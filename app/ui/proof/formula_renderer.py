"""Formula rendering for proof views.

This module is intentionally isolated:

- no proof data writes;
- no OCR/layout dependency;
- no import from UI panels;
- one public rendering function returning a ``QPixmap``.

MathJax SVG is preferred. Matplotlib MathText is the in-process backend used
when an external JavaScript or TeX executor is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import atexit
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import tempfile
import json
from threading import Lock, Thread

import numpy as np
from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


ENV_FLAG = "OCR_FORMULA_RENDER"
ENV_ENGINE = "OCR_FORMULA_ENGINE"
ENV_MATHJAX_NODE_MODULES = "OCR_MATHJAX_NODE_MODULES"
ENV_MATHJAX_NODE_BIN = "OCR_MATHJAX_NODE_BIN"
ENV_MATHJAX_TIMEOUT = "OCR_MATHJAX_TIMEOUT"
DEFAULT_ENGINE = "mathjax_svg"
DEFAULT_MATHJAX_TIMEOUT_SECONDS = 60


class _MathJaxRenderProcess:
    """One serialized MathJax worker reused across formula renders."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: Queue[bytes | None] = Queue()

    def render(self, latex_body: str) -> bytes:
        with self._lock:
            process = self._ensure_process()
            if process.stdin is None:
                raise RuntimeError("mathjax worker stdin is unavailable")
            payload = json.dumps({"latex": latex_body}, ensure_ascii=False).encode("utf-8")
            try:
                process.stdin.write(payload + b"\n")
                process.stdin.flush()
                response = self._responses.get(timeout=_mathjax_timeout_seconds())
            except (BrokenPipeError, OSError, Empty) as exc:
                self._stop_locked()
                raise RuntimeError("mathjax worker did not return a response") from exc
            if response is None:
                self._stop_locked()
                raise RuntimeError("mathjax worker stopped unexpectedly")
            result = json.loads(response.decode("utf-8"))
            error = str(result.get("error", ""))
            if error:
                raise RuntimeError(error)
            svg = _extract_svg_bytes(str(result.get("svg", "")))
            if not svg:
                raise RuntimeError("mathjax returned empty svg")
            return svg

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        node_exe = _mathjax_node_executable()
        if node_exe is None:
            raise RuntimeError("node unavailable")
        env = os.environ.copy()
        node_path = _mathjax_node_path()
        if node_path:
            env["NODE_PATH"] = node_path
        responses: Queue[bytes | None] = Queue()
        self._responses = responses
        process = subprocess.Popen(
            [node_exe, "-e", _mathjax_worker_script()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._process = process
        Thread(
            target=self._read_responses,
            args=(process, responses),
            daemon=True,
            name="mathjax-render-reader",
        ).start()
        return process

    @staticmethod
    def _read_responses(
        process: subprocess.Popen[bytes],
        responses: Queue[bytes | None],
    ) -> None:
        if process.stdout is None:
            responses.put(None)
            return
        for line in iter(process.stdout.readline, b""):
            responses.put(line)
        responses.put(None)

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


_MATHJAX_PROCESS = _MathJaxRenderProcess()
atexit.register(_MATHJAX_PROCESS.close)


@dataclass(frozen=True)
class FormulaRenderResult:
    pixmap: QPixmap
    normalized_latex: str
    logical_width: int
    logical_height: int
    device_pixel_ratio: float
    backend: str


def formula_rendering_enabled() -> bool:
    """Return whether formula rendering is active.

    Rendering is enabled by default. Set ``OCR_FORMULA_RENDER=0`` to disable it
    without changing code.
    """
    value = os.environ.get(ENV_FLAG, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def render_formula_pixmap(
    text: str,
    *,
    target_height: int,
    color: str = "#2C2C2C",
    dpi: int = 180,
    oversample: float = 2.0,
) -> FormulaRenderResult | None:
    """Render a formula-like text fragment to a transparent pixmap.

    Returns ``None`` for disabled rendering, invalid text, unsupported TeX, or
    unavailable renderers. Callers must treat ``None`` as "show the formula
    source without a rendered preview".

    The returned pixmap is already scaled to the requested logical height and
    tagged with the active screen DPR.  Callers should use ``logical_width`` and
    ``logical_height`` for layout instead of ``pixmap.width()`` when high-DPI
    screens are involved.
    """
    if not formula_rendering_enabled():
        return None
    if QGuiApplication.instance() is None:
        return None
    latex = normalize_formula_latex(text)
    if not latex:
        return None
    height = max(8, int(target_height))
    dpr = _current_device_pixel_ratio()
    physical_height = max(8, int(round(height * dpr)))
    engine = _formula_engine()
    if engine == "mathjax_svg":
        try:
            pixmap, logical_width, logical_height = _render_mathjax_svg_pixmap(
                latex,
                color,
                physical_height=physical_height,
                dpr=dpr,
            )
            return FormulaRenderResult(
                pixmap=pixmap,
                normalized_latex=latex,
                logical_width=logical_width,
                logical_height=logical_height,
                device_pixel_ratio=dpr,
                backend="mathjax_svg",
            )
        except Exception:
            pass
    elif engine == "latex_svg":
        try:
            pixmap, logical_width, logical_height = _render_latex_svg_pixmap(
                latex,
                color,
                physical_height=physical_height,
                dpr=dpr,
            )
            return FormulaRenderResult(
                pixmap=pixmap,
                normalized_latex=latex,
                logical_width=logical_width,
                logical_height=logical_height,
                device_pixel_ratio=dpr,
                backend="latex_svg",
            )
        except Exception:
            pass
    try:
        pixmap, logical_width, logical_height = _render_mathtext_pixmap(
            latex,
            color,
            physical_height=physical_height,
            dpr=dpr,
            dpi=dpi,
            oversample=oversample,
        )
    except Exception:
        return None
    return FormulaRenderResult(
        pixmap=pixmap,
        normalized_latex=latex,
        logical_width=logical_width,
        logical_height=logical_height,
        device_pixel_ratio=dpr,
        backend="mathtext",
    )


def clear_formula_render_cache() -> None:
    """Clear cached formula renderer outputs."""
    for renderer in (_render_mathjax_svg, _render_latex_svg, _render_mathtext_rgba):
        cache_clear = getattr(renderer, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def normalize_formula_latex(text: str) -> str:
    """Normalize Paddle/Hanwang formula strings for renderer input."""
    s = (text or "").strip()
    if not s:
        return ""
    if _contains_cjk(s):
        return ""
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"^\s*\$\$\s*", "", s)
    s = re.sub(r"\s*\$\$\s*$", "", s)
    s = re.sub(r"^\s*\$\s*", "", s)
    s = re.sub(r"\s*\$\s*$", "", s)
    s = s.strip()
    if not s:
        return ""
    replacements = {
        "×": r"\times",
        "·": r"\cdot",
        "−": "-",
        "β": r"\beta",
        "α": r"\alpha",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
        "φ": r"\phi",
        "ϕ": r"\phi",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    s = re.sub(r"\s+", " ", s)
    return f"${s}$"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _formula_engine() -> str:
    value = os.environ.get(ENV_ENGINE, DEFAULT_ENGINE).strip().lower()
    if value in {"mathtext", "matplotlib"}:
        return "mathtext"
    if value in {"latex", "latex_svg", "dvisvgm", "latex_dvisvgm"}:
        return "latex_svg"
    return "mathjax_svg"


def _render_mathtext_pixmap(
    latex: str,
    color: str,
    *,
    physical_height: int,
    dpr: float,
    dpi: int,
    oversample: float,
) -> tuple[QPixmap, int, int]:
    render_dpi = max(72, int(round(float(dpi) * dpr * max(1.0, float(oversample)))))
    rgba = _render_mathtext_rgba(latex, color, render_dpi)
    if rgba.size == 0:
        raise RuntimeError("mathtext returned an empty image")
    image = _rgba_to_qimage(rgba)
    if image.height() != physical_height:
        image = image.scaledToHeight(
            physical_height,
            Qt.TransformationMode.SmoothTransformation,
        )
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        raise RuntimeError("mathtext returned an empty pixmap")
    pixmap.setDevicePixelRatio(dpr)
    return (
        pixmap,
        max(1, int(round(pixmap.width() / dpr))),
        max(1, int(round(pixmap.height() / dpr))),
    )


def _rgba_to_qimage(rgba: np.ndarray) -> QImage:
    return QImage(
        rgba.data,
        rgba.shape[1],
        rgba.shape[0],
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()


@lru_cache(maxsize=256)
def _render_mathtext_rgba(latex: str, color: str, dpi: int) -> np.ndarray:
    _prepare_matplotlib_env()
    from matplotlib.mathtext import MathTextParser

    parsed = MathTextParser("agg").parse(latex, dpi=dpi)
    alpha = np.asarray(parsed.image, dtype=np.uint8)
    qcolor = QColor(color)
    rgb = np.array([qcolor.red(), qcolor.green(), qcolor.blue()], dtype=np.uint8)
    rgba = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0:3] = rgb
    rgba[:, :, 3] = alpha
    visible = alpha > 0
    if not np.any(visible):
        return rgba
    rows = np.where(np.any(visible, axis=1))[0]
    cols = np.where(np.any(visible, axis=0))[0]
    top = max(0, int(rows[0]) - 1)
    bottom = min(rgba.shape[0], int(rows[-1]) + 2)
    left = max(0, int(cols[0]) - 1)
    right = min(rgba.shape[1], int(cols[-1]) + 2)
    return np.ascontiguousarray(rgba[top:bottom, left:right, :])


def _prepare_matplotlib_env() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "ocr_process_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _current_device_pixel_ratio() -> float:
    app = QGuiApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    try:
        value = float(screen.devicePixelRatio()) if screen is not None else 1.0
    except Exception:
        value = 1.0
    if value < 1.0:
        return 1.0
    return min(value, 4.0)


def _render_latex_svg_pixmap(
    latex: str,
    color: str,
    *,
    physical_height: int,
    dpr: float,
) -> tuple[QPixmap, int, int]:
    svg = _render_latex_svg(latex, _color_to_latex_hex(color))
    renderer = QSvgRenderer(QByteArray(svg))
    if not renderer.isValid():
        raise RuntimeError("invalid formula svg")
    size = renderer.defaultSize()
    if size.width() <= 0 or size.height() <= 0:
        raise RuntimeError("formula svg has no size")
    physical_width = max(1, int(round(size.width() * physical_height / max(1, size.height()))))
    image = QImage(physical_width, physical_height, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        renderer.render(painter, QRectF(0, 0, physical_width, physical_height))
    finally:
        painter.end()
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        raise RuntimeError("empty formula pixmap")
    pixmap.setDevicePixelRatio(dpr)
    logical_width = max(1, int(round(pixmap.width() / dpr)))
    logical_height = max(1, int(round(pixmap.height() / dpr)))
    return pixmap, logical_width, logical_height


def _render_mathjax_svg_pixmap(
    latex: str,
    color: str,
    *,
    physical_height: int,
    dpr: float,
) -> tuple[QPixmap, int, int]:
    svg = _render_mathjax_svg(_strip_math_delimiters(latex), color)
    return _render_svg_pixmap(svg, physical_height=physical_height, dpr=dpr)


def _render_svg_pixmap(
    svg: bytes,
    *,
    physical_height: int,
    dpr: float,
) -> tuple[QPixmap, int, int]:
    renderer = QSvgRenderer(QByteArray(svg))
    if not renderer.isValid():
        raise RuntimeError("invalid formula svg")
    size = renderer.defaultSize()
    aspect_ratio = 0.0
    if size.width() > 0 and size.height() > 0:
        aspect_ratio = float(size.width()) / max(1.0, float(size.height()))
    if aspect_ratio <= 0:
        aspect_ratio = _svg_viewbox_aspect_ratio(svg)
    if aspect_ratio <= 0:
        raise RuntimeError("formula svg has no size")
    physical_width = max(1, int(round(physical_height * aspect_ratio)))
    image = QImage(physical_width, physical_height, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        renderer.render(painter, QRectF(0, 0, physical_width, physical_height))
    finally:
        painter.end()
    image = _trim_transparent_image(image)
    if image.height() != physical_height:
        image = image.scaledToHeight(
            physical_height,
            Qt.TransformationMode.SmoothTransformation,
        )
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        raise RuntimeError("empty formula pixmap")
    pixmap.setDevicePixelRatio(dpr)
    logical_width = max(1, int(round(pixmap.width() / dpr)))
    logical_height = max(1, int(round(pixmap.height() / dpr)))
    return pixmap, logical_width, logical_height


def _trim_transparent_image(image: QImage) -> QImage:
    """Remove renderer canvas padding while preserving every painted pixel."""

    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    buffer = rgba.bits()
    pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(
        rgba.height(), rgba.bytesPerLine() // 4, 4
    )[:, : rgba.width(), :]
    visible = pixels[:, :, 3] > 0
    if not np.any(visible):
        return image
    rows = np.where(np.any(visible, axis=1))[0]
    cols = np.where(np.any(visible, axis=0))[0]
    left = int(cols[0])
    top = int(rows[0])
    right = int(cols[-1]) + 1
    bottom = int(rows[-1]) + 1
    return rgba.copy(left, top, right - left, bottom - top)


def _svg_viewbox_aspect_ratio(svg: bytes) -> float:
    match = re.search(rb'viewBox="([^"]+)"', svg)
    if not match:
        return 0.0
    parts = match.group(1).decode("ascii", errors="ignore").replace(",", " ").split()
    if len(parts) != 4:
        return 0.0
    try:
        width = abs(float(parts[2]))
        height = abs(float(parts[3]))
    except ValueError:
        return 0.0
    if width <= 0 or height <= 0:
        return 0.0
    return width / height


@lru_cache(maxsize=256)
def _render_mathjax_svg(latex_body: str, color: str) -> bytes:
    svg = _MATHJAX_PROCESS.render(latex_body)
    color_text = str(color or "#2C2C2C")
    svg = svg.replace(b"currentColor", color_text.encode("ascii", errors="ignore") or b"#2C2C2C")
    return svg


def _mathjax_timeout_seconds() -> int:
    raw = os.environ.get(ENV_MATHJAX_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_MATHJAX_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MATHJAX_TIMEOUT_SECONDS
    return max(5, value)


def _mathjax_node_executable() -> str | None:
    explicit = os.environ.get(ENV_MATHJAX_NODE_BIN, "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    candidates = [
        root / "formula" / "node" / executable
        for root in _formula_resource_roots()
        for executable in ("node.exe", "bin/node")
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("node")


def _mathjax_node_path() -> str:
    candidates: list[str] = []
    explicit = os.environ.get(ENV_MATHJAX_NODE_MODULES, "").strip()
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        str(root / "formula" / "mathjax" / "node_modules")
        for root in _formula_resource_roots()
    )
    # Development-only scratch install used by local renderer experiments. It is
    # ignored in packaged builds unless the directory exists.
    candidates.append(str(Path(tempfile.gettempdir()) / "ocr_formula_node" / "node_modules"))
    existing = [path for path in candidates if path and Path(path).exists()]
    current = os.environ.get("NODE_PATH", "").strip()
    if current:
        existing.append(current)
    return os.pathsep.join(existing)


def _formula_resource_roots() -> tuple[Path, ...]:
    """Return source and packaged resource roots without assuming one layout."""

    module_path = Path(__file__).resolve()
    app_root = module_path.parents[2]
    package_root = module_path.parents[3]
    return tuple(dict.fromkeys((
        package_root / "resources",
        package_root / "app" / "resources",
        app_root / "resources",
    )))


def _mathjax_worker_script() -> str:
    return r"""
const readline = require('readline');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
require('mathjax-full/js/input/tex/boldsymbol/BoldsymbolConfiguration.js');
require('mathjax-full/js/input/tex/newcommand/NewcommandConfiguration.js');
require('mathjax-full/js/input/tex/configmacros/ConfigMacrosConfiguration.js');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: ['base', 'ams', 'boldsymbol', 'newcommand', 'configmacros']});
const svg = new SVG({fontCache: 'none'});
const html = mathjax.document('', {InputJax: tex, OutputJax: svg});
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
input.on('line', (line) => {
  try {
    const request = JSON.parse(line);
    const node = html.convert(String(request.latex || ''), {display: true});
    const markup = adaptor.outerHTML(node);
    if (markup.indexOf('data-mjx-error=') !== -1) {
      throw new Error(markup);
    }
    process.stdout.write(JSON.stringify({svg: markup}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({error: String(error)}) + '\n');
  }
});
"""


def _extract_svg_bytes(markup: str) -> bytes:
    match = re.search(r"(<svg\b.*?</svg>)", markup or "", flags=re.S)
    if not match:
        return b""
    return match.group(1).encode("utf-8")


def _strip_math_delimiters(latex: str) -> str:
    s = (latex or "").strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(left) and s.endswith(right):
            return s[len(left): len(s) - len(right)].strip()
    return s


@lru_cache(maxsize=128)
def _render_latex_svg(latex: str, color_hex: str) -> bytes:
    if shutil.which("latex") is None or shutil.which("dvisvgm") is None:
        raise RuntimeError("latex/dvisvgm unavailable")
    with tempfile.TemporaryDirectory(prefix="ocr_formula_") as tmp:
        tmp_path = Path(tmp)
        tex_path = tmp_path / "formula.tex"
        tex_path.write_text(_latex_document(latex, color_hex), encoding="utf-8")
        subprocess.run(
            [
                "latex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tmp_path),
                str(tex_path),
            ],
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=True,
        )
        dvi_path = tmp_path / "formula.dvi"
        if not dvi_path.exists():
            raise RuntimeError("latex did not create dvi")
        proc = subprocess.run(
            [
                "dvisvgm",
                "--no-fonts",
                "--exact",
                "--bbox=min",
                "--stdout",
                str(dvi_path),
            ],
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=True,
        )
        if not proc.stdout:
            raise RuntimeError("dvisvgm returned empty svg")
        return proc.stdout


def _latex_document(latex: str, color_hex: str) -> str:
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{amsmath,amssymb,xcolor}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            rf"{{\color[HTML]{{{color_hex}}}{latex}}}",
            r"\end{document}",
            "",
        ]
    )


def _color_to_latex_hex(color: str) -> str:
    qcolor = QColor(color)
    if not qcolor.isValid():
        qcolor = QColor("#2C2C2C")
    return f"{qcolor.red():02X}{qcolor.green():02X}{qcolor.blue():02X}"
