"""Experimental formula rendering for proof views.

This module is intentionally isolated:

- no proof data writes;
- no OCR/layout dependency;
- no import from UI panels;
- one public rendering function returning a ``QPixmap``;
- removable by deleting this file and the thin HProof call site.

Current default backend: MathJax SVG.  LaTeX + dvisvgm remains available as an
explicit opt-in compatibility backend, but it is not part of the normal
delivery path because it depends on a full TeX installation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import json

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


ENV_FLAG = "OCR_EXPERIMENTAL_FORMULA_RENDER"
ENV_ENGINE = "OCR_EXPERIMENTAL_FORMULA_ENGINE"
ENV_MATHJAX_NODE_MODULES = "OCR_MATHJAX_NODE_MODULES"
ENV_MATHJAX_NODE_BIN = "OCR_MATHJAX_NODE_BIN"
ENV_MATHJAX_TIMEOUT = "OCR_MATHJAX_TIMEOUT"
DEFAULT_ENGINE = "mathjax_svg"
DEFAULT_MATHJAX_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class FormulaRenderResult:
    pixmap: QPixmap
    normalized_latex: str
    logical_width: int
    logical_height: int
    device_pixel_ratio: float
    backend: str


def formula_rendering_enabled() -> bool:
    """Return whether the experiment is active.

    The experiment is enabled by default so formula-debug rows can be evaluated
    in normal manual testing.  Set ``OCR_EXPERIMENTAL_FORMULA_RENDER=0`` to
    disable it without changing code.
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

    Returns ``None`` for disabled experiment, invalid text, unsupported TeX, or
    unavailable renderers.  Callers must treat ``None`` as "use old display".

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
            return None
    if engine == "latex_svg":
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
            return None
    return None


def clear_formula_render_cache() -> None:
    """Clear cached formula renderer outputs for manual renderer experiments."""
    for renderer in (_render_mathjax_svg, _render_latex_svg):
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
    if value in {"latex", "latex_svg", "dvisvgm", "latex_dvisvgm"}:
        return "latex_svg"
    return "mathjax_svg"


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
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        raise RuntimeError("empty formula pixmap")
    pixmap.setDevicePixelRatio(dpr)
    logical_width = max(1, int(round(pixmap.width() / dpr)))
    logical_height = max(1, int(round(pixmap.height() / dpr)))
    return pixmap, logical_width, logical_height


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
    node_exe = _mathjax_node_executable()
    if node_exe is None:
        raise RuntimeError("node unavailable")
    script = _mathjax_renderer_script()
    payload = json.dumps({"latex": latex_body}, ensure_ascii=False).encode("utf-8")
    env = os.environ.copy()
    node_path = _mathjax_node_path()
    if node_path:
        env["NODE_PATH"] = node_path
    proc = subprocess.run(
        [node_exe, "-e", script],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=_mathjax_timeout_seconds(),
        check=True,
        env=env,
    )
    result = json.loads(proc.stdout.decode("utf-8"))
    svg = _extract_svg_bytes(str(result.get("svg", "")))
    if not svg:
        raise RuntimeError("mathjax returned empty svg")
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
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "app" / "resources" / "formula" / "node" / "node.exe",
        root / "app" / "resources" / "formula" / "node" / "bin" / "node",
        root / "resources" / "formula" / "node" / "node.exe",
        root / "resources" / "formula" / "node" / "bin" / "node",
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
    root = Path(__file__).resolve().parents[2]
    candidates.append(str(root / "app" / "resources" / "formula" / "mathjax" / "node_modules"))
    candidates.append(str(root / "resources" / "formula" / "mathjax" / "node_modules"))
    # Development-only scratch install used by local renderer experiments. It is
    # ignored in packaged builds unless the directory exists.
    candidates.append(str(Path(tempfile.gettempdir()) / "ocr_formula_node" / "node_modules"))
    existing = [path for path in candidates if path and Path(path).exists()]
    current = os.environ.get("NODE_PATH", "").strip()
    if current:
        existing.append(current)
    return os.pathsep.join(existing)


def _mathjax_renderer_script() -> str:
    return r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
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
const node = html.convert(String(input.latex || ''), {display: true});
const markup = adaptor.outerHTML(node);
if (markup.indexOf('data-mjx-error=') !== -1) {
  process.stderr.write(markup);
  process.exit(2);
}
process.stdout.write(JSON.stringify({svg: markup}));
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
