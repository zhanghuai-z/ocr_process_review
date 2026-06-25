"""Experimental formula rendering for proof views.

This module is intentionally isolated:

- no proof data writes;
- no OCR/layout dependency;
- no import from UI panels;
- one public rendering function returning a ``QPixmap``;
- removable by deleting this file and the thin HProof call site.

Current default backend: LaTeX + dvisvgm SVG, with Matplotlib mathtext as a
fallback.  This is still experimental, but it is closer to real formula
typesetting than mathtext alone and keeps the HProof call site unchanged.
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

import numpy as np
from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


ENV_FLAG = "OCR_EXPERIMENTAL_FORMULA_RENDER"
ENV_ENGINE = "OCR_EXPERIMENTAL_FORMULA_ENGINE"
DEFAULT_ENGINE = "latex_svg"


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
    render_dpi = max(72, int(round(float(dpi) * dpr * max(1.0, float(oversample)))))
    engine = _formula_engine()
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
            pass
    try:
        rgba = _render_mathtext_rgba(latex, color, render_dpi)
    except Exception:
        return None
    if rgba.size == 0:
        return None
    image = _rgba_to_qimage(rgba)
    if image.height() != physical_height:
        image = image.scaledToHeight(
            physical_height,
            Qt.TransformationMode.SmoothTransformation,
        )
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        return None
    pixmap.setDevicePixelRatio(dpr)
    logical_width = max(1, int(round(pixmap.width() / dpr)))
    logical_height = max(1, int(round(pixmap.height() / dpr)))
    return FormulaRenderResult(
        pixmap=pixmap,
        normalized_latex=latex,
        logical_width=logical_width,
        logical_height=logical_height,
        device_pixel_ratio=dpr,
        backend="mathtext",
    )


def clear_formula_render_cache() -> None:
    """Clear cached formula renderer outputs for manual renderer experiments."""
    _render_latex_svg.cache_clear()
    _render_mathtext_rgba.cache_clear()


def normalize_formula_latex(text: str) -> str:
    """Normalize Paddle/Hanwang formula strings for mathtext input."""
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
    return DEFAULT_ENGINE


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


def _rgba_to_qimage(rgba: np.ndarray) -> QImage:
    return QImage(
        rgba.data,
        rgba.shape[1],
        rgba.shape[0],
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()


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
    # Trim transparent margins to make line placement predictable.
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
