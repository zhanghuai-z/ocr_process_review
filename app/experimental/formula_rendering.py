"""Experimental formula rendering for proof views.

This module is intentionally isolated:

- no proof data writes;
- no OCR/layout dependency;
- no import from UI panels;
- one public rendering function returning a ``QPixmap``;
- removable by deleting this file and the thin HProof call site.

Current backend: Matplotlib mathtext.  It is not the final desired renderer,
but it gives us an offline, low-risk experiment before embedding a fuller
MathJax/KaTeX SVG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap


ENV_FLAG = "OCR_EXPERIMENTAL_FORMULA_RENDER"


@dataclass(frozen=True)
class FormulaRenderResult:
    pixmap: QPixmap
    normalized_latex: str


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
) -> FormulaRenderResult | None:
    """Render a formula-like text fragment to a transparent pixmap.

    Returns ``None`` for disabled experiment, invalid text, unsupported TeX, or
    unavailable Matplotlib.  Callers must treat ``None`` as "use old display".
    """
    if not formula_rendering_enabled():
        return None
    if QGuiApplication.instance() is None:
        return None
    latex = normalize_formula_latex(text)
    if not latex:
        return None
    try:
        rgba = _render_mathtext_rgba(latex, color, int(dpi))
    except Exception:
        return None
    if rgba.size == 0:
        return None
    image = QImage(
        rgba.data,
        rgba.shape[1],
        rgba.shape[0],
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        return None
    height = max(8, int(target_height))
    if pixmap.height() > height:
        pixmap = pixmap.scaledToHeight(height)
    return FormulaRenderResult(pixmap=pixmap, normalized_latex=latex)


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
