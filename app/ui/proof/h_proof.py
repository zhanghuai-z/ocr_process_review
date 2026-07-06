"""横校面板（PRD 3.2）：滚动列表，每行显示【图像行 + 识别文本】对。

布局（参照 ui-2.jpg）：
  ┌────────────────────────────────────────────────────────────┐
  │  ↑上一行  ↓下一行  [保存 Ctrl+S]  [标记 F5]  [跳过 F6]     │ ← 工具栏
  ├────────────────────────────────────────────────────────────┤
  │ 图像行  1 │ [扫描图像行] ─────────────────────────────────  │
  │ 识别文本 1│  识别出的文字文本                           ●待确认 │
  │───────────────────────────────────────────────────────────│
  │ 图像行  2 │ [扫描图像行]                                    │
  │ 识别文本 2│  文字文本                                   ●已确认 │
  │  ─ ─ ─ ─ ─ ─ ─ ─ active row: 蓝色左边框 ─ ─ ─ ─ ─ ─ ─ ─ ─│
  │▌图像行  3 │ [扫描图像行]                                    │  ← active
  │▌识别文本 3│▶ [可编辑文本区]                             1处差异│  ← active
  └────────────────────────────────────────────────────────────┘
  状态栏：总字数 12,523 | 差异 128 (1.02%)  ● 与原文一致 ● 疑似 ● 不确认 ● 已标记

快捷键（文本编辑框内有效）：
  Enter       — 确认当前行并跳到下一行
  Ctrl+↑/↓    — 上一行 / 下一行（不确认）
  F5          — 跳到下一疑点 / 标记当前行
  F6          — 跳过本行（状态不变）
  Escape      — 还原当前行到 OCR 原始文本
  Ctrl+S      — 保存全部修改
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
import re
from typing import List, Optional
import unicodedata

import cv2
from PySide6.QtCore import QEvent, Qt, QRect, QTimer, Signal, QSize, QPoint
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
    QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, BlockType, Line, Page, ProofStatus
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_character_observation import line_ocr_chars
from app.models.ocr_observation import (
    block_ocr_line_observations_by_uid,
)
from app.core.block_attributes import block_attributes, normalize_source_label, semantic_block_type
from app.core.ocr_ir import is_formula_marker_token
from app.core.page_image_cache import PageImageCache
from app.core.proof_change import ProofChangeSet
from app.core.proof_atom import ProofAtom, ProofAtomKind
from app.core.proof_char_text import char_display_text, chars_display_spans, chars_display_text
from app.core.proof_line_facts import proof_block_text, proof_display_text, proof_ocr_text, proof_status
from app.core.proof_occurrence import line_signature
from app.core.proof_projection import ProofLineProjection, build_proof_line_projection
from app.core.raw_ocr_artifact import raw_block_text_values
from app.ui.widgets.page_directory import PageDirectoryList
from app.utils.icon_manager import get_icon
from app.core.proof_line_utils import iter_unique_page_hproof_lines
from app.core.proof_state import (
    TOPIC_LINE_PROOF_CHANGED,
    ProofSelection,
    ProofUpdateRequest,
)
from app.core.proof_state_bus import ProofStateBus
from app.services.proof_probe_text_service import (
    displayed_text as _displayed_text,
)
from app.services.proof_edit_service import ProofEditService, ProofEditStatus
from app.services.proof_hproof_session import (
    HProofLineEditSession,
    HProofRuntimeSession,
)
from app.services.proof_image_service import clamp_line_box_pixels
from app.services.proof_rebuild_gate import (
    ProofEditorRebuildState,
    proof_rebuild_gate_for_editor_state,
)
from app.ui.proof.confidence_utils import char_confidence
from app.ui.proof import char_verdict as _cv
# NOTE: AlignmentRibbon 已从布局中移除（proof-layout-collections 第 1 任务）。
# 用户原话：“既然已经做图字对应，就不要第三行文本行”。图字 y 轴对应
# 现在改回只走 hover/click 联动（图像悬停 → editor 高亮当前字；editor
# 光标变 → 图像画当前字 bbox），不再用 ribbon 重复绘制一行文本。

# ── 样式常量 ──────────────────────────────────────────────────
ROW_PAD_Y    = 4     # 裁图上下各加 4px
IMAGE_ROW_H  = 38    # 当前行图像显示高度（px）
NEAR_IMAGE_ROW_H = 24
FAR_IMAGE_ROW_H = 20
# 脚注、数字、标点的 Hanwang 字符框通常比正文窄。字号按较小文本优先，
# 避免按 slot center 自绘时把标点和数字挤在一起。
TEXT_FONT_PX = 26
TEXT_FONT_WEIGHT = QFont.Weight.Bold
TEXT_FONT_WEIGHT_CSS = 700
TEXT_LINE_HEIGHT_PX = 34
TEXT_EDITOR_MAX_H = 40
NEAR_TEXT_EDITOR_H = 25
FAR_TEXT_EDITOR_H = 22
TEXT_SLOT_MIN_W = 10.0
TEXT_SLOT_GUTTER_W = 4.0
TEXT_SLOT_GAP_W = 1.0
TEXT_SLOT_CLIPPED_MIN_W = 1.0
IMAGE_DIVIDER_COLOR = "#D8D2C8"
TEXT_GUIDE_LINE_COLOR = "#AFC0D8"
ROW_DIVIDER_COLOR = "#E7E2D8"
FOCUS_BORDER_COLOR = "#7A7368"
FORMULA_VISUAL_HEIGHT_RATIO = 0.70
FORMULA_SOURCE_PANEL_H = 88
FORMULA_SOURCE_POPUP_OFFSET = QPoint(10, 18)
# 保持“图 + 文本”两层的既有总高度，减少布局连锁变化。
LINE_PAIR_H = 92
NEAR_LINE_PAIR_H = 34
FAR_LINE_PAIR_H = 28


class _ProofProgressRing(QWidget):
    """Clickable compact progress indicator for the proof status bar."""

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("proofProgressRing")
        self.setFixedSize(38, 38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._handled = 0
        self._total = 0

    def set_counts(self, handled: int, total: int) -> None:
        self._handled = max(0, int(handled))
        self._total = max(0, int(total))
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setPen(QPen(QColor("#E7E2D8"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 0, 360 * 16)
        if self._total > 0 and self._handled > 0:
            ratio = min(1.0, self._handled / max(1, self._total))
            painter.setPen(QPen(QColor("#2C2C2C"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(rect, 90 * 16, int(-360 * 16 * ratio))
        painter.setPen(QColor("#2C2C2C"))
        font = painter.font()
        font.setPixelSize(8)
        font.setBold(True)
        painter.setFont(font)
        pct = int(round((self._handled / max(1, self._total)) * 100)) if self._total else 0
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{pct}%")


TEXT_FONT_FAMILY = "'Noto Serif CJK SC','Source Han Serif SC','SimSun','Songti SC','Times New Roman',serif"
LABEL_W      = 88    # 左侧行号列宽
STATUS_W     = 24    # 右侧状态列宽
LOW_CONF     = 0.80

_STATUS_COLOR = {
    ProofStatus.OK:           "#4CAF50",
    ProofStatus.MODIFIED:     "#5C6B58",
    ProofStatus.AUTO_FLAGGED: "#FF9800",
    ProofStatus.UNCHECKED:    TEXT_GUIDE_LINE_COLOR,
}
_STATUS_GLYPH = {
    ProofStatus.OK: "●",
    ProofStatus.MODIFIED: "◆",
    ProofStatus.AUTO_FLAGGED: "▲",
    ProofStatus.UNCHECKED: "○",
}

_DEBUG_FORMULA_LINE_FLAGS = {"hanwang_route_inline_formula"}
_DEBUG_TABLE_LINE_FLAGS = {"hanwang_route_table"}
_DEBUG_FORMULA_EXCLUDED_LABELS = {"formula_number"}
_DEBUG_FORMULA_LABEL_TOKENS = ("formula", "equation", "math")
_DEBUG_TABLE_LABEL_TOKENS = ("table",)
_INLINE_FORMULA_RE = re.compile(
    r"(\$\$.*?\$\$|\$[^$\n]+?\$|\\\(.*?\\\)|\\\[.*?\\\])",
    re.S,
)


def _slot_visual_width(
    text_char: str,
    font_metrics: QFontMetrics,
) -> float:
    """Return the visual slot width used by HProof's painted text layer.

    This deliberately does not use ``Char.bbox.w``. OCR bbox is data identity
    for crop/highlight/export; the HProof text layer only needs enough visual
    room for the rendered glyph.
    """
    glyph_width = float(font_metrics.horizontalAdvance(text_char or " ")) + TEXT_SLOT_GUTTER_W
    return max(TEXT_SLOT_MIN_W, glyph_width)


def _is_punctuation_slot_text(text: str) -> bool:
    """Whether a slot should draw text centered inside its visual cell."""
    if not text:
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in text)


def _formula_overlay_slot_width(
    text_char: str,
    atom: ProofAtom,
    font_metrics: QFontMetrics,
    scale: float,
) -> float:
    width = _slot_visual_width(text_char, font_metrics)
    if not _is_punctuation_slot_text(text_char) or atom.bbox is None or scale <= 0:
        return width
    bbox_width = max(TEXT_SLOT_CLIPPED_MIN_W, float(atom.bbox.w) * scale)
    return min(width, bbox_width)


_LATEX_SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "approx": "≈",
    "infty": "∞",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "partial": "∂",
    "nabla": "∇",
    "rightarrow": "→",
    "to": "→",
    "leftarrow": "←",
}

_SUPERSCRIPT = str.maketrans({
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "i": "ⁱ",
    "n": "ⁿ",
})
_SUBSCRIPT = str.maketrans({
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
})


def _translate_script(text: str, table: dict[int, str], marker: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    translated = text.translate(table)
    if translated != text:
        return translated
    if len(text) == 1:
        return f"{marker}{text}"
    return f"{marker}{{{text}}}"


def _compact_formula_spacing(text: str) -> str:
    """Collapse OCR-spaced Latin formula tokens without touching prose spacing."""
    s = text
    latin = r"A-Za-zΑ-Ωα-ω"
    sub_sup = "₀-₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ"
    # OCR often returns formula words as "I n c e n t i v e _ t".
    s = re.sub(rf"(?<=[{latin}])\s+(?=[{latin}_])", "", s)
    s = re.sub(rf"(?<=[{latin}0-9])\s+(?=[{sub_sup}])", "", s)
    s = re.sub(rf"(?<=[{sub_sup}])\s+(?=[{latin}0-9])", "", s)
    s = re.sub(r"\s*([=+\-×·*/])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=\S)", "", s)
    s = re.sub(r"(?<=\S)\s+(?=[\u3400-\u9fff])", "", s)
    return s


def _render_formula_display(text: str) -> str:
    """Best-effort visual text for HProof formula rows.

    This is intentionally a display-only renderer. It handles the common
    Paddle/Hanwang LaTeX fragments we see in proofreading, but keeps the raw
    OCR text in the editor document so saving/export stays unchanged.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("$$", "").replace("$", "")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)⁄(\2)", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"√\1", s)
    s = re.sub(
        r"\\([A-Za-z]+)",
        lambda m: _LATEX_SYMBOLS.get(m.group(1), m.group(1)),
        s,
    )
    s = re.sub(r"\^\s*\{\s*([^{}]+?)\s*\}", lambda m: _translate_script(m.group(1), _SUPERSCRIPT, "^"), s)
    s = re.sub(r"_\s*\{\s*([^{}]+?)\s*\}", lambda m: _translate_script(m.group(1), _SUBSCRIPT, "_"), s)
    s = re.sub(r"\^\s*([A-Za-z0-9+\-=()])", lambda m: _translate_script(m.group(1), _SUPERSCRIPT, "^"), s)
    s = re.sub(r"_\s*([A-Za-z0-9+\-=()])", lambda m: _translate_script(m.group(1), _SUBSCRIPT, "_"), s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return _compact_formula_spacing(s)


def _contains_cjk_text(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _formula_visual_target_height(editor_height: int) -> int:
    return max(10, int(round(max(1, int(editor_height)) * FORMULA_VISUAL_HEIGHT_RATIO)))


def _render_formula_visual(text: str, *, target_height: int) -> _FormulaVisual | None:
    fallback_text = _render_formula_display(text)
    try:
        from app.experimental.formula_rendering import render_formula_pixmap

        result = render_formula_pixmap(
            text,
            target_height=max(8, int(target_height)),
            color="#2C2C2C",
        )
    except Exception:
        result = None
    if result is not None:
        return _FormulaVisual(
            text=fallback_text or text.strip(),
            pixmap=result.pixmap,
            logical_size=QSize(result.logical_width, result.logical_height),
            kind="formula",
        )
    if fallback_text:
        kind = "" if _contains_cjk_text(fallback_text) else "formula"
        return _FormulaVisual(text=fallback_text, kind=kind)
    return None


def _clip_slot_widths_to_centers(
    x_centers: List[Optional[float]],
    widths: List[float],
    *,
    gap: float = TEXT_SLOT_GAP_W,
) -> List[float]:
    """Return display-only slot widths that do not overlap neighbor centers.

    Raw centers still come from ``Char.bbox`` and remain unchanged. When a
    punctuation glyph wants a wider visual slot than the OCR spacing allows,
    clip the hover/selection block width instead of moving the glyph.
    """
    clipped: List[float] = list(widths)
    i = 0
    n = min(len(x_centers), len(widths))
    while i < n:
        while i < n and x_centers[i] is None:
            i += 1
        start = i
        while i < n and x_centers[i] is not None:
            i += 1
        if start >= i:
            continue
        indices = list(range(start, i))
        raw = [float(x_centers[idx]) for idx in indices if x_centers[idx] is not None]
        for offset, idx in enumerate(indices):
            limits: list[float] = []
            if offset > 0:
                limits.append(max(TEXT_SLOT_CLIPPED_MIN_W, raw[offset] - raw[offset - 1] - gap))
            if offset + 1 < len(raw):
                limits.append(max(TEXT_SLOT_CLIPPED_MIN_W, raw[offset + 1] - raw[offset] - gap))
            if limits:
                clipped[idx] = min(max(TEXT_SLOT_CLIPPED_MIN_W, float(widths[idx])), min(limits))
    return clipped


def _debug_block_labels(block: Block) -> set[str]:
    attrs = block_attributes(block)
    labels = {
        attrs.normalized_source_label,
        attrs.normalized_semantic_label,
    }
    return {label for label in labels if label}


def _line_has_formula_source(line: Line) -> bool:
    has_formula_route = any(flag in _DEBUG_FORMULA_LINE_FLAGS for flag in line.review_flags)
    formula_texts: list[str] = []
    for char in line_ocr_chars(line):
        source = normalize_source_label(getattr(char, "bbox_source", ""))
        if source == "paddle_inline_formula":
            formula_texts.append(char_display_text(char))
    if formula_texts:
        return any(not is_formula_marker_token(text) for text in formula_texts)
    if has_formula_route:
        return not is_formula_marker_token(proof_display_text(line))
    return False


def _line_is_formula_marker_only(line: Line) -> bool:
    text = proof_display_text(line)
    if text and is_formula_marker_token(text):
        return True
    formula_texts = [
        char_display_text(char)
        for char in line_ocr_chars(line)
        if normalize_source_label(getattr(char, "bbox_source", "")) == "paddle_inline_formula"
    ]
    return bool(formula_texts) and all(is_formula_marker_token(text) for text in formula_texts)


def _block_debug_content(page: Page, block: Block) -> str:
    keys = ("block_content", "content", "text", "latex", "formula", "formula_latex")
    for value in raw_block_text_values(block, keys, page):
        return value
    return proof_block_text(block)


def _synthetic_block_debug_line(page: Page, block: Block) -> Line | None:
    text = _block_debug_content(page, block).strip()
    if not text:
        return None
    return Line(
        text=text,
        confidence=1.0,
        bbox=block.bbox,
        ocr_text=text,
    )


def _line_has_table_source(line: Line) -> bool:
    return any(flag in _DEBUG_TABLE_LINE_FLAGS for flag in line.review_flags)


def _is_debug_formula_block(block: Block) -> bool:
    labels = _debug_block_labels(block)
    if labels & _DEBUG_FORMULA_EXCLUDED_LABELS:
        return False
    if semantic_block_type(block) == BlockType.EQUATION:
        return True
    return any(
        token in label
        for label in labels
        for token in _DEBUG_FORMULA_LABEL_TOKENS
    )


def _is_inline_formula_block(block: Block) -> bool:
    return any("inline_formula" in label for label in _debug_block_labels(block))


def _is_display_formula_unit(block: Block, line: Line) -> bool:
    if _line_is_formula_marker_only(line):
        return False
    if _is_inline_formula_block(block):
        return False
    if not _is_debug_formula_block(block):
        return False
    if semantic_block_type(block) == BlockType.EQUATION:
        return True
    labels = _debug_block_labels(block)
    return any(label in {"display_formula", "display_equation", "equation"} for label in labels)


def _is_debug_table_block(block: Block) -> bool:
    if semantic_block_type(block) == BlockType.TABLE:
        return True
    labels = _debug_block_labels(block)
    return any(
        token in label
        for label in labels
        for token in _DEBUG_TABLE_LABEL_TOKENS
    )


def _view_debug_labels(view: LayoutBlockView, block: Block) -> set[str]:
    labels = set(_debug_block_labels(block))
    labels.add(normalize_source_label(view.block_type.value))
    labels.add(normalize_source_label(view.source_label))
    labels.add(normalize_source_label(view.origin.source_label))
    labels.discard("")
    return labels


def _is_debug_formula_view(view: LayoutBlockView, block: Block) -> bool:
    labels = _view_debug_labels(view, block)
    if labels & _DEBUG_FORMULA_EXCLUDED_LABELS:
        return False
    if view.block_type == BlockType.EQUATION or semantic_block_type(block) == BlockType.EQUATION:
        return True
    return any(
        token in label
        for label in labels
        for token in _DEBUG_FORMULA_LABEL_TOKENS
    )


def _is_debug_table_view(view: LayoutBlockView, block: Block) -> bool:
    if view.block_type == BlockType.TABLE or semantic_block_type(block) == BlockType.TABLE:
        return True
    labels = _view_debug_labels(view, block)
    return any(
        token in label
        for label in labels
        for token in _DEBUG_TABLE_LABEL_TOKENS
    )


def _debug_line_kind(block: Block, line: Line) -> str:
    if (_is_debug_formula_block(block) and not _line_is_formula_marker_only(line)) or _line_has_formula_source(line):
        return "公式"
    if _is_debug_table_block(block) or _line_has_table_source(line):
        return "表格"
    return ""


def _is_duplicate_debug_line(line: Line, seen: list[tuple[str, object]]) -> bool:
    text = proof_display_text(line)
    bbox = line.bbox.normalize()
    for seen_text, seen_bbox in seen:
        if text == seen_text and bbox.iou(seen_bbox) >= 0.85:
            return True
    seen.append((text, bbox))
    return False


def iter_unique_page_hproof_debug_lines(
    page: Page,
    *,
    formulas: bool = False,
    tables: bool = False,
) -> Iterator[tuple[Block, Line, int]]:
    """Yield formula/table debug lines without changing normal HProof routing."""
    if not formulas and not tables:
        return
    seen: list[tuple[str, object]] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        formula_block = _is_debug_formula_view(view, block)
        table_block = _is_debug_table_view(view, block)
        lines = block_ocr_line_observations_by_uid(view.uid)
        if formulas and formula_block and not lines:
            synthetic = _synthetic_block_debug_line(page, block)
            if synthetic is not None and not _line_is_formula_marker_only(synthetic):
                if not _is_duplicate_debug_line(synthetic, seen):
                    yield block, synthetic, -1
            continue
        for line_idx, line in enumerate(lines):
            include_formula = formulas and (
                (formula_block and not _line_is_formula_marker_only(line))
                or _line_has_formula_source(line)
            )
            include_table = tables and (
                table_block or _line_has_table_source(line)
            )
            if not include_formula and not include_table:
                continue
            if _is_duplicate_debug_line(line, seen):
                continue
            yield block, line, line_idx


# ─────────────────────────────────────────────────────────────
# 固定槽位辅助函数
# ─────────────────────────────────────────────────────────────

def _chars_are_single_codepoint(chars) -> bool:
    """chars 列表里每个 char.char 是否都恰好一个字符（含 1 个空格）。

    word/token-granularity 的 chars 元素可能是 "2016"、"abc" 之类的多字符
    token；这种情况下 1 char ↔ 1 codepoint 的槽位模型不成立。
    """
    if not chars:
        return False
    for c in chars:
        ch = c.char or ""
        if len(ch) != 1:
            return False
    return True


def _canonicalize_text_to_slots(text: str, chars) -> tuple[str, bool]:
    """把 displayed_text 规范化到 ``len(chars)`` 槽位。

    返回 ``(canonical_text, slot_locked)``：

    - chars 为空 / 含多字符 token → 不动文本，``slot_locked=False``。
    - ``len(text) < len(chars)`` → 末尾用 ASCII 空格补到 len(chars)；锁定。
    - ``len(text) == len(chars)`` → 不动；锁定。
    - ``len(text) > len(chars)`` → **不**自动截断（可能截掉 quality probe
      插入的 fake_char 或用户已写入的有效字）；保持自由编辑，``slot_locked=False``。

    规则：OCR 元素数 = 槽位数，超短行自动补空白槽；超长行降级为自由编辑，
    不能静默截断用户文本。
    """
    if not _chars_are_single_codepoint(chars):
        return text, False
    n = len(chars)
    tl = len(text)
    if tl == n:
        return text, True
    if tl < n:
        return text + " " * (n - tl), True
    # tl > n：拒绝自动截断
    return text, False


# ─────────────────────────────────────────────────────────────
# 行内文本编辑器（拦截 Enter/方向键/F 键）
# ─────────────────────────────────────────────────────────────

class _RowEditor(QPlainTextEdit):
    """嵌入行内的单行文本编辑器，拦截专用快捷键。

    当 ``self._fixed_length`` 不为 None 时（= 行有固定 OCR 字符观测，图像元素数固定），
    输入行为强制为"覆写模式"：

    - 普通字符输入：若无 selection，自动选中光标处的下一字 → 由 super 替换；
      若有 selection，必须替换为等长文本（多了截断、少了不接受）。
    - Backspace / Delete / Cut：一律拒绝（弹一次性 tooltip 提示）。
    - 粘贴：通过 insertFromMimeData 拦截，截断到 selection 长度（或 0）。

    这保证用户不会"silently 把文本删短"，最终落盘文本与图像 char.bbox 数
    永远等长，图字一一对应不会跑偏。
    """

    confirm_requested = Signal()
    prev_requested    = Signal()
    next_requested    = Signal()
    flag_requested    = Signal()
    skip_requested    = Signal()
    revert_requested  = Signal()
    # 鼠标悬停字符位置变化（-1 = 离开 editor 区域）
    hover_char_changed = Signal(int)
    # 编辑器获取焦点 / 点击 → 自动激活本行。
    row_focus_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fixed_length: Optional[int] = None
        self._last_hover_idx: int = -1
        # 启用鼠标跟踪：无需按下也能收到 mouseMoveEvent，用于图字 hover 联动
        self.setMouseTracking(True)
        # 每个字的 x 坐标由 OCR 字符观测 bbox 映射到 editor 像素空间。
        # _LinePair 在每次 _render_line_image 后推入 x_centers / widths。
        # None / 空列表表示降级为 QPlainTextEdit 原生渲染。
        self._slot_x_centers: Optional[List[Optional[float]]] = None
        self._slot_widths: Optional[List[float]] = None

    def set_slot_geometry(
        self,
        x_centers: Optional[List[Optional[float]]],
        widths: Optional[List[float]],
    ) -> None:
        """由 _LinePair 在行图缩放就绪后推入；每字在 editor 视口内的 x 中心。

        x_centers / widths 为 None 或空 → 自动降级到 Qt 原生文本渲染。
        """
        if not x_centers:
            self._slot_x_centers = None
            self._slot_widths = None
        else:
            self._slot_x_centers = list(x_centers)
            self._slot_widths = list(widths) if widths else [12.0] * len(x_centers)
        try:
            self.viewport().update()
        except Exception:
            self.update()

    def has_slot_geometry(self) -> bool:
        return bool(self._slot_x_centers)

    def _slot_index_for_x(self, x: float, *, nearest: bool = False) -> int:
        """Return the visible slot index at editor-local x.

        In slot-paint mode the visible glyph positions no longer match Qt's
        native text layout. Mouse hit-testing must therefore use the same slot
        centers that paintEvent uses, otherwise a click on the visible glyph can
        move the cursor to a different text offset.
        """
        centers = self._slot_x_centers or []
        if not centers:
            return -1
        explicit_widths = self._slot_widths is not None
        widths = self._slot_widths or [TEXT_SLOT_MIN_W] * len(centers)
        best_idx = -1
        best_dist = float("inf")
        first_left: float | None = None
        last_right: float | None = None
        for idx, center in enumerate(centers):
            if center is None:
                continue
            width = widths[idx] if idx < len(widths) else TEXT_SLOT_MIN_W
            min_half = TEXT_SLOT_CLIPPED_MIN_W / 2.0 if explicit_widths else TEXT_SLOT_MIN_W / 2.0
            half = max(min_half, float(width) / 2.0)
            left = float(center) - half
            right = float(center) + half
            first_left = left if first_left is None else min(first_left, left)
            last_right = right if last_right is None else max(last_right, right)
            if left <= x <= right:
                return idx
            dist = abs(float(center) - x)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if not nearest or best_idx < 0:
            return -1
        # Clicks between adjacent narrow slots should still select the nearest
        # visible glyph. Large blank margins remain non-character area.
        margin = max(24.0, TEXT_SLOT_MIN_W * 2.0)
        if first_left is not None and last_right is not None:
            if x < first_left - margin or x > last_right + margin:
                return -1
        return best_idx

    def _event_pos(self, event):
        try:
            return event.position().toPoint()
        except AttributeError:
            return event.pos()

    def _select_slot_index(self, idx: int) -> None:
        """Select one visible slot so typing overwrites that character."""
        if idx < 0 or idx >= len(self.toPlainText()):
            return
        cur = self.textCursor()
        cur.setPosition(idx)
        cur.setPosition(idx + 1, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cur)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """图字 y 轴对应核心：当 _slot_x_centers 已就绪，**完全自绘文本**
        到 image bbox 决定的 x 位置；否则走 super 原生渲染。

        自绘路径里还要把 _refresh_extra_selections 生成的 verdict 前景色 /
        当前字背景体现出来；为此读 self.extraSelections() 的 ExtraSelection
        列表自己绘。原生 cursor 不再画（外部已 setCursorWidth(0)）。
        """
        if not self._slot_x_centers:
            super().paintEvent(event)
            return
        vp = self.viewport()
        p = QPainter(vp)
        try:
            p.fillRect(vp.rect(), self.palette().base())
            text = self.toPlainText()
            n = min(len(text), len(self._slot_x_centers))

            # 收集 per-index 的 verdict fg / bg（从 extraSelections）。
            fg_color: dict[int, QColor] = {}
            bg_color: dict[int, QColor] = {}
            try:
                for sel in self.extraSelections():
                    cur = sel.cursor
                    start = cur.selectionStart()
                    end = cur.selectionEnd()
                    fmt = sel.format
                    if fmt.foreground().style() != Qt.BrushStyle.NoBrush:
                        c = fmt.foreground().color()
                        for i in range(start, end):
                            fg_color[i] = c
                    if fmt.background().style() != Qt.BrushStyle.NoBrush:
                        c = fmt.background().color()
                        for i in range(start, end):
                            bg_color[i] = c
            except Exception:
                pass

            p.setFont(self.font())
            fm = p.fontMetrics()
            y_baseline = (vp.height() + fm.ascent() - fm.descent()) // 2

            for i in range(n):
                xc = self._slot_x_centers[i]
                if xc is None:
                    continue
                ch = text[i]
                slot_w = self._slot_widths[i] if i < len(self._slot_widths or []) else 12.0
                slot_w = max(TEXT_SLOT_MIN_W, float(slot_w))
                left = int(round(xc - slot_w / 2.0))
                right = int(round(xc + slot_w / 2.0))
                cell = QRect(left, 0, max(1, right - left), vp.height())

                # 1) 背景（当前字 / 错字底色）
                bg = bg_color.get(i)
                if bg is not None and bg.alpha() > 0:
                    p.fillRect(cell, bg)

                # 2) 文本
                col = fg_color.get(i)
                if col is None:
                    col = self.palette().text().color()
                p.setPen(QPen(col, 1))
                char_w = fm.horizontalAdvance(ch)
                tx = int(round(xc - char_w / 2.0))
                p.drawText(tx, y_baseline, ch)
        finally:
            p.end()

    def set_fixed_length(self, n: Optional[int]) -> None:
        """启用/关闭固定长度模式。n=None 表示自由编辑。"""
        self._fixed_length = n

    def fixed_length(self) -> Optional[int]:
        return self._fixed_length

    def apply_inline_y_axis_metrics(self) -> None:
        """把 QPlainTextEdit 压成单行文本承载层，而不是默认文本框。"""
        doc = self.document()
        doc.setDocumentMargin(0)
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.SelectionType.Document)
        block_fmt = QTextBlockFormat()
        block_fmt.setLineHeight(
            float(TEXT_LINE_HEIGHT_PX),
            QTextBlockFormat.LineHeightTypes.FixedHeight.value,
        )
        cursor.mergeBlockFormat(block_fmt)

    # ── 编辑约束 ──────────────────────────────────────────────

    def _is_fixed(self) -> bool:
        return self._fixed_length is not None

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        mod = event.modifiers()
        no_mod = mod == Qt.KeyboardModifier.NoModifier
        ctrl   = mod == Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Return and no_mod:
            self.confirm_requested.emit(); return
        if key == Qt.Key.Key_Up and ctrl:
            self.prev_requested.emit(); return
        if key == Qt.Key.Key_Down and ctrl:
            self.next_requested.emit(); return
        if key == Qt.Key.Key_F5:
            self.flag_requested.emit(); return
        if key == Qt.Key.Key_F6:
            self.skip_requested.emit(); return
        if key == Qt.Key.Key_Escape:
            self.revert_requested.emit(); return

        if self._is_fixed():
            # 固定槽位语义：删除/剪切只把槽位填空，普通输入覆写当前槽位，
            # 所有路径都保持 OCR 字符观测数量和文本槽位数量一致。
            blank = " "
            if key == Qt.Key.Key_Backspace:
                cur = self.textCursor()
                if cur.hasSelection():
                    text_len = len(cur.selectedText())
                    cur.insertText(blank * text_len)
                else:
                    pos = cur.position()
                    if pos <= 0:
                        return
                    cur.setPosition(pos - 1)
                    cur.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
                    cur.insertText(blank)
                    # Backspace 行为习惯：光标停在被填空槽位之前
                    cur.setPosition(pos - 1)
                    self.setTextCursor(cur)
                return
            if key == Qt.Key.Key_Delete:
                cur = self.textCursor()
                if cur.hasSelection():
                    text_len = len(cur.selectedText())
                    cur.insertText(blank * text_len)
                else:
                    pos = cur.position()
                    if pos >= len(self.toPlainText()):
                        return
                    cur.setPosition(pos)
                    cur.setPosition(pos + 1, QTextCursor.MoveMode.KeepAnchor)
                    cur.insertText(blank)
                    cur.setPosition(pos + 1)
                    self.setTextCursor(cur)
                return
            # 2) Ctrl+X 剪切：填空（保持长度）
            if ctrl and key == Qt.Key.Key_X:
                cur = self.textCursor()
                if cur.hasSelection():
                    text_len = len(cur.selectedText())
                    cur.insertText(blank * text_len)
                return
            # 3) 普通字符输入（含 IME 单字键）：转为覆写模式
            txt = event.text()
            if txt and txt.isprintable() and not ctrl and key not in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End,
            ):
                cur = self.textCursor()
                if not cur.hasSelection():
                    if cur.position() >= len(self.toPlainText()):
                        # 行末不允许追加（无 tooltip，静默拒绝，保持长度）
                        return
                    cur.setPosition(cur.position())
                    cur.setPosition(cur.position() + 1, QTextCursor.MoveMode.KeepAnchor)
                    self.setTextCursor(cur)
                # 输入长度 > 选区长度 → 裁断到选区长度，不再拒绝
                sel_len = len(cur.selectedText())
                if len(txt) > sel_len:
                    txt = txt[:sel_len]
                if len(txt) < sel_len:
                    # 输入短于选区 → 余位填空（保长度）
                    txt = txt + blank * (sel_len - len(txt))
                cur.insertText(txt)
                return

        super().keyPressEvent(event)

    def insertFromMimeData(self, source) -> None:  # type: ignore[override]
        """粘贴：固定长度模式下保长度。

        固定长度模式下不因长度不匹配拒绝粘贴，而是截断或用空格补足，
        与键盘输入语义一致。
        """
        if not self._is_fixed():
            super().insertFromMimeData(source)
            return
        text = source.text() if source is not None else ""
        if not text:
            return
        text = text.replace("\r", "").replace("\n", "")
        blank = " "
        cur = self.textCursor()
        if not cur.hasSelection():
            doc_len = len(self.toPlainText())
            avail = doc_len - cur.position()
            take = min(len(text), avail)
            if take <= 0:
                return
            cur.setPosition(cur.position() + take, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cur)
            text = text[:take]
        else:
            sel_len = len(cur.selectedText())
            if len(text) > sel_len:
                text = text[:sel_len]
            elif len(text) < sel_len:
                text = text + blank * (sel_len - len(text))
        cur.insertText(text)

    # ── 鼠标悬停 → 字符索引（hproof-visual-marking）─────────
    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._slot_x_centers:
            pos = self._event_pos(event)
            idx = self._slot_index_for_x(float(pos.x()), nearest=False)
        else:
            super().mouseMoveEvent(event)
            pos = self._event_pos(event)
            cur = self.cursorForPosition(pos)
            idx = cur.position()
            # 末尾点击会落到 len(text)；当成离开
            if idx >= len(self.toPlainText()):
                idx = -1
        if idx != self._last_hover_idx:
            self._last_hover_idx = idx
            self.hover_char_changed.emit(idx)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        if self._last_hover_idx != -1:
            self._last_hover_idx = -1
            self.hover_char_changed.emit(-1)

    # mousePressEvent / focusInEvent 都发 row_focus_requested，由 _LinePair
    # 统一激活当前行。
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.row_focus_requested.emit()
        if self._slot_x_centers:
            pos = self._event_pos(event)
            idx = self._slot_index_for_x(float(pos.x()), nearest=True)
            if idx >= 0:
                self._select_slot_index(idx)
                self.setFocus()
                try:
                    event.accept()
                except Exception:
                    pass
                return
        super().mousePressEvent(event)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        self.row_focus_requested.emit()
        super().focusInEvent(event)


class _SlotLineEditor(QWidget):
    """横校逐字 slot 编辑器。

    这个控件只把 ``QTextDocument`` 当作文本状态容器，不使用 Qt 原生文本布局。
    屏幕上的字符、命中区域、选中背景都按 OCR 字符观测 bbox 传入的
    slot geometry 绘制，避免“视觉字位”和 Qt 文本光标坐标不一致。
    """

    confirm_requested = Signal()
    prev_requested = Signal()
    next_requested = Signal()
    flag_requested = Signal()
    skip_requested = Signal()
    revert_requested = Signal()
    hover_char_changed = Signal(int)
    row_focus_requested = Signal()
    visual_edit_exit_requested = Signal()
    formula_source_requested = Signal(int, int, QPoint)
    textChanged = Signal()
    selectionChanged = Signal()
    cursorPositionChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(TEXT_EDITOR_MAX_H)
        self._document = QTextDocument(self)
        self._document.setDocumentMargin(0)
        self._cursor = QTextCursor(self._document)
        self._cursor_width = 0
        self._fixed_length: Optional[int] = None
        self._slot_x_centers: Optional[List[Optional[float]]] = None
        self._slot_widths: Optional[List[float]] = None
        self._extra_selections: list = []
        self._last_hover_idx = -1
        self._active_visual = False
        self._visual_text_override: Optional[str] = None
        self._visual_pixmap_override: Optional[QPixmap] = None
        self._visual_pixmap_logical_size: QSize | None = None
        self._visual_text_kind: str = ""
        self._atom_visual_overlays: list[_AtomVisualOverlay] = []
        self._undo_stack: list[tuple[str, int, int]] = []
        self._redo_stack: list[tuple[str, int, int]] = []
        self._max_undo = 100
        self._read_only = False
        self._apply_document_line_height()

    # ── QPlainTextEdit-like API used by _LinePair ─────────────

    def document(self) -> QTextDocument:
        return self._document

    def setFrameShape(self, *_args, **_kwargs) -> None:
        return

    def setLineWrapMode(self, *_args, **_kwargs) -> None:
        return

    def setVerticalScrollBarPolicy(self, *_args, **_kwargs) -> None:
        return

    def setCursorWidth(self, width: int) -> None:
        self._cursor_width = int(width)

    def cursorWidth(self) -> int:
        return self._cursor_width

    def set_active_visual(self, active: bool) -> None:
        self._active_visual = bool(active)
        self.update()

    def setReadOnly(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self.setCursor(Qt.CursorShape.ArrowCursor if self._read_only else Qt.CursorShape.IBeamCursor)

    def isReadOnly(self) -> bool:
        return self._read_only

    def set_visual_text_override(self, text: Optional[str], *, kind: str = "") -> None:
        next_text = text or None
        next_kind = kind if next_text else ""
        if (
            self._visual_text_override == next_text
            and self._visual_pixmap_override is None
            and self._visual_text_kind == next_kind
        ):
            return
        self._visual_text_override = next_text
        self._visual_pixmap_override = None
        self._visual_pixmap_logical_size = None
        self._visual_text_kind = next_kind
        self.update()

    def set_visual_formula_override(self, visual: _FormulaVisual | None) -> None:
        if visual is None:
            self.set_visual_text_override(None)
            return
        self._visual_text_override = visual.text or None
        self._visual_pixmap_override = visual.pixmap
        self._visual_pixmap_logical_size = visual.logical_size if visual.pixmap is not None else None
        self._visual_text_kind = visual.kind
        self.update()

    def has_visual_text_override(self) -> bool:
        return bool(self._visual_text_override) or self._visual_pixmap_override is not None

    def set_atom_visual_overlays(self, overlays: list[_AtomVisualOverlay] | None) -> None:
        self._atom_visual_overlays = list(overlays or [])
        self.update()

    def has_atom_visual_overlays(self) -> bool:
        return bool(self._atom_visual_overlays)

    def atom_visual_overlays(self) -> list[_AtomVisualOverlay]:
        return list(self._atom_visual_overlays)

    def visual_text_content_width(self) -> int:
        if self._visual_pixmap_override is not None:
            if self._visual_pixmap_logical_size is not None:
                return self._visual_pixmap_logical_size.width() + 16
            dpr = max(1.0, float(self._visual_pixmap_override.devicePixelRatio()))
            return int(round(self._visual_pixmap_override.width() / dpr)) + 16
        if not self._visual_text_override:
            return 0
        font = QFont(self.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        if self._visual_text_kind == "formula":
            font.setItalic(True)
        return QFontMetrics(font).horizontalAdvance(self._visual_text_override) + 16

    def setPlainText(self, text: str) -> None:
        old = self.toPlainText()
        self._document.setPlainText(text or "")
        self._apply_document_line_height()
        self._cursor = QTextCursor(self._document)
        if self._document.characterCount() > 1:
            self._cursor.setPosition(0)
            self._cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        if self.toPlainText() != old:
            self.textChanged.emit()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.selectionChanged.emit()
        self.cursorPositionChanged.emit()
        self.update()

    def toPlainText(self) -> str:
        return self._document.toPlainText()

    def textCursor(self) -> QTextCursor:
        return QTextCursor(self._cursor)

    def setTextCursor(self, cursor: QTextCursor) -> None:
        self._cursor = QTextCursor(cursor)
        self.selectionChanged.emit()
        self.cursorPositionChanged.emit()
        self.update()

    def setExtraSelections(self, selections) -> None:
        self._extra_selections = list(selections or [])
        self.update()

    def extraSelections(self):
        return list(self._extra_selections)

    def set_fixed_length(self, n: Optional[int]) -> None:
        self._fixed_length = n

    def fixed_length(self) -> Optional[int]:
        return self._fixed_length

    def apply_inline_y_axis_metrics(self) -> None:
        self._apply_document_line_height()

    def set_slot_geometry(
        self,
        x_centers: Optional[List[Optional[float]]],
        widths: Optional[List[float]],
    ) -> None:
        if not x_centers:
            self._slot_x_centers = None
            self._slot_widths = None
        else:
            self._slot_x_centers = list(x_centers)
            self._slot_widths = list(widths) if widths else [TEXT_SLOT_MIN_W] * len(x_centers)
        self.update()

    def has_slot_geometry(self) -> bool:
        return bool(self._slot_x_centers)

    def _apply_document_line_height(self) -> None:
        cursor = QTextCursor(self._document)
        cursor.select(QTextCursor.SelectionType.Document)
        block_fmt = QTextBlockFormat()
        block_fmt.setLineHeight(
            float(TEXT_LINE_HEIGHT_PX),
            QTextBlockFormat.LineHeightTypes.FixedHeight.value,
        )
        cursor.mergeBlockFormat(block_fmt)

    # ── slot 选择 / 文本修改 ───────────────────────────────────

    def _slot_index_for_x(self, x: float, *, nearest: bool = False) -> int:
        overlay = self._atom_overlay_for_x(x)
        if overlay is not None:
            return overlay.start
        text = self.toPlainText()
        centers = self._slot_x_centers or self._fallback_slot_centers(text)
        if not centers:
            return -1
        if self._slot_widths is not None:
            widths = self._slot_widths
        else:
            font = QFont(self.font())
            font.setWeight(TEXT_FONT_WEIGHT)
            fm = QFontMetrics(font)
            widths = [_slot_visual_width(ch, fm) for ch in text]
        best_idx = -1
        best_dist = float("inf")
        first_left: float | None = None
        last_right: float | None = None
        for idx, center in enumerate(centers):
            if center is None:
                continue
            width = widths[idx] if idx < len(widths) else TEXT_SLOT_MIN_W
            half = max(TEXT_SLOT_MIN_W / 2.0, float(width) / 2.0)
            left = float(center) - half
            right = float(center) + half
            first_left = left if first_left is None else min(first_left, left)
            last_right = right if last_right is None else max(last_right, right)
            if left <= x <= right:
                return idx
            dist = abs(float(center) - x)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if not nearest or best_idx < 0:
            return -1
        margin = max(24.0, TEXT_SLOT_MIN_W * 2.0)
        if first_left is not None and last_right is not None:
            if x < first_left - margin or x > last_right + margin:
                return -1
        return best_idx

    def _atom_overlay_for_x(self, x: float) -> _AtomVisualOverlay | None:
        for overlay in self._atom_visual_overlays:
            left = min(overlay.left, overlay.right)
            right = max(overlay.left, overlay.right)
            if left <= x <= right:
                return overlay
        return None

    def _select_slot_index(self, idx: int) -> None:
        text_len = len(self.toPlainText())
        if idx < 0 or idx >= text_len:
            return
        cur = QTextCursor(self._document)
        cur.setPosition(idx)
        cur.setPosition(idx + 1, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cur)

    def _selected_range(self) -> tuple[int, int]:
        if self._cursor.hasSelection():
            return self._cursor.selectionStart(), self._cursor.selectionEnd()
        pos = self._cursor.position()
        return pos, min(pos + 1, len(self.toPlainText()))

    def _replace_selected_slots(self, text: str) -> None:
        current = self.toPlainText()
        if not current:
            return
        before = self._snapshot()
        start, end = self._selected_range()
        start = max(0, min(start, len(current)))
        end = max(start, min(end, len(current)))
        if start == end:
            if start >= len(current):
                return
            end = start + 1
        width = end - start
        replacement = (text or "")[:width]
        if len(replacement) < width:
            replacement += " " * (width - len(replacement))
        new_text = current[:start] + replacement + current[end:]
        if new_text != current:
            self._push_undo_snapshot(before)
        self._document.setPlainText(new_text)
        self._apply_document_line_height()
        next_pos = min(len(new_text), start + max(1, len(replacement)))
        self._cursor = QTextCursor(self._document)
        if next_pos < len(new_text):
            self._cursor.setPosition(next_pos)
            self._cursor.setPosition(next_pos + 1, QTextCursor.MoveMode.KeepAnchor)
        else:
            self._cursor.setPosition(len(new_text))
        self.textChanged.emit()
        self.selectionChanged.emit()
        self.cursorPositionChanged.emit()
        self.update()

    def replace_text_range(self, start: int, end: int, replacement: str) -> None:
        """Replace a raw text range without fixed-slot truncation.

        This path is reserved for formula source editing, where the source
        length is allowed to change and the rendered atom is recomputed from the
        new source text.
        """
        current = self.toPlainText()
        start = max(0, min(int(start), len(current)))
        end = max(start, min(int(end), len(current)))
        replacement = replacement or ""
        if current[start:end] == replacement:
            return
        before = self._snapshot()
        self._push_undo_snapshot(before)
        new_text = current[:start] + replacement + current[end:]
        self._document.setPlainText(new_text)
        self._apply_document_line_height()
        self._cursor = QTextCursor(self._document)
        next_start = start
        next_end = start + len(replacement)
        self._cursor.setPosition(next_start)
        if next_end != next_start:
            self._cursor.setPosition(next_end, QTextCursor.MoveMode.KeepAnchor)
        self.textChanged.emit()
        self.selectionChanged.emit()
        self.cursorPositionChanged.emit()
        self.update()

    def _snapshot(self) -> tuple[str, int, int]:
        return (
            self.toPlainText(),
            int(self._cursor.selectionStart()),
            int(self._cursor.selectionEnd()),
        )

    def _push_undo_snapshot(self, snapshot: tuple[str, int, int]) -> None:
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack = self._undo_stack[-self._max_undo:]
        self._redo_stack.clear()

    def _restore_snapshot(self, snapshot: tuple[str, int, int]) -> None:
        text, start, end = snapshot
        self._document.setPlainText(text)
        self._apply_document_line_height()
        text_len = len(self.toPlainText())
        start = max(0, min(int(start), text_len))
        end = max(start, min(int(end), text_len))
        self._cursor = QTextCursor(self._document)
        self._cursor.setPosition(start)
        if end != start:
            self._cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        self.textChanged.emit()
        self.selectionChanged.emit()
        self.cursorPositionChanged.emit()
        self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore_snapshot(self._undo_stack.pop())

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore_snapshot(self._redo_stack.pop())

    def insertFromMimeData(self, source) -> None:
        text = source.text() if source is not None else ""
        text = text.replace("\r", "").replace("\n", "")
        if text:
            self._replace_selected_slots(text)

    def inputMethodQuery(self, query):  # type: ignore[override]
        if query == Qt.InputMethodQuery.ImEnabled:
            return True
        if query == Qt.InputMethodQuery.ImSurroundingText:
            return self.toPlainText()
        if query == Qt.InputMethodQuery.ImCurrentSelection:
            return self._cursor.selectedText() if self._cursor.hasSelection() else ""
        if query == Qt.InputMethodQuery.ImCursorRectangle:
            return self._cursor_rect()
        return super().inputMethodQuery(query)

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        commit = event.commitString() if event is not None else ""
        if commit:
            self._replace_selected_slots(commit)
            event.accept()
            return
        super().inputMethodEvent(event)

    # ── 绘制 ───────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), self.palette().base())
            if self._visual_pixmap_override is not None:
                self._paint_visual_pixmap_override(p, self._visual_pixmap_override)
                return
            if self._visual_text_override:
                self._paint_visual_text_override(p, self._visual_text_override)
                return
            text = self.toPlainText()
            font = QFont(self.font())
            font.setWeight(TEXT_FONT_WEIGHT)
            p.setFont(font)
            fm = QFontMetrics(font)
            centers = self._slot_x_centers or self._fallback_slot_centers(text)
            widths = self._slot_widths or [
                _slot_visual_width(ch, fm)
                for ch in text
            ]
            explicit_widths = self._slot_widths is not None
            fg_color, bg_color = self._selection_colors()
            selected_start, selected_end = self._selection_bounds_for_paint() if self._active_visual else (-1, -1)
            y_baseline = (self.height() + fm.ascent() - fm.descent()) // 2
            n = min(len(text), len(centers))
            hidden_indices = {
                idx
                for overlay in self._atom_visual_overlays
                for idx in range(max(0, overlay.start), min(len(text), overlay.end))
            }
            for i in range(n):
                if i in hidden_indices:
                    continue
                center = centers[i]
                if center is None:
                    continue
                width = widths[i] if i < len(widths) else TEXT_SLOT_MIN_W
                min_slot_w = TEXT_SLOT_CLIPPED_MIN_W if explicit_widths else TEXT_SLOT_MIN_W
                slot_w = max(min_slot_w, float(width))
                left = int(round(float(center) - slot_w / 2.0))
                cell = QRect(left, 2, max(1, int(round(slot_w))), self.height() - 4)
                if i == self._last_hover_idx:
                    p.fillRect(cell, QColor("#e8f0fe"))
                if selected_start <= i < selected_end:
                    p.fillRect(cell, QColor("#cfe2ff"))
                bg = bg_color.get(i)
                if bg is not None and bg.alpha() > 0:
                    p.fillRect(cell, bg)
                if i == self._last_hover_idx or selected_start <= i < selected_end:
                    p.setPen(QPen(QColor("#9cc2ff"), 1))
                    p.drawRect(cell.adjusted(0, 0, -1, -1))
                color = fg_color.get(i) or self.palette().text().color()
                p.setPen(QPen(color, 1))
                ch = text[i]
                if _is_punctuation_slot_text(ch):
                    ink = fm.tightBoundingRect(ch)
                    tx = float(center) - ink.width() / 2.0 - ink.left()
                    p.drawText(int(round(tx)), y_baseline, ch)
                else:
                    char_w = fm.horizontalAdvance(ch)
                    tx = int(round(float(center) - char_w / 2.0))
                    p.drawText(tx, y_baseline, ch)
            for overlay in self._atom_visual_overlays:
                self._paint_atom_visual_overlay(
                    p,
                    overlay,
                    selected_start=selected_start,
                    selected_end=selected_end,
                )
        finally:
            p.end()

    def _paint_visual_text_override(self, painter: QPainter, text: str) -> None:
        font = QFont(self.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        if self._visual_text_kind == "formula":
            font.setItalic(True)
        painter.setFont(font)
        painter.setPen(QPen(self.palette().text().color(), 1))
        rect = self.rect().adjusted(6, 0, -6, 0)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _paint_visual_pixmap_override(self, painter: QPainter, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        if self._visual_pixmap_logical_size is not None:
            width = self._visual_pixmap_logical_size.width()
            height = self._visual_pixmap_logical_size.height()
        else:
            dpr = max(1.0, float(pixmap.devicePixelRatio()))
            width = int(round(pixmap.width() / dpr))
            height = int(round(pixmap.height() / dpr))
        target = QRect(6, max(0, (self.height() - height) // 2), max(1, width), max(1, height))
        painter.drawPixmap(target, pixmap)

    def _paint_atom_visual_overlay(
        self,
        painter: QPainter,
        overlay: _AtomVisualOverlay,
        *,
        selected_start: int,
        selected_end: int,
    ) -> None:
        left = int(round(min(overlay.left, overlay.right)))
        right = int(round(max(overlay.left, overlay.right)))
        rect = QRect(left, 2, max(1, right - left), self.height() - 4)
        hovered = overlay.start <= self._last_hover_idx < overlay.end
        selected = selected_start < overlay.end and selected_end > overlay.start
        if hovered:
            painter.fillRect(rect, QColor("#e8f0fe"))
        if selected:
            painter.fillRect(rect, QColor("#cfe2ff"))
        if hovered or selected:
            painter.setPen(QPen(QColor("#9cc2ff"), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        if overlay.pixmap is not None and not overlay.pixmap.isNull():
            pixmap = overlay.pixmap
            max_w = max(1, rect.width() - 4)
            max_h = max(1, rect.height() - 4)
            target_w = overlay.logical_size.width() if overlay.logical_size is not None else max_w
            target_h = overlay.logical_size.height() if overlay.logical_size is not None else max_h
            scale = min(max_w / max(1, target_w), max_h / max(1, target_h), 1.0)
            draw_w = max(1, int(round(target_w * scale)))
            draw_h = max(1, int(round(target_h * scale)))
            target = QRect(
                rect.x() + max(0, (rect.width() - draw_w) // 2),
                rect.y() + max(0, (rect.height() - draw_h) // 2),
                draw_w,
                draw_h,
            )
            painter.drawPixmap(target, pixmap)
            return
        font = QFont(self.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        if overlay.kind == "formula":
            font.setItalic(True)
        painter.setFont(font)
        painter.setPen(QPen(self.palette().text().color(), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, overlay.text)

    def _fallback_slot_centers(self, text: str) -> list[Optional[float]]:
        if not text:
            return []
        font = QFont(self.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        fm = QFontMetrics(font)
        x = max(6.0, TEXT_SLOT_MIN_W / 2.0)
        centers: list[Optional[float]] = []
        for ch in text:
            w = _slot_visual_width(ch, fm)
            centers.append(x + w / 2.0)
            x += w
        return centers

    def _selection_colors(self) -> tuple[dict[int, QColor], dict[int, QColor]]:
        fg_color: dict[int, QColor] = {}
        bg_color: dict[int, QColor] = {}
        for sel in self._extra_selections:
            cur = sel.cursor
            start = cur.selectionStart()
            end = cur.selectionEnd()
            fmt = sel.format
            if fmt.foreground().style() != Qt.BrushStyle.NoBrush:
                color = fmt.foreground().color()
                for i in range(start, end):
                    fg_color[i] = color
            if fmt.background().style() != Qt.BrushStyle.NoBrush:
                color = fmt.background().color()
                for i in range(start, end):
                    bg_color[i] = color
        return fg_color, bg_color

    def _selection_bounds_for_paint(self) -> tuple[int, int]:
        if self._cursor.hasSelection():
            return self._cursor.selectionStart(), self._cursor.selectionEnd()
        pos = self._cursor.position()
        return pos, min(pos + 1, len(self.toPlainText()))

    def _cursor_rect(self) -> QRect:
        pos = self._cursor.selectionStart() if self._cursor.hasSelection() else self._cursor.position()
        centers = self._slot_x_centers or self._fallback_slot_centers(self.toPlainText())
        fm = QFontMetrics(self.font())
        explicit_widths = self._slot_widths is not None
        widths = self._slot_widths or [
            _slot_visual_width(ch, fm)
            for ch in self.toPlainText()
        ]
        if 0 <= pos < len(centers):
            center = centers[pos]
            if center is not None:
                width = widths[pos] if pos < len(widths) else TEXT_SLOT_MIN_W
                min_slot_w = TEXT_SLOT_CLIPPED_MIN_W if explicit_widths else TEXT_SLOT_MIN_W
                width = max(min_slot_w, float(width))
                left = int(round(float(center) - width / 2.0))
                return QRect(left, 2, max(1, int(round(width))), self.height() - 4)
        return QRect(0, 0, 1, self.height())

    # ── 事件 ───────────────────────────────────────────────────

    def _event_pos(self, event):
        try:
            return event.position().toPoint()
        except AttributeError:
            return event.pos()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            pos = self._event_pos(event)
            text_len = len(self.toPlainText())
            overlay = self._atom_overlay_for_x(float(pos.x()))
            if overlay is not None and overlay.kind == "formula":
                self.formula_source_requested.emit(
                    max(0, min(overlay.start, text_len)),
                    max(0, min(overlay.end, text_len)),
                    self.mapToGlobal(pos),
                )
            elif self.has_visual_text_override():
                self.formula_source_requested.emit(0, text_len, self.mapToGlobal(pos))
            else:
                self.visual_edit_exit_requested.emit()
            try:
                event.accept()
            except Exception:
                pass
            return
        self.row_focus_requested.emit()
        pos = self._event_pos(event)
        overlay = self._atom_overlay_for_x(float(pos.x()))
        if overlay is not None:
            cur = QTextCursor(self._document)
            cur.setPosition(max(0, min(overlay.start, len(self.toPlainText()))))
            cur.setPosition(
                max(0, min(overlay.end, len(self.toPlainText()))),
                QTextCursor.MoveMode.KeepAnchor,
            )
            self.setTextCursor(cur)
            self.setFocus()
            try:
                event.accept()
            except Exception:
                pass
            return
        idx = self._slot_index_for_x(float(pos.x()), nearest=True)
        if idx >= 0:
            self._select_slot_index(idx)
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = self._event_pos(event)
        idx = self._slot_index_for_x(float(pos.x()), nearest=False)
        if idx != self._last_hover_idx:
            self._last_hover_idx = idx
            self.hover_char_changed.emit(idx)
            self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        if self._last_hover_idx != -1:
            self._last_hover_idx = -1
            self.hover_char_changed.emit(-1)
            self.update()

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        self.row_focus_requested.emit()
        super().focusInEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        mod = event.modifiers()
        no_mod = mod == Qt.KeyboardModifier.NoModifier
        ctrl = bool(mod & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mod & Qt.KeyboardModifier.ShiftModifier)
        if self._read_only:
            if ctrl and key == Qt.Key.Key_C:
                selected = self._cursor.selectedText() if self._cursor.hasSelection() else ""
                QApplication.clipboard().setText(selected)
                return
            if ctrl and key == Qt.Key.Key_A:
                cur = QTextCursor(self._document)
                cur.setPosition(0)
                cur.setPosition(len(self.toPlainText()), QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cur)
                return
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                if key == Qt.Key.Key_Left:
                    self._select_slot_index(max(0, self._cursor.selectionStart() - 1))
                else:
                    self._select_slot_index(min(max(0, len(self.toPlainText()) - 1), self._cursor.selectionEnd()))
                return
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and no_mod:
            self.confirm_requested.emit()
            return
        if ctrl and key == Qt.Key.Key_Z:
            if shift:
                self.redo()
            else:
                self.undo()
            return
        if ctrl and key == Qt.Key.Key_Y:
            self.redo()
            return
        if key == Qt.Key.Key_Up and ctrl:
            self.prev_requested.emit()
            return
        if key == Qt.Key.Key_Down and ctrl:
            self.next_requested.emit()
            return
        if key == Qt.Key.Key_F5:
            self.flag_requested.emit()
            return
        if key == Qt.Key.Key_F6:
            self.skip_requested.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.revert_requested.emit()
            return
        if ctrl and key == Qt.Key.Key_A:
            cur = QTextCursor(self._document)
            cur.setPosition(0)
            cur.setPosition(len(self.toPlainText()), QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cur)
            return
        if ctrl and key == Qt.Key.Key_C:
            selected = self._cursor.selectedText() if self._cursor.hasSelection() else ""
            QApplication.clipboard().setText(selected)
            return
        if ctrl and key == Qt.Key.Key_V:
            text = QApplication.clipboard().text()
            if text:
                self._replace_selected_slots(text.replace("\r", "").replace("\n", ""))
            return
        if ctrl and key == Qt.Key.Key_X:
            selected = self._cursor.selectedText() if self._cursor.hasSelection() else ""
            if selected:
                QApplication.clipboard().setText(selected)
            self._replace_selected_slots(" ")
            return
        if key == Qt.Key.Key_Left and no_mod:
            self._select_slot_index(max(0, self._cursor.selectionStart() - 1))
            return
        if key == Qt.Key.Key_Right and no_mod:
            self._select_slot_index(min(max(0, len(self.toPlainText()) - 1), self._cursor.selectionEnd()))
            return
        if key == Qt.Key.Key_Backspace and no_mod:
            if self._cursor.hasSelection():
                self._replace_selected_slots(" ")
            else:
                self._select_slot_index(max(0, self._cursor.position() - 1))
                self._replace_selected_slots(" ")
            return
        if key == Qt.Key.Key_Delete and no_mod:
            self._replace_selected_slots(" ")
            return
        text = event.text()
        if text and text.isprintable() and not ctrl:
            self._replace_selected_slots(text)
            return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────
# 评测位 (quality probe) 显示↔真实 桥接
# 显示文本仍由 app.services.proof_probe_text_service 提供；写入统一走
# ProofEditService，避免横校绕过 proof change contract。
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 单行对（图像行 + 识别文本行）控件
# ─────────────────────────────────────────────────────────────

class ProofUnitKind(str, Enum):
    """横校画布中的可渲染校对单元类型。"""

    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"


@dataclass(frozen=True)
class _FormulaVisual:
    text: str
    pixmap: QPixmap | None = None
    logical_size: QSize | None = None
    kind: str = "formula"


@dataclass(frozen=True)
class _AtomVisualOverlay:
    start: int
    end: int
    left: float
    right: float
    text: str
    pixmap: QPixmap | None = None
    logical_size: QSize | None = None
    kind: str = ""


class _FormulaSourcePopup(QFrame):
    """Small formula source editor opened from a rendered formula atom."""

    source_changed = Signal(str)
    closed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("formulaSourcePopup")
        self.setStyleSheet(
            "QFrame#formulaSourcePopup { background:#FFFDF8; border:1px solid #CFC7BA; "
            "border-radius:8px; }"
            "QLabel { color:#5F574D; font-size:11px; font-weight:600; }"
            "QPlainTextEdit { background:#FFFFFF; border:1px solid #DED6CA; "
            "border-radius:6px; padding:6px; font-family:'JetBrains Mono','Consolas',monospace; "
            "font-size:13px; color:#27231E; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        label = QLabel("公式源码")
        layout.addWidget(label)
        self._source_edit = QPlainTextEdit()
        self._source_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._source_edit.setFixedSize(420, 76)
        self._source_edit.installEventFilter(self)
        layout.addWidget(self._source_edit)
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(160)
        self._emit_timer.timeout.connect(self._emit_source_changed)
        self._source_edit.textChanged.connect(self._schedule_source_changed)

    def open_for(self, source: str, global_pos: QPoint, *, editable: bool = True) -> None:
        self._source_edit.blockSignals(True)
        self._source_edit.setPlainText(source or "")
        self._source_edit.blockSignals(False)
        self._source_edit.setReadOnly(not editable)
        self.adjustSize()
        self.move(global_pos + FORMULA_SOURCE_POPUP_OFFSET)
        self.show()
        self.raise_()
        self._source_edit.setFocus()
        cursor = self._source_edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self._source_edit.setTextCursor(cursor)

    def source_text(self) -> str:
        return self._source_edit.toPlainText().replace("\r", "").replace("\n", " ").strip()

    def _schedule_source_changed(self) -> None:
        self._emit_timer.start()

    def _emit_source_changed(self) -> None:
        self.source_changed.emit(self.source_text())

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._source_edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if key == Qt.Key.Key_Escape or (ctrl and key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}):
                self.hide()
                return True
        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        if self._emit_timer.isActive():
            self._emit_timer.stop()
            self._emit_source_changed()
        super().hideEvent(event)
        self.closed.emit()


class _FormulaSourceInlinePanel(QFrame):
    """Embedded source editor for display formula rows.

    Display formulas are usually wider than a line, so they need a full-width
    third pane instead of the small inline popup used by inline formulas.
    """

    source_changed = Signal(str)
    close_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("formulaSourceInlinePanel")
        self.setFixedHeight(FORMULA_SOURCE_PANEL_H)
        self.setStyleSheet(
            "QFrame#formulaSourceInlinePanel { background:#FBF8F1; "
            "border-top:1px solid #DED6CA; }"
            "QLabel { color:#5F574D; font-size:11px; font-weight:600; }"
            "QPlainTextEdit { background:#FFFFFF; border:1px solid #DED6CA; "
            "border-radius:5px; padding:5px 6px; font-family:'JetBrains Mono','Consolas',monospace; "
            "font-size:13px; color:#27231E; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 7)
        layout.setSpacing(5)
        label = QLabel("公式源码")
        layout.addWidget(label)
        self._source_edit = QPlainTextEdit()
        self._source_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._source_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._source_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._source_edit.setFixedHeight(48)
        self._source_edit.installEventFilter(self)
        layout.addWidget(self._source_edit)
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(160)
        self._emit_timer.timeout.connect(self.flush_source_changed)
        self._source_edit.textChanged.connect(self._schedule_source_changed)
        self.hide()

    def open_for(self, source: str, *, editable: bool = True) -> None:
        self._source_edit.blockSignals(True)
        self._source_edit.setPlainText(source or "")
        self._source_edit.blockSignals(False)
        self._source_edit.setReadOnly(not editable)
        self.show()
        self._source_edit.setFocus()
        cursor = self._source_edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self._source_edit.setTextCursor(cursor)

    def source_text(self) -> str:
        return self._source_edit.toPlainText().replace("\r", "").replace("\n", " ").strip()

    def flush_source_changed(self) -> None:
        if self._emit_timer.isActive():
            self._emit_timer.stop()
        self.source_changed.emit(self.source_text())

    def _schedule_source_changed(self) -> None:
        self._emit_timer.start()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._source_edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if key == Qt.Key.Key_Escape or (ctrl and key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}):
                self.flush_source_changed()
                self.close_requested.emit()
                return True
        return super().eventFilter(watched, event)


@dataclass(frozen=True)
class ProofUnit:
    """横校内部的稳定渲染输入。

    当前只把正文行接入 renderer；公式、表格、图片先占位，后续可在不改
    HProofPanel 保存/统计逻辑的前提下接入专用渲染器。
    """

    uid: str
    kind: ProofUnitKind
    page: Page
    block: Block
    line: Line
    line_index: int
    page_line_number: int
    display_text: str
    debug_badge: str = ""
    atoms: tuple[ProofAtom, ...] = ()
    editable: bool = True

def _proof_unit_kind(projection: ProofLineProjection, debug_badge: str) -> ProofUnitKind:
    if debug_badge == "公式" and _is_display_formula_unit(projection.block, projection.line):
        return ProofUnitKind.FORMULA
    if debug_badge == "表格":
        return ProofUnitKind.TABLE
    return ProofUnitKind.TEXT


class _LinePair(QFrame):
    """显示一行的【扫描图像行 + OCR 识别文本】对。"""

    clicked       = Signal(int)   # 发出自身 idx
    text_saved    = Signal(int, str)  # (idx, new_text)
    confirmed     = Signal(int)   # Enter 确认
    prev_req      = Signal()
    next_req      = Signal()
    flag_req      = Signal()
    skip_req      = Signal()

    def __init__(
        self,
        idx: int,
        block: Block,
        line: Line,
        page: Page,
        line_in_page: int,   # 在页面内的行序号（1-based，用于显示）
        cache: PageImageCache,
        debug_badge: str = "",
        unit: ProofUnit | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._idx          = idx
        self._block        = block
        self._line         = line
        self._page         = page
        self._line_in_page = line_in_page
        self._cache        = cache
        self._debug_badge  = debug_badge
        self._unit         = unit
        self._edit_session = HProofLineEditSession(
            editable=True if unit is None else bool(unit.editable),
            loaded_display_text="",
            loaded_line_signature="",
        )
        self._active       = False
        self._image_loaded = False
        self._line_crop = None
        self._line_crop_origin: tuple[int, int] = (0, 0)
        self._focus_depth = "active"
        self._image_row_h = IMAGE_ROW_H
        self._editor_h = TEXT_EDITOR_MAX_H
        self._pair_h = LINE_PAIR_H
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)
        # 最近一次行图像缩放比例，用于把 _img_lbl 上的点击位置反查回原图坐标
        self._render_scale: float = 1.0
        # hproof-visual-marking：editor 鼠标悬停的字符索引（-1 = 未悬停）。
        # _render_line_image 会按 active 选区/光标 + 这个 hover idx 联合画框。
        self._hover_char_idx: int = -1
        self._formula_source_popup: _FormulaSourcePopup | None = None
        self._formula_source_panel: _FormulaSourceInlinePanel | None = None
        self._formula_source_range: tuple[int, int] | None = None
        self._formula_source_anchor_span: tuple[int, int] | None = None
        self._formula_source_anchor_rect: tuple[float, float] | None = None

        self.setObjectName("linePair")
        self.setStyleSheet(
            "QFrame#linePair { background:#FFFFFF; border:1px solid transparent; "
            f"border-bottom:1px solid {ROW_DIVIDER_COLOR}; border-radius:0; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    # ── 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFixedHeight(self._pair_h)
        # 横校当前固定为“上图下字”：行图像在上，slot editor 始终可见。
        #   root QHBoxLayout = [active_bar | content_v(img_row, editor) | status]
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 8, 4)
        root.setSpacing(6)

        # 蓝色激活条（左边框）
        self._active_bar = QWidget()
        self._active_bar.setFixedWidth(4)
        self._active_bar.setStyleSheet("background:transparent;")
        root.addWidget(self._active_bar)
        self._active_bar2 = self._active_bar

        # 中间内容容器：上图下字
        self._content = QWidget()
        self._content.setStyleSheet("background:#FFFFFF;")
        content_v = QVBoxLayout(self._content)
        content_v.setContentsMargins(0, 0, 0, 0)
        content_v.setSpacing(0)

        # ── 上：行图像（去掉左侧"图像 N"hdr，直接占满宽度）──────
        self._img_lbl = QLabel()
        self._img_lbl.setFixedHeight(self._image_row_h)
        self._img_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._img_lbl.setStyleSheet(
            f"background:#FFFFFF; border-bottom:1px solid {IMAGE_DIVIDER_COLOR};"
        )
        content_v.addWidget(self._img_lbl)

        # 不再额外渲染第三行文本；图字对应只通过 editor↔image hover/click 联动表达。

        # ── 下：整行文本框（永远可见；弱光标 + 等宽 + 与图像 y 对齐）──
        self._editor = _SlotLineEditor()
        # 弱化“文本框感”：无边框，底色跟随激活态，让用户看到的是
        # 一行可改文字，而不是一个独立输入框。
        self._editor.setStyleSheet(self._editor_style(active=False))
        self._editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._editor.setFixedHeight(self._editor_h)
        self._editor.document().setDocumentMargin(0)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # 弱光标：cursor width 0，通过 extraSelections + cursor.setPosition 表达当前位置。
        self._editor.setCursorWidth(0)
        self._editor.setReadOnly(not self._edit_session.editable)
        # 初始填入显示空间文本
        # 加载时把文本规范化到槽位数；仅在可锁定时补空，超长不动。
        _initial_disp = _displayed_text(self._line, self._page, self._block)
        _canon, _ = _canonicalize_text_to_slots(_initial_disp, self._line_chars())
        self._editor.setPlainText(_canon)
        self._edit_session.mark_saved(_canon, line_signature(self._line))
        self._apply_editor_visual_override()
        self._editor.apply_inline_y_axis_metrics()
        # 信号转发
        self._editor.confirm_requested.connect(lambda: self.confirmed.emit(self._idx))
        self._editor.prev_requested.connect(self.prev_req)
        self._editor.next_requested.connect(self.next_req)
        self._editor.flag_requested.connect(self.flag_req)
        self._editor.skip_requested.connect(self.skip_req)
        self._editor.revert_requested.connect(self._revert)
        self._editor.visual_edit_exit_requested.connect(self._exit_formula_edit_mode)
        self._editor.formula_source_requested.connect(self._open_formula_source_editor)
        # hproof-visual-marking：鼠标悬停 editor → 在行图上高亮对应字
        self._editor.hover_char_changed.connect(self._on_editor_hover_char)
        self._editor.selectionChanged.connect(self._refresh_extra_selections)
        self._editor.selectionChanged.connect(self._render_line_image)
        self._editor.cursorPositionChanged.connect(self._refresh_extra_selections)
        self._editor.cursorPositionChanged.connect(self._render_line_image)
        # 编辑触发置信度高亮重绘（修过的字按 OK 颜色处理）
        self._editor.textChanged.connect(self._refresh_extra_selections)
        # 文本变化后重新评估图字对齐和当前公式/文本显示模式。
        self._editor.textChanged.connect(self._sync_editor_slot_geometry)
        self._editor.textChanged.connect(self._apply_editor_visual_override)
        self._editor.textChanged.connect(self._clear_external_conflict_if_resolved)
        self._editor.textChanged.connect(self._refresh_status)
        # editor focus/click → 激活本行。
        self._editor.row_focus_requested.connect(self._on_editor_focus_in)
        # 有可靠 char boxes 时锁定编辑器长度，保持图字一一对应。
        self._apply_fixed_length_to_editor()
        self._sync_editor_slot_geometry()
        content_v.addWidget(self._editor)

        root.addWidget(self._content, 1)

        # 状态标签
        self._status_lbl = QLabel()
        self._status_lbl.setFixedWidth(STATUS_W)
        self._status_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._refresh_status()
        root.addWidget(self._status_lbl)

        # 行级单击（active_bar / 整体）走 _on_click 激活本行；图像点击走
        # _img_clicked_lookup（仍按字 bbox 反查 + cursor 定位到该字）。
        for w in (self, self._active_bar):
            w.mousePressEvent = self._on_click  # type: ignore[method-assign]
        self._img_lbl.mousePressEvent = self._img_clicked_lookup  # type: ignore[method-assign]

        # 首次绘制置信度底色
        self._refresh_extra_selections()

    # ── 对外接口 ──────────────────────────────────────────────

    @staticmethod
    def _editor_style(*, active: bool) -> str:
        bg = "#F6F2EA" if active else "#FFFFFF"
        return (
            f"font-family:{TEXT_FONT_FAMILY}; font-size:{TEXT_FONT_PX}px; "
            f"font-weight:{TEXT_FONT_WEIGHT_CSS}; padding:0 2px; background:{bg}; border:none;"
        )

    def _formula_source_panel_open(self) -> bool:
        return bool(
            self._formula_source_panel is not None
            and not self._formula_source_panel.isHidden()
            and self._focus_depth == "active"
        )

    def _apply_pair_height(self) -> None:
        extra = FORMULA_SOURCE_PANEL_H if self._formula_source_panel_open() else 0
        self.setFixedHeight(self._pair_h + extra)

    def _apply_editor_visual_override(self) -> None:
        visual: _FormulaVisual | None = None
        if self._unit is not None and self._unit.kind == ProofUnitKind.FORMULA:
            raw_formula = self._editor.toPlainText()
            visual = _render_formula_visual(raw_formula, target_height=_formula_visual_target_height(self._editor_h))
        self._editor.set_visual_formula_override(visual)
        if visual is not None:
            self._editor.set_atom_visual_overlays(None)
        if visual is not None:
            self._editor.set_slot_geometry(None, None)
            content_w = max(self._editor.visual_text_content_width(), int(self._line.bbox.w))
            self._editor.setMinimumWidth(content_w)
            self._content.setMinimumWidth(content_w)
            self.setMinimumWidth(content_w + STATUS_W + 36)
        else:
            self._editor.setMinimumWidth(0)
            self._content.setMinimumWidth(0)
            self.setMinimumWidth(0)

    def _formula_atom_visual_data(
        self,
        text: str,
    ) -> tuple[list[_AtomVisualOverlay], list[Optional[float]] | None, list[float] | None]:
        chars = self._line_chars()
        if self._unit is None or not self._unit.atoms or not chars:
            return self._fallback_formula_visual_data(text), None, None
        span_by_char_index = self._span_by_char_index_for_formula_text(text)
        if span_by_char_index is None:
            return self._fallback_formula_visual_data(text), None, None
        centers: list[Optional[float]] = [None] * len(text)
        widths: list[float] = [0.0] * len(text)
        scale = float(self._render_scale or 0.0)
        ox, _oy = self._line_crop_origin
        fm = QFontMetrics(self._editor.font())
        if scale > 0:
            for atom in self._unit.atoms:
                if atom.kind == ProofAtomKind.FORMULA:
                    continue
                if atom.bbox is None or not atom.char_indices:
                    continue
                span = span_by_char_index.get(atom.char_indices[0])
                if span is None:
                    continue
                span_start, span_end = span
                if span_end - span_start != 1:
                    continue
                idx = span_start
                if not (0 <= idx < len(text)):
                    continue
                centers[idx] = ((atom.bbox.x + atom.bbox.x2) / 2.0 - float(ox)) * scale
                widths[idx] = _formula_overlay_slot_width(text[idx], atom, fm, scale)

        overlays: list[_AtomVisualOverlay] = []
        for atom in self._unit.atoms:
            if atom.kind != ProofAtomKind.FORMULA or not atom.char_indices:
                continue
            span = span_by_char_index.get(atom.char_indices[0])
            if span is None:
                continue
            start = max(0, min(span[0], len(text)))
            end = max(start, min(span[1], len(text)))
            if end <= start:
                continue
            raw = text[start:end]
            left, right = self._formula_atom_rect(atom, text, start, end)
            visual = _render_formula_visual(raw, target_height=_formula_visual_target_height(self._editor_h))
            visual_text = visual.text if visual is not None and visual.text else raw
            overlays.append(
                _AtomVisualOverlay(
                    start=start,
                    end=end,
                    left=left,
                    right=right,
                    text=visual_text,
                    pixmap=visual.pixmap if visual is not None else None,
                    logical_size=visual.logical_size if visual is not None else None,
                    kind="formula",
                )
            )
        slot_centers = centers if any(center is not None for center in centers) else None
        slot_widths = widths if slot_centers is not None else None
        return overlays, slot_centers, slot_widths

    def _span_by_char_index_for_formula_text(
        self,
        text: str,
    ) -> dict[int, tuple[int, int]] | None:
        chars = self._line_chars()
        old_text = chars_display_text(chars)
        spans = chars_display_spans(chars)
        if text == old_text:
            return {
                char_index: (span.start, span.end)
                for span in spans
                for char_index in span.char_indices
            }
        anchor = self._formula_source_anchor_span
        current = self._formula_source_range
        if anchor is None or current is None:
            return None
        old_start, old_end = anchor
        new_start, new_end = current
        old_start = max(0, min(old_start, len(old_text)))
        old_end = max(old_start, min(old_end, len(old_text)))
        new_start = max(0, min(new_start, len(text)))
        new_end = max(new_start, min(new_end, len(text)))
        if text[:new_start] != old_text[:old_start]:
            return None
        if text[new_end:] != old_text[old_end:]:
            return None
        delta = (new_end - new_start) - (old_end - old_start)
        mapped: dict[int, tuple[int, int]] = {}
        for span in spans:
            if span.end <= old_start:
                next_span = (span.start, span.end)
            elif span.start >= old_end:
                next_span = (span.start + delta, span.end + delta)
            elif span.start == old_start and span.end == old_end:
                next_span = (new_start, new_end)
            else:
                return None
            for char_index in span.char_indices:
                mapped[char_index] = next_span
        return mapped

    def _fallback_formula_visual_data(self, text: str) -> list[_AtomVisualOverlay]:
        ranges: list[tuple[int, int]] = []
        if self._formula_source_range is not None:
            start, end = self._formula_source_range
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))
            if end > start:
                ranges.append((start, end))
        for match in _INLINE_FORMULA_RE.finditer(text or ""):
            span = (match.start(), match.end())
            if span not in ranges:
                ranges.append(span)
        overlays: list[_AtomVisualOverlay] = []
        for start, end in ranges:
            raw = text[start:end]
            if not raw.strip():
                continue
            if (
                self._formula_source_range is not None
                and (start, end) == self._formula_source_range
                and self._formula_source_anchor_rect is not None
            ):
                left, right = self._formula_source_anchor_rect
            else:
                left, right = self._fallback_text_range_rect(text, start, end)
            visual = _render_formula_visual(raw, target_height=_formula_visual_target_height(self._editor_h))
            overlays.append(
                _AtomVisualOverlay(
                    start=start,
                    end=end,
                    left=left,
                    right=right,
                    text=visual.text if visual is not None and visual.text else raw,
                    pixmap=visual.pixmap if visual is not None else None,
                    logical_size=visual.logical_size if visual is not None else None,
                    kind="formula",
                )
            )
        return overlays

    def _formula_atom_rect(
        self,
        atom: ProofAtom,
        text: str,
        start: int,
        end: int,
    ) -> tuple[float, float]:
        scale = float(self._render_scale or 0.0)
        if atom.bbox is not None and scale > 0:
            ox, _oy = self._line_crop_origin
            left = (atom.bbox.x - float(ox)) * scale
            right = (atom.bbox.x2 - float(ox)) * scale
            if right > left + 4:
                return max(0.0, left), max(4.0, right)
        return self._fallback_text_range_rect(text, start, end)

    def _fallback_text_range_rect(self, text: str, start: int, end: int) -> tuple[float, float]:
        font = QFont(self._editor.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        fm = QFontMetrics(font)
        x = max(6.0, TEXT_SLOT_MIN_W / 2.0)
        left = x
        right = x
        for idx, ch in enumerate(text):
            width = _slot_visual_width(ch, fm)
            if idx == start:
                left = x
            x += width
            if idx + 1 == end:
                right = x
                break
        if right <= left:
            right = left + TEXT_SLOT_MIN_W
        return left, right

    def _exit_formula_edit_mode(self) -> bool:
        if self._unit is None or self._unit.kind != ProofUnitKind.FORMULA:
            return False
        if self._editor.has_visual_text_override():
            return False
        self._apply_editor_visual_override()
        self._editor.clearFocus()
        self._refresh_extra_selections()
        self._render_line_image()
        return True

    def _open_formula_source_editor(self, start: int, end: int, global_pos: QPoint) -> None:
        if not self._edit_session.editable:
            return
        text = self._editor.toPlainText()
        start = max(0, min(int(start), len(text)))
        end = max(start, min(int(end), len(text)))
        if end <= start and self._unit is not None and self._unit.kind == ProofUnitKind.FORMULA:
            start, end = 0, len(text)
        if end <= start:
            return
        self._formula_source_range = (start, end)
        self._formula_source_anchor_span = (start, end)
        self._formula_source_anchor_rect = self._formula_source_rect_for_range(text, start, end)
        if self._should_use_formula_source_panel(start, end, text):
            self._open_formula_source_panel(text[start:end])
            return
        self._close_formula_source_panel()
        if self._formula_source_popup is None:
            self._formula_source_popup = _FormulaSourcePopup(self)
            self._formula_source_popup.source_changed.connect(self._apply_formula_source_text)
            self._formula_source_popup.closed.connect(self._on_formula_source_popup_closed)
        self._formula_source_popup.open_for(text[start:end], global_pos, editable=self._edit_session.editable)

    def _formula_source_rect_for_range(self, text: str, start: int, end: int) -> tuple[float, float]:
        for overlay in self._editor.atom_visual_overlays():
            if overlay.kind == "formula" and overlay.start == start and overlay.end == end:
                return overlay.left, overlay.right
        if self._unit is not None and self._unit.atoms:
            span_by_char_index = self._span_by_char_index_for_formula_text(text)
            if span_by_char_index is not None:
                for atom in self._unit.atoms:
                    if atom.kind != ProofAtomKind.FORMULA or not atom.char_indices:
                        continue
                    span = span_by_char_index.get(atom.char_indices[0])
                    if span == (start, end):
                        return self._formula_atom_rect(atom, text, start, end)
        return self._fallback_text_range_rect(text, start, end)

    def _should_use_formula_source_panel(self, start: int, end: int, text: str) -> bool:
        if self._unit is None or self._unit.kind != ProofUnitKind.FORMULA:
            return False
        return start <= 0 and end >= len(text)

    def _open_formula_source_panel(self, source: str) -> None:
        if self._formula_source_popup is not None:
            self._formula_source_popup.hide()
        panel = self._ensure_formula_source_panel()
        panel.open_for(source, editable=self._edit_session.editable)
        self._apply_pair_height()

    def _ensure_formula_source_panel(self) -> _FormulaSourceInlinePanel:
        if self._formula_source_panel is None:
            self._formula_source_panel = _FormulaSourceInlinePanel()
            self._formula_source_panel.source_changed.connect(self._apply_formula_source_text)
            self._formula_source_panel.close_requested.connect(self._close_formula_source_panel)
            layout = self._content.layout()
            if isinstance(layout, QVBoxLayout):
                layout.addWidget(self._formula_source_panel)
        return self._formula_source_panel

    def _close_formula_source_panel(self) -> None:
        panel = self._formula_source_panel
        if panel is None or panel.isHidden():
            return
        panel.flush_source_changed()
        panel.hide()
        self._apply_pair_height()

    def _apply_formula_source_text(self, source: str) -> None:
        if self._formula_source_range is None:
            return
        start, end = self._formula_source_range
        current = self._editor.toPlainText()
        start = max(0, min(start, len(current)))
        end = max(start, min(end, len(current)))
        if current[start:end] == source:
            return
        self._editor.replace_text_range(start, end, source)
        self._formula_source_range = (start, start + len(source))
        self._apply_editor_visual_override()
        self._sync_editor_slot_geometry()
        self._refresh_status()
        self._apply_pair_height()

    def _on_formula_source_popup_closed(self) -> None:
        self._apply_editor_visual_override()
        self._sync_editor_slot_geometry()
        self._refresh_extra_selections()

    @staticmethod
    def _focus_metrics(depth: str) -> tuple[int, int, int, float]:
        if depth == "active":
            return LINE_PAIR_H, IMAGE_ROW_H, TEXT_EDITOR_MAX_H, 1.0
        if depth == "near":
            return NEAR_LINE_PAIR_H, NEAR_IMAGE_ROW_H, NEAR_TEXT_EDITOR_H, 0.42
        return FAR_LINE_PAIR_H, FAR_IMAGE_ROW_H, FAR_TEXT_EDITOR_H, 0.28

    def set_focus_depth(self, depth: str) -> None:
        """Set visual focus depth without changing proofreading state.

        active: 当前校对行，完整高度和不透明；
        near: 上下相邻上下文，略缩小、略浅；
        far: 远离当前行的上下文，压缩并浅化。
        """
        if depth not in {"active", "near", "far"}:
            depth = "far"
        pair_h, image_h, editor_h, opacity = self._focus_metrics(depth)
        if (
            self._focus_depth == depth
            and self._pair_h == pair_h
            and self._image_row_h == image_h
            and self._editor_h == editor_h
        ):
            return
        self._focus_depth = depth
        self._pair_h = pair_h
        self._image_row_h = image_h
        self._editor_h = editor_h
        self._apply_pair_height()
        self._img_lbl.setFixedHeight(image_h)
        self._editor.setFixedHeight(editor_h)
        self._editor.setVisible(depth == "active")
        if self._formula_source_panel is not None:
            self._formula_source_panel.setVisible(
                (not self._formula_source_panel.isHidden()) and depth == "active"
            )
            self._apply_pair_height()
        if depth == "active" and self._active:
            self._editor.setFocus()
        self._opacity_effect.setOpacity(opacity)
        self._refresh_status()
        if self._line_crop is not None:
            self._render_line_image()
        else:
            self._sync_editor_slot_geometry()

    def _canonical_model_display_text(self) -> str:
        text = _displayed_text(self._line, self._page, self._block)
        text, _ = _canonicalize_text_to_slots(text, self._line_chars())
        return text

    def _set_editor_text(self, text: str) -> None:
        if self._editor.toPlainText() == text:
            return
        self._formula_source_range = None
        self._formula_source_anchor_span = None
        self._formula_source_anchor_rect = None
        self._editor.blockSignals(True)
        self._editor.setPlainText(text)
        self._editor.apply_inline_y_axis_metrics()
        self._editor.blockSignals(False)

    def is_editor_dirty(self) -> bool:
        return self._edit_session.is_dirty(self._editor.toPlainText())

    def has_external_conflict(self) -> bool:
        return self._edit_session.external_conflict

    @property
    def is_editable(self) -> bool:
        return self._edit_session.editable

    def loaded_line_signature(self) -> str:
        return self._edit_session.loaded_line_signature

    def mark_external_conflict(self) -> None:
        self._edit_session.mark_external_conflict()
        self._refresh_status()
        self._refresh_extra_selections()
        self._sync_editor_slot_geometry()

    def mark_editor_saved(self) -> None:
        self._edit_session.mark_saved(
            self._canonical_model_display_text(),
            line_signature(self._line),
        )
        self._refresh_status()
        self._refresh_extra_selections()
        self._sync_editor_slot_geometry()

    def dirty_editor_snapshot(self) -> tuple[str, str]:
        return self._edit_session.dirty_snapshot(self._editor.toPlainText())

    def _clear_external_conflict_if_resolved(self) -> None:
        current_model_text = self._canonical_model_display_text()
        if self._edit_session.clear_conflict_if_editor_matches_model(
            editor_text=self._editor.toPlainText(),
            model_display_text=current_model_text,
            model_line_signature=line_signature(self._line),
        ):
            self._refresh_status()
            self._refresh_extra_selections()
            self._sync_editor_slot_geometry()

    def restore_dirty_editor_text(self, text: str, previous_loaded_text: str | None = None) -> None:
        self._set_editor_text(text)
        self._edit_session.restore_dirty_editor_text(
            editor_text=text,
            previous_loaded_text=previous_loaded_text,
        )
        self._apply_editor_visual_override()
        self._apply_fixed_length_to_editor()
        self._refresh_status()
        self._refresh_extra_selections()
        self._sync_editor_slot_geometry()

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        active_color = FOCUS_BORDER_COLOR
        accent_color = self._status_accent_color()
        if active:
            bar_style = f"background:{accent_color}; border-radius:3px;"
            frame_style = (
                f"QFrame#linePair {{ background:#FFFDF8; border:1px solid {active_color}; "
                "border-radius:6px; }"
            )
            content_bg = "#FFFDF8"
            image_style = f"background:#FFFFFF; border-bottom:1px solid {IMAGE_DIVIDER_COLOR};"
            self._editor.setFocus()
        else:
            # 切走前先把公式源码栏刷回 editor，再保存 in-flight 文本。
            self._close_formula_source_panel()
            self._flush_editor_if_dirty()
            # 切行时清掉本行 editor 的选中状态和高亮，避免非焦点行仍有选中感。
            cur = self._editor.textCursor()
            if cur.hasSelection():
                cur.clearSelection()
                self._editor.setTextCursor(cur)
            self._editor.setExtraSelections([])
            self._hover_char_idx = -1
            bar_style = "background:transparent;"
            frame_style = (
                "QFrame#linePair { background:#FFFFFF; border:1px solid transparent; "
                f"border-bottom:1px solid {ROW_DIVIDER_COLOR}; border-radius:0; }}"
            )
            content_bg = "#FFFFFF"
            image_style = f"background:#FFFFFF; border-bottom:1px solid {IMAGE_DIVIDER_COLOR};"

        self._active_bar.setStyleSheet(bar_style)
        self._active_bar2.setStyleSheet(bar_style)
        self._content.setStyleSheet(f"background:{content_bg};")
        self._img_lbl.setStyleSheet(image_style)
        self._editor.setStyleSheet(self._editor_style(active=active))
        if hasattr(self._editor, "set_active_visual"):
            self._editor.set_active_visual(active)
        self.setProperty("active", active)
        self.setStyleSheet(frame_style)
        self._apply_pair_height()
        self._refresh_status()
        self._refresh_extra_selections()
        self._render_line_image()

    def _flush_editor_if_dirty(self) -> None:
        """若 editor 当前文本与显示空间文本不一致，发 text_saved 让面板落盘。
        editor 始终可见，不再判 isHidden。"""
        if self._edit_session.external_conflict or not self._edit_session.editable:
            return
        new_text = self._editor.toPlainText()
        if self._edit_session.is_dirty(new_text):
            self.text_saved.emit(self._idx, new_text)

    def _apply_fixed_length_to_editor(self) -> None:
        """按当前 OCR 字符观测状态启用/关闭固定长度覆写模式。

        不给 editor 设置 tooltip，避免空白 hover 框残影；固定模式本身仍启用，
        Backspace/Delete 走“填空字”路径，而不是拒绝输入。
        """
        chars = self._line_chars()
        text_len = len(self._editor.toPlainText())
        fixed = len(chars) if chars and text_len == len(chars) else None
        self._editor.set_fixed_length(fixed)

    def _on_editor_hover_char(self, idx: int) -> None:
        """editor 鼠标悬停字符 idx 变化 → 在行图上画 hover 框（图字对应升级）。

        只在 aligned 时生效；否则保留行级显示，不假装能定位到某字。
        """
        if not self._chars_aligned():
            new_idx = -1
        else:
            new_idx = idx if 0 <= idx < len(self._line_chars()) else -1
        if new_idx == self._hover_char_idx:
            return
        self._hover_char_idx = new_idx
        self._render_line_image()

    # ── 弱光标 + 逐字高亮 ───────────────

    def _chars_aligned(self) -> bool:
        """Editor 文本是否与 OCR 字符观测严格一一对应。

        图像 char.bbox 与文本下标的映射只有在 ``len(text) == len(chars)`` 且
        每个 ``char.char`` 恰好一个字符时才可靠。word/token granularity 或
        结构性编辑会让第 i 个文本字与第 i 个 bbox 错位，此时必须降级，不画
        逐字高亮。
        """
        chars = self._line_chars()
        if not chars:
            return False
        if len(self._editor.toPlainText()) != len(chars):
            return False
        for c in chars:
            ch = c.char or ""
            if len(ch) != 1:
                return False
        return True

    # ── 文本颜色规则 ─────────
    # 颜色 + 证据链由 :mod:`app.ui.proof.char_verdict` 集中负责，本文件只做调用。
    #
    # 颜色只读 OCR confidence；“用户修改过”不能自动洗白。

    def _classify_char_verdict(self, i: int) -> Optional[_cv.CharVerdict]:
        """返回第 i 个字的 verdict；下标越界 / 未对齐 → None。"""
        chars = self._line_chars()
        if not (0 <= i < len(chars)):
            return None
        text = self._editor.toPlainText()
        if i >= len(text):
            return None
        conf = char_confidence(self._line, i)
        ocr = proof_ocr_text(self._line) or ""
        # 只在等长时取同下标字符；长度不一致时退回 None，避免错位比对
        ocr_ch = ocr[i] if len(ocr) == len(text) and i < len(ocr) else None
        return _cv.classify_char(
            confidence=conf,
            text_char=text[i],
            ocr_char=ocr_ch,
        )

    def _refresh_extra_selections(self) -> None:
        """生成 editor 的 extraSelections：
        - 按 verdict 给每个字上前景色（绿/橙/红/灰）。
        - 当前光标所在/选中字：蓝色淡背景（"当前字"指示，配合弱光标）。

        caret 宽度为 0，弱光标完全靠这套 extraSelections 表达。
        """
        try:
            from PySide6.QtWidgets import QTextEdit
        except Exception:
            return
        editor = self._editor
        doc_text = editor.toPlainText()
        sels: list = []

        aligned = self._chars_aligned()
        # 1) 前景色：仅在图字严格一一对应时绘制
        if aligned:
            chars = self._line_chars()
            n = min(len(chars), len(doc_text))
            for i in range(n):
                verdict = self._classify_char_verdict(i)
                if verdict is None or verdict.color == _cv.COLOR_UNVERIFIED:
                    continue
                sel = QTextEdit.ExtraSelection()
                cur = QTextCursor(editor.document())
                cur.setPosition(i)
                cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(verdict.color))
                # 错字额外给一个非常淡的红底，"必须修"更醒目；其他颜色不加底色
                # 避免与"当前字"的蓝底冲突。
                if verdict.color == _cv.COLOR_ERROR:
                    fmt.setBackground(QColor("#fdecec"))
                sel.format = fmt
                sel.cursor = cur
                sels.append(sel)

        # 2) 当前字（光标所在位置或 selection 范围）边框/底色
        cursor = editor.textCursor()
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
        else:
            pos = cursor.position()
            start, end = pos, min(pos + 1, len(doc_text))
        if end > start and self._active:
            sel = QTextEdit.ExtraSelection()
            cur = QTextCursor(editor.document())
            cur.setPosition(start)
            cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#cfe2ff"))   # 蓝色高亮 = 当前字
            sel.format = fmt
            sel.cursor = cur
            sels.append(sel)

        editor.setExtraSelections(sels)

    def _img_clicked_lookup(self, event) -> None:
        """点击行图区域：先按 char.bbox 反查最近的字 → 把 editor 光标定到该字。
        若 chars 缺失 / 图字未严格一一对应，退回行级激活（不强行定位以免错位）。
        """
        if self._exit_formula_edit_mode():
            try:
                event.accept()
            except Exception:
                pass
            return
        if not self._chars_aligned():
            # 降级：不假装能定到某字，仅激活本行 + 提示原因
            self._on_click(event)
            return
        # 反查：把 _img_lbl 内 px 坐标 → 原图坐标 → 找最近 char
        try:
            pos = event.position()
            click_x = pos.x()
        except AttributeError:
            click_x = float(event.x())
        ox, _oy = self._line_crop_origin
        scale = self._render_scale or 1.0
        orig_x = ox + click_x / scale
        target_idx = None
        for i, ch in enumerate(self._line_chars()):
            if ch.bbox is None:
                continue
            if ch.bbox.x <= orig_x <= ch.bbox.x2:
                target_idx = i
                break
        if target_idx is None:
            best_dist = float("inf")
            for i, ch in enumerate(self._line_chars()):
                if ch.bbox is None:
                    continue
                cx = (ch.bbox.x + ch.bbox.x2) / 2.0
                d = abs(cx - orig_x)
                if d < best_dist:
                    best_dist = d
                    target_idx = i
        if target_idx is None:
            self._on_click(event)
            return
        # 先激活本行
        if not self._active:
            # 复用 _on_click 的 emit 路径
            self._on_click(event)
        # 把 editor 光标定到该字
        from PySide6.QtGui import QTextCursor
        cur = self._editor.textCursor()
        cur.setPosition(target_idx)
        cur.setPosition(target_idx + 1, QTextCursor.MoveMode.KeepAnchor)
        self._editor.setTextCursor(cur)
        self._editor.setFocus()
        self._refresh_extra_selections()

    def load_image(self) -> None:
        """懒加载行图像。"""
        if self._image_loaded:
            return
        self._image_loaded = True
        bb = self._line.bbox
        if bb.w <= 0 or bb.h <= 0:
            self._img_lbl.setText("—")
            return
        image = self._cache.get_page_image(self._page.display_image_path)
        if image is None:
            self._img_lbl.setText("（无图像）")
            return
        H, W = image.shape[:2]
        clamped = clamp_line_box_pixels(bb, W, H, pad_y=ROW_PAD_Y)
        if clamped is None:
            self._img_lbl.setText("（行框异常）")
            return
        x1, y1, x2, y2 = clamped
        crop = image[y1:y2, x1:x2].copy()
        self._line_crop = crop
        self._line_crop_origin = (x1, y1)
        h, w = crop.shape[:2]
        # 极小 bbox（OCR 出错时高/宽 < 5px）放大后会产生伪影/碎裂，
        # 而非真实行图。直接显示占位避免误导用户。
        if h < 5 or w < 5:
            self._img_lbl.setText("（行框异常）")
            return
        self._render_line_image()

    def _render_line_image(self) -> None:
        if self._line_crop is None:
            return
        crop = self._line_crop.copy()
        highlight_range: tuple[int, int] | None = None
        # editor 始终可见，按光标/选区在行图上高亮对应 char.bbox。
        # 只有文本与 chars 严格一一对应时才画框，避免把"第 N 字"高亮到错误 bbox。
        if self._active and self._chars_aligned():
            cursor = self._editor.textCursor()
            start = min(cursor.selectionStart(), cursor.selectionEnd())
            end = max(cursor.selectionStart(), cursor.selectionEnd())
            if end > start:
                highlight_range = (start, end)
            else:
                pos = cursor.position()
                if 0 <= pos < len(self._line_chars()):
                    highlight_range = (pos, pos + 1)
        if highlight_range is not None:
            start, end = highlight_range
            ox, oy = self._line_crop_origin
            chars = self._line_chars()
            for idx in range(start, min(end, len(chars))):
                char = chars[idx]
                if char.bbox is None:
                    continue
                x1 = max(0, char.bbox.x - ox)
                y1 = max(0, char.bbox.y - oy)
                x2 = min(crop.shape[1] - 1, char.bbox.x2 - ox)
                y2 = min(crop.shape[0] - 1, char.bbox.y2 - oy)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 128, 255), 2)
                    overlay = crop.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), -1)
                    crop = cv2.addWeighted(overlay, 0.18, crop, 0.82, 0)
        # editor 鼠标悬停 → 在行图上画绿色细框
        # （与"当前字"蓝框区分；当 hover idx 与当前字重叠时，蓝框已经画过，
        # 此处的绿框会叠在外侧 1px，仍能看出"鼠标正在指这个字"）。
        if (
            self._chars_aligned()
            and 0 <= self._hover_char_idx < len(self._line_chars())
        ):
            ox, oy = self._line_crop_origin
            char = self._line_chars()[self._hover_char_idx]
            if char.bbox is not None:
                x1 = max(0, char.bbox.x - ox)
                y1 = max(0, char.bbox.y - oy)
                x2 = min(crop.shape[1] - 1, char.bbox.x2 - ox)
                y2 = min(crop.shape[0] - 1, char.bbox.y2 - oy)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (40, 167, 69), 1)
        h, w = crop.shape[:2]
        # 缩放到当前视觉层级的行图高度，同时限制最大宽度（避免超宽行撑开布局）。
        # 严格保持宽高比：先按高度缩放；若超宽再按宽度缩放重算高度。
        target_h = max(1, int(self._image_row_h or IMAGE_ROW_H))
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        MAX_LINE_W = 1200
        if new_w > MAX_LINE_W:
            scale = MAX_LINE_W / w
            new_h = max(1, int(round(h * scale)))
            crop = cv2.resize(crop, (MAX_LINE_W, new_h), interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
        self._render_scale = scale
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), rw, rh, rw * 3, QImage.Format.Format_RGB888)
        self._img_lbl.setPixmap(QPixmap.fromImage(qimg))
        # 把图像里每个字的 x 中心推给 editor。editor.paintEvent 用这套坐标
        # 自绘每字，让图像字和文本字在同一水平位置对齐。
        # editor 与 _img_lbl 同为 content_v 的 full-width 子控件，且 _img_lbl
        # 内 pixmap 左对齐 → editor x=0 == _img_lbl x=0 == 行图左边缘。
        self._sync_editor_slot_geometry()

    def _sync_editor_slot_geometry(self) -> None:
        """按 OCR 字符观测 bbox + _render_scale + _line_crop_origin 推算
        每字在 editor 视口内的 x 中心 & 宽度；交给 editor 自绘文本。

        chars 未对齐 / 缺 bbox → 清空 → editor 走原生渲染（降级）。
        """
        editor = getattr(self, "_editor", None)
        if editor is None:
            return
        if hasattr(editor, "has_visual_text_override") and editor.has_visual_text_override():
            editor.set_slot_geometry(None, None)
            editor.set_atom_visual_overlays(None)
            return
        text = editor.toPlainText()
        atom_overlays, atom_centers, atom_widths = self._formula_atom_visual_data(text)
        editor.set_atom_visual_overlays(atom_overlays)
        if atom_centers is not None:
            editor.set_slot_geometry(atom_centers, _clip_slot_widths_to_centers(atom_centers, atom_widths or []))
            return
        if not self._chars_aligned():
            editor.set_slot_geometry(None, None)
            return
        ox, _oy = self._line_crop_origin
        scale = float(self._render_scale or 0.0)
        if scale <= 0:
            editor.set_slot_geometry(None, None)
            return
        x_centers: list = []
        widths: list = []
        fm = QFontMetrics(editor.font())
        text = editor.toPlainText()
        for idx, ch in enumerate(self._line_chars()):
            if ch.bbox is None:
                x_centers.append(None)
                widths.append(0.0)
                continue
            cx_src = (ch.bbox.x + ch.bbox.x2) / 2.0 - float(ox)
            x_centers.append(cx_src * scale)
            text_char = text[idx] if idx < len(text) else ch.char
            widths.append(_slot_visual_width(text_char, fm))
        editor.set_slot_geometry(x_centers, _clip_slot_widths_to_centers(x_centers, widths))

    def refresh_text(self, *, force: bool = False) -> str:
        """外部（VProof / probe 切换）更新 final_text 后同步 editor 文本。

        返回值：
        - "updated"/"merged": 已同步到新模型文本；
        - "local_dirty": 本地有未保存改动，模型未变化，不覆盖；
        - "conflict": 本地和外部同时改了同一行，不覆盖本地 editor。
        """
        new_disp = self._canonical_model_display_text()
        editor_text = self._editor.toPlainText()
        dirty = self._edit_session.is_dirty(editor_text)
        external_changed = new_disp != self._edit_session.loaded_display_text
        if not force and dirty and external_changed and new_disp != editor_text:
            self.mark_external_conflict()
            return "conflict"
        if not force and dirty and not external_changed:
            self._refresh_status()
            self._refresh_extra_selections()
            self._sync_editor_slot_geometry()
            return "local_dirty"
        if dirty and external_changed and new_disp == editor_text:
            self._edit_session.mark_saved(new_disp, line_signature(self._line))
            self._refresh_status()
            self._refresh_extra_selections()
            self._sync_editor_slot_geometry()
            return "merged"
        self._set_editor_text(new_disp)
        self._edit_session.mark_saved(new_disp, line_signature(self._line))
        self._apply_editor_visual_override()
        # 显示文本变了 → 字数可能变 → 重新评估 fixed_length
        self._apply_fixed_length_to_editor()
        self._refresh_status()
        self._refresh_extra_selections()
        # 文本/对齐状态变化后重算图字 x 映射；_render_scale 未就绪时会自动降级。
        self._sync_editor_slot_geometry()
        return "updated"

    def rebind(
        self,
        block: Block,
        line: Line,
        page: Page,
        line_in_page: int,
        debug_badge: str = "",
        unit: ProofUnit | None = None,
    ) -> None:
        """Point this UI row at the current project Line without rebuilding it."""
        editor_text = self._editor.toPlainText()
        self._block = block
        self._line = line
        self._page = page
        self._line_in_page = line_in_page
        self._debug_badge = debug_badge
        self._unit = unit
        self._image_loaded = False
        self._line_crop = None
        self._apply_editor_visual_override()
        new_loaded_text = self._canonical_model_display_text()
        was_dirty = self._edit_session.rebind_to_model(
            editor_text=editor_text,
            model_display_text=new_loaded_text,
            model_line_signature=line_signature(self._line),
        )
        if not was_dirty:
            self._set_editor_text(self._edit_session.loaded_display_text)
        # rebind 只在旧 editor clean 时同步新文本；dirty 文本保留。若后台同 key
        # 新 Line 也变了，则标冲突，禁止静默覆盖。新行可能 chars 数不同，需要
        # 重新评估 fixed_length。
        self._apply_fixed_length_to_editor()
        self._refresh_status()
        self._refresh_extra_selections()
        # 切到新行 → 旧的 slot geometry 立即失效；先清空避免短暂错位
        self._editor.set_slot_geometry(None, None)

    @property
    def line(self) -> Line:
        return self._line

    @property
    def page(self) -> Page:
        return self._page

    @property
    def block(self) -> Block:
        return self._block

    # ── 私有 ──────────────────────────────────────────────────

    def _line_chars(self):
        return line_ocr_chars(self._line)

    def _on_click(self, event) -> None:
        if hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton:
            if self._exit_formula_edit_mode():
                try:
                    event.accept()
                except Exception:
                    pass
                return
        self.clicked.emit(self._idx)

    def _on_editor_focus_in(self) -> None:
        """editor 内点击/取得焦点即激活本行。"""
        if self._active:
            return
        self.clicked.emit(self._idx)

    def _revert(self) -> None:
        # 还原到 OCR 原始文本。
        original_true = proof_ocr_text(self._line) or proof_display_text(self._line) or ""
        self._editor.blockSignals(True)
        # 还原后也按槽位规则补空。
        _rev_canon, _ = _canonicalize_text_to_slots(
            original_true, self._line_chars()
        )
        self._editor.setPlainText(_rev_canon)
        self._editor.apply_inline_y_axis_metrics()
        self._editor.blockSignals(False)
        self._apply_editor_visual_override()

    def _refresh_status(self) -> None:
        status = proof_status(self._line)
        color = self._status_accent_color()
        if self._edit_session.external_conflict:
            color = "#c62828"
            self._status_lbl.setText(
                f"<span style='color:{color};font-size:11px;font-weight:700;'>冲突</span>"
            )
            self._status_lbl.setStyleSheet("background:transparent;")
            if hasattr(self, "_active_bar"):
                if self._active:
                    self._active_bar.setStyleSheet(f"background:{color}; border-radius:3px;")
                else:
                    self._active_bar.setStyleSheet("background:transparent;")
            return
        # 这里只在真正异常时给 ⚠ 提示；正常情况下不显示额外 hover 文案。
        text_n = len(self._editor.toPlainText()) if hasattr(self, "_editor") else 0
        chars = self._line_chars()
        char_n = len(chars) if chars else 0
        unaligned = bool(chars) and text_n != char_n
        glyph = _STATUS_GLYPH.get(status, "○")
        if unaligned:
            color = "#c62828"
            glyph = "!"
            tip = (
                f"图字未对齐：文本 {text_n} 字 ≠ 图像字符 {char_n} 字\n"
                f"已停用逐字高亮；请把文本改回 {char_n} 字以恢复图字一一对应"
            )
        else:
            tip = ""
        if hasattr(self, "_active_bar"):
            if self._active:
                self._active_bar.setStyleSheet(f"background:{color}; border-radius:3px;")
            else:
                self._active_bar.setStyleSheet("background:transparent;")
        size = 15 if self._active or unaligned else 12
        status_text = f"<span style='color:{color};font-size:{size}px;font-weight:700;'>{glyph}</span>"
        self._status_lbl.setText(status_text)
        self._status_lbl.setStyleSheet("background:transparent;")
        # 不给 status_lbl 设 tooltip，避免 Qt 某些环境下出现空白 hover 框。

    def _status_accent_color(self) -> str:
        return _STATUS_COLOR.get(proof_status(self._line), TEXT_GUIDE_LINE_COLOR)


class _ProofUnitRenderer:
    """横校渲染器基类：把 ProofUnit 转成可插入列表的 QWidget。"""

    def create_pair(
        self,
        idx: int,
        unit: ProofUnit,
        cache: PageImageCache,
    ) -> _LinePair:
        raise NotImplementedError


class _TextProofUnitRenderer(_ProofUnitRenderer):
    """正文行渲染器。当前复用既有 _LinePair，保证行为不变。"""

    def create_pair(
        self,
        idx: int,
        unit: ProofUnit,
        cache: PageImageCache,
    ) -> _LinePair:
        return _LinePair(
            idx,
            unit.block,
            unit.line,
            unit.page,
            unit.page_line_number,
            cache,
            debug_badge=unit.debug_badge,
            unit=unit,
        )


# ─────────────────────────────────────────────────────────────
# 横校面板主体
# ─────────────────────────────────────────────────────────────

class HProofPanel(QWidget):
    """横校面板：滚动列表 + 工具栏，对照 ui-2.jpg 设计。"""

    proof_changed = Signal(object)
    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session = HProofRuntimeSession()
        self._pairs: List[_LinePair] = []
        self._cache = PageImageCache.instance()
        self._external_refresh_timer = QTimer(self)
        self._external_refresh_timer.setSingleShot(True)
        self._external_refresh_timer.setInterval(80)
        self._external_refresh_timer.timeout.connect(self._do_external_refresh)
        text_renderer = _TextProofUnitRenderer()
        self._renderers: dict[ProofUnitKind, _ProofUnitRenderer] = {
            ProofUnitKind.TEXT: text_renderer,
            ProofUnitKind.FORMULA: text_renderer,
            ProofUnitKind.TABLE: text_renderer,
            ProofUnitKind.IMAGE: text_renderer,
            ProofUnitKind.CAPTION: text_renderer,
        }
        self._bus = ProofStateBus.instance()
        # H/V 校对联动：订阅其他 panel 编辑事件；origin == id(self) 的事件忽略。
        # 保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            TOPIC_LINE_PROOF_CHANGED, self._on_external_line_changed,
        )
        self.destroyed.connect(lambda *_: self._teardown_bus())
        self.destroyed.connect(lambda *_: self._teardown_ui_filters())
        self._app_tooltip_filter_installed = False
        self._build_ui()
        self._suppress_hover_tooltips(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_tooltip_filter_installed = True

    def _teardown_bus(self) -> None:
        """释放 ProofStateBus 订阅。

        QObject.destroyed 信号在 Python 端仍可调用 unsubscribe；幂等，多次安全。"""
        unsub = getattr(self, "_bus_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._bus_unsub = None

    def _teardown_ui_filters(self) -> None:
        if not getattr(self, "_app_tooltip_filter_installed", False):
            return
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        self._app_tooltip_filter_installed = False

    def _suppress_hover_tooltips(self, widget: QWidget) -> None:
        """Disable hover tooltip popups inside HProof.

        Page switching rebuilds row widgets. Any tooltip attached to row/image/
        editor children becomes a rapid stream of small hover popups, so HProof
        treats hover tips as non-essential UI noise.
        """
        try:
            widget.setToolTip("")
            widget.installEventFilter(self)
        except Exception:
            pass
        for child in widget.findChildren(QWidget):
            try:
                child.setToolTip("")
                child.installEventFilter(self)
            except Exception:
                pass

    def _is_own_tooltip_target(self, obj) -> bool:
        if isinstance(obj, QWidget):
            return obj is self or self.isAncestorOf(obj)
        return self.isVisible()

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event is not None and event.type() == QEvent.Type.ToolTip:
            return self._is_own_tooltip_target(obj)
        return super().eventFilter(obj, event)

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 横校主视图仍由 _LinePair 承载；这里只调整 shell：
        # 左侧导航 + 中央校对列表 + 底部状态/操作区。
        self.setObjectName("proofRoot")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("hproofSplitter")

        # ── 左：侧边栏轨道 + 页面目录 ───────────────────────────
        left = QFrame()
        left.setObjectName("proofLeftPane")
        left.setMinimumWidth(250)
        left.setMaximumWidth(322)
        self._hproof_left_pane = left
        self._hproof_sidebar_collapsed = False
        left_lay = QHBoxLayout(left)
        left_lay.setContentsMargins(10, 10, 10, 10)
        left_lay.setSpacing(8)

        rail = QFrame()
        rail.setObjectName("proofNavRail")
        rail_lay = QVBoxLayout(rail)
        rail_lay.setContentsMargins(0, 0, 0, 0)
        rail_lay.setSpacing(6)

        self._btn_nav_toggle = QPushButton()
        self._btn_nav_toggle.setObjectName("proofRailBtn")
        self._btn_nav_toggle.setCheckable(True)
        self._btn_nav_toggle.setFixedSize(34, 32)
        self._btn_nav_toggle.setIcon(get_icon("sidebar", color="#6B6B6B"))
        self._btn_nav_toggle.setIconSize(QSize(17, 17))
        self._btn_nav_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        rail_lay.addWidget(self._btn_nav_toggle)

        self._btn_nav_pages = QPushButton()
        self._btn_nav_pages.setObjectName("proofRailBtn")
        self._btn_nav_pages.setCheckable(True)
        self._btn_nav_pages.setChecked(True)
        self._btn_nav_pages.setFixedSize(34, 32)
        self._btn_nav_pages.setIcon(get_icon("directory", color="#6B6B6B"))
        self._btn_nav_pages.setIconSize(QSize(17, 17))
        self._btn_nav_pages.setCursor(Qt.CursorShape.PointingHandCursor)
        rail_lay.addWidget(self._btn_nav_pages)
        rail_lay.addStretch(1)
        left_lay.addWidget(rail)

        page_pane = QWidget()
        page_pane.setObjectName("proofLeftStackPage")
        self._hproof_page_pane = page_pane
        page_lay = QVBoxLayout(page_pane)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(8)
        self._page_dir = PageDirectoryList()
        self._page_dir.page_selected.connect(self._on_page_dir_selected)
        page_lay.addWidget(self._page_dir, 1)
        left_lay.addWidget(page_pane, 1)
        splitter.addWidget(left)
        self._hproof_splitter = splitter

        # ── 中：滚动列表 ────────────────────────────────────────
        center = QWidget()
        center.setObjectName("proofCenterPane")
        center_v = QVBoxLayout(center)
        center_v.setContentsMargins(10, 10, 10, 10)
        center_v.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("proofScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_widget = QWidget()
        self._list_widget.setObjectName("proofLineList")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(6, 4, 6, 4)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()

        # 空状态提示（无数据时显示）
        self._empty_lbl = QLabel("完成 OCR 识别后，横校数据将在此展示")
        self._empty_lbl.setObjectName("proofEmpty")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setMinimumHeight(120)
        self._list_layout.insertWidget(0, self._empty_lbl)

        self._mode_banner = QLabel("")
        self._mode_banner.setObjectName("hproofModeBanner")
        self._mode_banner.setVisible(False)
        center_v.addWidget(self._mode_banner)
        self._scroll.setWidget(self._list_widget)
        center_v.addWidget(self._scroll, 1)
        splitter.addWidget(center)

        # 操作按钮保留原 slot，承载位置改到底部状态栏。
        self._btn_save = QPushButton("保存")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.setToolTip("保存所有修改  (Ctrl+S)")
        self._btn_flag = QPushButton("标记")
        self._btn_flag.setObjectName("secondaryBtn")
        self._btn_flag.setToolTip("标记当前行  (F5)")
        self._btn_skip = QPushButton("跳过")
        self._btn_skip.setObjectName("ghostBtn")
        self._btn_skip.setToolTip("跳过当前行  (F6)")
        for btn in (self._btn_save, self._btn_flag, self._btn_skip):
            btn.setFixedHeight(28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._current_scope_lbl = QLabel("全部页面")
        self._current_scope_lbl.setObjectName("muted")

        self._current_line_lbl = QLabel("当前行 0 / 0")
        self._current_line_lbl.setObjectName("proofStatusStrong")

        self._handled_lbl = QLabel("已处理 0 / 0")
        self._handled_lbl.setObjectName("muted")

        # Backing widgets kept for tests and internal state reads; they are no
        # longer part of the visible right sidebar.
        self._proof_progress_bar = QProgressBar()
        self._proof_progress_bar.setRange(0, 1)
        self._proof_progress_bar.setValue(0)
        self._proof_progress_bar.setVisible(False)
        self._pending_lbl = QLabel("待确认 0")
        self._confirmed_lbl = QLabel("已确认 0")
        self._modified_lbl = QLabel("已修改 0")
        self._flagged_lbl = QLabel("疑点 0")

        self._btn_debug_formula = QPushButton("公式")
        self._btn_debug_formula.setObjectName("ghostBtn")
        self._btn_debug_formula.setCheckable(True)
        self._btn_debug_formula.setToolTip("只显示被识别为公式或内联公式路由的行")
        self._btn_debug_table = QPushButton("表格")
        self._btn_debug_table.setObjectName("ghostBtn")
        self._btn_debug_table.setCheckable(True)
        self._btn_debug_table.setToolTip("只显示被识别为表格或表格路由的行")
        for btn in (self._btn_debug_formula, self._btn_debug_table):
            btn.setFixedHeight(28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._stat_lbl = QLabel("")
        self._stat_lbl.setObjectName("muted")

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([282, 9999])
        root.addWidget(splitter, 1)

        # ── 底部状态栏 ─────────────────────────────────────────
        statusbar = QWidget()
        statusbar.setObjectName("proofStatusBar")
        statusbar.setFixedHeight(44)
        sl = QHBoxLayout(statusbar)
        sl.setContentsMargins(12, 3, 12, 3)
        sl.setSpacing(10)

        self._progress_ring = _ProofProgressRing()
        sl.addWidget(self._progress_ring)
        sl.addWidget(self._current_scope_lbl)
        sl.addWidget(self._current_line_lbl)
        sl.addWidget(self._handled_lbl)

        sl.addSpacing(6)
        sl.addWidget(self._btn_save)
        sl.addWidget(self._btn_flag)
        sl.addWidget(self._btn_skip)

        sl.addSpacing(8)
        sl.addWidget(self._btn_debug_formula)
        sl.addWidget(self._btn_debug_table)

        self._total_lbl = QLabel("总字数 0")
        self._diff_lbl  = QLabel("差异 0 (0%)")
        for lbl in (self._total_lbl, self._diff_lbl):
            lbl.setStyleSheet("font-size:11px; color:#666;")

        sl.addStretch()
        sl.addWidget(self._total_lbl)
        sl.addWidget(self._diff_lbl)
        sl.addWidget(self._stat_lbl)
        root.addWidget(statusbar)

        # ── 信号 ───────────────────────────────────────────────
        self._btn_nav_toggle.clicked.connect(
            lambda checked=False: self._set_left_sidebar_collapsed(
                not self._hproof_sidebar_collapsed
            )
        )
        self._btn_nav_pages.clicked.connect(self._show_directory_sidebar)
        self._progress_ring.clicked.connect(self._show_progress_popup)
        self._btn_save.clicked.connect(self._save_all)
        self._btn_flag.clicked.connect(self._toggle_flag)
        self._btn_skip.clicked.connect(self._next)
        self._btn_debug_formula.toggled.connect(self._on_debug_filter_changed)
        self._btn_debug_table.toggled.connect(self._on_debug_filter_changed)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_all)
        # 滚动时触发懒加载
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda _: self._load_visible_images()
        )

    def _show_directory_sidebar(self) -> None:
        self._set_left_sidebar_collapsed(False)
        self._btn_nav_pages.setChecked(True)

    def _set_left_sidebar_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._hproof_sidebar_collapsed:
            return
        self._hproof_sidebar_collapsed = collapsed
        self._btn_nav_toggle.setChecked(collapsed)
        self._btn_nav_pages.setChecked(True)
        self._hproof_page_pane.setVisible(not collapsed)

        if collapsed:
            self._hproof_left_pane.setMinimumWidth(56)
            self._hproof_left_pane.setMaximumWidth(56)
            self._hproof_splitter.setSizes([56, max(800, self._hproof_splitter.width() - 56)])
        else:
            self._hproof_left_pane.setMinimumWidth(250)
            self._hproof_left_pane.setMaximumWidth(322)
            self._hproof_splitter.setSizes([282, max(800, self._hproof_splitter.width() - 282)])
        self._hproof_left_pane.updateGeometry()

    # ── 公共 API ───────────────────────────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._session.set_pages(pages, page_has_lines=self._page_has_lines)
        self._refresh_page_filter()
        self._render_pages(self._filtered_pages())

    def merge_pages(self, pages: List[Page]) -> None:
        """Merge OCR background updates without rebuilding active editors."""
        if not self._pairs:
            self.load_pages(pages)
            return
        self._session.set_pages(pages, page_has_lines=self._page_has_lines)
        self._refresh_page_filter()
        loaded_keys = {
            projection.identity_key: index
            for index, projection in enumerate(self._session.projections)
        }
        old_current_key = None
        current_index = self._session.current_projection_index
        if 0 <= current_index < len(self._session.projections):
            old_current_key = self._session.projections[current_index].identity_key
        preserved_dirty_texts = {
            projection.identity_key: self._pairs[index].dirty_editor_snapshot()
            for index, projection in enumerate(self._session.projections)
            if index < len(self._pairs) and self._pairs[index].is_editor_dirty()
        }
        filtered_pages = self._filtered_pages()
        new_entries: list[tuple[ProofLineProjection, int]] = []
        for page in filtered_pages:
            page_line_num = 1
            for block, line, li in self._iter_page_lines(page):
                projection = self._build_projection(page, block, line, li)
                new_entries.append((projection, page_line_num))
                page_line_num += 1
        new_keys = {projection.identity_key for projection, _page_line_num in new_entries}
        if set(loaded_keys) - new_keys:
            self._render_pages(filtered_pages)
            target_idx = 0
            for index, projection in enumerate(self._session.projections):
                if projection.identity_key == old_current_key:
                    target_idx = index
            if self._pairs:
                self._activate(target_idx)
            for index, projection in enumerate(self._session.projections):
                if projection.identity_key in preserved_dirty_texts and index < len(self._pairs):
                    dirty_text, previous_loaded_text = preserved_dirty_texts[projection.identity_key]
                    self._pairs[index].restore_dirty_editor_text(
                        dirty_text,
                        previous_loaded_text=previous_loaded_text,
                    )
            self._update_stats()
            return
        projections = self._session.projections
        prev_page_number = projections[-1].page.page_number if projections else -1
        added = False
        for projection, page_line_num in new_entries:
                existing_index = loaded_keys.get(projection.identity_key)
                if existing_index is not None:
                    unit = self._unit_from_projection(projection, page_line_num)
                    projections[existing_index] = projection
                    self._pairs[existing_index].rebind(
                        projection.block, projection.line, projection.page, page_line_num,
                        unit.debug_badge,
                        unit=unit,
                    )
                    continue
                if projection.page.page_number != prev_page_number:
                    sep = QLabel(f"── 第 {projection.page.page_number} 页 ──")
                    sep.setObjectName("pageSep")
                    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sep.setMinimumHeight(26)
                    self._list_layout.insertWidget(self._list_layout.count() - 1, sep)
                    prev_page_number = projection.page.page_number
                self._append_projection(projection, page_line_num)
                loaded_keys[projection.identity_key] = len(projections) - 1
                added = True
        if added:
            self._empty_lbl.setVisible(False)
            QTimer.singleShot(100, self._load_visible_images)
        self._update_stats()

    def _refresh_page_filter(self) -> None:
        """仅维护左侧 PageDirectoryList 的页面项。"""
        usable_pages = self._session.usable_pages(self._page_has_lines)
        page_numbers = [p.page_number for p in usable_pages]
        self._session.filter_updating = True
        if hasattr(self, "_page_dir"):
            self._page_dir.set_pages(usable_pages)
            if self._session.selected_page_number in page_numbers:
                self._page_dir.set_current_index(
                    page_numbers.index(self._session.selected_page_number)
                )
        # 若选中页消失（被移除），回退到"全部页面"
        self._session.ensure_selected_page_exists(self._page_has_lines)
        self._session.filter_updating = False

    def _filtered_pages(self) -> List[Page]:
        return self._session.filtered_pages()

    def set_current_page_number(self, page_number: int) -> None:
        """外部联动调用：把过滤器切换到指定页。"""
        page_number = int(page_number)
        if self._session.selected_page_number == page_number:
            return
        if not self._flush_current_editor_before_view_switch():
            self._sync_page_dir_selection()
            return
        # 同步左侧目录视觉
        if hasattr(self, "_page_dir"):
            usable_numbers = [
                p.page_number for p in self._session.pages
                if self._page_has_lines(p)
            ]
            if page_number in usable_numbers:
                self._page_dir.set_current_index(usable_numbers.index(page_number))
        self._session.selected_page_number = page_number
        self._render_pages(self._filtered_pages())

    def _on_page_dir_selected(self, dir_idx: int) -> None:
        """左栏页面目录是页面过滤入口。点击 → 切换过滤 + 重渲染。"""
        if self._session.filter_updating:
            return
        usable_pages = self._session.usable_pages(self._page_has_lines)
        if not (0 <= dir_idx < len(usable_pages)):
            return
        target = usable_pages[dir_idx]
        if self._session.selected_page_number == target.page_number:
            return
        if not self._flush_current_editor_before_view_switch():
            self._sync_page_dir_selection()
            return
        self._session.selected_page_number = target.page_number
        self.page_selected.emit(int(target.page_number))
        self._render_pages(self._filtered_pages())

    def _sync_page_dir_selection(self) -> None:
        if not hasattr(self, "_page_dir"):
            return
        usable_numbers = [
            p.page_number for p in self._session.pages
            if self._page_has_lines(p)
        ]
        selected_page_number = self._session.selected_page_number
        if selected_page_number is None:
            self._session.filter_updating = True
            try:
                self._page_dir.setCurrentRow(-1)
                self._page_dir.clearSelection()
            finally:
                self._session.filter_updating = False
            return
        if selected_page_number not in usable_numbers:
            return
        self._session.filter_updating = True
        try:
            self._page_dir.set_current_index(
                usable_numbers.index(selected_page_number)
            )
        finally:
            self._session.filter_updating = False

    def _flush_current_editor_before_view_switch(self) -> bool:
        """Flush the active row before destroying/rebuilding HProof rows.

        Returning False means the active editor is in an external-conflict state
        and the caller must keep the current view alive.
        """
        current_index = self._session.current_projection_index
        if not self._pairs or not (0 <= current_index < len(self._pairs)):
            return True
        pair = self._pairs[current_index]
        state = ProofEditorRebuildState(
            editable=pair.is_editable,
            external_conflict=pair.has_external_conflict(),
            dirty=pair.is_editor_dirty(),
        )
        save_status = ProofEditStatus.NOOP
        if state.needs_save:
            save_status = self._save_current(silent=True)
        gate = proof_rebuild_gate_for_editor_state(
            state,
            save_status=save_status,
            conflict_message="当前行存在保存冲突，处理后再切换页面",
        )
        if not gate.allow_rebuild:
            pair._refresh_status()
            if hasattr(self, "_stat_lbl"):
                self._stat_lbl.setText(gate.message)
        return gate.allow_rebuild

    def _clear_pending_external_refresh(self) -> None:
        self._external_refresh_timer.stop()
        self._session.clear_pending_external_refresh()

    def _render_pages(self, pages: List[Page]) -> None:
        self._clear_pending_external_refresh()
        self._session.projections.clear()
        self._pairs.clear()

        # 清空旧 _LinePair。从后往前递删；skip _empty_lbl 和布局末尾的 stretch。
        # 为什么要 skip ：之前代码会 delete _empty_lbl ，导致二次 load 时
        # `self._empty_lbl.setVisible(...)` 变成访问已销毁的 C++ 对象 →
        # RuntimeError 中断后续清理 → 旧 _LinePair 仍贴在 _list_widget 上 →
        # 产生“重复页面”现象。
        for i in range(self._list_layout.count() - 1, -1, -1):
            item = self._list_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is None or w is self._empty_lbl:
                continue
            self._list_layout.takeAt(i)
            w.setParent(None)
            w.deleteLater()

        # 无数据时显示空状态
        self._empty_lbl.setText(self._empty_text())
        has_data = any(True for page in pages for _ in self._iter_page_lines(page))
        self._empty_lbl.setVisible(not has_data)
        self._update_mode_banner()

        prev_page_number: int = -1
        for page in pages:
            page_line_num = 1
            for block, line, li in self._iter_page_lines(page):
                # 每页第一行前插入页面分隔条，让用户清晰知道当前所处页面
                if page.page_number != prev_page_number:
                    sep = QLabel(f"── 第 {page.page_number} 页 ──")
                    sep.setObjectName("pageSep")
                    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sep.setMinimumHeight(26)
                    self._list_layout.insertWidget(
                        self._list_layout.count() - 1, sep
                    )
                    prev_page_number = page.page_number
                self._append_projection(
                    self._build_projection(page, block, line, li),
                    page_line_num,
                )
                page_line_num += 1

        self._session.current_projection_index = 0
        self._update_stats()
        if self._pairs:
            self._activate(0)
            # 懒加载前 30 行图像
            QTimer.singleShot(100, self._load_visible_images)

    def _build_projection(
        self,
        page: Page,
        block: Block,
        line: Line,
        li: int,
    ) -> ProofLineProjection:
        return build_proof_line_projection(
            page,
            block,
            line,
            li,
            display_text=_displayed_text(line, page, block),
            editable=li >= 0,
        )

    def _unit_from_projection(
        self,
        projection: ProofLineProjection,
        page_line_num: int,
    ) -> ProofUnit:
        debug_badge = self._debug_badge_for(projection.block, projection.line)
        return ProofUnit(
            uid=projection.uid,
            kind=_proof_unit_kind(projection, debug_badge),
            page=projection.page,
            block=projection.block,
            line=projection.line,
            line_index=projection.line_index,
            page_line_number=page_line_num,
            display_text=projection.display_text,
            debug_badge=debug_badge,
            atoms=projection.atoms,
            editable=projection.editable,
        )

    def _renderer_for(self, unit: ProofUnit) -> _ProofUnitRenderer:
        return self._renderers.get(unit.kind, self._renderers[ProofUnitKind.TEXT])

    def _append_projection(self, projection: ProofLineProjection, page_line_num: int) -> None:
        unit = self._unit_from_projection(projection, page_line_num)
        self._session.projections.append(projection)
        pair = self._renderer_for(unit).create_pair(len(self._pairs), unit, self._cache)
        pair.clicked.connect(self._on_pair_clicked)
        pair.text_saved.connect(self._on_text_saved)
        pair.confirmed.connect(self._on_confirmed)
        pair.prev_req.connect(self._prev)
        pair.next_req.connect(self._next)
        pair.flag_req.connect(self._toggle_flag)
        pair.skip_req.connect(self._next)
        self._pairs.append(pair)
        self._suppress_hover_tooltips(pair)
        self._list_layout.insertWidget(self._list_layout.count() - 1, pair)

    def _debug_enabled(self) -> bool:
        return self._session.debug_enabled

    def _debug_label(self) -> str:
        return self._session.debug_label

    def _scope_label(self) -> str:
        selected_page_number = self._session.selected_page_number
        if selected_page_number is None:
            scope = "全部页面"
        else:
            scope = f"第 {selected_page_number} 页"
        if self._debug_enabled():
            return f"{scope} · {self._debug_label()}调试"
        return f"{scope} · 正文"

    def _update_mode_banner(self) -> None:
        if not hasattr(self, "_mode_banner"):
            return
        if self._debug_enabled():
            self._mode_banner.setText(
                f"{self._debug_label()}调试视图 · 当前只显示路由命中的校验行"
            )
            self._mode_banner.setVisible(True)
        else:
            self._mode_banner.setText("")
            self._mode_banner.setVisible(False)

    def _empty_text(self) -> str:
        if self._debug_enabled():
            return f"当前页面没有可显示的{self._debug_label()}调试行"
        return "完成 OCR 识别后，横校数据将在此展示"

    def _iter_page_lines(self, page: Page) -> Iterator[tuple[Block, Line, int]]:
        if not self._debug_enabled():
            yield from iter_unique_page_hproof_lines(page)
            return
        yield from iter_unique_page_hproof_debug_lines(
            page,
            formulas=self._session.show_formula_debug,
            tables=self._session.show_table_debug,
        )

    def _page_has_lines(self, page: Page) -> bool:
        return any(True for _ in self._iter_page_lines(page))

    def _debug_badge_for(self, block: Block, line: Line) -> str:
        if not self._debug_enabled():
            return ""
        return _debug_line_kind(block, line)

    def _on_debug_filter_changed(self, *_args) -> None:
        new_formula = bool(self._btn_debug_formula.isChecked())
        new_table = bool(self._btn_debug_table.isChecked())
        if (
            new_formula == self._session.show_formula_debug
            and new_table == self._session.show_table_debug
        ):
            return
        if not self._flush_current_editor_before_view_switch():
            for btn, checked in (
                (self._btn_debug_formula, self._session.show_formula_debug),
                (self._btn_debug_table, self._session.show_table_debug),
            ):
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)
            return
        self._session.set_debug_flags(formula=new_formula, table=new_table)
        self._refresh_page_filter()
        self._update_mode_banner()
        self._render_pages(self._filtered_pages())

    def reset(self) -> None:
        self._session.reset()
        for btn in (
            getattr(self, "_btn_debug_formula", None),
            getattr(self, "_btn_debug_table", None),
        ):
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.load_pages([])
        self._stat_lbl.setText("")
        self._total_lbl.setText("总字数 0")
        self._diff_lbl.setText("差异 0 (0%)")

    # ── 导航 ───────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._session.current_projection_index > 0:
            self._save_current(silent=True)
            self._activate(self._session.current_projection_index - 1)

    def _next(self) -> None:
        if self._session.current_projection_index < len(self._pairs) - 1:
            self._save_current(silent=True)
            self._activate(self._session.current_projection_index + 1)

    def _ensure_pair_visible_left_aligned(self, pair: _LinePair) -> None:
        """Keep the active row visible without horizontal auto-centering."""
        vbar = self._scroll.verticalScrollBar()
        hbar = self._scroll.horizontalScrollBar()
        viewport_h = max(1, self._scroll.viewport().height())
        margin = 40
        top = pair.y()
        bottom = top + pair.height()
        view_top = vbar.value()
        view_bottom = view_top + viewport_h
        if top < view_top + margin:
            vbar.setValue(max(0, top - margin))
        elif bottom > view_bottom - margin:
            vbar.setValue(max(0, bottom - viewport_h + margin))
        hbar.setValue(0)

    def _activate(self, idx: int) -> None:
        idx = max(0, min(idx, len(self._pairs) - 1))
        current_index = self._session.current_projection_index
        if 0 <= current_index < len(self._pairs) and current_index != idx:
            self._pairs[current_index].set_active(False)
        self._session.current_projection_index = idx
        pair = self._pairs[idx]
        pair.set_active(True)
        self._update_focus_depths()
        self._update_stats()
        # 滚动到可见
        QTimer.singleShot(30, lambda pair=pair: self._ensure_pair_visible_left_aligned(pair))

    def _update_focus_depths(self) -> None:
        for i, pair in enumerate(self._pairs):
            distance = abs(i - self._session.current_projection_index)
            if distance == 0:
                depth = "active"
            elif distance == 1:
                depth = "near"
            else:
                depth = "far"
            pair.set_focus_depth(depth)

    def _on_pair_clicked(self, idx: int) -> None:
        if idx != self._session.current_projection_index:
            self._save_current(silent=True)
            self._activate(idx)

    def _on_confirmed(self, idx: int) -> None:
        """Enter 键确认当前行。"""
        projections = self._session.projections
        if not projections:
            return
        if 0 <= idx < len(self._pairs) and not self._pairs[idx].is_editable:
            return
        if 0 <= idx < len(self._pairs) and self._pairs[idx].has_external_conflict():
            self._pairs[idx]._refresh_status()
            return
        projection = projections[idx]
        block, line, page, li = (
            projection.block,
            projection.line,
            projection.page,
            projection.line_index,
        )
        pair = self._pairs[idx] if 0 <= idx < len(self._pairs) else None
        if pair is None:
            return
        # 先保存文本
        save_result = self._save_current(silent=True)
        if save_result == ProofEditStatus.CONFLICT:
            return
        status_result = ProofEditService.set_line_status(
            page,
            block,
            line,
            ProofStatus.OK,
            expected_signature=pair.loaded_line_signature() if pair is not None else "",
        )
        if status_result.blocked:
            if pair is not None:
                pair.mark_external_conflict()
            return
        if not status_result.changed:
            self._next()
            return
        self._publish_line_update(
            page=page, block=block, line=line, line_index=li,
            status=ProofStatus.OK.value, source="hproof.confirm",
        )
        self._pairs[idx].refresh_text()
        self._update_stats()
        self._emit_proof_change(status_result.change)
        self._next()

    def _on_text_saved(self, idx: int, new_text: str) -> None:
        """_LinePair 在失焦或主动保存时提交文本。

        保存成功后，被改动行的 proof_status 通常会从 UNCHECKED 变 MODIFIED。
        这里只刷新状态点，避免重建 editor 干扰当前焦点。"""
        projections = self._session.projections
        if idx >= len(projections):
            return
        projection = projections[idx]
        block, line, page, li = (
            projection.block,
            projection.line,
            projection.page,
            projection.line_index,
        )
        pair = self._pairs[idx] if 0 <= idx < len(self._pairs) else None
        if pair is None:
            return
        result = ProofEditService.replace_line_text(
            page,
            block,
            line,
            new_text,
            expected_signature=pair.loaded_line_signature(),
        )
        if result.blocked:
            pair.mark_external_conflict()
            return
        if result.changed:
            self._publish_line_update(
                page=page, block=block, line=line, line_index=li,
                status=proof_status(line).value, source="hproof.text_saved",
            )
            self._update_stats()
            if pair is not None:
                pair.mark_editor_saved()
            self._emit_proof_change(result.change)

    def _save_current(self, *, silent: bool = False) -> ProofEditStatus:
        """将当前编辑器内容保存到 line 对象。"""
        current_index = self._session.current_projection_index
        if not self._pairs or current_index >= len(self._pairs):
            return ProofEditStatus.NOOP
        pair = self._pairs[current_index]
        if pair.has_external_conflict():
            pair._refresh_status()
            return ProofEditStatus.CONFLICT
        if not pair.is_editable:
            return ProofEditStatus.READONLY
        # editor 始终可见，直接读其当前文本与 line 比较保存。
        new_text = pair._editor.toPlainText()
        projection = self._session.projections[current_index]
        block, line, page, li = (
            projection.block,
            projection.line,
            projection.page,
            projection.line_index,
        )
        result = ProofEditService.replace_line_text(
            page,
            block,
            line,
            new_text,
            expected_signature=pair.loaded_line_signature(),
        )
        if result.blocked:
            pair.mark_external_conflict()
            return ProofEditStatus.CONFLICT
        if result.changed:
            self._publish_line_update(
                page=page, block=block, line=line, line_index=li,
                status=proof_status(line).value, source="hproof.save_current",
            )
            pair.refresh_text(force=True)
            self._update_stats()
            self._emit_proof_change(result.change)
            return ProofEditStatus.SAVED
        return ProofEditStatus.NOOP

    def _save_all(self) -> None:
        self._save_current()

    def _emit_proof_change(self, change: ProofChangeSet) -> None:
        if not change.needs_persist:
            return
        self.proof_changed.emit(change)

    def _toggle_flag(self) -> None:
        current_index = self._session.current_projection_index
        projections = self._session.projections
        if not projections or current_index >= len(projections):
            return
        projection = projections[current_index]
        block, line, page, li = (
            projection.block,
            projection.line,
            projection.page,
            projection.line_index,
        )
        pair = self._pairs[current_index]
        if li < 0 or not pair.is_editable:
            return
        new_status = (
            ProofStatus.UNCHECKED
            if proof_status(line) == ProofStatus.AUTO_FLAGGED
            else ProofStatus.AUTO_FLAGGED
        )
        result = ProofEditService.set_line_status(
            page,
            block,
            line,
            new_status,
            expected_signature=pair.loaded_line_signature(),
        )
        if result.blocked:
            pair.mark_external_conflict()
            return
        if not result.changed:
            return
        self._publish_line_update(
            page=page, block=block, line=line, line_index=li,
            status=new_status.value, source="hproof.flag",
        )
        pair.refresh_text()
        self._update_stats()
        self._emit_proof_change(result.change)

    # ── 统计 ───────────────────────────────────────────────────

    def _publish_line_update(
        self,
        *,
        page: Page,
        block: Block,
        line: Line,
        line_index: int,
        status: str,
        source: str,
    ) -> None:
        request = ProofUpdateRequest(
            page_id=page.id,
            line_id=line.id,
            status=status,
            page_uid=page.uid,
            line_uid=line.uid,
            origin=id(self),
            selection=ProofSelection.for_line(
                page=page, block=block, line=line, line_index=line_index, source=source,
            ),
            source=source,
        )
        self._bus.publish_line_update(request)

    def _on_external_line_changed(self, request: ProofUpdateRequest) -> None:
        """收到外部（纵校）发来的 line.proof_changed → 找到本 panel 中
        line.id 匹配的行，刷新该行显示并重算统计。

        回路保护：origin == id(self) 时直接跳过（自己 publish 的事件）。
        """
        if not isinstance(request, ProofUpdateRequest):
            return
        if request.origin == id(self):
            return
        if (request.line_uid or None) is None and request.line_id is None:
            return
        if not self._session.has_projection_for_request(request):
            return
        self._session.queue_external_refresh(request)
        self._external_refresh_timer.start()

    def _do_external_refresh(self) -> None:
        plan = self._session.consume_external_refresh_plan()
        if not plan.has_work:
            return
        for i in plan.touched_projection_indexes:
            if i < len(self._pairs):
                self._pairs[i].refresh_text()
        self._update_stats()

    def _update_stats(self) -> None:
        projections = self._session.projections
        total_chars = sum(len(projection.display_text) for projection in projections)
        diff_count  = sum(
            1 for projection in projections
            if proof_status(projection.line) in (ProofStatus.MODIFIED, ProofStatus.AUTO_FLAGGED)
        )
        pct = (diff_count / max(1, len(projections))) * 100
        self._total_lbl.setText(f"总字数 {total_chars:,}")
        self._diff_lbl.setText(f"差异 {diff_count} ({pct:.1f}%)")

        total_lines = len(projections)
        current = self._session.current_projection_index + 1 if self._pairs else 0
        confirmed = sum(
            1 for projection in projections
            if proof_status(projection.line) == ProofStatus.OK
        )
        modified = sum(
            1 for projection in projections
            if proof_status(projection.line) == ProofStatus.MODIFIED
        )
        flagged = sum(
            1 for projection in projections
            if proof_status(projection.line) == ProofStatus.AUTO_FLAGGED
        )
        pending = sum(
            1 for projection in projections
            if proof_status(projection.line) == ProofStatus.UNCHECKED
        )
        handled = confirmed + modified
        self._last_stats = {
            "scope": self._scope_label(),
            "current": current,
            "total_lines": total_lines,
            "handled": handled,
            "pending": pending,
            "confirmed": confirmed,
            "modified": modified,
            "flagged": flagged,
            "total_chars": total_chars,
            "diff_count": diff_count,
            "diff_pct": pct,
        }

        if hasattr(self, "_current_scope_lbl"):
            self._current_scope_lbl.setText(self._scope_label())
            self._current_line_lbl.setText(f"当前行 {current} / {total_lines}")
            self._proof_progress_bar.setRange(0, max(1, total_lines))
            self._proof_progress_bar.setValue(handled)
            self._handled_lbl.setText(f"已处理 {handled} / {total_lines}")
            self._pending_lbl.setText(f"待确认 {pending}")
            self._confirmed_lbl.setText(f"已确认 {confirmed}")
            self._modified_lbl.setText(f"已修改 {modified}")
            self._flagged_lbl.setText(f"疑点 {flagged}")
            if hasattr(self, "_progress_ring"):
                self._progress_ring.set_counts(handled, total_lines)
            self._stat_lbl.setText(self._debug_label() if self._debug_enabled() else "")

    def _show_progress_popup(self) -> None:
        stats = getattr(self, "_last_stats", None) or {}
        popup = QFrame(self, Qt.WindowType.Popup)
        popup.setObjectName("proofProgressPopup")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        title = QLabel(str(stats.get("scope") or "横校进度"))
        title.setObjectName("sectionTitle")
        lay.addWidget(title)

        rows = (
            ("当前行", f"{stats.get('current', 0)} / {stats.get('total_lines', 0)}"),
            ("已处理", f"{stats.get('handled', 0)} / {stats.get('total_lines', 0)}"),
            ("待确认", str(stats.get("pending", 0))),
            ("已确认", str(stats.get("confirmed", 0))),
            ("已修改", str(stats.get("modified", 0))),
            ("疑点", str(stats.get("flagged", 0))),
            ("总字数", f"{int(stats.get('total_chars', 0)):,}"),
            ("差异", f"{stats.get('diff_count', 0)} ({float(stats.get('diff_pct', 0.0)):.1f}%)"),
        )
        for key, value in rows:
            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(18)
            k = QLabel(key)
            k.setObjectName("muted")
            v = QLabel(value)
            v.setObjectName("fieldLabel")
            row_lay.addWidget(k)
            row_lay.addStretch(1)
            row_lay.addWidget(v)
            lay.addWidget(row)

        popup.adjustSize()
        anchor = self._progress_ring.mapToGlobal(self._progress_ring.rect().topLeft())
        hint = popup.sizeHint()
        popup.move(anchor.x(), max(0, anchor.y() - hint.height() - 8))
        popup.show()
        self._progress_popup = popup

    def refresh_quality_probe_state(self) -> None:
        """重新渲染所有可见行，使显示空间文本与全局 active store 对齐。

        这里只刷新 pair 显示，不改变模型事实。"""
        for pair in self._pairs:
            pair.refresh_text()

    # ── 懒加载图像 ─────────────────────────────────────────────

    def _load_visible_images(self) -> None:
        """加载当前视口附近行的图像。"""
        if not self._pairs or not self.isVisible():
            return
        vp = self._scroll.viewport()
        vp_top = self._scroll.verticalScrollBar().value()
        vp_bot = vp_top + vp.height()
        for pair in self._pairs:
            y = pair.y()
            h = pair.height()
            # 加载视口 ±2 屏范围内的图像
            if y + h >= vp_top - vp.height() * 2 and y <= vp_bot + vp.height() * 2:
                pair.load_image()
