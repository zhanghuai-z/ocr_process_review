"""Horizontal proof view over an immutable :class:`ProofWorkspaceView`.

The widget owns only display state and an editor's transient text buffer.  It
never reads a repository or applies a proof edit.  Every mutation request is
represented by a ``ProofEditCommand`` emitted to the application boundary.

Interaction model restored from the mature proof workspace (d4c6dfe):

- one row at a time, image and text aligned on the same x axis;
- a fixed-length slot editor: typing overwrites the current slot, deletion
  fills blanks, pastes are truncated/padded so text length stays locked to
  the OCR character geometry;
- per-character verdict colouring (never claims "absolutely correct", a user
  edit only adds an underline, it never whitewashes the OCR evidence);
- a pure keyboard loop: Enter confirms and advances, F5 flags, F6 skips,
  Esc reverts to the OCR text, Ctrl+S saves every dirty row;
- dirty editors survive workspace snapshot replacements; rows whose loaded
  text changed underneath a dirty editor are marked as conflicts instead of
  silently overwriting the user's work.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QFontMetricsF,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from app.application.contracts import ProofEditCommand
from app.application.proof_workspace import (
    ProofAtomView,
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspaceView,
)
from app.ui.proof import char_verdict as _cv
from app.ui.proof.confidence_view import ProofCharView, build_char_views, normalize_confidence
from app.ui.proof.formula_renderer import render_formula_pixmap
from app.ui.widgets.page_thumbnail import PAGE_ROW_H, PageDirectoryRow


ROW_IMAGE_SIZE = QSize(300, 54)
STATUS_COLORS = {
    "checked": "#4E7A63",
    "ok": "#4E7A63",
    "modified": "#5C6B58",
    "flagged": "#C67B22",
    "auto_flagged": "#C67B22",
    "conflict": "#C62828",
    "unchecked": "#8090A0",
}
STATUS_LABELS = {
    "unchecked": "待校对",
    "checked": "已校对",
    "ok": "已校对",
    "modified": "已修改",
    "flagged": "疑点",
    "auto_flagged": "疑点",
    "conflict": "冲突",
}

# Painted text layer constants (restored slot editor typography).
TEXT_FONT_FAMILY = "'Noto Serif CJK SC','Source Han Serif SC','SimSun','Songti SC','Times New Roman',serif"
TEXT_FONT_FAMILIES = [
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "SimSun",
    "Songti SC",
    "Times New Roman",
]
TEXT_FONT_WEIGHT = QFont.Weight.Bold
TEXT_LINE_HEIGHT_PX = 34
TEXT_EDITOR_DEFAULT_H = 40
TEXT_SLOT_MIN_W = 10.0
TEXT_SLOT_GUTTER_W = 4.0
TEXT_SLOT_CLIPPED_MIN_W = 1.0

# Row proportions restored from the mature workspace: the active row carries
# the editing focus; surrounding rows collapse both in height and opacity so
# attention stays on the line being edited.  All row images render at one
# uniform height per focus depth so every line shares the same scale.
IMAGE_ROW_H = 39
NEAR_IMAGE_ROW_H = 25
FAR_IMAGE_ROW_H = 21
MAX_LINE_IMAGE_W = 1200
LINE_PAIR_MIN_H = 108
LINE_PAIR_MAX_H = 120
NEAR_LINE_PAIR_MIN_H = 52
NEAR_LINE_PAIR_MAX_H = 58
FAR_LINE_PAIR_MIN_H = 48
FAR_LINE_PAIR_MAX_H = 54
FOCUS_OPACITY = {"active": 1.0, "near": 0.45, "far": 0.30}

FORMULA_IMAGE_ROW_H = 96
FORMULA_RENDER_TARGET_H = 72
FORMULA_RENDER_AREA_H = FORMULA_RENDER_TARGET_H + 8
FORMULA_SOURCE_PANEL_H = 82
FORMULA_VISUAL_HEIGHT_RATIO = 0.98
FORMULA_SOURCE_POPUP_OFFSET = QPoint(10, 18)
TABLE_ROW_IMAGE_H = 200
TABLE_LINE_PAIR_MIN_H = 268
TABLE_LINE_PAIR_MAX_H = 284

_CONFIRMED_STATUSES = {"checked", "ok"}
_FLAGGED_STATUSES = {"flagged", "auto_flagged"}
_STATUS_GLYPHS = {
    "checked": "✓",
    "ok": "✓",
    "modified": "✎",
    "flagged": "⚠",
    "auto_flagged": "⚠",
    "conflict": "!",
    "unchecked": "✕",
}

# 状态图标（iconify/tabler 素材，stroke=currentColor，运行时按状态色着色）
_STATUS_ICON_DIR = (
    Path(__file__).resolve().parents[3] / "resources" / "icons" / "proof"
)
_STATUS_ICON_FILES = {
    "checked": "check.svg",
    "ok": "check.svg",
    "modified": "pencil.svg",
    "flagged": "alert-triangle.svg",
    "auto_flagged": "alert-triangle.svg",
    "conflict": "exclamation-circle.svg",
    "unchecked": "x.svg",
}
_status_icon_cache: dict[tuple[str, str], QPixmap] = {}


def _tinted_status_icon(status: str, color: QColor, size: int = 14) -> QPixmap:
    """Status icon tinted to the status color; glyph fallback when missing."""

    key = (status, color.name())
    cached = _status_icon_cache.get(key)
    if cached is not None:
        return cached
    from PySide6.QtGui import QIcon

    path = _STATUS_ICON_DIR / _STATUS_ICON_FILES.get(status, "x.svg")
    base = QIcon(str(path)).pixmap(size, size) if path.is_file() else QPixmap()
    if base.isNull():
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QPen(color, 1))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            pixmap.rect(), Qt.AlignmentFlag.AlignCenter, _STATUS_GLYPHS.get(status, "?")
        )
        painter.end()
    else:
        pixmap = QPixmap(base.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, base)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
    _status_icon_cache[key] = pixmap
    return pixmap


def _slot_visual_width(text_char: str, font_metrics: QFontMetrics) -> float:
    """Visual slot width for the painted text layer (not the OCR bbox)."""

    glyph_width = float(font_metrics.horizontalAdvance(text_char or " ")) + TEXT_SLOT_GUTTER_W
    return max(TEXT_SLOT_MIN_W, glyph_width)


def _is_punctuation_slot_text(text: str) -> bool:
    """Whether a slot should draw text centered inside its visual cell."""

    if not text:
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in text)


def _char_slot_weight(ch: str) -> float:
    """Estimated horizontal ink share of one character inside an atom bbox.

    CJK and full-width forms occupy the full em; ASCII letters/digits are a
    bit over half; half-width punctuation is narrow; space is half.  Used to
    distribute a multi-character atom bbox across its characters.
    """

    if not ch:
        return 1.0
    if ch == " ":
        return 0.5
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    category = unicodedata.category(ch)
    if category.startswith("P"):
        return 0.35
    if category == "Nd":
        return 0.56
    if category.startswith("L"):
        return 0.56
    if category.startswith("N"):
        return 0.6
    return 0.8


@dataclass(frozen=True, slots=True)
class _FormulaVisual:
    text: str | None
    pixmap: QPixmap | None = None
    logical_size: QSize | None = None
    kind: str = "formula"


@dataclass(frozen=True, slots=True)
class _AtomVisualOverlay:
    """One rendered formula atom placed over its exact text span."""

    start: int
    end: int
    left: float
    right: float
    text: str
    pixmap: QPixmap | None = None
    logical_size: QSize | None = None
    kind: str = "formula"


_LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "cdot": "·", "pm": "±", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "neq": "≠", "approx": "≈", "infty": "∞", "sum": "∑",
    "prod": "∏", "int": "∫", "partial": "∂", "nabla": "∇",
    "rightarrow": "→", "to": "→", "leftarrow": "←",
}

_SUPERSCRIPT = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
    ")": "⁾", "i": "ⁱ", "n": "ⁿ",
})
_SUBSCRIPT = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
    ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ",
    "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
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
    s = re.sub(rf"(?<=[{latin}])\s+(?=[{latin}_])", "", s)
    s = re.sub(rf"(?<=[{latin}0-9])\s+(?=[{sub_sup}])", "", s)
    s = re.sub(rf"(?<=[{sub_sup}])\s+(?=[{latin}0-9])", "", s)
    s = re.sub(r"\s*([=+\-×·*/])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=\S)", "", s)
    s = re.sub(r"(?<=\S)\s+(?=[\u3400-\u9fff])", "", s)
    return s


def _render_formula_display(text: str) -> str:
    """Best-effort visual text for formula rows (display-only fallback).

    The editor document keeps the raw OCR/LaTeX text; this renderer only
    produces a readable approximation when the pixmap renderer is
    unavailable.
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


def _render_formula_visual(text: str, *, target_height: int) -> _FormulaVisual | None:
    """Render formula text to a visual layer: MathJax pixmap, text fallback."""

    fallback_text = _render_formula_display(text)
    result = render_formula_pixmap(
        text,
        target_height=max(8, int(target_height)),
        color="#2C2C2C",
    )
    if result is not None:
        return _FormulaVisual(
            fallback_text or None,
            result.pixmap,
            QSize(result.logical_width, result.logical_height),
            kind="formula",
        )
    if fallback_text:
        return _FormulaVisual(
            fallback_text,
            kind="" if _contains_cjk_text(fallback_text) else "formula",
        )
    return None


def _formula_preview_source(source: str, number_text: str = "") -> str:
    """Compose the linked number into a render-only formula preview."""

    body = (source or "").strip()
    number = (number_text or "").strip().strip("$ ")
    if len(number) >= 2 and (number[0], number[-1]) in {
        ("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"),
    }:
        number = number[1:-1].strip()
    if not body or not number:
        return body
    body = re.sub(r"^\s*\$\$?\s*", "", body)
    body = re.sub(r"\s*\$\$?\s*$", "", body)
    body = re.sub(r"\\tag\*?\s*\{[^{}]*\}", "", body)
    return rf"{body.strip()} \qquad ({number})"


@dataclass(frozen=True, slots=True)
class _AtomPlacement:
    atom: ProofAtomView
    char_indices: tuple[int, ...]


def _atom_kind(atom: ProofAtomView) -> str | None:
    return atom.render_kind if atom.render_kind in {"formula", "table"} else None


def _explicit_char_span(atom: ProofAtomView) -> tuple[int, int] | None:
    raw_span = atom.char_span
    if raw_span is None:
        return None
    if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 2:
        raise TypeError("ProofAtomView.char_span must contain two integers")
    start, end = raw_span
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError("ProofAtomView.char_span must contain two integers")
    if start < 0 or end <= start:
        raise ValueError("ProofAtomView.char_span must be a non-empty half-open span")
    return start, end


def _atom_placements(line: ProofLineView) -> tuple[_AtomPlacement, ...]:
    placements: list[_AtomPlacement] = []
    text_length = len(line.proof_text)
    for atom in sorted(line.atoms, key=lambda item: (item.atom_index, item.atom_uid)):
        span = _explicit_char_span(atom)
        char_indices = () if span is None else tuple(range(*span))
        if any(char_index >= text_length for char_index in char_indices):
            raise ValueError(
                f"atom {atom.atom_uid!r} char span exceeds proof text length"
            )
        placements.append(_AtomPlacement(atom=atom, char_indices=char_indices))
    return tuple(placements)


def _row_kind(line: ProofLineView, placements: tuple[_AtomPlacement, ...]) -> str:
    """Line-level presentation kind.

    Only the line's own render kind promotes a row: a text line containing
    inline formula atoms stays a text row (the atom spans get rendered
    overlays), while a standalone formula/table region becomes a
    formula/table row.  `placements` is retained for call-site symmetry.
    """

    if line.render_kind == "formula":
        return "formula"
    if line.render_kind == "table":
        return "table"
    return "text"


def _row_preview_source(
    line: ProofLineView,
    placements: tuple[_AtomPlacement, ...],
    kind: str,
) -> str:
    if kind == "formula":
        formula_text = "".join(
            item.atom.text
            for item in placements
            if _atom_kind(item.atom) == "formula" and item.atom.text
        )
        return formula_text or line.proof_text
    return line.proof_text


def _row_image_bbox(
    line: ProofLineView,
    placements: tuple[_AtomPlacement, ...],
    formula_number_line: ProofLineView | None = None,
) -> tuple[int, int, int, int] | None:
    boxes = [item.atom.bbox for item in placements]
    if line.bbox is not None:
        boxes.insert(0, line.bbox)
    if formula_number_line is not None and formula_number_line.bbox is not None:
        boxes.append(formula_number_line.bbox)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


@dataclass(frozen=True, slots=True)
class _ProofRow:
    key: tuple[str, str]
    page: ProofPageView
    state: ProofStateView
    unit: ProofTextUnitView
    line: ProofLineView
    entries: tuple[ProofCharView, ...]
    atom_placements: tuple[_AtomPlacement, ...]
    kind: str
    preview_source: str
    formula_number_line: ProofLineView | None = None


# ─────────────────────────────────────────────────────────────
# 逐字槽位编辑器（自绘，QTextDocument 仅作文本容器）
# ─────────────────────────────────────────────────────────────

class _SlotLineEditor(QWidget):
    """Per-character slot editor for horizontal proofing.

    The widget uses a ``QTextDocument`` only as the text container; glyphs,
    hit areas and selection backgrounds are painted from the slot geometry
    derived from OCR character boxes, so the visual position never drifts
    from the cursor model.  All edits funnel through
    ``_replace_selected_slots`` which truncates/pads the replacement so the
    total text length never changes implicitly.
    """

    confirm_requested = Signal()
    prev_requested = Signal()
    next_requested = Signal()
    flag_requested = Signal()
    skip_requested = Signal()
    revert_requested = Signal()
    save_all_requested = Signal()
    formula_source_requested = Signal(int, int, QPoint)
    hover_char_changed = Signal(int)
    row_focus_requested = Signal()
    textChanged = Signal()
    selectionChanged = Signal()
    cursorPositionChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(TEXT_EDITOR_DEFAULT_H)
        self._document = QTextDocument(self)
        self._document.setDocumentMargin(0)
        self._cursor = QTextCursor(self._document)
        self._cursor_width = 0
        self._slot_x_centers: list[float | None] | None = None
        self._slot_widths: list[float] | None = None
        self._extra_selections: list = []
        self._last_hover_idx = -1
        self._active_visual = False
        self._read_only = False
        self._placeholder = ""
        self._formula_editing_enabled = False
        self._visual_text_override: str | None = None
        self._visual_pixmap_override: QPixmap | None = None
        self._visual_pixmap_logical_size: QSize | None = None
        self._visual_text_kind: str = ""
        self._atom_visual_overlays: list[_AtomVisualOverlay] = []
        self._undo_stack: list[tuple[str, int, int]] = []
        self._redo_stack: list[tuple[str, int, int]] = []
        self._max_undo = 100
        self._apply_document_line_height()

    # ── QPlainTextEdit-like API used by the row widget ────────

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

    def setPlaceholderText(self, text: str) -> None:
        self._placeholder = text or ""
        self.update()

    def set_active_visual(self, active: bool) -> None:
        self._active_visual = bool(active)
        self.update()

    def setReadOnly(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self.setCursor(Qt.CursorShape.ArrowCursor if self._read_only else Qt.CursorShape.IBeamCursor)

    def isReadOnly(self) -> bool:
        return self._read_only

    def set_formula_editing_enabled(self, enabled: bool) -> None:
        self._formula_editing_enabled = bool(enabled)

    def set_atom_visual_overlays(self, overlays: list["_AtomVisualOverlay"] | None) -> None:
        self._atom_visual_overlays = list(overlays or [])
        self.update()

    def has_atom_visual_overlays(self) -> bool:
        return bool(self._atom_visual_overlays)

    def atom_visual_overlays(self) -> list["_AtomVisualOverlay"]:
        return list(self._atom_visual_overlays)

    def slot_geometry(self) -> tuple[list[float | None] | None, list[float] | None]:
        return self._slot_x_centers, self._slot_widths

    def set_visual_formula_override(self, visual: _FormulaVisual | None) -> None:
        if visual is None:
            self.set_visual_text_override(None)
            return
        self._visual_text_override = visual.text or None
        self._visual_pixmap_override = visual.pixmap
        self._visual_pixmap_logical_size = (
            visual.logical_size if visual.pixmap is not None else None
        )
        self._visual_text_kind = visual.kind
        self.update()

    def set_visual_text_override(self, text: str | None, *, kind: str = "") -> None:
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

    def has_visual_text_override(self) -> bool:
        return bool(self._visual_text_override) or self._visual_pixmap_override is not None

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

    def set_slot_geometry(
        self,
        x_centers: list[float | None] | None,
        widths: list[float] | None,
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

    # ── slot selection / text mutation ─────────────────────────

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

        Reserved for formula source editing, where the source length is
        allowed to change.
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

    # ── painting ───────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            option = QStyleOption()
            option.initFrom(self)
            self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, option, painter, self)
            if self._visual_pixmap_override is not None:
                self._paint_visual_pixmap_override(painter, self._visual_pixmap_override)
                return
            if self._visual_text_override:
                self._paint_visual_text_override(painter, self._visual_text_override)
                return
            text = self.toPlainText()
            if not text:
                if self._placeholder:
                    font = QFont(self.font())
                    painter.setFont(font)
                    painter.setPen(QPen(QColor("#8C8C8C"), 1))
                    painter.drawText(
                        self.rect().adjusted(6, 0, -6, 0),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        self._placeholder,
                    )
                return
            font = QFont(self.font())
            font.setWeight(TEXT_FONT_WEIGHT)
            painter.setFont(font)
            fm = QFontMetrics(font)
            centers = self._slot_x_centers or self._fallback_slot_centers(text)
            widths = self._slot_widths or [
                _slot_visual_width(ch, fm)
                for ch in text
            ]
            explicit_widths = self._slot_widths is not None
            fg_color, bg_color, underline_color = self._selection_colors()
            if self._active_visual:
                selected_start, selected_end = self._selection_bounds_for_paint()
            else:
                selected_start, selected_end = -1, -1
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
                    painter.fillRect(cell, QColor("#e8f0fe"))
                if selected_start <= i < selected_end:
                    painter.fillRect(cell, QColor("#cfe2ff"))
                bg = bg_color.get(i)
                if bg is not None and bg.alpha() > 0:
                    painter.fillRect(cell, bg)
                if i == self._last_hover_idx or selected_start <= i < selected_end:
                    painter.setPen(QPen(QColor("#9cc2ff"), 1))
                    painter.drawRect(cell.adjusted(0, 0, -1, -1))
                color = fg_color.get(i) or self.palette().text().color()
                painter.setPen(QPen(color, 1))
                ch = text[i]
                if _is_punctuation_slot_text(ch):
                    ink = fm.tightBoundingRect(ch)
                    tx = float(center) - ink.width() / 2.0 - ink.left()
                    painter.drawText(int(round(tx)), y_baseline, ch)
                else:
                    char_w = fm.horizontalAdvance(ch)
                    tx = int(round(float(center) - char_w / 2.0))
                    painter.drawText(tx, y_baseline, ch)
                underline = underline_color.get(i)
                if underline is not None:
                    painter.setPen(QPen(underline, 1))
                    baseline_y = cell.bottom() - 1
                    painter.drawLine(cell.left(), baseline_y, cell.right(), baseline_y)
            for overlay in self._atom_visual_overlays:
                self._paint_atom_visual_overlay(
                    painter,
                    overlay,
                    selected_start=selected_start,
                    selected_end=selected_end,
                )
        finally:
            painter.end()

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

    def _fallback_slot_centers(self, text: str) -> list[float | None]:
        if not text:
            return []
        font = QFont(self.font())
        font.setWeight(TEXT_FONT_WEIGHT)
        fm = QFontMetrics(font)
        x = max(6.0, TEXT_SLOT_MIN_W / 2.0)
        centers: list[float | None] = []
        for ch in text:
            w = _slot_visual_width(ch, fm)
            centers.append(x + w / 2.0)
            x += w
        return centers

    def _selection_colors(self) -> tuple[dict[int, QColor], dict[int, QColor], dict[int, QColor]]:
        fg_color: dict[int, QColor] = {}
        bg_color: dict[int, QColor] = {}
        underline_color: dict[int, QColor] = {}
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
            if fmt.underlineStyle() != QTextCharFormat.UnderlineStyle.NoUnderline:
                color = fmt.underlineColor()
                if not color.isValid():
                    color = fg_color.get(start) or self.palette().text().color()
                for i in range(start, end):
                    underline_color[i] = color
        return fg_color, bg_color, underline_color

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

    # ── events ─────────────────────────────────────────────────

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
            elif self._formula_editing_enabled or self.has_visual_text_override():
                self.formula_source_requested.emit(
                    0,
                    text_len,
                    self.mapToGlobal(pos),
                )
            else:
                self.revert_requested.emit()
            try:
                event.accept()
            except Exception:
                pass
            return
        self.row_focus_requested.emit()
        pos = self._event_pos(event)
        overlay = self._atom_overlay_for_x(float(pos.x()))
        if overlay is not None:
            # 点击渲染公式段：整段选中（等同一组槽位），不逐字定位
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

    def focusNextPrevChild(self, next_child: bool) -> bool:  # type: ignore[override]
        # Tab/Shift+Tab 交给 keyPressEvent 做行导航，不做控件间焦点跳转
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
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
        if key == Qt.Key.Key_Backtab or (key == Qt.Key.Key_Tab and shift):
            self.prev_requested.emit()
            return
        if key == Qt.Key.Key_Tab and no_mod:
            self.next_requested.emit()
            return
        if ctrl and key == Qt.Key.Key_S:
            self.save_all_requested.emit()
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


class _FormulaLineSourceEdit(QPlainTextEdit):
    """Inline source editor used as the third row of display-formula rows.

    Unlike the slot editor it owns plain source text (LaTeX), so its length
    may change freely; commits flow through the same panel pipeline as slot
    edits.  Long source scrolls horizontally instead of being truncated.
    Geometry/verdict/slot APIs are accepted as no-ops so the row widget can
    treat both editor kinds uniformly.
    """

    confirm_requested = Signal()
    prev_requested = Signal()
    next_requested = Signal()
    flag_requested = Signal()
    skip_requested = Signal()
    revert_requested = Signal()
    save_all_requested = Signal()
    formula_source_requested = Signal(int, int, QPoint)
    hover_char_changed = Signal(int)
    row_focus_requested = Signal()
    selectionChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(False)
        mono = QFont(self.font())
        mono.setFamilies(["JetBrains Mono", "Consolas", "monospace"])
        mono.setPixelSize(14)
        self.setFont(mono)
        self.cursorPositionChanged.connect(self.selectionChanged)

    # ── uniform no-op surface consumed by the row widget ───────

    def set_slot_geometry(self, *_args, **_kwargs) -> None:
        return

    def set_active_visual(self, *_args, **_kwargs) -> None:
        return

    def set_formula_editing_enabled(self, *_args, **_kwargs) -> None:
        return

    def set_visual_formula_override(self, *_args, **_kwargs) -> None:
        return

    def set_visual_text_override(self, *_args, **_kwargs) -> None:
        return

    def has_visual_text_override(self) -> bool:
        return False

    def visual_text_content_width(self) -> int:
        return 0

    def set_atom_visual_overlays(self, *_args, **_kwargs) -> None:
        return

    def slot_geometry(self):
        return None, None

    def replace_text_range(self, start: int, end: int, replacement: str) -> None:
        current = self.toPlainText()
        start = max(0, min(int(start), len(current)))
        end = max(start, min(int(end), len(current)))
        cursor = self.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement or "")

    # ── events ─────────────────────────────────────────────────

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        self.row_focus_requested.emit()
        super().focusInEvent(event)

    def focusNextPrevChild(self, next_child: bool) -> bool:  # type: ignore[override]
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        mod = event.modifiers()
        no_mod = mod == Qt.KeyboardModifier.NoModifier
        ctrl = bool(mod & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mod & Qt.KeyboardModifier.ShiftModifier)
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and no_mod:
            self.confirm_requested.emit()
            return
        if key == Qt.Key.Key_Backtab or (key == Qt.Key.Key_Tab and shift):
            self.prev_requested.emit()
            return
        if key == Qt.Key.Key_Tab and no_mod:
            self.next_requested.emit()
            return
        if ctrl and key == Qt.Key.Key_S:
            self.save_all_requested.emit()
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
        super().keyPressEvent(event)


class _FormulaSourcePopup(QFrame):
    """Small formula source editor opened from a rendered formula row.

    No confirm/cancel buttons: edits apply live (160ms debounce), closing
    (Esc / Ctrl+Enter / click outside) flushes the last change.
    """

    source_changed = Signal(str)
    closed = Signal()
    command_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("formulaSourcePopup")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        label = QLabel("公式源码")
        label.setObjectName("formulaSourceLabel")
        layout.addWidget(label)
        self._source_edit = QPlainTextEdit()
        self._source_edit.setObjectName("formulaSourceEdit")
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
        if self._emit_timer.isActive():
            self._emit_timer.stop()
        self.source_changed.emit(self.source_text())

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._source_edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            if key == Qt.Key.Key_Escape or (ctrl and key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}):
                self.hide()
                return True
            command = ""
            if key == Qt.Key.Key_Backtab or (key == Qt.Key.Key_Tab and shift):
                command = "previous"
            elif key == Qt.Key.Key_Tab:
                command = "next"
            elif ctrl and key == Qt.Key.Key_Up:
                command = "previous"
            elif ctrl and key == Qt.Key.Key_Down:
                command = "next"
            elif ctrl and key == Qt.Key.Key_S:
                command = "save"
            elif key == Qt.Key.Key_F5:
                command = "flag"
            elif key == Qt.Key.Key_F6:
                command = "skip"
            if command:
                self._emit_source_changed()
                if command != "save":
                    self.hide()
                self.command_requested.emit(command)
                return True
        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        if self._emit_timer.isActive():
            self._emit_timer.stop()
            self._emit_source_changed()
        super().hideEvent(event)
        self.closed.emit()


class _FormulaSourceInlinePanel(QFrame):
    """Full-width source editor used only for standalone formula rows."""

    source_changed = Signal(str)
    closed = Signal()
    command_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("formulaSourceInlinePanel")
        self.setFixedHeight(FORMULA_SOURCE_PANEL_H)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 6)
        layout.setSpacing(4)
        label = QLabel("公式源码")
        label.setObjectName("formulaSourceLabel")
        layout.addWidget(label)
        self._source_edit = QPlainTextEdit()
        self._source_edit.setObjectName("formulaSourceEdit")
        self._source_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._source_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._source_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._source_edit.setFixedHeight(48)
        self._source_edit.installEventFilter(self)
        layout.addWidget(self._source_edit)
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(160)
        self._emit_timer.timeout.connect(self._emit_source_changed)
        self._source_edit.textChanged.connect(self._emit_timer.start)
        self.hide()

    def open_for(self, source: str) -> None:
        self._source_edit.blockSignals(True)
        self._source_edit.setPlainText(source or "")
        self._source_edit.blockSignals(False)
        self.show()
        self._source_edit.setFocus()
        cursor = self._source_edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self._source_edit.setTextCursor(cursor)

    def _emit_source_changed(self) -> None:
        if self._emit_timer.isActive():
            self._emit_timer.stop()
        self.source_changed.emit(
            self._source_edit.toPlainText().replace("\r", "").replace("\n", " ").strip()
        )

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._source_edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            if key == Qt.Key.Key_Escape or (ctrl and key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}):
                self.hide()
                return True
            command = ""
            if key == Qt.Key.Key_Backtab or (key == Qt.Key.Key_Tab and shift):
                command = "previous"
            elif key == Qt.Key.Key_Tab:
                command = "next"
            elif ctrl and key == Qt.Key.Key_Up:
                command = "previous"
            elif ctrl and key == Qt.Key.Key_Down:
                command = "next"
            elif ctrl and key == Qt.Key.Key_S:
                command = "save"
            elif key == Qt.Key.Key_F5:
                command = "flag"
            elif key == Qt.Key.Key_F6:
                command = "skip"
            if command:
                self._emit_source_changed()
                if command != "save":
                    self.hide()
                self.command_requested.emit(command)
                return True
        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        if self._emit_timer.isActive():
            self._emit_timer.stop()
            self._emit_source_changed()
        super().hideEvent(event)
        self.closed.emit()


class _FormulaRenderLabel(QLabel):
    right_clicked = Signal(QPoint)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.mapToGlobal(event.position().toPoint()))
            event.accept()
            return
        super().mousePressEvent(event)


class _ProofLineImage(QLabel):
    """Clickable image projection with no OCR ownership."""

    clicked = Signal(QPoint)
    right_clicked = Signal(QPoint)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.mapToGlobal(event.position().toPoint()))
            event.accept()
            return
        super().mousePressEvent(event)


def _image_for_row(
    page: ProofPageView,
    bbox: tuple[int, int, int, int] | None,
    pixmap: QPixmap | None = None,
) -> QPixmap:
    if bbox is None:
        # 无行框几何时退回占位文案，绝不把整页当行图（公式行常见）
        return QPixmap()
    if pixmap is None:
        pixmap = QPixmap(page.image_path)
    if pixmap.isNull():
        return QPixmap()
    left, top, right, bottom = bbox
    x_scale = pixmap.width() / page.width if page.width > 0 else 1.0
    y_scale = pixmap.height() / page.height if page.height > 0 else 1.0
    rect = QRect(
        round(left * x_scale),
        round(top * y_scale),
        max(1, round((right - left) * x_scale)),
        max(1, round((bottom - top) * y_scale)),
    ).intersected(pixmap.rect())
    if rect.isEmpty():
        return QPixmap()
    return pixmap.copy(rect)


class _ProofPageCard(PageDirectoryRow):
    """Proof page card: shared Stitch-style thumbnail row + selection state."""

    def __init__(self, page: ProofPageView, parent=None) -> None:
        if not isinstance(page, ProofPageView):
            raise TypeError("proof page card requires ProofPageView")
        source = page.thumbnail_path or page.image_path
        super().__init__(
            page.page_number,
            page.source_path or page.image_path,
            source,
            badge_text=f"第 {page.page_number} 页",
            parent=parent,
        )
        self.page = page
        self._page_number = self.badge
        self._filename = self.filename
        self._source_pixmap = self.thumbnail
        self.setProperty("selected", False)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)


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


class _ProofRowWidget(QFrame):
    commit_requested = Signal()
    navigate_requested = Signal(int)
    history_requested = Signal(int)
    confirm_requested = Signal()
    flag_requested = Signal()
    skip_requested = Signal()
    save_all_requested = Signal()
    activated = Signal()

    def __init__(self, row: _ProofRow, parent=None, page_pixmap: QPixmap | None = None, large_image: bool = False) -> None:
        super().__init__(parent)
        self.row = row
        self._large_image = bool(large_image)
        self._page_pixmap = page_pixmap
        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(LINE_PAIR_MIN_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 8, 4)
        root.setSpacing(6)
        self._active_bar = QWidget()
        self._active_bar.setObjectName("proofRowActiveBar")
        self._active_bar.setFixedWidth(4)
        root.addWidget(self._active_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        header = QHBoxLayout()
        self._title = QLabel(f"第 {row.page.page_number} 页 · 第 {row.unit.order + 1} 行")
        self._title.setObjectName("muted")
        header.addWidget(self._title, 1)
        self._status = QLabel()
        header.addWidget(self._status)
        content_layout.addLayout(header)

        self._image = _ProofLineImage()
        self._image.setObjectName("proofLineImage")
        self._image.setFixedHeight(
            TABLE_ROW_IMAGE_H
            if self._large_image
            else (FORMULA_IMAGE_ROW_H if row.kind == "formula" else IMAGE_ROW_H)
        )
        self._image.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._image.setText("无可用行图像")
        self._line_bbox = _row_image_bbox(
            row.line, row.atom_placements, row.formula_number_line
        )
        self._line_crop = _image_for_row(row.page, self._line_bbox, page_pixmap)
        self._selected_char_index: int | None = None
        self._hover_char_index: int | None = None
        self._focus_depth = "far"
        self._displayed_pixmap_size = QSize()
        self._image.clicked.connect(self._on_image_clicked)
        self._image.right_clicked.connect(self._open_standalone_formula_editor)
        content_layout.addWidget(self._image)

        # display formula 三行呈现：crop → 渲染 → 源码编辑
        self._formula_render_area = QScrollArea()
        self._formula_render_area.setObjectName("proofFormulaRender")
        self._formula_render_area.setFixedHeight(FORMULA_RENDER_AREA_H)
        self._formula_render_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._formula_render_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._formula_render_area.setFrameShape(QFrame.Shape.NoFrame)
        self._formula_render_area.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._formula_render_area.customContextMenuRequested.connect(
            lambda pos: self._open_standalone_formula_editor(
                self._formula_render_area.mapToGlobal(pos)
            )
        )
        self._formula_render_label = _FormulaRenderLabel()
        self._formula_render_label.setObjectName("proofFormulaRenderLabel")
        self._formula_render_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._formula_render_area.setWidget(self._formula_render_label)
        self._formula_render_area.setVisible(False)
        content_layout.addWidget(self._formula_render_area)

        self._active = False
        self._external_conflict = False
        self._status_value = row.unit.status
        self._formula_source_range: tuple[int, int] | None = None
        self._formula_popup: _FormulaSourcePopup | None = None
        self._formula_source_panel: _FormulaSourceInlinePanel | None = None
        self._content_layout = content_layout
        self._formula_render_label.right_clicked.connect(
            self._open_standalone_formula_editor
        )
        self._formula_render_timer = QTimer(self)
        self._formula_render_timer.setSingleShot(True)
        self._formula_render_timer.setInterval(160)
        self._formula_render_timer.timeout.connect(self._refresh_formula_render)

        if row.kind == "formula":
            self.editor = _FormulaLineSourceEdit()
            self.editor.setObjectName("formulaSourceEdit")
            self.editor.setPlaceholderText("公式源码")
        else:
            self.editor = _SlotLineEditor()
            self.editor.setObjectName("proofSlotEditor")
            self.editor.setPlaceholderText("校对文本")
            self.editor.setCursorWidth(0)
        self.editor.set_formula_editing_enabled(row.kind == "formula")
        self.editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor.setFixedHeight(TEXT_EDITOR_DEFAULT_H)
        self.editor.document().setDocumentMargin(0)
        self.editor.confirm_requested.connect(self.confirm_requested)
        self.editor.prev_requested.connect(lambda: self.navigate_requested.emit(-1))
        self.editor.next_requested.connect(lambda: self.navigate_requested.emit(1))
        self.editor.flag_requested.connect(self.flag_requested)
        self.editor.skip_requested.connect(self.skip_requested)
        self.editor.revert_requested.connect(self._revert)
        self.editor.formula_source_requested.connect(self._open_formula_source_editor)
        self.editor.save_all_requested.connect(self.save_all_requested)
        self.editor.row_focus_requested.connect(self.activated)
        self.editor.textChanged.connect(self._refresh_editor_geometry)
        self.editor.textChanged.connect(self._refresh_extra_selections)
        self.editor.textChanged.connect(self._schedule_formula_render)
        self.editor.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.editor.cursorPositionChanged.connect(self._refresh_extra_selections)
        self.editor.selectionChanged.connect(self._refresh_extra_selections)
        self.editor.hover_char_changed.connect(self._on_hover_char_changed)
        content_layout.addWidget(self.editor)
        root.addWidget(content, 1)

        self.editor.setPlainText(row.unit.text)
        self.set_status(row.unit.status)
        self.set_active(False)
        self._refresh_formula_render()
        self._refresh_image()

    # ── status / conflict ──────────────────────────────────────

    def update_row(self, row: _ProofRow) -> None:
        """Swap the immutable row snapshot in place (structure unchanged).

        Used by the panel's incremental refresh path so a new workspace
        snapshot does not rebuild the widget tree (keeps focus, cursor and
        scroll position during a proofing session).
        """

        previous_bbox = self._line_bbox
        self.row = row
        self._status_value = row.unit.status
        self._line_bbox = _row_image_bbox(
            row.line, row.atom_placements, row.formula_number_line
        )
        if self._line_bbox != previous_bbox:
            self._line_crop = _image_for_row(
                row.page, self._line_bbox, self._page_pixmap
            )
            self._refresh_image()
        self._refresh_status()
        self._refresh_formula_render()
        self._refresh_editor_geometry()
        self._refresh_extra_selections()

    def displayed_status(self) -> str:
        return "conflict" if self._external_conflict else self._status_value

    def set_conflict(self, conflict: bool) -> None:
        self._external_conflict = bool(conflict)
        self.setProperty("conflict", self._external_conflict)
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh_status()

    def has_external_conflict(self) -> bool:
        return self._external_conflict

    def _chars_aligned(self) -> bool:
        entries = self.row.entries
        if not entries:
            return False
        return len(self.editor.toPlainText()) == len(entries)

    def _refresh_status(self) -> None:
        status = "conflict" if self._external_conflict else self._status_value
        color = QColor(STATUS_COLORS.get(status, STATUS_COLORS["unchecked"]))
        if self._external_conflict:
            icon_status = "conflict"
            tooltip = "冲突：外部变更与本地未提交编辑冲突；编辑到与最新文本一致即可解除"
        elif not self._chars_aligned():
            icon_status = "conflict"
            color = QColor(STATUS_COLORS["conflict"])
            tooltip = "图字未对齐：文本长度与 OCR 字符数不一致，图字对应已暂停"
        else:
            icon_status = status
            tooltip = STATUS_LABELS.get(status, status)
        self._status.setStyleSheet("")
        self._status.setPixmap(_tinted_status_icon(icon_status, color))
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._status.setToolTip(tooltip)

    def set_status(self, status: str) -> None:
        self._status_value = status
        self._refresh_status()

    def _revert(self) -> None:
        """Reset the editor to the OCR source text (saved later, on commit)."""

        original = self.row.line.ocr_text or self.row.unit.text
        self.editor.setPlainText(original)
        self.editor.setFocus()

    # ── verdict colouring ──────────────────────────────────────

    def _classify_char_verdict(self, index: int) -> _cv.CharVerdict | None:
        entries = self.row.entries
        if not (0 <= index < len(entries)):
            return None
        text = self.editor.toPlainText()
        if index >= len(text):
            return None
        entry = entries[index]
        # 只在等长时取同下标 OCR 字符；长度不一致退回 None，避免错位比对
        ocr_char = entry.ocr_char if self._chars_aligned() else None
        return _cv.classify_char(
            confidence=entry.confidence,
            text_char=text[index],
            ocr_char=ocr_char,
        )

    def _refresh_extra_selections(self) -> None:
        """Per-character verdict colours + current-slot highlight.

        Cursor width is 0; the weak cursor is expressed entirely through
        these selections.  ``user_modified`` only adds an underline — it
        never changes the verdict colour (no whitewashing).  Display-formula
        source rows skip verdict colouring: LaTeX source is not proof text.
        """

        from PySide6.QtWidgets import QTextEdit

        editor = self.editor
        doc_text = editor.toPlainText()
        sels: list = []

        if self.row.kind != "formula" and self._chars_aligned():
            for i in range(min(len(self.row.entries), len(doc_text))):
                verdict = self._classify_char_verdict(i)
                if verdict is None:
                    continue
                if verdict.color == _cv.COLOR_UNVERIFIED and not verdict.user_modified:
                    continue
                sel = QTextEdit.ExtraSelection()
                cur = QTextCursor(editor.document())
                cur.setPosition(i)
                cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                if verdict.color != _cv.COLOR_UNVERIFIED:
                    fmt.setForeground(QColor(verdict.color))
                # 错字额外给一个非常淡的红底；其他颜色不加底，避免与"当前字"蓝底冲突
                if verdict.color == _cv.COLOR_ERROR:
                    fmt.setBackground(QColor("#fdecec"))
                if verdict.user_modified:
                    fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
                    fmt.setUnderlineColor(QColor(verdict.color))
                sel.format = fmt
                sel.cursor = cur
                sels.append(sel)

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
            fmt.setBackground(QColor("#cfe2ff"))
            sel.format = fmt
            sel.cursor = cur
            sels.append(sel)

        editor.setExtraSelections(sels)
        self._refresh_status()

    # ── focus / activation ─────────────────────────────────────

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.setProperty("active", self._active)
        self._active_bar.setVisible(self._active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.editor.set_active_visual(self._active)
        self._refresh_extra_selections()

    def set_focus_depth(self, depth: str) -> None:
        if depth not in {"active", "near", "far"}:
            raise ValueError(f"unsupported proof row focus depth: {depth!r}")
        self._focus_depth = depth
        active = depth == "active"
        self.set_active(active)
        self.editor.setVisible(active and self.row.kind != "formula")
        self._formula_render_area.setVisible(active and self.row.kind == "formula")
        if not active and self._formula_source_panel is not None:
            self._formula_source_panel.hide()
        self._image.setFixedHeight(
            TABLE_ROW_IMAGE_H
            if (active and self._large_image)
            else (
                (FORMULA_IMAGE_ROW_H if self.row.kind == "formula" else IMAGE_ROW_H)
                if active
                else (NEAR_IMAGE_ROW_H if depth == "near" else FAR_IMAGE_ROW_H)
            )
        )
        extra = (
            FORMULA_RENDER_AREA_H - TEXT_EDITOR_DEFAULT_H
            + FORMULA_IMAGE_ROW_H - IMAGE_ROW_H
            if active and self.row.kind == "formula" and not self._large_image
            else 0
        )
        if self._formula_source_panel is not None and self._formula_source_panel.isVisible():
            extra += FORMULA_SOURCE_PANEL_H
        if active:
            if self._large_image:
                min_h, max_h = TABLE_LINE_PAIR_MIN_H, TABLE_LINE_PAIR_MAX_H
                extra = 0
            else:
                min_h, max_h = LINE_PAIR_MIN_H, LINE_PAIR_MAX_H
        elif depth == "near":
            min_h, max_h = NEAR_LINE_PAIR_MIN_H, NEAR_LINE_PAIR_MAX_H
        else:
            min_h, max_h = FAR_LINE_PAIR_MIN_H, FAR_LINE_PAIR_MAX_H
        self.setMinimumHeight(min_h + extra)
        self.setMaximumHeight(max_h + extra)
        self.setProperty("focusDepth", depth)
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh_image()

    # ── formula visual layer / source editing ─────────────────

    def _schedule_formula_render(self) -> None:
        if self.row.kind == "formula":
            self._formula_render_timer.start()

    def _refresh_formula_render(self) -> None:
        """Display-formula rows: middle row shows the rendered formula.

        The rendering is a pure preview rebuilt from the editor's source
        text; it is never stored as business state.
        """

        if self.row.kind != "formula":
            self._formula_render_area.setVisible(False)
            return
        self._formula_render_area.setVisible(self._focus_depth == "active")
        visual = _render_formula_visual(
            _formula_preview_source(
                self.editor.toPlainText(),
                self.row.formula_number_line.proof_text
                if self.row.formula_number_line is not None
                else "",
            ),
            target_height=FORMULA_RENDER_TARGET_H,
        )
        if visual is not None and visual.pixmap is not None:
            self._formula_render_label.setPixmap(visual.pixmap)
            self._formula_render_label.setText("")
        elif visual is not None and visual.text:
            self._formula_render_label.setPixmap(QPixmap())
            self._formula_render_label.setText(visual.text)
        else:
            self._formula_render_label.setPixmap(QPixmap())
            self._formula_render_label.setText("（无法渲染，请直接编辑下方源码）")
        self._formula_render_label.adjustSize()

    def _refresh_atom_visual_overlays(self) -> None:
        """Render inline formula atoms over their exact char spans.

        Placement comes only from ``ProofAtomView.char_span`` projected
        through the slot geometry; there is deliberately no regex or text
        search fallback.  When the text diverges from the snapshot length
        (e.g. mid-edit), overlays hide until the next snapshot.
        """

        overlays: list[_AtomVisualOverlay] = []
        text = self.editor.toPlainText()
        centers, widths = self.editor.slot_geometry()
        if (
            centers is not None
            and widths is not None
            and len(text) == len(self.row.line.proof_text)
        ):
            for placement in self.row.atom_placements:
                if _atom_kind(placement.atom) != "formula":
                    continue
                indices = placement.char_indices
                if not indices:
                    continue
                if any(
                    index >= len(text) or centers[index] is None
                    for index in indices
                ):
                    continue
                start, end = indices[0], indices[-1] + 1
                visual = _render_formula_visual(
                    text[start:end],
                    target_height=max(
                        8,
                        round(max(1, self.editor.height()) * FORMULA_VISUAL_HEIGHT_RATIO),
                    ),
                )
                if visual is None:
                    continue
                overlays.append(
                    _AtomVisualOverlay(
                        start=start,
                        end=end,
                        left=centers[indices[0]] - widths[indices[0]] / 2.0,
                        right=centers[indices[-1]] + widths[indices[-1]] / 2.0,
                        text=visual.text or "",
                        pixmap=visual.pixmap,
                        logical_size=visual.logical_size,
                        kind="formula",
                    )
                )
        self.editor.set_atom_visual_overlays(overlays)

    def _open_formula_source_editor(self, start: int, end: int, global_pos: QPoint) -> None:
        text = self.editor.toPlainText()
        if not text:
            return
        if end <= start:
            start, end = 0, len(text)
        self._formula_source_range = (start, end)
        if self.row.kind == "formula" and start == 0 and end == len(text):
            if self._formula_popup is not None:
                self._formula_popup.hide()
            if self._formula_source_panel is None:
                self._formula_source_panel = _FormulaSourceInlinePanel(self)
                self._formula_source_panel.source_changed.connect(
                    self._apply_formula_source_text
                )
                self._formula_source_panel.closed.connect(
                    self._on_formula_source_panel_closed
                )
                self._formula_source_panel.command_requested.connect(
                    self._handle_formula_source_command
                )
                self._content_layout.addWidget(self._formula_source_panel)
            self._formula_source_panel.open_for(text)
            self.set_focus_depth("active")
            return
        if self._formula_popup is None:
            self._formula_popup = _FormulaSourcePopup(self)
            self._formula_popup.source_changed.connect(self._apply_formula_source_text)
            self._formula_popup.closed.connect(self._on_formula_popup_closed)
            self._formula_popup.command_requested.connect(
                self._handle_formula_source_command
            )
        self._formula_popup.open_for(text[start:end], global_pos, editable=True)

    def _handle_formula_source_command(self, command: str) -> None:
        if command == "previous":
            self.navigate_requested.emit(-1)
        elif command == "next":
            self.navigate_requested.emit(1)
        elif command == "save":
            self.save_all_requested.emit()
        elif command == "flag":
            self.flag_requested.emit()
        elif command == "skip":
            self.skip_requested.emit()
        else:
            raise ValueError(f"unsupported formula source command: {command!r}")

    def _open_standalone_formula_editor(self, global_pos: QPoint) -> None:
        if self.row.kind != "formula":
            return
        self.activated.emit()
        self._open_formula_source_editor(0, len(self.editor.toPlainText()), global_pos)

    def _apply_formula_source_text(self, source: str) -> None:
        if self._formula_source_range is None:
            return
        start, end = self._formula_source_range
        current = self.editor.toPlainText()
        if current[start:end] == source:
            return
        self.editor.replace_text_range(start, end, source)
        self._formula_source_range = (start, start + len(source))
        self._refresh_status()

    def _on_formula_popup_closed(self) -> None:
        self._formula_source_range = None
        self._refresh_editor_geometry()
        self._refresh_status()

    def _on_formula_source_panel_closed(self) -> None:
        self._formula_source_range = None
        self.set_focus_depth(self._focus_depth)
        self._refresh_formula_render()
        self._refresh_status()

    # ── line image ─────────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_image()

    def _entry_char_box(self, char_index: int | None):
        if char_index is None or self._line_bbox is None:
            return None
        entry = next(
            (
                item
                for item in self.row.entries
                if item.char_index == char_index and item.bbox is not None
            ),
            None,
        )
        if entry is None or entry.bbox is None:
            return None
        return entry.bbox

    def _draw_char_box(self, painter: QPainter, source: QPixmap, bbox, color: QColor, width: int, fill: QColor | None) -> None:
        line_left, line_top = self._line_bbox[0], self._line_bbox[1]
        left, top, right, bottom = bbox
        line_width = max(1, self._line_bbox[2] - line_left)
        line_height = max(1, self._line_bbox[3] - line_top)
        painter.setPen(QPen(color, width))
        if fill is not None:
            painter.setBrush(fill)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            QRect(
                round((left - line_left) * source.width() / line_width),
                round((top - line_top) * source.height() / line_height),
                max(1, round((right - left) * source.width() / line_width)),
                max(1, round((bottom - top) * source.height() / line_height)),
            )
        )

    def _refresh_image(self) -> None:
        if self._line_crop.isNull():
            self._image.setPixmap(QPixmap())
            self._image.setText("无可用行图像")
            return
        target = self._image.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        source = QPixmap(self._line_crop)
        if self._line_bbox is not None and self._active:
            painter = QPainter(source)
            selected_box = self._entry_char_box(self._selected_char_index)
            if selected_box is not None:
                self._draw_char_box(
                    painter,
                    source,
                    selected_box,
                    QColor("#D45555"),
                    2,
                    QColor(212, 85, 85, 32),
                )
            hover_box = self._entry_char_box(self._hover_char_index)
            if hover_box is not None and hover_box != selected_box:
                self._draw_char_box(
                    painter,
                    source,
                    hover_box,
                    QColor(40, 167, 69),
                    1,
                    None,
                )
            painter.end()
        opacity = FOCUS_OPACITY[self._focus_depth]
        if opacity < 1.0:
            faded = QPixmap(source.size())
            faded.fill(Qt.GlobalColor.transparent)
            painter = QPainter(faded)
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, source)
            painter.end()
            source = faded
        row_image_h = (
            TABLE_ROW_IMAGE_H
            if (self._focus_depth == "active" and self._large_image)
            else (
                (FORMULA_IMAGE_ROW_H if self.row.kind == "formula" else IMAGE_ROW_H)
                if self._focus_depth == "active"
                else (NEAR_IMAGE_ROW_H if self._focus_depth == "near" else FAR_IMAGE_ROW_H)
            )
        )
        # 小裁剪区域最多放大 3 倍，避免个别小行框被拉成模糊大图
        row_image_h = min(row_image_h, max(1, source.height() * 3))
        scaled = source.scaledToHeight(
            row_image_h,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.width() > MAX_LINE_IMAGE_W:
            scaled = source.scaledToWidth(
                MAX_LINE_IMAGE_W,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._displayed_pixmap_size = scaled.size()
        self._image.setPixmap(scaled)
        self._image.setText("")
        self._refresh_editor_geometry()

    # ── editor geometry (letter-spacing contract + slot geometry) ──

    def _proof_char_spans(self) -> dict[int, tuple[float, float]]:
        """Per-character x spans in page coordinates.

        Each atom's bbox is distributed across its characters by estimated
        glyph width (CJK full width, Latin/digit ~0.56, half-width
        punctuation ~0.35, space 0.5) instead of equal division, so mixed
        CJK/digit/punctuation lines track the ink much better.  Formula and
        table atoms opt out: their rendered glyph positions do not follow
        the source text, so pretending an alignment would be dishonest.
        """

        spans: dict[int, tuple[float, float]] = {}
        proof_text = self.row.line.proof_text
        for placement in self.row.atom_placements:
            # display formula/table 行不使用槽位几何（源码编辑/表格视图）；
            # 但文本行内的 inline formula atom 必须参与投影，
            # 其 char_span 是覆盖层放置的唯一依据
            if self.row.kind != "text" and _atom_kind(placement.atom) is not None:
                continue
            indices = placement.char_indices
            if not indices:
                continue
            left, _top, right, _bottom = placement.atom.bbox
            weights = [
                max(0.05, _char_slot_weight(proof_text[index]))
                for index in indices
            ]
            total = sum(weights) or 1.0
            width = float(right - left)
            cursor = float(left)
            for index, weight in zip(indices, weights):
                span_w = width * weight / total
                spans[index] = (cursor, cursor + span_w)
                cursor += span_w
        return spans

    def _refresh_editor_geometry(self) -> None:
        """Project immutable atom geometry into the active text editor.

        Two coordinated projections:
        - the letter-spacing contract on the document (kept for the Qt text
          layout consumers and the existing geometry tests);
        - the painted slot geometry consumed by the slot editor's own
          paint/hit-test path.

        Both are visual-only: when an edit changes the number of characters,
        the OCR geometry no longer describes the text, so both projections
        are dropped until a new workspace snapshot arrives.  Display-formula
        rows use a plain source editor and skip geometry entirely.
        """

        if self.row.kind == "formula":
            self._refresh_atom_visual_overlays()
            return

        text = self.editor.toPlainText()
        displayed_width = self._displayed_pixmap_size.width()
        displayed_height = self._displayed_pixmap_size.height()
        font = QFont(self.editor.font())
        font.setFamilies(TEXT_FONT_FAMILIES)
        # 字号跟随行图高度，但不超过普通文本行档位（表格大图不放大字号）
        font_height = min(displayed_height, IMAGE_ROW_H)
        font.setPixelSize(max(18, round(font_height * 0.80)) if font_height else 20)
        self.editor.setFont(font)

        document = self.editor.document()
        selection = self.editor.textCursor()
        saved_anchor = selection.anchor()
        saved_position = selection.position()
        all_text = QTextCursor(document)
        all_text.select(QTextCursor.SelectionType.Document)
        base_format = QTextCharFormat()
        base_format.setFont(font)
        base_format.setFontLetterSpacingType(QFont.SpacingType.AbsoluteSpacing)
        base_format.setFontLetterSpacing(0.0)

        self.editor.blockSignals(True)
        all_text.setCharFormat(base_format)
        block_cursor = QTextCursor(document)
        block_format = QTextBlockFormat(block_cursor.blockFormat())
        block_format.setLeftMargin(0.0)
        block_cursor.setBlockFormat(block_format)

        spans = self._proof_char_spans()
        geometry_valid = (
            bool(text)
            and len(text) == len(self.row.line.proof_text)
            and self._line_bbox is not None
            and displayed_width > 0
            and len(spans) == len(text)
        )
        slot_centers: list[float | None] | None = None
        slot_widths: list[float] | None = None
        if geometry_valid and self._line_bbox is not None:
            line_left, _top, line_right, _bottom = self._line_bbox
            line_width = max(1, line_right - line_left)
            scale = displayed_width / line_width
            desired = {
                index: (span[0] - line_left) * scale
                for index, span in spans.items()
            }
            desired_widths = {
                index: max(TEXT_SLOT_CLIPPED_MIN_W, (span[1] - span[0]) * scale)
                for index, span in spans.items()
            }
            first_index = min(desired)
            block_format.setLeftMargin(max(0.0, desired[first_index]))
            block_cursor.setBlockFormat(block_format)
            metrics = QFontMetricsF(font)
            for index in range(len(text) - 1):
                current_x = desired.get(index)
                next_x = desired.get(index + 1)
                if current_x is None or next_x is None:
                    continue
                spacing = next_x - current_x - metrics.horizontalAdvance(text[index])
                char_cursor = QTextCursor(document)
                char_cursor.setPosition(index)
                char_cursor.setPosition(index + 1, QTextCursor.MoveMode.KeepAnchor)
                char_format = QTextCharFormat(base_format)
                char_format.setFontLetterSpacing(spacing)
                char_cursor.setCharFormat(char_format)
            slot_centers = [
                desired[index] + desired_widths[index] / 2.0
                for index in range(len(text))
            ]
            slot_widths = [desired_widths[index] for index in range(len(text))]

        selection.setPosition(min(saved_anchor, len(text)))
        selection.setPosition(
            min(saved_position, len(text)),
            QTextCursor.MoveMode.KeepAnchor,
        )
        self.editor.setTextCursor(selection)
        self.editor.blockSignals(False)
        if slot_centers is not None:
            self.editor.set_slot_geometry(slot_centers, slot_widths)
        else:
            self.editor.set_slot_geometry(None, None)
        self._refresh_atom_visual_overlays()

    # ── char lookup (cursor ↔ image) ───────────────────────────

    def _on_cursor_position_changed(self) -> None:
        if not self.row.entries:
            return
        cursor = self.editor.textCursor()
        position = cursor.selectionStart() if cursor.hasSelection() else cursor.position()
        if position >= len(self.editor.toPlainText()) and position > 0:
            position -= 1
        available = {entry.char_index for entry in self.row.entries if entry.bbox is not None}
        self._selected_char_index = position if position in available else None
        self._refresh_image()

    def _on_hover_char_changed(self, index: int) -> None:
        available = {entry.char_index for entry in self.row.entries if entry.bbox is not None}
        hover = index if index in available else None
        if hover == self._hover_char_index:
            return
        self._hover_char_index = hover
        self._refresh_image()

    def _on_image_clicked(self, point: QPoint) -> None:
        self.activated.emit()
        if self._line_bbox is None or self._displayed_pixmap_size.isEmpty():
            self.editor.setFocus()
            return
        y_offset = max(0, (self._image.height() - self._displayed_pixmap_size.height()) // 2)
        if not (
            0 <= point.x() < self._displayed_pixmap_size.width()
            and y_offset <= point.y() < y_offset + self._displayed_pixmap_size.height()
        ):
            return
        source_x = point.x() * self._line_crop.width() / self._displayed_pixmap_size.width()
        source_y = (point.y() - y_offset) * self._line_crop.height() / self._displayed_pixmap_size.height()
        line_left, line_top, line_right, line_bottom = self._line_bbox
        page_x = line_left + source_x * (line_right - line_left) / self._line_crop.width()
        page_y = line_top + source_y * (line_bottom - line_top) / self._line_crop.height()
        entries = tuple(entry for entry in self.row.entries if entry.bbox is not None)
        if not entries:
            self.editor.setFocus()
            return
        entry = min(
            entries,
            key=lambda item: (
                0
                if item.bbox is not None
                and item.bbox[0] <= page_x <= item.bbox[2]
                and item.bbox[1] <= page_y <= item.bbox[3]
                else 1,
                abs(((item.bbox[0] + item.bbox[2]) / 2) - page_x) if item.bbox else float("inf"),
            ),
        )
        cursor = self.editor.textCursor()
        cursor.setPosition(min(entry.char_index, len(self.editor.toPlainText())))
        cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 1)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()


class HProofPanel(QWidget):
    """Horizontal proof editor that consumes one immutable workspace snapshot."""

    proof_edit_requested = Signal(object)

    def __init__(
        self,
        workspace: ProofWorkspaceView | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._workspace: ProofWorkspaceView | None = None
        self._rows: tuple[_ProofRow, ...] = ()
        self._visible_rows: tuple[_ProofRow, ...] = ()
        self._row_widgets: dict[tuple[str, str], _ProofRowWidget] = {}
        self._page_cards: dict[str, _ProofPageCard] = {}
        self._dirty_text: dict[tuple[str, str], str] = {}
        self._dirty_origin: dict[tuple[str, str], str] = {}
        self._conflict_keys: set[tuple[str, str]] = set()
        self._page_pixmaps: dict[str, QPixmap] = {}
        self._selected_page_uid: str | None = None
        self._active_key: tuple[str, str] | None = None
        self._last_stats: dict[str, object] = {}
        self._stats_card: QFrame | None = None
        self._render_mode = ""
        self._build_ui()
        if workspace is not None:
            self.set_workspace(workspace)

    @property
    def workspace(self) -> ProofWorkspaceView | None:
        return self._workspace

    def _build_ui(self) -> None:
        self.setObjectName("proofRoot")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("hproofSplitter")
        self._splitter.setChildrenCollapsible(False)

        left = QFrame()
        left.setObjectName("proofLeftPane")
        left.setMinimumWidth(210)
        left.setMaximumWidth(270)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        directory_title = QLabel("页面")
        directory_title.setObjectName("sectionTitle")
        left_layout.addWidget(directory_title)
        self._page_directory = QListWidget()
        self._page_directory.setObjectName("pageDirectoryList")
        self._page_directory.setSpacing(10)
        self._page_directory.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._page_directory.currentItemChanged.connect(self._on_page_directory_selected)
        left_layout.addWidget(self._page_directory, 1)
        self._splitter.addWidget(left)

        center = QWidget()
        center.setObjectName("proofCenterPane")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(0)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("proofScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._mode_banner = QLabel("")
        self._mode_banner.setObjectName("hproofModeBanner")
        self._mode_banner.setVisible(False)
        center_layout.addWidget(self._mode_banner)
        self._rows_root = QWidget()
        self._rows_root.setObjectName("proofLineList")
        self._rows_layout = QVBoxLayout(self._rows_root)
        self._rows_layout.setContentsMargins(6, 4, 6, 6)
        self._rows_layout.setSpacing(3)
        self._empty_label = QLabel("完成 OCR 识别后，横校数据将在此展示")
        self._empty_label.setObjectName("proofEmpty")
        self._empty_label.setMinimumHeight(120)
        self._scroll.setWidget(self._rows_root)
        self._center_stack = QStackedWidget()
        self._center_stack.addWidget(self._scroll)
        self._table_scroll = QScrollArea()
        self._table_scroll.setObjectName("proofScroll")
        self._table_scroll.setWidgetResizable(True)
        self._table_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._table_root = QWidget()
        self._table_root.setObjectName("proofLineList")
        self._table_layout = QVBoxLayout(self._table_root)
        self._table_layout.setContentsMargins(6, 4, 6, 6)
        self._table_layout.setSpacing(6)
        self._table_empty_label = QLabel("当前页面没有表格行")
        self._table_empty_label.setObjectName("proofEmpty")
        self._table_empty_label.setMinimumHeight(120)
        self._table_scroll.setWidget(self._table_root)
        self._center_stack.addWidget(self._table_scroll)
        center_layout.addWidget(self._center_stack, 1)
        self._splitter.addWidget(center)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([236, 1000])
        root.addWidget(self._splitter, 1)

        self._status_bar = QWidget()
        self._status_bar.setObjectName("proofStatusBar")
        self._status_bar.setFixedHeight(44)
        toolbar = QHBoxLayout(self._status_bar)
        toolbar.setContentsMargins(12, 3, 12, 3)
        toolbar.setSpacing(8)
        self._progress_ring = _ProofProgressRing()
        self._progress_ring.setToolTip("校对进度统计")
        self._progress_ring.clicked.connect(self._toggle_stats_card)
        toolbar.addWidget(self._progress_ring)
        self._btn_prev = QPushButton("上一行")
        self._btn_next = QPushButton("下一行")
        self._btn_save = QPushButton("保存")
        self._btn_flag = QPushButton("标记")
        self._btn_skip = QPushButton("跳过")
        self._btn_refresh = QPushButton("刷新")
        self._btn_debug_formula = QPushButton("公式")
        self._btn_debug_table = QPushButton("表格")
        self._btn_debug_formula.setCheckable(True)
        self._btn_debug_table.setCheckable(True)
        self._btn_debug_formula.setToolTip("调试：只显示公式路由命中的校验行")
        self._btn_debug_table.setToolTip("调试：只显示表格路由命中的校验行")
        self._btn_debug_formula.clicked.connect(self._on_debug_filter_changed)
        self._btn_debug_table.clicked.connect(self._on_debug_filter_changed)
        self._btn_prev.setObjectName("ghostBtn")
        self._btn_next.setObjectName("ghostBtn")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_flag.setObjectName("ghostBtn")
        self._btn_skip.setObjectName("ghostBtn")
        self._btn_refresh.setObjectName("ghostBtn")
        self._btn_debug_formula.setObjectName("ghostBtn")
        self._btn_debug_table.setObjectName("ghostBtn")
        self._btn_save.setToolTip("保存全部已编辑行 (Ctrl+S)")
        self._btn_flag.setToolTip("标记/取消疑点 (F5)")
        self._btn_skip.setToolTip("跳过本行 (F6)")
        self._btn_prev.setToolTip("上一行 (Shift+Tab)")
        self._btn_next.setToolTip("下一行 (Tab)")
        self._btn_refresh.setToolTip("重新渲染当前快照")
        self._btn_prev.clicked.connect(lambda: self._focus_row(-1))
        self._btn_next.clicked.connect(lambda: self._focus_row(1))
        self._btn_save.clicked.connect(self.save)
        self._btn_flag.clicked.connect(self._flag_active_row)
        self._btn_skip.clicked.connect(self._skip_active_row)
        self._btn_refresh.clicked.connect(self.refresh_view)
        for button in (
            self._btn_prev,
            self._btn_next,
            self._btn_save,
            self._btn_flag,
            self._btn_skip,
            self._btn_refresh,
            self._btn_debug_formula,
            self._btn_debug_table,
        ):
            # 按钮不抢键盘焦点，Tab/Enter 始终属于行编辑器
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        toolbar.addWidget(self._btn_prev)
        toolbar.addWidget(self._btn_next)
        toolbar.addWidget(self._btn_save)
        toolbar.addWidget(self._btn_flag)
        toolbar.addWidget(self._btn_skip)
        toolbar.addWidget(self._btn_refresh)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._btn_debug_formula)
        toolbar.addWidget(self._btn_debug_table)
        self._status = QLabel("暂无可校对内容")
        self._status.setObjectName("muted")
        toolbar.addSpacing(8)
        toolbar.addWidget(self._status)
        toolbar.addStretch(1)
        self._total_label = QLabel("")
        self._total_label.setObjectName("muted")
        toolbar.addWidget(self._total_label)
        self._diff_label = QLabel("")
        self._diff_label.setObjectName("muted")
        toolbar.addWidget(self._diff_label)
        self._scope = QLabel("全部页面")
        self._scope.setObjectName("proofStatusStrong")
        toolbar.addWidget(self._scope)
        root.addWidget(self._status_bar)

    # ── workspace intake ───────────────────────────────────────

    def set_workspace(self, workspace: ProofWorkspaceView | None) -> None:
        """Replace the displayed immutable snapshot.

        Dirty editors are snapshotted before the swap and restored onto the
        rebuilt rows; a row whose underlying text changed while it was dirty
        becomes a conflict instead of losing the user's uncommitted edit.
        """

        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("HProofPanel requires ProofWorkspaceView or None")
        previous_rows = {row.key: row for row in self._rows}
        for key in tuple(self._dirty_text):
            if key in previous_rows:
                self._dirty_origin.setdefault(key, previous_rows[key].unit.text)
        selected = self._selected_page_uid
        self._workspace = workspace
        self._page_pixmaps.clear()
        pages = self._pages()
        page_uids = {page.page_uid for page in pages}
        self._selected_page_uid = selected if selected in page_uids else (pages[0].page_uid if pages else None)
        self._populate_page_selector()
        self._build_rows()
        self._reconcile_dirty_rows()
        self._render_rows()

    def clear_workspace(self) -> None:
        self.set_workspace(None)

    def _pages(self) -> tuple[ProofPageView, ...]:
        if self._workspace is None:
            return ()
        return tuple(sorted(self._workspace.pages, key=lambda item: (item.page_number, item.page_uid)))

    def _page_pixmap(self, page: ProofPageView) -> QPixmap:
        pixmap = self._page_pixmaps.get(page.page_uid)
        if pixmap is None:
            pixmap = QPixmap(page.image_path)
            self._page_pixmaps[page.page_uid] = pixmap
        return pixmap

    def _populate_page_selector(self) -> None:
        self._page_directory.blockSignals(True)
        self._page_directory.clear()
        self._page_cards.clear()
        pages = self._pages()
        for page in pages:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, page.page_uid)
            item.setSizeHint(QSize(0, PAGE_ROW_H))
            self._page_directory.addItem(item)
            card = _ProofPageCard(page)
            self._page_directory.setItemWidget(item, card)
            self._page_cards[page.page_uid] = card
        current = next(
            (index for index, page in enumerate(pages) if page.page_uid == self._selected_page_uid),
            -1,
        )
        if current >= 0:
            self._page_directory.setCurrentRow(current)
        self._page_directory.blockSignals(False)
        for page_uid, card in self._page_cards.items():
            card.set_selected(page_uid == self._selected_page_uid)
        self._scope.setText(
            f"第 {pages[current].page_number} 页" if current >= 0 else "全部页面"
        )

    def _on_page_directory_selected(self, current: QListWidgetItem | None, _previous) -> None:
        self._selected_page_uid = (
            current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        )
        page = next(
            (item for item in self._pages() if item.page_uid == self._selected_page_uid),
            None,
        )
        for page_uid, card in self._page_cards.items():
            card.set_selected(page_uid == self._selected_page_uid)
        self._scope.setText(f"第 {page.page_number} 页" if page is not None else "全部页面")
        self._render_rows()

    def _build_rows(self) -> None:
        if self._workspace is None:
            self._rows = ()
            return
        pages = {page.page_uid: page for page in self._workspace.pages}
        lines_by_key = {
            (line.proof_uid, line.text_unit_uid): line
            for line in self._workspace.lines
        }
        number_uid_by_formula = {
            (link.proof_uid, link.formula_text_unit_uid): link.number_text_unit_uid
            for link in self._workspace.formula_number_links
        }
        linked_number_keys = {
            (link.proof_uid, link.number_text_unit_uid)
            for link in self._workspace.formula_number_links
        }
        rows: list[_ProofRow] = []
        for state in self._workspace.proof_states:
            page = pages.get(state.page_uid)
            if page is None:
                raise ValueError(f"proof state {state.proof_uid!r} has no page view")
            units = {unit.text_unit_uid: unit for unit in state.text_units}
            for line in state.lines:
                if (state.proof_uid, line.text_unit_uid) in linked_number_keys:
                    continue
                unit = units.get(line.text_unit_uid)
                if unit is None:
                    raise ValueError(f"proof line {line.text_unit_uid!r} has no text unit view")
                atom_placements = _atom_placements(line)
                kind = _row_kind(line, atom_placements)
                number_uid = number_uid_by_formula.get(
                    (state.proof_uid, line.text_unit_uid)
                )
                formula_number_line = (
                    lines_by_key.get((state.proof_uid, number_uid))
                    if number_uid is not None
                    else None
                )
                if number_uid is not None and formula_number_line is None:
                    raise ValueError(
                        f"formula number link references missing text unit {number_uid!r}"
                    )
                rows.append(
                    _ProofRow(
                        key=(state.proof_uid, unit.text_unit_uid),
                        page=page,
                        state=state,
                        unit=unit,
                        line=line,
                        entries=build_char_views(line, page),
                        atom_placements=atom_placements,
                        kind=kind,
                        preview_source=_row_preview_source(line, atom_placements, kind),
                        formula_number_line=formula_number_line,
                    )
                )
        self._rows = tuple(
            sorted(rows, key=lambda row: (row.page.page_number, row.state.proof_uid, row.unit.order, row.unit.text_unit_uid))
        )

    def _reconcile_dirty_rows(self) -> None:
        rows = {row.key: row for row in self._rows}
        for key, dirty in tuple(self._dirty_text.items()):
            row = rows.get(key)
            if row is None:
                self._dirty_text.pop(key, None)
                self._dirty_origin.pop(key, None)
                self._conflict_keys.discard(key)
                continue
            if dirty == row.unit.text:
                self._dirty_text.pop(key, None)
                self._dirty_origin.pop(key, None)
                self._conflict_keys.discard(key)
                continue
            origin = self._dirty_origin.get(key, row.unit.text)
            if origin != row.unit.text:
                self._conflict_keys.add(key)
            else:
                self._conflict_keys.discard(key)

    def refresh_view(self) -> None:
        """Repaint the current snapshot without querying or mutating state."""

        self._build_rows()
        self._reconcile_dirty_rows()
        self._render_rows()

    def _debug_kinds(self) -> frozenset[str]:
        kinds: set[str] = set()
        if self._btn_debug_formula.isChecked():
            kinds.add("formula")
        if self._btn_debug_table.isChecked():
            kinds.add("table")
        return frozenset(kinds)

    def _on_debug_filter_changed(self) -> None:
        # 切换视图前先把未提交编辑落盘，避免调试过滤吞掉当前行编辑
        self._save_all()
        self._render_rows()

    def _update_mode_banner(self, kinds: frozenset[str]) -> None:
        if not kinds:
            self._mode_banner.setVisible(False)
            return
        if "table" in kinds:
            self._mode_banner.setText("表格校对视图 · 表格行已分流到大图校对模式")
        else:
            self._mode_banner.setText("公式调试视图 · 当前只显示路由命中的校验行")
        self._mode_banner.setVisible(True)

    def _create_row_widget(self, row: _ProofRow, *, large_image: bool = False) -> _ProofRowWidget:
        parent = self._table_root if large_image else self._rows_root
        widget = _ProofRowWidget(
            row,
            parent,
            page_pixmap=self._page_pixmap(row.page),
            large_image=large_image,
        )
        widget.editor.textChanged.connect(
            lambda row=row, widget=widget: self._on_text_changed(row, widget)
        )
        widget.confirm_requested.connect(
            lambda row=row, widget=widget: self._on_confirm_requested(row, widget)
        )
        widget.navigate_requested.connect(
            lambda direction, row=row: self._navigate_from(row, direction)
        )
        widget.history_requested.connect(
            lambda direction, row=row: self._apply_history(row, direction)
        )
        widget.flag_requested.connect(
            lambda row=row, widget=widget: self._toggle_flag(row, widget)
        )
        widget.skip_requested.connect(
            lambda row=row, widget=widget: self._skip_row(row, widget)
        )
        widget.save_all_requested.connect(self._save_all)
        widget.activated.connect(lambda row=row: self._activate_row(row.key))
        return widget

    def _render_table_view(self) -> None:
        """表格分流视图：表格行离开普通行列表，进入大图校对模式。"""

        self._center_stack.setCurrentWidget(self._table_scroll)
        self._visible_rows = tuple(
            row
            for row in self._rows
            if (self._selected_page_uid is None or row.page.page_uid == self._selected_page_uid)
            and row.kind == "table"
        )
        self._row_widgets.clear()
        while self._table_layout.count():
            item = self._table_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._table_empty_label:
                widget.deleteLater()
        self._table_layout.addWidget(self._table_empty_label)
        for row in self._visible_rows:
            widget = self._create_row_widget(row, large_image=True)
            self._table_layout.addWidget(widget)
            self._row_widgets[row.key] = widget
        self._table_layout.addStretch(1)
        self._table_empty_label.setVisible(not self._visible_rows)
        self._finish_render_rows()

    def _render_rows(self) -> None:
        kinds = self._debug_kinds()
        self._update_mode_banner(kinds)
        if "table" in kinds:
            self._render_mode = "table"
            self._render_table_view()
            return
        self._center_stack.setCurrentWidget(self._scroll)
        self._visible_rows = tuple(
            row
            for row in self._rows
            if (self._selected_page_uid is None or row.page.page_uid == self._selected_page_uid)
            and (not kinds or row.kind in kinds)
        )
        # 结构未变时原位更新：避免每次命令后整树重建（丢焦点/滚动位置）
        visible_keys = [row.key for row in self._visible_rows]
        same_widget_kinds = all(
            self._row_widgets[row.key].row.kind == row.kind
            for row in self._visible_rows
            if row.key in self._row_widgets
        )
        if (
            self._render_mode == "list"
            and visible_keys == list(self._row_widgets.keys())
            and same_widget_kinds
        ):
            for row in self._visible_rows:
                widget = self._row_widgets[row.key]
                if widget.row != row:
                    widget.update_row(row)
                if row.key not in self._dirty_text and widget.editor.toPlainText() != row.unit.text:
                    widget.editor.blockSignals(True)
                    widget.editor.setPlainText(row.unit.text)
                    widget.editor.blockSignals(False)
            self._finish_render_rows()
            return
        self._render_mode = "list"
        self._row_widgets.clear()
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._empty_label:
                widget.deleteLater()
        self._rows_layout.addWidget(self._empty_label)
        for row in self._visible_rows:
            widget = self._create_row_widget(row)
            self._rows_layout.addWidget(widget)
            self._row_widgets[row.key] = widget
        self._rows_layout.addStretch(1)
        if not self._visible_rows:
            self._empty_label.setText(
                "当前页面没有可显示的调试行"
                if kinds
                else "完成 OCR 识别后，横校数据将在此展示"
            )
        self._empty_label.setVisible(not self._visible_rows)
        self._finish_render_rows()

    def _finish_render_rows(self) -> None:
        # 把未提交的脏编辑恢复到重建后的行上（跨快照/跨视图存活）
        for key, widget in self._row_widgets.items():
            dirty = self._dirty_text.get(key)
            if dirty is not None and dirty != widget.editor.toPlainText():
                widget.editor.blockSignals(True)
                widget.editor.setPlainText(dirty)
                widget.editor.blockSignals(False)
                widget._refresh_editor_geometry()
                widget._refresh_extra_selections()
                widget._refresh_formula_render()
            widget.set_conflict(key in self._conflict_keys)
        if self._visible_rows:
            active_key = self._active_key if self._active_key in self._row_widgets else self._visible_rows[0].key
            self._activate_row(active_key)
        else:
            self._active_key = None
        self._status.setText(
            "暂无可校对内容"
            if not self._visible_rows
            else f"{len(self._visible_rows)} 行 · {sum(len(row.unit.text) for row in self._visible_rows)} 字符"
        )
        self._update_stats()

    def _activate_row(self, key: tuple[str, str] | None) -> None:
        self._active_key = key
        index = next(
            (index for index, row in enumerate(self._visible_rows) if row.key == key),
            -1,
        )
        for row_index, row in enumerate(self._visible_rows):
            widget = self._row_widgets.get(row.key)
            if widget is None:
                continue
            distance = abs(row_index - index) if index >= 0 else 99
            widget.set_focus_depth("active" if distance == 0 else "near" if distance == 1 else "far")

    # ── dirty tracking / conflict ──────────────────────────────

    def _on_text_changed(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        # 原位更新路径下 widget 会被复用，连接闭包里的 row 可能是旧快照，
        # 一律以 widget 当前持有的 row 为准（CAS 事实/文本都最新）
        row = widget.row
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
            self._dirty_origin.pop(row.key, None)
            if row.key in self._conflict_keys:
                self._conflict_keys.discard(row.key)
                widget.set_conflict(False)
        else:
            if row.key not in self._dirty_text:
                self._dirty_origin[row.key] = row.unit.text
            self._dirty_text[row.key] = text
        self._activate_row(row.key)

    # ── commands ───────────────────────────────────────────────

    def _emit(self, command: ProofEditCommand) -> None:
        self.proof_edit_requested.emit(command)

    def _replace_command(
        self,
        row: _ProofRow,
        text: str,
        *,
        status: str = "modified",
    ) -> ProofEditCommand:
        return ProofEditCommand(
            proof_uid=row.state.proof_uid,
            op="replace_text",
            expected_revision=row.state.revision,
            expected_fingerprint=row.state.fingerprint,
            text_unit_uid=row.unit.text_unit_uid,
            text=text,
            status=status,
            expected_unit_revision=row.unit.revision,
            expected_unit_fingerprint=row.unit.fingerprint,
        )

    def _status_command(self, row: _ProofRow, status: str) -> ProofEditCommand:
        return ProofEditCommand(
            proof_uid=row.state.proof_uid,
            op="set_status",
            expected_revision=row.state.revision,
            expected_fingerprint=row.state.fingerprint,
            text_unit_uid=row.unit.text_unit_uid,
            status=status,
            expected_unit_revision=row.unit.revision,
            expected_unit_fingerprint=row.unit.fingerprint,
        )

    def _apply_history(self, row: _ProofRow, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("history direction must be -1 or 1")
        self._emit(
            ProofEditCommand(
                proof_uid=row.state.proof_uid,
                op="undo" if direction < 0 else "redo",
                expected_revision=row.state.revision,
                expected_fingerprint=row.state.fingerprint,
            )
        )

    def _commit_row(self, row: _ProofRow, widget: _ProofRowWidget) -> bool:
        row = widget.row
        if row.key in self._conflict_keys:
            widget._refresh_status()
            return False
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
            self._dirty_origin.pop(row.key, None)
            return False
        self._emit(self._replace_command(row, text))
        self._dirty_text.pop(row.key, None)
        self._dirty_origin.pop(row.key, None)
        widget.set_status("modified")
        self._update_stats()
        return True

    def _save_all(self) -> bool:
        grouped: dict[str, list[tuple[_ProofRow, str]]] = defaultdict(list)
        rows = {row.key: row for row in self._rows}
        skipped = 0
        for key, text in tuple(self._dirty_text.items()):
            if key in self._conflict_keys:
                skipped += 1
                continue
            row = rows.get(key)
            if row is not None and text != row.unit.text:
                grouped[row.state.proof_uid].append((row, text))
        emitted = False
        for proof_uid, values in grouped.items():
            state = values[0][0].state
            replacements = tuple(
                (row.unit.text_unit_uid, text)
                for row, text in sorted(values, key=lambda item: (item[0].unit.order, item[0].unit.text_unit_uid))
            )
            self._emit(
                ProofEditCommand(
                    proof_uid=proof_uid,
                    op="replace_many",
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    replacements=replacements,
                )
            )
            emitted = True
        if emitted:
            for _proof_uid, values in grouped.items():
                for row, _text in values:
                    self._dirty_text.pop(row.key, None)
                    self._dirty_origin.pop(row.key, None)
        if skipped:
            self._status.setText(f"{skipped} 行存在冲突，未保存")
        return emitted

    def save(self) -> bool:
        return self._save_all()

    def _find_row(self, key: tuple[str, str]) -> _ProofRow | None:
        return next((row for row in self._rows if row.key == key), None)

    def _confirm_row(self, row: _ProofRow, widget: _ProofRowWidget) -> bool:
        row = widget.row
        if row.key in self._conflict_keys:
            widget._refresh_status()
            return False
        text = widget.editor.toPlainText()
        if text != row.unit.text:
            self._emit(self._replace_command(row, text, status="checked"))
            self._dirty_text.pop(row.key, None)
            self._dirty_origin.pop(row.key, None)
        else:
            self._emit(self._status_command(row, "checked"))
        widget.set_status("checked")
        self._update_stats()
        return True

    def _on_confirm_requested(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        if self._confirm_row(row, widget):
            self._focus_row(1, current_key=row.key)

    def _toggle_flag(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        row = widget.row
        if row.key in self._conflict_keys:
            widget._refresh_status()
            return
        current = widget.displayed_status()
        next_status = "unchecked" if current in _FLAGGED_STATUSES else "flagged"
        self._emit(self._status_command(row, next_status))
        widget.set_status(next_status)
        self._update_stats()

    def _skip_row(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        row = widget.row
        self._commit_row(row, widget)
        self._focus_row(1, current_key=row.key)

    def _active_row_and_widget(self) -> tuple[_ProofRow, _ProofRowWidget] | None:
        if self._active_key is None:
            return None
        row = self._find_row(self._active_key)
        widget = self._row_widgets.get(self._active_key)
        if row is None or widget is None:
            return None
        return row, widget

    def _flag_active_row(self) -> None:
        target = self._active_row_and_widget()
        if target is not None:
            self._toggle_flag(*target)

    def _skip_active_row(self) -> None:
        target = self._active_row_and_widget()
        if target is not None:
            self._skip_row(*target)

    def _navigate_from(self, row: _ProofRow, delta: int) -> None:
        widget = self._row_widgets.get(row.key)
        if widget is not None:
            self._commit_row(row, widget)
        self._focus_row(delta, current_key=row.key)

    def _focus_row(
        self,
        delta: int,
        *,
        current_key: tuple[str, str] | None = None,
    ) -> None:
        if not self._visible_rows:
            return
        key = current_key or self._active_key or self._visible_rows[0].key
        current = next(
            (index for index, row in enumerate(self._visible_rows) if row.key == key),
            0,
        )
        target = max(0, min(len(self._visible_rows) - 1, current + delta))
        target_key = self._visible_rows[target].key
        self._activate_row(target_key)
        widget = self._row_widgets.get(target_key)
        if widget is not None:
            widget.editor.setFocus()
            QTimer.singleShot(30, lambda w=widget: self._scroll.ensureWidgetVisible(w, 40, 60))

    # ── statistics / progress ──────────────────────────────────

    def _row_display_status(self, row: _ProofRow) -> str:
        widget = self._row_widgets.get(row.key)
        return widget.displayed_status() if widget is not None else row.unit.status

    def _update_stats(self) -> None:
        rows = self._visible_rows
        total = len(rows)
        confirmed = sum(1 for row in rows if self._row_display_status(row) in _CONFIRMED_STATUSES)
        modified = sum(1 for row in rows if self._row_display_status(row) == "modified")
        flagged = sum(1 for row in rows if self._row_display_status(row) in _FLAGGED_STATUSES)
        pending = total - confirmed - modified - flagged
        handled = confirmed + modified
        total_chars = sum(len(row.unit.text) for row in rows)
        diff = modified + flagged
        pct = diff / max(1, total) * 100
        active_index = next(
            (index for index, row in enumerate(rows) if row.key == self._active_key),
            -1,
        )
        self._last_stats = {
            "当前行": f"{active_index + 1} / {total}" if active_index >= 0 else "—",
            "已处理": handled,
            "待确认": pending,
            "已确认": confirmed,
            "已修改": modified,
            "疑点": flagged,
            "总字数": total_chars,
            "差异": f"{diff} ({pct:.0f}%)",
        }
        self._progress_ring.set_counts(handled, total)
        self._total_label.setText(f"总字数 {total_chars}" if total else "")
        self._diff_label.setText(f"差异 {diff} ({pct:.0f}%)" if total else "")

    def _toggle_stats_card(self) -> None:
        if self._stats_card is not None and self._stats_card.isVisible():
            self._hide_stats_card()
            return
        self._show_progress_popup()

    def _show_progress_popup(self) -> None:
        if not self._last_stats:
            return
        self._hide_stats_card()
        card = QFrame(self)
        card.setObjectName("proofProgressPopup")
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        title = QLabel(self._scope.text())
        title.setObjectName("sectionTitle")
        grid.addWidget(title, 0, 0, 1, 2)
        for row_index, (name, value) in enumerate(self._last_stats.items(), start=1):
            key_label = QLabel(str(name))
            key_label.setObjectName("muted")
            value_label = QLabel(str(value))
            grid.addWidget(key_label, row_index, 0)
            grid.addWidget(value_label, row_index, 1)
        card.adjustSize()
        anchor = self._progress_ring.mapTo(self, QPoint(0, 0))
        y = anchor.y() - card.height() - 8
        if y < 4:
            y = anchor.y() + self._progress_ring.height() + 8
        card.move(max(4, anchor.x() - 4), y)
        card.show()
        card.raise_()
        self._stats_card = card
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _hide_stats_card(self) -> None:
        if self._stats_card is not None:
            self._stats_card.hide()
            self._stats_card.deleteLater()
            self._stats_card = None
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if self._stats_card is not None:
            if event.type() == QEvent.Type.MouseButtonPress:
                if watched is self._progress_ring:
                    return False
                local = self._stats_card.mapFromGlobal(event.globalPosition().toPoint())
                if not self._stats_card.rect().contains(local):
                    self._hide_stats_card()
            elif event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._hide_stats_card()
        return False

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self._hide_stats_card()
        super().hideEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._hide_stats_card()
        super().resizeEvent(event)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        self._save_all()
        super().closeEvent(event)
