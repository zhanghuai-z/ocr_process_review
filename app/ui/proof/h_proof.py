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
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, QRect, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFontMetrics, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
    QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QProgressBar,
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, BlockType, Line, OcrProject, Page, ProofStatus
from app.core.block_attributes import block_attributes, normalize_source_label, semantic_block_type
from app.core.ocr_ir import is_formula_marker_token
from app.core.page_image_cache import PageImageCache
from app.ui.widgets.page_directory import PageDirectoryList
from app.core.proof_line_utils import iter_unique_page_hproof_lines
from app.core.proof_state import (
    TOPIC_LINE_PROOF_CHANGED,
    ProofLineViewModel,
    ProofSelection,
    ProofUpdateRequest,
    proof_request_matches_line,
)
from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp
from app.services.proof_probe_text_service import (
    displayed_text as _displayed_text,
    save_displayed_edit as _save_displayed_edit,
    resolve_block_line_index as _resolve_block_line_index,
)
from app.services.proof_image_service import clamp_line_box_pixels
from app.ui.proof.confidence_utils import char_confidence
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.proof import char_verdict as _cv
# NOTE: AlignmentRibbon 已从布局中移除（proof-layout-collections 第 1 任务）。
# 用户原话：“既然已经做图字对应，就不要第三行文本行”。图字 y 轴对应
# 现在改回只走 hover/click 联动（图像悬停 → editor 高亮当前字；editor
# 光标变 → 图像画当前字 bbox），不再用 ribbon 重复绘制一行文本。

# ── 样式常量 ──────────────────────────────────────────────────
ROW_PAD_Y    = 4     # 裁图上下各加 4px
IMAGE_ROW_H  = 32    # 行图像显示高度（px）
# 脚注、数字、标点的 Hanwang 字符框通常比正文窄。字号按较小文本优先，
# 避免按 slot center 自绘时把标点和数字挤在一起。
TEXT_FONT_PX = 22
TEXT_LINE_HEIGHT_PX = 28
TEXT_EDITOR_MAX_H = 32
TEXT_SLOT_MIN_W = 10.0
TEXT_SLOT_GUTTER_W = 4.0
TEXT_SLOT_GAP_W = 1.0
TEXT_SLOT_CLIPPED_MIN_W = 1.0
# 保持“图 + 文本”两层的既有总高度，减少布局连锁变化。
LINE_PAIR_H = 70
# Phase 17 blocker：字格模式下需要为 CharCellRow 留够竖向空间。
# CharCellRow 自身固定高 = IMG_H(36) + EDIT_H(26) + 6 内边距 = 68。
# Phase 24（上图下字后）：image_row(32) + cell_row(68) + spacing(4) +
# 上下 padding(8) = 112 px。
CELL_PAIR_H = 112
TEXT_FONT_FAMILY = "'SimHei','Microsoft YaHei UI','Noto Sans CJK SC','PingFang SC','SimSun',sans-serif"
LABEL_W      = 88    # 左侧行号列宽
STATUS_W     = 80    # 右侧状态列宽
LOW_CONF     = 0.80

_STATUS_COLOR = {
    ProofStatus.OK:           "#4CAF50",
    ProofStatus.MODIFIED:     "#1a73e8",
    ProofStatus.AUTO_FLAGGED: "#FF9800",
    ProofStatus.UNCHECKED:    "#c8d0db",
}
_STATUS_LABEL = {
    ProofStatus.OK:           "已确认",
    ProofStatus.MODIFIED:     "已修改",
    ProofStatus.AUTO_FLAGGED: "⚑ 疑点",
    ProofStatus.UNCHECKED:    "待确认",
}

_DEBUG_FORMULA_LINE_FLAGS = {"hanwang_route_inline_formula"}
_DEBUG_TABLE_LINE_FLAGS = {"hanwang_route_table"}
_DEBUG_FORMULA_EXCLUDED_LABELS = {"formula_number"}
_DEBUG_FORMULA_LABEL_TOKENS = ("formula", "equation", "math")
_DEBUG_TABLE_LABEL_TOKENS = ("table",)


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
        normalize_source_label(attrs.raw_label),
        attrs.normalized_semantic_label,
    }
    for key in ("block_label", "label", "type", "category"):
        value = attrs.raw_payload.get(key)
        if value:
            labels.add(normalize_source_label(value))
        value = attrs.app_payload.get(key)
        if value:
            labels.add(normalize_source_label(value))
    return {label for label in labels if label}


def _line_has_formula_source(line: Line) -> bool:
    has_formula_route = any(flag in _DEBUG_FORMULA_LINE_FLAGS for flag in line.review_flags)
    formula_texts: list[str] = []
    for char in line.chars or []:
        source = normalize_source_label(getattr(char, "bbox_source", ""))
        if source == "paddle_inline_formula":
            formula_texts.append(str(getattr(char, "token_text", "") or getattr(char, "char", "") or ""))
    if formula_texts:
        return any(not is_formula_marker_token(text) for text in formula_texts)
    if has_formula_route:
        return not is_formula_marker_token(line.display_text)
    return False


def _line_is_formula_marker_only(line: Line) -> bool:
    text = line.display_text
    if text and is_formula_marker_token(text):
        return True
    formula_texts = [
        str(getattr(char, "token_text", "") or getattr(char, "char", "") or "")
        for char in line.chars or []
        if normalize_source_label(getattr(char, "bbox_source", "")) == "paddle_inline_formula"
    ]
    return bool(formula_texts) and all(is_formula_marker_token(text) for text in formula_texts)


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


def _is_debug_table_block(block: Block) -> bool:
    if semantic_block_type(block) == BlockType.TABLE:
        return True
    labels = _debug_block_labels(block)
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
    text = line.display_text
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
    for block in page.blocks:
        formula_block = _is_debug_formula_block(block)
        table_block = _is_debug_table_block(block)
        for line_idx, line in enumerate(block.lines):
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
# proof-slot-residual：固定槽位辅助函数
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

    这就是 proof-slot-residual 第 1 任务要求的“OCR 元素数 = 槽位数，且
    超短行自动补空白槽，超长行老老实实降级而不是悄悄删字”。
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

    Task #2（固定元素数下编辑限制）：当 ``self._fixed_length`` 不为 None 时
    （= 行有 ``line.chars``，图像元素数固定），输入行为强制为"覆写模式"：

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
    length_violation  = Signal(str)  # 试图改变长度时发出原因字符串
    # hproof-visual-marking 升级：鼠标悬停字符位置变化（-1 = 离开 editor 区域）
    hover_char_changed = Signal(int)
    # proof-direct-input-closure round 10 任务 1：编辑器获取焦点 / 点击 →
    # 自动激活本行。signal 用 mouse + focus 两条路径覆盖，键盘 Tab 也算。
    row_focus_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fixed_length: Optional[int] = None
        self._last_hover_idx: int = -1
        # 启用鼠标跟踪：无需按下也能收到 mouseMoveEvent，用于图字 hover 联动
        self.setMouseTracking(True)
        # hproof-yaxis-quiet-load 本轮任务 1：图字 y 轴对应。
        # 每个字的 x 坐标改为由 line.chars[i].bbox 映射到 editor 像素空间；
        # _line_pair 在每次 _render_line_image 后调 set_slot_geometry()
        # 把 x_centers / widths 推进来。set 为 None / 空列表 → 走原生
        # QPlainTextEdit 渲染（降级，例如 chars 缺失 / 未对齐时）。
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
            # proof-interaction-slots 第 3 任务：固定槽位语义
            #   - Backspace / Delete：不删字、不改长度，**将当前 / 邻位槽位
            #     填充为空格**（line.chars 数量不变，文本对应字位置变成 " "）。
            #   - 普通字符输入：仍按"自动选下一字 + 覆写"路径。
            #   - 输入长度 > 选区长度：自动裁断到选区长度（不再拒绝，不再弹 tooltip）。
            #   - Ctrl+X 剪切：把选区填空，而非拒绝。
            #   注：blank 用 ASCII 空格 ' '；保存后 final_text 的相应字位置即为空。
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

        proof-interaction-slots 第 3 任务：不再因长度不匹配拒绝粘贴，
        而是截断 / 用空格补足，与键盘输入语义一致。
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

    # proof-direct-input-closure round 10 任务 1：mousePressEvent / focusInEvent
    # 都发 row_focus_requested。上层 _LinePair 用它激活本行（之前只有
    # 点 _active_bar / _img_lbl 才切行，点文本不行——这与用户直觉相反）。
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
    屏幕上的字符、命中区域、选中背景都按 ``line.chars[i].bbox`` 传入的
    slot geometry 绘制，避免“视觉字位”和 Qt 文本光标坐标不一致。
    """

    confirm_requested = Signal()
    prev_requested = Signal()
    next_requested = Signal()
    flag_requested = Signal()
    skip_requested = Signal()
    revert_requested = Signal()
    length_violation = Signal(str)
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
        self._apply_document_line_height()

    # ── 兼容 QPlainTextEdit 调用面 ─────────────────────────────

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
        centers = self._slot_x_centers or []
        if not centers:
            return -1
        widths = self._slot_widths or [TEXT_SLOT_MIN_W] * len(centers)
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
            text = self.toPlainText()
            p.setFont(self.font())
            fm = QFontMetrics(self.font())
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
            for i in range(n):
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
                char_w = fm.horizontalAdvance(ch)
                tx = int(round(float(center) - char_w / 2.0))
                p.drawText(tx, y_baseline, ch)
        finally:
            p.end()

    def _fallback_slot_centers(self, text: str) -> list[Optional[float]]:
        if not text:
            return []
        fm = QFontMetrics(self.font())
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
        self.row_focus_requested.emit()
        pos = self._event_pos(event)
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
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and no_mod:
            self.confirm_requested.emit()
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
# 共享实现见 app.services.proof_probe_text_service；本文件保留同名局部别名
# 以保持调用点不变（_displayed_text / _save_displayed_edit / _resolve_block_line_index）。
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 单行对（图像行 + 识别文本行）控件
# ─────────────────────────────────────────────────────────────

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
        self._active       = False
        self._image_loaded = False
        self._line_crop = None
        self._line_crop_origin: tuple[int, int] = (0, 0)
        # 最近一次行图像缩放比例，用于把 _img_lbl 上的点击位置反查回原图坐标
        self._render_scale: float = 1.0
        # hproof-visual-marking：editor 鼠标悬停的字符索引（-1 = 未悬停）。
        # _render_line_image 会按 active 选区/光标 + 这个 hover idx 联合画框。
        self._hover_char_idx: int = -1

        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    # ── 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFixedHeight(LINE_PAIR_H)
        # Phase 25：彻底取消字格模式 / 取消"图像/文本"hdr 标签 / 整行文本框
        # 始终可见。布局保留 Phase 24 的"上图下字"骨架：
        #   root QHBoxLayout = [active_bar | content_v(img_row, editor) | status]
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 1, 8, 1)
        root.setSpacing(4)

        # 蓝色激活条（左边框）
        self._active_bar = QWidget()
        self._active_bar.setFixedWidth(4)
        self._active_bar.setStyleSheet("background: transparent;")
        root.addWidget(self._active_bar)
        self._active_bar2 = self._active_bar

        # 中间内容容器：上图下字
        self._content = QWidget()
        content_v = QVBoxLayout(self._content)
        content_v.setContentsMargins(0, 0, 0, 0)
        content_v.setSpacing(0)

        # ── 上：行图像（去掉左侧"图像 N"hdr，直接占满宽度）──────
        self._img_lbl = QLabel()
        self._img_lbl.setFixedHeight(IMAGE_ROW_H)
        self._img_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._img_lbl.setStyleSheet("background:#fafbfc;")
        content_v.addWidget(self._img_lbl)

        # proof-layout-collections 第 1 任务：AlignmentRibbon 已删除。
        # 用户明确不要 “第三行文本行”。图字 y 轴对应改回只通过 editor↔image
        # 的 hover/click 联动表达（_on_editor_hover_char / _img_clicked_lookup）。

        # ── 下：整行文本框（永远可见；弱光标 + 等宽 + 与图像 y 对齐）──
        self._editor = _SlotLineEditor()
        # proof-direct-input-closure round 10 任务 2：继续弱化"文本框感"。
        # 之前的 border:1px solid #e3e8ef 让每行都像一个独立输入框，光
        # 标心智依然强烈。去掉边框、底色随激活态走（激活 = #f0f6ff，
        # 非激活 = transparent），让用户看到的是"一行可改的文字"，而
        # 非"一个文本框"。
        self._editor.setStyleSheet(self._editor_style(active=False))
        self._editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._editor.setFixedHeight(TEXT_EDITOR_MAX_H)
        self._editor.document().setDocumentMargin(0)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # Phase 25：弱光标 —— cursor width 0，不显示插入符；逐字定位/高亮
        # 通过 extraSelections + cursor.setPosition 体现。
        self._editor.setCursorWidth(0)
        # 初始填入显示空间文本
        # proof-slot-residual 第 1 任务：加载时就把文本规范化到槽位数
        # （仅在可锁定时补空；超长不动）。
        _initial_disp = _displayed_text(self._line, self._page, self._block)
        _canon, _ = _canonicalize_text_to_slots(_initial_disp, self._line.chars or [])
        self._editor.setPlainText(_canon)
        self._editor.apply_inline_y_axis_metrics()
        # 信号转发
        self._editor.confirm_requested.connect(lambda: self.confirmed.emit(self._idx))
        self._editor.prev_requested.connect(self.prev_req)
        self._editor.next_requested.connect(self.next_req)
        self._editor.flag_requested.connect(self.flag_req)
        self._editor.skip_requested.connect(self.skip_req)
        self._editor.revert_requested.connect(self._revert)
        # Task #2: 长度违规 → 浮动 tooltip 提示（不弹模态）
        self._editor.length_violation.connect(self._on_length_violation)
        # hproof-visual-marking：鼠标悬停 editor → 在行图上高亮对应字
        self._editor.hover_char_changed.connect(self._on_editor_hover_char)
        self._editor.selectionChanged.connect(self._refresh_extra_selections)
        self._editor.selectionChanged.connect(self._render_line_image)
        self._editor.cursorPositionChanged.connect(self._refresh_extra_selections)
        self._editor.cursorPositionChanged.connect(self._render_line_image)
        # 编辑触发置信度高亮重绘（修过的字按 OK 颜色处理）
        self._editor.textChanged.connect(self._refresh_extra_selections)
        # Task #1：编辑改变字数 → 重新评估图字是否对齐 → 刷新 ⚠ 标
        self._editor.textChanged.connect(self._sync_editor_slot_geometry)
        self._editor.textChanged.connect(self._refresh_status)
        # proof-direct-input-closure round 10 任务 1：editor focus/click → 激活本行
        self._editor.row_focus_requested.connect(self._on_editor_focus_in)
        # Task #2：按 line.chars 锁定编辑器固定长度（图字一一对应不变）
        self._apply_fixed_length_to_editor()
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
        bg = "#eaf3ff" if active else "transparent"
        return (
            f"font-family:{TEXT_FONT_FAMILY}; font-size:{TEXT_FONT_PX}px; "
            f"padding:0; background:{bg}; border:none;"
        )

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        blue = "#1a73e8"
        if active:
            bar_style = f"background:{blue}; border-radius:2px;"
            frame_style = (
                "QFrame#linePair { background:#eaf3ff; "
                "border-top:1px solid #c9ddff; border-bottom:1px solid #c9ddff; }"
            )
            content_bg = "#eaf3ff"
            image_style = "background:#f8fbff; border-bottom:1px solid #d9e7ff;"
            self._editor.setFocus()
        else:
            # 切走前先把 in-flight 文本保存（编辑器始终可见）
            self._flush_editor_if_dirty()
            # proof-interaction-slots 第 2 任务：切行时清掉本行 editor 里的选中状态
            # 和高亮调用。不清会让用户看到“他行还有选中感”。
            cur = self._editor.textCursor()
            if cur.hasSelection():
                cur.clearSelection()
                self._editor.setTextCursor(cur)
            self._editor.setExtraSelections([])
            self._hover_char_idx = -1
            bar_style = "background:transparent;"
            frame_style = ""
            content_bg = "transparent"
            image_style = "background:#fafbfc; border-bottom:1px solid #edf1f7;"

        self._active_bar.setStyleSheet(bar_style)
        self._active_bar2.setStyleSheet(bar_style)
        self._content.setStyleSheet(f"background:{content_bg};")
        self._img_lbl.setStyleSheet(image_style)
        self._editor.setStyleSheet(self._editor_style(active=active))
        if hasattr(self._editor, "set_active_visual"):
            self._editor.set_active_visual(active)
        self.setProperty("active", active)
        self.setStyleSheet(frame_style)
        self._refresh_status()
        self._refresh_extra_selections()
        self._render_line_image()

    def _flush_editor_if_dirty(self) -> None:
        """若 editor 当前文本与显示空间文本不一致，发 text_saved 让面板落盘。
        Phase 25：editor 始终可见，不再判 isHidden。"""
        new_text = self._editor.toPlainText()
        if new_text != _displayed_text(self._line, self._page, self._block):
            self.text_saved.emit(self._idx, new_text)

    def _apply_fixed_length_to_editor(self) -> None:
        """按当前 line.chars 状态启用/关闭固定长度覆写模式。

        proof-interaction-slots 第 1+3 任务：
        - 彻底不再给 editor 设任何 tooltip。以前为了提示"固定 N 字"
          会 setToolTip；反复设空会在 Qt 上重出空白 hover 框残影。
        - 固定模式本身仍启用（控制 keyPressEvent 里 Backspace/Delete 走
          "填空字"路径不是拒绝）。
        """
        chars = self._line.chars or []
        text_len = len(self._editor.toPlainText())
        fixed = len(chars) if chars and text_len == len(chars) else None
        self._editor.set_fixed_length(fixed)

    def _on_length_violation(self, reason: str) -> None:
        """保留接口以免旧信号连接报错，但不再弹 tooltip。

        proof-interaction-slots 第 1+3 任务：固定模式不再拒绝删除动作
        （改为将槽位填空），也不再拒绝超长输入（裁断）。原"超长"、"禁删"
        提示路径不再需要，也不再设 tooltip。
        """
        return

    def _on_editor_hover_char(self, idx: int) -> None:
        """editor 鼠标悬停字符 idx 变化 → 在行图上画 hover 框（图字对应升级）。

        只在 aligned 时生效；否则保留行级显示，不假装能定位到某字。
        """
        if not self._chars_aligned():
            new_idx = -1
        else:
            new_idx = idx if 0 <= idx < len(self._line.chars) else -1
        if new_idx == self._hover_char_idx:
            return
        self._hover_char_idx = new_idx
        self._render_line_image()

    # ── Phase 25：弱光标 + 逐字高亮（取代字格模式）──────────────

    def _chars_aligned(self) -> bool:
        """Editor 文本是否与 ``line.chars`` 严格一一对应。

        proof UI clarity（Task #1）：图像 char.bbox 与文本下标的映射只有在
        ``len(text) == len(chars)`` 时才可靠。一旦用户编辑增删字符、或 OCR
        本身就给出错位的 chars 列表，**就不要**伪装成"第 i 字 ↔ 第 i 个
        bbox"——而是改走降级路径。

        proof-layout-collections（第 2 任务 “内容偏移”）重点加强：
        即使 len(text) == len(chars)，只要任一 ``char.char`` 不是恰好 1 个
        字符（word/token-granularity 的 char 可能是多字 token），“第 i 个
        char ↔ 第 i 个文本字”也会被错位（OCR 给了 5 个 token 但文本 12 字
        刷后成 12 个 glyph）。这是上一轮“坐标对了但内容偏移”的根因。
        以后发现 chars 任一元素不是单字符 → 降级，不画逐字高亮。
        """
        chars = self._line.chars
        if not chars:
            return False
        if len(self._editor.toPlainText()) != len(chars):
            return False
        for c in chars:
            ch = c.char or ""
            if len(ch) != 1:
                return False
        return True

    # ── 文本颜色规则（hproof-yaxis-verdicts 第三任务）─────────
    # 颜色 + 证据链由 :mod:`app.ui.proof.char_verdict` 集中负责，本文件只做调用。
    #
    # 关键变更（vs hproof-visual-marking 旧版）：
    #   · "用户修改过 → 黑色 = 再无置信度问题" 已删除。用户本轮 TASK 明确：
    #     橙/红仍参与正确性判断，不能因人工改过自动洗白。颜色现在只读
    #     OCR confidence + LLM 双源一致性。
    #   · "黑 = 校对过绝对正确" 当前 codebase 没有可靠证据链来源（quality_probe
    #     禁改、Line 无字级核验字段）。本控件因此 **不再** 把任何字标成黑色
    #     "绝对正确"，默认 unverified 用普通深灰；handoff 写明缺什么。
    #   · "用户改过" 单独成为 ``user_modified`` 标志，UI 在 AlignmentRibbon
    #     上画细下划线提示，但不改变颜色。

    def _classify_char_verdict(self, i: int) -> Optional[_cv.CharVerdict]:
        """返回第 i 个字的 verdict；下标越界 / 未对齐 → None。"""
        chars = self._line.chars or []
        if not (0 <= i < len(chars)):
            return None
        text = self._editor.toPlainText()
        if i >= len(text):
            return None
        conf = char_confidence(self._line, i)
        ocr = self._line.ocr_text or self._line.original_text or ""
        llm = self._line.llm_suggestion or ""
        # 只在等长时取同下标字符；长度不一致时退回 None，避免错位比对
        ocr_ch = ocr[i] if len(ocr) == len(text) and i < len(ocr) else None
        llm_ch = llm[i] if len(llm) == len(text) and i < len(llm) else None
        return _cv.classify_char(
            confidence=conf,
            text_char=text[i],
            ocr_char=ocr_ch,
            llm_char=llm_ch,
        )

    def _refresh_extra_selections(self) -> None:
        """生成 editor 的 extraSelections：
        - 按 verdict 给每个字上前景色（绿/橙/红/灰）。
        - 当前光标所在/选中字：蓝色淡背景（"当前字"指示，配合弱光标）。

        Phase 25 起 caret 宽度=0，弱光标完全靠这套 extraSelections 表达。
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
            chars = self._line.chars or []
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
        for i, ch in enumerate(self._line.chars):
            if ch.bbox is None:
                continue
            if ch.bbox.x <= orig_x <= ch.bbox.x2:
                target_idx = i
                break
        if target_idx is None:
            best_dist = float("inf")
            for i, ch in enumerate(self._line.chars):
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
        # Phase 25：editor 始终可见，按光标/选区在行图上高亮对应 char.bbox。
        # proof UI clarity（Task #1）：仅当文本与 chars 严格一一对应才画框，
        # 否则不强行画——避免把"第 N 字"高亮到了别的字 bbox 上的假象。
        if self._active and self._chars_aligned():
            cursor = self._editor.textCursor()
            start = min(cursor.selectionStart(), cursor.selectionEnd())
            end = max(cursor.selectionStart(), cursor.selectionEnd())
            if end > start:
                highlight_range = (start, end)
            else:
                pos = cursor.position()
                if 0 <= pos < len(self._line.chars):
                    highlight_range = (pos, pos + 1)
        if highlight_range is not None:
            start, end = highlight_range
            ox, oy = self._line_crop_origin
            for idx in range(start, min(end, len(self._line.chars))):
                char = self._line.chars[idx]
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
        # hproof-visual-marking：editor 鼠标悬停 → 在行图上画绿色细框
        # （与"当前字"蓝框区分；当 hover idx 与当前字重叠时，蓝框已经画过，
        # 此处的绿框会叠在外侧 1px，仍能看出"鼠标正在指这个字"）。
        if (
            self._chars_aligned()
            and 0 <= self._hover_char_idx < len(self._line.chars)
        ):
            ox, oy = self._line_crop_origin
            char = self._line.chars[self._hover_char_idx]
            if char.bbox is not None:
                x1 = max(0, char.bbox.x - ox)
                y1 = max(0, char.bbox.y - oy)
                x2 = min(crop.shape[1] - 1, char.bbox.x2 - ox)
                y2 = min(crop.shape[0] - 1, char.bbox.y2 - oy)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (40, 167, 69), 1)
        h, w = crop.shape[:2]
        # 缩放到 IMAGE_ROW_H 高度，同时限制最大宽度（避免超宽行撑开布局）。
        # 严格保持宽高比：先按高度缩放；若超宽再按宽度缩放重算高度。
        scale = IMAGE_ROW_H / h
        new_w = max(1, int(round(w * scale)))
        MAX_LINE_W = 1200
        if new_w > MAX_LINE_W:
            scale = MAX_LINE_W / w
            new_h = max(1, int(round(h * scale)))
            crop = cv2.resize(crop, (MAX_LINE_W, new_h), interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(crop, (new_w, IMAGE_ROW_H), interpolation=cv2.INTER_AREA)
        self._render_scale = scale
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), rw, rh, rw * 3, QImage.Format.Format_RGB888)
        self._img_lbl.setPixmap(QPixmap.fromImage(qimg))
        # hproof-yaxis-quiet-load 本轮任务 1：把图像里每个字的 x 中心
        # （editor 像素坐标系）推给 editor。editor.paintEvent 用这套坐标
        # 自绘每字 —— 用户向正下方看，落到的文本字就是同一个字。
        # editor 与 _img_lbl 同为 content_v 的 full-width 子控件，且 _img_lbl
        # 内 pixmap 左对齐 → editor x=0 == _img_lbl x=0 == 行图左边缘。
        self._sync_editor_slot_geometry()

    def _sync_editor_slot_geometry(self) -> None:
        """按 line.chars[i].bbox + _render_scale + _line_crop_origin 推算
        每字在 editor 视口内的 x 中心 & 宽度；交给 editor 自绘文本。

        chars 未对齐 / 缺 bbox → 清空 → editor 走原生渲染（降级）。
        """
        editor = getattr(self, "_editor", None)
        if editor is None:
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
        for idx, ch in enumerate(self._line.chars):
            if ch.bbox is None:
                x_centers.append(None)
                widths.append(0.0)
                continue
            cx_src = (ch.bbox.x + ch.bbox.x2) / 2.0 - float(ox)
            x_centers.append(cx_src * scale)
            text_char = text[idx] if idx < len(text) else ch.char
            widths.append(_slot_visual_width(text_char, fm))
        editor.set_slot_geometry(x_centers, _clip_slot_widths_to_centers(x_centers, widths))

    def refresh_text(self) -> None:
        """外部（VProof / probe 切换）更新 final_text 后同步 editor 文本。

        Phase 25：editor 始终可见 → 直接 blockSignals + setPlainText 重写当前
        文本，避免触发 dirty flush。active 行上若用户正在编辑，会被覆盖
        （这是与 V 同步的既有行为；实时 dirty 已通过 _flush_editor_if_dirty
        在 set_active 切走前落盘）。
        """
        new_disp = _displayed_text(self._line, self._page, self._block)
        # proof-slot-residual 第 1 任务：同步时也走槽位规范化，让 V/H
        # 互同后本行的显示不会从“锁定”退回“自由”。
        new_disp, _ = _canonicalize_text_to_slots(new_disp, self._line.chars or [])
        if self._editor.toPlainText() != new_disp:
            self._editor.blockSignals(True)
            self._editor.setPlainText(new_disp)
            self._editor.apply_inline_y_axis_metrics()
            self._editor.blockSignals(False)
        # 显示文本变了 → 字数可能变 → 重新评估 fixed_length
        self._apply_fixed_length_to_editor()
        self._refresh_status()
        self._refresh_extra_selections()
        # hproof-yaxis-quiet-load 本轮任务 1：文本/对齐状态变化 → 重算
        # 图字 x 映射；当 _render_scale 还未就绪时 _sync 会自动清空进入降级。
        self._sync_editor_slot_geometry()

    def rebind(
        self,
        block: Block,
        line: Line,
        page: Page,
        line_in_page: int,
        debug_badge: str = "",
    ) -> None:
        """Point this UI row at the current project Line without rebuilding it."""
        self._block = block
        self._line = line
        self._page = page
        self._line_in_page = line_in_page
        self._debug_badge = debug_badge
        self._image_loaded = False
        self._line_crop = None
        # Phase 25：editor 始终可见。rebind 不触碰 editor 内容，
        # 与 Phase 24 之前 "isHidden() 才同步" 的行为等价 —— 这样用户在原行上
        # 未提交的编辑（dirty 文本）不会被 merge_pages 路径上的 rebind 覆盖。
        # 真正的"显示空间已变"由 refresh_text() 单独负责。
        # Task #2：新行可能 chars 数不同 → 重新评估 fixed_length
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

    def _on_click(self, event) -> None:
        self.clicked.emit(self._idx)

    def _on_editor_focus_in(self) -> None:
        """proof-direct-input-closure round 10 任务 1：editor 内点击/取得焦点
        即激活本行。不传 event；本行已经是 active 时静默忽略，避免对
        cursor/focus 链路造成多余刷新。"""
        if self._active:
            return
        self.clicked.emit(self._idx)

    def _revert(self) -> None:
        # 还原到 OCR 原始文本。Round 15 后评测不再修改显示文本，
        # 所以直接用 original_true 即可，无需 apply_probes_to_display。
        original_true = self._line.original_text or self._line.ocr_text or self._line.text or ""
        self._editor.blockSignals(True)
        # proof-slot-residual 第 1 任务：还原后也补空到槽位数
        _rev_canon, _ = _canonicalize_text_to_slots(
            original_true, self._line.chars or []
        )
        self._editor.setPlainText(_rev_canon)
        self._editor.apply_inline_y_axis_metrics()
        self._editor.blockSignals(False)

    def _refresh_status(self) -> None:
        status = self._line.proof_status
        color = _STATUS_COLOR.get(status, "#ccc")
        label = _STATUS_LABEL.get(status, "")
        # Task #1 升级：fixed_length 模式下 text_n 永远等于 char_n，aligned 一直成立。
        # 这里只在真正异常时给 ⚠ 提示；正常情况下不再"为了诚实而提示"，避免
        # 用户每行都看到一条空有解释、没有动作意义的 hover。
        text_n = len(self._editor.toPlainText()) if hasattr(self, "_editor") else 0
        char_n = len(self._line.chars) if self._line.chars else 0
        unaligned = bool(self._line.chars) and text_n != char_n
        warn = ""
        if unaligned:
            warn = " <span style='color:#c62828;font-size:11px;'>⚠</span>"
            tip = (
                f"图字未对齐：文本 {text_n} 字 ≠ 图像字符 {char_n} 字\n"
                f"已停用逐字高亮；请把文本改回 {char_n} 字以恢复图字一一对应"
            )
        else:
            tip = ""
        badge = ""
        if self._debug_badge:
            badge = (
                f"<span style='color:#555;font-size:10px;'>"
                f"{self._debug_badge}</span><br/>"
            )
        self._status_lbl.setText(
            f"{badge}<span style='color:{color};font-size:11px;'>● {label}</span>{warn}"
        )
        # proof-interaction-slots 第 1 任务：彻底不给 status_lbl 设 tooltip（连空串都不设）
        # 避免 Qt 某些环境下“空白 hover 框”。是否未对齐已经用 warn 图标表达。


# ─────────────────────────────────────────────────────────────
# 横校面板主体
# ─────────────────────────────────────────────────────────────

class HProofPanel(QWidget):
    """横校面板：滚动列表 + 工具栏，对照 ui-2.jpg 设计。"""

    proof_saved = Signal()
    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: List[Page] = []
        self._items: List[Tuple[Block, Line, Page, int]] = []
        self._pairs: List[_LinePair] = []
        self._line_view_models: List[ProofLineViewModel] = []
        self._current_idx: int = 0
        self._filter_updating = False
        self._selected_page_number: int | None = None  # Phase 25：左侧目录唯一过滤源
        self._show_formula_debug = False
        self._show_table_debug = False
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        # H/V 校对联动：订阅其他 panel 编辑事件；origin == id(self) 的事件忽略。
        # Phase 18 blocker 3：保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            TOPIC_LINE_PROOF_CHANGED, self._on_external_line_changed,
        )
        self.destroyed.connect(lambda *_: self._teardown_bus())
        self._build_ui()

    def _teardown_bus(self) -> None:
        """Phase 18 blocker 3：释放 ProofStateBus 订阅。

        QObject.destroyed 信号在 Python 端仍可调用 unsubscribe；幂等，多次安全。"""
        unsub = getattr(self, "_bus_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._bus_unsub = None

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Phase 24：横校改为三栏布局：
        #   左：复用版面分析的页面目录（PageDirectoryList）
        #   中：原滚动列表（_LinePair 已改为上图下字）
        #   右：工具栏 + 快捷键说明（用户明确要求"放进界面"）
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("hproofSplitter")

        # ── 左：页面目录 ────────────────────────────────────────
        self._page_dir = PageDirectoryList()
        self._page_dir.page_selected.connect(self._on_page_dir_selected)
        splitter.addWidget(self._page_dir)

        # ── 中：滚动列表 ────────────────────────────────────────
        center = QWidget()
        center_v = QVBoxLayout(center)
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()

        # 空状态提示（无数据时显示）
        self._empty_lbl = QLabel("完成 OCR 识别后，横校数据将在此展示")
        self._empty_lbl.setObjectName("proofEmpty")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setMinimumHeight(120)
        self._list_layout.insertWidget(0, self._empty_lbl)

        self._mode_banner = QLabel("")
        self._mode_banner.setObjectName("hproofModeBanner")
        self._mode_banner.setStyleSheet(
            "background:#fff7ed; color:#9a3412; border-bottom:1px solid #fed7aa; "
            "padding:6px 10px; font-size:12px;"
        )
        self._mode_banner.setVisible(False)
        center_v.addWidget(self._mode_banner)
        self._scroll.setWidget(self._list_widget)
        center_v.addWidget(self._scroll, 1)
        splitter.addWidget(center)

        # ── 右：工具栏（垂直）+ 快捷键说明 ──────────────────────
        right = QWidget()
        right.setObjectName("hproofRightDock")
        right.setMinimumWidth(220)
        right.setMaximumWidth(300)
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(10, 10, 10, 10)
        right_v.setSpacing(10)

        # 操作按钮组（Phase 25：右栏精简，只剩保存 / 标记 / 跳过）
        actions_lbl = QLabel("操作")
        actions_lbl.setStyleSheet("font-weight:600; color:#444; font-size:12px;")
        right_v.addWidget(actions_lbl)

        self._btn_save = QPushButton("保存")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.setToolTip("保存所有修改  (Ctrl+S)")
        self._btn_flag = QPushButton("⚑ 标记  F5")
        self._btn_skip = QPushButton("跳过  F6")
        for btn in (self._btn_save, self._btn_flag, self._btn_skip):
            btn.setMinimumHeight(30)
            right_v.addWidget(btn)

        status_lbl = QLabel("当前")
        status_lbl.setStyleSheet("font-weight:600; color:#444; font-size:12px;")
        right_v.addWidget(status_lbl)

        self._current_scope_lbl = QLabel("全部页面")
        self._current_scope_lbl.setObjectName("muted")
        self._current_scope_lbl.setWordWrap(True)
        self._current_scope_lbl.setStyleSheet("font-size:12px; color:#5f6b7a;")
        right_v.addWidget(self._current_scope_lbl)

        self._current_line_lbl = QLabel("当前行 0 / 0")
        self._current_line_lbl.setStyleSheet("font-size:13px; color:#1f2937;")
        right_v.addWidget(self._current_line_lbl)

        self._proof_progress_bar = QProgressBar()
        self._proof_progress_bar.setTextVisible(False)
        self._proof_progress_bar.setFixedHeight(8)
        self._proof_progress_bar.setRange(0, 1)
        self._proof_progress_bar.setValue(0)
        self._proof_progress_bar.setStyleSheet(
            "QProgressBar { background:#e5e7eb; border:0; border-radius:4px; }"
            "QProgressBar::chunk { background:#1a73e8; border-radius:4px; }"
        )
        right_v.addWidget(self._proof_progress_bar)

        self._handled_lbl = QLabel("已处理 0 / 0")
        self._handled_lbl.setObjectName("muted")
        self._handled_lbl.setStyleSheet("font-size:11px; color:#667085;")
        right_v.addWidget(self._handled_lbl)

        stats_lbl = QLabel("统计")
        stats_lbl.setStyleSheet("font-weight:600; color:#444; font-size:12px;")
        right_v.addWidget(stats_lbl)

        self._pending_lbl = QLabel("待确认 0")
        self._confirmed_lbl = QLabel("已确认 0")
        self._modified_lbl = QLabel("已修改 0")
        self._flagged_lbl = QLabel("疑点 0")
        for lbl in (
            self._pending_lbl,
            self._confirmed_lbl,
            self._modified_lbl,
            self._flagged_lbl,
        ):
            lbl.setMinimumHeight(22)
            lbl.setStyleSheet(
                "font-size:12px; color:#344054; padding:2px 0;"
            )
            right_v.addWidget(lbl)

        debug_lbl = QLabel("调试")
        debug_lbl.setStyleSheet("font-weight:600; color:#444; font-size:12px;")
        right_v.addWidget(debug_lbl)

        self._btn_debug_formula = QPushButton("公式")
        self._btn_debug_formula.setObjectName("ghostBtn")
        self._btn_debug_formula.setCheckable(True)
        self._btn_debug_formula.setToolTip("只显示被识别为公式或内联公式路由的行")
        self._btn_debug_table = QPushButton("表格")
        self._btn_debug_table.setObjectName("ghostBtn")
        self._btn_debug_table.setCheckable(True)
        self._btn_debug_table.setToolTip("只显示被识别为表格或表格路由的行")
        for btn in (self._btn_debug_formula, self._btn_debug_table):
            btn.setMinimumHeight(28)
            right_v.addWidget(btn)

        # 统计徽章（保留：用于全局进度小字提示）
        self._stat_lbl = QLabel("")
        self._stat_lbl.setObjectName("muted")
        self._stat_lbl.setStyleSheet("font-size:11px; color:#666;")
        right_v.addWidget(self._stat_lbl)

        right_v.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([180, 9999, 180])
        root.addWidget(splitter, 1)

        # ── 底部状态栏 ─────────────────────────────────────────
        statusbar = QWidget()
        statusbar.setFixedHeight(28)
        statusbar.setStyleSheet("background:#f5f7fb; border-top:1px solid #e3e8ef;")
        sl = QHBoxLayout(statusbar)
        sl.setContentsMargins(12, 0, 12, 0)
        sl.setSpacing(16)

        self._total_lbl = QLabel("总字数 0")
        self._diff_lbl  = QLabel("差异 0 (0%)")
        for lbl in (self._total_lbl, self._diff_lbl):
            lbl.setStyleSheet("font-size:11px; color:#666;")
            sl.addWidget(lbl)

        for color, label in (
            ("#2e7d32", "高置信"),
            ("#e8801f", "可疑"),
            ("#c62828", "错字 / 高风险"),
            ("#222222", "已修正"),
        ):
            dot = QLabel(
                f"<span style='color:{color}'>●</span>"
                f"<span style='color:#666;font-size:11px;'> {label}</span>"
            )
            dot.setTextFormat(Qt.TextFormat.RichText)
            sl.addWidget(dot)

        sl.addStretch()
        root.addWidget(statusbar)

        # ── 信号 ───────────────────────────────────────────────
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

    # ── 公共 API ───────────────────────────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        self._refresh_page_filter()
        self._render_pages(self._filtered_pages())

    def merge_pages(self, pages: List[Page]) -> None:
        """Merge OCR background updates without rebuilding active editors."""
        if not self._pairs:
            self.load_pages(pages)
            return
        self._pages = pages
        self._refresh_page_filter()
        loaded_keys = {
            self._line_key(block, line, page, li): index
            for index, (block, line, page, li) in enumerate(self._items)
        }
        prev_page_number = self._items[-1][2].page_number if self._items else -1
        added = False
        for page in self._filtered_pages():
            page_line_num = 1
            for block, line, li in self._iter_page_lines(page):
                key = self._line_key(block, line, page, li)
                existing_index = loaded_keys.get(key)
                if existing_index is not None:
                    self._items[existing_index] = (block, line, page, li)
                    self._line_view_models[existing_index] = ProofLineViewModel.from_model(
                        page=page,
                        block=block,
                        line=line,
                        line_index=li,
                        display_text=_displayed_text(line, page, block),
                        source="hproof.merge",
                    )
                    self._pairs[existing_index].rebind(
                        block, line, page, page_line_num,
                        self._debug_badge_for(block, line),
                    )
                    page_line_num += 1
                    continue
                if page.page_number != prev_page_number:
                    sep = QLabel(f"── 第 {page.page_number} 页 ──")
                    sep.setObjectName("pageSep")
                    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sep.setMinimumHeight(26)
                    self._list_layout.insertWidget(self._list_layout.count() - 1, sep)
                    prev_page_number = page.page_number
                self._append_pair(block, line, page, li, page_line_num)
                loaded_keys[key] = len(self._items) - 1
                page_line_num += 1
                added = True
        if added:
            self._empty_lbl.setVisible(False)
            self._update_stats()
            QTimer.singleShot(100, self._load_visible_images)

    def _refresh_page_filter(self) -> None:
        """Phase 25：仅维护左侧 PageDirectoryList 的页面项；不再有 combo。"""
        usable_pages = [
            page for page in self._pages
            if self._page_has_lines(page)
        ]
        page_numbers = [p.page_number for p in usable_pages]
        self._filter_updating = True
        if hasattr(self, "_page_dir"):
            self._page_dir.set_pages(usable_pages)
            if self._selected_page_number in page_numbers:
                self._page_dir.set_current_index(
                    page_numbers.index(self._selected_page_number)
                )
        # 若选中页消失（被移除），回退到"全部页面"
        if (
            self._selected_page_number is not None
            and self._selected_page_number not in page_numbers
        ):
            self._selected_page_number = None
        self._filter_updating = False

    def _filtered_pages(self) -> List[Page]:
        if self._selected_page_number is None:
            return self._pages
        return [p for p in self._pages if p.page_number == self._selected_page_number]

    def set_current_page_number(self, page_number: int) -> None:
        """外部联动调用：把过滤器切换到指定页。"""
        self._selected_page_number = int(page_number)
        # 同步左侧目录视觉
        if hasattr(self, "_page_dir"):
            usable_numbers = [
                p.page_number for p in self._pages
                if self._page_has_lines(p)
            ]
            if page_number in usable_numbers:
                self._page_dir.set_current_index(usable_numbers.index(page_number))
        self._render_pages(self._filtered_pages())

    def _on_page_dir_selected(self, dir_idx: int) -> None:
        """Phase 25：左栏页面目录是页面过滤的唯一入口。点击 → 切换过滤 + 重渲染。"""
        if self._filter_updating:
            return
        usable_pages = [
            page for page in self._pages
            if self._page_has_lines(page)
        ]
        if not (0 <= dir_idx < len(usable_pages)):
            return
        target = usable_pages[dir_idx]
        if self._selected_page_number == target.page_number:
            return
        self._selected_page_number = target.page_number
        self.page_selected.emit(int(target.page_number))
        self._render_pages(self._filtered_pages())

    def _render_pages(self, pages: List[Page]) -> None:
        self._items.clear()
        self._pairs.clear()
        self._line_view_models.clear()

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
                self._append_pair(block, line, page, li, page_line_num)
                page_line_num += 1

        self._current_idx = 0
        self._update_stats()
        if self._pairs:
            self._activate(0)
            # 懒加载前 30 行图像
            QTimer.singleShot(100, self._load_visible_images)

    def _append_pair(self, block: Block, line: Line, page: Page, li: int, page_line_num: int) -> None:
        self._items.append((block, line, page, li))
        self._line_view_models.append(ProofLineViewModel.from_model(
            page=page,
            block=block,
            line=line,
            line_index=li,
            display_text=_displayed_text(line, page, block),
            source="hproof",
        ))
        pair = _LinePair(
            len(self._pairs), block, line, page, page_line_num,
            self._cache,
            debug_badge=self._debug_badge_for(block, line),
        )
        pair.clicked.connect(self._on_pair_clicked)
        pair.text_saved.connect(self._on_text_saved)
        pair.confirmed.connect(self._on_confirmed)
        pair.prev_req.connect(self._prev)
        pair.next_req.connect(self._next)
        pair.flag_req.connect(self._toggle_flag)
        pair.skip_req.connect(self._next)
        self._pairs.append(pair)
        self._list_layout.insertWidget(self._list_layout.count() - 1, pair)

    def _debug_enabled(self) -> bool:
        return self._show_formula_debug or self._show_table_debug

    def _debug_label(self) -> str:
        labels = []
        if self._show_formula_debug:
            labels.append("公式")
        if self._show_table_debug:
            labels.append("表格")
        return " / ".join(labels)

    def _scope_label(self) -> str:
        if self._selected_page_number is None:
            scope = "全部页面"
        else:
            scope = f"第 {self._selected_page_number} 页"
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
            formulas=self._show_formula_debug,
            tables=self._show_table_debug,
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
            new_formula == self._show_formula_debug
            and new_table == self._show_table_debug
        ):
            return
        self._save_current(silent=True)
        self._show_formula_debug = new_formula
        self._show_table_debug = new_table
        self._refresh_page_filter()
        self._update_mode_banner()
        self._render_pages(self._filtered_pages())

    def _line_key(self, block: Block, line: Line, page: Page, line_idx: int) -> tuple:
        bbox = line.bbox.normalize()
        return (
            page.display_image_path,
            page.source_path,
            int(page.source_page_index),
            int(page.page_number),
            block.block_type.value,
            int(block.order),
            int(line_idx),
            int(bbox.x),
            int(bbox.y),
            int(bbox.w),
            int(bbox.h),
        )

    def reset(self) -> None:
        self._selected_page_number = None
        for btn in (
            getattr(self, "_btn_debug_formula", None),
            getattr(self, "_btn_debug_table", None),
        ):
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._show_formula_debug = False
        self._show_table_debug = False
        self.load_pages([])
        self._stat_lbl.setText("")
        self._total_lbl.setText("总字数 0")
        self._diff_lbl.setText("差异 0 (0%)")

    # ── 导航 ───────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._current_idx > 0:
            self._save_current(silent=True)
            self._activate(self._current_idx - 1)

    def _next(self) -> None:
        if self._current_idx < len(self._pairs) - 1:
            self._save_current(silent=True)
            self._activate(self._current_idx + 1)

    def _activate(self, idx: int) -> None:
        idx = max(0, min(idx, len(self._pairs) - 1))
        if 0 <= self._current_idx < len(self._pairs) and self._current_idx != idx:
            self._pairs[self._current_idx].set_active(False)
        self._current_idx = idx
        pair = self._pairs[idx]
        pair.set_active(True)
        self._update_stats()
        # 滚动到可见
        QTimer.singleShot(30, lambda: self._scroll.ensureWidgetVisible(pair, 0, 40))

    def _on_pair_clicked(self, idx: int) -> None:
        if idx != self._current_idx:
            self._save_current(silent=True)
            self._activate(idx)

    def _on_confirmed(self, idx: int) -> None:
        """Enter 键确认当前行。"""
        if not self._items:
            return
        block, line, page, li = self._items[idx]
        # 先保存文本
        self._save_current(silent=True)
        line.proof_status = ProofStatus.OK
        self._publish_line_update(
            page=page, block=block, line=line, line_index=li,
            status=ProofStatus.OK.value, source="hproof.confirm",
        )
        self.proof_saved.emit()
        self._pairs[idx].refresh_text()
        self._update_stats()
        self._next()

    def _on_text_saved(self, idx: int, new_text: str) -> None:
        """_LinePair 在 set_active(False) 时保存；cell mode 下每次 cell 编辑
        也走这里（_LinePair._on_cell_text_committed → text_saved）。

        Phase 22 blocker 2：保存成功后，被改动行的 proof_status 通常会从
        UNCHECKED 变 MODIFIED。必须立即刷新该 pair 的状态点 ——
        但**不能**调 pair.refresh_text()（active+cell_mode 下它会作废重建
        _cell_row → 把焦点拉回第 0 格）。直接调 _refresh_status() 只更新
        状态标签，不触碰 _cell_row / editor / 焦点。"""
        if idx >= len(self._items):
            return
        block, line, page, _ = self._items[idx]
        if _save_displayed_edit(line, page, block, new_text):
            self._publish_line_update(
                page=page, block=block, line=line, line_index=self._items[idx][3],
                status=line.proof_status.value, source="hproof.text_saved",
            )
            self.proof_saved.emit()
            self._update_stats()
            # Phase 22 blocker 2：即时刷新 active pair 状态点
            if 0 <= idx < len(self._pairs):
                self._pairs[idx]._refresh_status()

    def _save_current(self, *, silent: bool = False) -> None:
        """将当前编辑器内容保存到 line 对象。"""
        if not self._pairs or self._current_idx >= len(self._pairs):
            return
        pair = self._pairs[self._current_idx]
        # Phase 25：editor 始终可见，直接读其当前文本与 line 比较保存
        new_text = pair._editor.toPlainText()
        block, line, page, _ = self._items[self._current_idx]
        if _save_displayed_edit(line, page, block, new_text):
            self._publish_line_update(
                page=page, block=block, line=line, line_index=self._items[self._current_idx][3],
                status=line.proof_status.value, source="hproof.save_current",
            )
            self.proof_saved.emit()
            pair.refresh_text()
            self._update_stats()

    def _save_all(self) -> None:
        self._save_current()
        self.proof_saved.emit()

    def _toggle_flag(self) -> None:
        if not self._items or self._current_idx >= len(self._items):
            return
        block, line, page, li = self._items[self._current_idx]
        new_status = (
            ProofStatus.UNCHECKED
            if line.proof_status == ProofStatus.AUTO_FLAGGED
            else ProofStatus.AUTO_FLAGGED
        )
        line.proof_status = new_status
        self._publish_line_update(
            page=page, block=block, line=line, line_index=li,
            status=new_status.value, source="hproof.flag",
        )
        self._pairs[self._current_idx].refresh_text()
        self._update_stats()

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

    def _on_external_line_changed(self, event=None, **kwargs) -> None:
        """收到外部（纵校）发来的 line.proof_changed → 找到本 panel 中
        line.id 匹配的行，刷新该行显示并重算统计。

        回路保护：origin == id(self) 时直接跳过（自己 publish 的事件）。
        """
        request = ProofUpdateRequest.from_legacy(event, **kwargs)
        if request.origin == id(self):
            return
        if (request.line_uid or None) is None and request.line_id is None:
            return
        touched = False
        for i, (block, line, page, _li) in enumerate(self._items):
            if proof_request_matches_line(request, line):
                pair = self._pairs[i]
                # Phase 25：editor 始终可见 → 直接走 refresh_text 同步显示文本。
                pair.refresh_text()
                touched = True
        if touched:
            self._update_stats()

    def _update_stats(self) -> None:
        total_chars = sum(len(ln.display_text) for _, ln, _, _ in self._items)
        diff_count  = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status in (ProofStatus.MODIFIED, ProofStatus.AUTO_FLAGGED)
        )
        pct = (diff_count / max(1, len(self._items))) * 100
        self._total_lbl.setText(f"总字数 {total_chars:,}")
        self._diff_lbl.setText(f"差异 {diff_count} ({pct:.1f}%)")

        total_lines = len(self._items)
        current = self._current_idx + 1 if self._pairs else 0
        confirmed = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status == ProofStatus.OK
        )
        modified = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status == ProofStatus.MODIFIED
        )
        flagged = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status == ProofStatus.AUTO_FLAGGED
        )
        pending = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status == ProofStatus.UNCHECKED
        )
        handled = confirmed + modified

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
            self._stat_lbl.setText(f"{self._scope_label()} · {total_lines} 行")


    def refresh_quality_probe_state(self) -> None:
        """重新渲染所有可见行，使显示空间文本与全局 active store 对齐。

        Phase 11：「评测：开/关」工具栏按钮已迁移到设置→正确率统计 对话框，
        所以这里只剩刷新 pair 显示一件事。"""
        for pair in self._pairs:
            pair.refresh_text()

    # ── 懒加载图像 ─────────────────────────────────────────────

    def _load_visible_images(self) -> None:
        """加载当前视口附近行的图像。"""
        if not self._pairs:
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
