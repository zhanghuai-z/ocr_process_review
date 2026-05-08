"""横校面板（PRD 3.2）：上图下文，相邻行上下文，全键盘操作。

快捷键（在文本编辑框内有效）：
  Enter       — 确认当前行并跳到下一行
  Ctrl+↑      — 上一行
  Ctrl+↓      — 下一行（等同 Enter 但不确认）
  F5          — 跳到下一疑点
  Escape      — 还原当前行到 OCR 原始文本
  Ctrl+S      — 保存全部修改

上下文图像渲染：
  - 当前行及上下各 2 行裁自同一页图，纵向堆叠
  - 上下文行 55% 透明灰化；当前行橙色 2px 边框高亮
  - 点击上下文行可直接跳转
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import (
    QColor, QImage, QKeySequence, QPixmap, QShortcut,
    QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, Line, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.core.proof_state_bus import ProofStateBus
from app.ui.widgets.confidence_badge import ConfidenceBadge

CONTEXT_LINES = 2
ROW_PAD_Y = 6
GAP = 8
DIM_ALPHA = 0.55
HIGHLIGHT_COLOR = (255, 140, 0)   # BGR orange
LOW_CONF = 0.80


# ─────────────────────────────────────────────────────────────
# 自定义文本编辑：拦截专用快捷键
# ─────────────────────────────────────────────────────────────

class _ProofEdit(QPlainTextEdit):
    """QPlainTextEdit 子类：Enter/Ctrl+↑↓/F5/Escape 发信号而非默认行为。"""

    confirm_requested    = Signal()
    prev_requested       = Signal()
    next_requested       = Signal()
    flagged_requested    = Signal()
    revert_requested     = Signal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        mod = event.modifiers()
        no_mod = mod == Qt.KeyboardModifier.NoModifier
        ctrl   = mod == Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Return and no_mod:
            self.confirm_requested.emit()
            return
        if key == Qt.Key.Key_Up and ctrl:
            self.prev_requested.emit()
            return
        if key == Qt.Key.Key_Down and ctrl:
            self.next_requested.emit()
            return
        if key == Qt.Key.Key_F5:
            self.flagged_requested.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.revert_requested.emit()
            return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────
# 状态色标
# ─────────────────────────────────────────────────────────────

_STATUS_COLOR = {
    ProofStatus.OK:           "#4CAF50",
    ProofStatus.MODIFIED:     "#1a73e8",
    ProofStatus.AUTO_FLAGGED: "#FF9800",
    ProofStatus.UNCHECKED:    "#c8d0db",
}
_STATUS_LABEL = {
    ProofStatus.OK:           "已确认",
    ProofStatus.MODIFIED:     "已修改",
    ProofStatus.AUTO_FLAGGED: "疑点",
    ProofStatus.UNCHECKED:    "待确认",
}


def _status_dot(status: ProofStatus) -> str:
    color = _STATUS_COLOR.get(status, "#ccc")
    label = _STATUS_LABEL.get(status, "")
    return (
        f"<span style='display:inline-block;width:10px;height:10px;"
        f"border-radius:5px;background:{color};'></span> {label}"
    )


# ─────────────────────────────────────────────────────────────
# 上下文图像渲染
# ─────────────────────────────────────────────────────────────

def _render_context_stack(
    page_path: str,
    lines: List[Line],
    current_idx: int,       # 在 lines 列表中的索引
    display_width: int,
    cache: PageImageCache,
) -> Optional[QPixmap]:
    """将 current_idx ± CONTEXT_LINES 行裁图纵向堆叠，返回 QPixmap。"""
    img = cache.get_image(page_path)
    if img is None:
        return None
    H, W = img.shape[:2]

    # 收集要渲染的行索引
    rows: List[Tuple[int, bool]] = []  # (line_idx, is_current)
    for off in range(-CONTEXT_LINES, CONTEXT_LINES + 1):
        li = current_idx + off
        if 0 <= li < len(lines):
            rows.append((li, off == 0))

    if not rows:
        return None

    # 裁剪各行
    crops: List[Tuple[np.ndarray, bool]] = []
    for li, is_cur in rows:
        ln = lines[li]
        bb = ln.bbox
        x1 = max(0, bb.x)
        y1 = max(0, bb.y - ROW_PAD_Y)
        x2 = min(W, bb.x + bb.w)
        y2 = min(H, bb.y + bb.h + ROW_PAD_Y)
        if x2 <= x1 or y2 <= y1:
            crops.append((np.zeros((20, max(bb.w, 1), 3), np.uint8), is_cur))
        else:
            crops.append((img[y1:y2, x1:x2].copy(), is_cur))

    # 统一宽度（取最宽那行）
    max_w = max(c.shape[1] for c, _ in crops) if crops else 1
    padded: List[np.ndarray] = []
    for crop, is_cur in crops:
        h, w = crop.shape[:2]
        canvas = np.full((h, max_w, 3), 245, dtype=np.uint8)
        canvas[:h, :w] = crop
        if not is_cur:
            overlay = np.full_like(canvas, 245)
            canvas = cv2.addWeighted(canvas, 1 - DIM_ALPHA, overlay, DIM_ALPHA, 0)
        else:
            cv2.rectangle(canvas, (0, 0), (max_w - 1, h - 1), HIGHLIGHT_COLOR, 2)
        padded.append(canvas)

    # 纵向堆叠
    gap = np.full((GAP, max_w, 3), 245, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for i, p in enumerate(padded):
        if i:
            parts.append(gap)
        parts.append(p)
    stacked = np.vstack(parts)

    # 按显示宽度等比缩放
    sh, sw = stacked.shape[:2]
    if display_width > 10 and sw > 0:
        scale = display_width / sw
        new_h = max(1, int(sh * scale))
        stacked = cv2.resize(stacked, (display_width, new_h),
                             interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(np.ascontiguousarray(stacked), cv2.COLOR_BGR2RGB)
    rh, rw = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), rw, rh, rw * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ─────────────────────────────────────────────────────────────
# 横校面板主体
# ─────────────────────────────────────────────────────────────

class HProofPanel(QWidget):
    """横校（横向校对）面板：上图下文 + 上下文行 + 完整快捷键。"""

    proof_saved = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # items: (block, line, page, line_idx_in_block)
        self._items: List[Tuple[Block, Line, Page, int]] = []
        # 当前 block 内的所有行（用于上下文渲染）
        self._block_lines: List[Line] = []
        self._block_line_offset: int = 0  # _items 中该 block 起始 idx
        self._current_idx: int = 0
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        self._build_ui()

    # ─────────────────── 构建 UI ────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶部工具栏 ──────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setFixedHeight(46)
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 0, 12, 0)
        tl.setSpacing(6)

        title = QLabel("④ 横向校对")
        title.setObjectName("pageTitle")
        tl.addWidget(title)

        tl.addSpacing(16)

        self._btn_prev = QPushButton("⇡ 上一行")
        self._btn_next = QPushButton("⇣ 下一行")
        self._btn_ok   = QPushButton("✓ 确认")
        self._btn_ok.setObjectName("primaryBtn")
        self._btn_flag = QPushButton("⚑ 下一疑点  F5")
        self._btn_save = QPushButton("✎ 保存  Ctrl+S")

        for btn in (self._btn_prev, self._btn_next, self._btn_ok,
                    self._btn_flag, self._btn_save):
            btn.setMinimumHeight(30)
            tl.addWidget(btn)

        tl.addStretch()

        self._progress_lbl = QLabel("0 / 0")
        self._progress_lbl.setObjectName("muted")
        tl.addWidget(self._progress_lbl)

        self._conf_badge = ConfidenceBadge(1.0)
        tl.addWidget(self._conf_badge)

        # 状态色点
        self._status_dot = QLabel()
        self._status_dot.setTextFormat(Qt.TextFormat.RichText)
        tl.addWidget(self._status_dot)

        root.addWidget(toolbar)

        # ── 主体：上图下文 ───────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)

        # 上：上下文行堆叠图
        img_frame = QWidget()
        img_frame.setObjectName("card")
        img_frame.setStyleSheet("QWidget#card{background:#fafbfc; border:none;}")
        il = QVBoxLayout(img_frame)
        il.setContentsMargins(8, 8, 8, 8)

        self._context_label = QLabel("（尚未加载）")
        self._context_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._context_label.setScaledContents(False)
        self._context_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._context_label.setMinimumHeight(80)
        il.addWidget(self._context_label)

        splitter.addWidget(img_frame)

        # 下：文本编辑区
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(12, 8, 12, 8)
        bl.setSpacing(6)

        info_row = QHBoxLayout()
        self._line_label = QLabel("当前行：—")
        self._line_label.setObjectName("noteLabel")
        info_row.addWidget(self._line_label)
        info_row.addStretch()
        hint = QLabel("Enter 确认  Ctrl+↑/↓ 切行  F5 疑点  Esc 还原")
        hint.setObjectName("muted")
        info_row.addWidget(hint)
        bl.addLayout(info_row)

        self._text_edit = _ProofEdit()
        self._text_edit.setStyleSheet("font-size:18px; padding:8px 12px;")
        self._text_edit.setMaximumHeight(120)
        self._text_edit.confirm_requested.connect(self._mark_ok_and_next)
        self._text_edit.prev_requested.connect(self._prev)
        self._text_edit.next_requested.connect(self._next)
        self._text_edit.flagged_requested.connect(self._next_flagged)
        self._text_edit.revert_requested.connect(self._revert)
        bl.addWidget(self._text_edit)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("noteLabel")
        bl.addWidget(self._status_lbl)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # ── 信号 ────────────────────────────────────────────
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next.clicked.connect(self._next)
        self._btn_ok.clicked.connect(self._mark_ok_and_next)
        self._btn_flag.clicked.connect(self._next_flagged)
        self._btn_save.clicked.connect(self._save_current)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_current)

    # ─────────────────── 公共 API ───────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._items = []
        for page in pages:
            for block in page.text_blocks:
                for li, line in enumerate(block.lines):
                    self._items.append((block, line, page, li))
        self._current_idx = 0
        self._show_index(0)

    def reset(self) -> None:
        self._items.clear()
        self._block_lines.clear()
        self._current_idx = 0
        self._progress_lbl.setText("0 / 0")
        self._line_label.setText("当前行：—")
        self._text_edit.clear()
        self._context_label.clear()

    # ─────────────────── 内部逻辑 ───────────────────────────

    def _show_index(self, idx: int) -> None:
        if not self._items:
            self.reset()
            return
        idx = max(0, min(idx, len(self._items) - 1))
        # 先保存上一条
        if 0 <= self._current_idx < len(self._items) and self._current_idx != idx:
            self._save_current(emit=False, silent=True)
        self._current_idx = idx
        block, line, page, line_idx = self._items[idx]

        # 收集同 block 的所有行 + 当前行在其中的偏移
        block_items = [(j, ln) for j, (b, ln, _p, _li) in enumerate(self._items)
                       if b is block]
        self._block_lines = [ln for _, ln in block_items]
        pos_in_block = next(
            (k for k, (j, _) in enumerate(block_items) if j == idx), 0
        )

        # 渲染上下文图
        display_w = max(self._context_label.width() - 16, 200)
        pix = _render_context_stack(
            page.display_image_path,
            self._block_lines,
            pos_in_block,
            display_w,
            self._cache,
        )
        if pix:
            self._context_label.setPixmap(pix)
        else:
            self._context_label.setText("（无图像）")

        # 文本
        self._text_edit.blockSignals(True)
        self._text_edit.setPlainText(line.text)
        self._highlight_low_conf(line)
        self._text_edit.blockSignals(False)
        self._text_edit.setFocus()

        self._conf_badge.set_score(line.confidence)
        flagged = "⚑ " if line.proof_status == ProofStatus.AUTO_FLAGGED else ""
        self._line_label.setText(
            f"{flagged}第 {page.page_number} 页 / 块#{block.order} / 行 {line_idx + 1}"
        )
        self._progress_lbl.setText(f"{idx + 1} / {len(self._items)}")
        self._status_dot.setText(_status_dot(line.proof_status))
        self._status_lbl.setText("")

    def _highlight_low_conf(self, line: Line) -> None:
        if not line.chars:
            return
        red = QTextCharFormat()
        red.setBackground(QColor(255, 140, 0, 70))
        normal = QTextCharFormat()
        for i, ch in enumerate(line.chars):
            if i >= len(line.text):
                break
            cur = self._text_edit.textCursor()
            cur.setPosition(i)
            cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
            cur.setCharFormat(red if ch.confidence < LOW_CONF else normal)

    def _save_current(self, *, emit: bool = True, silent: bool = False) -> None:
        if not self._items:
            return
        _, line, page, _ = self._items[self._current_idx]
        new_text = self._text_edit.toPlainText()
        changed = new_text != line.text
        if changed:
            line.update_text(new_text)
            self._bus.publish(
                "line.proof_changed",
                page_id=page.id,
                line_id=line.id,
                status=line.proof_status.value,
            )
        if emit:
            self.proof_saved.emit()
        if not silent:
            self._status_lbl.setText("已保存" if changed else "无变更")

    def _revert(self) -> None:
        if not self._items:
            return
        _, line, _, _ = self._items[self._current_idx]
        original = line.original_text or line.ocr_text or line.text
        self._text_edit.blockSignals(True)
        self._text_edit.setPlainText(original)
        self._text_edit.blockSignals(False)
        self._status_lbl.setText("已还原到 OCR 原文")

    def _mark_ok_and_next(self) -> None:
        if not self._items:
            return
        self._save_current(emit=False, silent=True)
        _, line, page, _ = self._items[self._current_idx]
        line.proof_status = ProofStatus.OK
        self._bus.publish(
            "line.proof_changed",
            page_id=page.id,
            line_id=line.id,
            status=ProofStatus.OK.value,
        )
        self.proof_saved.emit()
        self._next()

    def _prev(self) -> None:
        self._show_index(self._current_idx - 1)

    def _next(self) -> None:
        self._show_index(self._current_idx + 1)

    def _next_flagged(self) -> None:
        if not self._items:
            return
        start = self._current_idx + 1
        for i in range(start, len(self._items)):
            if self._items[i][1].proof_status == ProofStatus.AUTO_FLAGGED:
                self._show_index(i)
                return
        # 从头找
        for i in range(0, start):
            if self._items[i][1].proof_status == ProofStatus.AUTO_FLAGGED:
                self._show_index(i)
                self._status_lbl.setText("（已回绕到开头）")
                return
        self._status_lbl.setText("无疑点行")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # 重新渲染以适应新宽度
        if self._items and 0 <= self._current_idx < len(self._items):
            _, line, page, _ = self._items[self._current_idx]
            block_items = [
                (j, ln) for j, (b, ln, _p, _li) in enumerate(self._items)
                if b is self._items[self._current_idx][0]
            ]
            block_lines = [ln for _, ln in block_items]
            pos = next(
                (k for k, (j, _) in enumerate(block_items) if j == self._current_idx), 0
            )
            display_w = max(self._context_label.width() - 16, 200)
            pix = _render_context_stack(
                page.display_image_path, block_lines, pos, display_w, self._cache
            )
            if pix:
                self._context_label.setPixmap(pix)
