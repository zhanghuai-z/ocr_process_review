"""横校面板：上图下文，带上下文行。

设计：
- 上半部分：纵向堆叠 5 行图像（前2 + 当前 + 后2），当前行高亮，其它灰化
- 下半部分：当前行的可编辑文字，低置信度字符红底高亮
- 键盘：↑/↓ 切换上一/下一条；Enter 确认本行并跳到下一条
- 鼠标点击堆叠图中的某行，可直接切换到该行
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import (
    QColor, QImage, QKeySequence, QShortcut, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, Line, Page, ProofStatus
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge

CONTEXT_BEFORE = 2
CONTEXT_AFTER = 2
ROW_PADDING = 6                 # 每行裁剪上下额外像素
LOW_CONF_THRESHOLD = 0.80
GAP_BETWEEN_ROWS = 8            # 堆叠行间距
HIGHLIGHT_BORDER = (255, 87, 34)  # 当前行红色边框 (RGB)
DIM_OVERLAY_ALPHA = 0.55          # 上下文行灰化透明度


class HProofPanel(QWidget):
    """横校（横向校对）面板：上图下文 + 上下文。"""

    proof_saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 全部行（含 block 引用，便于取上下文）
        self._items: List[Tuple[Block, Line, Page, int]] = []
        # 索引：(block_idx_in_items list-of-block, line_idx_in_block)
        self._current_idx: int = 0
        # 记录当前堆叠中点击区域 -> 在 _items 中的索引
        self._stack_hit_zones: List[Tuple[int, int, int]] = []
        self._build_ui()
        self._install_shortcuts()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题行
        top = QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 4)
        title = QLabel("④ 横校")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        top.addWidget(title)
        top.addStretch()
        self._progress_lbl = QLabel("0 / 0")
        self._progress_lbl.setStyleSheet("color:#aaa;")
        top.addWidget(self._progress_lbl)
        self._conf_badge = ConfidenceBadge(1.0)
        top.addWidget(self._conf_badge)
        layout.addLayout(top)

        # 主体：上图下文
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._viewer = ImageViewer()
        splitter.addWidget(self._viewer)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(8, 4, 8, 4)
        self._line_label = QLabel("当前行：—")
        self._line_label.setStyleSheet("color:#aaa; font-size:12px;")
        bl.addWidget(self._line_label)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:18px;")
        self._text_edit.setMaximumHeight(110)
        bl.addWidget(self._text_edit)

        btn_row = QHBoxLayout()
        self._btn_prev = QPushButton("← 上一条 (↑)")
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next = QPushButton("下一条 (↓) →")
        self._btn_next.clicked.connect(self._next)
        self._btn_ok = QPushButton("✓ 确认 (Enter)")
        self._btn_ok.clicked.connect(self._mark_ok_and_next)
        self._btn_save = QPushButton("✎ 保存修改")
        self._btn_save.clicked.connect(self._save_current)
        self._btn_next_flag = QPushButton("⚑ 下一标记")
        self._btn_next_flag.clicked.connect(self._next_flagged)
        for b in (self._btn_prev, self._btn_ok, self._btn_save, self._btn_next, self._btn_next_flag):
            btn_row.addWidget(b)
        btn_row.addStretch()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:12px;")
        btn_row.addWidget(self._status_lbl)
        bl.addLayout(btn_row)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _install_shortcuts(self) -> None:
        # 全局快捷键
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, activated=self._prev)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, activated=self._next)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._mark_ok_and_next)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._mark_ok_and_next)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_current)

    # ------------------------------------------------------------------ public

    def load_pages(self, pages: List[Page]) -> None:
        self._items = []
        for page in pages:
            for block in page.text_blocks:
                for li, line in enumerate(block.lines):
                    self._items.append((block, line, page, li))
        self._current_idx = 0
        self._show_index(0)

    # ------------------------------------------------------------------ private

    def _show_index(self, idx: int) -> None:
        if not self._items:
            self._progress_lbl.setText("0 / 0")
            self._line_label.setText("（无可校对行）")
            self._text_edit.clear()
            self._viewer.clear_overlays()
            return
        idx = max(0, min(idx, len(self._items) - 1))
        # 保存上一条修改
        if 0 <= self._current_idx < len(self._items) and self._current_idx != idx:
            self._save_current(emit=False, silent=True)
        self._current_idx = idx
        block, line, page, line_idx = self._items[idx]

        # 构造堆叠图
        self._render_stacked(idx)
        # 文本
        self._text_edit.blockSignals(True)
        self._text_edit.setPlainText(line.text)
        self._highlight_low_conf_chars(line)
        self._text_edit.blockSignals(False)
        self._text_edit.setFocus()
        self._conf_badge.set_score(line.confidence)
        flagged = "⚑ " if line.proof_status == ProofStatus.AUTO_FLAGGED else ""
        self._line_label.setText(
            f"{flagged}第 {page.page_number} 页 / 块#{block.order} / 行 {line_idx+1}"
        )
        self._progress_lbl.setText(f"{idx+1} / {len(self._items)}")

    def _highlight_low_conf_chars(self, line: Line) -> None:
        """对低置信度字符用红底高亮。"""
        if not line.chars:
            return
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        normal = QTextCharFormat()
        red = QTextCharFormat()
        red.setBackground(QColor(255, 87, 34, 90))
        # 简单按位置遍历（chars 数量可能 < text 长度时走位置）
        for i, ch in enumerate(line.chars):
            if i >= len(line.text):
                break
            cur = self._text_edit.textCursor()
            cur.setPosition(i)
            cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
            if ch.confidence < LOW_CONF_THRESHOLD:
                cur.setCharFormat(red)
            else:
                cur.setCharFormat(normal)

    def _render_stacked(self, idx: int) -> None:
        """渲染当前行 ± 上下文 N 行，纵向堆叠成一张图。"""
        block, _, page, _ = self._items[idx]
        # 当前 block 内的行索引范围（不跨 block 取上下文）
        current_block_items: List[Tuple[int, Line]] = []
        for j, (b, ln, _p, li) in enumerate(self._items):
            if b is block:
                current_block_items.append((j, ln))
        # 当前在 block 中位置
        pos_in_block = next(
            (k for k, (j, _) in enumerate(current_block_items) if j == idx),
            0,
        )
        start = max(0, pos_in_block - CONTEXT_BEFORE)
        end = min(len(current_block_items), pos_in_block + CONTEXT_AFTER + 1)
        window = current_block_items[start:end]

        img = cv2.imread(page.display_image_path)
        if img is None:
            self._viewer.clear_overlays()
            return
        H, W = img.shape[:2]

        # 裁剪每行
        rows = []  # list of (idx_in_items, np.ndarray BGR, is_current)
        for global_idx, ln in window:
            bb = ln.bbox
            x1 = max(0, bb.x - ROW_PADDING)
            y1 = max(0, bb.y - ROW_PADDING)
            x2 = min(W, bb.x + bb.w + ROW_PADDING)
            y2 = min(H, bb.y + bb.h + ROW_PADDING)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2].copy()
            rows.append((global_idx, crop, global_idx == idx))

        if not rows:
            self._viewer.clear_overlays()
            return

        # 统一宽度（按最大宽度 padding）
        max_w = max(c.shape[1] for _, c, _ in rows)
        padded_rows = []
        for global_idx, crop, is_cur in rows:
            h, w = crop.shape[:2]
            if w < max_w:
                pad = np.full((h, max_w - w, 3), 32, dtype=np.uint8)  # 深灰填充
                crop = np.hstack([crop, pad])
            if not is_cur:
                # 上下文行灰化
                gray_overlay = np.full_like(crop, 32)
                crop = cv2.addWeighted(crop, 1 - DIM_OVERLAY_ALPHA, gray_overlay, DIM_OVERLAY_ALPHA, 0)
            else:
                # 当前行：加红色边框
                cv2.rectangle(
                    crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1),
                    (HIGHLIGHT_BORDER[2], HIGHLIGHT_BORDER[1], HIGHLIGHT_BORDER[0]),
                    3,
                )
            padded_rows.append((global_idx, crop, is_cur))

        # 拼接（带间隔条）
        gap = np.full((GAP_BETWEEN_ROWS, max_w, 3), 24, dtype=np.uint8)
        composed_parts = []
        self._stack_hit_zones = []
        cur_y = 0
        for k, (global_idx, crop, is_cur) in enumerate(padded_rows):
            composed_parts.append(crop)
            self._stack_hit_zones.append((cur_y, cur_y + crop.shape[0], global_idx))
            cur_y += crop.shape[0]
            if k < len(padded_rows) - 1:
                composed_parts.append(gap)
                cur_y += GAP_BETWEEN_ROWS
        composed = np.vstack(composed_parts)

        # 转 QImage 显示
        h, w = composed.shape[:2]
        rgb = cv2.cvtColor(np.ascontiguousarray(composed), cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._viewer.set_image_from_qimage(qimg)

    def _save_current(self, emit: bool = True, silent: bool = False) -> None:
        if not self._items:
            return
        _, line, _, _ = self._items[self._current_idx]
        new_text = self._text_edit.toPlainText()
        if new_text != line.text:
            line.update_text(new_text)
            if emit:
                self.proof_saved.emit()
            if not silent:
                self._status_lbl.setText("✎ 已保存")

    def _mark_ok_and_next(self) -> None:
        if not self._items:
            return
        self._save_current(emit=False, silent=True)
        _, line, _, _ = self._items[self._current_idx]
        line.proof_status = ProofStatus.OK
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 已确认")
        self._next()

    def _prev(self) -> None:
        self._show_index(self._current_idx - 1)

    def _next(self) -> None:
        self._show_index(self._current_idx + 1)

    def _next_flagged(self) -> None:
        if not self._items:
            return
        n = len(self._items)
        for k in range(1, n + 1):
            j = (self._current_idx + k) % n
            _, ln, _, _ = self._items[j]
            if ln.proof_status == ProofStatus.AUTO_FLAGGED:
                self._show_index(j)
                return
        self._status_lbl.setText("（无更多标记行）")

    # ---- 鼠标点击堆叠图切换行 ----
    def mousePressEvent(self, event):
        # 把 viewer 上的点击映射到行
        if event.button() == Qt.MouseButton.LeftButton and self._stack_hit_zones:
            global_pos = event.globalPosition().toPoint()
            view_pos = self._viewer.mapFromGlobal(global_pos)
            if self._viewer.rect().contains(view_pos):
                scene_pos = self._viewer.mapToScene(view_pos)
                y = scene_pos.y()
                for top, bottom, gi in self._stack_hit_zones:
                    if top <= y < bottom:
                        self._show_index(gi)
                        break
        super().mousePressEvent(event)
