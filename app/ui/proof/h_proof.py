"""横校面板：左图右文，逐行校对，低置信字符红底高亮。"""
from __future__ import annotations
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from app.models import Line, Page, ProofStatus
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge

# 低置信度阈值
LOW_CONFIDENCE = 0.80


class HProofPanel(QWidget):
    """
    横校（水平校对）面板。
    - 左：当前行切图
    - 右：可编辑文字（低置信字符红底标记）
    - 导航：上一行 / 下一行 / 跳至下一可疑
    发出 proof_saved 信号（Line 已更新）。
    """
    proof_saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_lines: List[tuple[Line, Page]] = []  # (line, page)
        self._current_idx: int = 0
        self._flagged_indices: List[int] = []
        self._flagged_ptr: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        top = QHBoxLayout()
        top.setContentsMargins(12, 12, 12, 4)
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

        # 主体
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：行图
        self._viewer = ImageViewer()
        self._viewer.setMinimumWidth(300)
        splitter.addWidget(self._viewer)

        # 右：编辑
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        info = QLabel("识别文字（可直接编辑修改）：")
        info.setStyleSheet("color:#aaa; font-size:12px;")
        rl.addWidget(info)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:20px; line-height:1.6;")
        rl.addWidget(self._text_edit)

        # 状态按钮
        btn_row = QHBoxLayout()
        self._btn_ok = QPushButton("✓ 确认正确")
        self._btn_ok.clicked.connect(self._mark_ok)
        self._btn_save = QPushButton("✎ 保存修改")
        self._btn_save.clicked.connect(self._save_current)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_save)
        rl.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 导航栏
        nav = QHBoxLayout()
        nav.setContentsMargins(12, 4, 12, 12)
        self._btn_prev = QPushButton("← 上一行")
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next_flagged = QPushButton("⚠ 下一可疑")
        self._btn_next_flagged.clicked.connect(self._next_flagged)
        self._btn_next = QPushButton("下一行 →")
        self._btn_next.clicked.connect(self._next)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:12px;")

        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next_flagged)
        nav.addStretch()
        nav.addWidget(self._status_lbl)
        nav.addWidget(self._btn_next)
        layout.addLayout(nav)

    # ------------------------------------------------------------------ public

    def load_pages(self, pages: List[Page]) -> None:
        """加载所有页面，展平所有行，构建可疑索引。"""
        self._all_lines = []
        for page in pages:
            for block in page.text_blocks:
                for line in block.lines:
                    self._all_lines.append((line, page))

        self._flagged_indices = [
            i for i, (l, _) in enumerate(self._all_lines)
            if l.proof_status == ProofStatus.AUTO_FLAGGED
        ]
        self._flagged_ptr = 0
        self._current_idx = 0
        self._show_line(0)
        self._update_nav()

    # ------------------------------------------------------------------ private

    def _show_line(self, idx: int) -> None:
        if not self._all_lines:
            return
        idx = max(0, min(idx, len(self._all_lines) - 1))
        self._current_idx = idx
        line, page = self._all_lines[idx]

        # 从原图裁剪行区域
        import cv2
        import numpy as np
        from PySide6.QtGui import QImage

        img = cv2.imread(page.display_image_path)
        if img is not None:
            bb = line.bbox
            pad = 4
            x1 = max(0, bb.x - pad)
            y1 = max(0, bb.y - pad)
            x2 = min(img.shape[1], bb.x + bb.w + pad)
            y2 = min(img.shape[0], bb.y + bb.h + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                h, w = crop.shape[:2]
                # 用 np.ascontiguousarray 确保内存连续，tobytes() 确保 QImage 持有独立副本
                rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
                self._viewer.set_image_from_qimage(qimg)

        # 显示文字
        self._text_edit.setPlainText(line.text)
        self._conf_badge.set_score(line.confidence)

        # 标红低置信字符
        if line.chars:
            self._highlight_chars(line)

        total = len(self._all_lines)
        flagged = len(self._flagged_indices)
        self._progress_lbl.setText(
            f"{idx+1} / {total}  (可疑: {flagged})"
        )
        status_map = {
            ProofStatus.UNCHECKED: "",
            ProofStatus.AUTO_FLAGGED: "⚠ 低置信度",
            ProofStatus.MODIFIED: "✎ 已修改",
            ProofStatus.OK: "✓ 已确认",
        }
        self._status_lbl.setText(status_map.get(line.proof_status, ""))

    def _highlight_chars(self, line: Line) -> None:
        """将低置信字符背景标红。"""
        cursor = self._text_edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        fmt_normal = QTextCharFormat()
        cursor.setCharFormat(fmt_normal)

        fmt_bad = QTextCharFormat()
        fmt_bad.setBackground(QColor(0xF4, 0x43, 0x36, 160))
        fmt_bad.setForeground(QColor(0xFF, 0xFF, 0xFF))

        for i, char in enumerate(line.chars):
            if char.confidence < LOW_CONFIDENCE:
                cur = self._text_edit.textCursor()
                cur.setPosition(i)
                cur.movePosition(QTextCursor.MoveOperation.NextCharacter,
                                 QTextCursor.MoveMode.KeepAnchor)
                cur.setCharFormat(fmt_bad)

    def _save_current(self) -> None:
        if not self._all_lines:
            return
        line, _ = self._all_lines[self._current_idx]
        new_text = self._text_edit.toPlainText()
        if new_text != line.text:
            line.update_text(new_text)
            self.proof_saved.emit()
        self._status_lbl.setText("✎ 已保存")

    def _mark_ok(self) -> None:
        if not self._all_lines:
            return
        line, _ = self._all_lines[self._current_idx]
        line.proof_status = ProofStatus.OK
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 已确认")

    def _prev(self) -> None:
        self._show_line(self._current_idx - 1)

    def _next(self) -> None:
        self._show_line(self._current_idx + 1)

    def _next_flagged(self) -> None:
        if not self._flagged_indices:
            return
        self._flagged_ptr = (self._flagged_ptr + 1) % len(self._flagged_indices)
        self._show_line(self._flagged_indices[self._flagged_ptr])

    def _update_nav(self) -> None:
        has = bool(self._all_lines)
        self._btn_prev.setEnabled(has)
        self._btn_next.setEnabled(has)
        self._btn_next_flagged.setEnabled(bool(self._flagged_indices))
