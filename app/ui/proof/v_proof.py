"""纵校面板：上图下文，按整个 Block（列/段）导航。"""
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, Page, ProofStatus
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge


class VProofPanel(QWidget):
    """
    纵校（垂直校对）面板。
    - 上：当前块的裁剪图片
    - 下：对应整块文字，可整体编辑
    适合竖排文字或整段对比场景。
    """
    proof_saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blocks: List[Tuple[Block, Page]] = []
        self._current_idx: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        top = QHBoxLayout()
        top.setContentsMargins(12, 12, 12, 4)
        title = QLabel("⑤ 纵校")
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
        bl.addWidget(QLabel("块文字（可整体编辑）："))
        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:16px;")
        bl.addWidget(self._text_edit)

        btn_row = QHBoxLayout()
        self._btn_ok = QPushButton("✓ 确认本块")
        self._btn_ok.clicked.connect(self._mark_block_ok)
        self._btn_save = QPushButton("✎ 保存修改")
        self._btn_save.clicked.connect(self._save_current)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_save)
        bl.addLayout(btn_row)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 导航
        nav = QHBoxLayout()
        nav.setContentsMargins(12, 4, 12, 12)
        self._btn_prev = QPushButton("← 上一块")
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next = QPushButton("下一块 →")
        self._btn_next.clicked.connect(self._next)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:12px;")
        nav.addWidget(self._btn_prev)
        nav.addStretch()
        nav.addWidget(self._status_lbl)
        nav.addWidget(self._btn_next)
        layout.addLayout(nav)

    # ------------------------------------------------------------------ public

    def load_pages(self, pages: List[Page]) -> None:
        self._blocks = []
        for page in pages:
            for block in page.text_blocks:
                self._blocks.append((block, page))
        self._current_idx = 0
        self._show_block(0)

    # ------------------------------------------------------------------ private

    def _show_block(self, idx: int) -> None:
        if not self._blocks:
            return
        idx = max(0, min(idx, len(self._blocks) - 1))
        self._current_idx = idx
        block, page = self._blocks[idx]

        # 裁剪块图片
        img = cv2.imread(page.display_image_path)
        if img is not None:
            bb = block.bbox
            pad = 8
            x1 = max(0, bb.x - pad)
            y1 = max(0, bb.y - pad)
            x2 = min(img.shape[1], bb.x + bb.w + pad)
            y2 = min(img.shape[0], bb.y + bb.h + pad)
            crop = img[y1:y2, x1:x2]
            h, w, ch = crop.shape
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self._viewer.set_image_from_qimage(qimg)

        self._text_edit.setPlainText(block.full_text)
        self._conf_badge.set_score(block.avg_confidence)
        self._progress_lbl.setText(f"{idx+1} / {len(self._blocks)}")

    def _save_current(self) -> None:
        if not self._blocks:
            return
        block, _ = self._blocks[self._current_idx]
        new_full = self._text_edit.toPlainText()
        new_lines = new_full.splitlines()
        for i, line in enumerate(block.lines):
            new_text = new_lines[i] if i < len(new_lines) else ""
            if new_text != line.text:
                line.update_text(new_text)
        self.proof_saved.emit()
        self._status_lbl.setText("✎ 已保存")

    def _mark_block_ok(self) -> None:
        if not self._blocks:
            return
        block, _ = self._blocks[self._current_idx]
        for line in block.lines:
            line.proof_status = ProofStatus.OK
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 块已确认")

    def _prev(self) -> None:
        self._show_block(self._current_idx - 1)

    def _next(self) -> None:
        self._show_block(self._current_idx + 1)
