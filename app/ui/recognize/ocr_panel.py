"""OCR 识别面板：进度条 + 结果树状展示。"""
from __future__ import annotations
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from app.core.block_attributes import block_display_label
from app.core.proof_line_facts import proof_line_facts
from app.models import Block, Page, ProofStatus
from app.models.layout_projection import page_layout_blocks
from app.models.ocr_observation import block_avg_confidence, block_ocr_line_count, block_ocr_lines
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge


class OcrPanel(QWidget):
    """
    步骤3: OCR 识别结果展示。
    左：图像预览；右：结果树（Block → Line → Char）。

    信号：
    - go_to_proof_requested: 用户点击"进入校对"按钮（纯导航意图）
      （注意：不要在按钮点击中重新发射 OCR 完成信号）
    """
    go_to_proof_requested = Signal()   # 纯导航意图

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题 + 进度
        top = QHBoxLayout()
        top.setContentsMargins(12, 12, 12, 4)
        title = QLabel("③ OCR 识别")
        title.setObjectName("pageTitle")
        top.addWidget(title)
        top.addStretch()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(200)
        self._progress.setFixedHeight(16)
        self._progress.setFormat("%p%")
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        top.addWidget(self._progress)
        layout.addLayout(top)

        # 主体：左图右树
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._viewer = ImageViewer()
        splitter.addWidget(self._viewer)

        # 结果树
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 4, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["内容", "置信度", "状态"])
        self._tree.setColumnWidth(0, 300)
        self._tree.setColumnWidth(1, 70)
        self._tree.currentItemChanged.connect(self._on_item_selected)
        right_layout.addWidget(self._tree)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 底部
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 8)
        self._status_lbl = QLabel("等待识别…")
        self._status_lbl.setObjectName("muted")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        self._btn_next = QPushButton("进入校对 →")
        self._btn_next.setEnabled(False)
        self._btn_next.setObjectName("primaryBtn"); self._btn_next.setMinimumHeight(34)
        self._btn_next.clicked.connect(self.go_to_proof_requested.emit)
        btn_row.addWidget(self._btn_next)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ public

    def set_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        if pages:
            self._viewer.set_image(pages[0].display_image_path)

    def on_progress(self, page_idx: int, total: int) -> None:
        self._progress.setVisible(True)
        pct = int((page_idx + 1) / total * 100)
        self._progress.setValue(pct)
        self._progress.setFormat("%p%")
        self._status_lbl.setText("识别中…")

    def on_recognition_complete(self, pages: List[Page]) -> None:
        self._pages = pages
        self._progress.setVisible(False)
        self._populate_tree(pages)
        flagged = sum(
            1 for p in pages for b in page_layout_blocks(p)
            for l in block_ocr_lines(b) if proof_line_facts(l).status == ProofStatus.AUTO_FLAGGED
        )
        total_lines = sum(block_ocr_line_count(b) for p in pages for b in page_layout_blocks(p))
        self._status_lbl.setText(
            f"识别完成：{total_lines} 行，其中 {flagged} 行置信度偏低（已自动标记）"
        )
        self._btn_next.setEnabled(True)

    # ------------------------------------------------------------------ private

    def _populate_tree(self, pages: List[Page]) -> None:
        self._tree.clear()
        for page in pages:
            page_item = QTreeWidgetItem(self._tree, [f"第 {page.page_number} 页", "", ""])
            page_item.setData(0, Qt.ItemDataRole.UserRole, page)
            for block in page_layout_blocks(page):
                block_item = QTreeWidgetItem(
                    page_item,
                    [f"[{block_display_label(block)}]", f"{block_avg_confidence(block):.2f}", ""],
                )
                block_item.setData(0, Qt.ItemDataRole.UserRole, block)
                for line in block_ocr_lines(block):
                    facts = proof_line_facts(line)
                    line_text = facts.text
                    preview = line_text[:40] + ("…" if len(line_text) > 40 else "")
                    status_str = {
                        ProofStatus.UNCHECKED: "",
                        ProofStatus.AUTO_FLAGGED: "⚠ 低置信",
                        ProofStatus.MODIFIED: "✎ 已修改",
                        ProofStatus.OK: "✓ 确认",
                    }.get(facts.status, "")
                    line_item = QTreeWidgetItem(
                        block_item,
                        [preview, f"{facts.confidence:.2f}", status_str],
                    )
                    line_item.setData(0, Qt.ItemDataRole.UserRole, line)
                    if facts.status == ProofStatus.AUTO_FLAGGED:
                        line_item.setForeground(1, Qt.GlobalColor.red)
            page_item.setExpanded(True)

    def _on_item_selected(self, current: QTreeWidgetItem, _) -> None:
        if current is None:
            return
        obj = current.data(0, Qt.ItemDataRole.UserRole)
        from app.models import Line, Page as PageModel
        if isinstance(obj, PageModel):
            self._viewer.set_image(obj.display_image_path)
            self._viewer.show_blocks(page_layout_blocks(obj))
        elif isinstance(obj, Block):
            # 找到对应页面
            for page in self._pages:
                if obj in page_layout_blocks(page):
                    self._viewer.set_image(page.display_image_path)
                    self._viewer.show_blocks([obj])
                    break
