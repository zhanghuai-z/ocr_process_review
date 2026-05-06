"""版面分析面板：图像 + BBox 叠加可视化，属性侧边栏。"""
from __future__ import annotations
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from app.models import Block, Page
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge


class BlockPropertyPanel(QWidget):
    """侧边属性面板：显示选中 Block 的信息。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QLabel("块属性")
        hdr.setStyleSheet("font-weight:bold; font-size:13px;")
        layout.addWidget(hdr)

        self._type_lbl = QLabel("类型：—")
        self._conf_lbl = ConfidenceBadge(1.0)
        self._bbox_lbl = QLabel("位置：—")
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlaceholderText("（点击版面块查看文字内容）")
        self._text_edit.setMaximumHeight(200)

        for w in (self._type_lbl, self._conf_lbl, self._bbox_lbl):
            layout.addWidget(w)
        layout.addWidget(QLabel("内容预览："))
        layout.addWidget(self._text_edit)
        layout.addStretch()

    def show_block(self, block: Block) -> None:
        bb = block.bbox
        self._type_lbl.setText(f"类型：{block.block_type.value}")
        self._conf_lbl.set_score(block.avg_confidence)
        self._bbox_lbl.setText(f"位置：x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self._text_edit.setPlainText(block.full_text)

    def clear(self) -> None:
        self._type_lbl.setText("类型：—")
        self._bbox_lbl.setText("位置：—")
        self._text_edit.clear()


class LayoutPanel(QWidget):
    """
    步骤2: 版面分析结果可视化。
    左侧：页面缩略图列表；中间：图像+BBox；右侧：块属性。
    发出 analysis_confirmed 信号，触发 OCR 识别。
    """
    analysis_confirmed = Signal()
    page_selected = Signal(int)   # payload: page index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 标题行
        title_row = QHBoxLayout()
        title_lbl = QLabel("② 版面分析")
        title_lbl.setStyleSheet("font-size:18px; font-weight:bold; padding:12px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        self._status_lbl = QLabel("请先导入文件并运行版面分析")
        self._status_lbl.setStyleSheet("color:#aaa;")
        title_row.addWidget(self._status_lbl)
        main_layout.addLayout(title_row)

        # 主区域（三栏）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：页面列表
        self._page_list = QListWidget()
        self._page_list.setMaximumWidth(160)
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        splitter.addWidget(self._page_list)

        # 中：图像查看器
        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        splitter.addWidget(self._viewer)

        # 右：属性面板
        self._prop_panel = BlockPropertyPanel()
        self._prop_panel.setMinimumWidth(200)
        self._prop_panel.setMaximumWidth(300)
        splitter.addWidget(self._prop_panel)

        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 8)
        self._btn_run = QPushButton("运行版面分析")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._request_analysis)

        self._btn_next = QPushButton("开始 OCR 识别 →")
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self.analysis_confirmed)
        self._btn_next.setStyleSheet("font-size:14px; padding:8px;")

        btn_row.addWidget(self._btn_run)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_next)
        main_layout.addLayout(btn_row)

    # ------------------------------------------------------------------ public

    def set_pages(self, pages: List[Page]) -> None:
        """设置待分析的页面（已加载图片路径）。"""
        self._pages = pages
        self._page_list.clear()
        for i, page in enumerate(pages):
            from pathlib import Path
            self._page_list.addItem(f"第 {page.page_number} 页\n{Path(page.image_path).name}")
        self._btn_run.setEnabled(bool(pages))
        if pages:
            self._page_list.setCurrentRow(0)

    def show_analysis_result(self, pages: List[Page]) -> None:
        """版面分析完成后，更新显示（保持当前选中页）。"""
        self._pages = pages
        # 保持当前页索引，不跳回第 0 页
        current_idx = min(self._current_page_idx, len(pages) - 1)
        self._update_viewer(current_idx)
        self._btn_next.setEnabled(True)
        total_blocks = sum(len(p.blocks) for p in pages)
        self._status_lbl.setText(f"共 {len(pages)} 页，{total_blocks} 个版面块")

    # ------------------------------------------------------------------ private

    def _request_analysis(self) -> None:
        # 由主窗口连接到 LayoutAnalyzer Worker
        pass

    def _on_page_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._pages):
            self._current_page_idx = idx
            self._update_viewer(idx)
            self.page_selected.emit(idx)

    def _update_viewer(self, idx: int) -> None:
        if not self._pages:
            return
        page = self._pages[idx]
        self._viewer.set_image(page.image_path)
        if page.is_analyzed:
            self._viewer.show_blocks(page.blocks)
        self._prop_panel.clear()

    def _on_block_clicked(self, block: Block) -> None:
        self._prop_panel.show_block(block)

    @property
    def run_button(self) -> QPushButton:
        return self._btn_run
