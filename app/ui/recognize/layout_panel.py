"""版面分析面板：图像 + BBox 叠加可视化，块信息内嵌底部栏。"""
from __future__ import annotations
from typing import List, Optional
import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from app.models import BBox, Block, BlockSource, BlockType, Page
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge


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
        title_lbl.setObjectName("pageTitle"); title_lbl.setStyleSheet("padding:12px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        self._status_lbl = QLabel("请先导入文件并运行版面分析")
        self._status_lbl.setObjectName("muted")
        title_row.addWidget(self._status_lbl)
        main_layout.addLayout(title_row)

        # 主区域（两栏：页面列表 + 图像查看器）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：页面列表（缩窄）
        self._page_list = QListWidget()
        self._page_list.setMaximumWidth(110)
        self._page_list.setMinimumWidth(60)
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        splitter.addWidget(self._page_list)

        # 中：图像查看器
        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        self._viewer.block_moved.connect(self._on_block_moved)
        self._viewer.block_created.connect(self._on_block_created)
        splitter.addWidget(self._viewer)

        # 左:中 = 1:9
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)
        splitter.setSizes([90, 9999])
        main_layout.addWidget(splitter)

        # 底部工具栏（按钮 + 属性信息内嵌）
        bottom = QFrame()
        bottom.setObjectName("toolbar")
        bottom.setFixedHeight(46)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(12, 0, 12, 0)
        bottom_layout.setSpacing(8)

        self._btn_run = QPushButton("▶ 分析")
        self._btn_run.setToolTip("运行版面分析")
        self._btn_run.setEnabled(False)
        self._btn_run.setObjectName("ghostBtn")
        self._btn_run.clicked.connect(self._request_analysis)

        self._btn_edit = QPushButton("✎ 编辑框")
        self._btn_edit.setToolTip("开启后可拖动/缩放 BBox，右键拖拽新建框")
        self._btn_edit.setCheckable(True)
        self._btn_edit.setObjectName("ghostBtn")
        self._btn_edit.toggled.connect(self._viewer.set_edit_mode)

        self._btn_next = QPushButton("→ 开始 OCR")
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self.analysis_confirmed)
        self._btn_next.setObjectName("primaryBtn")
        self._btn_next.setMinimumHeight(34)

        bottom_layout.addWidget(self._btn_run)
        bottom_layout.addWidget(self._btn_edit)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color:#e3e8ef; margin:8px 4px;")
        bottom_layout.addWidget(sep)

        # 块属性内嵌标签
        self._prop_type = QLabel("—")
        self._prop_type.setObjectName("fieldLabel")
        self._prop_bbox = QLabel("")
        self._prop_bbox.setObjectName("muted")
        self._prop_conf = ConfidenceBadge(1.0)
        self._prop_conf.hide()
        bottom_layout.addWidget(self._prop_type)
        bottom_layout.addWidget(self._prop_bbox)
        bottom_layout.addWidget(self._prop_conf)

        bottom_layout.addStretch()
        bottom_layout.addWidget(self._btn_next)
        main_layout.addWidget(bottom)

    # ------------------------------------------------------------------ public

    def set_pages(self, pages: List[Page]) -> None:
        """设置待分析的页面（已加载图片路径）。"""
        self._pages = pages
        self._page_list.clear()
        for i, page in enumerate(pages):
            from pathlib import Path
            self._page_list.addItem(
                f"第 {page.page_number} 页\n{Path(page.source_path or page.image_path).name}"
            )
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
        self._viewer.set_image(page.display_image_path)
        if page.is_analyzed:
            self._viewer.show_blocks(page.blocks)
        # 清空底部属性栏
        self._prop_type.setText("—")
        self._prop_bbox.setText("")
        self._prop_conf.hide()

    def _on_block_clicked(self, block: Block) -> None:
        bb = block.bbox
        self._prop_type.setText(f"[{block.block_type.value}]")
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self._prop_conf.set_score(block.avg_confidence)
        self._prop_conf.show()

    def _on_block_moved(self, block: Block) -> None:
        bb = block.bbox
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")

    def _on_block_created(self, bbox: BBox) -> None:
        """右键拖拽创建新 Block，添加到当前页并刷新画布。"""
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        new_block = Block(
            block_type=BlockType.TEXT,
            bbox=bbox,
            source=BlockSource.MANUAL_DRAW,
        )
        page.blocks.append(new_block)
        self._viewer.show_blocks(page.blocks)

    @property
    def run_button(self) -> QPushButton:
        return self._btn_run
