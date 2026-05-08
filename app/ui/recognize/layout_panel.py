"""版面分析面板：图像 + BBox 叠加可视化，块信息内嵌底部栏。"""
from __future__ import annotations
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from app.models import BBox, Block, BlockSource, BlockType, Page
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge


class LayoutPanel(QWidget):
    """
    步骤2: 版面分析结果可视化。
    左侧：页面缩略图列表；中间：图像+BBox；右侧：块属性。
    analysis_confirmed 信号由 show_analysis_result() 自动发出，触发 OCR 识别。
    """
    analysis_confirmed = Signal()
    page_selected = Signal(int)   # payload: page index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._selected_block: Optional[Block] = None
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 标题行（含 ▶ 分析 按钮 + 进度条）
        title_row = QHBoxLayout()
        title_row.setContentsMargins(12, 6, 12, 0)

        self._btn_run = QPushButton("▶")
        self._btn_run.setToolTip("运行版面分析")
        self._btn_run.setEnabled(False)
        self._btn_run.setObjectName("runBtn")
        self._btn_run.setFixedSize(36, 36)
        self._btn_run.clicked.connect(self._request_analysis)
        title_row.addWidget(self._btn_run)

        # 进度条（默认隐藏，分析期间显示不确定动画）
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # 不确定模式
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()
        title_row.addWidget(self._progress_bar, 1)

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

        # 中：图像查看器（始终可编辑）
        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        self._viewer.block_moved.connect(self._on_block_moved)
        self._viewer.block_created.connect(self._on_block_created)
        self._viewer.block_deleted.connect(self._on_block_deleted)
        splitter.addWidget(self._viewer)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)
        splitter.setSizes([90, 9999])
        main_layout.addWidget(splitter)

        # 底部属性栏（Delete 键删除；类型操作分两组）
        bottom = QFrame()
        bottom.setObjectName("toolbar")
        bottom.setFixedHeight(46)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(12, 0, 12, 0)
        bl.setSpacing(8)

        # 新建框类型
        bl.addWidget(QLabel("新建:"))
        self._new_type_combo = QComboBox()
        self._new_type_combo.setMinimumWidth(90)
        for bt in BlockType:
            self._new_type_combo.addItem(bt.value, bt)
        bl.addWidget(self._new_type_combo)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setFrameShadow(QFrame.Shadow.Sunken)
        sep0.setStyleSheet("color:#e3e8ef; margin:8px 4px;")
        bl.addWidget(sep0)

        # 选中块类型
        bl.addWidget(QLabel("选中:"))
        self._type_combo = QComboBox()
        self._type_combo.setEnabled(False)
        self._type_combo.setMinimumWidth(90)
        for bt in BlockType:
            self._type_combo.addItem(bt.value, bt)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        bl.addWidget(self._type_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color:#e3e8ef; margin:8px 4px;")
        bl.addWidget(sep)

        # bbox / conf 属性标签
        self._prop_bbox = QLabel("")
        self._prop_bbox.setObjectName("muted")
        self._prop_conf = ConfidenceBadge(1.0)
        self._prop_conf.hide()
        bl.addWidget(self._prop_bbox)
        bl.addWidget(self._prop_conf)

        bl.addStretch()
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
        """版面分析完成后，更新显示（保持当前选中页）并自动触发 OCR 流程。"""
        self._progress_bar.hide()
        self._pages = pages
        current_idx = min(self._current_page_idx, len(pages) - 1)
        self._update_viewer(current_idx)
        total_blocks = sum(len(p.blocks) for p in pages)
        self._status_lbl.setText(f"共 {len(pages)} 页，{total_blocks} 个版面块")

    # ------------------------------------------------------------------ private

    def _request_analysis(self) -> None:
        """触发版面分析（由主窗口的 run_button.clicked 同时连接），显示进度条。"""
        self._progress_bar.show()
        self._status_lbl.setText("正在分析…")

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
        self._selected_block = None
        self._type_combo.setEnabled(False)
        self._prop_bbox.setText("")
        self._prop_conf.hide()

    def _on_block_clicked(self, block: Block) -> None:
        self._selected_block = block
        bb = block.bbox
        self._type_combo.setEnabled(True)
        # 同步类型下拉到当前块
        self._type_combo.blockSignals(True)
        for i in range(self._type_combo.count()):
            if self._type_combo.itemData(i) == block.block_type:
                self._type_combo.setCurrentIndex(i)
                break
        self._type_combo.blockSignals(False)
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self._prop_conf.set_score(block.avg_confidence)
        self._prop_conf.show()

    def _on_block_moved(self, block: Block) -> None:
        bb = block.bbox
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")

    def _on_block_created(self, bbox: BBox) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        bt = self._new_type_combo.currentData() or BlockType.TEXT
        new_block = Block(
            block_type=bt,
            bbox=bbox,
            source=BlockSource.MANUAL_DRAW,
        )
        page.blocks.append(new_block)
        self._viewer.show_blocks(page.blocks)

    def _on_block_deleted(self, block: Block) -> None:
        """viewer 键盘 Delete 已删除框 → 从 page 数据中移除。"""
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        page.blocks = [b for b in page.blocks if b is not block]
        if self._selected_block is block:
            self._selected_block = None
            self._type_combo.setEnabled(False)
            self._prop_bbox.setText("")
            self._prop_conf.hide()

    def _delete_selected(self) -> None:
        """底部栏 ✕ 删除框 按钮。"""
        self._viewer.delete_selected()

    def _on_type_changed(self, _index: int) -> None:
        if self._selected_block is None:
            return
        new_type: BlockType = self._type_combo.currentData()
        if new_type:
            self._selected_block.block_type = new_type

    @property
    def run_button(self) -> QPushButton:
        return self._btn_run
