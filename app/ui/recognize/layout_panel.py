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
from app.ui.widgets.block_inspector import BlockInspector
from app.core.proof_state_bus import ProofStateBus
from app.ui.widgets.confidence_badge import ConfidenceBadge


class LayoutPanel(QWidget):
    """
    步骤2: 版面分析结果可视化。
    左侧：页面缩略图列表；中间：图像+BBox；右侧：块属性。
    analysis_confirmed 信号由 show_analysis_result() 自动发出，触发 OCR 识别。
    """
    analysis_confirmed = Signal()
    page_selected = Signal(int)   # payload: page_number
    geometry_changed = Signal()
    block_contract_changed = Signal(int, str)  # page_number, change_kind
    ocr_entry_requested = Signal(str, int)  # source, page_number
    page_completed = Signal(int)  # 用户点「完成」时（payload: page idx）
    edits_cancelled = Signal(int) # 用户点「取消」时（payload: page idx）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._selected_block: Optional[Block] = None
        self._page_gate_states: dict[int, tuple[str, bool, str, str]] = {}
        self._primary_actions: dict[int, tuple[str, str, bool]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # run 按钮 / 进度条 / 状态标签 仍然保留，但不再占用顶栏空间：
        #   run 按钮：隐藏（由 TopBar 的「运行版面分析」按钮接管）
        #   progress / status：移动到底栏（见下文 bottom bar 区段）
        self._btn_run = QPushButton("▶")
        self._btn_run.setObjectName("runBtn")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._request_analysis)
        self._btn_run.hide()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()

        self._status_lbl = QLabel("请先导入文件并运行版面分析")
        self._status_lbl.setObjectName("muted")

        # 主区域（两栏：页面列表 + 图像查看器）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：页面目录——宍依设计稿的带缩略图 + 状态徽章的 PageDirectoryList。
        from app.ui.widgets.page_directory import PageDirectoryList
        self._page_list = PageDirectoryList()
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        splitter.addWidget(self._page_list)

        # 中：图像查看器 + 上方编辑工具条
        viewer_wrap = QWidget()
        vw_lay = QVBoxLayout(viewer_wrap)
        vw_lay.setContentsMargins(0, 0, 0, 0)
        vw_lay.setSpacing(0)

        viewer_tb = QFrame()
        viewer_tb.setObjectName("viewerToolbar")
        viewer_tb.setFixedHeight(34)
        vtl = QHBoxLayout(viewer_tb)
        vtl.setContentsMargins(10, 0, 10, 0)
        vtl.setSpacing(8)

        self._btn_char_boxes = QPushButton("\u2318 \u5b57\u6846")
        self._btn_char_boxes.setObjectName("toolToggle")
        self._btn_char_boxes.setCheckable(True)
        self._btn_char_boxes.setChecked(True)
        self._btn_char_boxes.setToolTip("\u663e\u793a/\u9690\u85cf OCR \u5b57\u6846")
        self._btn_char_boxes.clicked.connect(lambda: self._update_viewer(self._current_page_idx))
        vtl.addWidget(self._btn_char_boxes)

        vtl.addSpacing(6)
        vtl.addWidget(QLabel("\u65b0\u5efa:"))
        self._new_type_combo = QComboBox()
        self._new_type_combo.setMinimumWidth(90)
        for bt in BlockType:
            self._new_type_combo.addItem(bt.value, bt)
        vtl.addWidget(self._new_type_combo)

        vtl.addSpacing(6)
        vtl.addWidget(QLabel("\u9009\u4e2d:"))
        self._type_combo = QComboBox()
        self._type_combo.setEnabled(False)
        self._type_combo.setMinimumWidth(90)
        for bt in BlockType:
            self._type_combo.addItem(bt.value, bt)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        vtl.addWidget(self._type_combo)

        self._btn_merge = QPushButton("合并框")
        self._btn_merge.setObjectName("secondaryBtn")
        self._btn_merge.setToolTip("合并当前多选框；合并后清空旧 OCR 文本，提交时重新识别")
        self._btn_merge.clicked.connect(self._merge_selected_blocks)
        vtl.addWidget(self._btn_merge)

        # bbox / conf 标签随选中状态填充
        self._prop_bbox = QLabel("")
        self._prop_bbox.setObjectName("muted")
        self._prop_conf = ConfidenceBadge(1.0)
        self._prop_conf.hide()
        vtl.addSpacing(8)
        vtl.addWidget(self._prop_bbox)
        vtl.addWidget(self._prop_conf)
        vtl.addStretch(1)
        vw_lay.addWidget(viewer_tb)

        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        self._viewer.block_moved.connect(self._on_block_moved)
        self._viewer.block_created.connect(self._on_block_created)
        self._viewer.block_deleted.connect(self._on_block_deleted)
        self._viewer.char_bbox_moved.connect(self._on_char_bbox_moved)
        vw_lay.addWidget(self._viewer, 1)
        splitter.addWidget(viewer_wrap)

        # 右：Inspector（选中块详情）
        self._inspector = BlockInspector()
        splitter.addWidget(self._inspector)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 9999, 320])
        main_layout.addWidget(splitter, 1)

        # 底栏（设计稿语义：翻页 + 状态 + 完成 / 提交 / 取消）
        bottom = QFrame()
        bottom.setObjectName("toolbar")
        bottom.setFixedHeight(46)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(12, 0, 12, 0)
        bl.setSpacing(8)

        # 左：翻页 < n/N >  ＋  状态
        self._btn_prev = QPushButton("<")
        self._btn_prev.setObjectName("pageNavBtn")
        self._btn_prev.setFixedSize(30, 30)
        self._btn_prev.setToolTip("上一页")
        self._btn_prev.clicked.connect(lambda: self._goto_relative(-1))
        bl.addWidget(self._btn_prev)

        self._lbl_page_no = QLabel("0 / 0")
        self._lbl_page_no.setObjectName("pageNavLabel")
        self._lbl_page_no.setMinimumWidth(48)
        self._lbl_page_no.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self._lbl_page_no)

        self._btn_next = QPushButton(">")
        self._btn_next.setObjectName("pageNavBtn")
        self._btn_next.setFixedSize(30, 30)
        self._btn_next.setToolTip("下一页")
        self._btn_next.clicked.connect(lambda: self._goto_relative(+1))
        bl.addWidget(self._btn_next)

        bl.addSpacing(16)
        bl.addWidget(self._status_lbl)
        bl.addWidget(self._progress_bar)
        # progress 默认隐藏，但需要伸展能力
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setFixedWidth(120)

        bl.addStretch(1)

        # 右：取消 / 完成 / 提交
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setObjectName("secondaryBtn")
        self._btn_cancel.setMinimumHeight(30)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        bl.addWidget(self._btn_cancel)

        self._btn_done = QPushButton("完成本页")
        self._btn_done.setObjectName("secondaryBtn")
        self._btn_done.setMinimumHeight(30)
        self._btn_done.clicked.connect(self._on_done_clicked)
        bl.addWidget(self._btn_done)

        self._btn_submit = QPushButton("提交并进入 OCR")
        self._btn_submit.setObjectName("primaryBtn")
        self._btn_submit.setMinimumHeight(30)
        self._btn_submit.clicked.connect(self._on_submit_clicked)
        bl.addWidget(self._btn_submit)

        main_layout.addWidget(bottom)
        self._update_page_nav()
    # ------------------------------------------------------------------ public

    def set_pages(self, pages: List[Page]) -> None:
        """设置待分析的页面（已加载图片路径）。"""
        self._pages = pages
        self._page_list.set_pages(pages)
        self._btn_run.setEnabled(bool(pages))
        if pages:
            self._page_list.set_current_index(0)
        self._update_page_nav()

    def show_analysis_result(self, pages: List[Page]) -> None:
        """版面分析完成后，更新显示（保持当前选中页）并自动触发 OCR 流程。"""
        self.finish_analysis_progress()
        self._pages = pages
        current_idx = min(self._current_page_idx, len(pages) - 1)
        self._update_viewer(current_idx)
        self._update_page_nav()
        total_blocks = sum(len(p.blocks) for p in pages)
        failed = sum(1 for page in pages if page.error_message)
        if failed:
            self._status_lbl.setText(
                f"共 {len(pages)} 页，{total_blocks} 个版面块，{failed} 页分析失败"
            )
        else:
            self._status_lbl.setText(f"共 {len(pages)} 页，{total_blocks} 个版面块")

    def start_analysis_progress(self, total_pages: int) -> None:
        self._progress_bar.setRange(0, max(1, total_pages))
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._status_lbl.setText(f"正在分析版面… 0/{total_pages}")

    def update_analysis_progress(self, current: int, total: int) -> None:
        current_done = max(0, min(current + 1, total))
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(current_done)
        self._progress_bar.show()
        self._status_lbl.setText(f"正在分析版面… {current_done}/{total}")

    def finish_analysis_progress(self, message: str = "") -> None:
        self._progress_bar.hide()
        if message:
            self._status_lbl.setText(message)

    def set_current_page_number(self, page_number: int) -> None:
        for idx, page in enumerate(self._pages):
            if page.page_number == page_number:
                self._page_list.set_current_index(idx)
                self._current_page_idx = idx
                self._update_viewer(idx)
                self._update_page_nav()
                return

    def set_page_gate_state(
        self,
        page_number: int,
        page_state: str,
        is_pending: bool,
        reason_code: str,
        reason_text: str,
    ) -> None:
        self._page_gate_states[page_number] = (page_state, is_pending, reason_code, reason_text)
        if self._pages and self._pages[self._current_page_idx].page_number == page_number:
            self._status_lbl.setText(reason_text)

    def set_primary_action(self, page_number: int, action_key: str, label: str, enabled: bool) -> None:
        self._primary_actions[page_number] = (action_key, label, enabled)
        if self._pages and self._pages[self._current_page_idx].page_number == page_number:
            self._btn_submit.setText(label)
            self._btn_submit.setEnabled(enabled)

    # ------------------------------------------------------------------ private

    def _request_analysis(self) -> None:
        """触发版面分析（由主窗口的 run_button.clicked 同时连接），显示进度条。"""
        self.start_analysis_progress(len(self._pages))

    def _on_page_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._pages):
            self._current_page_idx = idx
            self._update_viewer(idx)
            self._update_page_nav()
            self.page_selected.emit(self._pages[idx].page_number)

    def _update_viewer(self, idx: int) -> None:
        if not self._pages:
            return
        page = self._pages[idx]
        self._viewer.set_image(page.display_image_path)
        if page.is_analyzed:
            self._viewer.show_blocks(page.blocks)
            if self._btn_char_boxes.isChecked():
                self._viewer.show_char_boxes(self._collect_page_chars(page))
        elif page.error_message:
            self._status_lbl.setText(f"第 {page.page_number} 页分析失败：{page.error_message}")
        self._selected_block = None
        self._type_combo.setEnabled(False)
        self._prop_bbox.setText("")
        self._prop_conf.hide()
        self._inspector.clear()
        self._inspector.set_page_stats(page)
        gate = self._page_gate_states.get(page.page_number)
        if gate is not None:
            self._status_lbl.setText(gate[3])
        action = self._primary_actions.get(page.page_number)
        if action is not None:
            self._btn_submit.setText(action[1])
            self._btn_submit.setEnabled(action[2])

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
        self._inspector.set_block(block)

    def _on_block_moved(self, block: Block) -> None:
        bb = block.bbox
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self._inspector.set_block(block)
        self.geometry_changed.emit()
        self.block_contract_changed.emit(self._pages[self._current_page_idx].page_number, "block_moved")

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
        if self._btn_char_boxes.isChecked():
            self._viewer.show_char_boxes(self._collect_page_chars(page))
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_created")

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
            self._inspector.clear()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_deleted")

    def _delete_selected(self) -> None:
        """底部栏 ✕ 删除框 按钮。"""
        self._viewer.delete_selected()

    def _merge_selected_blocks(self) -> None:
        if not self._pages:
            return
        selected = [block for block in self._viewer.selected_blocks() if block in self._pages[self._current_page_idx].blocks]
        if len(selected) < 2:
            self._status_lbl.setText("请先在画布中多选至少两个框再合并")
            return

        page = self._pages[self._current_page_idx]
        selected.sort(key=lambda block: (block.order, block.bbox.y, block.bbox.x))
        primary = selected[0]
        x1 = min(block.bbox.x1 for block in selected)
        y1 = min(block.bbox.y1 for block in selected)
        x2 = max(block.bbox.x2 for block in selected)
        y2 = max(block.bbox.y2 for block in selected)
        primary.bbox = BBox.from_xyxy(x1, y1, x2, y2)
        primary.lines = []
        primary.source = BlockSource.USER_EDITED
        primary.recognizable = primary.block_type not in (BlockType.FIGURE, BlockType.TABLE, BlockType.UNKNOWN)
        primary.note = "manual_merge_requires_ocr_rerun"
        primary.raw_payload = {
            **dict(primary.raw_payload),
            "manual_merge_from": [
                {
                    "block_type": block.block_type.value,
                    "bbox": list(block.bbox.to_xyxy()),
                    "source_label": block.source_label,
                }
                for block in selected
            ],
            "ocr_text_invalidated": True,
        }

        remove_ids = {id(block) for block in selected[1:]}
        page.blocks = [block for block in page.blocks if id(block) not in remove_ids]
        for order, block in enumerate(page.blocks):
            block.order = order

        self._selected_block = primary
        self._viewer.show_blocks(page.blocks)
        if self._btn_char_boxes.isChecked():
            self._viewer.show_char_boxes(self._collect_page_chars(page))
        self._inspector.set_block(primary)
        self._prop_bbox.setText(f"x={primary.bbox.x} y={primary.bbox.y} w={primary.bbox.w} h={primary.bbox.h}")
        self._status_lbl.setText("已合并选中框；旧 OCR 文本已清空，提交后会按新框重新识别")
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "blocks_merged")

    def _on_type_changed(self, _index: int) -> None:
        if self._selected_block is None:
            return
        new_type: BlockType = self._type_combo.currentData()
        if new_type:
            self._selected_block.block_type = new_type
            self.geometry_changed.emit()
            self.block_contract_changed.emit(self._pages[self._current_page_idx].page_number, "block_type_changed")

    def _on_char_bbox_moved(self, char) -> None:
        bb = char.bbox
        if bb is not None:
            self._prop_bbox.setText(f"字框 x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self.geometry_changed.emit()

    def _collect_page_chars(self, page: Page):
        chars = []
        for block in page.blocks:
            for line in block.lines:
                chars.extend([char for char in line.chars if char.bbox is not None])
        return chars

    # ── 底栏：翻页 / 完成 / 取消 / 提交 ─────────────────────────

    def _update_page_nav(self) -> None:
        n = len(self._pages)
        cur = (self._current_page_idx + 1) if n else 0
        self._lbl_page_no.setText(f"{cur} / {n}")
        self._btn_prev.setEnabled(n > 0 and self._current_page_idx > 0)
        self._btn_next.setEnabled(n > 0 and self._current_page_idx < n - 1)
        has_pages = n > 0
        self._btn_done.setEnabled(has_pages)
        self._btn_cancel.setEnabled(has_pages)
        if has_pages:
            page_number = self._pages[self._current_page_idx].page_number
            action = self._primary_actions.get(page_number)
            if action is not None:
                self._btn_submit.setText(action[1])
                self._btn_submit.setEnabled(action[2])
            else:
                self._btn_submit.setText("提交并进入 OCR")
                self._btn_submit.setEnabled(True)
        else:
            self._btn_submit.setText("提交并进入 OCR")
            self._btn_submit.setEnabled(False)

    def _goto_relative(self, delta: int) -> None:
        if not self._pages:
            return
        new_idx = self._current_page_idx + delta
        if 0 <= new_idx < len(self._pages):
            self._page_list.set_current_index(new_idx)
            # 手动触发 viewer 更新（set_current_index 不会发 currentRowChanged）
            self._current_page_idx = new_idx
            self._update_viewer(new_idx)
            self._update_page_nav()
            self.page_selected.emit(self._pages[new_idx].page_number)

    def _on_done_clicked(self) -> None:
        """完成本页编辑（占位：发出信号供控制器处理；当前仅状态提示）。"""
        if not self._pages:
            return
        self._status_lbl.setText(f"第 {self._pages[self._current_page_idx].page_number} 页编辑已记录")
        self.page_completed.emit(self._current_page_idx)

    def _on_cancel_clicked(self) -> None:
        """取消本页未提交的编辑（占位：发出信号供控制器处理）。"""
        if not self._pages:
            return
        self._status_lbl.setText(f"已取消第 {self._pages[self._current_page_idx].page_number} 页编辑")
        self.edits_cancelled.emit(self._current_page_idx)

    def _on_submit_clicked(self) -> None:
        """提交版面分析结果，进入 OCR 阶段。"""
        if not self._pages:
            return
        page_number = self._pages[self._current_page_idx].page_number
        self.ocr_entry_requested.emit("layout_submit", page_number)
        self.analysis_confirmed.emit()

    @property
    def run_button(self) -> QPushButton:
        return self._btn_run
