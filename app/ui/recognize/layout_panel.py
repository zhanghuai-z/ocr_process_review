"""版面分析面板：图像 + BBox 叠加可视化，块信息内嵌底部栏。"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from app.core.bbox_extraction import bbox_from_variant
from app.core.block_payload import (
    MANUAL_MERGE_FROM_KEY,
    MANUAL_DRAW_BBOX_KEY,
    OCR_TEXT_INVALIDATED_KEY,
    PADDLE_BLOCK_BBOX_KEY,
    PADDLE_BLOCK_LABEL_KEY,
    UI_DEFAULT_LOCKED_KEY,
    UI_DELETED_INLINE_FORMULA_KEY,
    UI_GENERATED_INLINE_FORMULA_BLOCK_KEY,
    UI_INLINE_FORMULA_ORIGIN_BBOX_KEY,
    UI_INLINE_FORMULA_PARENT_LABEL_KEY,
    UI_LOCK_OVERRIDDEN_KEY,
    payload_bool,
    payload_get,
    set_payload_entries,
)
from app.core.ocr_dispatch_policy import is_text_ocr_candidate
from app.core.paddle_artifact_index import (
    BINDING_AMBIGUOUS,
    BINDING_EMPTY_REVIEW,
    PaddleArtifactIndex,
    apply_paddle_binding_to_block,
)
from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import (
    ROUTE_SUBBLOCKS_FIELD,
    formula_texts_by_subblock_bbox,
    line_routes_for_block,
)
from app.core.ocr_ir import is_formula_marker_token
from app.models import BBox, Block, BlockSource, BlockType, Page
from app.ui.widgets.image_viewer import BLOCK_COLORS, ImageViewer
from app.ui.widgets.block_inspector import BlockInspector
from app.core.proof_state_bus import ProofStateBus
from app.ui.widgets.confidence_badge import ConfidenceBadge

STATUS_LABEL_MAX_CHARS = 96


@dataclass(frozen=True)
class LayoutSubtypeSpec:
    """UI button spec for one Paddle layout label."""

    label: str
    source_label: str
    block_type: BlockType
    note: str = ""

    @property
    def normalized_source_label(self) -> str:
        return normalize_paddle_label(self.source_label)


BLOCK_TYPE_BUTTON_GROUPS = (
    ("文本", (
        LayoutSubtypeSpec("正文", "text", BlockType.TEXT, "普通正文段落"),
        LayoutSubtypeSpec("段落标题", "paragraph_title", BlockType.TITLE, "章节/小节标题"),
        LayoutSubtypeSpec("文档标题", "doc_title", BlockType.TITLE, "页面或文档主标题"),
        LayoutSubtypeSpec("摘要", "abstract", BlockType.TEXT, "摘要内容"),
    )),
    ("页边", (
        LayoutSubtypeSpec("页眉", "header", BlockType.TEXT, "页眉区域，通常不进入正文校对"),
        LayoutSubtypeSpec("页脚", "footer", BlockType.TEXT, "页脚区域，通常不进入正文校对"),
        LayoutSubtypeSpec("页码", "number", BlockType.TEXT, "页码/编号类位置元素"),
        LayoutSubtypeSpec("脚注", "footnote", BlockType.TEXT, "脚注文本"),
    )),
    ("公式表格", (
        LayoutSubtypeSpec("公式", "display_formula", BlockType.EQUATION, "独立公式块"),
        LayoutSubtypeSpec("行内公式", "inline_formula", BlockType.EQUATION, "正文行内公式"),
        LayoutSubtypeSpec("公式序号", "formula_number", BlockType.EQUATION, "公式右侧或附近的编号"),
        LayoutSubtypeSpec("表格", "table", BlockType.TABLE, "表格主体区域"),
    )),
    ("图像图表", (
        LayoutSubtypeSpec("图片", "figure", BlockType.FIGURE, "图片/插图区域"),
        LayoutSubtypeSpec("图表", "chart", BlockType.FIGURE, "统计图、坐标图等图表区域"),
        LayoutSubtypeSpec("图题", "figure_title", BlockType.FIGURE_CAPTION, "图片或图表标题"),
        LayoutSubtypeSpec("表题", "table_title", BlockType.TABLE_CAPTION, "表格标题"),
    )),
    ("引用注释", (
        LayoutSubtypeSpec("参考内容", "reference_content", BlockType.REFERENCE, "参考文献或引用条目"),
        LayoutSubtypeSpec("视觉脚注", "vision_footnote", BlockType.TEXT, "Paddle 视觉模型判定的脚注区域"),
    )),
)
BLOCK_SUBTYPE_BUTTON_ORDER = tuple(
    spec
    for _group_title, specs in BLOCK_TYPE_BUTTON_GROUPS
    for spec in specs
)
BLOCK_TYPE_BUTTON_ORDER = tuple(dict.fromkeys(spec.block_type for spec in BLOCK_SUBTYPE_BUTTON_ORDER))
DEFAULT_SUBTYPE_BY_SOURCE_LABEL = {
    spec.normalized_source_label: spec
    for spec in BLOCK_SUBTYPE_BUTTON_ORDER
}
DEFAULT_SUBTYPE_BY_BLOCK_TYPE: dict[BlockType, LayoutSubtypeSpec] = {}
for _spec in BLOCK_SUBTYPE_BUTTON_ORDER:
    DEFAULT_SUBTYPE_BY_BLOCK_TYPE.setdefault(_spec.block_type, _spec)
BLOCK_TYPE_LABELS = {
    BlockType.TEXT: "正文",
    BlockType.TITLE: "标题",
    BlockType.EQUATION: "公式",
    BlockType.TABLE: "表格",
    BlockType.FIGURE: "图片",
    BlockType.FIGURE_CAPTION: "图注",
    BlockType.TABLE_CAPTION: "表注",
    BlockType.REFERENCE: "引用",
    BlockType.UNKNOWN: "其他",
}


def _compact_status_text(text: str) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= STATUS_LABEL_MAX_CHARS:
        return value
    return value[: STATUS_LABEL_MAX_CHARS - 1] + "…"


def _block_type_label(block_type: BlockType) -> str:
    return BLOCK_TYPE_LABELS.get(block_type, block_type.value)


def _block_type_button_stylesheet(block_type: BlockType) -> str:
    color = BLOCK_COLORS.get(block_type, BLOCK_COLORS[BlockType.UNKNOWN])
    border = color.name()
    return f"""
        QPushButton {{
            min-height: 24px;
            padding: 3px 6px;
            border-radius: 6px;
            border: 1px solid rgba({color.red()}, {color.green()}, {color.blue()}, 150);
            background: rgba({color.red()}, {color.green()}, {color.blue()}, 28);
            color: #202124;
        }}
        QPushButton:checked {{
            border: 2px solid {border};
            background: rgba({color.red()}, {color.green()}, {color.blue()}, 76);
            font-weight: 600;
        }}
        QPushButton:disabled {{
            border: 1px solid #dfe3e7;
            background: #f5f6f7;
            color: #9aa0a6;
            font-weight: 400;
        }}
    """


def _block_type_group_stylesheet() -> str:
    return """
        QFrame#blockTypeGroup {
            border: 1px solid #e5e8ec;
            border-radius: 6px;
            background: #fbfcfd;
        }
        QLabel#blockTypeGroupTitle {
            color: #5f6b7a;
            font-size: 12px;
            font-weight: 600;
        }
    """


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
    DRAW_SNAP_TOLERANCE = 8
    DRAW_INK_SNAP_TOLERANCE = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._selected_block: Optional[Block] = None
        self._page_gate_states: dict[int, tuple[str, bool, str, str]] = {}
        self._primary_actions: dict[int, tuple[str, str, bool]] = {}
        self._undo_stack: list[tuple[int, list[Block]]] = []
        self._ink_mask_cache: dict[str, tuple[object, int, int, list[tuple[int, int, int, int, int]]]] = {}
        self._new_subtype: LayoutSubtypeSpec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"]
        self._new_block_type: BlockType = self._new_subtype.block_type
        self._new_type_buttons: dict[BlockType, QPushButton] = {}
        self._new_subtype_buttons: dict[str, QPushButton] = {}
        self._selected_type_buttons: dict[BlockType, QPushButton] = {}
        self._selected_subtype_buttons: dict[str, QPushButton] = {}
        self._type_group: QButtonGroup | None = None
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
        self._status_lbl.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        # 主区域（两栏：页面列表 + 图像查看器）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter

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
        self._btn_char_boxes.clicked.connect(self._refresh_current_page_layers)
        vtl.addWidget(self._btn_char_boxes)

        vtl.addSpacing(10)
        self._viewer_hint = QLabel("编辑：拖动/缩放框；Shift+拖拽添加或合并；按住空格移动画布")
        self._viewer_hint.setObjectName("muted")
        vtl.addWidget(self._viewer_hint)

        # bbox / conf 标签随选中状态填充
        self._prop_bbox = QLabel("")
        self._prop_bbox.setObjectName("muted")
        self._prop_conf = ConfidenceBadge(1.0)
        self._prop_conf.hide()
        vtl.addSpacing(8)
        vtl.addWidget(self._prop_bbox)
        vtl.addWidget(self._prop_conf)
        vtl.addStretch(1)
        self._selection_type_status = QLabel("")
        self._selection_type_status.setObjectName("selectionTypeStatus")
        self._selection_type_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._selection_type_status.setMinimumWidth(190)
        vtl.addWidget(self._selection_type_status)
        vw_lay.addWidget(viewer_tb)

        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        self._viewer.block_edit_started.connect(self._on_block_edit_started)
        self._viewer.block_moved.connect(self._on_block_moved)
        self._viewer.block_created.connect(self._on_block_created)
        self._viewer.block_deleted.connect(self._on_block_deleted)
        self._viewer.edit_blocked.connect(self._on_viewer_edit_blocked)
        self._viewer.char_bbox_moved.connect(self._on_char_bbox_moved)
        self._viewer.set_bbox_snapper(self._snap_current_draw_bbox)
        vw_lay.addWidget(self._viewer, 1)
        splitter.addWidget(viewer_wrap)

        # 右：工具栏 + Inspector（选中块详情）
        right_wrap = QWidget()
        right_lay = QVBoxLayout(right_wrap)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        tool_panel = QFrame()
        tool_panel.setObjectName("sidebarBar")
        tool_lay = QVBoxLayout(tool_panel)
        tool_lay.setContentsMargins(12, 12, 12, 10)
        tool_lay.setSpacing(8)

        tool_title = QLabel("工具")
        tool_title.setObjectName("sectionTitle")
        tool_lay.addWidget(tool_title)

        self._type_context_title = QLabel("新建框类型")
        tool_lay.addWidget(self._type_context_title)
        self._type_group = QButtonGroup(self)
        self._type_group.setExclusive(True)
        self._new_type_group = self._type_group
        self._selected_type_group = self._type_group
        self._new_type_buttons, self._new_subtype_buttons = self._create_type_button_grid(
            tool_lay,
            group=self._type_group,
            callback=self._on_type_button_clicked,
        )
        self._selected_type_buttons = self._new_type_buttons
        self._selected_subtype_buttons = self._new_subtype_buttons
        self._sync_type_buttons(None)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)

        self._btn_lock = QPushButton("锁定框")
        self._btn_lock.setObjectName("secondaryBtn")
        self._btn_lock.setCheckable(True)
        self._btn_lock.setEnabled(False)
        self._btn_lock.setToolTip("锁定后不可拖动、缩放或删除；再次点击解锁")
        self._btn_lock.clicked.connect(self._toggle_selected_lock)
        action_row.addWidget(self._btn_lock)

        self._btn_undo = QPushButton("撤销")
        self._btn_undo.setObjectName("secondaryBtn")
        self._btn_undo.setEnabled(False)
        self._btn_undo.setToolTip("撤销上一步版面编辑（Ctrl+Z）")
        self._btn_undo.clicked.connect(self._undo_last_edit)
        action_row.addWidget(self._btn_undo)
        tool_lay.addLayout(action_row)

        action_row_2 = QHBoxLayout()
        action_row_2.setContentsMargins(0, 0, 0, 0)
        action_row_2.setSpacing(6)

        self._btn_merge = QPushButton("合并选中框")
        self._btn_merge.setObjectName("secondaryBtn")
        self._btn_merge.setToolTip("合并当前多选框；合并后清空旧 OCR 文本，提交时重新识别")
        self._btn_merge.clicked.connect(self._merge_selected_blocks)
        action_row_2.addWidget(self._btn_merge)

        self._btn_delete = QPushButton("删除选中")
        self._btn_delete.setObjectName("secondaryBtn")
        self._btn_delete.setToolTip("删除当前选中的非锁定框（Delete）")
        self._btn_delete.clicked.connect(self._delete_selected)
        action_row_2.addWidget(self._btn_delete)
        tool_lay.addLayout(action_row_2)

        self._btn_unlock_page = QPushButton("解除本页锁定")
        self._btn_unlock_page.setObjectName("secondaryBtn")
        self._btn_unlock_page.setToolTip("需要修改正文框时，先解除本页锁定")
        self._btn_unlock_page.clicked.connect(self._unlock_page_blocks)
        tool_lay.addWidget(self._btn_unlock_page)

        tool_hint = QLabel("Shift+拖拽：添加框；覆盖非锁定框时自动合并为大框。\nSpace 长按：只移动画布，不编辑框。")
        tool_hint.setObjectName("muted")
        tool_hint.setWordWrap(True)
        tool_lay.addWidget(tool_hint)

        right_lay.addWidget(tool_panel)

        self._inspector = BlockInspector()
        right_lay.addWidget(self._inspector, 1)

        right_scroll = QScrollArea()
        right_scroll.setObjectName("layoutToolScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(380)
        right_scroll.setWidget(right_wrap)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 980, 320])
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
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._undo_last_edit)
        self._update_page_nav()
    # ------------------------------------------------------------------ public

    def set_pages(self, pages: List[Page]) -> None:
        """设置待分析的页面（已加载图片路径）。"""
        self._pages = pages
        self._ink_mask_cache.clear()
        self._page_list.set_pages(pages)
        self._btn_run.setEnabled(bool(pages))
        if pages:
            self._page_list.set_current_index(0)
        self._update_page_nav()

    def reset(self) -> None:
        self.finish_analysis_progress()
        self._pages = []
        self._current_page_idx = 0
        self._selected_block = None
        self._page_gate_states.clear()
        self._primary_actions.clear()
        self._undo_stack.clear()
        self._ink_mask_cache.clear()
        self._page_list.set_pages([])
        self._viewer.clear()
        self._set_status_text("请先导入文件并运行版面分析")
        self._btn_run.setEnabled(False)
        self._btn_undo.setEnabled(False)
        self._set_selected_type_buttons_enabled(False)
        self._sync_lock_button(None)
        self._prop_bbox.setText("")
        self._prop_conf.hide()
        self._inspector.clear()
        self._update_page_nav()

    def show_analysis_result(self, pages: List[Page]) -> None:
        """版面分析完成后，更新显示（保持当前选中页）并自动触发 OCR 流程。"""
        self.finish_analysis_progress()
        self._pages = pages
        self._ink_mask_cache.clear()
        current_idx = min(self._current_page_idx, len(pages) - 1)
        self._update_viewer(current_idx)
        self._update_page_nav()
        total_blocks = sum(len(p.blocks) for p in pages)
        failed = sum(1 for page in pages if page.error_message)
        if failed:
            self._set_status_text(
                f"共 {len(pages)} 页，{total_blocks} 个版面块，{failed} 页分析失败"
            )
        else:
            self._set_status_text(f"共 {len(pages)} 页，{total_blocks} 个版面块")

    def start_analysis_progress(self, total_pages: int) -> None:
        self._progress_bar.setRange(0, max(1, total_pages))
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._set_status_text(f"正在分析版面… 0/{total_pages}")

    def update_analysis_progress(self, current: int, total: int) -> None:
        current_done = max(0, min(current + 1, total))
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(current_done)
        self._progress_bar.show()
        self._set_status_text(f"正在分析版面… {current_done}/{total}")

    def finish_analysis_progress(self, message: str = "") -> None:
        self._progress_bar.hide()
        if message:
            self._set_status_text(message)

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
            self._set_status_text(reason_text)

    def set_primary_action(self, page_number: int, action_key: str, label: str, enabled: bool) -> None:
        self._primary_actions[page_number] = (action_key, label, enabled)
        if self._pages and self._pages[self._current_page_idx].page_number == page_number:
            self._btn_submit.setText(label)
            self._btn_submit.setEnabled(enabled)

    def _set_status_text(self, text: str) -> None:
        full = str(text or "")
        compact = _compact_status_text(full)
        self._status_lbl.setText(compact)
        self._status_lbl.setToolTip(full if compact != full else "")

    def _create_type_button_grid(
        self,
        parent_layout: QVBoxLayout,
        *,
        group: QButtonGroup,
        callback,
    ) -> tuple[dict[BlockType, QPushButton], dict[str, QPushButton]]:
        type_buttons: dict[BlockType, QPushButton] = {}
        subtype_buttons: dict[str, QPushButton] = {}
        wrap = QWidget()
        column = QVBoxLayout(wrap)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        for group_title, subtype_specs in BLOCK_TYPE_BUTTON_GROUPS:
            group_frame = QFrame()
            group_frame.setObjectName("blockTypeGroup")
            group_frame.setStyleSheet(_block_type_group_stylesheet())
            group_lay = QVBoxLayout(group_frame)
            group_lay.setContentsMargins(8, 6, 8, 8)
            group_lay.setSpacing(5)

            title = QLabel(group_title)
            title.setObjectName("blockTypeGroupTitle")
            group_lay.addWidget(title)

            grid_wrap = QWidget()
            grid = QGridLayout(grid_wrap)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(6)
            for index, spec in enumerate(subtype_specs):
                button = QPushButton(spec.label)
                button.setObjectName("blockTypeButton")
                button.setCheckable(True)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                button.setStyleSheet(_block_type_button_stylesheet(spec.block_type))
                tooltip = f"{group_title} · {spec.label} / {spec.source_label} → {spec.block_type.value}"
                if spec.note:
                    tooltip = f"{tooltip}\n{spec.note}"
                button.setToolTip(tooltip)
                button.clicked.connect(lambda _checked=False, value=spec: callback(value))
                group.addButton(button)
                subtype_buttons[spec.normalized_source_label] = button
                type_buttons.setdefault(spec.block_type, button)
                grid.addWidget(button, index // 2, index % 2)
            group_lay.addWidget(grid_wrap)
            column.addWidget(group_frame)
        parent_layout.addWidget(wrap)
        return type_buttons, subtype_buttons

    @staticmethod
    def _set_type_button_checked(
        group: QButtonGroup,
        buttons: dict[str, QPushButton],
        source_label: str | None,
    ) -> None:
        normalized_source_label = normalize_paddle_label(source_label)
        group.setExclusive(False)
        for current_label, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(bool(normalized_source_label) and current_label == normalized_source_label)
            button.blockSignals(False)
        group.setExclusive(True)

    def _set_type_buttons_enabled(self, enabled: bool) -> None:
        for button in self._new_subtype_buttons.values():
            button.setEnabled(enabled)

    def _set_selected_type_buttons_enabled(self, enabled: bool) -> None:
        self._set_type_buttons_enabled(enabled)

    def _sync_type_buttons(self, block: Block | None) -> None:
        if block is None:
            self._set_type_buttons_enabled(True)
            self._type_context_title.setText("新建框类型")
            self._set_type_button_checked(
                self._type_group,
                self._new_subtype_buttons,
                self._new_subtype.source_label,
            )
            self._update_selection_type_status(None)
            return
        enabled = not bool(getattr(block, "is_locked", False))
        self._set_type_buttons_enabled(enabled)
        self._type_context_title.setText("选中框类型")
        self._set_type_button_checked(
            self._type_group,
            self._new_subtype_buttons,
            self._button_source_label_for_block(block),
        )
        self._update_selection_type_status(block)

    def _sync_selected_type_buttons(self, block: Block | None) -> None:
        self._sync_type_buttons(block)

    def _set_new_block_type(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        self._new_subtype = self._coerce_subtype_spec(subtype, DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"])
        self._new_block_type = self._new_subtype.block_type
        if self._selected_block is None:
            self._sync_type_buttons(None)

    def _on_type_button_clicked(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        if self._selected_block is not None:
            self._on_selected_type_button_clicked(subtype)
            return
        self._set_new_block_type(subtype)

    def _update_selection_type_status(self, block: Block | None) -> None:
        if block is None:
            spec = self._new_subtype
            text = f"新建：{spec.label} / {spec.source_label}"
            self._selection_type_status.setText(text)
            self._selection_type_status.setToolTip(f"当前新建框类型：{spec.label} / {spec.source_label} → {spec.block_type.value}")
            return
        label = self._button_source_label_for_block(block)
        spec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL.get(normalize_paddle_label(label))
        if spec is not None:
            text = f"选中：{spec.label} / {spec.source_label}"
            tooltip = f"当前选中框属性：{spec.label} / {spec.source_label} → {spec.block_type.value}"
        else:
            raw_label = block.source_label or block.block_type.value
            text = f"选中：{_block_type_label(block.block_type)} / {raw_label}"
            tooltip = f"当前选中框属性：{raw_label} → {block.block_type.value}"
        if getattr(block, "is_locked", False):
            text = f"{text} · 已锁定"
            tooltip = f"{tooltip}\n该框已锁定，不能修改属性"
        self._selection_type_status.setText(text)
        self._selection_type_status.setToolTip(tooltip)

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
        self._apply_default_locks(page)
        self._viewer.set_image(page.display_image_path)
        if page.is_analyzed:
            self._show_page_layers(page)
        elif page.error_message:
            self._set_status_text(f"第 {page.page_number} 页分析失败：{page.error_message}")
        self._selected_block = None
        self._sync_selected_type_buttons(None)
        self._btn_lock.setEnabled(False)
        self._btn_lock.setChecked(False)
        self._btn_lock.setText("锁定框")
        self._prop_bbox.setText("")
        self._prop_conf.hide()
        self._inspector.clear()
        self._inspector.set_page_stats(page)
        gate = self._page_gate_states.get(page.page_number)
        if gate is not None:
            self._set_status_text(gate[3])
        action = self._primary_actions.get(page.page_number)
        if action is not None:
            self._btn_submit.setText(action[1])
            self._btn_submit.setEnabled(action[2])

    def _refresh_current_page_layers(self) -> None:
        """Refresh layout overlays without reloading the image or resetting zoom."""
        if not self._pages:
            return
        self._show_page_layers(self._pages[self._current_page_idx])

    def _clear_selection_ui(self, page: Page) -> None:
        self._selected_block = None
        self._sync_selected_type_buttons(None)
        self._btn_lock.setEnabled(False)
        self._btn_lock.setChecked(False)
        self._btn_lock.setText("锁定框")
        self._prop_bbox.setText("")
        self._prop_conf.hide()
        self._inspector.clear()
        self._inspector.set_page_stats(page)

    def _on_block_clicked(self, block: Block) -> None:
        if getattr(block, "is_locked", False):
            self._set_status_text("该框已锁定；如需编辑，请先解除本页锁定")
            return
        self._selected_block = block
        bb = block.bbox
        self._sync_lock_button(block)
        self._sync_selected_type_buttons(block)
        self._prop_bbox.setText(f"x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self._prop_conf.set_score(block.avg_confidence)
        self._prop_conf.show()
        self._inspector.set_block(block)

    def _on_block_edit_started(self, block: Block) -> None:
        if getattr(block, "is_locked", False):
            return
        self._push_undo_snapshot()

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
        bbox = self._snap_drawn_bbox(page, bbox)
        if bbox.area <= 0:
            return
        self._push_undo_snapshot()
        subtype = self._new_subtype
        bt = self._coerce_block_type(subtype.block_type, BlockType.TEXT)
        intersecting = self._blocks_intersecting_bbox(page, bbox)
        if intersecting:
            merged = self._merge_blocks_into_bbox(page, intersecting, bbox, bt, subtype.source_label)
            self._show_page_layers(page)
            self._select_block_for_edit(merged)
            self._set_status_text("已按拖拽范围合并框；旧 OCR 文本已清空，提交后会重新识别")
            self.geometry_changed.emit()
            self.block_contract_changed.emit(page.page_number, "blocks_merged_by_draw")
            return
        new_block = Block(
            block_type=bt,
            bbox=bbox,
            source=BlockSource.MANUAL_DRAW,
            source_label=subtype.source_label,
        )
        new_block.recognizable = is_text_ocr_candidate(new_block)
        self._bind_manual_block_to_paddle(page, new_block)
        page.blocks.append(new_block)
        self._show_page_layers(page)
        self._select_block_for_edit(new_block)
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_created")

    def _on_block_deleted(self, block: Block) -> None:
        """viewer 键盘 Delete 已删除框 → 从 page 数据中移除。"""
        if not self._pages:
            return
        if getattr(block, "is_locked", False):
            self._set_status_text("选中框已锁定，需先解锁后删除")
            return
        self._push_undo_snapshot()
        page = self._pages[self._current_page_idx]
        self._mark_generated_inline_formula_handled(page, block)
        page.blocks = [b for b in page.blocks if b is not block]
        if self._selected_block is block:
            self._selected_block = None
            self._sync_selected_type_buttons(None)
            self._prop_bbox.setText("")
            self._prop_conf.hide()
            self._inspector.clear()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_deleted")

    def _delete_selected(self) -> None:
        """底部栏 ✕ 删除框 按钮。"""
        if not self._viewer.selected_blocks():
            self._set_status_text("请先选择要删除的非锁定框")
            return
        self._viewer.delete_selected()

    def _merge_selected_blocks(self) -> None:
        if not self._pages:
            return
        selected = [block for block in self._viewer.selected_blocks() if block in self._pages[self._current_page_idx].blocks]
        if len(selected) < 2:
            self._set_status_text("请先在画布中多选至少两个框再合并")
            return

        page = self._pages[self._current_page_idx]
        self._push_undo_snapshot()
        selected.sort(key=lambda block: (block.order, block.bbox.y, block.bbox.x))
        primary = selected[0]
        x1 = min(block.bbox.x1 for block in selected)
        y1 = min(block.bbox.y1 for block in selected)
        x2 = max(block.bbox.x2 for block in selected)
        y2 = max(block.bbox.y2 for block in selected)
        primary.bbox = BBox.from_xyxy(x1, y1, x2, y2)
        primary.lines = []
        primary.source = BlockSource.USER_EDITED
        primary.recognizable = is_text_ocr_candidate(primary)
        primary.note = "manual_merge_requires_ocr_rerun"
        set_payload_entries(primary, {
            MANUAL_MERGE_FROM_KEY: [
                {
                    "block_type": getattr(block.block_type, "value", str(block.block_type)),
                    "bbox": list(block.bbox.to_xyxy()),
                    "source_label": block.source_label,
                }
                for block in selected
            ],
            OCR_TEXT_INVALIDATED_KEY: True,
        })

        for block in selected[1:]:
            self._mark_generated_inline_formula_handled(page, block)
        remove_ids = {id(block) for block in selected[1:]}
        page.blocks = [block for block in page.blocks if id(block) not in remove_ids]
        for order, block in enumerate(page.blocks):
            block.order = order

        self._selected_block = primary
        self._show_page_layers(page)
        self._select_block_for_edit(primary)
        self._prop_bbox.setText(f"x={primary.bbox.x} y={primary.bbox.y} w={primary.bbox.w} h={primary.bbox.h}")
        self._set_status_text("已合并选中框；旧 OCR 文本已清空，提交后会按新框重新识别")
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "blocks_merged")

    def _on_selected_type_button_clicked(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        if self._selected_block is None:
            return
        if getattr(self._selected_block, "is_locked", False):
            self._set_status_text("该框已锁定；如需修改属性，请先解除本页锁定")
            self._sync_selected_type_buttons(self._selected_block)
            return
        new_subtype = self._coerce_subtype_spec(subtype, DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"])
        new_label = new_subtype.normalized_source_label
        current_label = normalize_paddle_label(self._selected_block.source_label)
        if self._selected_block.block_type == new_subtype.block_type and current_label == new_label:
            self._sync_selected_type_buttons(self._selected_block)
            return
        self._push_undo_snapshot()
        self._selected_block.block_type = new_subtype.block_type
        self._selected_block.source_label = new_subtype.source_label
        self._selected_block.source = BlockSource.USER_EDITED
        self._selected_block.recognizable = is_text_ocr_candidate(self._selected_block)
        self._bind_manual_block_to_paddle(
            self._pages[self._current_page_idx],
            self._selected_block,
        )
        self._show_page_layers(self._pages[self._current_page_idx])
        self._viewer.select_block(self._selected_block)
        self._sync_lock_button(self._selected_block)
        self._sync_selected_type_buttons(self._selected_block)
        self.geometry_changed.emit()
        self.block_contract_changed.emit(self._pages[self._current_page_idx].page_number, "block_type_changed")

    def _on_char_bbox_moved(self, char) -> None:
        bb = char.bbox
        if bb is not None:
            self._prop_bbox.setText(f"字框 x={bb.x} y={bb.y} w={bb.w} h={bb.h}")
        self.geometry_changed.emit()

    @staticmethod
    def _collect_page_chars(page: Page):
        if page.needs_ocr_rerun:
            return []
        chars = []
        for block in page.blocks:
            if payload_bool(block, OCR_TEXT_INVALIDATED_KEY):
                continue
            for line in block.lines:
                chars.extend([
                    char
                    for char in line.chars
                    if (
                        char.bbox is not None
                        and (char.char or char.token_text)
                        and char.bbox_source != "paddle_inline_formula"
                    )
                ])
        return chars

    def _show_page_layers(self, page: Page) -> None:
        self._ensure_inline_formula_blocks(page)
        self._apply_default_locks(page)
        self._viewer.show_blocks(page.blocks)
        self._viewer.show_readonly_overlays(self._collect_readonly_layout_overlays(page))
        if self._btn_char_boxes.isChecked():
            self._viewer.show_char_boxes(self._collect_page_chars(page), editable=False)

    def _select_block_for_edit(self, block: Block) -> None:
        if getattr(block, "is_locked", False):
            return
        self._viewer.select_block(block)
        self._on_block_clicked(block)

    def _apply_default_locks(self, page: Page) -> None:
        """初始锁定自动版面分析生成的 text 框，避免行内公式校正时误拖正文框。"""
        for block in page.blocks:
            if block.block_type != BlockType.TEXT:
                continue
            if payload_bool(block, UI_LOCK_OVERRIDDEN_KEY):
                continue
            if block.source == BlockSource.AUTO_LAYOUT:
                block.is_locked = True
                set_payload_entries(block, {UI_DEFAULT_LOCKED_KEY: True})

    def _bind_manual_block_to_paddle(self, page: Page, block: Block) -> None:
        if block.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
            return
        explicit_source_label = block.source_label
        binding = PaddleArtifactIndex.from_page(page).bind_manual_bbox(block.bbox, block.block_type)
        apply_paddle_binding_to_block(block, binding)
        if normalize_paddle_label(explicit_source_label) in DEFAULT_SUBTYPE_BY_SOURCE_LABEL:
            block.source_label = explicit_source_label
            set_payload_entries(block, {PADDLE_BLOCK_LABEL_KEY: explicit_source_label})
        if binding.status == BINDING_EMPTY_REVIEW:
            self._set_status_text("已创建空校验框；Paddle 父框没有可直接召回的真值")
        elif binding.status == BINDING_AMBIGUOUS:
            self._set_status_text("已创建校验框；Paddle 父框存在多个候选，需要人工确认")
        elif binding.text:
            self._set_status_text("已绑定 Paddle 父框真值，提交后不会交给 Hanwang 强识别")

    def _sync_lock_button(self, block: Optional[Block]) -> None:
        if block is None:
            self._btn_lock.setEnabled(False)
            self._btn_lock.setChecked(False)
            self._btn_lock.setText("锁定框")
            self._sync_selected_type_buttons(None)
            return
        locked = bool(getattr(block, "is_locked", False))
        self._btn_lock.setEnabled(True)
        self._btn_lock.blockSignals(True)
        self._btn_lock.setChecked(locked)
        self._btn_lock.setText("解锁框" if locked else "锁定框")
        self._btn_lock.blockSignals(False)
        self._sync_selected_type_buttons(block)

    def _toggle_selected_lock(self) -> None:
        block = self._selected_block
        if block is None:
            return
        self._push_undo_snapshot()
        block.is_locked = not block.is_locked
        set_payload_entries(block, {UI_LOCK_OVERRIDDEN_KEY: True})
        if block.is_locked:
            set_payload_entries(block, {UI_DEFAULT_LOCKED_KEY: block.block_type == BlockType.TEXT})
        block.source = BlockSource.USER_EDITED
        self._sync_lock_button(block)
        self._show_page_layers(self._pages[self._current_page_idx])
        self._inspector.set_block(block)
        self.geometry_changed.emit()
        self.block_contract_changed.emit(
            self._pages[self._current_page_idx].page_number,
            "block_lock_changed",
        )

    def _on_viewer_edit_blocked(self, block: Block, reason: str) -> None:
        if reason == "locked":
            self._set_status_text("选中框已锁定，需先解锁后编辑")

    def _unlock_page_blocks(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        locked = [block for block in page.blocks if getattr(block, "is_locked", False)]
        if not locked:
            self._set_status_text("本页没有锁定框")
            return
        self._push_undo_snapshot()
        for block in locked:
            block.is_locked = False
            set_payload_entries(block, {UI_LOCK_OVERRIDDEN_KEY: True})
        self._show_page_layers(page)
        self._sync_lock_button(None)
        self._selected_block = None
        self._inspector.set_page_stats(page)
        self._set_status_text(f"已解除本页 {len(locked)} 个锁定框")
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_lock_changed")

    def _push_undo_snapshot(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        self._undo_stack.append((self._current_page_idx, copy.deepcopy(page.blocks)))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._btn_undo.setEnabled(True)

    def _undo_last_edit(self) -> None:
        if not self._undo_stack or not self._pages:
            return
        page_idx, blocks = self._undo_stack.pop()
        if not (0 <= page_idx < len(self._pages)):
            return
        previous_idx = self._current_page_idx
        page = self._pages[page_idx]
        page.blocks = copy.deepcopy(blocks)
        self._btn_undo.setEnabled(bool(self._undo_stack))
        if page_idx == previous_idx:
            self._show_page_layers(page)
            self._clear_selection_ui(page)
        else:
            self._current_page_idx = page_idx
            self._page_list.blockSignals(True)
            try:
                self._page_list.set_current_index(page_idx)
            finally:
                self._page_list.blockSignals(False)
            self._update_viewer(page_idx)
        self._update_page_nav()
        self._set_status_text("已撤销上一步版面编辑")
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "layout_undo")

    @staticmethod
    def _bbox_intersects(a: BBox, b: BBox) -> bool:
        return a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2

    def _snap_drawn_bbox(self, page: Page, bbox: BBox) -> BBox:
        bbox = bbox.clamp(page.width, page.height)
        if bbox.area <= 0:
            return bbox
        ink_snapped = self._snap_bbox_to_image_ink(page, bbox)
        if ink_snapped is not None:
            return ink_snapped
        x1, y1, x2, y2 = bbox.to_xyxy()
        x_edges: list[tuple[int, int]] = [(0, 6), (page.width, 6)]
        y_edges: list[tuple[int, int]] = [(0, 6), (page.height, 6)]

        def snap(value: int, candidates: list[tuple[int, int]]) -> int:
            nearest, tolerance = min(candidates, key=lambda edge: abs(edge[0] - value))
            if abs(nearest - value) <= tolerance:
                return nearest
            return value

        snapped_x1 = snap(x1, x_edges)
        snapped_y1 = snap(y1, y_edges)
        snapped_x2 = snap(x2, x_edges)
        snapped_y2 = snap(y2, y_edges)
        if snapped_x2 <= snapped_x1:
            snapped_x1, snapped_x2 = x1, x2
        if snapped_y2 <= snapped_y1:
            snapped_y1, snapped_y2 = y1, y2
        return BBox.from_xyxy(snapped_x1, snapped_y1, snapped_x2, snapped_y2).clamp(page.width, page.height)

    def _snap_current_draw_bbox(self, bbox: BBox) -> BBox:
        if not self._pages:
            return bbox
        return self._snap_drawn_bbox(self._pages[self._current_page_idx], bbox)

    def _snap_bbox_to_image_ink(self, page: Page, bbox: BBox) -> Optional[BBox]:
        mask_info = self._get_page_ink_mask(page)
        if mask_info is None:
            return None
        _mask, img_w, img_h, components = mask_info
        if page.width <= 0 or page.height <= 0 or img_w <= 0 or img_h <= 0:
            return None
        sx = img_w / float(page.width)
        sy = img_h / float(page.height)
        bx1 = max(0, int(bbox.x1 * sx))
        by1 = max(0, int(bbox.y1 * sy))
        bx2 = min(img_w, int(bbox.x2 * sx))
        by2 = min(img_h, int(bbox.y2 * sy))
        if bx2 <= bx1 or by2 <= by1:
            return None

        selected: list[tuple[int, int, int, int, int]] = []
        dark_pixels = 0
        for left, top, right, bottom, area in components:
            if left >= bx2 or bx1 >= right or top >= by2 or by1 >= bottom:
                continue
            selected.append((left, top, right, bottom, area))
            dark_pixels += area
        if not selected or dark_pixels < 8:
            return None

        box_area = max(1, (bx2 - bx1) * (by2 - by1))
        if dark_pixels / float(box_area) > 0.92:
            return None

        ink_x1 = int(round(min(item[0] for item in selected) / sx))
        ink_y1 = int(round(min(item[1] for item in selected) / sy))
        ink_x2 = int(round(max(item[2] for item in selected) / sx))
        ink_y2 = int(round(max(item[3] for item in selected) / sy))

        x1, y1, x2, y2 = bbox.to_xyxy()
        snapped_x1 = ink_x1 if abs(ink_x1 - x1) <= self.DRAW_INK_SNAP_TOLERANCE else x1
        snapped_y1 = ink_y1 if abs(ink_y1 - y1) <= self.DRAW_INK_SNAP_TOLERANCE else y1
        snapped_x2 = ink_x2 if abs(ink_x2 - x2) <= self.DRAW_INK_SNAP_TOLERANCE else x2
        snapped_y2 = ink_y2 if abs(ink_y2 - y2) <= self.DRAW_INK_SNAP_TOLERANCE else y2
        if (snapped_x1, snapped_y1, snapped_x2, snapped_y2) == (x1, y1, x2, y2):
            return None
        if snapped_x2 <= snapped_x1 or snapped_y2 <= snapped_y1:
            return None
        return BBox.from_xyxy(snapped_x1, snapped_y1, snapped_x2, snapped_y2).clamp(page.width, page.height)

    def _get_page_ink_mask(self, page: Page) -> Optional[tuple[object, int, int, list[tuple[int, int, int, int, int]]]]:
        image_path = getattr(page, "display_image_path", None) or getattr(page, "image_path", None)
        if not image_path:
            return None
        key = str(image_path)
        cached = self._ink_mask_cache.get(key)
        if cached is not None:
            return cached
        try:
            import cv2
            gray = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
        if gray is None or gray.size == 0:
            return None
        img_h, img_w = gray.shape[:2]
        mask = (gray < 245).astype("uint8")
        try:
            labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        except Exception:
            return None
        components: list[tuple[int, int, int, int, int]] = []
        for label_idx in range(1, labels_count):
            left, top, width, height, area = [int(v) for v in stats[label_idx]]
            if area < 4:
                continue
            components.append((left, top, left + width, top + height, area))
        cached = (mask, img_w, img_h, components)
        self._ink_mask_cache[key] = cached
        return cached

    def _blocks_intersecting_bbox(self, page: Page, bbox: BBox) -> list[Block]:
        return [
            block for block in page.blocks
            if not getattr(block, "is_locked", False)
            and self._bbox_intersects(bbox, block.bbox)
        ]

    def _ensure_inline_formula_blocks(self, page: Page) -> None:
        """Promote Paddle inline_formula subblocks to editable equation blocks."""
        for parent, subblock, bbox in self._iter_inline_formula_subblocks(page):
            if subblock.get(UI_DELETED_INLINE_FORMULA_KEY):
                continue
            origin = list(bbox.to_xyxy())
            if self._has_inline_formula_origin_block(page, origin):
                continue
            page.blocks.append(Block(
                block_type=BlockType.EQUATION,
                bbox=bbox,
                order=len(page.blocks),
                source=BlockSource.AUTO_LAYOUT,
                source_label="inline_formula",
                raw_payload={
                    **dict(subblock.get("raw_payload") if isinstance(subblock.get("raw_payload"), dict) else {}),
                },
                app_payload={
                    PADDLE_BLOCK_LABEL_KEY: "inline_formula",
                    PADDLE_BLOCK_BBOX_KEY: origin,
                    UI_GENERATED_INLINE_FORMULA_BLOCK_KEY: True,
                    UI_INLINE_FORMULA_ORIGIN_BBOX_KEY: origin,
                    UI_INLINE_FORMULA_PARENT_LABEL_KEY: str(parent.get("block_label") or parent.get("label") or ""),
                },
            ))

    def _iter_inline_formula_subblocks(self, page: Page):
        for parent in page.ppvl_parsing_res_list:
            subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
            if not isinstance(subblocks, list):
                continue
            for subblock in subblocks:
                if not isinstance(subblock, dict):
                    continue
                label = str(
                    subblock.get("block_label")
                    or subblock.get("label")
                    or subblock.get("type")
                    or ""
                )
                if normalize_paddle_label(label) != "inline_formula":
                    continue
                bbox = bbox_from_variant(
                    subblock.get("block_bbox") or subblock.get("bbox") or subblock.get("coordinate"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if bbox is None or bbox.area <= 0:
                    continue
                formula_text = self._inline_formula_subblock_text(page, parent, bbox)
                if formula_text and is_formula_marker_token(formula_text):
                    continue
                yield parent, subblock, bbox.clamp(page.width, page.height)

    @staticmethod
    def _inline_formula_subblock_text(page: Page, parent: dict, bbox: BBox) -> str:
        target = bbox.clamp(page.width, page.height).to_xyxy()
        for route in line_routes_for_block(parent, page.width, page.height):
            for segment in route.get("segments", []):
                if segment.get("kind") != "formula":
                    continue
                segment_bbox = bbox_from_variant(
                    segment.get("bbox"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if segment_bbox is None:
                    continue
                if LayoutPanel._same_inline_formula_route_span(segment_bbox.to_xyxy(), target):
                    return str(segment.get("text") or "")
        for formula_bbox, text in formula_texts_by_subblock_bbox(parent, page.width, page.height).items():
            if LayoutPanel._same_inline_formula_route_span(formula_bbox, target):
                return text
        return ""

    @staticmethod
    def _same_inline_formula_route_span(
        route_bbox: tuple[int, int, int, int],
        raw_bbox: tuple[int, int, int, int],
    ) -> bool:
        if route_bbox == raw_bbox:
            return True
        if route_bbox[0] != raw_bbox[0] or route_bbox[2] != raw_bbox[2]:
            return False
        overlap = max(0, min(route_bbox[3], raw_bbox[3]) - max(route_bbox[1], raw_bbox[1]))
        denom = max(1, min(route_bbox[3] - route_bbox[1], raw_bbox[3] - raw_bbox[1]))
        return overlap / denom >= 0.5

    @staticmethod
    def _has_inline_formula_origin_block(page: Page, origin_bbox: list[int]) -> bool:
        origin_tuple = tuple(origin_bbox)
        for block in page.blocks:
            if tuple(payload_get(block, UI_INLINE_FORMULA_ORIGIN_BBOX_KEY) or ()) == origin_tuple:
                return True
            if normalize_paddle_label(getattr(block, "source_label", "")) != "inline_formula":
                continue
            if block.bbox.to_xyxy() == origin_tuple:
                return True
        return False

    def _mark_generated_inline_formula_handled(self, page: Page, block: Block) -> None:
        origin = payload_get(block, UI_INLINE_FORMULA_ORIGIN_BBOX_KEY)
        if not origin:
            return
        origin_tuple = tuple(origin)
        for _parent, subblock, bbox in self._iter_inline_formula_subblocks(page):
            if bbox.to_xyxy() == origin_tuple:
                subblock[UI_DELETED_INLINE_FORMULA_KEY] = True

    @staticmethod
    def _coerce_block_type(value: object, default: BlockType) -> BlockType:
        if isinstance(value, BlockType):
            return value
        try:
            return BlockType(str(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_subtype_spec(
        value: object,
        default: LayoutSubtypeSpec,
    ) -> LayoutSubtypeSpec:
        if isinstance(value, LayoutSubtypeSpec):
            return value
        normalized_label = normalize_paddle_label(value)
        if normalized_label in DEFAULT_SUBTYPE_BY_SOURCE_LABEL:
            return DEFAULT_SUBTYPE_BY_SOURCE_LABEL[normalized_label]
        block_type = LayoutPanel._coerce_block_type(value, BlockType.UNKNOWN)
        return DEFAULT_SUBTYPE_BY_BLOCK_TYPE.get(block_type, default)

    @staticmethod
    def _button_source_label_for_block(block: Block) -> str:
        normalized = normalize_paddle_label(block.source_label)
        if normalized in DEFAULT_SUBTYPE_BY_SOURCE_LABEL:
            return normalized
        spec = DEFAULT_SUBTYPE_BY_BLOCK_TYPE.get(block.block_type)
        return spec.source_label if spec is not None else ""

    def _merge_blocks_into_bbox(
        self,
        page: Page,
        blocks: list[Block],
        bbox: BBox,
        block_type: BlockType,
        source_label: str = "",
    ) -> Block:
        blocks.sort(key=lambda block: (block.order, block.bbox.y, block.bbox.x))
        primary = blocks[0]
        x1 = min([bbox.x1, *(block.bbox.x1 for block in blocks)])
        y1 = min([bbox.y1, *(block.bbox.y1 for block in blocks)])
        x2 = max([bbox.x2, *(block.bbox.x2 for block in blocks)])
        y2 = max([bbox.y2, *(block.bbox.y2 for block in blocks)])
        primary.bbox = BBox.from_xyxy(x1, y1, x2, y2).clamp(page.width, page.height)
        primary.block_type = block_type
        primary.source_label = source_label
        primary.lines = []
        primary.source = BlockSource.USER_EDITED
        primary.is_locked = False
        primary.recognizable = is_text_ocr_candidate(primary)
        primary.note = "manual_draw_merge_requires_ocr_rerun"
        set_payload_entries(primary, {
            MANUAL_MERGE_FROM_KEY: [
                {
                    "block_type": getattr(block.block_type, "value", str(block.block_type)),
                    "bbox": list(block.bbox.to_xyxy()),
                    "source_label": block.source_label,
                    "was_locked": bool(block.is_locked),
                }
                for block in blocks
            ],
            MANUAL_DRAW_BBOX_KEY: list(bbox.to_xyxy()),
            OCR_TEXT_INVALIDATED_KEY: True,
        })
        self._bind_manual_block_to_paddle(page, primary)
        for block in blocks[1:]:
            self._mark_generated_inline_formula_handled(page, block)
        remove_ids = {id(block) for block in blocks[1:]}
        page.blocks = [block for block in page.blocks if id(block) not in remove_ids]
        for order, block in enumerate(page.blocks):
            block.order = order
        self._selected_block = primary
        self._sync_lock_button(primary)
        self._inspector.set_block(primary)
        return primary

    def _collect_readonly_layout_overlays(self, page: Page) -> List[tuple[str, BBox]]:
        overlays: List[tuple[str, BBox]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for parent in page.ppvl_parsing_res_list:
            subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
            if not isinstance(subblocks, list):
                continue
            for subblock in subblocks:
                if not isinstance(subblock, dict):
                    continue
                label = str(
                    subblock.get("block_label")
                    or subblock.get("label")
                    or subblock.get("type")
                    or ""
                )
                if normalize_paddle_label(label) == "inline_formula":
                    continue
                bbox = bbox_from_variant(
                    subblock.get("block_bbox") or subblock.get("bbox") or subblock.get("coordinate"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if bbox is None or bbox.area <= 0:
                    continue
                bbox = bbox.clamp(page.width, page.height)
                key = (label, bbox.to_xyxy())
                if key in seen:
                    continue
                seen.add(key)
                overlays.append((label or "inline_formula", bbox))
        return overlays

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
        self._set_status_text(f"第 {self._pages[self._current_page_idx].page_number} 页编辑已记录")
        self.page_completed.emit(self._current_page_idx)

    def _on_cancel_clicked(self) -> None:
        """取消本页未提交的编辑（占位：发出信号供控制器处理）。"""
        if not self._pages:
            return
        self._set_status_text(f"已取消第 {self._pages[self._current_page_idx].page_number} 页编辑")
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
