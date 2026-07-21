"""版面分析面板：图像 + BBox 叠加可视化，右侧提供框类型与项目统计。"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget
)

from app.utils.icon_manager import get_icon
from app.application.contracts import (
    BlockView,
    LayoutEditCommand,
    LayoutWorkspaceView,
    PageView,
)
from app.core.paddle_labels import normalize_paddle_label
from app.models.enums import BlockType, OcrPolicy
from app.models.geometry import BBox
from app.ui.widgets.image_viewer import ImageViewer, OcrAtomBox
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.effects import apply_soft_shadow
from app.ui.widgets.page_thumbnail import PAGE_ROW_H, PageDirectoryRow
from app.utils.image_io import read_cv_image

STATUS_LABEL_MAX_CHARS = 96
STRUCTURAL_DRAW_BLOCK_TYPES = {BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE}


@dataclass(frozen=True, slots=True)
class LayoutBlockPresentation:
    """Layout block view prepared for the existing viewer presentation."""

    page_uid: str
    block: BlockView

    @property
    def uid(self) -> str:
        return self.block.block_uid

    @property
    def block_type(self) -> BlockType:
        return self.block.block_type

    @property
    def bbox(self) -> BBox:
        return self.block.bbox

    @property
    def order(self) -> int:
        return self.block.order

    @property
    def source_label(self) -> str:
        return self.block.source_label

    @property
    def ocr_policy(self) -> OcrPolicy:
        return self.block.ocr_policy


def _layout_blocks(page: PageView) -> tuple[LayoutBlockPresentation, ...]:
    return tuple(
        LayoutBlockPresentation(
            page_uid=page.page_uid,
            block=block,
        )
        for block in page.blocks
    )


def _ocr_observation_line_count(_page: PageView) -> int:
    return 0


def _ocr_observation_avg_confidence(_page: PageView, _block_uid: str) -> float:
    return 0.0


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
    ("核心与结构", (
        LayoutSubtypeSpec("正文", "text", BlockType.TEXT, "普通正文段落"),
        LayoutSubtypeSpec("摘要", "abstract", BlockType.TEXT, "摘要内容"),
        LayoutSubtypeSpec("公式", "formula", BlockType.EQUATION, "自动判断行内公式/独立公式"),
        LayoutSubtypeSpec("参考文献", "reference_content", BlockType.REFERENCE, "参考文献或引用条目"),
    )),
    ("标题层级", (
        LayoutSubtypeSpec("H1", "heading_1", BlockType.TITLE, "一级标题"),
        LayoutSubtypeSpec("H2", "heading_2", BlockType.TITLE, "二级标题"),
        LayoutSubtypeSpec("H3", "heading_3", BlockType.TITLE, "三级标题"),
        LayoutSubtypeSpec("H4", "heading_4", BlockType.TITLE, "四级标题"),
        LayoutSubtypeSpec("H5", "heading_5", BlockType.TITLE, "五级标题"),
        LayoutSubtypeSpec("H6", "heading_6", BlockType.TITLE, "六级标题"),
    )),
    ("图表元素", (
        LayoutSubtypeSpec("图片", "figure", BlockType.FIGURE, "图片/插图区域"),
        LayoutSubtypeSpec("图题", "figure_title", BlockType.FIGURE_CAPTION, "图片或图表标题"),
        LayoutSubtypeSpec("表格", "table", BlockType.TABLE, "表格主体区域"),
        LayoutSubtypeSpec("表题", "table_title", BlockType.TABLE_CAPTION, "表格标题"),
        LayoutSubtypeSpec("图表", "chart", BlockType.FIGURE, "统计图、坐标图等图表区域"),
    )),
    ("页边元素", (
        LayoutSubtypeSpec("页眉", "header", BlockType.TEXT, "页眉区域，通常不进入正文校对"),
        LayoutSubtypeSpec("页脚", "footer", BlockType.TEXT, "页脚区域，通常不进入正文校对"),
        LayoutSubtypeSpec("页码", "number", BlockType.TEXT, "页码/编号类位置元素"),
        LayoutSubtypeSpec("脚注", "footnote", BlockType.TEXT, "脚注文本"),
    )),
)
TYPE_BUTTON_GROUP_COLUMNS = {
    "核心与结构": 2,
    "标题层级": 3,
    "图表元素": 2,
    "页边元素": 2,
}
TYPE_BUTTON_FULL_ROW_LABELS = frozenset({"chart"})
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
LAYOUT_SEARCH_TEXT_ONLY_ROLE = Qt.ItemDataRole.UserRole + 1
LAYOUT_SEARCH_PRESETS = (
    ("手动输入", "", False),
    ("中文序号标题：一、", r"^\s*[一二三四五六七八九十百千万零〇]+[、.．]", True),
    ("括号中文标题：（一）", r"^\s*[（(][一二三四五六七八九十百千万零〇]+[）)]", True),
    ("章节标题：第一章/第1节", r"^\s*第[一二三四五六七八九十百千万零〇\d]+[章节篇]", True),
    ("数字标题：1./1.1", r"^\s*\d+(?:[.．]\d+)*(?:[、.．)\)]|\s+)", True),
)


def _compact_status_text(text: str) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= STATUS_LABEL_MAX_CHARS:
        return value
    return value[: STATUS_LABEL_MAX_CHARS - 1] + "…"


def _block_type_label(block_type: BlockType) -> str:
    return BLOCK_TYPE_LABELS.get(block_type, block_type.value)


def _block_type_button_stylesheet(block_type: BlockType) -> str:
    return """
        QPushButton {{
            min-height: 32px;
            padding: 0 8px;
            border-radius: 8px;
            border: none;
            background: rgba(255, 255, 255, 150);
            color: #2C2C2C;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background: #ffffff;
            color: #2C2C2C;
        }}
        QPushButton:checked {{
            background: #2C2C2C;
            color: #ffffff;
            font-weight: 500;
        }}
        QPushButton:disabled {{
            background: #F2F0EB;
            color: #CCCCCC;
            font-weight: 400;
        }}
    """


def _block_type_group_stylesheet() -> str:
    return """
        QFrame#blockTypeGroup {
            border: none;
            border-radius: 8px;
            background: #ffffff;
        }
        QLabel#blockTypeGroupTitle {
            color: #6B6B6B;
            font-size: 10px;
            font-weight: 500;
        }
    """


def _type_badge_stylesheet(block_type: BlockType) -> str:
    return """
        QLabel#typeBadge {{
            padding: 2px 10px;
            border-radius: 10px;
            border: 1px solid rgba(44, 44, 44, 25);
            background: rgba(44, 44, 44, 12);
            color: #2C2C2C;
            font-weight: 500;
            font-size: 10px;
        }}
    """


class _PageViewDirectory(QListWidget):
    """Small page directory projection that accepts only ``PageView`` values."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(236)
        self.setMinimumWidth(196)
        self.setSpacing(10)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self._page_uids: tuple[str, ...] = ()
        self._suppress_signal = False

    def set_pages(self, pages: Iterable[PageView]) -> None:
        values = tuple(pages)
        if any(not isinstance(page, PageView) for page in values):
            raise TypeError("layout page directory requires PageView values")
        self._suppress_signal = True
        try:
            self.clear()
            self._page_uids = tuple(page.page_uid for page in values)
            for page in values:
                source = page.source_path or page.image_path
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, page.page_uid)
                item.setSizeHint(QSize(0, PAGE_ROW_H))
                self.addItem(item)
                row = PageDirectoryRow(
                    page.page_number,
                    source,
                    page.thumbnail_path or page.image_path,
                )
                self.setItemWidget(item, row)
        finally:
            self._suppress_signal = False

    def set_current_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("directory index must be an integer")
        if index < 0 or index >= len(self._page_uids):
            raise IndexError(f"directory index is out of range: {index}")
        self._suppress_signal = True
        try:
            self.setCurrentRow(index)
        finally:
            self._suppress_signal = False

    def _on_row_changed(self, index: int) -> None:
        if self._suppress_signal or index < 0 or index >= len(self._page_uids):
            return


class LayoutPanel(QWidget):
    """
    步骤2: 版面分析结果可视化。
    左侧：页面缩略图列表；中间：图像+BBox；右侧：框类型与项目统计。
    OCR 入口由用户提交当前版面后发出，不在结果展示时自动触发。
    """
    analysis_confirmed = Signal()
    page_selected = Signal(str)   # payload: stable page UID
    layout_edit_requested = Signal(object)  # immutable LayoutEditCommand
    atom_geometry_change_requested = Signal(str, object)  # atom_uid, BBox
    ocr_entry_requested = Signal(str, str)  # source, page_uid
    page_completed = Signal(str)  # stable page UID
    edits_cancelled = Signal(str)  # stable page UID
    analysis_cancel_requested = Signal()
    DRAW_SNAP_TOLERANCE = 8
    DRAW_INK_SNAP_TOLERANCE = 16

    def __init__(self, workspace: LayoutWorkspaceView | None = None, parent=None):
        super().__init__(parent)
        self._pages: list[PageView] = []
        self._current_page_idx: int = 0
        self._selected_block_uid: str | None = None
        self._atom_boxes_by_page: dict[str, tuple[OcrAtomBox, ...]] = {}
        self._page_gate_states: dict[str, tuple[str, bool, str, str]] = {}
        self._primary_actions: dict[str, tuple[str, str, bool]] = {}
        self._ink_mask_cache: dict[str, tuple[object, int, int, list[tuple[int, int, int, int, int]]]] = {}
        self._new_subtype: LayoutSubtypeSpec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"]
        self._analysis_running = False
        self._new_block_type: BlockType = self._new_subtype.block_type
        self._new_type_buttons: dict[BlockType, QPushButton] = {}
        self._new_subtype_buttons: dict[str, QPushButton] = {}
        self._selected_type_buttons: dict[BlockType, QPushButton] = {}
        self._selected_subtype_buttons: dict[str, QPushButton] = {}
        self._type_group: QButtonGroup | None = None
        self._block_search_matches: list[tuple[int, str]] = []
        self._search_text_fields_only = False
        self._build_ui()
        if workspace is not None:
            self.set_workspace(workspace)

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
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()
        self._status_lbl = QLabel("请先导入文件并运行版面分析")
        self._status_lbl.setObjectName("muted")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        # 主区域（两栏：页面列表 + 图像查看器）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        self._splitter = splitter

        # 左：页面目录 / 标题目录切换。
        self._page_list = _PageViewDirectory()
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        self._outline_tree = QTreeWidget()
        self._outline_tree.setObjectName("headingOutlineTree")
        self._outline_tree.setHeaderHidden(True)
        self._outline_tree.setIndentation(16)
        self._outline_tree.itemClicked.connect(self._on_outline_item_clicked)
        self._left_tabs = QTabWidget()
        self._left_tabs.setObjectName("layoutLeftTabs")
        self._left_tabs.addTab(self._page_list, "页面")
        self._left_tabs.addTab(self._outline_tree, "标题")
        splitter.addWidget(self._left_tabs)

        # 中：图像查看器 + 上方编辑工具条
        vw_wrap = QWidget()
        vw_wrap.setObjectName("layoutCanvasPane")
        vw_lay = QVBoxLayout(vw_wrap)
        vw_lay.setContentsMargins(0, 28, 0, 0)
        vw_lay.setSpacing(16)

        tb_wrap = QWidget()
        tb_lay = QHBoxLayout(tb_wrap)
        tb_lay.setContentsMargins(0, 0, 0, 0)

        # viewer 工具条 (悬浮胶囊)
        viewer_tb = QFrame()
        viewer_tb.setObjectName("pillToolbar")
        apply_soft_shadow(viewer_tb, blur_radius=18, y_offset=4, alpha=16)

        viewer_tb.setFixedHeight(40)
        vtl = QHBoxLayout(viewer_tb)
        vtl.setContentsMargins(16, 0, 16, 0)
        vtl.setSpacing(12)

        self._btn_atom_boxes = QPushButton("字框")
        self._btn_atom_boxes.setIcon(get_icon("layout_box", color="#6B6B6B"))
        self._btn_atom_boxes.setIconSize(QSize(16, 16))
        self._btn_atom_boxes.setObjectName("pillToolBtn")
        self._btn_atom_boxes.setCheckable(True)
        self._btn_atom_boxes.setChecked(True)
        self._btn_atom_boxes.setFixedHeight(28)
        self._btn_atom_boxes.setToolTip("\u663e\u793a/\u9690\u85cf OCR \u539f\u5b50\u6846")
        self._btn_atom_boxes.clicked.connect(self._refresh_current_page_layers)
        vtl.addWidget(self._btn_atom_boxes)

        self._btn_delete = QPushButton("删除")
        self._btn_delete.setIcon(get_icon("delete", color="#6B6B6B"))
        self._btn_delete.setIconSize(QSize(16, 16))
        self._btn_delete.setObjectName("pillToolBtn")
        self._btn_delete.setFixedHeight(28)
        self._btn_delete.setToolTip("删除右键框选中的框（Delete）")
        self._btn_delete.clicked.connect(self._delete_selected)
        vtl.addWidget(self._btn_delete)

        # 添加翻页导航到图像区域顶部
        vtl.addSpacing(16)
        self._btn_prev = QPushButton("‹")
        self._btn_prev.setObjectName("pillToolBtn")
        self._btn_prev.setFixedSize(28, 28)
        self._btn_prev.setToolTip("上一页")
        self._btn_prev.clicked.connect(lambda: self._goto_relative(-1))
        vtl.addWidget(self._btn_prev)

        self._lbl_page_no = QLabel("0 / 0")
        self._lbl_page_no.setObjectName("pageNavLabel")
        self._lbl_page_no.setMinimumWidth(48)
        self._lbl_page_no.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vtl.addWidget(self._lbl_page_no)

        self._btn_next = QPushButton("›")
        self._btn_next.setObjectName("pillToolBtn")
        self._btn_next.setFixedSize(28, 28)
        self._btn_next.setToolTip("下一页")
        self._btn_next.clicked.connect(lambda: self._goto_relative(+1))
        vtl.addWidget(self._btn_next)

        vtl.addStretch(1)
        self._selection_mode_lbl = QLabel("新建模式:")
        self._selection_mode_lbl.setObjectName("muted")
        vtl.addWidget(self._selection_mode_lbl)
        self._selection_type_status = QLabel("")
        self._selection_type_status.setObjectName("typeBadge")
        self._selection_type_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._selection_type_status.setMinimumWidth(64)
        vtl.addWidget(self._selection_type_status)

        tb_lay.addStretch(1)
        tb_lay.addWidget(viewer_tb)
        tb_lay.addStretch(1)
        vw_lay.addWidget(tb_wrap)

        self._find_dialog = self._build_find_dialog()

        self._viewer = ImageViewer()
        self._viewer.block_clicked_uid.connect(self._on_block_clicked_uid)
        self._viewer.block_edit_started_uid.connect(self._on_block_edit_started_uid)
        self._viewer.block_geometry_change_requested.connect(self._on_block_geometry_change_requested)
        self._viewer.block_created.connect(self._on_block_created)
        self._viewer.block_deleted_uid.connect(self._on_block_deleted_uid)
        self._viewer.atom_geometry_change_requested.connect(
            self._on_atom_geometry_change_requested
        )
        self._viewer.set_bbox_snapper(self._snap_current_draw_bbox)
        vw_lay.addWidget(self._viewer, 1)
        splitter.addWidget(vw_wrap)

        # 右：工具栏 + 项目统计
        right_wrap = QWidget()
        right_wrap.setObjectName("layoutRightContent")
        right_lay = QVBoxLayout(right_wrap)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        sidebar_header = QFrame()
        sidebar_header.setObjectName("layoutSidebarHeader")
        sidebar_header.setFixedHeight(44)
        sidebar_header_lay = QHBoxLayout(sidebar_header)
        sidebar_header_lay.setContentsMargins(16, 0, 16, 0)
        self._type_context_title = QLabel("新建框类型")
        self._type_context_title.setObjectName("sidebarTitle")
        sidebar_header_lay.addWidget(self._type_context_title)
        self._prop_conf = ConfidenceBadge()
        self._prop_conf.hide()
        sidebar_header_lay.addWidget(self._prop_conf)
        right_lay.addWidget(sidebar_header)

        tool_panel = QFrame()
        tool_panel.setObjectName("sidebarBar")
        tool_lay = QVBoxLayout(tool_panel)
        tool_lay.setContentsMargins(16, 16, 16, 16)
        tool_lay.setSpacing(18)

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

        self._btn_undo = QPushButton("撤销")
        self._btn_undo.setObjectName("secondaryBtn")
        self._btn_undo.setEnabled(False)
        self._btn_undo.setToolTip("撤销上一步版面编辑（Ctrl+Z）")
        self._btn_undo.clicked.connect(self._undo_last_edit)

        action_wrap = QWidget()
        action_lay = QVBoxLayout(action_wrap)
        action_lay.setContentsMargins(0, 0, 0, 0)
        action_lay.setSpacing(8)
        action_lay.addWidget(self._btn_undo)
        tool_lay.addWidget(action_wrap)

        right_lay.addWidget(tool_panel)

        stats_panel = QFrame()
        stats_panel.setObjectName("layoutStatsPanel")
        apply_soft_shadow(stats_panel, blur_radius=16, y_offset=4, alpha=14)
        stats_lay = QVBoxLayout(stats_panel)
        stats_lay.setContentsMargins(12, 10, 12, 10)
        stats_lay.setSpacing(6)
        stats_title = QLabel("项目统计")
        stats_title.setObjectName("sectionTitle")
        stats_lay.addWidget(stats_title)
        self._project_stats_lbl = QLabel("暂无项目")
        self._project_stats_lbl.setObjectName("muted")
        self._project_stats_lbl.setWordWrap(True)
        stats_lay.addWidget(self._project_stats_lbl)
        stats_lay.addStretch(1)
        right_lay.addWidget(stats_panel, 1)

        right_scroll = QScrollArea()
        right_scroll.setObjectName("layoutToolScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(380)
        right_scroll.setWidget(right_wrap)
        # Create right pane
        right_pane = QWidget()
        right_pane.setObjectName("layoutRightPane")
        right_pane_lay = QVBoxLayout(right_pane)
        right_pane_lay.setContentsMargins(0, 0, 0, 0)
        right_pane_lay.setSpacing(0)
        right_pane_lay.addWidget(right_scroll, 1)

        # Right Action Area (Replaces Bottom Bar)
        action_area = QFrame()
        action_area.setObjectName("actionArea")
        action_lay = QVBoxLayout(action_area)
        action_lay.setContentsMargins(16, 16, 16, 16)
        action_lay.setSpacing(12)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(12)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setObjectName("secondaryBtn")
        self._btn_cancel.setFixedHeight(36)
        self._btn_cancel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._btn_cancel)

        self._btn_done = QPushButton("完成本页")
        self._btn_done.setObjectName("secondaryBtn")
        self._btn_done.setFixedHeight(36)
        self._btn_done.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_done.clicked.connect(self._on_done_clicked)
        btn_row.addWidget(self._btn_done)
        action_lay.addLayout(btn_row)

        self._btn_submit = QPushButton("提交 OCR")
        self._btn_submit.setObjectName("darkBtn")
        self._btn_submit.setFixedHeight(40)
        self._btn_submit.clicked.connect(self._on_submit_clicked)
        apply_soft_shadow(self._btn_submit, blur_radius=16, y_offset=3, alpha=18)
        action_lay.addWidget(self._btn_submit)

        status_row = QVBoxLayout()
        status_row.setContentsMargins(4, 0, 4, 0)
        status_row.setSpacing(4)
        status_row.addWidget(self._status_lbl)
        self._progress_bar.setFixedHeight(16)
        status_row.addWidget(self._progress_bar)
        action_lay.addLayout(status_row)

        right_pane_lay.addWidget(action_area, 0)

        splitter.addWidget(right_pane)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([236, 980, 320])
        main_layout.addWidget(splitter, 1)
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._undo_last_edit)
        self._page_up_shortcut = QShortcut(QKeySequence("PageUp"), self)
        self._page_up_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._page_up_shortcut.activated.connect(lambda: self._goto_relative(-1))
        self._page_down_shortcut = QShortcut(QKeySequence("PageDown"), self)
        self._page_down_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._page_down_shortcut.activated.connect(lambda: self._goto_relative(1))
        self._update_page_nav()

    def _build_find_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setObjectName("layoutFindDialog")
        dialog.setWindowTitle("基于属性与文本过滤的批量赋值")
        dialog.setModal(True)
        dialog.resize(580, 450)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("layoutFindHeader")
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(24, 16, 18, 14)
        title = QLabel("基于属性与文本过滤的批量赋值")
        title.setObjectName("sidebarTitle")
        header_lay.addWidget(title)
        header_lay.addStretch(1)
        self._btn_close_find = QPushButton("×")
        self._btn_close_find.setObjectName("iconBtn")
        self._btn_close_find.setFixedSize(28, 28)
        self._btn_close_find.clicked.connect(self.hide_find_dialog)
        header_lay.addWidget(self._btn_close_find)
        root.addWidget(header)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(24, 18, 24, 24)
        body_lay.setSpacing(16)

        input_title = QLabel("包含文本特征")
        input_title.setObjectName("fieldLabel")
        body_lay.addWidget(input_title)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("输入文本或正则，例如：^一、")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textEdited.connect(self._on_search_text_edited)
        self._search_input.textChanged.connect(self._refresh_block_search)
        body_lay.addWidget(self._search_input)

        option_row = QHBoxLayout()
        option_row.setContentsMargins(0, 0, 0, 0)
        option_row.setSpacing(8)
        self._search_preset = QComboBox()
        self._search_preset.setObjectName("layoutSearchPreset")
        self._search_preset.setMinimumWidth(170)
        for label, pattern, text_fields_only in LAYOUT_SEARCH_PRESETS:
            self._search_preset.addItem(label, pattern)
            self._search_preset.setItemData(
                self._search_preset.count() - 1,
                bool(text_fields_only),
                LAYOUT_SEARCH_TEXT_ONLY_ROLE,
            )
        self._search_preset.currentIndexChanged.connect(self._on_search_preset_changed)
        option_row.addWidget(self._search_preset, 1)

        self._search_regex = QCheckBox("正则")
        self._search_regex.toggled.connect(self._refresh_block_search)
        option_row.addWidget(self._search_regex)

        self._search_case = QCheckBox("区分大小写")
        self._search_case.toggled.connect(self._refresh_block_search)
        option_row.addWidget(self._search_case)
        body_lay.addLayout(option_row)

        assign_row = QHBoxLayout()
        assign_row.setContentsMargins(0, 0, 0, 0)
        assign_row.setSpacing(16)

        source_col = QVBoxLayout()
        source_col.setContentsMargins(0, 0, 0, 0)
        source_col.setSpacing(4)
        source_label = QLabel("原分类限制")
        source_label.setObjectName("fieldLabel")
        source_col.addWidget(source_label)
        self._search_source_filter = QComboBox()
        self._search_source_filter.addItem("不限分类", "any")
        self._search_source_filter.addItem("标题", "title")
        self._search_source_filter.addItem("正文/文字", "text")
        self._search_source_filter.addItem("公式", "equation")
        self._search_source_filter.addItem("图像/图表", "figure")
        self._search_source_filter.addItem("表格", "table")
        self._search_source_filter.addItem("参考文献", "reference")
        self._search_source_filter.currentIndexChanged.connect(self._refresh_block_search)
        source_col.addWidget(self._search_source_filter)
        assign_row.addLayout(source_col, 1)

        target_col = QVBoxLayout()
        target_col.setContentsMargins(0, 0, 0, 0)
        target_col.setSpacing(4)
        target_label = QLabel("目标赋值分类")
        target_label.setObjectName("fieldLabel")
        target_col.addWidget(target_label)
        self._search_target_combo = QComboBox()
        for spec in BLOCK_SUBTYPE_BUTTON_ORDER:
            self._search_target_combo.addItem(spec.label, spec.normalized_source_label)
        self._search_target_combo.setCurrentIndex(0)
        target_col.addWidget(self._search_target_combo)
        assign_row.addLayout(target_col, 1)
        body_lay.addLayout(assign_row)

        result_row = QHBoxLayout()
        result_row.setContentsMargins(0, 0, 0, 0)
        result_row.setSpacing(8)
        self._search_results = QListWidget()
        self._search_results.setObjectName("layoutSearchResults")
        self._search_results.setMinimumHeight(130)
        self._search_results.itemClicked.connect(self._on_search_result_clicked)
        result_row.addWidget(self._search_results, 1)

        action_col = QVBoxLayout()
        action_col.setContentsMargins(0, 0, 0, 0)
        action_col.setSpacing(8)
        self._btn_preview_matches = QPushButton("预览匹配块")
        self._btn_preview_matches.setObjectName("secondaryBtn")
        self._btn_preview_matches.clicked.connect(self._preview_first_search_match)
        action_col.addWidget(self._btn_preview_matches)
        self._btn_apply_filter_type = QPushButton("批量应用属性")
        self._btn_apply_filter_type.setObjectName("primaryBtn")
        self._btn_apply_filter_type.setToolTip("把目标赋值分类批量应用到查找结果")
        self._btn_apply_filter_type.clicked.connect(self._apply_search_target_to_matches)
        action_col.addWidget(self._btn_apply_filter_type)
        self._search_count_lbl = QLabel("0")
        self._search_count_lbl.setObjectName("muted")
        action_col.addWidget(self._search_count_lbl)
        action_col.addStretch(1)
        result_row.addLayout(action_col)
        body_lay.addLayout(result_row)
        root.addWidget(body)
        return dialog

    # ------------------------------------------------------------------ public

    def set_workspace(self, workspace: LayoutWorkspaceView) -> None:
        """Display one immutable application workspace projection."""
        if not isinstance(workspace, LayoutWorkspaceView):
            raise TypeError("layout panel requires LayoutWorkspaceView")
        self._set_page_views(workspace.pages)
        if not self._pages:
            self._update_page_nav()
            return
        self._update_viewer(min(self._current_page_idx, len(self._pages) - 1))
        total_blocks = sum(self._layout_block_count(page) for page in self._pages)
        failed = sum(1 for page in self._pages if bool(page.error))
        if failed:
            self._set_status_text(
                f"共 {len(self._pages)} 页，{total_blocks} 个版面块，{failed} 页分析失败"
            )
        else:
            self._set_status_text(f"共 {len(self._pages)} 页，{total_blocks} 个版面块")

    def _set_page_views(self, inputs: Iterable[PageView]) -> None:
        values = tuple(inputs)
        if any(not isinstance(value, PageView) for value in values):
            raise TypeError("layout panel pages must be PageView values")
        if len({value.page_uid for value in values}) != len(values):
            raise ValueError("layout panel pages contain duplicate page UIDs")
        self._pages = sorted(values, key=lambda value: (value.page_number, value.page_uid))
        self._ink_mask_cache.clear()
        self._page_list.set_pages(self._pages)
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
        self._btn_run.setEnabled(bool(self._pages))
        if self._pages:
            self._page_list.set_current_index(0)
        self._update_page_nav()

    def reset(self) -> None:
        self.finish_analysis_progress()
        self._pages = []
        self._current_page_idx = 0
        self._selected_block_uid = None
        self._atom_boxes_by_page = {}
        self._page_gate_states.clear()
        self._primary_actions.clear()
        self._ink_mask_cache.clear()
        self._page_list.set_pages(())
        self._outline_tree.clear()
        self._search_results.clear()
        self._block_search_matches.clear()
        self._search_count_lbl.setText("0")
        self._find_dialog.hide()
        self._viewer.clear()
        self._set_status_text("请先导入文件并运行版面分析")
        self._btn_run.setEnabled(False)
        self._btn_undo.setEnabled(False)
        self._set_selected_type_buttons_enabled(False)
        self._prop_conf.hide()
        self._update_project_stats()
        self._update_page_nav()

    def start_analysis_progress(self, total_pages: int) -> None:
        # 进度统一由主窗口状态栏进度条展示（版面/OCR 共用），本面板只维护状态文案
        self._analysis_running = True
        self._progress_bar.hide()
        self._btn_cancel.setText("停止分析")
        self._btn_cancel.setEnabled(True)
        self._set_status_text("版面分析中…")

    def update_analysis_progress(self, current: int, total: int) -> None:
        self._set_status_text(f"版面分析中… {max(0, min(current + 1, total))}/{max(1, total)} 页")

    def update_analysis_stage(self, message: str) -> None:
        if self._analysis_running:
            self._set_status_text(f"版面分析：{message}")

    def finish_analysis_progress(self, message: str = "") -> None:
        self._analysis_running = False
        self._progress_bar.hide()
        self._btn_cancel.setText("取消")
        self._update_page_nav()
        if message:
            self._set_status_text(message)

    def set_current_page_uid(self, page_uid: str) -> None:
        for idx, page in enumerate(self._pages):
            if page.uid == page_uid:
                self._page_list.set_current_index(idx)
                self._current_page_idx = idx
                self._update_viewer(idx)
                self._update_page_nav()
                return

    def set_page_gate_state(
        self,
        page_uid: str,
        page_state: str,
        is_pending: bool,
        reason_code: str,
        reason_text: str,
    ) -> None:
        self._page_gate_states[page_uid] = (page_state, is_pending, reason_code, reason_text)
        if self._pages and self._pages[self._current_page_idx].uid == page_uid:
            self._set_status_text(reason_text)

    def set_primary_action(self, page_uid: str, action_key: str, label: str, enabled: bool) -> None:
        self._primary_actions[page_uid] = (action_key, label, enabled)
        if self._pages and self._pages[self._current_page_idx].uid == page_uid:
            self._btn_submit.setText(label)
            self._btn_submit.setEnabled(enabled)

    def show_find_dialog(self) -> None:
        self._sync_search_target_combo()
        self._find_dialog.show()
        self._find_dialog.raise_()
        self._find_dialog.activateWindow()
        self._search_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._search_input.selectAll()
        self._refresh_block_search()

    def hide_find_dialog(self) -> None:
        self._find_dialog.hide()

    def toggle_find_dialog(self) -> None:
        if self._find_dialog.isVisible():
            self.hide_find_dialog()
        else:
            self.show_find_dialog()

    def refresh_text_indexes(self) -> None:
        """Refresh derived text views after OCR/proof updates block lines."""
        self._rebuild_heading_outline()
        self._refresh_block_search()

    def _set_status_text(self, text: str) -> None:
        full = str(text or "")
        compact = _compact_status_text(full)
        self._status_lbl.setText(compact)
        self._status_lbl.setToolTip(full if compact != full else "")

    def _update_project_stats(self) -> None:
        if not hasattr(self, "_project_stats_lbl"):
            return
        if not self._pages:
            self._project_stats_lbl.setText("暂无项目")
            return
        total_pages = len(self._pages)
        analyzed_pages = sum(1 for page in self._pages if page.layout_revision is not None)
        failed_pages = sum(1 for page in self._pages if bool(page.error))
        layout_pages = [page for page in self._pages if page.layout_revision is not None]
        total_blocks = sum(self._layout_block_count(page) for page in layout_pages)
        text_ocr_blocks = sum(
            1
            for page in layout_pages
            for view in _layout_blocks(page)
            if view.ocr_policy is OcrPolicy.TEXT_OCR
        )
        total_lines = sum(_ocr_observation_line_count(page) for page in self._pages)
        counts: dict[str, int] = {}
        for page in layout_pages:
            for view in _layout_blocks(page):
                label = _block_type_label(view.block_type)
                counts[label] = counts.get(label, 0) + 1
        count_text = "，".join(
            f"{label} {count}"
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ) or "无"
        self._project_stats_lbl.setText(
            "\n".join([
                f"页面：{analyzed_pages}/{total_pages} 已分析"
                + (f"，失败 {failed_pages}" if failed_pages else ""),
                f"框：{total_blocks} 个",
                f"文字 OCR 框：{text_ocr_blocks} 个",
                f"OCR 行：{total_lines} 行",
                f"类型：{count_text}",
            ])
        )

    @staticmethod
    def _layout_block_count(page: PageView) -> int:
        return len(_layout_blocks(page))

    @staticmethod
    def _layout_view_search_fields(view: LayoutBlockPresentation) -> list[str]:
        parts: list[str] = [
            view.source_label,
            getattr(view.block_type, "value", str(view.block_type)),
        ]
        return [str(part or "").strip() for part in parts if str(part or "").strip()]

    @staticmethod
    def _layout_view_preview_text(view: LayoutBlockPresentation) -> str:
        return view.source_label or getattr(view.block_type, "value", str(view.block_type))

    @staticmethod
    def _layout_view_matches_search_source_filter(view: LayoutBlockPresentation, filter_key: str) -> bool:
        if filter_key == "any":
            return True
        label = normalize_paddle_label(view.source_label or view.block_type.value)
        if filter_key == "title":
            return LayoutPanel._is_title_like_layout_view(view)
        if filter_key == "text":
            return view.block_type == BlockType.TEXT and label not in {"header", "footer", "number", "footnote"}
        if filter_key == "equation":
            return view.block_type == BlockType.EQUATION or label in {
                "formula", "inline_formula", "display_formula", "equation",
            }
        if filter_key == "figure":
            return view.block_type in {BlockType.FIGURE, BlockType.FIGURE_CAPTION} or label in {
                "figure", "chart", "figure_title",
            }
        if filter_key == "table":
            return view.block_type in {BlockType.TABLE, BlockType.TABLE_CAPTION} or label in {
                "table", "table_title",
            }
        if filter_key == "reference":
            return view.block_type == BlockType.REFERENCE or label == "reference_content"
        return True

    def _refresh_block_search(self) -> None:
        if not hasattr(self, "_search_results"):
            return
        query = self._search_input.text() if hasattr(self, "_search_input") else ""
        query = str(query or "")
        source_filter = (
            self._search_source_filter.currentData()
            if hasattr(self, "_search_source_filter")
            else "any"
        )
        self._search_results.clear()
        self._block_search_matches.clear()
        if not query.strip() or not self._pages:
            self._search_count_lbl.setText("0")
            return

        matcher = None
        if self._search_regex.isChecked():
            flags = 0 if self._search_case.isChecked() else re.IGNORECASE
            try:
                matcher = re.compile(query, flags)
            except re.error as exc:
                self._search_count_lbl.setText("正则错误")
                self._set_status_text(f"正则表达式错误：{exc}")
                return
        else:
            needle = query if self._search_case.isChecked() else query.lower()

        for page_idx, page in enumerate(self._pages):
            if page.layout_revision is None:
                continue
            for view in _layout_blocks(page):
                if not self._layout_view_matches_search_source_filter(view, str(source_filter or "any")):
                    continue
                fields = self._layout_view_search_fields(view)
                if matcher is not None:
                    matched = any(bool(matcher.search(field)) for field in fields)
                else:
                    matched = any(
                        needle in (field if self._search_case.isChecked() else field.lower())
                        for field in fields
                    )
                if not matched:
                    continue
                match_idx = len(self._block_search_matches)
                self._block_search_matches.append((page_idx, view.uid))
                item = QListWidgetItem(self._layout_view_preview_text(view))
                item.setData(Qt.ItemDataRole.UserRole, match_idx)
                self._search_results.addItem(item)

        count = len(self._block_search_matches)
        self._search_count_lbl.setText(f"{count} 个")
        if count:
            self._set_status_text(f"查找到 {count} 个版面块")

    def _preview_first_search_match(self) -> None:
        self._refresh_block_search()
        if not self._block_search_matches:
            self._set_status_text("没有匹配的版面块")
            return
        page_idx, block_uid = self._block_search_matches[0]
        self._focus_block_uid(page_idx, block_uid)
        self._set_status_text(f"已预览第 1 个匹配块，共 {len(self._block_search_matches)} 个")

    def _sync_search_target_combo(self) -> None:
        if not hasattr(self, "_search_target_combo"):
            return
        label = self._current_checked_subtype().normalized_source_label
        index = self._search_target_combo.findData(label)
        if index >= 0:
            self._search_target_combo.setCurrentIndex(index)

    def _apply_search_target_to_matches(self) -> None:
        label = self._search_target_combo.currentData() if hasattr(self, "_search_target_combo") else ""
        subtype = DEFAULT_SUBTYPE_BY_SOURCE_LABEL.get(normalize_paddle_label(label))
        if subtype is None:
            self._set_status_text("目标赋值分类无效")
            return
        self._apply_current_type_to_search_matches(subtype)

    def _on_search_preset_changed(self, index: int) -> None:
        if not hasattr(self, "_search_preset"):
            return
        pattern = self._search_preset.itemData(index)
        self._search_text_fields_only = bool(
            self._search_preset.itemData(index, LAYOUT_SEARCH_TEXT_ONLY_ROLE)
        )
        if not pattern:
            self._search_text_fields_only = False
            self._refresh_block_search()
            return
        self._search_regex.setChecked(True)
        if self._search_input.text() != pattern:
            self._search_input.setText(pattern)
        else:
            self._refresh_block_search()
        self._search_input.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_search_text_edited(self, _text: str) -> None:
        self._search_text_fields_only = False

    def _on_search_result_clicked(self, item: QListWidgetItem) -> None:
        match_idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(match_idx, int) or not (0 <= match_idx < len(self._block_search_matches)):
            return
        page_idx, block_uid = self._block_search_matches[match_idx]
        self._focus_block_uid(page_idx, block_uid)

    def _focus_block_uid(self, page_idx: int, block_uid: str) -> None:
        if not (0 <= page_idx < len(self._pages)):
            return
        view = None
        for candidate in _layout_blocks(self._pages[page_idx]):
            if candidate.uid == block_uid:
                view = candidate
                break
        if view is None:
            return
        self._current_page_idx = page_idx
        self._page_list.set_current_index(page_idx)
        self._update_viewer(page_idx)
        if self._viewer.select_block_uid(block_uid):
            self._on_block_clicked_uid(block_uid)
        else:
            self._viewer.highlight_bbox(view.bbox, zoom=True)
            self._selected_block_uid = block_uid
            self._sync_selected_type_buttons(view.block)

    def _current_checked_subtype(self) -> LayoutSubtypeSpec:
        for source_label, button in self._new_subtype_buttons.items():
            if button.isChecked():
                spec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL.get(source_label)
                if spec is not None:
                    return spec
        return self._new_subtype

    def _apply_current_type_to_search_matches(self, subtype: LayoutSubtypeSpec | None = None) -> None:
        if not self._block_search_matches:
            self._set_status_text("没有可应用的查找结果")
            return
        subtype = subtype or self._current_checked_subtype()
        affected: dict[int, list[str]] = {}
        for page_idx, block_uid in self._block_search_matches:
            if 0 <= page_idx < len(self._pages) and block_uid:
                affected.setdefault(page_idx, []).append(block_uid)
        if not affected:
            return
        changed = 0
        for page_idx, block_uids in affected.items():
            page = self._pages[page_idx]
            for block_uid in block_uids:
                block = self._layout_block_by_uid(page, block_uid)
                if block is None:
                    continue
                source_label = self._source_label_for_subtype(page, block, subtype)
                is_changed = (
                    block.block_type != subtype.block_type
                    or normalize_paddle_label(block.source_label) != normalize_paddle_label(source_label)
                )
                if not is_changed:
                    continue
                revision = page.layout_revision
                if revision is None:
                    continue
                self._emit_layout_command(LayoutEditCommand.change_type(
                    page.page_uid,
                    revision,
                    block.block_uid,
                    block_type=subtype.block_type,
                    source_label=source_label,
                ))
                changed += 1
        self._set_status_text(f"已将 {changed} 个查找结果设为 {subtype.label}")

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
        column.setSpacing(18)
        for group_title, subtype_specs in BLOCK_TYPE_BUTTON_GROUPS:
            section_wrap = QWidget()
            section_lay = QVBoxLayout(section_wrap)
            section_lay.setContentsMargins(0, 0, 0, 0)
            section_lay.setSpacing(8)

            title = QLabel(group_title)
            title.setObjectName("blockTypeGroupTitle")
            section_lay.addWidget(title)

            # Floating White Card Wrap
            grid_wrap = QFrame()
            grid_wrap.setObjectName("whiteCard")
            apply_soft_shadow(grid_wrap, blur_radius=18, y_offset=4, alpha=14)
            grid_lay = QVBoxLayout(grid_wrap)
            grid_lay.setContentsMargins(12, 12, 12, 12)

            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(8)
            columns = TYPE_BUTTON_GROUP_COLUMNS.get(group_title, 2)
            row = 0
            col = 0
            for spec in subtype_specs:
                button = QPushButton(spec.label)
                button.setObjectName("blockTypeButton")
                button.setCheckable(True)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                button.setFixedHeight(32)
                tooltip = f"{group_title} · {spec.label} / {spec.source_label} → {spec.block_type.value}"
                if spec.note:
                    tooltip = f"{tooltip}\n{spec.note}"
                button.setToolTip(tooltip)
                button.clicked.connect(lambda _checked=False, value=spec: callback(value))
                group.addButton(button)
                subtype_buttons[spec.normalized_source_label] = button
                type_buttons.setdefault(spec.block_type, button)
                span = columns if spec.normalized_source_label in TYPE_BUTTON_FULL_ROW_LABELS else 1
                if span > 1 and col != 0:
                    row += 1
                    col = 0
                grid.addWidget(button, row, col, 1, min(span, columns))
                col += span
                if col >= columns:
                    col = 0
                    row += 1
            grid_lay.addLayout(grid)
            section_lay.addWidget(grid_wrap)
            column.addWidget(section_wrap)

        column.addStretch(1)
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

    def _sync_type_buttons(self, block: BlockView | None) -> None:
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
        self._set_type_buttons_enabled(True)
        self._type_context_title.setText("选中框类型")
        self._set_type_button_checked(
            self._type_group,
            self._new_subtype_buttons,
            self._button_source_label_for_block(block),
        )
        self._update_selection_type_status(block)

    def _sync_selected_type_buttons(self, block: BlockView | None) -> None:
        self._sync_type_buttons(block)

    def _set_new_block_type(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        self._new_subtype = self._coerce_subtype_spec(subtype, DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"])
        self._new_block_type = self._new_subtype.block_type
        if self._selected_block_uid is None:
            self._sync_type_buttons(None)

    def _on_type_button_clicked(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        if self._selected_block_for_current_page() is not None:
            self._on_selected_type_button_clicked(subtype)
            return
        self._set_new_block_type(subtype)

    def _update_selection_type_status(self, block: BlockView | None) -> None:
        if block is None:
            spec = self._new_subtype
            self._selection_mode_lbl.setText("新建模式:")
            self._selection_type_status.setText(_block_type_label(spec.block_type))
            self._selection_type_status.setToolTip(f"当前新建框类型：{spec.label} / {spec.source_label} → {spec.block_type.value}")
            return
        label = self._button_source_label_for_block(block)
        spec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL.get(normalize_paddle_label(label))
        if spec is not None:
            badge = spec.label
            badge_type = spec.block_type
            tooltip = f"当前选中框属性：{spec.label} / {spec.source_label} → {spec.block_type.value}"
        else:
            raw_label = block.source_label or block.block_type.value
            badge = _block_type_label(block.block_type)
            badge_type = block.block_type
            tooltip = f"当前选中框属性：{raw_label} → {block.block_type.value}"
        self._selection_mode_lbl.setText("选中类型:")
        self._selection_type_status.setText(_block_type_label(badge_type))
        self._selection_type_status.setToolTip(tooltip)

    @staticmethod
    def _heading_level_for_layout_view(view: LayoutBlockPresentation) -> int:
        label = normalize_paddle_label(view.source_label)
        match = re.fullmatch(r"heading_([1-6])", label)
        if match:
            return int(match.group(1))
        if label == "doc_title":
            return 1
        if label in {"paragraph_title", "section_title", "chapter_title", "title"}:
            return 0
        return 0

    @staticmethod
    def _is_title_like_layout_view(view: LayoutBlockPresentation) -> bool:
        label = normalize_paddle_label(view.source_label)
        return view.block_type == BlockType.TITLE or label in {
            "doc_title",
            "paragraph_title",
            "section_title",
            "chapter_title",
            "title",
        } or bool(re.fullmatch(r"heading_[1-6]", label))

    def _rebuild_heading_outline(self) -> None:
        if not hasattr(self, "_outline_tree"):
            return
        self._outline_tree.clear()
        stack: list[tuple[int, QTreeWidgetItem]] = []
        for page_idx, page in enumerate(self._pages):
            if page.layout_revision is None:
                continue
            heading_blocks = [
                view
                for view in _layout_blocks(page)
                if self._is_title_like_layout_view(view)
            ]
            if not heading_blocks:
                continue
            for view in sorted(heading_blocks, key=lambda item: (item.bbox.y, item.bbox.x, item.order)):
                level = self._heading_level_for_layout_view(view)
                outline_level = level if 1 <= level <= 6 else 1
                level_text = f"H{level}" if level else "标题候选"
                text = self._layout_view_preview_text(view)
                item = QTreeWidgetItem([text])
                item.setData(0, Qt.ItemDataRole.UserRole, (page_idx, view.uid))
                item.setToolTip(0, f"级别：{level_text}\n第 {page.page_number} 页")
                while stack and stack[-1][0] >= outline_level:
                    stack.pop()
                if stack:
                    stack[-1][1].addChild(item)
                else:
                    self._outline_tree.addTopLevelItem(item)
                stack.append((outline_level, item))
        self._outline_tree.expandAll()

    def _on_outline_item_clicked(self, item: QTreeWidgetItem) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        page_idx, block_uid = payload
        if not isinstance(page_idx, int) or not (0 <= page_idx < len(self._pages)):
            return
        if isinstance(block_uid, str) and block_uid:
            self._focus_block_uid(page_idx, block_uid)

    # ------------------------------------------------------------------ private

    def _request_analysis(self) -> None:
        """触发版面分析（由主窗口的 run_button.clicked 同时连接），显示进度条。"""
        self.start_analysis_progress(len(self._pages))

    def _on_page_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._pages):
            self._current_page_idx = idx
            self._update_viewer(idx)
            self._update_page_nav()
            self.page_selected.emit(self._pages[idx].uid)

    def _update_viewer(self, idx: int) -> None:
        if not self._pages:
            return
        page = self._pages[idx]
        self._viewer.set_image(page.image_path)
        if page.layout_revision is not None:
            self._show_page_layers(page)
        elif page.error:
            self._set_status_text(f"第 {page.page_number} 页分析失败：{page.error}")
        self._selected_block_uid = None
        self._sync_selected_type_buttons(None)
        self._prop_conf.hide()
        self._update_project_stats()
        gate = self._page_gate_states.get(page.uid)
        if gate is not None:
            self._set_status_text(gate[3])
        action = self._primary_actions.get(page.uid)
        if action is not None:
            self._btn_submit.setText(action[1])
            self._btn_submit.setEnabled(action[2])

    def _refresh_current_page_layers(self) -> None:
        """Refresh layout overlays without reloading the image or resetting zoom."""
        if not self._pages:
            return
        self._show_page_layers(self._pages[self._current_page_idx])

    def _clear_selection_ui(self, page: PageView) -> None:
        self._selected_block_uid = None
        self._sync_selected_type_buttons(None)
        self._prop_conf.hide()
        self._update_project_stats()

    def _on_block_clicked_uid(self, block_uid: str) -> None:
        if not block_uid or not self._pages:
            return
        page = self._pages[self._current_page_idx]
        block = self._layout_block_by_uid(page, block_uid)
        if block is None:
            return
        self._selected_block_uid = block_uid
        self._sync_selected_type_buttons(block)
        self._prop_conf.set_score(_ocr_observation_avg_confidence(page, block.uid))

    def _on_block_edit_started_uid(self, block_uid: str) -> None:
        if not block_uid or not self._pages:
            return
        block = self._layout_block_by_uid(self._pages[self._current_page_idx], block_uid)
        if block is None:
            return

    def _layout_block_by_uid(
        self,
        page: PageView,
        block_uid: str,
    ) -> BlockView | None:
        for view in page.blocks:
            if view.uid == block_uid:
                return view
        return None

    def _selected_block_for_current_page(self) -> BlockView | None:
        if self._selected_block_uid is None or not self._pages:
            return None
        page = self._pages[self._current_page_idx]
        return self._layout_block_by_uid(page, self._selected_block_uid)

    def _layout_revision(self, page: PageView) -> int | None:
        if page.layout_revision is None:
            self._set_status_text("当前页面没有可编辑的版面快照")
            return None
        return page.layout_revision

    def _emit_layout_command(self, command: LayoutEditCommand) -> LayoutEditCommand:
        if not isinstance(command, LayoutEditCommand):
            raise TypeError("layout panel emits only application LayoutEditCommand")
        self.layout_edit_requested.emit(command)
        return command

    def _on_block_geometry_change_requested(self, block_uid: str, bbox: BBox) -> None:
        if not block_uid or not self._pages:
            return
        page = self._pages[self._current_page_idx]
        block = self._layout_block_by_uid(page, block_uid)
        if block is None:
            return
        revision = self._layout_revision(page)
        if revision is None:
            return
        command_factory = LayoutEditCommand.move if (
            bbox.w == block.bbox.w and bbox.h == block.bbox.h
        ) else LayoutEditCommand.resize
        self._emit_layout_command(command_factory(
            page.page_uid,
            revision,
            block.block_uid,
            bbox,
        ))
        self._set_status_text("已发出版面几何编辑请求")

    def _on_block_created(self, bbox: BBox) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        bbox = self._snap_drawn_bbox(page, bbox)
        if bbox.area <= 0:
            return
        subtype = self._new_subtype
        bt = self._coerce_block_type(subtype.block_type, BlockType.TEXT)
        source_label = self._source_label_for_bbox_subtype(page, bbox, subtype)
        intersecting = self._draw_merge_candidates(
            self._blocks_intersecting_bbox(page, bbox),
            bt,
        )
        revision = self._layout_revision(page)
        if revision is None:
            return
        if intersecting:
            self._emit_layout_command(LayoutEditCommand.merge(
                page.page_uid,
                revision,
                [block.block_uid for block in intersecting],
                bbox,
                bt,
                source_label,
            ))
            self._set_status_text("已发出合并版面框请求")
            return
        self._emit_layout_command(LayoutEditCommand.draw(
            page.page_uid,
            revision,
            bbox,
            bt,
            source_label,
            new_block_uid=f"block-{uuid4().hex}",
        ))
        self._set_status_text("已发出新建版面框请求")

    def _on_block_deleted_uid(self, block_uid: str) -> None:
        """Emit a delete intent addressed to the current layout revision."""
        if not block_uid or not self._pages:
            return
        page = self._pages[self._current_page_idx]
        revision = self._layout_revision(page)
        if revision is None:
            return
        self._emit_layout_command(LayoutEditCommand.delete(
            page.page_uid,
            revision,
            block_uid,
        ))
        self._set_status_text("已发出删除版面框请求")

    def _delete_selected(self) -> None:
        """顶部栏删除框按钮。"""
        if not self._viewer.selected_block_uids():
            self._set_status_text("请先用右键拉框选中要删除的框")
            return
        self._viewer.delete_selected()

    def _on_selected_type_button_clicked(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        selected_block = self._selected_block_for_current_page()
        if selected_block is None:
            self._selected_block_uid = None
            self._sync_selected_type_buttons(None)
            return
        new_subtype = self._coerce_subtype_spec(subtype, DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"])
        new_label = new_subtype.normalized_source_label
        current_label = self._button_source_label_for_block(selected_block)
        if selected_block.block_type == new_subtype.block_type and current_label == new_label:
            self._sync_selected_type_buttons(selected_block)
            return
        page = self._pages[self._current_page_idx]
        source_label = self._source_label_for_subtype(page, selected_block, new_subtype)
        revision = self._layout_revision(page)
        if revision is None:
            return
        self._emit_layout_command(LayoutEditCommand.change_type(
            page.page_uid,
            revision,
            selected_block.block_uid,
            new_subtype.block_type,
            source_label,
        ))
        self._set_status_text("已发出版面类型变更请求")

    def _on_atom_geometry_change_requested(self, atom_uid: str, bbox: BBox) -> None:
        if not atom_uid or not isinstance(bbox, BBox):
            return
        self.atom_geometry_change_requested.emit(atom_uid, bbox)

    def _show_page_layers(self, page: PageView) -> None:
        if page.layout_revision is None:
            self._viewer.clear_overlays()
            return
        atom_boxes = self._atom_boxes_by_page.get(page.uid, ())
        self._viewer.show_layout_blocks(page.blocks, atom_boxes=atom_boxes)
        self._viewer.show_readonly_overlays([])
        self._viewer.set_block_frame_occlusions(atom_boxes)
        self._viewer.show_atom_boxes(
            atom_boxes if self._btn_atom_boxes.isChecked() else (),
            editable=False,
        )

    def set_ocr_workspace(self, workspace) -> None:
        """Receive immutable OCR observations used by the atom-box overlay."""
        from app.application.ocr_workspace import OcrWorkspaceView

        if workspace is not None and not isinstance(workspace, OcrWorkspaceView):
            raise TypeError("layout panel OCR overlay requires OcrWorkspaceView or None")
        boxes: dict[str, tuple[OcrAtomBox, ...]] = {}
        if workspace is not None:
            for page in workspace.pages:
                page_boxes: list[OcrAtomBox] = []
                for region in page.regions:
                    for line in region.lines:
                        for atom in line.atoms:
                            if atom.bbox.w <= 0 or atom.bbox.h <= 0:
                                # 退化几何的 atom 无法入画（真实数据存在），跳过
                                continue
                            confidence = float(atom.confidence)
                            if confidence > 1.0:
                                confidence = confidence / 100.0
                            confidence = max(0.0, min(1.0, confidence))
                            page_boxes.append(
                                OcrAtomBox(
                                    uid=atom.atom_uid,
                                    bbox=atom.bbox,
                                    text=atom.text,
                                    confidence=confidence,
                                    line_uid=line.line_uid,
                                    block_uid=region.block_uid or "",
                                )
                            )
                boxes[page.page_uid] = tuple(page_boxes)
        self._atom_boxes_by_page = boxes
        if self._pages:
            self._refresh_current_page_layers()

    def _select_block_uid_for_edit(self, block_uid: str) -> None:
        self._viewer.select_block_uid(block_uid)
        self._on_block_clicked_uid(block_uid)

    def _set_status_text_for_layout_edit_command(self, command: LayoutEditCommand) -> None:
        status_text = {
            "draw": "已发出新建版面框请求",
            "delete": "已发出删除版面框请求",
            "move": "已发出版面移动请求",
            "resize": "已发出版面缩放请求",
            "change_type": "已发出版面类型变更请求",
            "merge": "已发出合并版面框请求",
        }.get(command.op, "已发出版面编辑请求")
        self._set_status_text(status_text)

    def _undo_last_edit(self) -> None:
        self._set_status_text("撤销由应用层处理")

    @staticmethod
    def _bbox_hits_frame(a: BBox, b: BBox, tolerance: int = 6) -> bool:
        if not (a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2):
            return False
        if b.w <= tolerance * 2 or b.h <= tolerance * 2:
            return True
        bands = (
            BBox(b.x, b.y, b.w, tolerance),
            BBox(b.x, b.y2 - tolerance, b.w, tolerance),
            BBox(b.x, b.y, tolerance, b.h),
            BBox(b.x2 - tolerance, b.y, tolerance, b.h),
        )
        return any(a.x1 < band.x2 and band.x1 < a.x2 and a.y1 < band.y2 and band.y1 < a.y2 for band in bands)

    def _snap_drawn_bbox(self, page: PageView, bbox: BBox) -> BBox:
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

    def _snap_bbox_to_image_ink(self, page: PageView, bbox: BBox) -> Optional[BBox]:
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

    def _get_page_ink_mask(
        self,
        page: PageView,
    ) -> Optional[tuple[object, int, int, list[tuple[int, int, int, int, int]]]]:
        image_path = page.image_path
        if not image_path:
            return None
        key = str(image_path)
        cached = self._ink_mask_cache.get(key)
        if cached is not None:
            return cached
        try:
            import cv2
            gray = read_cv_image(key, cv2.IMREAD_GRAYSCALE)
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

    def _blocks_intersecting_bbox(
        self,
        page: PageView,
        bbox: BBox,
    ) -> list[BlockView]:
        blocks: list[BlockView] = []
        for view in _layout_blocks(page):
            if self._bbox_hits_frame(bbox, view.bbox):
                blocks.append(view.block)
        return blocks

    @staticmethod
    def _draw_merge_candidates(
        blocks: list[BlockView],
        block_type: BlockType,
    ) -> list[BlockView]:
        target_structural = block_type in STRUCTURAL_DRAW_BLOCK_TYPES
        candidates: list[BlockView] = []
        for block in blocks:
            existing_structural = block.block_type in STRUCTURAL_DRAW_BLOCK_TYPES
            if target_structural != existing_structural:
                continue
            if target_structural and block.block_type != block_type:
                continue
            candidates.append(block)
        return candidates

    def _source_label_for_subtype(
        self,
        page: PageView,
        block: BlockView,
        subtype: LayoutSubtypeSpec,
    ) -> str:
        if normalize_paddle_label(subtype.source_label) == "formula":
            return self._infer_formula_source_label_for_bbox(page, block.bbox, exclude_uid=block.uid)
        return subtype.source_label

    def _source_label_for_bbox_subtype(
        self,
        page: PageView,
        bbox: BBox,
        subtype: LayoutSubtypeSpec,
    ) -> str:
        if normalize_paddle_label(subtype.source_label) == "formula":
            return self._infer_formula_source_label_for_bbox(page, bbox)
        return subtype.source_label

    def _infer_formula_source_label_for_bbox(
        self,
        page: PageView,
        bbox: BBox,
        *,
        exclude_uid: str = "",
    ) -> str:
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        for view in _layout_blocks(page):
            if view.uid == exclude_uid or view.block_type != BlockType.TEXT:
                continue
            if not (view.bbox.x1 <= cx <= view.bbox.x2 and view.bbox.y1 <= cy <= view.bbox.y2):
                continue
            return "inline_formula"
        return "display_formula"

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
    def _button_source_label_for_block(block: BlockView) -> str:
        normalized = normalize_paddle_label(block.source_label)
        if normalized in {
            "display_formula",
            "inline_formula",
            "formula_number",
            "equation",
            "formula",
        }:
            return "formula"
        if normalized in DEFAULT_SUBTYPE_BY_SOURCE_LABEL:
            return normalized
        if block.block_type == BlockType.TITLE:
            return ""
        spec = DEFAULT_SUBTYPE_BY_BLOCK_TYPE.get(block.block_type)
        return spec.source_label if spec is not None else ""

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
            page_uid = self._pages[self._current_page_idx].uid
            action = self._primary_actions.get(page_uid)
            if action is not None:
                self._btn_submit.setText(action[1])
                self._btn_submit.setEnabled(action[2])
            else:
                self._btn_submit.setText("提交 OCR")
                self._btn_submit.setProperty("class", "darkBtn")
                self._btn_submit.setEnabled(True)
        else:
            self._btn_submit.setText("提交 OCR")
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
            self.page_selected.emit(self._pages[new_idx].uid)

    def _on_done_clicked(self) -> None:
        """完成本页编辑（占位：发出信号供控制器处理；当前仅状态提示）。"""
        if not self._pages:
            return
        self._set_status_text(f"第 {self._pages[self._current_page_idx].page_number} 页编辑已记录")
        self.page_completed.emit(self._pages[self._current_page_idx].uid)

    def _on_cancel_clicked(self) -> None:
        """取消本页未提交的编辑（占位：发出信号供控制器处理）。"""
        if self._analysis_running:
            self._set_status_text("正在停止版面分析…")
            self.analysis_cancel_requested.emit()
            return
        if not self._pages:
            return
        self._set_status_text(f"已取消第 {self._pages[self._current_page_idx].page_number} 页编辑")
        self.edits_cancelled.emit(self._pages[self._current_page_idx].uid)

    def _on_submit_clicked(self) -> None:
        """提交版面分析结果，进入 OCR 阶段。"""
        if not self._pages:
            return
        page_uid = self._pages[self._current_page_idx].uid
        self.ocr_entry_requested.emit("layout_submit", page_uid)
        self.analysis_confirmed.emit()

    @property
    def run_button(self) -> QPushButton:
        return self._btn_run
