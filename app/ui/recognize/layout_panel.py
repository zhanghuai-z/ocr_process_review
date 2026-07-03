"""版面分析面板：图像 + BBox 叠加可视化，右侧提供框类型与项目统计。"""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QKeySequence, QShortcut, QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget
)

from app.utils.icon_manager import get_icon
from app.core.bbox_extraction import bbox_from_variant
from app.models.block_state import is_ocr_text_invalidated
from app.core.inline_formula_edit_state import (
    handled_inline_formula_origin_bboxes,
    inline_formula_origin_bbox,
)
from app.core.paddle_labels import normalize_paddle_label
from app.core.proof_line_facts import proof_display_text, proof_search_texts
from app.core.raw_ocr_artifact import raw_block_payload, raw_layout_records
from app.core.paddle_line_routing import (
    ROUTE_SUBBLOCKS_FIELD,
    block_text,
    formula_texts_by_subblock_bbox,
    line_routes_for_block,
)
from app.core.ocr_ir import is_formula_marker_token
from app.core.proof_char_text import char_display_text
from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Page
from app.models.ocr_observation import block_ocr_lines
from app.services.layout_edit_service import LayoutEditResult, LayoutEditService
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.effects import apply_soft_shadow

STATUS_LABEL_MAX_CHARS = 96
STRUCTURAL_DRAW_BLOCK_TYPES = {BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE}
INLINE_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


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


class LayoutPanel(QWidget):
    """
    步骤2: 版面分析结果可视化。
    左侧：页面缩略图列表；中间：图像+BBox；右侧：框类型与项目统计。
    OCR 入口由用户提交当前版面后发出，不在结果展示时自动触发。
    """
    analysis_confirmed = Signal()
    page_selected = Signal(int)   # payload: page_number
    geometry_changed = Signal()
    block_contract_changed = Signal(int, str)  # page_number, change_kind
    ocr_entry_requested = Signal(str, int)  # source, page_number
    page_completed = Signal(int)  # 用户点「完成」时（payload: page idx）
    edits_cancelled = Signal(int) # 用户点「取消」时（payload: page idx）
    analysis_cancel_requested = Signal()
    DRAW_SNAP_TOLERANCE = 8
    DRAW_INK_SNAP_TOLERANCE = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._selected_block: Optional[Block] = None
        self._page_gate_states: dict[int, tuple[str, bool, str, str]] = {}
        self._primary_actions: dict[int, tuple[str, str, bool]] = {}
        self._layout_edit_service = LayoutEditService()
        self._undo_stack: list[list[tuple[int, list[Block]]]] = []
        self._layout_edit_start_state: dict[str, dict] = {}
        self._ink_mask_cache: dict[str, tuple[object, int, int, list[tuple[int, int, int, int, int]]]] = {}
        self._new_subtype: LayoutSubtypeSpec = DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"]
        self._analysis_running = False
        self._new_block_type: BlockType = self._new_subtype.block_type
        self._new_type_buttons: dict[BlockType, QPushButton] = {}
        self._new_subtype_buttons: dict[str, QPushButton] = {}
        self._selected_type_buttons: dict[BlockType, QPushButton] = {}
        self._selected_subtype_buttons: dict[str, QPushButton] = {}
        self._type_group: QButtonGroup | None = None
        self._block_search_matches: list[tuple[int, Block]] = []
        self._search_text_fields_only = False
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
        from app.ui.widgets.page_directory import PageDirectoryList
        self._page_list = PageDirectoryList()
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

        self._btn_char_boxes = QPushButton("字框")
        self._btn_char_boxes.setIcon(get_icon("layout_box", color="#6B6B6B"))
        self._btn_char_boxes.setIconSize(QSize(16, 16))
        self._btn_char_boxes.setObjectName("pillToolBtn")
        self._btn_char_boxes.setCheckable(True)
        self._btn_char_boxes.setChecked(True)
        self._btn_char_boxes.setFixedHeight(28)
        self._btn_char_boxes.setToolTip("\u663e\u793a/\u9690\u85cf OCR \u5b57\u6846")
        self._btn_char_boxes.clicked.connect(self._refresh_current_page_layers)
        vtl.addWidget(self._btn_char_boxes)

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
        self._viewer.block_clicked.connect(self._on_block_clicked)
        self._viewer.block_edit_started.connect(self._on_block_edit_started)
        self._viewer.block_moved.connect(self._on_block_moved)
        self._viewer.block_created.connect(self._on_block_created)
        self._viewer.block_deleted.connect(self._on_block_deleted)
        self._viewer.char_bbox_moved.connect(self._on_char_bbox_moved)
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

    def set_pages(self, pages: List[Page]) -> None:
        """设置待分析的页面（已加载图片路径）。"""
        self._pages = pages
        self._ink_mask_cache.clear()
        self._page_list.set_pages(pages)
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
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

    def show_analysis_result(self, pages: List[Page]) -> None:
        """版面分析完成后，更新显示并保持当前选中页。"""
        self.finish_analysis_progress()
        self._pages = pages
        self._ink_mask_cache.clear()
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
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
        self._analysis_running = True
        self._progress_bar.setRange(0, max(1, total_pages))
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.show()
        self._btn_cancel.setText("停止分析")
        self._btn_cancel.setEnabled(True)
        self._set_status_text("版面分析中…")

    def update_analysis_progress(self, current: int, total: int) -> None:
        current_done = max(0, min(current + 1, total))
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(current_done)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.show()
        self._set_status_text("版面分析中…")

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
        analyzed_pages = sum(1 for page in self._pages if page.is_analyzed)
        failed_pages = sum(1 for page in self._pages if page.error_message)
        total_blocks = sum(len(page.blocks) for page in self._pages)
        text_ocr_blocks = sum(len(page.text_ocr_blocks) for page in self._pages)
        total_lines = sum(page.total_lines for page in self._pages)
        counts: dict[str, int] = {}
        for page in self._pages:
            for block in page.blocks:
                label = _block_type_label(block.block_type)
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
    def _payload_strings(value: object):
        if isinstance(value, dict):
            for item in value.values():
                yield from LayoutPanel._payload_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from LayoutPanel._payload_strings(item)
        elif value is not None:
            text = str(value).strip()
            if text:
                yield text

    @staticmethod
    def _block_search_fields(block: Block) -> list[str]:
        parts: list[str] = [
            block.source_label,
            getattr(block.block_type, "value", str(block.block_type)),
            block.note,
        ]
        for line in block_ocr_lines(block):
            parts.extend(proof_search_texts(line))
        parts.extend(LayoutPanel._payload_strings(raw_block_payload(block)))
        return [str(part or "").strip() for part in parts if str(part or "").strip()]

    @staticmethod
    def _block_text_search_fields(block: Block) -> list[str]:
        parts: list[str] = []
        for line in block_ocr_lines(block):
            parts.extend(proof_search_texts(line))
        if block.note:
            parts.append(block.note.split("|", 1)[0])
        return [str(part or "").strip() for part in parts if str(part or "").strip()]

    @staticmethod
    def _block_display_text(block: Block) -> str:
        return " ".join(LayoutPanel._block_search_fields(block))

    @staticmethod
    def _block_preview_text(block: Block) -> str:
        for line in block_ocr_lines(block):
            text = proof_display_text(line)
            if text:
                return _compact_status_text(text)
        if block.note:
            return _compact_status_text(block.note.split("|", 1)[0])
        return block.source_label or getattr(block.block_type, "value", str(block.block_type))

    @staticmethod
    def _block_matches_search_source_filter(block: Block, filter_key: str) -> bool:
        if filter_key == "any":
            return True
        label = normalize_paddle_label(block.source_label or block.block_type.value)
        if filter_key == "title":
            return LayoutPanel._is_title_like_block(block)
        if filter_key == "text":
            return block.block_type == BlockType.TEXT and label not in {"header", "footer", "number", "footnote"}
        if filter_key == "equation":
            return block.block_type == BlockType.EQUATION or label in {
                "formula", "inline_formula", "display_formula", "equation",
            }
        if filter_key == "figure":
            return block.block_type in {BlockType.FIGURE, BlockType.FIGURE_CAPTION} or label in {
                "figure", "chart", "figure_title",
            }
        if filter_key == "table":
            return block.block_type in {BlockType.TABLE, BlockType.TABLE_CAPTION} or label in {
                "table", "table_title",
            }
        if filter_key == "reference":
            return block.block_type == BlockType.REFERENCE or label == "reference_content"
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
            for block in page.blocks:
                if not self._block_matches_search_source_filter(block, str(source_filter or "any")):
                    continue
                fields = (
                    self._block_text_search_fields(block)
                    if self._search_text_fields_only
                    else self._block_search_fields(block)
                )
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
                self._block_search_matches.append((page_idx, block))
                item = QListWidgetItem(self._block_preview_text(block))
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
        page_idx, block = self._block_search_matches[0]
        self._focus_block(page_idx, block)
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
        page_idx, block = self._block_search_matches[match_idx]
        self._focus_block(page_idx, block)

    def _focus_block(self, page_idx: int, block: Block) -> None:
        if not (0 <= page_idx < len(self._pages)):
            return
        self._current_page_idx = page_idx
        self._page_list.set_current_index(page_idx)
        self._update_viewer(page_idx)
        if self._viewer.select_block(block):
            self._on_block_clicked(block)
        else:
            self._viewer.highlight_bbox(block.bbox, zoom=True)
            self._selected_block = block
            self._sync_selected_type_buttons(block)

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
        affected: dict[int, list[Block]] = {}
        for page_idx, block in self._block_search_matches:
            if 0 <= page_idx < len(self._pages):
                affected.setdefault(page_idx, []).append(block)
        if not affected:
            return
        self._push_undo_snapshot_for_pages(affected.keys())
        changed = 0
        for page_idx, blocks in affected.items():
            page = self._pages[page_idx]
            for block in blocks:
                source_label = self._source_label_for_subtype(page, block, subtype)
                is_changed = (
                    block.block_type != subtype.block_type
                    or normalize_paddle_label(block.source_label) != normalize_paddle_label(source_label)
                )
                if not is_changed:
                    continue
                self._layout_edit_service.change_block_kind(
                    page,
                    block,
                    block_type=subtype.block_type,
                    source_label=source_label,
                )
                if is_changed:
                    changed += 1
            for order, block in enumerate(page.blocks):
                block.order = order
            self.block_contract_changed.emit(page.page_number, "block_type_changed")
        self._show_page_layers(self._pages[self._current_page_idx])
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self.geometry_changed.emit()
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
        self._set_type_buttons_enabled(True)
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
    def _heading_level_for_block(block: Block) -> int:
        label = normalize_paddle_label(block.source_label)
        match = re.fullmatch(r"heading_([1-6])", label)
        if match:
            return int(match.group(1))
        if label == "doc_title":
            return 1
        if label in {"paragraph_title", "section_title", "chapter_title", "title"}:
            return 0
        return 0

    @staticmethod
    def _is_title_like_block(block: Block) -> bool:
        label = normalize_paddle_label(block.source_label)
        return block.block_type == BlockType.TITLE or label in {
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
            heading_blocks = [block for block in page.blocks if self._is_title_like_block(block)]
            if not heading_blocks:
                continue
            for block in sorted(heading_blocks, key=lambda item: (item.bbox.y, item.bbox.x, item.order)):
                level = self._heading_level_for_block(block)
                outline_level = level if 1 <= level <= 6 else 1
                level_text = f"H{level}" if level else "标题候选"
                text = self._block_preview_text(block)
                item = QTreeWidgetItem([text])
                item.setData(0, Qt.ItemDataRole.UserRole, (page_idx, id(block)))
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
        page_idx, block_identity = payload
        if not isinstance(page_idx, int) or not (0 <= page_idx < len(self._pages)):
            return
        block = next((candidate for candidate in self._pages[page_idx].blocks if id(candidate) == block_identity), None)
        if block is not None:
            self._focus_block(page_idx, block)

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
            self._show_page_layers(page)
        elif page.error_message:
            self._set_status_text(f"第 {page.page_number} 页分析失败：{page.error_message}")
        self._selected_block = None
        self._sync_selected_type_buttons(None)
        self._prop_conf.hide()
        self._update_project_stats()
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
        self._prop_conf.hide()
        self._update_project_stats()

    def _on_block_clicked(self, block: Block) -> None:
        self._selected_block = block
        bb = block.bbox
        self._sync_selected_type_buttons(block)
        self._prop_conf.set_score(block.avg_confidence)

    @staticmethod
    def _layout_block_state(block: Block) -> dict:
        return LayoutEditService.block_state(block)

    def _record_layout_edit(
        self,
        page: Page,
        op: str,
        block: Block | None,
        *,
        before: dict,
        after: dict,
    ) -> None:
        self._layout_edit_service.record_edit(page, op, block, before=before, after=after)

    def _on_block_edit_started(self, block: Block) -> None:
        self._push_undo_snapshot()
        self._layout_edit_start_state[block.uid] = self._layout_block_state(block)

    def _on_block_moved(self, block: Block) -> None:
        bb = block.bbox
        page = self._pages[self._current_page_idx]
        before = self._layout_edit_start_state.pop(block.uid, self._layout_block_state(block))
        self._persist_user_block_geometry(page, block)
        self._record_layout_edit(
            page,
            "resize_block",
            block,
            before=before,
            after=self._layout_block_state(block),
        )
        self._update_project_stats()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(self._pages[self._current_page_idx].page_number, "block_moved")

    def _persist_user_block_geometry(self, page: Page, block: Block) -> None:
        self._layout_edit_service.persist_user_block_geometry(page, block)

    @staticmethod
    def _inline_formula_origin_bbox(block: Block) -> tuple[int, int, int, int] | None:
        return inline_formula_origin_bbox(block)

    @staticmethod
    def _is_generated_inline_formula_block(block: Block) -> bool:
        return LayoutEditService.is_generated_inline_formula_block(block)

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
        source_label = self._source_label_for_bbox_subtype(page, bbox, subtype)
        intersecting = self._draw_merge_candidates(
            self._blocks_intersecting_bbox(page, bbox),
            bt,
        )
        if intersecting:
            result = self._layout_edit_service.merge_blocks_into_bbox(
                page,
                intersecting,
                bbox,
                block_type=bt,
                source_label=source_label,
            )
            merged = result.block
            if merged is None:
                return
            self._show_page_layers(page)
            self._select_block_for_edit(merged)
            self._set_status_text("已合并框，需重新识别")
            self._rebuild_heading_outline()
            self._refresh_block_search()
            self._update_project_stats()
            self.geometry_changed.emit()
            self.block_contract_changed.emit(page.page_number, "blocks_merged_by_draw")
            return
        result = self._layout_edit_service.create_block(
            page,
            bbox,
            bt,
            source_label,
        )
        new_block = result.block
        if new_block is None:
            return
        self._set_status_text_for_layout_edit_result(result)
        self._show_page_layers(page)
        self._select_block_for_edit(new_block)
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_created")

    def _on_block_deleted(self, block: Block) -> None:
        """viewer 键盘 Delete 已删除框 → 从 page 数据中移除。"""
        if not self._pages:
            return
        self._push_undo_snapshot()
        page = self._pages[self._current_page_idx]
        self._layout_edit_service.delete_block(page, block)
        if self._selected_block is block:
            self._selected_block = None
            self._selected_char_box_index = -1
            self._prop_conf.hide()
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(page.page_number, "block_deleted")

    def _delete_selected(self) -> None:
        """顶部栏删除框按钮。"""
        if not self._viewer.selected_blocks():
            self._set_status_text("请先用右键拉框选中要删除的框")
            return
        self._viewer.delete_selected()

    def _on_selected_type_button_clicked(self, subtype: LayoutSubtypeSpec | BlockType | str) -> None:
        if self._selected_block is None:
            return
        new_subtype = self._coerce_subtype_spec(subtype, DEFAULT_SUBTYPE_BY_SOURCE_LABEL["text"])
        new_label = new_subtype.normalized_source_label
        current_label = self._button_source_label_for_block(self._selected_block)
        if self._selected_block.block_type == new_subtype.block_type and current_label == new_label:
            self._sync_selected_type_buttons(self._selected_block)
            return
        self._push_undo_snapshot()
        page = self._pages[self._current_page_idx]
        source_label = self._source_label_for_subtype(page, self._selected_block, new_subtype)
        result = self._layout_edit_service.change_block_kind(
            page,
            self._selected_block,
            block_type=new_subtype.block_type,
            source_label=source_label,
        )
        self._set_status_text_for_layout_edit_result(result)
        self._show_page_layers(page)
        self._viewer.select_block(self._selected_block)
        self._sync_selected_type_buttons(self._selected_block)
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self._update_project_stats()
        self.geometry_changed.emit()
        self.block_contract_changed.emit(self._pages[self._current_page_idx].page_number, "block_type_changed")

    def _on_char_bbox_moved(self, char) -> None:
        bb = char.bbox
        self.geometry_changed.emit()

    @staticmethod
    def _collect_page_chars(page: Page):
        if page.needs_ocr_rerun:
            return []
        chars = []
        for block in page.blocks:
            if is_ocr_text_invalidated(block):
                continue
            for line in block_ocr_lines(block):
                chars.extend([
                    char
                    for char in line.chars
                    if (
                        char.bbox is not None
                        and char.bbox_source != "paddle_inline_formula"
                        and char_display_text(char)
                    )
                ])
        return chars

    def _show_page_layers(self, page: Page) -> None:
        self._ensure_inline_formula_blocks(page)
        self._viewer.show_blocks(page.blocks)
        self._viewer.show_readonly_overlays(self._collect_readonly_layout_overlays(page))
        if self._btn_char_boxes.isChecked():
            self._viewer.show_char_boxes(self._collect_page_chars(page), editable=False)

    def _select_block_for_edit(self, block: Block) -> None:
        self._viewer.select_block(block)
        self._on_block_clicked(block)

    def _set_status_text_for_layout_edit_result(self, result: LayoutEditResult) -> None:
        if result.binding_empty_review:
            self._set_status_text("已创建校验框")
        elif result.binding_ambiguous:
            self._set_status_text("已创建校验框，需确认")
        elif result.binding_text:
            self._set_status_text("已绑定识别结果")

    def _push_undo_snapshot(self) -> None:
        if not self._pages:
            return
        self._push_undo_snapshot_for_page(self._current_page_idx)

    def _push_undo_snapshot_for_page(self, page_idx: int) -> None:
        self._push_undo_snapshot_for_pages([page_idx])

    def _push_undo_snapshot_for_pages(self, page_indices) -> None:
        if not self._pages:
            return
        snapshots: list[tuple[int, list[Block]]] = []
        seen: set[int] = set()
        for page_idx in page_indices:
            if not isinstance(page_idx, int) or page_idx in seen:
                continue
            if not (0 <= page_idx < len(self._pages)):
                continue
            seen.add(page_idx)
            page = self._pages[page_idx]
            snapshots.append((page_idx, copy.deepcopy(page.blocks)))
        if not snapshots:
            return
        self._undo_stack.append(snapshots)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._btn_undo.setEnabled(True)

    def _undo_last_edit(self) -> None:
        if not self._undo_stack or not self._pages:
            return
        snapshots = self._undo_stack.pop()
        previous_idx = self._current_page_idx
        restored_page_indices: list[int] = []
        for page_idx, blocks in snapshots:
            if not (0 <= page_idx < len(self._pages)):
                continue
            self._pages[page_idx].blocks = copy.deepcopy(blocks)
            restored_page_indices.append(page_idx)
        if not restored_page_indices:
            self._btn_undo.setEnabled(bool(self._undo_stack))
            return
        self._btn_undo.setEnabled(bool(self._undo_stack))
        focus_idx = previous_idx if previous_idx in restored_page_indices else restored_page_indices[-1]
        page = self._pages[focus_idx]
        if focus_idx == previous_idx:
            self._show_page_layers(page)
            self._clear_selection_ui(page)
        else:
            self._current_page_idx = focus_idx
            self._page_list.blockSignals(True)
            try:
                self._page_list.set_current_index(focus_idx)
            finally:
                self._page_list.blockSignals(False)
            self._update_viewer(focus_idx)
        self._update_page_nav()
        self._set_status_text("已撤销上一步版面编辑")
        self._rebuild_heading_outline()
        self._refresh_block_search()
        self.geometry_changed.emit()
        for page_idx in restored_page_indices:
            self.block_contract_changed.emit(self._pages[page_idx].page_number, "layout_undo")

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
            if self._bbox_hits_frame(bbox, block.bbox)
        ]

    @staticmethod
    def _draw_merge_candidates(blocks: list[Block], block_type: BlockType) -> list[Block]:
        target_structural = block_type in STRUCTURAL_DRAW_BLOCK_TYPES
        candidates: list[Block] = []
        for block in blocks:
            existing_structural = block.block_type in STRUCTURAL_DRAW_BLOCK_TYPES
            if target_structural != existing_structural:
                continue
            if target_structural and block.block_type != block_type:
                continue
            candidates.append(block)
        return candidates

    def _ensure_inline_formula_blocks(self, page: Page) -> None:
        """Promote Paddle inline_formula subblocks to editable equation blocks."""
        handled_origins = handled_inline_formula_origin_bboxes(page)
        for parent_index, parent, subblock, bbox in self._iter_inline_formula_subblocks(page):
            origin_tuple = bbox.to_xyxy()
            if origin_tuple in handled_origins:
                continue
            origin = list(origin_tuple)
            if self._has_inline_formula_origin_block(page, origin):
                continue
            page.blocks.append(Block(
                block_type=BlockType.EQUATION,
                bbox=bbox,
                order=len(page.blocks),
                source=BlockSource.AUTO_LAYOUT,
                source_label="inline_formula",
                origin=BlockOrigin(
                    created_by=BlockSource.AUTO_LAYOUT.value,
                    source_engine="paddleocr-vl",
                    source_label="inline_formula",
                    original_bbox=bbox,
                    original_kind=BlockType.EQUATION,
                    raw_artifact_uid=page.raw_layout_artifact.uid if page.raw_layout_artifact else "",
                    raw_index=parent_index,
                ),
            ))

    def _iter_inline_formula_subblocks(self, page: Page):
        for parent_index, parent in enumerate(raw_layout_records(page)):
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
                yield parent_index, parent, subblock, bbox.clamp(page.width, page.height)

    @staticmethod
    def _inline_formula_subblock_text(page: Page, parent: dict, bbox: BBox) -> str:
        target = bbox.clamp(page.width, page.height).to_xyxy()
        marker_text = LayoutPanel._inline_formula_marker_text_from_parent_order(page, parent, target)
        if marker_text:
            return marker_text
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
    def _inline_formula_marker_text_from_parent_order(
        page: Page,
        parent: dict,
        target: tuple[int, int, int, int],
    ) -> str:
        spans = [match.group(0) for match in INLINE_FORMULA_SPAN_RE.finditer(block_text(parent))]
        if not spans:
            return ""
        subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
        if not isinstance(subblocks, list):
            return ""
        formula_bboxes: list[tuple[int, int, int, int]] = []
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
            formula_bboxes.append(bbox.clamp(page.width, page.height).to_xyxy())
        formula_bboxes.sort(key=lambda item: (item[1], item[0]))
        for index, formula_bbox in enumerate(formula_bboxes):
            if index >= len(spans):
                break
            if not LayoutPanel._same_inline_formula_route_span(formula_bbox, target):
                continue
            span = spans[index]
            if is_formula_marker_token(span):
                return span
            return ""
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
            if LayoutPanel._inline_formula_origin_bbox(block) == origin_tuple:
                return True
            if normalize_paddle_label(getattr(block, "source_label", "")) != "inline_formula":
                continue
            if block.bbox.to_xyxy() == origin_tuple:
                return True
        return False

    def _source_label_for_subtype(self, page: Page, block: Block, subtype: LayoutSubtypeSpec) -> str:
        if normalize_paddle_label(subtype.source_label) == "formula":
            return self._infer_formula_source_label_for_bbox(page, block.bbox, exclude=block)
        return subtype.source_label

    def _source_label_for_bbox_subtype(self, page: Page, bbox: BBox, subtype: LayoutSubtypeSpec) -> str:
        if normalize_paddle_label(subtype.source_label) == "formula":
            return self._infer_formula_source_label_for_bbox(page, bbox)
        return subtype.source_label

    def _infer_formula_source_label_for_bbox(self, page: Page, bbox: BBox, *, exclude: Block | None = None) -> str:
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        for block in page.blocks:
            if block is exclude or block.block_type != BlockType.TEXT:
                continue
            if not (block.bbox.x1 <= cx <= block.bbox.x2 and block.bbox.y1 <= cy <= block.bbox.y2):
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
    def _button_source_label_for_block(block: Block) -> str:
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

    def _collect_readonly_layout_overlays(self, page: Page) -> List[tuple[str, BBox]]:
        overlays: List[tuple[str, BBox]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for parent in raw_layout_records(page):
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
            self.page_selected.emit(self._pages[new_idx].page_number)

    def _on_done_clicked(self) -> None:
        """完成本页编辑（占位：发出信号供控制器处理；当前仅状态提示）。"""
        if not self._pages:
            return
        self._set_status_text(f"第 {self._pages[self._current_page_idx].page_number} 页编辑已记录")
        self.page_completed.emit(self._current_page_idx)

    def _on_cancel_clicked(self) -> None:
        """取消本页未提交的编辑（占位：发出信号供控制器处理）。"""
        if self._analysis_running:
            self._set_status_text("正在停止版面分析…")
            self.analysis_cancel_requested.emit()
            return
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
