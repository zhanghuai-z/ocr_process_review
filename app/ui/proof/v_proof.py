"""纵校面板（PRD 3.3）：参照 ui.jpg 布局重构。

布局：
  ┌─────────────┬──────────────────────────────────────────────┐
  │ 单字列表    │  相同字索引 gallery（水平条，可左右滚动）    │
  │（频次排序） │──────────────────────────────────────────────│
  │             │  OCR 文本（可编辑）  │ 原图 + 高亮框         │
  └─────────────┴──────────────────────┴─────────────────────── ┘

联动：
  - 点单字列表 → gallery 刷新 + 文本高亮 + 原图定位
  - 点 gallery 缩略图 → 原图跳到对应页并高亮
  - 原图 block 点击 → 文本滚动到对应行

裁图坐标验证：
  - get_char_crop 传入的 bbox 必须在 page 原图像素空间内
  - 若坐标超出图像尺寸则记录 WARNING 日志
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    QAbstractListModel, QModelIndex, QSize, Qt, Signal,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtGui import (
    QColor, QImage, QIcon, QPainter, QPen, QPixmap,
    QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSplitter, QStyle, QTextEdit,
    QStyledItemDelegate, QVBoxLayout, QWidget,
)

from app.models import BBox, Block, Line, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.core.proof_line_utils import iter_unique_page_text_lines
from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp
from app.services.char_index_service import CharEntry, CharIndexService
from app.services.proof_probe_text_service import (
    displayed_text as _proof_displayed_text,
    save_displayed_edit as _proof_save_displayed_edit,
    resolve_block_line_index as _proof_resolve_block_line_index,
)
from app.services.proof_image_service import (
    verified_char_crop as _shared_verified_char_crop,
)
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

CHAR_LIST_THUMB = 18
GALLERY_THUMB   = 27
GALLERY_ITEMS_PER_ROW = 8
GALLERY_MAX_ROWS = 4
LOW_CONF        = 0.80
DEFAULT_CONFUSABLE_CANDIDATES = {
    # 田/由/甲/申 系
    "田": ["由", "甲", "申", "曲"],
    "由": ["田", "甲", "申"],
    "甲": ["田", "由", "申"],
    "申": ["田", "由", "甲"],
    # 日/曰/目/口 系
    "日": ["曰", "目", "口"],
    "曰": ["日", "目", "口"],
    "目": ["日", "曰", "自", "首"],
    "口": ["日", "曰", "囗", "回"],
    "囗": ["口", "回", "国"],
    # 己/已/巳 系
    "己": ["已", "巳"],
    "已": ["己", "巳"],
    "巳": ["己", "已"],
    # 末/未 系
    "未": ["末", "朱"],
    "末": ["未", "朱"],
    "朱": ["未", "末"],
    # 人/入/八 系
    "人": ["入", "八", "个"],
    "入": ["人", "八"],
    "八": ["人", "入"],
    # 大/太/犬 系
    "大": ["太", "犬", "夫"],
    "太": ["大", "犬", "夫"],
    "犬": ["大", "太"],
    "夫": ["大", "天", "夭"],
    "天": ["夫", "夭", "无"],
    "夭": ["天", "夫"],
    # 干/千/午/壬 系
    "干": ["千", "午", "于"],
    "千": ["干", "午"],
    "午": ["干", "千", "牛"],
    "牛": ["午", "牟"],
    # 土/士/王/玉/主/王 系
    "土": ["士", "工"],
    "士": ["土", "仕"],
    "王": ["玉", "主", "壬"],
    "玉": ["王", "主"],
    "主": ["王", "玉", "住"],
    "壬": ["王", "工", "土"],
    # 工/匚 系
    "工": ["土", "士", "壬"],
    # 又/叉/义 系
    "又": ["叉", "义"],
    "叉": ["又"],
    "义": ["又", "乂"],
    # 力/刀/办 系
    "力": ["刀", "办"],
    "刀": ["力", "刃"],
    "刃": ["刀"],
    # 木/术/本/朩
    "木": ["术", "本", "朩"],
    "术": ["木", "朮"],
    "本": ["木", "未", "末"],
    # 水/氺/永/冰
    "水": ["氺", "永", "冰"],
    "永": ["水", "求"],
    "冰": ["水", "永"],
    # 火/灬
    "火": ["灬", "炎"],
    # 心/必
    "心": ["必", "忄"],
    "必": ["心"],
    # 巾/币/市
    "巾": ["币", "市", "布"],
    "币": ["巾", "市"],
    "市": ["巾", "币", "布"],
    "布": ["巾", "市"],
    # 戊/戌/戍/戎/成
    "戊": ["戌", "戍", "戎", "成"],
    "戌": ["戊", "戍", "戎"],
    "戍": ["戊", "戌", "戎"],
    "戎": ["戊", "戌", "戍"],
    "成": ["戊", "戌"],
    # 凡/几/丸
    "凡": ["几", "丸"],
    "几": ["凡", "九"],
    "丸": ["凡", "九"],
    "九": ["几", "丸"],
    # 北/比/此
    "北": ["比", "兆"],
    "比": ["北", "此"],
    "此": ["比"],
    # 卜/上/下/不
    "卜": ["上", "下", "不"],
    "上": ["卜", "下"],
    "下": ["卜", "上"],
    "不": ["卜", "丕"],
    # 千/午/牛/年
    "年": ["午", "牛"],
    # 自/白/百
    "自": ["白", "百", "目"],
    "白": ["自", "百"],
    "百": ["白", "自"],
    # 干/于/亏
    "于": ["干", "亏", "乎"],
    "亏": ["于", "夸"],
    # 风/凤/凡
    "风": ["凤", "凡"],
    "凤": ["风", "凡"],
    # 鸟/乌/马
    "鸟": ["乌"],
    "乌": ["鸟"],
    # 兔/免
    "兔": ["免", "兑"],
    "免": ["兔"],
    # 衣/农/表
    "衣": ["农", "表"],
    "农": ["衣"],
    "表": ["衣"],
    # 万/方
    "万": ["方"],
    "方": ["万"],
    # 历/厉/历
    "历": ["厉"],
    "厉": ["历"],
    # 凉/京/亨/享
    "京": ["亨", "享", "亰"],
    "亨": ["京", "享"],
    "享": ["京", "亨"],
    # 末未本 family already above
    # 人/入/八/丶/丿 系（去重，已含）
    # 长/张/麦/麸
    "长": ["镸"],
    # 圣/圣
    # 兵/丘/丛
    "兵": ["丘"],
    "丘": ["兵", "兵"],
    # 体/休
    "休": ["体"],
    "体": ["休"],
    # 化/华
    "化": ["华"],
    "华": ["化"],
    # 兮/分/今
    "今": ["令", "兮"],
    "令": ["今"],
    "兮": ["今"],
    # 半/羊
    "半": ["羊", "丰"],
    "丰": ["半", "羊"],
    "羊": ["半", "丰"],
    # 兄/见/光
    "兄": ["见", "克"],
    "见": ["兄", "贝"],
    "贝": ["见", "页"],
    "页": ["贝", "顶"],
    # 圆/园/团
    "圆": ["园", "团"],
    "园": ["圆", "园"],
    "团": ["圆", "园"],
    # 句/旬/勺
    "句": ["旬", "勺"],
    "旬": ["句"],
    "勺": ["句"],
}


@dataclass(frozen=True)
class LlmCandidateRequest:
    token: str
    page_number: int
    line_text: str
    char_index: int
    context_text: str
    bbox_source: str = ""
    bbox_granularity: str = ""
    collection_kind: str = ""
    confidence: float = 0.0


class LlmCandidateProvider(Protocol):
    def suggest_candidates(self, request: LlmCandidateRequest) -> List[str]:
        ...


# ─────────────────────────────────────────────────────────────
# 裁图坐标验证辅助
# ─────────────────────────────────────────────────────────────

def _verified_char_crop(
    cache: PageImageCache,
    page_path: str,
    bbox: BBox,
    size: int = GALLERY_THUMB,
    pad: Optional[int] = None,
) -> Optional[QPixmap]:
    """裁图坐标校验薄封装 —— 实际逻辑在 ``app.services.proof_image_service``。

    保留本地名以维持调用点不变，并保留 ``size`` 默认 ``GALLERY_THUMB`` 这个
    UI 侧默认值（service 函数不带默认 size，由这里集中决定）。
    """
    return _shared_verified_char_crop(cache, page_path, bbox, size, pad=pad)


# ─────────────────────────────────────────────────────────────
# 虚拟 Gallery Model / Delegate（水平条）
# ─────────────────────────────────────────────────────────────

class _GalleryModel(QAbstractListModel):
    def __init__(self, cache: PageImageCache, parent=None) -> None:
        super().__init__(parent)
        self._entries: List[CharEntry] = []
        self._cache = cache

    def set_entries(self, entries: List[CharEntry]) -> None:
        self.beginResetModel()
        self._entries = entries
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._entries):
            return None
        entry = self._entries[index.row()]
        if role == Qt.ItemDataRole.DecorationRole:
            return _verified_char_crop(self._cache, entry.page_path, entry.bbox, GALLERY_THUMB)
        if role == Qt.ItemDataRole.DisplayRole:
            display_key = entry.token_text or entry.char
            return f"{index.row() + 1:03d}\nP{entry.page_number}-{display_key}"
        if role == Qt.ItemDataRole.ToolTipRole:
            display_key = entry.token_text or entry.char
            kind = entry.collection_kind
            kind_label = "token" if kind == "token" else "char"
            return (
                f"第 {entry.page_number} 页，位#{entry.char_idx + 1}  "
                f"[{display_key}] ({kind_label}, {entry.bbox_source}/{entry.bbox_granularity})"
            )
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None


class _GalleryDelegate(QStyledItemDelegate):
    SIZE = GALLERY_THUMB + 8

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        r = option.rect
        pix: Optional[QPixmap] = index.data(Qt.ItemDataRole.DecorationRole)
        img_r = r.adjusted(2, 2, -2, -2)
        if pix and not pix.isNull():
            scaled = pix.scaled(
                img_r.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            dx = (img_r.width() - scaled.width()) // 2
            painter.drawPixmap(img_r.x() + dx, img_r.y(), scaled)
        else:
            painter.fillRect(img_r, QColor("#ffffff"))
            # 裁图失败时显示 token 内容作为占位，避免白块无信息
            entry = index.data(Qt.ItemDataRole.UserRole)
            fallback = (entry.token_text or entry.char) if entry else "?"
            painter.setPen(QColor("#5580a0"))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(
                img_r,
                Qt.AlignmentFlag.AlignCenter,
                fallback[:4],  # 最多显4字
            )

        # 选中边框
        if option.state & QStyle.StateFlag.State_Selected:
            pen = QPen(QColor("#1a73e8"), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r.adjusted(1, 1, -1, -1))

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        return QSize(self.SIZE, self.SIZE)


# ─────────────────────────────────────────────────────────────
# 文本映射辅助
# ─────────────────────────────────────────────────────────────

def _build_text_map(
    page: Page,
) -> Tuple[str, List[Tuple[Line, int, int, int]]]:
    parts: List[str] = []
    mapping: List[Tuple[Line, int, int, int]] = []
    pos = 0
    last_block: Optional[Block] = None
    for block, line, _line_idx in iter_unique_page_text_lines(page):
        if last_block is not None and block is not last_block:
            parts.append("\n")
            pos += 1
        last_block = block
        # 显示空间文本：若 quality_probe 启用，行内若干位置会被替换成 fake_char。
        # 因为 apply_probes_to_display 是位置等长替换，char_idx 在真实空间和显示
        # 空间是一一对应的，所以单字 mapping 仍然有效。
        display_text = _vproof_displayed_text(page, block, line)
        for ci, _char in enumerate(display_text):
            mapping.append((line, ci, pos, pos + 1))
            pos += 1
        parts.append(display_text)
        parts.append("\n")
        pos += 1
    return "".join(parts), mapping


# ─── quality_probe display ↔ true text 桥接 ──────────────────────
# 共享实现见 app.services.proof_probe_text_service。
# 这里保留 _vproof_* 名称以维持调用点不变，但内部直接转发；签名调换 page/line
# 顺序仅为兼容 v_proof 的旧调用习惯。

def _vproof_resolve_block_line_index(page: Page, block: Block, line: Line):
    return _proof_resolve_block_line_index(page, block, line)


def _vproof_displayed_text(page: Page, block: Block, line: Line) -> str:
    return _proof_displayed_text(line, page, block)


def _vproof_save_displayed_line(
    page: Page, block: Block, line: Line, displayed_new_text: str
) -> bool:
    return _proof_save_displayed_edit(line, page, block, displayed_new_text)


# ─────────────────────────────────────────────────────────────
# 纵校面板
# ─────────────────────────────────────────────────────────────

class VProofPanel(QWidget):
    """纵校面板：参照 ui.jpg 三区域布局（左单字列表 + 顶gallery + 底OCR/图）。"""

    proof_saved = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        self._char_svc = CharIndexService()
        self._text_map: List[Tuple[Line, int, int, int]] = []
        self._entry_pos_by_key: dict[tuple[int, int], int] = {}
        self._gallery_model = _GalleryModel(self._cache)
        self._selected_char: str = ""
        self._current_candidate_entry: Optional[CharEntry] = None
        self._candidate_buttons: List[QPushButton] = []
        self._candidate_provider: Optional[LlmCandidateProvider] = None
        self._updating = False
        # Phase 22 blocker 1：_load_page 后存基线文本，用于 _on_external_line_changed
        # 判断"_text_edit 是否有未保存输入"，避免外部同步覆盖用户在编辑的内容。
        self._loaded_text: str = ""
        self._build_ui()
        # H/V 校对联动：订阅其他 panel 的编辑事件，本 panel 自己 publish 的事件
        # 通过 origin == id(self) 过滤掉以避免回路。
        # Phase 18 blocker 3：保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            "line.proof_changed", self._on_external_line_changed,
        )
        self.destroyed.connect(lambda *_: self._teardown_bus())

    def _teardown_bus(self) -> None:
        """Phase 18 blocker 3：释放 ProofStateBus 订阅，幂等。"""
        unsub = getattr(self, "_bus_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._bus_unsub = None

    # ─────────────────── UI ───────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶部工具栏 ──────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setFixedHeight(46)
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 0, 12, 0)
        tl.setSpacing(6)

        self._btn_prev_page = QPushButton("← 上一页")
        self._btn_next_page = QPushButton("下一页 →")
        self._btn_save = QPushButton("✎ 保存  Ctrl+S")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_ok = QPushButton("✓ 确认本页")

        for btn in (self._btn_prev_page, self._btn_next_page,
                    self._btn_save, self._btn_ok):
            btn.setMinimumHeight(30)
            tl.addWidget(btn)

        tl.addStretch()
        self._page_label = QLabel("页 0 / 0")
        self._page_label.setObjectName("muted")
        tl.addWidget(self._page_label)
        self._conf_badge = ConfidenceBadge(1.0)
        tl.addWidget(self._conf_badge)

        root.addWidget(toolbar)

        # ── 主体：水平分割（左单字列表 | 右主区域）─────────────
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1)

        # 左：单字列表 + 搜索
        self._main_split = h_split
        self._left_box = self._build_char_list()
        self._left_box.setMinimumWidth(110)
        self._left_box.setMaximumWidth(170)
        h_split.addWidget(self._left_box)

        # 右：垂直分割（上gallery | 下文本/图）
        right_box = self._build_right_area()
        h_split.addWidget(right_box)
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 7)

        root.addWidget(h_split, 1)

        # ── 信号 ────────────────────────────────────────────
        self._btn_prev_page.clicked.connect(self._prev_page)
        self._btn_next_page.clicked.connect(self._next_page)
        self._btn_save.clicked.connect(self._save_page_text)
        self._btn_ok.clicked.connect(self._mark_page_ok)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_page_text)
        QShortcut(QKeySequence("PageUp"), self, activated=self._prev_page)
        QShortcut(QKeySequence("PageDown"), self, activated=self._next_page)
        # proof-interaction-slots 第 6 任务：点击切图 → 焦点走 _text_edit。
        # 为了不把用户“困”在文本光标里，增加 Alt+← / Alt+→ 在 gallery 里切
        # 到上/下一个同字出现。这些快捷键不抢占普通方向键。
        QShortcut(QKeySequence("Alt+Right"), self, activated=self._go_next_gallery)
        QShortcut(QKeySequence("Alt+Left"), self, activated=self._go_prev_gallery)

    def _build_char_list(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel("字符索引")
        lbl.setObjectName("sectionTitle")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._char_count_lbl = QLabel("共 0 项")
        self._char_count_lbl.setObjectName("muted")
        hdr.addWidget(self._char_count_lbl)
        layout.addLayout(hdr)

        self._char_search = QLineEdit()
        self._char_search.setPlaceholderText("搜索字符…")
        self._char_search.textChanged.connect(self._filter_char_list)
        layout.addWidget(self._char_search)

        self._char_list = QListWidget()
        # proof-interaction-slots 第 4 任务：字符索引不再显示单字图，icon 区清零。
        self._char_list.setIconSize(QSize(0, 0))
        self._char_list.setSpacing(2)
        self._char_list.setStyleSheet("QListWidget { background:#ffffff; } QListWidget::item { background:#ffffff; }")
        self._char_list.itemClicked.connect(self._on_char_clicked)
        self._char_list.currentItemChanged.connect(
            lambda cur, _prev: self._on_char_clicked(cur) if cur else None
        )
        layout.addWidget(self._char_list)
        return box

    def _build_right_area(self) -> QWidget:
        box = QWidget()
        root = QHBoxLayout(box)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._content_split = QSplitter(Qt.Orientation.Horizontal)
        self._content_split.setHandleWidth(1)

        self._proof_column = QWidget()
        col = QVBoxLayout(self._proof_column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # 左列上：gallery 网格。与同列 OCR 文本自然等宽。
        self._gallery_box = self._build_gallery_strip()
        self._resize_gallery_for_entries(0)
        col.addWidget(self._gallery_box)

        self._candidate_box = self._build_candidate_panel()
        col.addWidget(self._candidate_box)

        self._ocr_text_box = self._build_ocr_text()
        col.addWidget(self._ocr_text_box, 1)

        self._viewer_box = self._build_viewer()
        self._content_split.addWidget(self._proof_column)
        self._content_split.addWidget(self._viewer_box)
        self._content_split.setStretchFactor(0, 1)
        self._content_split.setStretchFactor(1, 4)
        root.addWidget(self._content_split)
        return box

    def _build_candidate_panel(self) -> QWidget:
        # proof-layout-collections 第 4 任务：候选面板不再打印“不代表此字
        # 正确 + 原因”这样的可见解释文本。诊断信息改走面板 tooltip，作为
        # "鼠标悬停才看到"的不打扰提示。
        #
        # proof-layout-collections 第 5 任务：候选按钮不能压在下方 OCR 文本区
        # 上。高度从 54 提到 64，并增加上下 padding，让 24px 按钮 + 标题 不
        # 被压出边界。
        box = QWidget()
        box.setObjectName("candidatePanel")
        box.setFixedHeight(64)
        box.setStyleSheet("QWidget#candidatePanel { background:#ffffff; border:0; }")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self._candidate_title = QLabel("候选字")
        self._candidate_title.setObjectName("sectionTitle")
        self._candidate_buttons_row = QHBoxLayout()
        self._candidate_buttons_row.setSpacing(6)
        # 保留 _candidate_hint 作为不可见的 QLabel。代码里多处 _candidate_hint.setText /
        # show / hide 有历史调用点。这里让它始终 hidden，文本只用于 tooltip
        # 传递，不进入可见布局。
        self._candidate_hint = QLabel("")
        self._candidate_hint.setObjectName("muted")
        self._candidate_hint.hide()
        layout.addWidget(self._candidate_title)
        layout.addLayout(self._candidate_buttons_row)
        return box

    def _build_gallery_strip(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        self._gallery_hdr = QLabel("相同字索引（请先在左侧选择一个字)")
        self._gallery_hdr.setObjectName("sectionTitle")
        hdr.addWidget(self._gallery_hdr)
        hdr.addStretch()
        layout.addLayout(hdr)

        # proof-slot-residual 第 2 任务：批量改字输入条。
        # 上一轮 (interaction-slots) 只让"候选按钮"在多选时批量；本轮再加
        #   - 行内输入框：键入任意字符 + Enter / 按"应用"立即批量替换。
        #     这是用户最直接的"我自己来打字"路径，不依赖 LLM 候选。
        #   - Ctrl+A：选中本字所有出现（widget 内置 ExtendedSelection 已支持，
        #     这里再装一次显式快捷键，覆盖鼠标不在 gallery 时的场景）。
        #   - Esc：清空 gallery 多选，回到单选当前项。
        batch_row = QHBoxLayout()
        batch_row.setSpacing(4)
        self._batch_input = QLineEdit()
        self._batch_input.setPlaceholderText("批量改为…（多选时生效）")
        self._batch_input.setMaxLength(8)  # 允许 token-粒度多字符
        self._batch_input.setClearButtonEnabled(True)
        self._batch_input.returnPressed.connect(self._apply_batch_input)
        self._batch_btn = QPushButton("批量应用")
        self._batch_btn.clicked.connect(self._apply_batch_input)
        self._batch_select_all_btn = QPushButton("全选 (Ctrl+A)")
        self._batch_select_all_btn.clicked.connect(self._select_all_gallery)
        self._batch_clear_btn = QPushButton("清选 (Esc)")
        self._batch_clear_btn.clicked.connect(self._clear_gallery_selection)
        batch_row.addWidget(self._batch_input, 1)
        batch_row.addWidget(self._batch_btn)
        batch_row.addWidget(self._batch_select_all_btn)
        batch_row.addWidget(self._batch_clear_btn)
        layout.addLayout(batch_row)

        self._gallery_view = QListView()
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.setItemDelegate(_GalleryDelegate(self._gallery_view))
        self._gallery_view.setViewMode(QListView.ViewMode.IconMode)
        self._gallery_view.setFlow(QListView.Flow.LeftToRight)  # 水平排列
        self._gallery_view.setWrapping(True)
        self._gallery_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._gallery_view.setMovement(QListView.Movement.Static)
        self._gallery_view.setUniformItemSizes(True)
        self._gallery_view.setSpacing(4)
        self._gallery_view.setStyleSheet("QListView { background:#ffffff; } QListView::item { background:#ffffff; }")
        self._gallery_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gallery_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._gallery_view.setSelectionMode(
            # proof-interaction-slots 第 7 任务：改为扩展选择（Ctrl 点选 + Shift 跨选）
            # 用于批量修改。原单选模式在需要一次改多个相同字的切图时不可用。
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # 让方向键 / Shift+方向键 / Ctrl+点击 走 Qt 原生 ExtendedSelection 行为。
        self._gallery_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._gallery_view.selectionModel().currentChanged.connect(
            self._on_gallery_current_changed
        )
        # proof-slot-residual 第 2 任务：选区数量变化 → 刷新标题 + 按钮状态
        self._gallery_view.selectionModel().selectionChanged.connect(
            self._on_gallery_selection_changed
        )
        self._gallery_view.clicked.connect(self._on_gallery_clicked)
        # widget-scoped Ctrl+A / Esc：仅当焦点在 gallery 内时触发
        sc_all = QShortcut(QKeySequence("Ctrl+A"), self._gallery_view)
        sc_all.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_all.activated.connect(self._select_all_gallery)
        sc_esc = QShortcut(QKeySequence("Escape"), self._gallery_view)
        sc_esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_esc.activated.connect(self._clear_gallery_selection)
        layout.addWidget(self._gallery_view)
        # 初始按钮态
        self._on_gallery_selection_changed()
        return box

    def _resize_gallery_for_entries(self, count: int) -> None:
        # proof-interaction-slots 第 5 任务：相同字索引窗口更疏开。
        # item 行高从 +8 提到 +14；外框冗余从 +30 提到 +44，让按钮/标题/
        # gallery 三者之间有清晰间距，不再压字。
        rows = min(
            GALLERY_MAX_ROWS,
            max(1, (max(1, count) + GALLERY_ITEMS_PER_ROW - 1) // GALLERY_ITEMS_PER_ROW),
        )
        item_h = GALLERY_THUMB + 14
        self._gallery_box.setFixedHeight(rows * item_h + 44)

    def _go_next_gallery(self) -> None:
        """proof-interaction-slots 第 6 任务：Alt+→ 在 gallery 里跳下一个出现。"""
        self._step_gallery(+1)

    def _go_prev_gallery(self) -> None:
        """proof-interaction-slots 第 6 任务：Alt+← 在 gallery 里跳上一个出现。"""
        self._step_gallery(-1)

    def _step_gallery(self, delta: int) -> None:
        count = self._gallery_model.rowCount()
        if count <= 0:
            return
        sel = self._gallery_view.selectionModel()
        cur = sel.currentIndex()
        cur_row = cur.row() if cur.isValid() else -1
        new_row = (cur_row + delta) % count
        new_idx = self._gallery_model.index(new_row, 0)
        sel.setCurrentIndex(
            new_idx, sel.SelectionFlag.ClearAndSelect
        )
        # 主动触发同步（currentChanged 已连，但保险）
        self._sync_gallery_entry(new_idx)

    def _build_ocr_text(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("OCR 文本（直接编辑）"))

        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:16px; padding:8px;")
        self._text_edit.document().contentsChanged.connect(self._on_text_changed)
        layout.addWidget(self._text_edit)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("noteLabel")
        layout.addWidget(self._status_lbl)
        return box

    def _build_viewer(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("原图"))
        self._viewer = ImageViewer()
        layout.addWidget(self._viewer)
        return box

    # ─────────────────── 公共 API ───────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        self._current_page_idx = 0
        self._char_svc.build(pages)
        self._rebuild_char_list()
        if pages:
            self._load_page(0)

    def merge_pages(self, pages: List[Page]) -> None:
        """Merge background OCR pages without overwriting current page text."""
        if not self._pages or not self._text_map:
            self.load_pages(pages)
            return
        current_page = self._pages[self._current_page_idx]
        self._pages = pages
        self._current_page_idx = self._find_page_index(current_page)
        if self._pages:
            self._rebuild_text_lookup(self._pages[self._current_page_idx])
        self._char_svc.build(pages)
        selected = self._selected_char
        self._rebuild_char_list()
        if selected:
            self._selected_char = selected
            entries = self._char_svc.query(selected)
            self._gallery_model.set_entries(entries)
            self._resize_gallery_for_entries(len(entries))
            self._gallery_hdr.setText(f'"{selected}"  共 {len(entries)} 处')
        else:
            self._resize_gallery_for_entries(0)
        self._page_label.setText(f"页 {self._current_page_idx + 1} / {len(self._pages)}")

    def _find_page_index(self, target: Page) -> int:
        for idx, page in enumerate(self._pages):
            if page is target:
                return idx
        for idx, page in enumerate(self._pages):
            if (
                page.display_image_path == target.display_image_path
                and page.page_number == target.page_number
            ):
                return idx
        return min(self._current_page_idx, max(0, len(self._pages) - 1))

    def reset(self) -> None:
        self._pages = []
        self._char_svc = CharIndexService()
        self._char_list.clear()
        self._text_edit.clear()
        self._gallery_model.set_entries([])
        self._resize_gallery_for_entries(0)
        self._current_candidate_entry = None
        self._clear_candidate_buttons()
        self._candidate_box.setToolTip("")
        self._page_label.setText("页 0 / 0")

    def set_candidate_provider(self, provider: Optional[LlmCandidateProvider]) -> None:
        self._candidate_provider = provider

    # ─────────────────── 字符列表 ────────────────────────────

    def _rebuild_char_list(self) -> None:
        # proof-interaction-slots 第 4 任务：字符索引改为纯文本索引。
        # 只展示 "文本 ×数量"；不再设 icon（单字切图）、不再设
        # “[char] / [token]” 后缀、不再设 tooltip。
        self._char_list.blockSignals(True)
        self._char_list.clear()
        # 同时把 icon_size 设为 0，以防某些主题留出空白 icon 区。
        self._char_list.setIconSize(QSize(0, 0))
        freqs = self._char_svc.sorted_chars()
        self._char_count_lbl.setText(f"共 {len(freqs)} 项")
        for tok, count in freqs:
            item = QListWidgetItem(f"{tok}  ×{count}")
            item.setData(Qt.ItemDataRole.UserRole, tok)
            self._char_list.addItem(item)
        self._char_list.blockSignals(False)

    def _filter_char_list(self, text: str) -> None:
        for i in range(self._char_list.count()):
            item = self._char_list.item(i)
            char = item.data(Qt.ItemDataRole.UserRole) or ""
            item.setHidden(text != "" and text not in char)

    # ─────────────────── 页面加载 ────────────────────────────

    def _load_page(self, idx: int) -> None:
        if not self._pages:
            return
        idx = max(0, min(idx, len(self._pages) - 1))
        self._current_page_idx = idx
        page = self._pages[idx]
        self._page_label.setText(f"页 {idx + 1} / {len(self._pages)}")

        lines = [line for _block, line, _line_idx in iter_unique_page_text_lines(page)]
        if lines:
            avg_conf = sum(ln.confidence for ln in lines) / len(lines)
            self._conf_badge.set_score(avg_conf)

        self._viewer.set_image(page.display_image_path)
        # 纵校视图仅供参考，不显示版面标注框（show_blocks 不调用）
        # 字符高亮由 highlight_bbox 单独绘制，避免与块框混淆

        flat_text, self._text_map = _build_text_map(page)
        self._rebuild_text_lookup(page)
        self._updating = True
        self._text_edit.setPlainText(flat_text)
        self._text_edit.setExtraSelections([])
        self._updating = False
        # Phase 22 blocker 1：保存"刚加载完成时的纯净文本"作为 dirty 判断基线
        self._loaded_text = flat_text
        self._status_lbl.setText("")
        self._status_lbl.setStyleSheet("")

    # ─────────────────── 单字列表点击 ───────────────────────

    def _on_char_clicked(self, item: QListWidgetItem) -> None:
        tok = item.data(Qt.ItemDataRole.UserRole)
        if not tok:
            return
        self._selected_char = tok
        entries = self._char_svc.query(tok)

        # 重置 gallery：清选中、滚回顶部
        self._gallery_model.set_entries(entries)
        self._resize_gallery_for_entries(len(entries))
        self._gallery_view.clearSelection()
        if entries:
            self._gallery_view.scrollTo(
                self._gallery_model.index(0, 0),
                QAbstractItemView.ScrollHint.PositionAtTop,
            )
        self._gallery_hdr.setText(f'"{tok}"  共 {len(entries)} 处')
        target = entries[0] if entries else None
        if target:
            first_idx = self._gallery_model.index(0, 0)
            self._gallery_view.setCurrentIndex(first_idx)
            self._sync_gallery_entry(first_idx)
        else:
            self._current_candidate_entry = None
            self._clear_candidate_buttons()
            self._candidate_box.setToolTip("无候选")
            self._highlight_char_in_text(tok, focus_entry=None)

    def _rebuild_text_lookup(self, page: Page) -> None:
        _flat_text, self._text_map = _build_text_map(page)
        self._entry_pos_by_key = {
            (id(line), ci): start
            for line, ci, start, _end in self._text_map
        }

    def _entry_text_pos(self, entry: CharEntry) -> Optional[int]:
        """查找 CharEntry 在当前 _text_map 中的起始光标位置。"""
        return self._entry_pos_by_key.get((id(entry.line), entry.char_idx))

    def _highlight_char_in_text(
        self, char: str, focus_entry: Optional[CharEntry] = None,
    ) -> None:
        doc = self._text_edit.document()
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#ffe8a3"))
        fmt.setForeground(QColor("#0b57d0"))
        target_pos: Optional[int] = None
        if focus_entry is not None:
            target_pos = self._entry_text_pos(focus_entry)
        if target_pos is None:
            first = doc.find(char)
            if not first.isNull():
                target_pos = first.selectionStart()
        if target_pos is not None:
            place = QTextCursor(doc)
            place.setPosition(target_pos)
            for _ in range(len(char)):
                place.movePosition(
                    QTextCursor.MoveOperation.NextCharacter,
                    QTextCursor.MoveMode.KeepAnchor,
                )
            selection = QTextEdit.ExtraSelection()
            selection.cursor = QTextCursor(place)
            selection.format = fmt
            self._text_edit.setExtraSelections([selection])
            self._text_edit.setTextCursor(place)
            self._text_edit.ensureCursorVisible()
        else:
            self._text_edit.setExtraSelections([])

    def _highlight_char_in_viewer(self, entry: CharEntry) -> None:
        if not self._pages:
            return
        cur_page = self._pages[self._current_page_idx]
        if entry.page_path == cur_page.display_image_path:
            self._viewer.highlight_bbox(entry.bbox, zoom=True)
        else:
            for i, p in enumerate(self._pages):
                if p.display_image_path == entry.page_path:
                    self._load_page(i)
                    self._viewer.highlight_bbox(entry.bbox, zoom=True)
                    break

    def _candidate_request_for_entry(self, entry: CharEntry) -> LlmCandidateRequest:
        line_text = entry.line.text or ""
        return LlmCandidateRequest(
            token=entry.token_text or entry.char,
            page_number=entry.page_number,
            line_text=line_text,
            char_index=entry.char_idx,
            context_text=self._text_edit.toPlainText(),
            bbox_source=entry.bbox_source,
            bbox_granularity=entry.bbox_granularity,
            collection_kind=entry.collection_kind,
            confidence=entry.confidence,
        )

    def _update_candidate_panel(self, entry: CharEntry) -> None:
        # 极简版：UI 上只摆 ≤5 个候选按钮；不显示分数/来源/LLM 字样/解释段。
        # 第一候选由 _ranked_candidates 决定（最高可信来源优先）。
        request = self._candidate_request_for_entry(entry)
        candidates = self._ranked_candidates(entry, request)[:5]
        self._set_candidate_buttons(candidates)
        # proof-layout-collections 第 4 任务：原本在面板底部画一段 “仅 1 候选，
        # 不代表此字正确…原因…” 的解释文本。用户要求拿掉。诊断现在只作为
        # 整个候选面板的 toolTip（鼠标悬停才看到），不侵占任何可见布局。
        if not candidates:
            self._candidate_box.setToolTip("无候选")
            return
        if len(candidates) == 1:
            reason = self._diagnose_single_candidate(entry, request)
            self._candidate_box.setToolTip(
                f"仅 1 候选：暂无替代建议，不代表此字正确。{reason}"
            )
        else:
            self._candidate_box.setToolTip("")

    def _diagnose_single_candidate(
        self, entry: CharEntry, request: "LlmCandidateRequest",
    ) -> str:
        """生成"仅 1 候选"的诚实诊断字符串。

        说明 4 个候选来源各自为什么没贡献新字（去重后只剩当前字）：
          1. line.ocr_text[i] 与当前字相同 / 缺失
          2. line.llm_suggestion[i] 与当前字相同 / 缺失
          3. LlmCandidateProvider 未注入 / 调用失败 / 返回空
          4. DEFAULT_CONFUSABLE_CANDIDATES 字典里 token 无 entry

        本函数 **只读** 不写状态；返回一段短文本拼接到 hint label。
        """
        reasons: List[str] = []
        token = entry.token_text or entry.char
        ocr_ch = self._line_char_at(entry.line.ocr_text, entry.char_idx)
        llm_ch = self._line_char_at(entry.line.llm_suggestion, entry.char_idx)
        if not ocr_ch:
            reasons.append("OCR 无对应字")
        elif ocr_ch == token:
            reasons.append("OCR 与当前字一致")
        if not llm_ch:
            reasons.append("LLM 未给建议")
        elif llm_ch == token:
            reasons.append("LLM 建议与当前字一致")
        if self._candidate_provider is None:
            reasons.append("候选 provider 未接入")
        if token not in DEFAULT_CONFUSABLE_CANDIDATES:
            reasons.append(f"易混淆字典无 '{token}' 条目")
        if not reasons:
            return "（来源均无新字）"
        return "原因：" + " / ".join(reasons) + "。"

    def _ranked_candidates(
        self, entry: CharEntry, request: LlmCandidateRequest,
    ) -> List[str]:
        """按可信度从高到低聚合候选字，去重后返回 list。

        优先级（高 → 低）：
          1. 当前显示字 / token 自身（保留第一位 = 现有识别结果）
          2. line.ocr_text 在同 char_index 上的字（如果与显示字不同，说明本次有 probe 或后续修正）
          3. line.llm_suggestion 在同 char_index 上的字
          4. 注入的 LlmCandidateProvider 返回值（不在 UI 标 LLM）
          5. DEFAULT_CONFUSABLE_CANDIDATES 易混淆字（兜底）

        provider 失败/未注入时静默跳过，不显示错误提示，保持 UI 干净。
        """
        ranked: List[str] = []

        def _push(value: str) -> None:
            if value and value not in ranked:
                ranked.append(value)

        _push(entry.token_text or entry.char)
        _push(self._line_char_at(entry.line.ocr_text, entry.char_idx))
        _push(self._line_char_at(entry.line.llm_suggestion, entry.char_idx))

        if self._candidate_provider is not None:
            try:
                provider_out = self._candidate_provider.suggest_candidates(request)
            except Exception:
                provider_out = []
            for value in provider_out:
                _push(value)

        token = entry.token_text or entry.char
        # 一阶易混淆
        first_level = DEFAULT_CONFUSABLE_CANDIDATES.get(token, [])
        for value in first_level:
            _push(value)
        # Task #3 修复：真实流程中 provider 默认未注入、ocr_text/llm_suggestion
        # 与当前字常常相同，绝大多数字字典又只有一级 → UI 上经常只剩 1 项。
        # 这里做二阶易混淆展开：对一阶里的每个字再 lookup 一次，把它们的
        # 易混淆兄弟字合并进来。仍受 _push 去重保护，最终在 _update_candidate_panel
        # 处截断到 5 项。
        for first in first_level:
            for second in DEFAULT_CONFUSABLE_CANDIDATES.get(first, []):
                _push(second)

        return ranked

    def _explainable_candidates_for_entry(self, entry: CharEntry) -> List[str]:
        # 保留以兼容旧调用点；内部委托给 _ranked_candidates。
        request = self._candidate_request_for_entry(entry)
        return self._ranked_candidates(entry, request)[:5]

    def _clear_candidate_buttons(self) -> None:
        while self._candidate_buttons_row.count():
            item = self._candidate_buttons_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidate_buttons = []

    def _set_candidate_buttons(self, candidates: List[str]) -> None:
        # 极简版：第一候选用 primaryBtn 样式强调，其余 candidateButton。
        # 不再附带 LLM/来源标签到按钮可见文本上。
        self._clear_candidate_buttons()
        for idx, candidate in enumerate(candidates):
            button = QPushButton(candidate)
            button.setObjectName("primaryBtn" if idx == 0 else "candidateButton")
            button.setMinimumHeight(24)
            button.setToolTip(
                f"替换当前选中字为：{candidate}"
                + ("（最高可信候选）" if idx == 0 else "")
            )
            button.clicked.connect(
                lambda _checked=False, value=candidate: self._apply_candidate(value)
            )
            self._candidate_buttons_row.addWidget(button)
            self._candidate_buttons.append(button)
        self._candidate_buttons_row.addStretch()

    def _apply_candidate(self, candidate: str) -> None:
        entry = self._current_candidate_entry
        if entry is None:
            return
        applied = self._apply_replacement_to_selected(candidate, fallback_entry=entry)
        self._text_edit.setFocus()
        if applied > 1:
            self._status_lbl.setText(f"● 已批量应用候选到 {applied} 处，待保存")
        else:
            self._status_lbl.setText("● 已应用候选，待保存")
        self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")

    # ─────────────────── proof-slot-residual 批量改字共享路径 ─────────
    def _apply_replacement_to_selected(
        self, new_text: str, *, fallback_entry: Optional[CharEntry] = None,
    ) -> int:
        """把 ``new_text`` 应用到 gallery 当前选中的所有"当前页"出现。

        - 多选 (>1)：仅替换在当前页的那部分 entry；跨页 entry 跳过，避免
          无声修改不可见页（status 里会写出"跨页 N 处未改"）。
        - 单选 / 无选区：退化为 ``fallback_entry`` 单点替换。
        返回真正成功 insertText 的处数。
        """
        sel_model = self._gallery_view.selectionModel()
        selected = list(sel_model.selectedIndexes()) if sel_model else []
        entries: list[CharEntry] = []
        skipped_offpage = 0
        if len(selected) > 1:
            cur_page_path = self._pages[self._current_page_idx].display_image_path \
                if self._pages else None
            for idx in selected:
                e = idx.data(Qt.ItemDataRole.UserRole)
                if e is None:
                    continue
                if cur_page_path is not None and e.page_path != cur_page_path:
                    skipped_offpage += 1
                    continue
                entries.append(e)
        if not entries and fallback_entry is not None:
            entries = [fallback_entry]
        if not entries:
            return 0
        anchor = entries[0]
        token = anchor.token_text or anchor.char
        token_len = max(1, len(token))
        doc = self._text_edit.document()
        positions: list[int] = []
        for e in entries:
            pos = self._entry_text_pos(e)
            if pos is not None:
                positions.append(pos)
        positions.sort(reverse=True)
        applied = 0
        for pos in positions:
            cur = QTextCursor(doc)
            cur.setPosition(pos)
            for _ in range(token_len):
                cur.movePosition(
                    QTextCursor.MoveOperation.NextCharacter,
                    QTextCursor.MoveMode.KeepAnchor,
                )
            if cur.hasSelection():
                cur.insertText(new_text)
                applied += 1
        if applied == 0 and fallback_entry is not None:
            # 兜底：用旧的"先 highlight 再 insertText"路径
            self._highlight_char_in_text(token, focus_entry=fallback_entry)
            cursor = self._text_edit.textCursor()
            if cursor.hasSelection():
                cursor.insertText(new_text)
                self._text_edit.setTextCursor(cursor)
                applied = 1
        if skipped_offpage:
            # 在 status 里附带"未改"数量，老实告诉用户
            self._status_lbl.setText(
                f"● 已应用到 {applied} 处（跨页 {skipped_offpage} 处未改），待保存"
            )
            self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")
        return applied

    def _apply_batch_input(self) -> None:
        """proof-slot-residual 第 2 任务：行内输入框 + Enter / 应用按钮。"""
        new_text = self._batch_input.text()
        if not new_text:
            self._status_lbl.setText("批量改字：请先输入替换文本")
            return
        anchor = self._current_candidate_entry
        applied = self._apply_replacement_to_selected(new_text, fallback_entry=anchor)
        if applied == 0:
            self._status_lbl.setText("批量改字：当前页没有可替换的目标")
            return
        self._text_edit.setFocus()
        if applied > 1:
            self._status_lbl.setText(f"● 已批量应用 \"{new_text}\" 到 {applied} 处，待保存")
        else:
            self._status_lbl.setText(f"● 已应用 \"{new_text}\"，待保存")
        self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")

    def _select_all_gallery(self) -> None:
        """proof-slot-residual 第 2 任务：Ctrl+A / 按钮全选当前字所有出现。"""
        count = self._gallery_model.rowCount()
        if count <= 0:
            return
        sel = self._gallery_view.selectionModel()
        top = self._gallery_model.index(0, 0)
        bot = self._gallery_model.index(count - 1, 0)
        from PySide6.QtCore import QItemSelection
        sel.select(
            QItemSelection(top, bot),
            sel.SelectionFlag.ClearAndSelect,
        )
        sel.setCurrentIndex(top, sel.SelectionFlag.NoUpdate)

    def _clear_gallery_selection(self) -> None:
        """proof-slot-residual 第 2 任务：Esc / 按钮清空多选回到单选。"""
        sel = self._gallery_view.selectionModel()
        cur = sel.currentIndex()
        sel.clearSelection()
        if cur.isValid():
            sel.select(cur, sel.SelectionFlag.Select)
            sel.setCurrentIndex(cur, sel.SelectionFlag.NoUpdate)

    def _on_gallery_selection_changed(self, *_args) -> None:
        """选区数量变化 → 刷新批量按钮可用态、刷新标题里的"已选 N 个"。"""
        sel = self._gallery_view.selectionModel() if hasattr(self, "_gallery_view") else None
        n = len(sel.selectedIndexes()) if sel else 0
        multi = n > 1
        if hasattr(self, "_batch_btn"):
            self._batch_btn.setEnabled(n >= 1)
            self._batch_input.setEnabled(n >= 1)
            self._batch_clear_btn.setEnabled(multi)
        # 标题刷新：复用 _sync_gallery_entry 末尾的格式化，但只在有 current 时
        cur = sel.currentIndex() if sel else None
        if cur is not None and cur.isValid():
            self._refresh_gallery_header(cur, n)

    def _refresh_gallery_header(self, index: QModelIndex, sel_count: int) -> None:
        entry = index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None
        if entry is None or not self._selected_char:
            return
        total = len(self._char_svc.query(self._selected_char))
        extra = f"  ·  已选 {sel_count} 个" if sel_count > 1 else ""
        self._gallery_hdr.setText(
            f'"{self._selected_char}"  共 {total} 处  '
            f'[第 {entry.page_number} 页 / 第 {entry.char_idx + 1} 位]{extra}'
        )

    def _line_char_at(self, text: str, index: int) -> str:
        if 0 <= index < len(text):
            return text[index]
        return ""

    def _candidate_signals_for_entry(self, entry: CharEntry) -> str:
        block = self._block_for_entry(entry)
        block_info = block.block_type.value if block else "unknown"
        note = (block.note or "").split("|", 1)[0].strip() if block and block.note else ""
        parts = [
            f"bbox={entry.bbox_source}/{entry.bbox_granularity}",
            f"collection={entry.collection_kind}",
            f"conf={entry.confidence:.2f}",
            f"layout={block_info}",
        ]
        if note:
            parts.append(f"vl_note={note[:24]}")
        return "；".join(parts)

    def _block_for_entry(self, entry: CharEntry) -> Optional[Block]:
        for page in self._pages:
            if page.page_number != entry.page_number:
                continue
            for block in page.blocks:
                if block.order == entry.block_order:
                    return block
        return None

    # ─────────────────── Gallery 点击 ───────────────────────

    def _on_gallery_clicked(self, index: QModelIndex) -> None:
        self._sync_gallery_entry(index)

    def _on_gallery_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        self._sync_gallery_entry(current)

    def _sync_gallery_entry(self, index: QModelIndex) -> None:
        entry: Optional[CharEntry] = index.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        self._current_candidate_entry = entry
        self._highlight_char_in_viewer(entry)  # 可能触发翻页
        self._update_candidate_panel(entry)
        # 更新标题：让用户清楚当前看的是哪页
        if self._selected_char:
            n = len(self._char_svc.query(self._selected_char))
            sel_count = len(self._gallery_view.selectionModel().selectedIndexes())
            extra = f" · 已选 {sel_count} 个" if sel_count > 1 else ""
            self._gallery_hdr.setText(
                f'"{self._selected_char}"  共 {n} 处  '
                f'[第 {entry.page_number} 页 / 第 {entry.char_idx + 1} 位]{extra}'
            )
            # 传入 entry 以精准定位到该出现，而非首次出现
            self._highlight_char_in_text(self._selected_char, focus_entry=entry)
            # proof-interaction-slots 第 6 任务：点 gallery 切图 = 直接进入该字
            # 编辑态。_highlight_char_in_text 已把该字选中，这里另外把焦点转到
            # _text_edit，用户可以直接键入替代。不抢击键快捷：Alt+←/→ 仍
            # 可在 gallery 里跳下一个字（见 _go_prev_gallery / _go_next_gallery）。
            self._text_edit.setFocus()

    # ─────────────────── 原图 block 点击 ────────────────────

    def _on_block_clicked(self, block: Block) -> None:
        for entry in self._text_map:
            line, ci, start, end = entry
            if any(line is ln for ln in block.lines):
                cursor = self._text_edit.textCursor()
                cursor.setPosition(start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.ensureCursorVisible()
                break

    # ─────────────────── 保存 ───────────────────────────────

    def _on_text_changed(self) -> None:
        if not self._updating:
            self._status_lbl.setText("\u25cf \u672a\u4fdd\u5b58")
            self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")

    def _save_page_text(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        flat = self._text_edit.toPlainText()
        lines_text = flat.split("\n")
        changed = False
        idx = 0
        last_block: Optional[Block] = None
        for block, line, _line_idx in iter_unique_page_text_lines(page):
            if last_block is not None and block is not last_block:
                idx += 1  # 跳过 block 间空行
            last_block = block
            if idx < len(lines_text):
                new_displayed = lines_text[idx].rstrip()
                # 走 quality_probe 桥：如有 probe，显示空间 → 真实空间转换 + observation 落地
                if _vproof_save_displayed_line(page, block, line, new_displayed):
                    changed = True
                    self._bus.publish(
                        "line.proof_changed",
                        page_id=page.id,
                        line_id=line.id,
                        status=line.proof_status.value,
                        origin=id(self),
                    )
            idx += 1
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 已保存" if changed else "无变更")
        self._status_lbl.setStyleSheet(
            "color: #4CAF50; font-size: 12px;" if changed else ""
        )
        # Phase 23 blocker：保存后必须把 _loaded_text 同步到当前 _text_edit
        # 内容，否则后续同页外部 line.proof_changed 触发的 dirty 检查
        # 会一直为 True，把 HProof 刚改的别行又用 _text_edit 里 stale
        # 的整页文本（由 _save_page_text 回写）覆盖掉。
        # 取 toPlainText 而非重建 _build_text_map：保存路径会做 scrub
        # （fake_char 落盘前移除）；以"用户当前所见"为基线最直观，
        # 也避免在保存后强制 setPlainText 把光标位置打断。
        self._loaded_text = self._text_edit.toPlainText()
        if changed:
            self._char_svc.build(self._pages)
            self._rebuild_char_list()

    def _mark_page_ok(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        for _block, line, _line_idx in iter_unique_page_text_lines(page):
            if line.proof_status == ProofStatus.UNCHECKED:
                line.proof_status = ProofStatus.OK
                self._bus.publish(
                    "line.proof_changed",
                    page_id=page.id,
                    line_id=line.id,
                    status=ProofStatus.OK.value,
                    origin=id(self),
                )
        self.proof_saved.emit()
        self._status_lbl.setText("本页已确认")

    # ─────────────────── 翻页 ───────────────────────────────

    def _prev_page(self) -> None:
        self._load_page(self._current_page_idx - 1)

    def _next_page(self) -> None:
        self._load_page(self._current_page_idx + 1)

    def _on_external_line_changed(self, **kwargs) -> None:
        """收到外部（横校）发来的 line.proof_changed → 当前页若包含这一行，
        重渲染当前页文本，让显示和 line.text 同步。

        line.text 已被对方 update_text 改过，self._text_map 里的 (Line, ...) 引用
        指向同一对象，所以重新 _build_text_map 即拿到新值。
        """
        if kwargs.get("origin") == id(self):
            return
        if not self._pages:
            return
        line_id = kwargs.get("line_id")
        page_id = kwargs.get("page_id")
        if line_id is None:
            return
        cur_page = self._pages[self._current_page_idx]
        if page_id is not None and cur_page.id != page_id:
            return
        # 当前页有这一行才重渲染
        for _block, line, _li in iter_unique_page_text_lines(cur_page):
            if line.id == line_id:
                # Phase 22 blocker 1：先把用户在 _text_edit 里尚未保存的输入
                # 落盘（走 _save_page_text 同样的 quality_probe 桥），否则
                # 紧接着的 _load_page 会用 page 当前 line.text 重新渲染，
                # 把用户在编辑的文本静默覆盖掉。
                if self._text_edit.toPlainText() != self._loaded_text:
                    self._save_page_text()
                self._load_page(self._current_page_idx)
                # char list / gallery 也需要重建以反映新文本
                self._char_svc.build(self._pages)
                self._rebuild_char_list()
                return

    def refresh_quality_probe_state(self) -> None:
        """供 main_window 进入纵校步骤时调用：当前页若已加载，重新渲染让显示
        空间文本与全局 active store 对齐。"""
        if not self._pages:
            return
        # 重新触发 _load_page，让 _build_text_map 用最新的 active store 渲染
        self._load_page(self._current_page_idx)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_PageUp:
            self._prev_page()
        elif key == Qt.Key.Key_PageDown:
            self._next_page()
        else:
            super().keyPressEvent(event)
