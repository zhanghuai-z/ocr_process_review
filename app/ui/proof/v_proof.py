"""纵校面板（PRD 3.3）：参照 ui.jpg 布局重构。

布局：
  ┌─────────────┬──────────────────────────────────────────────┐
  │ 单字列表    │  相同字索引 gallery（水平条，可左右滚动）    │
  │（频次排序） │──────────────────────────────────────────────│
  │             │  OCR 文本（只读）    │ 原图 + 高亮框         │
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
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    QAbstractListModel, QEvent, QModelIndex, QPoint, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtGui import (
    QColor, QImage, QIcon, QPainter, QPen, QPixmap,
    QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSplitter, QStyle, QTextEdit,
    QStyledItemDelegate, QVBoxLayout, QWidget,
)

from app.core.block_attributes import block_display_label
from app.models import BBox, Block, Line, Page, ProofStatus
from app.models.layout_block_view import iter_page_layout_block_views
from app.models.ocr_observation import (
    block_ocr_line_observations,
    line_ocr_bbox,
)
from app.core.page_image_cache import PageImageCache
from app.core.proof_change import ProofChangeSet
from app.core.proof_line_facts import proof_display_text, proof_ocr_text, proof_status
from app.core.proof_line_utils import iter_unique_page_text_line_views, iter_unique_page_text_lines
from app.core.proof_occurrence import (
    ProofOccurrence,
    line_signature,
    occurrence_from_entry,
    proof_entry_page_identity_key,
    proof_occurrence_key,
    proof_page_identity_key,
    resolve_entry_owner,
    resolve_occurrence_key_in_pages,
)
from app.core.proof_state import (
    TOPIC_LINE_PROOF_CHANGED,
    CandidateSet,
    ProofSelection,
    ProofUpdateRequest,
    proof_request_matches_line,
    proof_request_matches_page,
)
from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp
from app.services.char_index_service import CharEntry, CharIndexService, char_entry_display_text
from app.services.proof_edit_service import (
    ProofEditService,
    ProofSpanReplacement,
)
from app.services.proof_image_service import verified_char_crop
from app.services.proof_occurrence_session import VProofOccurrenceSession
from app.services.proof_rebuild_gate import proof_rebuild_gate_for_reference_context
from app.ui.proof.confidence_utils import line_confidence, normalize_confidence
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.effects import apply_soft_shadow
from app.ui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)


def _observation_lines_for_block(block: Block) -> list[Line]:
    return block_ocr_line_observations(block)


def _line_belongs_to_observation_block(block: Block, line: Line | None) -> bool:
    return line is not None and any(candidate is line for candidate in _observation_lines_for_block(block))


CHAR_LIST_THUMB = 18
# 相同字索引用较大的缩略图，优先保证 CJK 字形和标点细节可读。
GALLERY_THUMB   = 56
GALLERY_ITEMS_PER_ROW = 6
GALLERY_MAX_ROWS = 3
LOW_CONF        = 0.80
PROOF_TEXT_FONT_FAMILY = "'SimHei','Microsoft YaHei UI','Noto Sans CJK SC','PingFang SC','SimSun',sans-serif"
PROOF_TEXT_FONT_CSS = f"font-family:{PROOF_TEXT_FONT_FAMILY};"
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
class _VProofLineEdit:
    occurrence_keys: Tuple[tuple[object, ...], ...]
    page_key: tuple[object, ...] | None
    page_uid: str
    page_id: int | None
    block_uid: str
    block_id: int | None
    block_index: int
    line_uid: str
    line_id: int | None
    line_index: int
    before_text: str
    after_text: str
    before_signature: str
    after_signature: str


@dataclass(frozen=True)
class _VProofEditAction:
    edits: Tuple[_VProofLineEdit, ...]


def _gallery_crop_pad_for_entry(entry: CharEntry) -> Optional[int]:
    """纵校 gallery 的字符裁图 padding 策略。

    CJK 字符保留 shared service 的自适应 padding，避免缩略图太贴边。英文、
    数字字符框通常来自 EngCut 或 micro_recblock 的单字符 bbox，字距很窄；
    只给 1px 安全边，避免既贴边又把邻字明显带进来。半角标点/符号本体通常
    很小（例如 "." 只有数 px），需要 2px 上下文才能看清形态。
    """
    if entry.collection_kind != "char":
        return None
    if (entry.bbox_granularity or "").strip().lower() != "char":
        return None
    char = entry.char or ""
    if len(char) == 1 and char.isascii() and not char.isspace():
        return 1 if char.isalnum() else 2
    return None


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
            # 源图高于实际显示像素密度，让 delegate 只做 down-scale，避免字形发糊。
            return verified_char_crop(
                self._cache, entry.page_path, entry.bbox,
                _GalleryDelegate.SOURCE_PX,
                pad=_gallery_crop_pad_for_entry(entry),
            )
        if role == Qt.ItemDataRole.DisplayRole:
            display_key = char_entry_display_text(entry)
            return f"{index.row() + 1:03d}\nP{entry.page_number}-{display_key}"
        if role == Qt.ItemDataRole.ToolTipRole:
            display_key = char_entry_display_text(entry)
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
    # 给缩略图周围留呼吸空间，避免 item rect 裁切边缘笔画。
    SIZE = GALLERY_THUMB + 14
    # 源 pixmap 像素密度 = 2× cell size；delegate 始终 down-scale。
    SOURCE_PX = (GALLERY_THUMB + 14) * 2

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        r = option.rect
        pix: Optional[QPixmap] = index.data(Qt.ItemDataRole.DecorationRole)
        # 白底打底，避免 list view 默认背景透出导致相邻 cell 视觉粘连。
        painter.fillRect(r, QColor("#FFFDF8"))
        img_r = r.adjusted(2, 2, -2, -2)
        if pix and not pix.isNull():
            scaled = pix.scaled(
                img_r.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            dx = (img_r.width() - scaled.width()) // 2
            dy = (img_r.height() - scaled.height()) // 2
            painter.drawPixmap(img_r.x() + dx, img_r.y() + dy, scaled)
            # cell 之间画 1px 浅灰分隔，让相邻字不再视觉粘在一起。
            painter.setPen(QColor("#E7E2D8"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r.adjusted(0, 0, -1, -1))
        else:
            painter.fillRect(img_r, QColor("#FFFDF8"))
            # 裁图失败时显示 token 内容作为占位，避免白块无信息
            entry = index.data(Qt.ItemDataRole.UserRole)
            fallback = char_entry_display_text(entry) if entry else "?"
            painter.setPen(QColor("#5C6B58"))
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
            pen = QPen(QColor("#5C6B58"), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r.adjusted(1, 1, -1, -1))

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        return QSize(self.SIZE, self.SIZE)


# ─────────────────────────────────────────────────────────────
# 纵校面板
# ─────────────────────────────────────────────────────────────


class _GalleryListView(QListView):
    """相同字索引 gallery 的 QListView 子类。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gallery_owner: Optional["VProofPanel"] = None
        # QListView 默认不接 IME；这里显式打开中文输入法提交路径。
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def set_gallery_owner(self, owner: "VProofPanel") -> None:
        self._gallery_owner = owner

    def inputMethodQuery(self, query):  # type: ignore[override]
        # 必须告诉 IME 自己 enabled，否则 inputMethodEvent 不会派过来
        if query == Qt.InputMethodQuery.ImEnabled:
            return True
        return super().inputMethodQuery(query)

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        """Handle committed IME text through the same direct-overwrite path.

        中文/日文 IME 不走 keyPressEvent 带 text；它走 QInputMethodEvent：
          - preeditString()：候选条还在拼，不要落到模型上（吞掉，不渲染）
          - commitString()：用户确认了，整段字符串送过来——这里就是真实输入。

        commit string 直接当一段新文本覆盖当前 entry（可能多字，比如"好的"，
        _gallery_direct_overwrite 内部走 _apply_replacement_to_selected，已
        经支持多字 token）。"""
        owner = self._gallery_owner
        commit = event.commitString() if event else ""
        if owner is None or not commit:
            super().inputMethodEvent(event)
            return
        if owner._gallery_direct_overwrite(commit):
            event.accept()
            return
        super().inputMethodEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        owner = self._gallery_owner
        if owner is None:
            super().keyPressEvent(event)
            return
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        key = event.key()
        if ctrl and key == Qt.Key.Key_Z:
            if shift:
                consumed = owner._redo_vproof_edit()
            else:
                consumed = owner._undo_vproof_edit()
            if consumed:
                event.accept()
                return
            super().keyPressEvent(event)
            return
        if ctrl and key == Qt.Key.Key_Y:
            if owner._redo_vproof_edit():
                event.accept()
                return
            super().keyPressEvent(event)
            return
        # 任何 Ctrl/Alt/Meta（含 Ctrl+A 全选、Ctrl+Alt+→ 步进、Alt+↑ 跨行）一律放行
        disallowed = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if mods & disallowed:
            super().keyPressEvent(event)
            return
        # Backspace / Delete 裸键 → 把当前槽位填空白（保留长度）
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if owner._gallery_direct_blank():
                event.accept()
                return
            super().keyPressEvent(event)
            return
        # 其它可打印单字符 → 覆盖当前 entry
        text = event.text()
        if len(text) == 1 and text.isprintable() and not text.isspace():
            if owner._gallery_direct_overwrite(text):
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        owner = self._gallery_owner
        if owner is not None and event.button() == Qt.MouseButton.RightButton:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            index = self.indexAt(pos)
            if index.isValid():
                sel = self.selectionModel()
                if sel is not None and not sel.isSelected(index):
                    sel.select(index, sel.SelectionFlag.ClearAndSelect)
                    sel.setCurrentIndex(index, sel.SelectionFlag.Current)
                owner._sync_gallery_entry(index)
            owner._show_edit_bubble_at(pos)
            event.accept()
            return
        super().mousePressEvent(event)

class VProofPanel(QWidget):
    """纵校面板：参照 ui.jpg 三区域布局（左单字列表 + 顶gallery + 底OCR/图）。"""

    proof_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session = VProofOccurrenceSession()
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        # VProof is an editing surface: when old or degraded OCR lacks
        # per-character boxes, line-level fallback entries are preferable to an
        # empty index.  Verified Hanwang/EngCut char boxes still win whenever
        # they are present.
        self._char_svc = CharIndexService(include_non_cjk=True, include_fallback=True)
        self._char_index_page_signatures: dict[tuple[object, ...], tuple] = {}
        self._gallery_model = _GalleryModel(self._cache)
        self._selected_char: str = ""
        self._current_selection: Optional[ProofSelection] = None
        self._current_candidate_entry: Optional[CharEntry] = None
        self._candidate_buttons: List[QPushButton] = []
        self._updating = False
        self._vproof_undo_stack: list[_VProofEditAction] = []
        self._vproof_redo_stack: list[_VProofEditAction] = []
        self._vproof_history_limit = 5
        self._vproof_restoring_history = False
        # _text_edit 是只读上下文。这里保留加载时的文本/行身份，供刷新、
        # 高亮和历史恢复校验使用，不再作为可提交的编辑缓冲。
        # 外部事件 → 重渲染通过 QTimer 单次延后合并。
        self._external_refresh_timer = QTimer(self)
        self._external_refresh_timer.setSingleShot(True)
        self._external_refresh_timer.setInterval(80)
        self._external_refresh_timer.timeout.connect(self._do_external_refresh)
        self._build_ui()
        # H/V 校对联动：订阅其他 panel 的编辑事件，本 panel 自己 publish 的事件
        # 通过 origin == id(self) 过滤掉以避免回路。
        # 保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            TOPIC_LINE_PROOF_CHANGED, self._on_external_line_changed,
        )
        self.destroyed.connect(lambda *_: self._teardown_bus())

    def _teardown_bus(self) -> None:
        """释放 ProofStateBus 订阅，幂等。"""
        unsub = getattr(self, "_bus_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._bus_unsub = None

    # ─────────────────── UI ───────────────────────────────────

    def _build_ui(self) -> None:
        self.setObjectName("proofRoot")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._btn_save = QPushButton("刷新")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.setToolTip("刷新当前纵校上下文（Ctrl+S）")
        self._btn_save.setMinimumHeight(30)
        self._page_label = QLabel("页 0 / 0")
        self._page_label.setObjectName("muted")
        self._page_label.hide()
        self._conf_badge = ConfidenceBadge(1.0)
        self._conf_badge.hide()

        # ── 主体：水平分割（左单字列表 | 右主区域）─────────────
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setObjectName("proofSplitter")
        h_split.setHandleWidth(10)

        # 左：单字列表 + 搜索
        self._main_split = h_split
        self._left_box = self._build_char_list()
        self._left_box.setMinimumWidth(170)
        self._left_box.setMaximumWidth(230)
        h_split.addWidget(self._left_box)

        # 右：垂直分割（上gallery | 下文本/图）
        right_box = self._build_right_area()
        h_split.addWidget(right_box)
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 7)
        h_split.setSizes([190, 980])

        root.addWidget(h_split, 1)
        self._build_edit_bubble()

        # ── 信号 ────────────────────────────────────────────
        self._btn_save.clicked.connect(self._refresh_reference_context)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._refresh_reference_context)
        # Gallery navigation keeps the OCR context focused without trapping
        # users in the read-only reference text cursor.
        QShortcut(QKeySequence("Alt+Right"), self, activated=self._go_next_gallery)
        QShortcut(QKeySequence("Alt+Left"), self, activated=self._go_prev_gallery)
        # Global char-index navigation works regardless of focus location.
        QShortcut(QKeySequence("Ctrl+."), self, activated=self._step_char_list_next)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self._step_char_list_prev)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_vproof_or_native)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo_vproof_or_native)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo_vproof_or_native)
        # VProof has no page-level confirmation action; edits are scoped to
        # concrete occurrences/lines.

    def _build_edit_bubble(self) -> None:
        self._edit_bubble = QFrame(self)
        self._edit_bubble.setObjectName("vproofEditBubble")
        self._edit_bubble.setStyleSheet(
            "QFrame#vproofEditBubble {"
            " background: rgba(255, 253, 248, 245);"
            " border: 1px solid #E7E2D8;"
            " border-radius: 16px;"
            "}"
            "QLineEdit#vproofEditBubbleInput {"
            " background: transparent;"
            " border: none;"
            " padding: 7px 12px;"
            " font-size: 18px;"
            " color: #2C2C2C;"
            " selection-background-color: #ECE8DF;"
            "}"
        )
        apply_soft_shadow(self._edit_bubble, blur_radius=24, y_offset=4, alpha=24)
        row = QHBoxLayout(self._edit_bubble)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(0)
        self._edit_bubble_input = QLineEdit()
        self._edit_bubble_input.setObjectName("vproofEditBubbleInput")
        self._edit_bubble_input.setFrame(False)
        self._edit_bubble_input.setMaxLength(16)
        self._edit_bubble_input.returnPressed.connect(self._apply_edit_bubble)
        self._edit_bubble_input.installEventFilter(self)
        row.addWidget(self._edit_bubble_input)
        self._edit_bubble.resize(132, 42)
        self._edit_bubble.hide()

    def _build_char_list(self) -> QWidget:
        box = QFrame()
        box.setObjectName("proofLeftPane")
        apply_soft_shadow(box, blur_radius=18, y_offset=3, alpha=10)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

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
        self._char_list.setObjectName("charIndexList")
        self._char_list.setStyleSheet(
            "QListWidget { background:#ffffff; border: none; }"
            "QListWidget::item { background: transparent; }"
        )
        # The char index is text-only; gallery thumbnails carry the image detail.
        self._char_list.setIconSize(QSize(0, 0))
        self._char_list.setSpacing(2)
        # Multi-select merges occurrences for several glyph buckets into one gallery.
        self._char_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._char_list.itemClicked.connect(self._on_char_clicked)
        self._char_list.itemSelectionChanged.connect(self._on_char_selection_changed)
        self._char_list.currentItemChanged.connect(
            lambda cur, _prev: self._on_char_clicked(cur) if cur else None
        )
        sc_charlist_all = QShortcut(QKeySequence("Ctrl+A"), self._char_list)
        sc_charlist_all.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_charlist_all.activated.connect(self._select_all_visible_chars)
        sc_charlist_esc = QShortcut(QKeySequence("Escape"), self._char_list)
        sc_charlist_esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_charlist_esc.activated.connect(self._clear_char_list_multi)
        layout.addWidget(self._char_list)
        return box

    def _build_right_area(self) -> QWidget:
        box = QWidget()
        box.setObjectName("proofContentPane")
        root = QHBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._content_split = QSplitter(Qt.Orientation.Horizontal)
        self._content_split.setObjectName("proofContentSplitter")
        self._content_split.setHandleWidth(10)

        self._proof_column = QFrame()
        self._proof_column.setObjectName("proofCenterPane")
        apply_soft_shadow(self._proof_column, blur_radius=18, y_offset=3, alpha=10)
        col = QVBoxLayout(self._proof_column)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(10)

        # 左列上：gallery 网格。与同列 OCR 文本自然等宽。
        self._gallery_box = self._build_gallery_strip()
        self._resize_gallery_for_entries(0)
        col.addWidget(self._gallery_box)

        self._candidate_box = self._build_candidate_panel()
        col.addWidget(self._candidate_box)

        self._ocr_text_box = self._build_ocr_text()
        col.addWidget(self._ocr_text_box, 1)

        self._viewer_box = self._build_viewer()
        # 左侧 proof 列和右侧原图列保持均衡，避免最大化后只放大原图。
        # 1) proof_column 必须至少装得下 gallery 一行 6 个 56px 缩略图：
        #    6 *(THUMB + 8 spacing margin) + 内框 padding ≈ 6*64 + 40 ≈ 424。
        #    再留一点 OCR 文本/候选区可用空间 → 设 520。
        # 2) viewer_box 不再"独吞 4 倍"；给它一个合理下限即可，让用户拖
        #    splitter 时不会缩没。
        # 3) stretchFactor 改成 1:1，让用户最大化窗口时两边等比例增长，
        #    而不是把所有新空间都给图片。
        # 4) setSizes 给出第一次显示时的明确尺寸，避免 Qt 用 sizeHint
        #    自动给图片偏大的初值（这正是"怎么图片还变大了"的来源）。
        proof_min_w = max(
            520,
            GALLERY_ITEMS_PER_ROW * (GALLERY_THUMB + 8) + 40,
        )
        self._proof_column.setMinimumWidth(proof_min_w)
        self._viewer_box.setMinimumWidth(300)
        self._content_split.addWidget(self._proof_column)
        self._content_split.addWidget(self._viewer_box)
        self._content_split.setStretchFactor(0, 1)
        self._content_split.setStretchFactor(1, 1)
        self._content_split.setSizes([proof_min_w + 220, 600])
        root.addWidget(self._content_split)
        return box

    def _build_candidate_panel(self) -> QWidget:
        # 候选区只显示可操作候选，诊断信息放到 tooltip，避免挤占 OCR 文本区。
        box = QFrame()
        box.setObjectName("candidatePanel")
        box.setStyleSheet("border:0;")
        box.setFixedHeight(72)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self._candidate_title = QLabel("候选字")
        self._candidate_title.setObjectName("sectionTitle")
        self._candidate_buttons_row = QHBoxLayout()
        self._candidate_buttons_row.setSpacing(6)
        layout.addWidget(self._candidate_title)
        layout.addLayout(self._candidate_buttons_row)
        return box

    def _build_gallery_strip(self) -> QWidget:
        box = QFrame()
        box.setObjectName("proofCard")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        hdr = QHBoxLayout()
        self._gallery_hdr = QLabel("相同字索引（请先在左侧选择一个字)")
        self._gallery_hdr.setObjectName("sectionTitle")
        hdr.addWidget(self._gallery_hdr)
        hdr.addStretch()
        hdr.addWidget(self._btn_save)
        layout.addLayout(hdr)

        self._gallery_view = _GalleryListView()
        self._gallery_view.set_gallery_owner(self)
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.setItemDelegate(_GalleryDelegate(self._gallery_view))
        self._gallery_view.setViewMode(QListView.ViewMode.IconMode)
        self._gallery_view.setFlow(QListView.Flow.LeftToRight)  # 水平排列
        self._gallery_view.setWrapping(True)
        self._gallery_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._gallery_view.setMovement(QListView.Movement.Static)
        self._gallery_view.setUniformItemSizes(True)
        self._gallery_view.setSpacing(4)
        self._gallery_view.setStyleSheet(
            "QListView { background:#ffffff; border: none; }"
            "QListView::item { background: transparent; }"
        )
        self._gallery_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gallery_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._gallery_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # 让方向键 / Shift+方向键 / Ctrl+点击 走 Qt 原生 ExtendedSelection 行为。
        self._gallery_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # QAbstractItemView 初始化后可能清掉 IME 标志，这里兜底打开。
        self._gallery_view.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self._gallery_view.selectionModel().currentChanged.connect(
            self._on_gallery_current_changed
        )
        # 选区数量变化 → 刷新标题 + 按钮状态。
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
        # 这些快捷键都是 widget-scoped，不抢全局；Qt 原生方向键在 gallery 上
        # 已经移动 currentIndex，下面补充跨 wrap 顺序和多选扩展语义：
        #   Ctrl+Alt+Right / Ctrl+Alt+Left = 强制按 entry 编号 ±1 步进（不论
        #       grid 行布局如何，跨 wrap 也保证顺序），并把单选切到那里。
        #   Shift+Alt+Right / Shift+Alt+Left = 在不动 current 的前提下，把
        #       ±1 邻居加入/移出 selection（扩展选择）。
        #   Alt+Up / Alt+Down = 跨 wrap 行步进（按当前实际 ITEMS_PER_ROW）。
        for keystr, fn in (
            ("Ctrl+Alt+Right", lambda: self._step_gallery_singleton(+1)),
            ("Ctrl+Alt+Left",  lambda: self._step_gallery_singleton(-1)),
            ("Shift+Alt+Right", lambda: self._extend_gallery_selection(+1)),
            ("Shift+Alt+Left",  lambda: self._extend_gallery_selection(-1)),
            ("Alt+Up",   lambda: self._step_gallery_row(-1)),
            ("Alt+Down", lambda: self._step_gallery_row(+1)),
        ):
            sc = QShortcut(QKeySequence(keystr), self._gallery_view)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        layout.addWidget(self._gallery_view)
        # 初始按钮态
        self._on_gallery_selection_changed()
        return box

    def _resize_gallery_for_entries(self, count: int) -> None:
        # gallery 现在只保留标题 + 缩略索引，编辑入口改为右键气泡。
        # chrome 高度按当前单一 proof shell 估算。
        rows = min(
            GALLERY_MAX_ROWS,
            max(1, (max(1, count) + GALLERY_ITEMS_PER_ROW - 1) // GALLERY_ITEMS_PER_ROW),
        )
        item_h = GALLERY_THUMB + 18
        chrome_h = 48
        view_h = rows * item_h + 8
        self._gallery_box.setFixedHeight(view_h + chrome_h)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is getattr(self, "_edit_bubble_input", None):
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._hide_edit_bubble()
                    return True
            if event.type() == QEvent.Type.FocusOut:
                QTimer.singleShot(0, self._hide_edit_bubble)
        return super().eventFilter(watched, event)

    def _show_edit_bubble_at(self, gallery_pos: QPoint) -> None:
        if not hasattr(self, "_edit_bubble"):
            return
        entry = self._current_candidate_entry
        if entry is None:
            self._status_lbl.setText("请先选中一个字位")
            return
        sel = self._gallery_view.selectionModel()
        selected_count = len(sel.selectedIndexes()) if sel else 0
        token = char_entry_display_text(entry)
        self._edit_bubble_input.blockSignals(True)
        self._edit_bubble_input.setText(token)
        self._edit_bubble_input.selectAll()
        self._edit_bubble_input.blockSignals(False)
        self._edit_bubble_input.setPlaceholderText(
            f"替换 {selected_count if selected_count > 1 else 1} 处"
        )
        panel_pos = self._gallery_view.mapTo(self, gallery_pos)
        x = max(8, min(panel_pos.x() + 8, self.width() - self._edit_bubble.width() - 8))
        y = max(8, min(panel_pos.y() + 8, self.height() - self._edit_bubble.height() - 8))
        self._edit_bubble.move(x, y)
        self._edit_bubble.show()
        self._edit_bubble.raise_()
        self._edit_bubble_input.setFocus()

    def _hide_edit_bubble(self) -> None:
        if hasattr(self, "_edit_bubble"):
            self._edit_bubble.hide()

    def _apply_edit_bubble(self) -> None:
        new_text = self._edit_bubble_input.text()
        if not new_text:
            self._status_lbl.setText("改字：请输入替换文本")
            return
        entry = self._current_candidate_entry
        if entry is None:
            self._status_lbl.setText("改字：先在索引中选中一个字位")
            return
        applied = self._apply_replacement_to_selected(new_text, fallback_entry=entry)
        if applied == 0:
            self._status_lbl.setText("改字：当前页没有可替换的目标")
            return
        self._hide_edit_bubble()
        self._gallery_view.setFocus()
        if applied > 1:
            self._status_lbl.setText(f'✓ 已应用 "{new_text}" 到 {applied} 处')
        else:
            self._status_lbl.setText(f'✓ 已应用 "{new_text}"')
        self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")

    def _make_vproof_line_edit(
        self,
        page: Page,
        block: Block,
        line: Line,
        line_index: int,
        *,
        occurrence_keys: Tuple[tuple[object, ...], ...] = tuple(),
        before_text: str,
        before_signature: str,
    ) -> _VProofLineEdit:
        return _VProofLineEdit(
            occurrence_keys=occurrence_keys,
            page_key=self._char_index_page_key(page),
            page_uid=page.uid,
            page_id=page.id,
            block_uid=block.uid,
            block_id=block.id,
            block_index=self._layout_block_index(page, block),
            line_uid=line.uid,
            line_id=line.id,
            line_index=line_index,
            before_text=before_text,
            after_text=proof_display_text(line),
            before_signature=before_signature,
            after_signature=line_signature(line),
        )

    def _push_vproof_edit_action(self, action: _VProofEditAction) -> None:
        if self._vproof_restoring_history or not action.edits:
            return
        if self._vproof_undo_stack and self._vproof_undo_stack[-1] == action:
            return
        self._vproof_undo_stack.append(action)
        if len(self._vproof_undo_stack) > self._vproof_history_limit:
            self._vproof_undo_stack = self._vproof_undo_stack[-self._vproof_history_limit:]
        self._vproof_redo_stack.clear()

    def _clear_vproof_history(self) -> None:
        self._vproof_undo_stack.clear()
        self._vproof_redo_stack.clear()

    def _reject_vproof_history_restore(self, label: str) -> bool:
        self._status_lbl.setText(f"{label}已取消：当前文本与页面状态不一致，请重新进入纵校")
        self._status_lbl.setStyleSheet("color: #D32F2F; font-size: 12px;")
        return False

    def _resolve_vproof_history_edit(
        self, edit: _VProofLineEdit,
    ) -> Optional[tuple[Page, Block, Line, int]]:
        page = self._session.current_page()
        if page is None:
            return None
        if edit.page_key is None or edit.page_key != self._char_index_page_key(page):
            return None
        for occurrence_key in edit.occurrence_keys:
            target = resolve_occurrence_key_in_pages(self._session.pages, occurrence_key)
            if target is None:
                continue
            resolved_page, block, line, line_index = target
            if self._char_index_page_key(resolved_page) == edit.page_key:
                return resolved_page, block, line, line_index
        block: Optional[Block] = None
        if edit.block_uid:
            block = self._layout_block_by_uid(page, edit.block_uid)
        if block is None and edit.block_id is not None:
            block = self._layout_block_by_id(page, edit.block_id)
        if block is None:
            block = self._layout_block_at_index(page, edit.block_index)
        if block is None:
            return None

        line: Optional[Line] = None
        line_index = -1
        observation_lines = _observation_lines_for_block(block)
        if edit.line_uid:
            for idx, candidate in enumerate(observation_lines):
                if candidate.uid == edit.line_uid:
                    line = candidate
                    line_index = idx
                    break
        if line is None and edit.line_id is not None:
            for idx, candidate in enumerate(observation_lines):
                if candidate.id == edit.line_id:
                    line = candidate
                    line_index = idx
                    break
        if line is None and 0 <= edit.line_index < len(observation_lines):
            line = observation_lines[edit.line_index]
            line_index = edit.line_index
        if line is None:
            return None
        return page, block, line, line_index

    def _apply_vproof_history_action(
        self, action: _VProofEditAction, *, undo: bool, label: str,
    ) -> tuple[bool, Optional[_VProofEditAction]]:
        if not action.edits:
            return False, None
        resolved: list[tuple[Page, Block, Line, int, _VProofLineEdit]] = []
        for edit in action.edits:
            target = self._resolve_vproof_history_edit(edit)
            if target is None:
                return self._reject_vproof_history_restore(label), None
            page, block, line, line_index = target
            expected = edit.after_signature if undo else edit.before_signature
            if line_signature(line) != expected:
                return self._reject_vproof_history_restore(label), None
            resolved.append((page, block, line, line_index, edit))

        change = ProofChangeSet()
        changed_lines: list[tuple[Page, Block, Line, int]] = []
        next_edits: list[_VProofLineEdit] = []
        self._vproof_restoring_history = True
        try:
            for page, block, line, line_index, edit in resolved:
                target_text = edit.before_text if undo else edit.after_text
                expected = edit.after_signature if undo else edit.before_signature
                result = ProofEditService.replace_line_text(
                    page,
                    block,
                    line,
                    target_text,
                    expected_signature=expected,
                    write_chars=True,
                )
                if result.blocked:
                    return self._reject_vproof_history_restore(label), None
                if result.changed:
                    change = change.merge(result.change)
                    changed_lines.append((page, block, line, line_index))
                current_signature = line_signature(line)
                if undo:
                    next_edits.append(replace(
                        edit,
                        before_text=proof_display_text(line),
                        before_signature=current_signature,
                    ))
                else:
                    next_edits.append(replace(
                        edit,
                        after_text=proof_display_text(line),
                        after_signature=current_signature,
                    ))
        finally:
            self._vproof_restoring_history = False

        affected_pages: list[Page] = []
        for page, _block, _line, _line_index in changed_lines:
            if page not in affected_pages:
                affected_pages.append(page)
        if affected_pages:
            self._refresh_char_index_for_pages(affected_pages)
            current_page = self._session.current_page()
            if current_page in affected_pages:
                self._refresh_page_text_view(current_page)
        for page, block, line, line_index in changed_lines:
            self._publish_line_update(
                page=page,
                block=block,
                line=line,
                line_index=line_index,
                status=proof_status(line).value,
                source=f"vproof.{label}",
            )
        if change.needs_persist:
            self._emit_proof_change(change)
        self._status_lbl.setText(f"✓ 已{label}")
        self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")
        return True, _VProofEditAction(tuple(next_edits))

    def _apply_native_undo_redo_for_focus(self, *, redo: bool) -> bool:
        focus = QApplication.focusWidget()
        if focus is getattr(self, "_edit_bubble_input", None):
            if redo:
                focus.redo()
            else:
                focus.undo()
            return True
        if focus is self._text_edit:
            doc = self._text_edit.document()
            available = doc.isRedoAvailable() if redo else doc.isUndoAvailable()
            if not available:
                return False
            if redo:
                self._text_edit.redo()
            else:
                self._text_edit.undo()
            return True
        return False

    def _undo_vproof_or_native(self) -> None:
        if QApplication.focusWidget() is getattr(self, "_edit_bubble_input", None):
            self._apply_native_undo_redo_for_focus(redo=False)
            return
        if self._undo_vproof_edit():
            return
        self._apply_native_undo_redo_for_focus(redo=False)

    def _redo_vproof_or_native(self) -> None:
        if QApplication.focusWidget() is getattr(self, "_edit_bubble_input", None):
            self._apply_native_undo_redo_for_focus(redo=True)
            return
        if self._redo_vproof_edit():
            return
        self._apply_native_undo_redo_for_focus(redo=True)

    def _undo_vproof_edit(self) -> bool:
        if not self._vproof_undo_stack:
            return False
        self._hide_edit_bubble()
        action = self._vproof_undo_stack[-1]
        ok, redo_action = self._apply_vproof_history_action(action, undo=True, label="撤销")
        if not ok:
            return True
        if redo_action is not None:
            self._vproof_redo_stack.append(redo_action)
        if len(self._vproof_redo_stack) > self._vproof_history_limit:
            self._vproof_redo_stack = self._vproof_redo_stack[-self._vproof_history_limit:]
        self._vproof_undo_stack.pop()
        return True

    def _redo_vproof_edit(self) -> bool:
        if not self._vproof_redo_stack:
            return False
        self._hide_edit_bubble()
        action = self._vproof_redo_stack[-1]
        ok, undo_action = self._apply_vproof_history_action(action, undo=False, label="重做")
        if not ok:
            return True
        if undo_action is not None:
            self._vproof_undo_stack.append(undo_action)
        if len(self._vproof_undo_stack) > self._vproof_history_limit:
            self._vproof_undo_stack = self._vproof_undo_stack[-self._vproof_history_limit:]
        self._vproof_redo_stack.pop()
        return True

    def _go_next_gallery(self) -> None:
        """Alt+Right: move to the next gallery occurrence."""
        self._step_gallery(+1)

    def _go_prev_gallery(self) -> None:
        """Alt+Left: move to the previous gallery occurrence."""
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

    # Global Ctrl+, / Ctrl+. char-bucket navigation.
    def _step_char_list_next(self) -> None:
        self._step_char_list(+1)

    def _step_char_list_prev(self) -> None:
        self._step_char_list(-1)

    def _step_char_list(self, delta: int) -> None:
        """跨可见 item 步进（跳过被搜索框过滤掉的）。"""
        count = self._char_list.count()
        if count <= 0:
            return
        cur_row = self._char_list.currentRow()
        if cur_row < 0:
            cur_row = 0
        # 在可见 item 中找下一个
        for step in range(1, count + 1):
            row = (cur_row + delta * step) % count
            it = self._char_list.item(row)
            if it is not None and not it.isHidden():
                self._char_list.setCurrentRow(row)
                # 单选语义：清掉之前 Ctrl+A 留下的多选
                self._char_list.clearSelection()
                it.setSelected(True)
                self._on_char_clicked(it)
                return

    def _build_ocr_text(self) -> QWidget:
        box = QFrame()
        box.setObjectName("proofCard")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._text_edit.setStyleSheet(
            f"{PROOF_TEXT_FONT_CSS} font-size:16px; padding:8px;"
        )
        layout.addWidget(self._text_edit)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("noteLabel")
        layout.addWidget(self._status_lbl)
        return box

    def _build_viewer(self) -> QWidget:
        box = QFrame()
        box.setObjectName("proofRightPane")
        apply_soft_shadow(box, blur_radius=18, y_offset=3, alpha=10)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("原图")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self._viewer = ImageViewer()
        layout.addWidget(self._viewer)
        return box

    # ─────────────────── 公共 API ───────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._clear_vproof_history()
        self._session.set_pages(pages)
        self._session.clear_pending_external_refresh()
        self._rebuild_full_char_index(pages)
        if pages:
            self._load_page(0)

    def merge_pages(self, pages: List[Page]) -> None:
        """Merge background OCR pages and reload the read-only context."""
        if not self._session.pages or self._session.reference_context is None:
            self.load_pages(pages)
            return
        current_page = self._session.current_page()
        selected_tokens = self._selected_char_tokens()
        selected_occurrence_keys = tuple(self._session.selected_occurrence_keys)
        self._clear_vproof_history()
        current_key = self._session.current_page_key
        self._session.set_pages(pages, current_page_key=current_key)
        if current_page is not None and self._session.current_page_key != current_key:
            self._session.set_current_page_by_index(self._find_page_index(current_page))
        if self._session.pages:
            self._load_page(self._session.current_page_index())
        self._refresh_char_index_for_changed_pages(pages)
        self._restore_gallery_after_context_reload(selected_tokens, selected_occurrence_keys)
        self._page_label.setText(
            f"页 {self._session.current_page_index() + 1} / {len(self._session.pages)}"
        )

    def _find_page_index(self, target: Page) -> int:
        return self._session.find_page_index(target)

    def reset(self) -> None:
        self._session.reset()
        self._char_svc = CharIndexService(include_non_cjk=True, include_fallback=True)
        self._char_index_page_signatures = {}
        self._char_list.clear()
        self._text_edit.clear()
        self._gallery_model.set_entries([])
        self._resize_gallery_for_entries(0)
        self._current_selection = None
        self._current_candidate_entry = None
        self._clear_vproof_history()
        self._clear_candidate_buttons()
        self._candidate_box.setToolTip("")
        self._page_label.setText("页 0 / 0")

    # ─────────────────── 字符列表 ────────────────────────────

    def _rebuild_char_list(self) -> None:
        # Char index only shows "glyph × count"; gallery carries image evidence.
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

    @staticmethod
    def _bbox_signature(bbox: Optional[BBox]) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        return int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)

    @staticmethod
    def _char_index_page_key(page: Page) -> tuple[object, ...]:
        return proof_page_identity_key(page)

    def _set_reference_context(self, page: Page, loaded_text: str) -> None:
        self._session.loaded_text = loaded_text
        self._session.current_page_key = self._char_index_page_key(page)

    def _entry_belongs_to_current_page(self, entry: CharEntry) -> bool:
        current_page = self._session.current_page()
        if current_page is None:
            return False
        return proof_entry_page_identity_key(entry) == proof_page_identity_key(current_page)

    def _char_index_page_signature(self, page: Page) -> tuple:
        parts: list[tuple] = [(
            "page",
            page.uid,
            page.id,
            page.page_number,
            page.display_image_path,
            page.width,
            page.height,
            id(page),
        )]
        for view, block, line, line_idx in iter_unique_page_text_line_views(page):
            parts.append((
                "line",
                view.uid,
                block.id,
                view.order,
                view.block_type.value,
                line.uid,
                line.id,
                line_idx,
                line_signature(line),
                self._bbox_signature(line_ocr_bbox(line)),
                id(block),
                id(line),
                ))
        return tuple(parts)

    def _rebuild_full_char_index(self, pages: List[Page]) -> None:
        self._char_svc.build(pages)
        self._char_index_page_signatures = {
            self._char_index_page_key(page): self._char_index_page_signature(page)
            for page in pages
        }
        self._rebuild_char_list()

    def _refresh_char_index_for_pages(self, pages: List[Page]) -> bool:
        changed_pages: list[Page] = []
        for page in pages:
            key = self._char_index_page_key(page)
            signature = self._char_index_page_signature(page)
            if self._char_index_page_signatures.get(key) != signature:
                changed_pages.append(page)
                self._char_index_page_signatures[key] = signature
        if not changed_pages:
            return False
        self._char_svc.replace_pages(changed_pages)
        self._rebuild_char_list()
        return True

    def _refresh_char_index_for_changed_pages(self, pages: List[Page]) -> bool:
        current_keys = {self._char_index_page_key(page) for page in pages}
        if set(self._char_index_page_signatures) - current_keys:
            self._rebuild_full_char_index(pages)
            return True
        return self._refresh_char_index_for_pages(pages)

    def _filter_char_list(self, text: str) -> None:
        for i in range(self._char_list.count()):
            item = self._char_list.item(i)
            char = item.data(Qt.ItemDataRole.UserRole) or ""
            item.setHidden(text != "" and text not in char)

    # ─────────────────── 页面加载 ────────────────────────────

    def _load_page(self, idx: int) -> None:
        if not self._session.pages:
            return
        page = self._session.set_current_page_by_index(idx)
        if page is None:
            return
        idx = self._session.current_page_index()
        self._page_label.setText(f"页 {idx + 1} / {len(self._session.pages)}")

        lines = [line for _block, line, _line_idx in iter_unique_page_text_lines(page)]
        scores = [score for ln in lines if (score := line_confidence(ln)) is not None]
        if scores:
            avg_conf = sum(scores) / len(scores)
            self._conf_badge.set_score(avg_conf)
        else:
            self._conf_badge.set_unavailable("无置信度")

        self._viewer.set_image(page.display_image_path)
        # 纵校视图仅供参考，不显示版面标注框（show_blocks 不调用）
        # 字符高亮由 highlight_bbox 单独绘制，避免与块框混淆

        context = self._session.load_reference_context(page)
        flat_text = context.text
        self._updating = True
        self._text_edit.setPlainText(flat_text)
        self._text_edit.setExtraSelections([])
        self._updating = False
        # 保存加载时的会话槽位：line slot 与 block separator slot 都要固定。
        self._set_reference_context(page, flat_text)
        self._status_lbl.setText("")
        self._status_lbl.setStyleSheet("")

    def _flush_dirty_before_reload(self) -> bool:
        if not self._session.pages:
            return False
        return proof_rebuild_gate_for_reference_context(
            dirty=self._text_edit.toPlainText() != self._session.loaded_text,
        ).allow_rebuild

    def _safe_load_page(self, idx: int) -> bool:
        if not self._flush_dirty_before_reload():
            return False
        self._load_page(idx)
        return True

    # ─────────────────── 单字列表点击 ───────────────────────

    def _on_char_clicked(self, item: QListWidgetItem) -> None:
        # 单击 / 当前项变化的路径：保持原"切到这个字"语义；如果用户当前
        # 多选了多个字（itemSelectionChanged 后续会被触发），最终以
        # _on_char_selection_changed 看到的合并集为准。
        tok = item.data(Qt.ItemDataRole.UserRole)
        if not tok:
            return
        self._selected_char = tok
        entries = list(self._char_svc.query(tok))

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
            self._current_selection = None
            self._session.clear_selection()
            self._clear_candidate_buttons()
            self._candidate_box.setToolTip("无候选")
            self._highlight_char_in_text(tok, focus_entry=None)

    # ───────── 跨字批量入口 ─────────

    def _selected_char_tokens(self) -> list[str]:
        """char_list 当前所有被选中的 token。顺序按它们在列表里的展示顺序。"""
        items = self._char_list.selectedItems()
        out: list[str] = []
        for it in items:
            tok = it.data(Qt.ItemDataRole.UserRole)
            if tok and tok not in out:
                out.append(tok)
        return out

    def _selected_tokens_for_restore(self) -> list[str]:
        tokens = self._selected_char_tokens()
        if tokens:
            return tokens
        return [self._selected_char] if self._selected_char else []

    def _restore_gallery_after_context_reload(
        self,
        selected_tokens: list[str],
        occurrence_keys: tuple[tuple[object, ...], ...],
    ) -> None:
        entries: list[CharEntry] = []
        if len(selected_tokens) > 1:
            for tok in selected_tokens:
                entries.extend(self._char_svc.query(tok))
            entries.sort(key=lambda e: (e.page_number, e.char_idx))
            self._selected_char = "".join(selected_tokens)
            joined = " ".join(f'"{t}"' for t in selected_tokens)
            self._gallery_hdr.setText(
                f"跨字索引（{len(selected_tokens)} 字 / 共 {len(entries)} 处）：{joined}"
            )
        elif len(selected_tokens) == 1:
            selected = selected_tokens[0]
            self._selected_char = selected
            entries = list(self._char_svc.query(selected))
            self._gallery_hdr.setText(f'"{selected}"  共 {len(entries)} 处')
        else:
            self._selected_char = ""
            self._gallery_hdr.setText("")
        self._gallery_model.set_entries(entries)
        self._resize_gallery_for_entries(len(entries))
        if not entries:
            self._current_candidate_entry = None
            self._current_selection = None
            self._session.clear_selection()
            self._clear_candidate_buttons()
            return
        if self._restore_gallery_selection_by_occurrence_keys(occurrence_keys):
            return
        first = self._gallery_model.index(0, 0)
        self._gallery_view.setCurrentIndex(first)
        self._sync_gallery_entry(first)

    def _restore_gallery_selection_by_occurrence_keys(
        self,
        occurrence_keys: tuple[tuple[object, ...], ...],
    ) -> bool:
        if not occurrence_keys:
            return False
        wanted = set(occurrence_keys)
        matched_entries: list[CharEntry] = []
        matched_indexes: list[QModelIndex] = []
        for row in range(self._gallery_model.rowCount()):
            idx = self._gallery_model.index(row, 0)
            entry = idx.data(Qt.ItemDataRole.UserRole)
            if entry is None:
                continue
            occurrence = self._occurrence_for_entry(entry)
            if occurrence is None or proof_occurrence_key(occurrence) not in wanted:
                continue
            matched_entries.append(entry)
            matched_indexes.append(idx)
        if not matched_entries:
            self._session.clear_selection()
            return False
        sel = self._gallery_view.selectionModel()
        if sel is not None:
            sel.clearSelection()
            for idx in matched_indexes:
                sel.select(idx, sel.SelectionFlag.Select)
            sel.setCurrentIndex(matched_indexes[0], sel.SelectionFlag.NoUpdate)
        self._sync_gallery_entry(matched_indexes[0])
        self._set_session_selected_entries(matched_entries)
        return True

    def _on_char_selection_changed(self) -> None:
        """char_list 多选变化 → 把所有选中字的 entries 合并塞进 gallery。

        - 0 选：保持原 _selected_char 状态（极少触发；clearSelection 路径）。
        - 1 选：等价于 _on_char_clicked（由它先调用过，这里直接 return）。
        - >=2 选：合并所有选中字的 entries，按 (page_number, char_idx) 排序后
          一次塞进 gallery_model；批量改字现在跨字生效。
        """
        toks = self._selected_char_tokens()
        if len(toks) <= 1:
            return
        merged: list[CharEntry] = []
        for tok in toks:
            merged.extend(self._char_svc.query(tok))
        merged.sort(key=lambda e: (e.page_number, e.char_idx))
        # 用拼接串当 "selected_char" 显示锚（仅用于标题）
        self._selected_char = "".join(toks)
        self._gallery_model.set_entries(merged)
        self._resize_gallery_for_entries(len(merged))
        self._gallery_view.clearSelection()
        if merged:
            first_idx = self._gallery_model.index(0, 0)
            self._gallery_view.scrollTo(
                first_idx, QAbstractItemView.ScrollHint.PositionAtTop
            )
            self._gallery_view.setCurrentIndex(first_idx)
            self._sync_gallery_entry(first_idx)
        joined = " ".join(f'"{t}"' for t in toks)
        self._gallery_hdr.setText(
            f"跨字索引（{len(toks)} 字 / 共 {len(merged)} 处）：{joined}"
        )

    def _select_all_visible_chars(self) -> None:
        """Ctrl+A selects all visible char-list rows.

        受 _filter_char_list（搜索框）影响：只选当前未隐藏的 item。
        """
        self._char_list.blockSignals(True)
        for i in range(self._char_list.count()):
            it = self._char_list.item(i)
            if not it.isHidden():
                it.setSelected(True)
        self._char_list.blockSignals(False)
        # 手动触发一次合并
        self._on_char_selection_changed()

    def _clear_char_list_multi(self) -> None:
        """Esc collapses multi-select back to the current char-list row."""
        cur = self._char_list.currentItem()
        self._char_list.blockSignals(True)
        self._char_list.clearSelection()
        if cur is not None:
            cur.setSelected(True)
        self._char_list.blockSignals(False)
        if cur is not None:
            self._on_char_clicked(cur)

    def _entry_text_pos(self, entry: CharEntry) -> Optional[int]:
        """查找 CharEntry 在当前 reference context 中的起始光标位置。"""
        context = self._session.reference_context
        if context is None:
            return None
        return context.position_for(entry.line, entry.char_idx)

    def _page_block_for_entry(self, entry: CharEntry) -> Optional[Tuple[Page, Block]]:
        return resolve_entry_owner(self._session.pages, entry)

    def _occurrence_for_entry(self, entry: CharEntry) -> Optional[ProofOccurrence]:
        owner = self._page_block_for_entry(entry)
        if owner is None:
            return None
        page, block = owner
        return occurrence_from_entry(entry, page, block)

    def _set_session_selected_entries(self, entries: list[CharEntry]) -> None:
        occurrences = [
            occurrence
            for entry in entries
            if (occurrence := self._occurrence_for_entry(entry)) is not None
        ]
        self._session.set_selected_occurrences(occurrences)

    def _refresh_page_text_view(self, page: Page) -> None:
        context = self._session.load_reference_context(page)
        flat_text = context.text
        self._updating = True
        try:
            self._text_edit.setPlainText(flat_text)
            self._text_edit.setExtraSelections([])
        finally:
            self._updating = False
        self._set_reference_context(page, flat_text)

    def _commit_entry_replacements(
        self,
        new_text: str,
        entries: list[CharEntry],
    ) -> int:
        grouped: dict[
            tuple[int, int, int, str],
            tuple[
                Page,
                Block,
                Line,
                int,
                str,
                list[ProofSpanReplacement],
                list[tuple[object, ...]],
            ],
        ] = {}
        for entry in entries:
            owner = self._page_block_for_entry(entry)
            if owner is None:
                continue
            page, block = owner
            line = entry.line
            line_text = proof_display_text(line)
            occurrence = occurrence_from_entry(entry, page, block)
            occurrence_key = proof_occurrence_key(occurrence)
            span_start = max(0, min(len(line_text), occurrence.span_start))
            span_end = max(span_start, min(len(line_text), occurrence.span_end))
            if span_end == span_start:
                continue
            key = (
                id(page),
                id(block),
                id(line),
                occurrence.line_signature,
            )
            group = grouped.get(key)
            replacement = ProofSpanReplacement(span_start, span_end, new_text)
            if group is None:
                grouped[key] = (
                    page,
                    block,
                    line,
                    entry.line_idx,
                    occurrence.line_signature,
                    [replacement],
                    [occurrence_key],
                )
            else:
                group[5].append(replacement)
                group[6].append(occurrence_key)

        if not grouped:
            return 0

        change = ProofChangeSet()
        affected_pages: list[Page] = []
        published: list[tuple[Page, Block, Line, int]] = []
        history_edits: list[_VProofLineEdit] = []
        applied = 0
        conflict_count = 0
        for page, block, line, line_idx, signature, replacements, occurrence_keys in grouped.values():
            unique: dict[tuple[int, int], ProofSpanReplacement] = {}
            for item in replacements:
                unique[(item.start, item.end)] = item
            before_text = proof_display_text(line)
            before_signature = line_signature(line)
            result = ProofEditService.replace_spans(
                page,
                block,
                line,
                unique.values(),
                expected_signature=signature,
            )
            if result.blocked:
                conflict_count += len(unique)
                continue
            applied += len(unique)
            if result.changed:
                change = change.merge(result.change)
                history_edits.append(self._make_vproof_line_edit(
                    page,
                    block,
                    line,
                    line_idx,
                    occurrence_keys=tuple(dict.fromkeys(occurrence_keys)),
                    before_text=before_text,
                    before_signature=before_signature,
                ))
                if page not in affected_pages:
                    affected_pages.append(page)
                published.append((page, block, line, line_idx))

        if change.changed:
            for page, block, line, line_idx in published:
                self._publish_line_update(
                    page=page,
                    block=block,
                    line=line,
                    line_index=line_idx,
                    status=proof_status(line).value,
                    source="vproof.occurrence_edit",
                )
            self._refresh_char_index_for_pages(affected_pages)
            current_page = self._session.current_page()
            if current_page in affected_pages:
                self._refresh_page_text_view(current_page)
            self._emit_proof_change(change)
            self._push_vproof_edit_action(_VProofEditAction(tuple(history_edits)))
        if conflict_count:
            self._status_lbl.setText(
                f"保存冲突：{conflict_count} 个字位已被外部更新，未覆盖"
            )
            self._status_lbl.setStyleSheet("color: #D32F2F; font-size: 12px;")
        return applied

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
                self._text_edit.setExtraSelections([])
                return
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

    def _highlight_char_in_viewer(self, entry: CharEntry) -> bool:
        cur_page = self._session.current_page()
        if cur_page is None:
            return False
        if entry.page_path == cur_page.display_image_path:
            self._viewer.highlight_bbox(entry.bbox, zoom=True)
            return True
        else:
            for i, p in enumerate(self._session.pages):
                if p.display_image_path == entry.page_path:
                    if self._safe_load_page(i):
                        self._viewer.highlight_bbox(entry.bbox, zoom=True)
                        return True
                    return False
        return False

    def _update_candidate_panel(self, entry: CharEntry) -> None:
        # 极简版：UI 上只摆 ≤5 个候选按钮；不显示分数/来源解释段。
        # 第一候选由 _ranked_candidates 决定（最高可信来源优先）。
        candidates = self._ranked_candidates(entry)[:5]
        self._current_selection = ProofSelection.for_char_entry(entry, source="vproof.gallery")
        candidate_set = CandidateSet.from_values(
            selection=self._current_selection,
            values=candidates,
            source="vproof.ranked",
        )
        self._set_candidate_buttons(candidate_set)
        # proof-layout-collections 第 4 任务：原本在面板底部画一段 “仅 1 候选，
        # 不代表此字正确…原因…” 的解释文本。用户要求拿掉。诊断现在只作为
        # 整个候选面板的 toolTip（鼠标悬停才看到），不侵占任何可见布局。
        if not candidates:
            self._candidate_box.setToolTip("无候选")
            return
        if len(candidates) == 1:
            reason = self._diagnose_single_candidate(entry)
            self._candidate_box.setToolTip(
                f"仅 1 候选：暂无替代建议，不代表此字正确。{reason}"
            )
        else:
            self._candidate_box.setToolTip("")

    def _diagnose_single_candidate(self, entry: CharEntry) -> str:
        """生成"仅 1 候选"的诚实诊断字符串。

        说明候选来源各自为什么没贡献新字（去重后只剩当前字）：
          1. OCR 原文在同位置与当前字相同 / 缺失
          2. DEFAULT_CONFUSABLE_CANDIDATES 字典里 token 无 entry

        本函数 **只读** 不写状态；返回一段短文本拼接到 hint label。
        """
        reasons: List[str] = []
        token = char_entry_display_text(entry)
        ocr_ch = self._line_char_at(proof_ocr_text(entry.line), entry.char_idx)
        if not ocr_ch:
            reasons.append("OCR 无对应字")
        elif ocr_ch == token:
            reasons.append("OCR 与当前字一致")
        if token not in DEFAULT_CONFUSABLE_CANDIDATES:
            reasons.append(f"易混淆字典无 '{token}' 条目")
        if not reasons:
            return "（来源均无新字）"
        return "原因：" + " / ".join(reasons) + "。"

    def _ranked_candidates(self, entry: CharEntry) -> List[str]:
        """按可信度从高到低聚合候选字，去重后返回 list。

        优先级（高 → 低）：
          1. 当前显示字 / token 自身（保留第一位 = 现有识别结果）
          2. OCR 原文在同 char_index 上的字（如果与显示字不同，说明本次有 probe 或后续修正）
          3. DEFAULT_CONFUSABLE_CANDIDATES 易混淆字（兜底）
        """
        ranked: List[str] = []

        def _push(value: str) -> None:
            if value and value not in ranked:
                ranked.append(value)

        _push(char_entry_display_text(entry))
        _push(self._line_char_at(proof_ocr_text(entry.line), entry.char_idx))

        token = char_entry_display_text(entry)
        # 一阶易混淆
        first_level = DEFAULT_CONFUSABLE_CANDIDATES.get(token, [])
        for value in first_level:
            _push(value)
        # Built-in confusable glyphs provide a deterministic fallback list
        # even when OCR suggestions are empty.
        for first in first_level:
            for second in DEFAULT_CONFUSABLE_CANDIDATES.get(first, []):
                _push(second)

        return ranked

    def _clear_candidate_buttons(self) -> None:
        while self._candidate_buttons_row.count():
            item = self._candidate_buttons_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidate_buttons = []

    def _set_candidate_buttons(self, candidate_set: CandidateSet | List[str]) -> None:
        # 极简版：第一候选用 primaryBtn 样式强调，其余 candidateButton。
        # 不再附带来源标签到按钮可见文本上。
        if isinstance(candidate_set, CandidateSet):
            candidates = candidate_set.texts
        else:
            candidates = candidate_set
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
        if applied == 0:
            self._status_lbl.setText("当前页没有可替换的目标")
            self._status_lbl.setStyleSheet("color: #D32F2F; font-size: 12px;")
            return
        if applied > 1:
            self._status_lbl.setText(f"✓ 已批量应用候选到 {applied} 处")
        else:
            self._status_lbl.setText("✓ 已应用候选")
        self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")

    # ─────────────────── 批量改字共享路径 ─────────
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
            current_page = self._session.current_page()
            cur_page_path = current_page.display_image_path if current_page is not None else None
            for idx in selected:
                e = idx.data(Qt.ItemDataRole.UserRole)
                if e is None:
                    continue
                if cur_page_path is not None and not self._entry_belongs_to_current_page(e):
                    skipped_offpage += 1
                    continue
                entries.append(e)
        if not entries and fallback_entry is not None:
            if not self._entry_belongs_to_current_page(fallback_entry):
                return 0
            entries = [fallback_entry]
        if not entries:
            self._session.clear_selection()
            return 0
        self._set_session_selected_entries(entries)
        applied = self._commit_entry_replacements(new_text, entries)
        if skipped_offpage:
            # 在 status 里附带"未改"数量，老实告诉用户
            self._status_lbl.setText(
                f"✓ 已应用到 {applied} 处（跨页 {skipped_offpage} 处未改）"
            )
            self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")
        return applied

    def _refresh_current_char_gallery(self) -> None:
        """原地重建当前 selected_char 的 gallery（保持 _selected_char 不变）。

        被 probe.observation 翻转后调用：当前 gallery 只读 CharIndexService；
        corrected 后重新 query 即可反映最新文本/索引事实。
        """
        tok = self._selected_char
        if not tok:
            return
        try:
            entries = list(self._char_svc.query(tok))
            self._gallery_model.set_entries(entries)
            self._resize_gallery_for_entries(len(entries))
        except Exception:
            pass

    # ─── gallery 直输回调 ───
    def _gallery_direct_overwrite(self, text: str) -> bool:
        """_GalleryListView.keyPressEvent 调用：把单字 ``text`` 覆盖当前 entry。

        普通键盘直输始终只覆盖当前 entry；多选批量替换只通过右键气泡触发，
        避免误触。
        """
        entry = self._current_candidate_entry
        if entry is None:
            return False
        sel = self._gallery_view.selectionModel()
        saved = list(sel.selectedIndexes()) if sel else []
        cur_idx = sel.currentIndex() if sel else QModelIndex()
        if sel:
            sel.clearSelection()
        try:
            applied = self._apply_replacement_to_selected(text, fallback_entry=entry)
        finally:
            if sel:
                for idx in saved:
                    sel.select(idx, sel.SelectionFlag.Select)
                if cur_idx.isValid():
                    sel.setCurrentIndex(cur_idx, sel.SelectionFlag.NoUpdate)
        if applied:
            self._status_lbl.setText(
                f'✓ 直输 P{entry.page_number}·#{entry.char_idx + 1} → "{text}"'
            )
            self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")
            return True
        return False

    def _gallery_direct_blank(self) -> bool:
        """_GalleryListView.keyPressEvent 调用：Backspace/Delete 直接把当前
        槽位填空白（保持长度）。返回 True 表示已处理。"""
        entry = self._current_candidate_entry
        if entry is None:
            return False
        tok = char_entry_display_text(entry)
        applied = self._apply_replacement_to_selected(
            " " * max(1, len(tok)),
            fallback_entry=entry,
        )
        if applied == 0:
            return False
        self._status_lbl.setText(
            f'✓ 直输 P{entry.page_number}·#{entry.char_idx + 1} 已清空为空白'
        )
        self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")
        return True

    def _refresh_current_selection_context(self) -> None:
        """Refresh current text/gallery context after target edits."""
        if not self._session.pages:
            return
        prev_char = self._selected_char
        sel_model = self._gallery_view.selectionModel() if self._gallery_view else None
        prev_row = (
            sel_model.currentIndex().row()
            if sel_model and sel_model.currentIndex().isValid()
            else 0
        )
        self._refresh_reference_context()
        target_char = prev_char
        if target_char:
            for i in range(self._char_list.count()):
                it = self._char_list.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == target_char:
                    self._char_list.blockSignals(True)
                    self._char_list.setCurrentRow(i)
                    self._char_list.blockSignals(False)
                    self._on_char_clicked(it)
                    break
        new_count = self._gallery_model.rowCount()
        if new_count > 0:
            row = max(0, min(prev_row, new_count - 1))
            idx = self._gallery_model.index(row, 0)
            self._gallery_view.setCurrentIndex(idx)
            self._sync_gallery_entry(idx)

    def _lookup_token_at_entry_position(self, entry: "CharEntry") -> Optional[str]:
        """根据 (page_number, block_order, line_idx, char_idx) 在当前 session
        模型里查那个槽位现在是什么字。改字后原字消失时用它找到新字。"""
        if not self._session.pages:
            return None
        owner = self._page_block_for_entry(entry)
        if owner is None:
            return None
        _page, block = owner
        observation_lines = _observation_lines_for_block(block)
        line = entry.line if _line_belongs_to_observation_block(block, entry.line) else None
        if line is None and 0 <= entry.line_idx < len(observation_lines):
            line = observation_lines[entry.line_idx]
        if line is None:
            return None
        txt = proof_display_text(line)
        if 0 <= entry.char_idx < len(txt):
            return txt[entry.char_idx]
        return None

    def _select_all_gallery(self) -> None:
        """Ctrl+A / 按钮全选当前字所有出现。"""
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
        """Esc / 按钮清空多选回到单选。"""
        sel = self._gallery_view.selectionModel()
        cur = sel.currentIndex()
        sel.clearSelection()
        if cur.isValid():
            sel.select(cur, sel.SelectionFlag.Select)
            sel.setCurrentIndex(cur, sel.SelectionFlag.NoUpdate)

    # ─────── modifier + 方向键微调 bbox ────
    def _gallery_items_per_row(self) -> int:
        """估算当前 IconMode 一行可放多少 item。view 太窄时回退到 1。"""
        view = self._gallery_view
        try:
            vw = view.viewport().width()
            gx = view.gridSize().width() or (GALLERY_THUMB + 10)
            n = max(1, vw // max(1, gx))
            return int(n)
        except Exception:
            return 1

    def _step_gallery_singleton(self, delta: int) -> None:
        """Ctrl+Alt+←/→：按 entry 编号 ±1 单选切换。"""
        count = self._gallery_model.rowCount()
        if count <= 0:
            return
        sel = self._gallery_view.selectionModel()
        cur = sel.currentIndex()
        row = cur.row() if cur.isValid() else 0
        new_row = max(0, min(count - 1, row + delta))
        new_idx = self._gallery_model.index(new_row, 0)
        sel.select(new_idx, sel.SelectionFlag.ClearAndSelect)
        sel.setCurrentIndex(new_idx, sel.SelectionFlag.Current)
        self._sync_gallery_entry(new_idx)

    def _extend_gallery_selection(self, delta: int) -> None:
        """Shift+Alt+←/→：不动 current，把相邻 ±1 加入 selection。"""
        count = self._gallery_model.rowCount()
        if count <= 0:
            return
        sel = self._gallery_view.selectionModel()
        cur = sel.currentIndex()
        if not cur.isValid():
            return
        target_row = cur.row() + delta
        if not (0 <= target_row < count):
            return
        target = self._gallery_model.index(target_row, 0)
        # 若目标已选中 → 收回；否则添加。
        if sel.isSelected(target):
            sel.select(target, sel.SelectionFlag.Deselect)
        else:
            sel.select(target, sel.SelectionFlag.Select)

    def _step_gallery_row(self, delta: int) -> None:
        """Alt+↑/↓：跨 wrap 行步进。"""
        per_row = self._gallery_items_per_row()
        self._step_gallery_singleton(delta * per_row)

    def _on_gallery_selection_changed(self, *_args) -> None:
        """选区数量变化 → 刷新标题里的"已选 N 个"。"""
        sel = self._gallery_view.selectionModel() if hasattr(self, "_gallery_view") else None
        n = len(sel.selectedIndexes()) if sel else 0
        cur = sel.currentIndex() if sel else None
        if cur is not None and cur.isValid():
            self._refresh_gallery_header(cur, n)

    def _refresh_gallery_header(self, index: QModelIndex, sel_count: int) -> None:
        entry = index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None
        if entry is None or not self._selected_char:
            return
        tokens = self._selected_char_tokens()
        if len(tokens) > 1:
            extra = f"  ·  已选 {sel_count} 个" if sel_count > 1 else ""
            joined = " ".join(f'"{t}"' for t in tokens)
            self._gallery_hdr.setText(
                f"跨字索引（{len(tokens)} 字 / 共 {self._gallery_model.rowCount()} 处）"
                f"{extra}：[第 {entry.page_number} 页 / 第 {entry.char_idx + 1} 位] {joined}"
            )
            return
        # 直接读 gallery_model 的 rowCount，避免标题和当前筛选结果分叉。
        total = self._gallery_model.rowCount()
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
        block_info = block_display_label(block) if block else "unknown"
        note = (block.note or "").split("|", 1)[0].strip() if block and block.note else ""
        parts = [
            f"bbox={entry.bbox_source}/{entry.bbox_granularity}",
            f"collection={entry.collection_kind}",
            f"conf={self._format_entry_confidence(entry)}",
            f"layout={block_info}",
        ]
        if note:
            parts.append(f"vl_note={note[:24]}")
        return "；".join(parts)

    def _confidence_for_entry(self, entry: CharEntry) -> Optional[float]:
        score = normalize_confidence(getattr(entry, "confidence", None))
        if score is not None:
            return score
        return line_confidence(entry.line)

    def _format_entry_confidence(self, entry: CharEntry) -> str:
        score = self._confidence_for_entry(entry)
        return "缺失" if score is None else f"{score:.2f}"

    def _block_for_entry(self, entry: CharEntry) -> Optional[Block]:
        owner = self._page_block_for_entry(entry)
        if owner is None:
            return None
        return owner[1]

    def _layout_block_index(self, page: Page, block: Block) -> int:
        for view in iter_page_layout_block_views(page):
            if view.runtime_block is block or view.uid == block.uid:
                return view.snapshot_index
        return -1

    def _layout_block_by_uid(self, page: Page, block_uid: str) -> Optional[Block]:
        for view in iter_page_layout_block_views(page):
            if view.uid == block_uid and view.runtime_block is not None:
                return view.runtime_block
        return None

    def _layout_block_by_id(self, page: Page, block_id: int) -> Optional[Block]:
        for view in iter_page_layout_block_views(page):
            block = view.runtime_block
            if block is not None and block.id == block_id:
                return block
        return None

    def _layout_block_at_index(self, page: Page, block_index: int) -> Optional[Block]:
        if block_index < 0:
            return None
        for view in iter_page_layout_block_views(page):
            if view.snapshot_index == block_index:
                return view.runtime_block
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
        if not self._highlight_char_in_viewer(entry):  # 可能触发翻页
            self._current_candidate_entry = None
            self._current_selection = None
            self._session.clear_selection()
            self._clear_candidate_buttons()
            self._candidate_box.setToolTip("")
            self._text_edit.setExtraSelections([])
            self._status_lbl.setText("切页失败：当前索引无法定位到页面")
            self._status_lbl.setStyleSheet("color: #D32F2F; font-size: 12px;")
            return
        self._current_candidate_entry = entry
        self._set_session_selected_entries([entry])
        self._update_candidate_panel(entry)
        # 更新标题：让用户清楚当前看的是哪页
        if self._selected_char:
            sel_count = len(self._gallery_view.selectionModel().selectedIndexes())
            self._refresh_gallery_header(index, sel_count)
            # 传入 entry 以精准定位到该出现，而非首次出现
            self._highlight_char_in_text(char_entry_display_text(entry), focus_entry=entry)
            side_inputs = (
                getattr(self, "_edit_bubble_input", None),
                self._text_edit,
            )
            if not any(w is not None and w.hasFocus() for w in side_inputs):
                self._gallery_view.setFocus()

    # ─────────────────── 原图 block 点击 ────────────────────

    def _on_block_clicked(self, block: Block) -> None:
        context = self._session.reference_context
        if context is None:
            return
        for slot in context.slots:
            if _line_belongs_to_observation_block(block, slot.line):
                cursor = self._text_edit.textCursor()
                cursor.setPosition(slot.start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.ensureCursorVisible()
                break

    # ─────────────────── 持久化 / 刷新 ───────────────────────

    def _emit_proof_change(self, change: ProofChangeSet) -> None:
        if not change.needs_persist:
            return
        self.proof_changed.emit(change)

    def _refresh_reference_context(self) -> bool:
        page = self._session.current_page()
        if page is None:
            return False
        change = ProofChangeSet()
        try:
            detected = qp.detect_corrections(qp.get_active_store(), self._session.pages)
        except Exception:
            detected = 0
        if detected > 0:
            change = change.merge(ProofChangeSet(probe_changed=True, index_changed=True))
        self._refresh_char_index_for_pages([page])
        self._refresh_page_text_view(page)
        if change.needs_persist:
            self._emit_proof_change(change)
            self._status_lbl.setText("✓ 已刷新")
            self._status_lbl.setStyleSheet("color: #4CAF50; font-size: 12px;")
            return True
        self._status_lbl.setText("已刷新")
        self._status_lbl.setStyleSheet("")
        return False

    def _publish_line_update(
        self,
        *,
        page: Page,
        block: Block,
        line: Line,
        line_index: int,
        status: str,
        source: str,
    ) -> None:
        request = ProofUpdateRequest(
            page_id=page.id,
            line_id=line.id,
            status=status,
            page_uid=page.uid,
            line_uid=line.uid,
            origin=id(self),
            selection=ProofSelection.for_line(
                page=page, block=block, line=line, line_index=line_index, source=source,
            ),
            source=source,
        )
        self._bus.publish_line_update(request)

    def _external_pages_for_request(self, request: ProofUpdateRequest) -> list[Page]:
        matched: list[Page] = []
        for page in self._session.pages:
            if not proof_request_matches_page(request, page):
                continue
            for _block, line, _li in iter_unique_page_text_lines(page):
                if proof_request_matches_line(request, line):
                    matched.append(page)
                    break
        return matched

    def _on_external_line_changed(self, request: ProofUpdateRequest) -> None:
        """收到外部（横校）发来的 line.proof_changed。

        ProofStateBus 是同步单线程 dispatch。横校批量保存会发布多条
        ``line.proof_changed``，纵校必须把这些事件折叠到一次当前页重建，
        避免主线程反复重建文本、viewer、字符索引和 gallery。
        """
        if not isinstance(request, ProofUpdateRequest):
            return
        if request.origin == id(self):
            return
        if not self._session.pages:
            return
        line_uid = request.line_uid or None
        if line_uid is None and request.line_id is None:
            return
        affected_pages = self._external_pages_for_request(request)
        if not affected_pages:
            return
        line_key = line_uid if line_uid is not None else request.line_id
        self._session.queue_external_refresh(
            line_key=line_key,
            page_keys=[self._char_index_page_key(page) for page in affected_pages],
        )
        self._external_refresh_timer.start()  # 80ms 内的 N 次 publish 合并成 1 次

    def _do_external_refresh(self) -> None:
        """Debounced external refresh for current-page proof context.

        从 _on_external_line_changed 累积的 pending 事件里只触发一次重建。
        与原先的同步路径相比，重建本身的代价没变，但 N→1 折叠掉了重复。
        """
        plan = self._session.consume_external_refresh_plan()
        if not plan.has_work:
            return
        current_page = self._session.current_page()
        if current_page is None:
            return
        pending_page_keys = set(plan.affected_page_keys)
        affected_pages = [
            page for page in self._session.pages
            if self._char_index_page_key(page) in pending_page_keys
        ]
        if affected_pages:
            self._refresh_char_index_for_pages(affected_pages)
        if not plan.reload_current_page:
            return
        gate = proof_rebuild_gate_for_reference_context(
            dirty=self._text_edit.toPlainText() != self._session.loaded_text,
        )
        if not gate.allow_rebuild:
            return
        selected_tokens = self._selected_tokens_for_restore()
        selected_occurrence_keys = tuple(self._session.selected_occurrence_keys)
        self._clear_vproof_history()
        self._load_page(self._session.current_page_index())
        self._restore_gallery_after_context_reload(selected_tokens, selected_occurrence_keys)

    def refresh_quality_probe_state(self) -> None:
        """供 main_window 进入纵校步骤时调用：当前页若已加载，重新渲染让显示
        空间文本与全局 active store 对齐。"""
        if not self._session.pages:
            return
        # 重新触发 _load_page，让 reference context 用最新的 active store 渲染
        self._safe_load_page(self._session.current_page_index())

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        super().keyPressEvent(event)
