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
    QAbstractListModel, QModelIndex, QSize, Qt, QTimer, Signal,
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

from app.core.block_attributes import block_display_label
from app.models import BBox, Block, Line, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.core.proof_line_utils import iter_unique_page_text_lines
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
from app.services.char_index_service import CharEntry, CharIndexService
from app.services.proof_probe_text_service import (
    displayed_text as _proof_displayed_text,
    save_displayed_edit as _proof_save_displayed_edit,
    resolve_block_line_index as _proof_resolve_block_line_index,
)
from app.services.proof_image_service import (
    verified_char_crop as _shared_verified_char_crop,
)
from app.ui.proof.confidence_utils import line_confidence, normalize_confidence
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

CHAR_LIST_THUMB = 18
# vproof-ime-persist-visibility round 12 任务 1：相同字索引"挤到看不见"，再
# 抬尺寸：thumb 34→56（CJK 56px 才真正易读），每行 8→6，最多 4→3 行。这把
# 单 cell 像素面积 ×2.6，行间距由 +14 提到 +18，外框 +50。
GALLERY_THUMB   = 56
GALLERY_ITEMS_PER_ROW = 6
GALLERY_MAX_ROWS = 3
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
            # vproof-gallery-rendering round 14：源 pix 必须 >= 实际绘制区域，
            # 否则 delegate.paint 会从 56 → 66 上采样，糊成「图被放大」的观感。
            # 用 2× 像素密度裁图，让 delegate 永远做 down-scale，CJK 笔画
            # 清晰、不糊。
            return _verified_char_crop(
                self._cache, entry.page_path, entry.bbox,
                _GalleryDelegate.SOURCE_PX,
            )
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
    # vproof-ime-persist-visibility round 12：+10 → +14，让 56px 缩略图周围
    # 留出 7px 内圈呼吸空间；总 cell 70×70，CJK 字在里面是真易读了。
    SIZE = GALLERY_THUMB + 14
    # vproof-gallery-rendering round 14：源 pixmap 像素密度 = 2× cell size。
    # 之前 model 用 GALLERY_THUMB(56) 取图，delegate 再 scale 到 66×66，
    # 是 up-scale，CJK 笔画被插值放粗、看上去「图被放大」。
    # 现在源 132×132，delegate 始终 down-scale → 清晰。
    SOURCE_PX = (GALLERY_THUMB + 14) * 2

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        r = option.rect
        pix: Optional[QPixmap] = index.data(Qt.ItemDataRole.DecorationRole)
        # vproof-gallery-rendering round 14：白底打底，避免 list view 默认
        # 背景透出造成相邻 cell 看上去「粘连/挤」。
        painter.fillRect(r, QColor("#ffffff"))
        img_r = r.adjusted(2, 2, -2, -2)
        if pix and not pix.isNull():
            scaled = pix.scaled(
                img_r.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # vproof-gallery-rendering round 14：同时居中 X 和 Y。
            # 之前只算 dx，drawPixmap 用 img_r.y() 顶对齐，导致瘦长/扁宽
            # CJK 字（一、丨、丁）贴在格子顶部，下方留大白边 → 视觉「压字」。
            dx = (img_r.width() - scaled.width()) // 2
            dy = (img_r.height() - scaled.height()) // 2
            painter.drawPixmap(img_r.x() + dx, img_r.y() + dy, scaled)
            # vproof-gallery-rendering round 14：cell 之间画 1px 浅灰分隔，
            # 让相邻字不再视觉粘在一起；选中边框照旧覆盖此线。
            painter.setPen(QColor("#e0e4ea"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r.adjusted(0, 0, -1, -1))
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


def _line_block_index(
    pages: List[Page], page_number: int, line: Line
) -> Optional[int]:
    """在 ``pages`` 中找到 ``line`` 所属 block 的 index（quality_probe key 用）。"""
    for page in pages:
        if page.page_number != page_number:
            continue
        for bi, block in enumerate(page.blocks):
            if line in block.lines:
                return bi
    return None


def _vproof_displayed_text(page: Page, block: Block, line: Line) -> str:
    return _proof_displayed_text(line, page, block)


def _vproof_save_displayed_line(
    page: Page, block: Block, line: Line, displayed_new_text: str
) -> bool:
    return _proof_save_displayed_edit(line, page, block, displayed_new_text)


# ─────────────────────────────────────────────────────────────
# proof-bbox-boxedit (round 9)：槽位心智的 OCR 文本编辑器
# ─────────────────────────────────────────────────────────────


class _SlotAwareTextEdit(QPlainTextEdit):
    """OCR 文本框的轻包装：把 Backspace / Delete / Cut 改成"槽位填空白"。

    设计原则：
    - OCR 文本只是参照；正常工作流应该走右侧 gallery 槽位编辑器。
    - 但用户偶尔在文本框里直接删字时，"位置不能因为删字而被吞掉"，
      所以对应到任何"行内字符槽位（_text_map 中有记录的位置）"的删除请求，
      都改成用 ASCII 空格 (U+0020) 填这个槽，槽边界不动。
    - 跨行换行符 / block 间分隔符 不在 _text_map 内，仍允许真删除。

    通过 ``set_slot_owner`` 注入 VProofPanel 作回调宿主，避免循环 import。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._slot_owner: Optional["VProofPanel"] = None

    def set_slot_owner(self, owner: "VProofPanel") -> None:
        self._slot_owner = owner

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        owner = self._slot_owner
        if owner is None:
            super().keyPressEvent(event)
            return
        key = event.key()
        mods = event.modifiers()
        # 只在"裸 Backspace / Delete"时拦截；Ctrl+Backspace（删词）保留原生行为。
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        cursor = self.textCursor()
        if key == Qt.Key.Key_Backspace and not ctrl:
            if owner._slot_aware_backspace(cursor):
                return
        elif key == Qt.Key.Key_Delete and not ctrl:
            if owner._slot_aware_delete_forward(cursor):
                return
        # Ctrl+X：剪切覆盖到槽位 → 改填空白
        elif key == Qt.Key.Key_X and ctrl and cursor.hasSelection():
            if owner._slot_aware_fill_selection_with_blanks(cursor):
                # 仍把内容拷到剪贴板
                from PySide6.QtGui import QGuiApplication
                QGuiApplication.clipboard().setText(cursor.selectedText())
                return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────
# 纵校面板
# ─────────────────────────────────────────────────────────────


class _GalleryListView(QListView):
    """vproof-direct-overwrite-residual：相同字索引 gallery 的 QListView 子类。

    在 keyPressEvent 里拦截"光标焦点在 gallery 时的直接输入"：
      - 单字符可打印键（无 Ctrl/Alt/Meta）→ 覆盖当前 entry。
      - Backspace / Delete（无修饰键）→ 把当前 entry 填成空白槽。
    其它键全部 super()，让 Qt 原生 ExtendedSelection / Ctrl+A / Esc /
    Ctrl+Alt+方向键等都正常工作。

    这是 round 11 的核心修复：上一轮 round 10 把直输逻辑挂在 panel 的
    eventFilter 上，但 `_sync_gallery_entry` 末尾会 `_slot_edit_input.setFocus()`，
    导致用户选完 entry 之后键盘焦点根本不在 gallery_view，eventFilter
    永远不会被触发。本轮把焦点保留在 gallery 上，并且把直输逻辑直接做
    成 view 子类的 override，焦点流是真的。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gallery_owner: Optional["VProofPanel"] = None
        # vproof-ime-persist-visibility round 12 任务 2：QListView 默认不接 IME，
        # 中文输入法不会把 commit 字符串发到这里。打开 WA_InputMethodEnabled，
        # 并通过 inputMethodQuery 声明 ImEnabled=True，让 IME 真正把 commit
        # string 通过 inputMethodEvent 投到本控件。
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def set_gallery_owner(self, owner: "VProofPanel") -> None:
        self._gallery_owner = owner

    def inputMethodQuery(self, query):  # type: ignore[override]
        # 必须告诉 IME 自己 enabled，否则 inputMethodEvent 不会派过来
        if query == Qt.InputMethodQuery.ImEnabled:
            return True
        return super().inputMethodQuery(query)

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        """vproof-ime-persist-visibility round 12 任务 2：真实中文 IME 闭环。

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
        # 任何 Ctrl/Alt/Meta（含 Ctrl+A 全选、Ctrl+Alt+→ 步进、Alt+↑ 跨行）一律放行
        disallowed = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if mods & disallowed:
            super().keyPressEvent(event)
            return
        key = event.key()
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
        self._current_selection: Optional[ProofSelection] = None
        self._current_candidate_entry: Optional[CharEntry] = None
        self._candidate_buttons: List[QPushButton] = []
        self._candidate_provider: Optional[LlmCandidateProvider] = None
        self._updating = False
        # Phase 22 blocker 1：_load_page 后存基线文本，用于 _on_external_line_changed
        # 判断"_text_edit 是否有未保存输入"，避免外部同步覆盖用户在编辑的内容。
        self._loaded_text: str = ""
        # vproof-direct-overwrite-residual round 11 任务 2：外部事件 → 重渲染
        # 通过 QTimer 单次延后合并。详见 _on_external_line_changed 的注释。
        self._external_refresh_timer = QTimer(self)
        self._external_refresh_timer.setSingleShot(True)
        self._external_refresh_timer.setInterval(80)
        self._external_refresh_timer.timeout.connect(self._do_external_refresh)
        self._pending_external_lines: set[int | str] = set()
        # vproof-ime-persist-visibility round 12 任务 3+4：本地直输/槽位编辑
        # 之后需要把 _text_edit 落盘到 line.final_text 并重建 _char_svc，
        # 否则切走再回来 / 字索引计数都是旧的。debounce 120ms 合并连续按键。
        self._local_commit_timer = QTimer(self)
        self._local_commit_timer.setSingleShot(True)
        self._local_commit_timer.setInterval(120)
        self._local_commit_timer.timeout.connect(self._commit_local_edits_and_refresh)
        self._build_ui()
        # H/V 校对联动：订阅其他 panel 的编辑事件，本 panel 自己 publish 的事件
        # 通过 origin == id(self) 过滤掉以避免回路。
        # Phase 18 blocker 3：保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            TOPIC_LINE_PROOF_CHANGED, self._on_external_line_changed,
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
        # Round 17：移除"确认本页"按钮。
        # 纵校以"字"为单位推进（gallery → 槽位逐字），整页一键 OK 的语义不属于
        # 纵校；行级 OK 标记由横校承担（HProof 行右上角已有"✓"），这里不再重复。

        for btn in (self._btn_prev_page, self._btn_next_page, self._btn_save):
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

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_page_text)
        QShortcut(QKeySequence("PageUp"), self, activated=self._prev_page)
        QShortcut(QKeySequence("PageDown"), self, activated=self._next_page)
        # proof-interaction-slots 第 6 任务：点击切图 → 焦点走 _text_edit。
        # 为了不把用户“困”在文本光标里，增加 Alt+← / Alt+→ 在 gallery 里切
        # 到上/下一个同字出现。这些快捷键不抢占普通方向键。
        QShortcut(QKeySequence("Alt+Right"), self, activated=self._go_next_gallery)
        QShortcut(QKeySequence("Alt+Left"), self, activated=self._go_prev_gallery)
        # proof-crosschar-batch 第 2 任务：真正全局的"字索引导航"，无论焦点
        # 在 text_edit / batch_input / gallery 任何地方都能切到上/下一个字。
        QShortcut(QKeySequence("Ctrl+."), self, activated=self._step_char_list_next)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self._step_char_list_prev)
        # Round 17：去掉 Ctrl+Return / Ctrl+Enter 的"整页确认"绑定 ——
        # 纵校无"页级 OK"语义。

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
        # proof-crosschar-batch 第 1 任务：允许多选不同的字符。选中 N 项
        # 后 gallery 会合并呈现这 N 个字的所有出现，批量改字可以跨字生效。
        self._char_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._char_list.setStyleSheet("QListWidget { background:#ffffff; } QListWidget::item { background:#ffffff; }")
        self._char_list.itemClicked.connect(self._on_char_clicked)
        # proof-crosschar-batch 第 1 任务：多选/取消都走这个路径
        self._char_list.itemSelectionChanged.connect(self._on_char_selection_changed)
        self._char_list.currentItemChanged.connect(
            lambda cur, _prev: self._on_char_clicked(cur) if cur else None
        )
        # proof-crosschar-batch 第 2 任务：键盘协同
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
        # vproof-window-balance round 13：窗口本身平衡，不是再放大缩略图。
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
        self._content_split.setSizes([proof_min_w + 40, 720])
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

        # proof-bbox-boxedit round 9 第 1 任务：让"靠近 gallery 的编辑入口"
        # 真存在 —— 在 batch_row 上再加一条"单槽位编辑器"：
        #   [当前槽位: P? idx? "x"]  [改为: ____ Enter]  [清空(填空白)]
        # 用户注意力不离开 gallery 就能改/清空当前选中的那个槽。
        slot_row = QHBoxLayout()
        slot_row.setSpacing(4)
        self._slot_info_lbl = QLabel("当前槽位：—")
        self._slot_info_lbl.setStyleSheet("color:#555;font-size:11px;")
        self._slot_edit_input = QLineEdit()
        self._slot_edit_input.setPlaceholderText("改当前槽位为…（Enter 应用）")
        self._slot_edit_input.setMaxLength(8)
        self._slot_edit_input.setClearButtonEnabled(True)
        self._slot_edit_input.returnPressed.connect(self._apply_slot_edit_input)
        self._slot_apply_btn = QPushButton("应用")
        self._slot_apply_btn.clicked.connect(self._apply_slot_edit_input)
        self._slot_blank_btn = QPushButton("清空(填空白)")
        self._slot_blank_btn.setToolTip("把当前槽位填成 ASCII 空格，保留槽位边界")
        self._slot_blank_btn.clicked.connect(self._apply_slot_blank_to_current)
        slot_row.addWidget(self._slot_info_lbl)
        slot_row.addWidget(self._slot_edit_input, 1)
        slot_row.addWidget(self._slot_apply_btn)
        slot_row.addWidget(self._slot_blank_btn)
        layout.addLayout(slot_row)

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
        # Round 18：移除"标记已识别"无损路径。新机制以"文本即锚点"——
        # displayed_text 把 fake_char 注入到 OCR 文本窗口；用户在文本窗口里
        # 真正改字才算 corrected（save_displayed_edit 反向写回 final_text 时
        # 自动报点；_apply_replacement_to_selected 等批改路径走 detect_corrections）。
        batch_row.addWidget(self._batch_input, 1)
        batch_row.addWidget(self._batch_btn)
        batch_row.addWidget(self._batch_select_all_btn)
        batch_row.addWidget(self._batch_clear_btn)
        layout.addLayout(batch_row)

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
        # round 12 任务 2：QAbstractItemView 构造时会把 WA_InputMethodEnabled
        # 设回 false，需要在所有 setXxx 之后再补一次，确保 IME 真的能投到这。
        self._gallery_view.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
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
        # proof-bbox-boxedit round 9 第 3 任务：modifier + 方向键真协同。
        # 这些都是 widget-scoped，不抢全局；Qt 原生方向键在 gallery 上已经
        # 移动 currentIndex，但下面这些是"真新增"的语义：
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
        # vproof-gallery-rendering follow-up：之前只给整个 gallery_box 预留 +50，
        # 但 box 里除了 QListView 还有标题行、单槽位编辑行、批量编辑行和
        # layout 的 spacing/margins。真实 chrome 高度接近 100px，导致
        # QListView 可用高度被吃掉，1 行时 item 会被纵向裁切，用户看到的仍是
        # “压字”。这里按真实 chrome 预算给固定高度。
        rows = min(
            GALLERY_MAX_ROWS,
            max(1, (max(1, count) + GALLERY_ITEMS_PER_ROW - 1) // GALLERY_ITEMS_PER_ROW),
        )
        item_h = GALLERY_THUMB + 18
        chrome_h = 110
        view_h = rows * item_h + 8
        self._gallery_box.setFixedHeight(view_h + chrome_h)

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

    # proof-crosschar-batch 第 2 任务：全局 Ctrl+, / Ctrl+. 切字
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
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        # proof-bbox-boxedit round 9 第 6 任务：明示"只是参照、不要光标心智"。
        hint = QLabel("OCR 文本（仅参照；编辑请用 gallery 旁的槽位输入或方向键导航。"
                      "在此删字会自动填空白以保留槽位）")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;font-size:11px;")
        layout.addWidget(hint)

        # proof-bbox-boxedit round 9 第 2 任务：用 _SlotAwareTextEdit
        # 让 Backspace / Delete / Ctrl+X 对槽位填空白而不是真删除。
        self._text_edit = _SlotAwareTextEdit()
        self._text_edit.set_slot_owner(self)
        self._text_edit.setStyleSheet("font-size:16px; padding:8px; background:#fafafa;")
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
        selected_tokens = self._selected_char_tokens()
        self._pages = pages
        self._current_page_idx = self._find_page_index(current_page)
        if self._pages:
            self._rebuild_text_lookup(self._pages[self._current_page_idx])
        self._char_svc.build(pages)
        self._rebuild_char_list()
        if len(selected_tokens) > 1:
            merged: list[CharEntry] = []
            for tok in selected_tokens:
                merged.extend(self._char_svc.query(tok))
            merged.extend(self._extras_for_tokens(selected_tokens))
            merged.sort(key=lambda e: (e.page_number, e.char_idx))
            self._selected_char = "".join(selected_tokens)
            self._gallery_model.set_entries(merged)
            self._resize_gallery_for_entries(len(merged))
            joined = " ".join(f'"{t}"' for t in selected_tokens)
            self._gallery_hdr.setText(
                f"跨字索引（{len(selected_tokens)} 字 / 共 {len(merged)} 处）：{joined}"
            )
        elif len(selected_tokens) == 1:
            selected = selected_tokens[0]
            self._selected_char = selected
            entries = list(self._char_svc.query(selected))
            entries.extend(self._extras_for_tokens([selected]))
            self._gallery_model.set_entries(entries)
            self._resize_gallery_for_entries(len(entries))
            self._gallery_hdr.setText(f'"{selected}"  共 {len(entries)} 处')
        else:
            self._resize_gallery_for_entries(0)
        self._page_label.setText(f"页 {self._current_page_idx + 1} / {len(self._pages)}")

    # ───── quality_probe 装饰者：把探针补成顺手在同字 gallery 上多出现 ─────
    def _extras_for_tokens(self, tokens: list[str]) -> list[CharEntry]:
        """Round 18：废弃。

        新机制把 fake_char 注入到 OCR 文本窗口的 displayed_text；final_text
        始终持有 true_char，所以 ``_char_svc.query(true_char)`` 已经包含
        probe 位置，gallery 不再需要任何 extras。保留方法签名仅为不动调用方。
        """
        return []

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
        self._current_selection = None
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
        scores = [score for ln in lines if (score := line_confidence(ln)) is not None]
        if scores:
            avg_conf = sum(scores) / len(scores)
            self._conf_badge.set_score(avg_conf)
        else:
            self._conf_badge.set_unavailable("无置信度")

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
        # 单击 / 当前项变化的路径：保持原"切到这个字"语义；如果用户当前
        # 多选了多个字（itemSelectionChanged 后续会被触发），最终以
        # _on_char_selection_changed 看到的合并集为准。
        tok = item.data(Qt.ItemDataRole.UserRole)
        if not tok:
            return
        self._selected_char = tok
        entries = list(self._char_svc.query(tok))
        entries.extend(self._extras_for_tokens([tok]))

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
            self._clear_candidate_buttons()
            self._candidate_box.setToolTip("无候选")
            self._highlight_char_in_text(tok, focus_entry=None)

    # ───────── proof-crosschar-batch：跨字 batch 入口 ─────────

    def _selected_char_tokens(self) -> list[str]:
        """char_list 当前所有被选中的 token。顺序按它们在列表里的展示顺序。"""
        items = self._char_list.selectedItems()
        out: list[str] = []
        for it in items:
            tok = it.data(Qt.ItemDataRole.UserRole)
            if tok and tok not in out:
                out.append(tok)
        return out

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
        merged.extend(self._extras_for_tokens(toks))
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
        """proof-crosschar-batch 第 2 任务：Ctrl+A 选中 char_list 全部可见项。

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
        """proof-crosschar-batch 第 2 任务：Esc 收回到单选（当前项）。"""
        cur = self._char_list.currentItem()
        self._char_list.blockSignals(True)
        self._char_list.clearSelection()
        if cur is not None:
            cur.setSelected(True)
        self._char_list.blockSignals(False)
        if cur is not None:
            self._on_char_clicked(cur)

    def _rebuild_text_lookup(self, page: Page) -> None:
        _flat_text, self._text_map = _build_text_map(page)
        self._entry_pos_by_key = {
            (id(line), ci): start
            for line, ci, start, _end in self._text_map
        }

    def _entry_text_pos(self, entry: CharEntry) -> Optional[int]:
        """查找 CharEntry 在当前 _text_map 中的起始光标位置。"""
        return self._entry_pos_by_key.get((id(entry.line), entry.char_idx))

    # ───── proof-bbox-boxedit round 9：槽位心智 —— 删除不丢位 ─────

    def _slot_at_pos(self, pos: int) -> Optional[Tuple[Line, int, int, int]]:
        """位置 ``pos`` 是否落在某个 _text_map 槽位内。

        _text_map 每条 entry 都是 (line, char_idx, start, start+1) 单字范围。
        线性扫描；页内字符数有限（典型 < 几千），开销忽略。
        """
        for entry in self._text_map:
            _line, _ci, start, end = entry
            if start <= pos < end:
                return entry
        return None

    def _fill_slot_with_blank(self, start: int, end: int) -> None:
        """把文本框 [start, end) 区间替换成等长 ASCII 空格。

        长度严格保持，所以 _text_map 索引不会失效。
        """
        cur = QTextCursor(self._text_edit.document())
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cur.insertText(" " * (end - start))

    def _slot_aware_backspace(self, cursor: QTextCursor) -> bool:
        """Backspace 拦截：若覆盖到槽位，改成填空白。

        - 有选区：选区内每个槽位都填空白，选区两端的非槽字符（换行/分隔）
          被保留——因为它们不在 _text_map 内，本来也不能 backspace 一次性删。
        - 无选区：看 cursor.position-1 是不是某个槽位的字符，是则填空白。
        返回 True 表示已消化按键、调用者不要再走原生。
        """
        if cursor.hasSelection():
            return self._slot_aware_fill_selection_with_blanks(cursor)
        pos = cursor.position()
        if pos <= 0:
            return False
        target = self._slot_at_pos(pos - 1)
        if target is None:
            return False
        _line, _ci, start, end = target
        self._fill_slot_with_blank(start, end)
        new_cur = self._text_edit.textCursor()
        new_cur.setPosition(start)  # 光标落到槽位起点：保持位置感
        self._text_edit.setTextCursor(new_cur)
        return True

    def _slot_aware_delete_forward(self, cursor: QTextCursor) -> bool:
        """Delete 拦截：与 backspace 对称，但定位 cursor.position 当前槽。"""
        if cursor.hasSelection():
            return self._slot_aware_fill_selection_with_blanks(cursor)
        pos = cursor.position()
        target = self._slot_at_pos(pos)
        if target is None:
            return False
        _line, _ci, start, end = target
        self._fill_slot_with_blank(start, end)
        new_cur = self._text_edit.textCursor()
        new_cur.setPosition(end)  # 跳过被填空白的槽
        self._text_edit.setTextCursor(new_cur)
        return True

    def _slot_aware_fill_selection_with_blanks(self, cursor: QTextCursor) -> bool:
        """选区跨多个字符 / 槽位时：每个被覆盖到的槽位都填空白。"""
        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()
        if sel_end <= sel_start:
            return False
        targets = [
            e for e in self._text_map
            if not (e[3] <= sel_start or e[2] >= sel_end)
        ]
        if not targets:
            return False
        # 从后向前替换，避免前面替换改变后面 offset（虽然等长，但稳一点）
        targets.sort(key=lambda e: e[2], reverse=True)
        for _line, _ci, start, end in targets:
            s = max(start, sel_start)
            e = min(end, sel_end)
            if e > s:
                self._fill_slot_with_blank(s, e)
        final = self._text_edit.textCursor()
        final.setPosition(sel_start)
        self._text_edit.setTextCursor(final)
        return True

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
        line_text = entry.line.display_text
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

    def _clear_candidate_buttons(self) -> None:
        while self._candidate_buttons_row.count():
            item = self._candidate_buttons_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidate_buttons = []

    def _set_candidate_buttons(self, candidate_set: CandidateSet | List[str]) -> None:
        # 极简版：第一候选用 primaryBtn 样式强调，其余 candidateButton。
        # 不再附带 LLM/来源标签到按钮可见文本上。
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
        # proof-crosschar-batch 第 1 任务：跨字 batch 时不同 entry 有不同
        # token 长度，必须按 entry 自己的长度算 selection，不能再用 anchor.token_len
        # 一刀切，否则会把多字 token 替换成多字时把后面字也吃掉 / 或留下尾巴。
        doc = self._text_edit.document()
        pos_and_len: list[tuple[int, int]] = []
        for e in entries:
            pos = self._entry_text_pos(e)
            if pos is None:
                continue
            tok = e.token_text or e.char or ""
            pos_and_len.append((pos, max(1, len(tok))))
        # 从后往前改，避免前面替换改变后面的 pos
        pos_and_len.sort(key=lambda x: x[0], reverse=True)
        applied = 0
        for pos, tlen in pos_and_len:
            cur = QTextCursor(doc)
            cur.setPosition(pos)
            for _ in range(tlen):
                cur.movePosition(
                    QTextCursor.MoveOperation.NextCharacter,
                    QTextCursor.MoveMode.KeepAnchor,
                )
            if cur.hasSelection():
                cur.insertText(new_text)
                applied += 1
        if applied == 0 and fallback_entry is not None:
            # 兜底：用旧的"先 highlight 再 insertText"路径
            tok = fallback_entry.token_text or fallback_entry.char
            self._highlight_char_in_text(tok, focus_entry=fallback_entry)
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
        # 质量探针观测：用户针对某个槽位动了手，如果该位是 probe，则记 corrected
        any_probe_hit = False
        if applied:
            store = qp.get_active_store()
            if store is not None:
                for e in entries:
                    bi = _line_block_index(self._pages, e.page_number, e.line)
                    if bi is None:
                        continue
                    if qp.observe_slot_edit(
                        store, e.page_number, bi, e.line_idx, e.char_idx,
                    ):
                        any_probe_hit = True
        # Round 18：批改路径也走文本锚点兜底——可能改到了 probe 位置
        # 但 observe_slot_edit 因为 char_idx 不在 probe key 上而漏报。
        try:
            
            extra_hits = qp.detect_corrections(qp.get_active_store(), self._pages)
        except Exception:
            extra_hits = 0
        if any_probe_hit or extra_hits:
            # corrected probe 立刻进入正确集合 —— gallery 原地刷新。
            self._refresh_current_char_gallery()
        return applied

    def _refresh_current_char_gallery(self) -> None:
        """Round 17：原地重建当前 selected_char 的 gallery（保持 _selected_char 不变）。

        被 probe.observation 翻转后调用：保证 corrected probe 立即从 extras 退出，
        以及"真正改字成 true_char"后该位置以 _char_svc 正常 entry 形式出现。
        """
        tok = self._selected_char
        if not tok:
            return
        try:
            entries = list(self._char_svc.query(tok))
            entries.extend(self._extras_for_tokens([tok]))
            self._gallery_model.set_entries(entries)
            self._resize_gallery_for_entries(len(entries))
        except Exception:
            pass

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
        # proof-bbox-boxedit round 9：批量后焦点回 batch 输入框，方便连续操作；
        # 不再抢 _text_edit 焦点，注意力仍在 gallery 旁。
        self._batch_input.selectAll()
        self._batch_input.setFocus()
        if applied > 1:
            self._status_lbl.setText(f"● 已批量应用 \"{new_text}\" 到 {applied} 处，待保存")
        else:
            self._status_lbl.setText(f"● 已应用 \"{new_text}\"，待保存")
        self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")

    # ─────────── proof-bbox-boxedit round 9：单槽位编辑器 ───────────

    def _refresh_slot_info(self, entry: Optional[CharEntry]) -> None:
        """更新 _slot_info_lbl + 预填 _slot_edit_input 当前槽位文本。

        预填方便用户"轻改一下"再 Enter；如果用户不想动当前文本，删掉重输即可。
        """
        if not hasattr(self, "_slot_info_lbl"):
            return
        if entry is None:
            self._slot_info_lbl.setText("当前槽位：—")
            self._slot_edit_input.clear()
            self._slot_apply_btn.setEnabled(False)
            self._slot_blank_btn.setEnabled(False)
            return
        tok = entry.token_text or entry.char
        self._slot_info_lbl.setText(
            f'当前槽位：P{entry.page_number}·#{entry.char_idx + 1} "{tok}"'
        )
        # 预填但不 selectAll —— 让用户点输入框时不会立即覆盖
        if not self._slot_edit_input.hasFocus():
            self._slot_edit_input.setText(tok)
        self._slot_apply_btn.setEnabled(True)
        self._slot_blank_btn.setEnabled(True)

    def _apply_slot_edit_input(self) -> None:
        """把 _slot_edit_input 的当前值应用到当前 entry（单槽位）。

        与批量路径区别：忽略 gallery 多选，直接对 _current_candidate_entry 改。
        如果当前 entry 是跨页的，按 _apply_replacement_to_selected 的旧逻辑兜底。
        """
        entry = self._current_candidate_entry
        if entry is None:
            self._status_lbl.setText("槽位编辑：先在 gallery 选一个槽位")
            return
        new_text = self._slot_edit_input.text()
        if new_text == "":
            self._status_lbl.setText('槽位编辑：空文本请用 "清空(填空白)" 按钮')
            return
        # 临时清掉 gallery 多选，让 _apply_replacement_to_selected 走 fallback 路径
        sel = self._gallery_view.selectionModel()
        saved = list(sel.selectedIndexes())
        sel.clearSelection()
        try:
            applied = self._apply_replacement_to_selected(new_text, fallback_entry=entry)
        finally:
            # 还原多选
            for idx in saved:
                sel.select(idx, sel.SelectionFlag.Select)
        if applied:
            self._status_lbl.setText(
                f'● 槽位 P{entry.page_number}·#{entry.char_idx + 1} → "{new_text}"，待保存'
            )
            self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")
        else:
            self._status_lbl.setText("槽位编辑：当前槽不在当前页，已跳过")

    def _apply_slot_blank_to_current(self) -> None:
        """proof-bbox-boxedit round 9 第 2 任务：把当前槽位填成 ASCII 空格，
        位置不丢失。直接走 _fill_slot_with_blank。"""
        entry = self._current_candidate_entry
        if entry is None:
            self._status_lbl.setText("清空槽位：先在 gallery 选一个槽位")
            return
        pos = self._entry_text_pos(entry)
        if pos is None:
            self._status_lbl.setText("清空槽位：当前槽不在当前页，已跳过")
            return
        tok = entry.token_text or entry.char
        self._fill_slot_with_blank(pos, pos + max(1, len(tok)))
        self._status_lbl.setText(
            f'● 槽位 P{entry.page_number}·#{entry.char_idx + 1} 已清空为空白，待保存'
        )
        self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")

    # ─── vproof-direct-overwrite-residual round 11：gallery 直输回调 ───
    def _gallery_direct_overwrite(self, text: str) -> bool:
        """_GalleryListView.keyPressEvent 调用：把单字 ``text`` 覆盖当前 entry。

        与 _apply_slot_edit_input 同一管道（清掉 gallery 多选 → 走
        _apply_replacement_to_selected 的 fallback 路径 → 还原选区），但不读
        _slot_edit_input、不需要用户切焦点。返回 True 表示已处理（按键应被吞）。
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
                f'● 直输 P{entry.page_number}·#{entry.char_idx + 1} → "{text}"，待保存'
            )
            self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")
            # 同步预填 slot 输入框（不抢焦点）
            if hasattr(self, "_slot_edit_input") and not self._slot_edit_input.hasFocus():
                self._slot_edit_input.setText(text)
            # round 12 任务 3+4：debounce 120ms 把编辑落到 line.final_text + 重建索引
            self._local_commit_timer.start()
            return True
        return False

    def _gallery_direct_blank(self) -> bool:
        """_GalleryListView.keyPressEvent 调用：Backspace/Delete 直接把当前
        槽位填空白（保持长度）。返回 True 表示已处理。"""
        entry = self._current_candidate_entry
        if entry is None:
            return False
        pos = self._entry_text_pos(entry)
        if pos is None:
            return False
        tok = entry.token_text or entry.char
        self._fill_slot_with_blank(pos, pos + max(1, len(tok)))
        self._status_lbl.setText(
            f'● 直输 P{entry.page_number}·#{entry.char_idx + 1} 已清空为空白，待保存'
        )
        self._status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")
        if hasattr(self, "_slot_edit_input") and not self._slot_edit_input.hasFocus():
            self._slot_edit_input.clear()
        # round 12 任务 3+4：同样要落到 page 模型 + 重建字索引
        self._local_commit_timer.start()
        return True

    # ───── vproof-ime-persist-visibility round 12 任务 3+4：本地编辑落盘 ─────
    def _commit_local_edits_and_refresh(self) -> None:
        """把 _text_edit 当前内容保存到 line.final_text，并重建字索引/字列表。

        本轮硬验收项 3 + 4 的核心：
          - 任务 3：把"也"改成"好"切走再回来不应回退 —— 之前
            _gallery_direct_overwrite 只动 _text_edit 的 QTextDocument，没回写
            到 line.final_text，所以 _on_char_clicked 再次从 _char_svc.query 取
            出来还是旧"也"。
          - 任务 4：改"也"→"好"以后"也"的计数不变 —— 同样因为 _char_svc 没
            重建，频次是冻结的。
        debounce 120ms 走这里一次：保存 → 重建 _char_svc → 重建 _char_list →
        恢复用户视角（之前选中的字 / gallery 行）。

        诚实交代：走的是和 _do_external_refresh 同样的全量重建
        (_char_svc.build(all pages))，没改成增量；超大工程上单次会有几百
        ms 卡顿。本轮不动这一层。"""
        if not self._pages:
            return
        if self._text_edit.toPlainText() == self._loaded_text:
            return
        prev_char = self._selected_char
        sel_model = self._gallery_view.selectionModel() if self._gallery_view else None
        prev_row = (
            sel_model.currentIndex().row()
            if sel_model and sel_model.currentIndex().isValid()
            else 0
        )
        prev_entry = self._current_candidate_entry
        # 1) 落盘
        self._save_page_text()
        # 2) 重建字索引
        self._char_svc.build(self._pages)
        # 3) 重建左侧字列表
        self._rebuild_char_list()
        # 4) 恢复字列表选中
        target_char = prev_char
        if prev_char:
            still_there = any(
                self._char_list.item(i).data(Qt.ItemDataRole.UserRole) == prev_char
                for i in range(self._char_list.count())
            )
            if not still_there and prev_entry is not None:
                new_char = self._lookup_token_at_entry_position(prev_entry)
                if new_char:
                    target_char = new_char
        if target_char:
            for i in range(self._char_list.count()):
                it = self._char_list.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == target_char:
                    self._char_list.blockSignals(True)
                    self._char_list.setCurrentRow(i)
                    self._char_list.blockSignals(False)
                    self._on_char_clicked(it)
                    break
        # 5) 恢复 gallery 当前位置
        new_count = self._gallery_model.rowCount()
        if new_count > 0:
            row = max(0, min(prev_row, new_count - 1))
            idx = self._gallery_model.index(row, 0)
            self._gallery_view.setCurrentIndex(idx)
            self._sync_gallery_entry(idx)

    def _lookup_token_at_entry_position(self, entry: "CharEntry") -> Optional[str]:
        """根据 (page_number, block_order, line_idx, char_idx) 在当前 _pages
        模型里查那个槽位现在是什么字。改字后原字消失时用它找到新字。"""
        if not self._pages:
            return None
        for page in self._pages:
            if page.page_number != entry.page_number:
                continue
            for block in page.blocks:
                if block.order != entry.block_order:
                    continue
                if not (0 <= entry.line_idx < len(block.lines)):
                    continue
                line = block.lines[entry.line_idx]
                txt = line.display_text
                if 0 <= entry.char_idx < len(txt):
                    return txt[entry.char_idx]
        return None

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

    # ─────── proof-bbox-boxedit round 9 第 3 任务：modifier + 方向键 ────
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
        tokens = self._selected_char_tokens()
        if len(tokens) > 1:
            extra = f"  ·  已选 {sel_count} 个" if sel_count > 1 else ""
            joined = " ".join(f'"{t}"' for t in tokens)
            self._gallery_hdr.setText(
                f"跨字索引（{len(tokens)} 字 / 共 {self._gallery_model.rowCount()} 处）"
                f"{extra}：[第 {entry.page_number} 页 / 第 {entry.char_idx + 1} 位] {joined}"
            )
            return
        # 探针 extras 也计入 total；直接读 gallery_model 的 rowCount 最不会错
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
        # proof-bbox-boxedit round 9 第 1 任务：刷新"当前槽位"提示+输入框预填
        self._refresh_slot_info(entry)
        # 更新标题：让用户清楚当前看的是哪页
        if self._selected_char:
            sel_count = len(self._gallery_view.selectionModel().selectedIndexes())
            self._refresh_gallery_header(index, sel_count)
            # 传入 entry 以精准定位到该出现，而非首次出现
            self._highlight_char_in_text(entry.token_text or entry.char, focus_entry=entry)
            # vproof-direct-overwrite-residual round 11 任务 1：上一轮把焦点
            # 推到 _slot_edit_input，导致"在 gallery 上直接输入"假闭环——
            # 用户敲键时键盘焦点根本不在 gallery_view，eventFilter 永远不响应。
            # 本轮：选完 entry 把焦点留在 gallery_view，让 _GalleryListView
            # 的 keyPressEvent 真正接到按键。_slot_edit_input 仍可用，但是
            # 用户必须自己点过去才编辑，不再主动抢焦点。
            # 注意：如果用户当前正在 _slot_edit_input / _batch_input / _text_edit
            # 里打字（hasFocus），就别抢，以免吞他正打到一半的输入。
            side_inputs = (
                getattr(self, "_slot_edit_input", None),
                getattr(self, "_batch_input", None),
                self._text_edit,
            )
            if not any(w is not None and w.hasFocus() for w in side_inputs):
                self._gallery_view.setFocus()

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
                    self._publish_line_update(
                        page=page,
                        block=block,
                        line=line,
                        line_index=_line_idx,
                        status=line.proof_status.value,
                        source="vproof.page_text",
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
        # Round 18：以文本为锚的最终兜底——保存之后扫一遍所有 probe，
        # 任何"final_text 不再持有 true_char"的位置都标 corrected 并广播。
        try:
            
            qp.detect_corrections(qp.get_active_store(), self._pages)
        except Exception:
            pass

    # Round 17：页级 OK 标记已删除（不属于纵校语义）。

    # ─────────────────── 翻页 ───────────────────────────────

    def _prev_page(self) -> None:
        self._load_page(self._current_page_idx - 1)

    def _next_page(self) -> None:
        self._load_page(self._current_page_idx + 1)

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

    def _on_external_line_changed(self, event=None, **kwargs) -> None:
        """收到外部（横校）发来的 line.proof_changed。

        vproof-direct-overwrite-residual round 11 任务 2 —— 卡死链路诚实排查：

        ProofStateBus 是同步单线程 dispatch；上一轮 round 10 加的
        ``_handling_external`` 重入闸其实是 placebo——同线程同步派发不会真的
        递归进自己（自反消息也被 ``origin == id(self)`` 挡掉）。

        真正会让 V-校"无响应"的原因是：HProof 一次大批保存会按行 publish N 条
        ``line.proof_changed``。每条进到这里都会做一次：
          (1) _save_page_text 把 V 侧本地 _text_edit 落盘；
          (2) _load_page 重建当前页文本 + viewer 图像；
          (3) _char_svc.build(self._pages) —— 扫全部 page 重建字索引；
          (4) _rebuild_char_list —— 整个 QListWidget 重建。
        N 行 × O(全工程字数) 的同步重建，主线程被占满，从外面看就是"卡死/无
        响应"。

        本轮做的是把这串重建动作 debounce 到 QTimer 单次延后：bus 收到 N 条
        事件只会把 line.id 累到 pending set 里 + 重启 80ms timer；timer 到点
        触发 _do_external_refresh，把"是否还要重建"做一次判断后只跑一次。这
        样 N → 1。仍然是 best-effort 缓解，不是从根上让 char_svc.build 变成
        增量；如果工程很大，单次重建本身仍会卡 ~几百毫秒，但不会再被 N 倍放
        大。
        """
        request = ProofUpdateRequest.from_legacy(event, **kwargs)
        if request.origin == id(self):
            return
        if not self._pages:
            return
        line_uid = request.line_uid or None
        if line_uid is None and request.line_id is None:
            return
        cur_page = self._pages[self._current_page_idx]
        if not proof_request_matches_page(request, cur_page):
            return
        # 当前页有这一行才入队；不在当前页的事件直接丢，因为切页时会自然刷新
        for _block, line, _li in iter_unique_page_text_lines(cur_page):
            if proof_request_matches_line(request, line):
                line_key = line_uid if line_uid is not None else request.line_id
                self._pending_external_lines.add(line_key)
                self._external_refresh_timer.start()  # 80ms 内的 N 次 publish 合并成 1 次
                return

    def _do_external_refresh(self) -> None:
        """vproof-direct-overwrite-residual round 11：debounced 外部刷新。

        从 _on_external_line_changed 累积的 pending 事件里只触发一次重建。
        与原先的同步路径相比，重建本身的代价没变，但 N→1 折叠掉了重复。
        """
        if not self._pages:
            self._pending_external_lines.clear()
            return
        if not self._pending_external_lines:
            return
        self._pending_external_lines.clear()
        # Phase 22 blocker 1：先把用户在 _text_edit 里尚未保存的输入落盘（走
        # _save_page_text 同样的 quality_probe 桥），否则紧接着的 _load_page
        # 会用 page 当前 final_text 重新渲染，把用户在编辑的文本静默覆盖掉。
        if self._text_edit.toPlainText() != self._loaded_text:
            self._save_page_text()
        self._load_page(self._current_page_idx)
        self._char_svc.build(self._pages)
        self._rebuild_char_list()

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
