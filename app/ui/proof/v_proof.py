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
from typing import List, Optional, Tuple

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
    QAbstractItemView, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSplitter, QStyle,
    QStyledItemDelegate, QVBoxLayout, QWidget,
)

from app.models import BBox, Block, Line, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.core.proof_state_bus import ProofStateBus
from app.services.char_index_service import CharEntry, CharIndexService
from app.ui.widgets.confidence_badge import ConfidenceBadge
from app.ui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

CHAR_LIST_THUMB = 44
GALLERY_THUMB   = 72   # gallery 水平条高度（较大缩略图）
LOW_CONF        = 0.80


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
    """带坐标校验的裁图。坐标超出图像范围时记录 WARNING 并尝试修正。

    pad 为 None 时根据 bbox 尺寸自适应：~10% 且不超 6 像素，
    避免纵排字中高度 ~40px 的字被固定 12px 填充拽进邻字。
    """
    img = cache.get_image(page_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    if pad is None:
        pad = max(1, min(int(min(bbox.w, bbox.h) * 0.10), 6))
    # 检查坐标合理性
    if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.w > W or bbox.y + bbox.h > H:
        logger.warning(
            "char bbox out of bounds: bbox=(%d,%d,%d,%d) img=(%d×%d) path=%s",
            bbox.x, bbox.y, bbox.w, bbox.h, W, H, page_path,
        )
        # 修正到图像边界内
        from app.models import BBox as BBox2
        bbox = BBox2(
            max(0, min(bbox.x, W - 1)),
            max(0, min(bbox.y, H - 1)),
            max(1, min(bbox.w, W - bbox.x)),
            max(1, min(bbox.h, H - bbox.y)),
        )
    return cache.get_char_crop(page_path, bbox, size, pad=pad)


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
            return f"{index.row() + 1:03d}\nP{entry.page_number}-{entry.char_idx + 1}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"第 {entry.page_number} 页，字#{entry.char_idx + 1}"
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None


class _GalleryDelegate(QStyledItemDelegate):
    SIZE = GALLERY_THUMB + 28  # 图 + 标签

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        r = option.rect
        pix: Optional[QPixmap] = index.data(Qt.ItemDataRole.DecorationRole)
        img_r = r.adjusted(2, 2, -2, -(28))
        if pix and not pix.isNull():
            scaled = pix.scaled(
                img_r.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            dx = (img_r.width() - scaled.width()) // 2
            painter.drawPixmap(img_r.x() + dx, img_r.y(), scaled)
        else:
            painter.fillRect(img_r, QColor("#f0f6ff"))

        # 标签
        lbl = index.data(Qt.ItemDataRole.DisplayRole) or ""
        painter.setPen(QColor("#888"))
        lbl_r = r.adjusted(0, GALLERY_THUMB + 2, 0, 0)
        painter.drawText(
            lbl_r, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            lbl,
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
    for block in page.text_blocks:
        for line in block.lines:
            text = line.text or ""
            for ci, c in enumerate(text):
                mapping.append((line, ci, pos, pos + 1))
                pos += 1
            if text:
                parts.append(text)
                parts.append("\n")
                pos += 1
            else:
                parts.append("\n")
                pos += 1
        parts.append("\n")
        pos += 1
    return "".join(parts), mapping


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
        self._gallery_model = _GalleryModel(self._cache)
        self._selected_char: str = ""
        self._updating = False
        self._build_ui()

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
        left_box = self._build_char_list()
        left_box.setMinimumWidth(160)
        left_box.setMaximumWidth(240)
        h_split.addWidget(left_box)

        # 右：垂直分割（上gallery | 下文本/图）
        right_box = self._build_right_area()
        h_split.addWidget(right_box)
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 4)

        root.addWidget(h_split, 1)

        # ── 信号 ────────────────────────────────────────────
        self._btn_prev_page.clicked.connect(self._prev_page)
        self._btn_next_page.clicked.connect(self._next_page)
        self._btn_save.clicked.connect(self._save_page_text)
        self._btn_ok.clicked.connect(self._mark_page_ok)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_page_text)
        QShortcut(QKeySequence("PageUp"), self, activated=self._prev_page)
        QShortcut(QKeySequence("PageDown"), self, activated=self._next_page)

    def _build_char_list(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel("单字列表")
        lbl.setObjectName("sectionTitle")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._char_count_lbl = QLabel("共 0 字")
        self._char_count_lbl.setObjectName("muted")
        hdr.addWidget(self._char_count_lbl)
        layout.addLayout(hdr)

        self._char_search = QLineEdit()
        self._char_search.setPlaceholderText("搜索字符…")
        self._char_search.textChanged.connect(self._filter_char_list)
        layout.addWidget(self._char_search)

        self._char_list = QListWidget()
        self._char_list.setIconSize(QSize(CHAR_LIST_THUMB, CHAR_LIST_THUMB))
        self._char_list.setSpacing(2)
        self._char_list.itemClicked.connect(self._on_char_clicked)
        self._char_list.currentItemChanged.connect(
            lambda cur, _prev: self._on_char_clicked(cur) if cur else None
        )
        layout.addWidget(self._char_list)
        return box

    def _build_right_area(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # 上：gallery 水平条（固定高度）
        gallery_box = self._build_gallery_strip()
        gallery_box.setFixedHeight(GALLERY_THUMB + 50)
        v.addWidget(gallery_box)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#e3e8ef;")
        v.addWidget(sep)

        # 下：OCR 文本 | 原图（QSplitter）
        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        bottom_split.setHandleWidth(1)

        bl_box = self._build_ocr_text()
        br_box = self._build_viewer()

        bottom_split.addWidget(bl_box)
        bottom_split.addWidget(br_box)
        bottom_split.setStretchFactor(0, 1)
        bottom_split.setStretchFactor(1, 2)

        v.addWidget(bottom_split, 1)
        return box

    def _build_gallery_strip(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        self._gallery_hdr = QLabel("相同字索引（请先在左侧选择一个字）")
        self._gallery_hdr.setObjectName("sectionTitle")
        hdr.addWidget(self._gallery_hdr)
        hdr.addStretch()
        layout.addLayout(hdr)

        self._gallery_view = QListView()
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.setItemDelegate(_GalleryDelegate(self._gallery_view))
        self._gallery_view.setViewMode(QListView.ViewMode.IconMode)
        self._gallery_view.setFlow(QListView.Flow.LeftToRight)  # 水平排列
        self._gallery_view.setWrapping(False)                   # 不换行
        self._gallery_view.setResizeMode(QListView.ResizeMode.Fixed)
        self._gallery_view.setMovement(QListView.Movement.Static)
        self._gallery_view.setUniformItemSizes(True)
        self._gallery_view.setSpacing(4)
        self._gallery_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._gallery_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gallery_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._gallery_view.clicked.connect(self._on_gallery_clicked)
        layout.addWidget(self._gallery_view)
        return box

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

    def reset(self) -> None:
        self._pages = []
        self._char_svc = CharIndexService()
        self._char_list.clear()
        self._text_edit.clear()
        self._gallery_model.set_entries([])
        self._page_label.setText("页 0 / 0")

    # ─────────────────── 单字列表 ────────────────────────────

    def _rebuild_char_list(self) -> None:
        self._char_list.clear()
        # 按拼音 / 字母 a-z 优先，后续为数字 → 标点 → 符号 → 其他
        freqs = self._char_svc.sorted_chars()
        self._char_count_lbl.setText(f"共 {len(freqs)} 字")
        for char, count in freqs:
            item = QListWidgetItem(f"{char} ×{count}")
            entry = self._char_svc.first_entry(char)
            if entry:
                pix = _verified_char_crop(
                    self._cache, entry.page_path, entry.bbox, CHAR_LIST_THUMB
                )
                if pix:
                    item.setIcon(QIcon(pix))
            item.setData(Qt.ItemDataRole.UserRole, char)
            self._char_list.addItem(item)

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

        lines = [ln for b in page.text_blocks for ln in b.lines]
        if lines:
            avg_conf = sum(ln.confidence for ln in lines) / len(lines)
            self._conf_badge.set_score(avg_conf)

        self._viewer.set_image(page.display_image_path)
        # 纵校视图仅供参考，不显示版面标注框（show_blocks 不调用）
        # 字符高亮由 highlight_bbox 单独绘制，避免与块框混淆

        flat_text, self._text_map = _build_text_map(page)
        self._updating = True
        self._text_edit.setPlainText(flat_text)
        self._updating = False
        self._status_lbl.setText("")
        self._status_lbl.setStyleSheet("")

    # ─────────────────── 单字列表点击 ───────────────────────

    def _on_char_clicked(self, item: QListWidgetItem) -> None:
        char = item.data(Qt.ItemDataRole.UserRole)
        if not char:
            return
        self._selected_char = char
        entries = self._char_svc.query(char)

        # 重置 gallery：清选中、滚回顶部
        self._gallery_model.set_entries(entries)
        self._gallery_view.clearSelection()
        if entries:
            self._gallery_view.scrollTo(
                self._gallery_model.index(0, 0),
                QAbstractItemView.ScrollHint.PositionAtTop,
            )
        self._gallery_hdr.setText(f'"{char}"  共 {len(entries)} 处')

        # 先定位原图（可能触发翻页 → setPlainText 重置文本），再高亮文本
        target = entries[0] if entries else None
        if target:
            self._highlight_char_in_viewer(target)
        self._highlight_char_in_text(char, focus_entry=target)

    def _entry_text_pos(self, entry: CharEntry) -> Optional[int]:
        """查找某个 CharEntry 在当前 _text_map 中的起始 start 光标位置。"""
        for line, ci, start, _end in self._text_map:
            if line is entry.line and ci == entry.char_idx:
                return start
        return None

    def _highlight_char_in_text(
        self, char: str, focus_entry: Optional[CharEntry] = None,
    ) -> None:
        doc = self._text_edit.document()
        # 先清除全文格式
        clear_cur = QTextCursor(doc)
        clear_cur.select(QTextCursor.SelectionType.Document)
        clear_cur.setCharFormat(QTextCharFormat())
        # 高亮全部出现位置
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#e3f0ff"))
        fmt.setForeground(QColor("#1a73e8"))
        cursor = doc.find(char)
        while not cursor.isNull():
            cursor.setCharFormat(fmt)
            cursor = doc.find(char, cursor)
        # 定位到具体 entry。若未提供则定位到首出现。
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
            place.movePosition(
                QTextCursor.MoveOperation.NextCharacter,
                QTextCursor.MoveMode.KeepAnchor,
            )
            self._text_edit.setTextCursor(place)
            self._text_edit.ensureCursorVisible()

    def _highlight_char_in_viewer(self, entry: CharEntry) -> None:
        if not self._pages:
            return
        cur_page = self._pages[self._current_page_idx]
        if entry.page_path == cur_page.display_image_path:
            self._viewer.highlight_bbox(entry.bbox)
        else:
            for i, p in enumerate(self._pages):
                if p.display_image_path == entry.page_path:
                    self._load_page(i)
                    self._viewer.highlight_bbox(entry.bbox)
                    break

    # ─────────────────── Gallery 点击 ───────────────────────

    def _on_gallery_clicked(self, index: QModelIndex) -> None:
        entry: Optional[CharEntry] = index.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        self._highlight_char_in_viewer(entry)  # 可能触发翻页
        if self._selected_char:
            # 传入 entry 以精准定位到该出现，而非首次出现
            self._highlight_char_in_text(self._selected_char, focus_entry=entry)

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
        for block in page.text_blocks:
            for line in block.lines:
                if idx < len(lines_text):
                    new_text = lines_text[idx].rstrip()
                    if new_text != (line.text or ""):
                        line.update_text(new_text)
                        changed = True
                        self._bus.publish(
                            "line.proof_changed",
                            page_id=page.id,
                            line_id=line.id,
                            status=line.proof_status.value,
                        )
                idx += 1
            idx += 1  # 跳过 block 末尾空行
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 已保存" if changed else "无变更")
        self._status_lbl.setStyleSheet(
            "color: #4CAF50; font-size: 12px;" if changed else ""
        )
        if changed:
            self._char_svc.build(self._pages)
            self._rebuild_char_list()

    def _mark_page_ok(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        for block in page.text_blocks:
            for line in block.lines:
                if line.proof_status == ProofStatus.UNCHECKED:
                    line.proof_status = ProofStatus.OK
                    self._bus.publish(
                        "line.proof_changed",
                        page_id=page.id,
                        line_id=line.id,
                        status=ProofStatus.OK.value,
                    )
        self.proof_saved.emit()
        self._status_lbl.setText("本页已确认")

    # ─────────────────── 翻页 ───────────────────────────────

    def _prev_page(self) -> None:
        self._load_page(self._current_page_idx - 1)

    def _next_page(self) -> None:
        self._load_page(self._current_page_idx + 1)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_PageUp:
            self._prev_page()
        elif key == Qt.Key.Key_PageDown:
            self._next_page()
        else:
            super().keyPressEvent(event)
