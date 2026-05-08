"""纵校面板（PRD 3.3）：四象限协同，虚拟 gallery，全书字频。

布局：
  ┌──────────────┬──────────────────┐
  │ TL: 单字列表  │ TR: 相同字图像集 │
  │  (freq 排序)  │  (虚拟滚动)      │
  ├──────────────┼──────────────────┤
  │ BL: OCR 文本  │ BR: 原图 + 高亮  │
  │   (可编辑)    │                  │
  └──────────────┴──────────────────┘

联动逻辑：
  - 点单字列表 → gallery 刷新 + 文本定位高亮 + 原图红框
  - 点 gallery 某个字 → 原图跳到对应页
  - 文本保存 → 重建字符索引

性能：
  - gallery 使用 QAbstractListModel + QListView + QStyledItemDelegate（虚拟渲染）
  - 页面图像通过 PageImageCache（LRU）缓存，不重复读盘
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    QAbstractListModel, QModelIndex, QSize, Qt, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap,
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

CHAR_LIST_THUMB = 52
GALLERY_THUMB = 56
LOW_CONF = 0.80


# ─────────────────────────────────────────────────────────────
# 虚拟缩略图 Model / Delegate（gallery TR）
# ─────────────────────────────────────────────────────────────

class _GalleryModel(QAbstractListModel):
    """存储 CharEntry 列表；data() 按需从 PageImageCache 取 QPixmap。"""

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
            return self._cache.get_char_crop(entry.page_path, entry.bbox, GALLERY_THUMB)
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"第 {entry.page_number} 页"
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None


class _GalleryDelegate(QStyledItemDelegate):
    """绘制缩略图；选中时蓝色边框。"""

    SIZE = GALLERY_THUMB + 8

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        r = option.rect.adjusted(2, 2, -2, -2)
        pix: Optional[QPixmap] = index.data(Qt.ItemDataRole.DecorationRole)
        if pix and not pix.isNull():
            scaled = pix.scaled(
                r.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # 居中绘制
            dx = (r.width()  - scaled.width())  // 2
            dy = (r.height() - scaled.height()) // 2
            painter.drawPixmap(r.x() + dx, r.y() + dy, scaled)
        else:
            painter.fillRect(r, QColor("#f0f6ff"))

        if option.state & QStyle.StateFlag.State_Selected:
            pen = QPen(QColor("#1a73e8"), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        return QSize(self.SIZE, self.SIZE)


# ─────────────────────────────────────────────────────────────
# 文本位置映射：(line, char_idx, text_pos_start, text_pos_end)
# ─────────────────────────────────────────────────────────────

def _build_text_map(
    page: Page,
) -> Tuple[str, List[Tuple[Line, int, int, int]]]:
    """把当前页所有文本拼为一个大字符串，并建立 (line, char_idx, start, end) 映射。

    返回 (flat_text, mapping)。
    """
    parts: List[str] = []
    mapping: List[Tuple[Line, int, int, int]] = []
    pos = 0
    for block in page.text_blocks:
        for line in block.lines:
            text = line.text
            for ci, c in enumerate(text):
                mapping.append((line, ci, pos, pos + 1))
                pos += 1
            # 行末换行
            if text:
                parts.append(text)
                parts.append("\n")
                pos += 1  # 换行符占位
            else:
                parts.append("\n")
                pos += 1
        # block 间额外空行
        parts.append("\n")
        pos += 1
    return "".join(parts), mapping


# ─────────────────────────────────────────────────────────────
# 纵校面板主体
# ─────────────────────────────────────────────────────────────

class VProofPanel(QWidget):
    """纵校面板：四象限协同。"""

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
        self._updating = False          # 防止循环触发
        self._build_ui()

    # ─────────────────── 构建 UI ────────────────────────────

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

        title = QLabel("⑤ 纵向校对")
        title.setObjectName("pageTitle")
        tl.addWidget(title)
        tl.addSpacing(16)

        self._btn_prev_page = QPushButton("← 上一页  PgUp")
        self._btn_next_page = QPushButton("下一页  PgDn →")
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

        # ── 四象限主体 ───────────────────────────────────────
        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.setHandleWidth(1)
        left  = QSplitter(Qt.Orientation.Vertical)
        right = QSplitter(Qt.Orientation.Vertical)
        left.setHandleWidth(1)
        right.setHandleWidth(1)

        # TL：单字列表（按频次排序）
        tl_box = self._build_tl()
        left.addWidget(tl_box)

        # BL：OCR 文本编辑器
        bl_box = self._build_bl()
        left.addWidget(bl_box)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 1)

        # TR：相同字图像 gallery（虚拟）
        tr_box = self._build_tr()
        right.addWidget(tr_box)

        # BR：原图 + 高亮
        br_box = self._build_br()
        right.addWidget(br_box)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 2)

        outer.addWidget(left)
        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 2)
        root.addWidget(outer)

        # ── 信号 ────────────────────────────────────────────
        self._btn_prev_page.clicked.connect(self._prev_page)
        self._btn_next_page.clicked.connect(self._next_page)
        self._btn_save.clicked.connect(self._save_page_text)
        self._btn_ok.clicked.connect(self._mark_page_ok)

    def _build_tl(self) -> QWidget:
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

        # 搜索框
        self._char_search = QLineEdit()
        self._char_search.setPlaceholderText("搜索字符…")
        self._char_search.textChanged.connect(self._filter_char_list)
        layout.addWidget(self._char_search)

        self._char_list = QListWidget()
        self._char_list.setIconSize(QSize(CHAR_LIST_THUMB, CHAR_LIST_THUMB))
        self._char_list.setSpacing(2)
        self._char_list.itemClicked.connect(self._on_char_clicked)
        layout.addWidget(self._char_list)
        return box

    def _build_bl(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("OCR 文本（直接编辑）"))
        layout.addLayout(hdr)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:16px; padding:8px;")
        layout.addWidget(self._text_edit)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("noteLabel")
        layout.addWidget(self._status_lbl)
        return box

    def _build_tr(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._gallery_hdr = QLabel("相同字图像（请先在左上选择一个字）")
        self._gallery_hdr.setObjectName("sectionTitle")
        layout.addWidget(self._gallery_hdr)

        self._gallery_view = QListView()
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.setItemDelegate(_GalleryDelegate(self._gallery_view))
        self._gallery_view.setViewMode(QListView.ViewMode.IconMode)
        self._gallery_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._gallery_view.setMovement(QListView.Movement.Static)
        self._gallery_view.setUniformItemSizes(True)
        self._gallery_view.setSpacing(4)
        self._gallery_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._gallery_view.clicked.connect(self._on_gallery_clicked)
        layout.addWidget(self._gallery_view)
        return box

    def _build_br(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("原图 + 版面框"))
        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        layout.addWidget(self._viewer)
        return box

    # ─────────────────── 公共 API ───────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        self._current_page_idx = 0
        # 重建全书字符索引
        self._char_svc.build(pages)
        self._rebuild_char_list()
        self._load_page(0)

    def reset(self) -> None:
        self._pages = []
        self._char_svc = CharIndexService()
        self._char_list.clear()
        self._text_edit.clear()
        self._gallery_model.set_entries([])
        self._page_label.setText("页 0 / 0")

    # ─────────────────── 字符列表 TL ────────────────────────

    def _rebuild_char_list(self) -> None:
        """按词频重建 TL 单字列表（唯一字 + 频次 + 缩略图）。"""
        self._char_list.clear()
        freqs = self._char_svc.char_frequency()
        self._char_count_lbl.setText(f"共 {len(freqs)} 字")
        for char, count in freqs:
            item = QListWidgetItem(f"{char} ×{count}")
            # 取首次出现的字符图像作为 icon
            entry = self._char_svc.first_entry(char)
            if entry:
                pix = self._cache.get_char_crop(
                    entry.page_path, entry.bbox, CHAR_LIST_THUMB
                )
                if pix:
                    from PySide6.QtGui import QIcon
                    item.setIcon(QIcon(pix))
            item.setData(Qt.ItemDataRole.UserRole, char)
            self._char_list.addItem(item)

    def _filter_char_list(self, text: str) -> None:
        for i in range(self._char_list.count()):
            item = self._char_list.item(i)
            char = item.data(Qt.ItemDataRole.UserRole) or ""
            item.setHidden(text != "" and text not in char)

    # ─────────────────── 页面加载 ───────────────────────────

    def _load_page(self, idx: int) -> None:
        if not self._pages:
            return
        idx = max(0, min(idx, len(self._pages) - 1))
        self._current_page_idx = idx
        page = self._pages[idx]
        self._page_label.setText(f"页 {idx + 1} / {len(self._pages)}")

        # 平均置信度
        lines = [ln for b in page.text_blocks for ln in b.lines]
        if lines:
            avg_conf = sum(ln.confidence for ln in lines) / len(lines)
            self._conf_badge.set_score(avg_conf)

        # 原图
        self._viewer.set_image(page.display_image_path)
        self._viewer.show_blocks(page.blocks)

        # OCR 文本 + 映射表
        flat_text, self._text_map = _build_text_map(page)
        self._updating = True
        self._text_edit.setPlainText(flat_text)
        self._updating = False
        self._status_lbl.setText("")

    # ─────────────────── 单字列表点击 ───────────────────────

    def _on_char_clicked(self, item: QListWidgetItem) -> None:
        char = item.data(Qt.ItemDataRole.UserRole)
        if not char:
            return
        self._selected_char = char
        entries = self._char_svc.query(char)
        self._gallery_model.set_entries(entries)
        # gallery 切字时强制滚回顶部并清空选中
        self._gallery_view.clearSelection()
        if entries:
            self._gallery_view.scrollTo(
                self._gallery_model.index(0, 0),
                QAbstractItemView.ScrollHint.PositionAtTop,
            )
        self._gallery_hdr.setText(
            f'"{char}"  共 {len(entries)} 处'
        )
        # 高亮文本中该字的首个出现
        self._highlight_char_in_text(char)
        # 原图定位到首次出现
        if entries:
            self._highlight_char_in_viewer(entries[0])

    def _highlight_char_in_text(self, char: str) -> None:
        """在文本编辑器中把该字的第一个出现位置滚动到视野内并高亮。"""
        doc = self._text_edit.document()
        cursor = doc.find(char)
        if not cursor.isNull():
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#e3f0ff"))
            fmt.setForeground(QColor("#1a73e8"))
            # 清除旧高亮
            clear_cursor = self._text_edit.textCursor()
            clear_cursor.select(QTextCursor.SelectionType.Document)
            clear_cursor.setCharFormat(QTextCharFormat())
            # 高亮首个
            cursor.setCharFormat(fmt)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.ensureCursorVisible()

    def _highlight_char_in_viewer(self, entry: CharEntry) -> None:
        """原图：若 entry 在当前页则高亮框，否则切换到对应页。"""
        if not self._pages:
            return
        cur_page = self._pages[self._current_page_idx]
        if entry.page_path == cur_page.display_image_path:
            self._viewer.highlight_bbox(entry.bbox)
        else:
            # 切页
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
        self._highlight_char_in_viewer(entry)

    # ─────────────────── 原图 block 点击 ────────────────────

    def _on_block_clicked(self, block: Block) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        # 在文本编辑器中找到该 block 的第一行第一个字
        for entry in self._text_map:
            line, ci, start, end = entry
            if any(line is ln for ln in block.lines):
                cursor = self._text_edit.textCursor()
                cursor.setPosition(start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.ensureCursorVisible()
                break

    # ─────────────────── 页面保存 ───────────────────────────

    def _save_page_text(self) -> None:
        if not self._pages:
            return
        page = self._pages[self._current_page_idx]
        flat = self._text_edit.toPlainText()
        lines_text = flat.split("\n")
        # 按 _build_text_map 的结构遍历：每行占一个 slot，每个 block 末尾占一个 slot
        changed = False
        idx = 0
        for block in page.text_blocks:
            for line in block.lines:
                if idx < len(lines_text):
                    new_text = lines_text[idx].rstrip()
                    if new_text != line.text:
                        line.update_text(new_text)
                        changed = True
                        self._bus.publish(
                            "line.proof_changed",
                            page_id=page.id,
                            line_id=line.id,
                            status=line.proof_status.value,
                        )
                idx += 1
            idx += 1  # 跳过 block 末尾的空行分隔符
        self.proof_saved.emit()
        self._status_lbl.setText("已保存" if changed else "无变更")
        # 重建字符索引（文本改变后）
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
