"""共享页面目录列表组件。

PageDir 行版式（对齐 Stitch Minimalist Studio）：

    ┌─────────────────────────────────────┐
    │        ┌─────────────── 01 ┐        │
    │        │    thumbnail      │        │
    │        └───────────────────┘        │
    │              filename.tif            │
    └─────────────────────────────────────┘

对外接口（保持稳定）：
- ``set_pages(pages)`` 重建条目
- ``set_current_index(idx)`` 同步选中行（不触发 page_selected）
- ``page_selected: Signal(int)`` 用户点击切换时派发当前 index
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from app.models import Page
from app.ui.widgets.effects import apply_soft_shadow

# 目录是大缩略图卡片，不再做紧凑行。
_THUMB_W = 150
_THUMB_H = 198
_ROW_H = 236

class _PageRow(QWidget):
    """单条页面行：缩略图 + 标题 + 文件名 + 状态徽章。"""

    def __init__(self, page: Page, thumbnail: QPixmap | None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageRow")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedHeight(_ROW_H)

        row = QVBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        thumb_card = QFrame()
        thumb_card.setObjectName("pageThumbCard")
        thumb_card.setFixedSize(_THUMB_W + 8, _THUMB_H + 8)
        thumb_layout = QVBoxLayout(thumb_card)
        thumb_layout.setContentsMargins(4, 4, 4, 4)
        thumb_layout.setSpacing(0)

        thumb_lbl = QLabel()
        thumb_lbl.setObjectName("pageThumb")
        thumb_lbl.setFixedSize(_THUMB_W, _THUMB_H)
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setScaledContents(False)
        if thumbnail is not None and not thumbnail.isNull():
            thumb_lbl.setPixmap(thumbnail)
        else:
            thumb_lbl.setText("DOC")
        thumb_layout.addWidget(thumb_lbl)
        apply_soft_shadow(thumb_card, blur_radius=16, y_offset=3, alpha=18)

        page_badge = QLabel(f"{page.page_number:02d}", thumb_card)
        page_badge.setObjectName("pageBadge")
        page_badge.setAlignment(Qt.AlignCenter)
        page_badge.setFixedSize(28, 20)
        page_badge.move(_THUMB_W - 24, 8)
        page_badge.raise_()
        row.addWidget(thumb_card, 0, Qt.AlignmentFlag.AlignHCenter)

        src = getattr(page, "source_path", None) or getattr(page, "image_path", None) or ""
        fname = Path(src).name if src else ""
        fname_lbl = QLabel(fname)
        fname_lbl.setObjectName("pageRowFile")
        # hproof-yaxis-quiet-load 本轮任务 2：加载页面时 directory 列表上
        # 这条 setToolTip 会让每个行都有 hover 弹窗，鼠标扫过即冒一串小框。
        # 改成静默：文件名本身已在行内显示（下方 elidedText），鼠标停留不再
        # 弹气泡。需要全名时可用状态栏 / 复制路径菜单。
        # 文件名过长省略
        fm = fname_lbl.fontMetrics()
        elided = fm.elidedText(fname, Qt.ElideMiddle, _THUMB_W)
        fname_lbl.setText(elided)
        fname_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(fname_lbl)


class PageDirectoryList(QListWidget):
    """轻量子类：自定义行 widget + 固定 sizeHint。"""

    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(236)
        self.setMinimumWidth(196)
        self.setSpacing(10)
        self.setUniformItemSizes(False)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self._suppress_signal = False
        self.currentRowChanged.connect(self._on_row_changed)

    # ── public ────────────────────────────────────────────────

    def set_pages(self, pages: List[Page]) -> None:
        self._suppress_signal = True
        try:
            self.clear()
            for page in pages:
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, _ROW_H))
                self.addItem(item)
                pm = self._make_thumbnail(page)
                widget = _PageRow(page, pm)
                self.setItemWidget(item, widget)
        finally:
            self._suppress_signal = False

    def set_current_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.count():
            return
        self._suppress_signal = True
        try:
            self.setCurrentRow(idx)
        finally:
            self._suppress_signal = False

    # ── internal ──────────────────────────────────────────────

    def _make_thumbnail(self, page: Page) -> QPixmap | None:
        img_path = getattr(page, "image_path", None) or getattr(page, "source_path", None)
        if not img_path:
            return None
        try:
            from app.core.page_image_cache import PageImageCache
            cache = PageImageCache.instance()
            pm = cache.get_pixmap(img_path) if hasattr(cache, "get_pixmap") else None
            if pm is None or pm.isNull():
                pm = QPixmap(img_path)
        except Exception:
            pm = QPixmap(img_path)
        if pm is None or pm.isNull():
            return None
        scaled = pm.scaled(
            _THUMB_W, _THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - _THUMB_W) // 2)
        y = max(0, (scaled.height() - _THUMB_H) // 2)
        return scaled.copy(x, y, _THUMB_W, _THUMB_H)

    def _on_row_changed(self, idx: int) -> None:
        if self._suppress_signal:
            return
        if 0 <= idx < self.count():
            self.page_selected.emit(idx)
