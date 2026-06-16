"""共享页面目录列表组件。

PageDir 行版式（对齐 Pencil 设计稿 N56wj）：

    ┌─────────────────────────────────────┐
    │ │ ┌──┐  第 1 页              [校] │  62h（选中态左侧 3px brand 条）
    │ │ │📄│  filename.jpg              │
    │ │ └──┘                            │
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
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from app.models import Page
from app.models.enums import PageStatus

# 缩略图固定 38×46（对齐设计稿）
_THUMB_W = 38
_THUMB_H = 46
_ROW_H   = 62

# PageStatus → (badge_text, badge_kind)  kind 命中 QLabel#pageBadge[kind=...]
_STATUS_BADGE: dict[str, tuple[str, str]] = {
    PageStatus.LAYOUT_DONE.value:       ("版", "done"),
    PageStatus.LAYOUT_CONFIRMED.value:  ("版", "done"),
    PageStatus.OCR_DONE.value:          ("识", "running"),
    PageStatus.PRE_REVIEW_DONE.value:   ("预", "running"),
    PageStatus.PROOFING.value:          ("校", "warn"),
    PageStatus.PROOF_DONE.value:        ("✓", "done"),
    PageStatus.ERROR.value:             ("!", "err"),
}


class _PageRow(QWidget):
    """单条页面行：缩略图 + 标题 + 文件名 + 状态徽章。"""

    def __init__(self, page: Page, thumbnail: QPixmap | None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageRow")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedHeight(_ROW_H)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(10)

        # 缩略图
        thumb_lbl = QLabel()
        thumb_lbl.setObjectName("pageThumb")
        thumb_lbl.setFixedSize(_THUMB_W, _THUMB_H)
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setScaledContents(False)
        if thumbnail is not None and not thumbnail.isNull():
            thumb_lbl.setPixmap(thumbnail)
        else:
            thumb_lbl.setText("📄")
        row.addWidget(thumb_lbl)

        # 文本
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        title = QLabel(f"第 {page.page_number} 页")
        title.setObjectName("pageRowTitle")
        text_col.addWidget(title)

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
        elided = fm.elidedText(fname, Qt.ElideMiddle, 130)
        fname_lbl.setText(elided)
        text_col.addWidget(fname_lbl)
        row.addLayout(text_col, 1)

        # 状态徽章
        status_val = getattr(getattr(page, "status", None), "value", None)
        badge = _STATUS_BADGE.get(status_val)
        if badge is not None:
            text, kind = badge
            badge_lbl = QLabel(text)
            badge_lbl.setObjectName("pageBadge")
            badge_lbl.setProperty("kind", kind)
            badge_lbl.setAlignment(Qt.AlignCenter)
            badge_lbl.setFixedSize(22, 18)
            row.addWidget(badge_lbl)


class PageDirectoryList(QListWidget):
    """轻量子类：自定义行 widget + 固定 sizeHint。"""

    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(240)
        self.setMinimumWidth(180)
        self.setSpacing(0)
        self.setUniformItemSizes(True)
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
        return pm.scaled(
            _THUMB_W, _THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _on_row_changed(self, idx: int) -> None:
        if self._suppress_signal:
            return
        if 0 <= idx < self.count():
            self.page_selected.emit(idx)
