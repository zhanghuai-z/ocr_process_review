"""共享页面目录列表组件。

Phase 24：原 LayoutPanel 左侧页面列表抽为可复用 widget，由 LayoutPanel /
HProofPanel 共用。
Phase 25：每个页面项加上缩略图（PageImageCache 加载 → 等比缩放 → setIcon），
   提升页面定位效率，maxWidth 同步放大以容纳 thumbnail 列。

对外接口：
- ``set_pages(pages)`` 重建条目（每页一行：缩略图 + "第 N 页 / 文件名"）
- ``set_current_index(idx)`` 同步选中行（不触发 page_selected）
- ``page_selected: Signal(int)`` 用户点击切换时派发当前 index
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from app.models import Page

# Phase 25：缩略图尺寸 —— 80x60 兼顾纵横，足以辨识版面
_THUMB_W = 80
_THUMB_H = 60


class PageDirectoryList(QListWidget):
    """轻量子类，仅约束样式 / 信号契约，不引入额外业务状态。"""

    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        # Phase 25：缩略图列需要更宽
        self.setMaximumWidth(180)
        self.setMinimumWidth(100)
        self.setIconSize(QSize(_THUMB_W, _THUMB_H))
        self.setSpacing(2)
        self._suppress_signal = False
        self.currentRowChanged.connect(self._on_row_changed)

    # ── public ────────────────────────────────────────────────

    def set_pages(self, pages: List[Page]) -> None:
        self._suppress_signal = True
        try:
            self.clear()
            for page in pages:
                src = getattr(page, "source_path", None) or page.image_path or ""
                fname = Path(src).name if src else ""
                label = f"第 {page.page_number} 页"
                if fname:
                    label += f"\n{fname}"
                item = QListWidgetItem(label)
                # Phase 25：缩略图（失败则留空 icon，列表仍可用）
                pm = self._make_thumbnail(page)
                if pm is not None and not pm.isNull():
                    item.setIcon(QIcon(pm))
                self.addItem(item)
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
        """优先用 PageImageCache，fallback 直接 QPixmap.load。"""
        img_path = getattr(page, "image_path", None) or getattr(page, "source_path", None)
        if not img_path:
            return None
        # 1) PageImageCache 已经加载/缓存的可走快路径
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
