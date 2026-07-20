"""Session page directory projection for immutable records."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.project_session import PageRecord
from app.ui.widgets.effects import apply_soft_shadow


_THUMB_W = 150
_THUMB_H = 198
_ROW_H = 236


class _PageRow(QWidget):
    """One directory item rendered from an immutable page record."""

    def __init__(self, page: PageRecord, thumbnail: QPixmap | None, parent=None) -> None:
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

        thumb_label = QLabel()
        thumb_label.setObjectName("pageThumb")
        thumb_label.setFixedSize(_THUMB_W, _THUMB_H)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if thumbnail is not None and not thumbnail.isNull():
            thumb_label.setPixmap(thumbnail)
        else:
            thumb_label.setText("DOC")
        thumb_layout.addWidget(thumb_label)
        apply_soft_shadow(thumb_card, blur_radius=16, y_offset=3, alpha=18)

        page_badge = QLabel(f"{page.page_number:02d}", thumb_card)
        page_badge.setObjectName("pageBadge")
        page_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page_badge.setFixedSize(28, 20)
        page_badge.move(_THUMB_W - 24, 8)
        page_badge.raise_()
        row.addWidget(thumb_card, 0, Qt.AlignmentFlag.AlignHCenter)

        source = page.source_path or page.image_path
        filename = Path(source).name if source else ""
        filename_label = QLabel()
        filename_label.setObjectName("pageRowFile")
        filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filename_label.setText(
            filename_label.fontMetrics().elidedText(
                filename,
                Qt.TextElideMode.ElideMiddle,
                _THUMB_W,
            )
        )
        row.addWidget(filename_label)


class PageDirectoryList(QListWidget):
    """Directory list whose user selection emits a stable page UID."""

    page_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(236)
        self.setMinimumWidth(196)
        self.setSpacing(10)
        self.setUniformItemSizes(False)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self._suppress_signal = False
        self._page_uids: tuple[str, ...] = ()
        self.currentRowChanged.connect(self._on_row_changed)

    def set_pages(self, pages: Iterable[PageRecord]) -> None:
        records = tuple(pages)
        if any(not isinstance(page, PageRecord) for page in records):
            raise TypeError("page directory requires PageRecord values")
        self._suppress_signal = True
        try:
            self.clear()
            self._page_uids = tuple(page.uid for page in records)
            for page in records:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, page.uid)
                item.setSizeHint(QSize(0, _ROW_H))
                self.addItem(item)
                self.setItemWidget(item, _PageRow(page, self._make_thumbnail(page)))
        finally:
            self._suppress_signal = False

    def set_current_uid(self, page_uid: str) -> None:
        try:
            index = self._page_uids.index(page_uid)
        except ValueError as exc:
            raise ValueError(f"page UID is not present in directory: {page_uid!r}") from exc
        self._suppress_signal = True
        try:
            self.setCurrentRow(index)
        finally:
            self._suppress_signal = False

    def set_current_index(self, index: int) -> None:
        """Synchronize a caller-owned row index without emitting selection."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("directory index must be an integer")
        if index < 0 or index >= len(self._page_uids):
            raise IndexError(f"directory index is out of range: {index}")
        self._suppress_signal = True
        try:
            self.setCurrentRow(index)
        finally:
            self._suppress_signal = False

    def _make_thumbnail(self, page: PageRecord) -> QPixmap | None:
        image_path = page.thumbnail_path or page.image_path or page.source_path
        if not image_path:
            return None
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return None
        scaled = pixmap.scaled(
            _THUMB_W,
            _THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - _THUMB_W) // 2)
        y = max(0, (scaled.height() - _THUMB_H) // 2)
        return scaled.copy(x, y, _THUMB_W, _THUMB_H)

    def _on_row_changed(self, index: int) -> None:
        if self._suppress_signal or index < 0 or index >= len(self._page_uids):
            return
        self.page_selected.emit(self._page_uids[index])


__all__ = ["PageDirectoryList"]
