"""共享页面目录缩略图组件（Stitch Minimalist Studio 卡片风格）。

行版式：

    ┌─────────────────────────┐
    │   ┌────────────── 01 ┐  │
    │   │    thumbnail     │  │
    │   └──────────────────┘  │
    │       filename.tif      │
    └─────────────────────────┘

缩略图采用 KeepAspectRatioByExpanding + 居中裁剪（封面式填充），并按
devicePixelRatio 生成高分 pixmap，避免高 DPI 下模糊；纯函数输入，不依赖
任何项目数据模型，横校与版面两个页面目录共用。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from app.ui.image_orientation import rotate_image, rotated_size
from app.ui.widgets.effects import apply_soft_shadow

PAGE_THUMB_W = 150
PAGE_THUMB_H = 198
PAGE_ROW_H = 236


def make_page_thumbnail(
    image_path: str,
    *,
    width: int = PAGE_THUMB_W,
    height: int = PAGE_THUMB_H,
    device_pixel_ratio: float = 1.0,
    rotation_quarters_clockwise: int = 0,
) -> QPixmap:
    """Cover-style thumbnail: expanding scale + centered crop, DPR aware.

    Decoding goes through ``QImageReader.setScaledSize`` so a 600 DPI page
    is never fully decoded on the UI thread just to make a 150px thumb.
    """

    if not image_path:
        return QPixmap()
    dpr = max(1.0, float(device_pixel_ratio))
    target_w = max(1, round(width * dpr))
    target_h = max(1, round(height * dpr))
    reader = QImageReader(image_path)
    source_size = reader.size()
    if source_size.width() <= 0 or source_size.height() <= 0:
        return QPixmap()
    display_width, display_height = rotated_size(
        source_size.width(),
        source_size.height(),
        rotation_quarters_clockwise,
    )
    scale = max(target_w / display_width, target_h / display_height)
    reader.setScaledSize(
        QSize(
            max(1, round(source_size.width() * scale)),
            max(1, round(source_size.height() * scale)),
        )
    )
    image = reader.read()
    if image.isNull():
        return QPixmap()
    image = rotate_image(image, rotation_quarters_clockwise)
    x = max(0, (image.width() - target_w) // 2)
    y = max(0, (image.height() - target_h) // 2)
    thumb = QPixmap.fromImage(image.copy(x, y, target_w, target_h))
    thumb.setDevicePixelRatio(dpr)
    return thumb


class PageDirectoryRow(QWidget):
    """单条页面行：缩略图卡片 + 页码徽章 + 省略文件名。"""

    def __init__(
        self,
        page_number: int,
        source_name: str,
        image_path: str,
        *,
        badge_text: str | None = None,
        rotation_quarters_clockwise: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("pageRow")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedHeight(PAGE_ROW_H)

        row = QVBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        thumb_card = QFrame()
        thumb_card.setObjectName("pageThumbCard")
        thumb_card.setFixedSize(PAGE_THUMB_W + 8, PAGE_THUMB_H + 8)
        thumb_layout = QVBoxLayout(thumb_card)
        thumb_layout.setContentsMargins(4, 4, 4, 4)
        thumb_layout.setSpacing(0)

        self._thumb_label = QLabel()
        self._thumb_label.setObjectName("pageThumb")
        self._thumb_label.setFixedSize(PAGE_THUMB_W, PAGE_THUMB_H)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setScaledContents(False)
        self.thumbnail = make_page_thumbnail(
            image_path,
            device_pixel_ratio=self.devicePixelRatioF(),
            rotation_quarters_clockwise=rotation_quarters_clockwise,
        )
        if not self.thumbnail.isNull():
            self._thumb_label.setPixmap(self.thumbnail)
        else:
            self._thumb_label.setText("DOC")
        thumb_layout.addWidget(self._thumb_label)
        apply_soft_shadow(thumb_card, blur_radius=16, y_offset=3, alpha=18)

        self.badge = QLabel(badge_text or f"{page_number:02d}", thumb_card)
        self.badge.setObjectName("pageBadge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.adjustSize()
        self.badge.setFixedHeight(20)
        if self.badge.width() < 28:
            self.badge.setFixedWidth(28)
        self.badge.move(
            max(4, PAGE_THUMB_W + 4 - self.badge.width()),
            8,
        )
        self.badge.raise_()
        row.addWidget(thumb_card, 0, Qt.AlignmentFlag.AlignHCenter)

        name = Path(source_name).name if source_name else ""
        self.filename = QLabel(name)
        self.filename.setObjectName("pageRowFile")
        elided = self.filename.fontMetrics().elidedText(
            name,
            Qt.TextElideMode.ElideMiddle,
            PAGE_THUMB_W,
        )
        self.filename.setText(elided)
        self.filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.filename)


__all__ = [
    "PAGE_ROW_H",
    "PAGE_THUMB_H",
    "PAGE_THUMB_W",
    "PageDirectoryRow",
    "make_page_thumbnail",
]
