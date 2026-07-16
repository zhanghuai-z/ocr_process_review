from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem

from app.ui.widgets.image_viewer import BBoxItem


def test_block_frame_avoids_char_ink_without_changing_geometry():
    app = QApplication.instance() or QApplication([])
    item = BBoxItem(QRectF(0, 0, 30, 20), QColor("#ff0000"), pen_width=1.0)
    item.setPos(10, 10)
    original_rect = QRectF(item.rect())
    original_pos = item.pos()
    item.set_stroke_occlusions([QRectF(15, 8, 8, 8)])

    image = QImage(40, 30, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.translate(item.pos())
    item.paint(painter, QStyleOptionGraphicsItem())
    painter.end()
    app.processEvents()

    assert image.pixelColor(20, 10) == QColor("white")
    assert image.pixelColor(35, 10) == QColor("red")
    assert item.rect() == original_rect
    assert item.pos() == original_pos
