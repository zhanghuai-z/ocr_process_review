from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem

from app.models import BBox, Block, BlockType, Char
from app.ui.widgets.image_viewer import BBoxItem, ImageViewer


def test_block_frame_is_drawn_outside_content_without_changing_geometry():
    app = QApplication.instance() or QApplication([])
    item = BBoxItem(
        QRectF(0, 0, 30, 20),
        QColor("#ff0000"),
        pen_width=2.0,
        stroke_outside=True,
    )
    item.setPos(10, 10)
    item.set_stroke_occlusions([QRectF(15, 25, 8, 8)])
    original_rect = QRectF(item.rect())
    original_pos = item.pos()

    image = QImage(50, 40, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.translate(item.pos())
    item.paint(painter, QStyleOptionGraphicsItem())
    painter.end()
    app.processEvents()

    assert image.pixelColor(20, 10) == QColor("white")
    assert image.pixelColor(20, 8) == QColor("red")
    assert image.pixelColor(20, 32) == QColor("white")
    assert image.pixelColor(35, 32) == QColor("red")
    assert item.rect() == original_rect
    assert item.pos() == original_pos


def test_viewer_uses_outside_stroke_only_for_layout_blocks():
    QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(80, 60, QImage.Format.Format_RGB888))
    viewer.show_blocks([Block(BlockType.TEXT, BBox(10, 10, 30, 20))])
    viewer.show_char_boxes([Char(char="字", confidence=0.9, bbox=BBox(12, 10, 12, 18))])

    assert viewer._block_items[0][0]._stroke_outside is True
    assert viewer._char_items[0][0]._stroke_outside is False
    viewer.close()


def test_block_frame_occlusions_do_not_depend_on_visible_char_boxes():
    QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(80, 60, QImage.Format.Format_RGB888))
    viewer.show_blocks([Block(BlockType.TEXT, BBox(10, 10, 30, 20))])
    char = Char(char="字", confidence=0.9, bbox=BBox(12, 8, 12, 18))

    viewer.set_block_frame_occlusions([char])

    assert viewer._char_items == []
    assert viewer._block_items[0][0]._stroke_occlusions
    viewer.close()


def test_block_frame_precomputes_only_edge_occlusions():
    QApplication.instance() or QApplication([])
    item = BBoxItem(
        QRectF(0, 0, 100, 80),
        QColor("#ff0000"),
        stroke_outside=True,
    )
    item.setPos(10, 10)

    item.set_stroke_occlusions([
        QRectF(25, 8, 10, 12),   # crosses the top frame
        QRectF(40, 40, 10, 10),  # wholly inside the block
        QRectF(200, 200, 10, 10),
    ])

    assert len(item._stroke_occlusions) == 1
    assert item._stroke_clip_path is not None

    cached_path = item._stroke_clip_path
    image = QImage(130, 110, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.translate(item.pos())
    item.paint(painter, QStyleOptionGraphicsItem())
    item.paint(painter, QStyleOptionGraphicsItem())
    painter.end()

    assert item._stroke_clip_path is cached_path


def test_block_frame_rebuilds_occlusions_after_geometry_edit():
    QApplication.instance() or QApplication([])
    item = BBoxItem(
        QRectF(0, 0, 100, 80),
        QColor("#ff0000"),
        stroke_outside=True,
    )
    item.setPos(10, 10)
    item.set_stroke_occlusions([QRectF(25, 8, 10, 12)])
    original_path = item._stroke_clip_path

    item._begin_geometry_edit()
    item.setPos(12, 12)
    item.setRect(QRectF(0, 0, 120, 90))
    assert item._stroke_clip_path is None

    item._end_geometry_edit()
    assert item._stroke_clip_path is not None
    assert item._stroke_clip_path is not original_path
    assert len(item._stroke_occlusions) == 1
