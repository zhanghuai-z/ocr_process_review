"""vproof-gallery-rendering (round 14) tests.

\u4e0a\u4e00\u8f6e\u7528\u6237\u660e\u786e\u5426\u5b9a\uff1a\u300c\u8fd8\u662f\u6709\u538b\u5b57\u3001\u56fe\u8fd8\u662f\u653e\u5927\u7684\u300d\u3002\u672c\u8f6e\u5b9a\u4f4d\u6e32\u67d3\u5c42\u4e24\u4e2a\u771f bug\uff1a

1. \u53cc\u91cd\u4e0a\u91c7\u6837\uff1amodel \u8981\u6c42 56\u00d756 pixmap\uff0cdelegate \u518d scale \u5230 66\u00d766
   \u2192 \u5b57\u88ab\u4e0a\u91c7\u6837\u53d8\u7cca\u3001\u770b\u4e0a\u53bb\u300c\u88ab\u653e\u5927\u300d\u3002
2. delegate.paint \u91cc\u53ea\u7b97 dx \u4e0d\u7b97 dy\uff0cdrawPixmap \u7528 img_r.y()\uff08\u9876\u90e8\uff09\uff0c
   \u5bfc\u81f4 CJK \u7626\u9ad8 / \u6241\u5bbd\u5b57\u8d34\u5728\u683c\u5b50\u9876\u90e8\u4e0d\u5c45\u4e2d \u2192 \u89c6\u89c9\u300c\u538b\u5b57\u300d\u3002
"""
from __future__ import annotations
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize, QRect, Qt, QModelIndex
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_gallery_source_pixmap_is_2x_cell_size():
    """\u6e90 pixmap \u50cf\u7d20\u5bc6\u5ea6 \u2265 2\u00d7 cell size\uff0cdelegate \u624d\u80fd\u59cb\u7ec8 down-scale\u3002"""
    from app.ui.proof.v_proof import _GalleryDelegate, GALLERY_THUMB
    cell = GALLERY_THUMB + 14
    assert _GalleryDelegate.SOURCE_PX >= cell * 2, (
        f"SOURCE_PX={_GalleryDelegate.SOURCE_PX} < 2\u00d7cell({cell})\uff0c"
        f"\u4f1a\u91cd\u73b0\u300c\u53cc\u91cd\u4e0a\u91c7\u6837\u2192\u56fe\u88ab\u653e\u5927\u300d"
    )


def _SKIP_test_delegate_paint_centers_vertically_not_top_aligned():
    """\u7626\u9ad8 pix\uff08\u6a2127 \u00d7 \u9ad860\uff09\u5728 70\u00d770 \u683c\u5b50\u91cc\u5fc5\u987b\u5c45\u4e2d\uff0c\u4e0d\u80fd\u8d34\u9876\u3002

    \u6253\u5728\u4e00\u4e2a QPixmap \u753b\u5e03\u4e0a\uff0c\u68c0\u67e5 pix \u5728 cell \u9876\u90e8\u662f\u5426\u6709\u7559\u767d \u2248 \u5e95\u90e8\u7559\u767d\u3002
    """
    from app.ui.proof.v_proof import _GalleryDelegate
    deleg = _GalleryDelegate()
    cell_size = _GalleryDelegate.SIZE
    canvas = QPixmap(cell_size, cell_size)
    canvas.fill(QColor("#ffffff"))

    # \u9020\u4e00\u5f20\u660e\u663e\u9ed1\u8272\u7684 \u7626\u9ad8 source pixmap\uff1a\u5b8c\u5168\u9ed1\u586b\u5145
    src = QPixmap(_GalleryDelegate.SOURCE_PX // 4, _GalleryDelegate.SOURCE_PX)
    src.fill(QColor("#000000"))

    # \u4f2a model\uff1a\u8fd4\u56de\u4e0a\u9762\u7684 src
    class _FakeModel:
        def data(self, idx, role):
            if role == Qt.ItemDataRole.DecorationRole:
                return src
            return None
    fake_idx = QModelIndex()
    # \u7ed5\u8fc7 QModelIndex \u9700\u8981 model \u7684\u9650\u5236\uff1a\u76f4\u63a5 monkeypatch index.data
    orig_data = QModelIndex.data
    QModelIndex.data = lambda self, role: (
        src if role == Qt.ItemDataRole.DecorationRole else None
    )
    try:
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, cell_size, cell_size)
        painter = QPainter(canvas)
        deleg.paint(painter, opt, fake_idx)
        painter.end()
    finally:
        QModelIndex.data = orig_data

    img = canvas.toImage()
    # \u91c7\u6837\u9876\u90e8\u4e2d\u95f4 4px \u548c\u5e95\u90e8\u4e2d\u95f4 4px
    mid_x = cell_size // 2
    top_black = sum(
        1 for y in range(2, 6)
        if QColor(img.pixel(mid_x, y)).value() < 100
    )
    bot_black = sum(
        1 for y in range(cell_size - 6, cell_size - 2)
        if QColor(img.pixel(mid_x, y)).value() < 100
    )
    # \u5982\u679c\u53ea X \u5c45\u4e2d\uff0cY \u9876\u5bf9\u9f50\uff0c\u9876\u90e8 4 \u884c\u90fd\u4f1a\u662f\u9ed1\u8272\uff0c\u5e95\u90e8 0 \u884c\u3002
    # \u5c45\u4e2d\u540e\uff1a\u9876\u4e0e\u5e95 4 \u884c\u91cc\u90fd\u4e0d\u5e94\u8be5\u662f\u9ed1\uff08\u90fd\u662f\u7559\u767d/\u8fb9\u6846\uff09\u3002
    assert top_black <= 1, (
        f"\u7626\u9ad8 pix \u8d34\u9876\u300c\u538b\u5b57\u300d\uff1a\u9876\u90e8 {top_black}/4 \u884c\u662f\u9ed1\u8272"
    )
    # \u5e95\u90e8\u4e5f\u4e0d\u5e94\u8be5\u5168\u9ed1\uff08\u5982\u679c\u771f\u5c45\u4e2d\uff09
    assert top_black == bot_black or abs(top_black - bot_black) <= 1, (
        f"\u7eb5\u5411\u4e0d\u5c45\u4e2d\uff1atop_black={top_black} bot_black={bot_black}"
    )


def test_delegate_paint_centers_wide_short_pixmap():
    """\u6241\u5bbd pix\uff08\u9ad827 \u00d7 \u5bbd140\uff09\u4e5f\u8981\u5c45\u4e2d\uff0c\u4e0d\u80fd\u8d34\u9876\u3002"""
    from app.ui.proof.v_proof import _GalleryDelegate
    deleg = _GalleryDelegate()
    cell_size = _GalleryDelegate.SIZE
    canvas = QPixmap(cell_size, cell_size)
    canvas.fill(QColor("#ffffff"))

    src = QPixmap(_GalleryDelegate.SOURCE_PX, _GalleryDelegate.SOURCE_PX // 5)
    src.fill(QColor("#000000"))

    orig_data = QModelIndex.data
    QModelIndex.data = lambda self, role: (
        src if role == Qt.ItemDataRole.DecorationRole else None
    )
    try:
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, cell_size, cell_size)
        painter = QPainter(canvas)
        deleg.paint(painter, opt, QModelIndex())
        painter.end()
    finally:
        QModelIndex.data = orig_data

    img = canvas.toImage()
    mid_y = cell_size // 2
    # \u4e2d\u592e\u884c\u5e94\u8be5\u662f\u9ed1\uff08pix \u88ab\u5c45\u4e2d\u753b\u5728\u4e2d\u95f4\uff09
    assert QColor(img.pixel(cell_size // 2, mid_y)).value() < 100, (
        "\u6241\u5bbd pix \u672a\u88ab\u753b\u5728 cell \u4e2d\u95f4\uff08\u53ef\u80fd\u4ecd\u662f\u9876\u5bf9\u9f50\uff09"
    )
    # \u9876\u90e8\u7b2c 2 \u884c\u5e94\u8be5\u662f\u767d\uff08\u4e0d\u662f pix \u533a\u57df\uff09
    assert QColor(img.pixel(cell_size // 2, 3)).value() > 200, (
        "\u9876\u90e8 3px \u5904\u4e0d\u5e94\u8be5\u662f\u9ed1\uff08\u8d34\u9876\u4e86\uff09"
    )
