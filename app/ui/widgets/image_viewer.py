"""图像查看器：支持缩放、平移，以及 BBox 框叠加显示与编辑。

模式：
- view 模式（默认）：ScrollHandDrag，鼠标拖拽=平移
- edit 模式：RubberBandDrag，BBox 可拖动；移动后发出 block_moved
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView,
)

from app.core.build_info import BUILD_MARKER
from app.core.logging import get_logger
from app.models import BBox, Block, BlockType

logger = get_logger(__name__)


def _pixmap_from_path(image_path: str) -> QPixmap:
    """通过 cv2 加载图片并转为 QPixmap，避免 EXIF 自动旋转造成的坐标错位。"""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return QPixmap(image_path)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# 各块类型对应的边框颜色

BLOCK_COLORS: dict[BlockType, QColor] = {
    BlockType.TEXT:           QColor(0x4C, 0xAF, 0x50),
    BlockType.TITLE:          QColor(0x21, 0x96, 0xF3),
    BlockType.FIGURE:         QColor(0xFF, 0x98, 0x00),
    BlockType.FIGURE_CAPTION: QColor(0xFF, 0xC1, 0x07),
    BlockType.TABLE:          QColor(0x9C, 0x27, 0xB0),
    BlockType.TABLE_CAPTION:  QColor(0xE0, 0x91, 0xFF),
    BlockType.REFERENCE:      QColor(0x00, 0xBC, 0xD4),
    BlockType.EQUATION:       QColor(0xF4, 0x43, 0x36),
    BlockType.UNKNOWN:        QColor(0x9E, 0x9E, 0x9E),
}

_LINE_HIGHLIGHT = QColor(0xFF, 0x57, 0x22, 160)
_LINE_OK_COLOR  = QColor(0x4C, 0xAF, 0x50, 100)


class _BBoxSignals(QObject):
    """BBoxItem 内部信号代理（QGraphicsRectItem 不能多继承 QObject）。"""
    moved = Signal(object)  # payload = block


class BBoxItem(QGraphicsRectItem):
    """可选中、可拖动的包围盒矩形。

    坐标方案：item.rect() 始终为 QRectF(0,0,w,h)，位置由 item.setPos(x,y) 决定。
    sceneBoundingRect() 返回的就是 block.bbox 在场景中的真实位置。
    """

    def __init__(self, rect: QRectF, color: QColor, label: str = "", parent=None):
        super().__init__(rect, parent)
        pen = QPen(color, 2)
        self.setPen(pen)
        self._label = label
        self._color = color
        self._block: Optional[Block] = None
        self.signals = _BBoxSignals()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._update_tooltip()

    def set_block(self, block: Block) -> None:
        self._block = block

    def set_editable(self, editable: bool) -> None:
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        if editable:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()

    def _update_tooltip(self) -> None:
        r = self.sceneBoundingRect()
        coord = f"x={int(r.x())} y={int(r.y())} w={int(r.width())} h={int(r.height())}"
        if self._label:
            self.setToolTip(f"{self._label}\n{coord}")
        else:
            self.setToolTip(coord)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_tooltip()
            if self._block is not None:
                r = self.sceneBoundingRect()
                bb = self._block.bbox
                self._block.bbox = BBox(int(r.x()), int(r.y()), bb.w, bb.h)
                self.signals.moved.emit(self._block)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            fill = QColor(self._color)
            fill.setAlpha(60)
            painter.fillRect(self.rect(), fill)
        super().paint(painter, option, widget)


class ImageViewer(QGraphicsView):
    """通用图像查看组件。"""

    block_clicked = Signal(object)
    block_moved   = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._block_items: List[Tuple[BBoxItem, Block]] = []
        self._image_path: str = ""
        self._edit_mode: bool = False
        self._highlight_item = None  # highlight_bbox 使用

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # ------------------------------------------------------------------ public

    def set_image(self, image_path: str) -> None:
        self._scene.clear()
        self._block_items.clear()
        pixmap = _pixmap_from_path(image_path)
        pixmap.setDevicePixelRatio(1.0)
        self._image_path = image_path
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_image_from_qimage(self, qimage: QImage) -> None:
        self._scene.clear()
        self._block_items.clear()
        pixmap = QPixmap.fromImage(qimage)
        pixmap.setDevicePixelRatio(1.0)
        self._image_path = ""
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def show_blocks(self, blocks: List[Block]) -> None:
        self._clear_overlays()
        for block in blocks:
            color = BLOCK_COLORS.get(block.block_type, BLOCK_COLORS[BlockType.UNKNOWN])
            bb = block.bbox
            rect = QRectF(0, 0, bb.w, bb.h)
            label = f"[{block.block_type.value}] 置信度: {block.avg_confidence:.2f}"
            item = BBoxItem(rect, color, label)
            item.setPos(bb.x, bb.y)
            item.set_block(block)
            item.set_editable(self._edit_mode)
            item.setData(0, block)
            item.signals.moved.connect(self.block_moved.emit)
            self._scene.addItem(item)
            self._block_items.append((item, block))
        self._write_viewer_debug(blocks)

    def highlight_bbox(self, bbox: BBox) -> None:
        """高亮某个 BBox（橙色边框），并将其滚动到视野中心。用于纵校定位字符。"""
        # 清除旧的高亮
        if hasattr(self, "_highlight_item") and self._highlight_item is not None:
            if self._highlight_item.scene() is self._scene:
                self._scene.removeItem(self._highlight_item)
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPen, QColor
        rect = QGraphicsRectItem(QRectF(bbox.x, bbox.y, bbox.w, bbox.h))
        pen = QPen(QColor("#FF8C00"), 2)
        rect.setPen(pen)
        rect.setBrush(QColor(255, 140, 0, 40))
        rect.setZValue(10)
        self._scene.addItem(rect)
        self._highlight_item = rect
        # 滚动到该位置
        self.ensureVisible(rect)

    def show_line_highlight(self, bbox: BBox, flagged: bool = False) -> QGraphicsRectItem:
        color = _LINE_HIGHLIGHT if flagged else _LINE_OK_COLOR
        rect_item = QGraphicsRectItem(QRectF(bbox.x, bbox.y, bbox.w, bbox.h))
        pen = QPen(Qt.PenStyle.NoPen)
        rect_item.setPen(pen)
        rect_item.setBrush(color)
        rect_item.setZValue(1)
        self._scene.addItem(rect_item)
        return rect_item

    def clear_overlays(self) -> None:
        self._clear_overlays()

    def fit_to_window(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_edit_mode(self, on: bool) -> None:
        """切换编辑模式：开启后 BBox 可拖动，关闭后只能浏览/平移。"""
        self._edit_mode = bool(on)
        if self._edit_mode:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        for item, _ in self._block_items:
            item.set_editable(self._edit_mode)

    # ------------------------------------------------------------------ events

    def wheelEvent(self, event):
        mods = event.modifiers()
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - (dy or dx))
        else:
            factor = 1.15 if dy > 0 else 1 / 1.15
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            for item, block in self._block_items:
                if item.sceneBoundingRect().contains(pos) and item.isSelected():
                    self.block_clicked.emit(block)
                    break

    # ------------------------------------------------------------------ private

    def _clear_overlays(self) -> None:
        for item, _ in self._block_items:
            self._scene.removeItem(item)
        self._block_items.clear()

    def _rect_to_dict(self, rect: QRectF) -> dict[str, float]:
        return {
            "x": rect.x(),
            "y": rect.y(),
            "w": rect.width(),
            "h": rect.height(),
        }

    def _write_viewer_debug(self, blocks: List[Block]) -> None:
        if not self._image_path or not self._pixmap_item:
            return

        pixmap = self._pixmap_item.pixmap()
        transform = self.transform()
        payload = {
            "build": BUILD_MARKER,
            "image_path": self._image_path,
            "pixmap": {
                "width": pixmap.width(),
                "height": pixmap.height(),
                "device_pixel_ratio": pixmap.devicePixelRatio(),
                "item_bounding_rect": self._rect_to_dict(self._pixmap_item.boundingRect()),
            },
            "scene_rect": self._rect_to_dict(self._scene.sceneRect()),
            "viewport": {
                "width": self.viewport().width(),
                "height": self.viewport().height(),
                "transform_m11": transform.m11(),
                "transform_m22": transform.m22(),
                "horizontal_scroll": self.horizontalScrollBar().value(),
                "vertical_scroll": self.verticalScrollBar().value(),
            },
            "blocks": [
                {
                    "order": block.order,
                    "type": block.block_type.value,
                    "bbox": {
                        "x": block.bbox.x,
                        "y": block.bbox.y,
                        "w": block.bbox.w,
                        "h": block.bbox.h,
                        "xyxy": list(block.bbox.to_xyxy()),
                    },
                }
                for block in blocks
            ],
        }
        try:
            Path(self._image_path).with_suffix(".viewer-debug.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write viewer debug file: %s", exc)
