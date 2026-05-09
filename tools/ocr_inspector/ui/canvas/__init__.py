"""Canvas overlay module for OCR Inspector.

Coordinate spaces:
  image_space  == scene_space (image placed at origin, 1:1 px, no scene scale)
  viewport_space — after pan/zoom QGraphicsView transform
"""
from __future__ import annotations
from typing import Any, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

from tools.ocr_inspector.models.ir import BBox, BlockNode, CharNode, LineNode, PageNode, Polygon
from tools.ocr_inspector.state import AppState

_COLOURS = {
    "block":    QColor(60, 120, 255, 160),
    "line":     QColor(0, 200, 80, 180),
    "char":     QColor(255, 160, 0, 200),
    "polygon":  QColor(220, 50, 50, 160),
    "selected": QColor(255, 230, 0, 230),
}
_LABEL_FONT = QFont("Consolas", 8)


class _BBoxItem(QGraphicsRectItem):
    def __init__(self, bbox: BBox, node: Any, colour: QColor, parent=None):
        super().__init__(bbox.x, bbox.y, bbox.w, bbox.h, parent)
        self._node = node
        self._base_colour = colour
        pen = QPen(colour)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCursor(Qt.PointingHandCursor)

    @property
    def ir_node(self) -> Any:
        return self._node

    def hoverEnterEvent(self, event):
        pen = QPen(_COLOURS["selected"])
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        pen = QPen(self._base_colour)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
        super().hoverLeaveEvent(event)

    def highlight(self, on: bool) -> None:
        if on:
            pen = QPen(_COLOURS["selected"])
            pen.setWidth(2)
            pen.setCosmetic(True)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 230, 0, 40)))
        else:
            pen = QPen(self._base_colour)
            pen.setWidth(1)
            pen.setCosmetic(True)
            self.setPen(pen)
            self.setBrush(QBrush(Qt.NoBrush))


class _PolygonItem(QGraphicsPolygonItem):
    def __init__(self, poly: Polygon, node: Any, colour: QColor, parent=None):
        pts = QPolygonF([QPointF(p[0], p[1]) for p in poly.points])
        super().__init__(pts, parent)
        self._node = node
        pen = QPen(colour)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCursor(Qt.PointingHandCursor)

    @property
    def ir_node(self) -> Any:
        return self._node


class OcrCanvas(QGraphicsView):
    """Image viewer with OCR bbox/polygon overlays.

    Signals:
      node_clicked(node)           — IR node when user clicks a bbox
      coord_changed(ix, iy, sx, sy) — mouse coords in image/scene space
    """
    node_clicked = Signal(object)
    coord_changed = Signal(float, float, float, float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self.setMouseTracking(True)

        self._bbox_items: list[_BBoxItem] = []
        self._selected_item: Optional[_BBoxItem] = None

        state.on_page_changed(self._on_page_changed)
        state.on_selection_changed(self._on_selection_changed)
        state.on_overlay_changed(self._on_overlay_changed)

    def load_page(self, page: PageNode) -> None:
        self._scene.clear()
        self._bbox_items = []
        self._selected_item = None

        if page.image_path:
            pix = QPixmap(page.image_path)
            if not pix.isNull():
                img_item = self._scene.addPixmap(pix)
                img_item.setZValue(0)
                self._scene.setSceneRect(QRectF(pix.rect()))

        flags = self._state.overlay_flags
        self._draw_blocks(page, flags)
        self._draw_lines(page, flags)
        self._draw_chars(page, flags)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear(self) -> None:
        self._scene.clear()
        self._bbox_items = []
        self._selected_item = None

    def _draw_blocks(self, page: PageNode, flags: dict) -> None:
        visible = flags.get("blocks", True)
        for block in page.blocks:
            if block.bbox:
                item = _BBoxItem(block.bbox, block, _COLOURS["block"])
                item.setZValue(1)
                item.setVisible(visible)
                self._scene.addItem(item)
                self._bbox_items.append(item)
                if flags.get("labels") and block.label:
                    self._add_label(block.bbox, block.label, _COLOURS["block"], visible)

    def _draw_lines(self, page: PageNode, flags: dict) -> None:
        visible = flags.get("lines", True)
        poly_vis = flags.get("polygons", False)
        for line in page.all_lines:
            if line.bbox:
                item = _BBoxItem(line.bbox, line, _COLOURS["line"])
                item.setZValue(2)
                item.setVisible(visible)
                self._scene.addItem(item)
                self._bbox_items.append(item)
            if line.polygon and poly_vis:
                pi = _PolygonItem(line.polygon, line, _COLOURS["polygon"])
                pi.setZValue(2)
                self._scene.addItem(pi)

    def _draw_chars(self, page: PageNode, flags: dict) -> None:
        if not flags.get("chars", False):
            return
        seen: set = set()
        for char in page.all_chars:
            if char.bbox:
                key = (char.bbox.x, char.bbox.y, char.bbox.w, char.bbox.h)
                if key not in seen:
                    seen.add(key)
                    item = _BBoxItem(char.bbox, char, _COLOURS["char"])
                    item.setZValue(3)
                    self._scene.addItem(item)
                    self._bbox_items.append(item)

    def _add_label(self, bbox: BBox, text: str, colour: QColor, visible: bool) -> None:
        lbl = QGraphicsSimpleTextItem(text[:30])
        lbl.setFont(_LABEL_FONT)
        lbl.setBrush(QBrush(colour))
        lbl.setPos(bbox.x, bbox.y - 12)
        lbl.setZValue(10)
        lbl.setVisible(visible)
        self._scene.addItem(lbl)

    def _on_page_changed(self, page) -> None:
        if page is not None:
            self.load_page(page)
        else:
            self.clear()

    def _on_selection_changed(self, node) -> None:
        if self._selected_item is not None:
            self._selected_item.highlight(False)
            self._selected_item = None
        if node is None:
            return
        for item in self._bbox_items:
            if item.ir_node is node:
                item.highlight(True)
                self._selected_item = item
                self.ensureVisible(item)
                break

    def _on_overlay_changed(self, key: str, value: bool) -> None:
        page = self._state.active_page
        if page is not None:
            self.load_page(page)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            for item in self._scene.items(scene_pos):
                if isinstance(item, (_BBoxItem, _PolygonItem)):
                    self._state.set_selection(item.ir_node)
                    self.node_clicked.emit(item.ir_node)
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.position().toPoint())
        self.coord_changed.emit(sp.x(), sp.y(), sp.x(), sp.y())
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
