"""Canvas overlay module for OCR Inspector.

Coordinate spaces:
  image_space  == scene_space (image placed at origin, 1:1 px, no scene scale)
  viewport_space — after pan/zoom QGraphicsView transform

Element source colours:
  overall_ocr_res   (行级 OCR)           — blue  #3C78FF
  parsing_res_list  (结构化块)           — green #00C850
  layout_det_res    (版面检测框)         — purple #A050FF
  text_word_region  (字/词级 bbox)       — orange #FFA000
  fallback / other                        — gray  #909090
"""
from __future__ import annotations
from typing import Any, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsTextItem, QGraphicsView,
)

from tools.ocr_inspector.models.ir import BBox, BlockNode, CharNode, LineNode, PageNode, Polygon
from tools.ocr_inspector.state import AppState

# ---------------------------------------------------------------------------
# Source-field colour palette
# ---------------------------------------------------------------------------
# Each source field gets its own colour so users can immediately see which
# JSON section each bbox came from.
SOURCE_COLOURS: dict[str, QColor] = {
    "overall_ocr_res":         QColor(60,  120, 255, 180),   # blue
    "overall_ocr_res.rec_texts": QColor(60, 120, 255, 180),  # same
    "parsing_res_list":        QColor(0,   200,  80, 180),   # green
    "layout_det_res":          QColor(160,  80, 255, 180),   # purple
    "text_word_region":        QColor(255, 160,   0, 200),   # orange
    "fallback":                QColor(140, 140, 140, 160),   # gray
}

_SELECTED_COLOUR = QColor(255, 230, 0, 230)
_POLYGON_COLOUR  = QColor(220,  50, 50, 160)
_LABEL_FONT      = QFont("Consolas", 7)
_LEGEND_FONT     = QFont("Consolas", 8)

_SOURCE_LABELS = {
    "overall_ocr_res":           "行级 OCR (overall_ocr_res)",
    "overall_ocr_res.rec_texts": "行级 OCR (overall_ocr_res)",
    "parsing_res_list":          "结构块 (parsing_res_list)",
    "layout_det_res":            "版面检测 (layout_det_res)",
    "text_word_region":          "字/词框 (text_word_region)",
    "fallback":                  "其他/回退",
}


def _source_colour(source_field: str) -> QColor:
    return SOURCE_COLOURS.get(source_field, SOURCE_COLOURS["fallback"])


# ---------------------------------------------------------------------------
# Graphics items
# ---------------------------------------------------------------------------

class _BBoxItem(QGraphicsRectItem):
    def __init__(self, bbox: BBox, node: Any, source_field: str, parent=None):
        super().__init__(bbox.x, bbox.y, bbox.w, bbox.h, parent)
        self._node = node
        self._source_field = source_field
        self._base_colour = _source_colour(source_field)
        pen = QPen(self._base_colour)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCursor(Qt.PointingHandCursor)
        # Tooltip
        kind = type(node).__name__
        text = getattr(node, "text", None) or getattr(node, "label", None) or getattr(node, "char", "")
        conf = getattr(node, "confidence", None)
        tip = f"{kind}  [{source_field}]\n"
        if text:
            tip += f"text: {str(text)[:60]}\n"
        if conf is not None:
            tip += f"conf: {conf:.3f}\n"
        tip += f"bbox: ({bbox.x:.0f}, {bbox.y:.0f}, {bbox.w:.0f}×{bbox.h:.0f})"
        self.setToolTip(tip)

    @property
    def ir_node(self) -> Any:
        return self._node

    def hoverEnterEvent(self, event):
        pen = QPen(_SELECTED_COLOUR)
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
            pen = QPen(_SELECTED_COLOUR)
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
    def __init__(self, poly: Polygon, node: Any, source_field: str, parent=None):
        pts = QPolygonF([QPointF(p[0], p[1]) for p in poly.points])
        super().__init__(pts, parent)
        self._node = node
        colour = _source_colour(source_field)
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


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _build_legend(scene: QGraphicsScene, visible_sources: set[str]) -> None:
    """Draw a colour legend in the top-left corner of the scene."""
    entries = [
        ("overall_ocr_res", "行级 OCR"),
        ("parsing_res_list", "结构化块"),
        ("layout_det_res", "版面检测"),
        ("text_word_region", "字/词框"),
        ("fallback", "其他/回退"),
    ]
    x0, y0 = 8, 8
    row_h = 16
    shown = [(sf, label) for sf, label in entries if sf in visible_sources]
    if not shown:
        return
    for i, (sf, label) in enumerate(shown):
        y = y0 + i * row_h
        colour = _source_colour(sf)
        # Colour swatch
        swatch = QGraphicsRectItem(x0, y, 12, 10)
        swatch.setPen(QPen(colour))
        swatch.setBrush(QBrush(colour))
        swatch.setZValue(20)
        swatch.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        scene.addItem(swatch)
        # Label
        txt = QGraphicsSimpleTextItem(label)
        txt.setFont(_LEGEND_FONT)
        txt.setBrush(QBrush(QColor(240, 240, 240)))
        txt.setPos(x0 + 16, y - 1)
        txt.setZValue(20)
        txt.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        scene.addItem(txt)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class OcrCanvas(QGraphicsView):
    """Image viewer with OCR bbox/polygon overlays, colour-coded by source field.

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
        visible_sources: set[str] = set()

        self._draw_layout_det(page, flags, visible_sources)
        self._draw_blocks(page, flags, visible_sources)
        self._draw_lines(page, flags, visible_sources)
        self._draw_chars(page, flags, visible_sources)

        if flags.get("legend", True):
            _build_legend(self._scene, visible_sources)

        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear(self) -> None:
        self._scene.clear()
        self._bbox_items = []
        self._selected_item = None

    # ── draw helpers ──────────────────────────────────────────────────────

    def _draw_layout_det(self, page: PageNode, flags: dict, visible_sources: set) -> None:
        """Draw layout_det_res blocks (detection only, dashed outline)."""
        if not flags.get("layout_det", False):
            return
        for block in page.layout_det_blocks:
            if block.bbox:
                colour = _source_colour("layout_det_res")
                item = _BBoxItem(block.bbox, block, "layout_det_res")
                item.setZValue(1)
                # Dashed pen to distinguish from structured blocks
                pen = QPen(colour)
                pen.setWidth(1)
                pen.setCosmetic(True)
                pen.setStyle(Qt.DashLine)
                item.setPen(pen)
                self._scene.addItem(item)
                self._bbox_items.append(item)
                if flags.get("labels") and block.label:
                    self._add_label(block.bbox, f"[det]{block.label}", colour, True)
                visible_sources.add("layout_det_res")

    def _draw_blocks(self, page: PageNode, flags: dict, visible_sources: set) -> None:
        visible = flags.get("blocks", True)
        for block in page.blocks:
            if block.bbox:
                sf = getattr(block, "source_field", "") or "parsing_res_list"
                colour = _source_colour(sf)
                item = _BBoxItem(block.bbox, block, sf)
                item.setZValue(2)
                item.setVisible(visible)
                self._scene.addItem(item)
                self._bbox_items.append(item)
                if flags.get("labels") and block.label and visible:
                    self._add_label(block.bbox, block.label, colour, True)
                visible_sources.add(sf)

    def _draw_lines(self, page: PageNode, flags: dict, visible_sources: set) -> None:
        visible = flags.get("lines", True)
        poly_vis = flags.get("polygons", False)
        for line in page.all_lines:
            sf = getattr(line, "source_field", "") or "overall_ocr_res"
            colour = _source_colour(sf)
            if line.bbox:
                item = _BBoxItem(line.bbox, line, sf)
                item.setZValue(3)
                item.setVisible(visible)
                self._scene.addItem(item)
                self._bbox_items.append(item)
                visible_sources.add(sf)
            if line.polygon and poly_vis:
                pi = _PolygonItem(line.polygon, line, sf)
                pi.setZValue(3)
                self._scene.addItem(pi)
                visible_sources.add(sf)

    def _draw_chars(self, page: PageNode, flags: dict, visible_sources: set) -> None:
        if not flags.get("chars", False):
            return
        seen: set = set()
        for char in page.all_chars:
            if char.bbox:
                key = (char.bbox.x, char.bbox.y, char.bbox.w, char.bbox.h)
                if key not in seen:
                    seen.add(key)
                    bs = getattr(char, "bbox_source", "") or "fallback"
                    sf = "text_word_region" if bs == "ocr" else "fallback"
                    item = _BBoxItem(char.bbox, char, sf)
                    item.setZValue(4)
                    self._scene.addItem(item)
                    self._bbox_items.append(item)
                    visible_sources.add(sf)

    def _add_label(self, bbox: BBox, text: str, colour: QColor, visible: bool) -> None:
        lbl = QGraphicsSimpleTextItem(text[:32])
        lbl.setFont(_LABEL_FONT)
        lbl.setBrush(QBrush(colour))
        lbl.setPos(bbox.x, bbox.y - 11)
        lbl.setZValue(10)
        lbl.setVisible(visible)
        self._scene.addItem(lbl)

    # ── state callbacks ──────────────────────────────────────────────────

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

    # ── input events ─────────────────────────────────────────────────────

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
