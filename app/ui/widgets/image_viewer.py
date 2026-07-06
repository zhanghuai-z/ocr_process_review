"""图像查看器：支持缩放、平移，以及 BBox 框叠加显示与编辑。

模式：
- edit 模式（默认）：BBox 可拖动/缩放；Shift+左键拖拽画出新矩形。
- 右键拖拽：按框线选中 BBox；框内部是镂空区域，不触发选中。
- pan 模式：按住 Space 临时进入平移，期间 BBox 不可编辑。
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPainterPath, QPainterPathStroker,
    QPen, QPixmap, QCursor,
)
from PySide6.QtWidgets import (
    QApplication, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView,
)

from app.core.block_attributes import block_attributes
from app.core.proof_char_text import char_display_text
from app.models import BBox, Block, BlockType, Char
from app.models.layout_block_view import LayoutBlockView
from app.models.ocr_character_observation import set_ocr_char_bbox
from app.models.ocr_observation import block_ocr_line_observations
from app.models.ocr_text_observation import line_ocr_confidence


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
_CHAR_BOX_COLOR = QColor("#ff8c00")
_FORMULA_CHAR_BOX_COLOR = QColor("#2563eb")
_CHAR_BOX_Z = 16


def _block_observation_avg_confidence(block: Block) -> float:
    lines = block_ocr_line_observations(block)
    if not lines:
        return 0.0
    return sum(line_ocr_confidence(line) for line in lines) / len(lines)

# ── 缩放手柄 ──────────────────────────────────────────────────

_TL, _TM, _TR, _ML, _MR, _BL, _BM, _BR = range(8)

_HANDLE_CURSORS = {
    _TL: Qt.CursorShape.SizeFDiagCursor,
    _TR: Qt.CursorShape.SizeBDiagCursor,
    _BL: Qt.CursorShape.SizeBDiagCursor,
    _BR: Qt.CursorShape.SizeFDiagCursor,
    _TM: Qt.CursorShape.SizeVerCursor,
    _BM: Qt.CursorShape.SizeVerCursor,
    _ML: Qt.CursorShape.SizeHorCursor,
    _MR: Qt.CursorShape.SizeHorCursor,
}
_HS = 7  # handle half-size in pixels
_FRAME_HIT_TOLERANCE = 6.0


class _ResizeHandle(QGraphicsRectItem):
    """BBoxItem 的缩放手柄（8 个方向）。

    作为 BBoxItem 的子 Item，坐标系为父 item 的本地坐标（原点=bbox左上角）。
    拖拽时计算新 bbox 并更新父 item。
    """

    def __init__(self, pos: int, parent: "BBoxItem") -> None:
        super().__init__(-_HS, -_HS, _HS * 2, _HS * 2, parent)
        self._pos = pos
        self._bbox_item = parent
        pen = QPen(QColor("#2C2C2C"), 1)
        self.setPen(pen)
        self.setBrush(QColor("#ffffff"))
        self.setZValue(20)
        self.setCursor(_HANDLE_CURSORS[pos])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self._drag_start: Optional[QPointF] = None   # scene coords
        self._orig_scene_rect: Optional[QRectF] = None

    # ------------------------------------------------------------------

    def update_position(self) -> None:
        """把手柄移到父矩形对应的角/边中心（父本地坐标）。"""
        r = self._bbox_item.rect()
        cx = r.width() / 2
        cy = r.height() / 2
        p = self._pos
        if   p == _TL: self.setPos(r.left(),  r.top())
        elif p == _TM: self.setPos(cx,        r.top())
        elif p == _TR: self.setPos(r.right(), r.top())
        elif p == _ML: self.setPos(r.left(),  cy)
        elif p == _MR: self.setPos(r.right(), cy)
        elif p == _BL: self.setPos(r.left(),  r.bottom())
        elif p == _BM: self.setPos(cx,        r.bottom())
        elif p == _BR: self.setPos(r.right(), r.bottom())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.scenePos()
            # 记录父 item 在场景中的原始矩形
            bi = self._bbox_item
            bi._emit_edit_started_once()
            sp = bi.scenePos()
            r  = bi.rect()
            self._orig_scene_rect = QRectF(
                sp.x() + r.x(), sp.y() + r.y(), r.width(), r.height()
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None or self._orig_scene_rect is None:
            return
        delta = event.scenePos() - self._drag_start
        dx, dy = delta.x(), delta.y()
        r = QRectF(self._orig_scene_rect)
        p = self._pos
        if p in (_TL, _TM, _TR): r.setTop(r.top()    + dy)
        if p in (_BL, _BM, _BR): r.setBottom(r.bottom() + dy)
        if p in (_TL, _ML, _BL): r.setLeft(r.left()   + dx)
        if p in (_TR, _MR, _BR): r.setRight(r.right()  + dx)
        r = r.normalized()
        if r.width() < 8 or r.height() < 8:
            return
        # 更新父 item
        bi = self._bbox_item
        bi._suppress_geometry_emit = True
        try:
            bi.prepareGeometryChange()
            bi.setPos(r.topLeft())
            bi.setRect(QRectF(0, 0, r.width(), r.height()))
            for h in bi._handles:
                h.update_position()
        finally:
            bi._suppress_geometry_emit = False
        bi._emit_geometry_changed(BBox(int(r.x()), int(r.y()), int(r.width()), int(r.height())))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start = None
        self._orig_scene_rect = None
        self._bbox_item._edit_started_for_drag = False
        event.accept()


# ── BBoxItem ──────────────────────────────────────────────────

class _BBoxSignals(QObject):
    """BBoxItem 内部信号代理（QGraphicsRectItem 不能多继承 QObject）。"""
    edit_started = Signal(str)  # payload = block_uid
    moved = Signal(object)  # payload = block
    geometry_changed = Signal(str, object)  # payload = block_uid, BBox


class BBoxItem(QGraphicsRectItem):
    """可选中、可拖动、可缩放的包围盒矩形。

    坐标方案：item.rect() 始终为 QRectF(0,0,w,h)，位置由 item.setPos(x,y) 决定。
    sceneBoundingRect() 返回的就是 block.bbox 在场景中的真实位置。
    """

    def __init__(
        self,
        rect: QRectF,
        color: QColor,
        label: str = "",
        parent=None,
        *,
        pen_width: float = 2.0,
    ):
        super().__init__(rect, parent)
        pen = QPen(color)
        pen.setWidthF(pen_width)
        self.setPen(pen)
        self._label = label
        self._color = color
        self._block: Optional[object] = None
        self.signals = _BBoxSignals()
        self._handles: List[_ResizeHandle] = []
        self._editable = True
        self._selectable = True
        self._edit_started_for_drag = False
        self._suppress_geometry_emit = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._update_tooltip()
        # 初始创建 8 个手柄（隐藏状态）
        for pos in range(8):
            h = _ResizeHandle(pos, self)
            h.setVisible(False)
            h.update_position()
            self._handles.append(h)

    def set_block(self, block: object) -> None:
        self._block = block

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        self._refresh_mouse_acceptance()
        if editable:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()
        # 只在选中且可编辑时显示手柄
        show_handles = editable and self.isSelected()
        for h in self._handles:
            h.setVisible(show_handles)

    def is_editable(self) -> bool:
        return self._editable

    def set_selectable(self, selectable: bool) -> None:
        self._selectable = selectable
        if not selectable and self.isSelected():
            self.setSelected(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, selectable)
        self._refresh_mouse_acceptance()

    def _refresh_mouse_acceptance(self) -> None:
        buttons = Qt.MouseButton.LeftButton if (self._selectable or self._editable) else Qt.MouseButton.NoButton
        self.setAcceptedMouseButtons(buttons)

    def _update_tooltip(self) -> None:
        r = self.sceneBoundingRect()
        coord = f"x={int(r.x())} y={int(r.y())} w={int(r.width())} h={int(r.height())}"
        if self._label:
            self.setToolTip(f"{self._label}\n{coord}")
        else:
            self.setToolTip(coord)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self._emit_edit_started_once()
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_tooltip()
            if not self._suppress_geometry_emit:
                self._emit_geometry_changed(self._scene_bbox())
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # 选中时显示手柄，取消选中时隐藏
            editable = bool(self.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            show = bool(value) and editable
            for h in self._handles:
                h.setVisible(show)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._edit_started_for_drag = False
        super().mouseReleaseEvent(event)

    def _emit_edit_started_once(self) -> None:
        if not self._editable or not self._selectable or self._block is None or self._edit_started_for_drag:
            return
        if isinstance(self._block, Block) and self._block.uid:
            self._edit_started_for_drag = True
            self.signals.edit_started.emit(self._block.uid)

    def _scene_bbox(self) -> BBox:
        pos = self.scenePos()
        rect = self.rect()
        return BBox(
            int(pos.x() + rect.x()),
            int(pos.y() + rect.y()),
            int(rect.width()),
            int(rect.height()),
        )

    def _emit_geometry_changed(self, bbox: BBox) -> None:
        if self._block is None or not hasattr(self._block, "bbox"):
            return
        if isinstance(self._block, Block):
            if self._block.uid:
                self.signals.geometry_changed.emit(self._block.uid, bbox)
            return
        if isinstance(self._block, Char):
            set_ocr_char_bbox(self._block, bbox)
            self.signals.moved.emit(self._block)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        stroker = QPainterPathStroker()
        stroker.setWidth(max(_FRAME_HIT_TOLERANCE * 2, self.pen().widthF() * 4))
        return stroker.createStroke(path)

    def contains(self, point: QPointF) -> bool:
        return self.shape().contains(point)

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            fill = QColor(self._color)
            fill.setAlpha(60)
            painter.fillRect(self.rect(), fill)
        super().paint(painter, option, widget)


class ImageViewer(QGraphicsView):
    """通用图像查看组件。"""

    block_clicked_uid = Signal(str)
    block_edit_started_uid = Signal(str)  # 用于上层在几何变化前记录撤销点
    block_geometry_change_requested = Signal(str, object)  # block_uid, BBox
    block_created  = Signal(object)  # BBox — Shift+左键拖拽画出新矩形
    block_deleted_uid = Signal(str)
    char_bbox_moved = Signal(object)  # Char

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._block_items: List[Tuple[BBoxItem, Block]] = []
        self._char_items: List[Tuple[BBoxItem, Char]] = []
        self._readonly_overlay_items: List[QGraphicsRectItem] = []
        self._highlight_item = None  # highlight_bbox 使用

        # Shift+左键拖拽画框状态
        self._draw_start: Optional[QPointF] = None   # scene 坐标
        self._draw_item: Optional[QGraphicsRectItem] = None
        self._selection_start: Optional[QPointF] = None
        self._selection_item: Optional[QGraphicsRectItem] = None
        self._bbox_snapper: Optional[Callable[[BBox], BBox]] = None
        self._edit_mode = True
        self._space_pan_active = False

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # ------------------------------------------------------------------ public

    def set_image(self, image_path: str) -> None:
        self._highlight_item = None
        self._scene.clear()
        self._block_items.clear()
        self._char_items.clear()
        self._readonly_overlay_items.clear()
        pixmap = _pixmap_from_path(image_path)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_image_from_qimage(self, qimage: QImage) -> None:
        self._highlight_item = None
        self._scene.clear()
        self._block_items.clear()
        self._char_items.clear()
        self._readonly_overlay_items.clear()
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self) -> None:
        """清空当前图像和所有叠加层。"""
        self._highlight_item = None
        self._draw_start = None
        self._draw_item = None
        self._selection_start = None
        self._selection_item = None
        self._pixmap_item = None
        self._block_items.clear()
        self._char_items.clear()
        self._readonly_overlay_items.clear()
        self._scene.clear()
        self._scene.setSceneRect(QRectF())
        self.resetTransform()

    def show_blocks(self, blocks: List[Block]) -> None:
        self._clear_overlays()
        for block in blocks:
            attrs = block_attributes(block)
            color = BLOCK_COLORS.get(attrs.semantic_block_type, BLOCK_COLORS[BlockType.UNKNOWN])
            label = f"[{attrs.display_label}] 置信度: {_block_observation_avg_confidence(block):.2f}"
            self._add_block_item(block, bbox=block.bbox, color=color, label=label)

    def show_layout_block_views(self, views: List[LayoutBlockView]) -> None:
        self._clear_overlays()
        for view in views:
            block = view.runtime_block
            if block is None:
                continue
            color = BLOCK_COLORS.get(view.block_type, BLOCK_COLORS[BlockType.UNKNOWN])
            display_label = view.source_label or getattr(view.block_type, "value", str(view.block_type))
            label = f"[{display_label}] 置信度: {_block_observation_avg_confidence(block):.2f}"
            self._add_block_item(block, bbox=view.bbox, color=color, label=label)

    def _add_block_item(
        self,
        block: Block,
        *,
        bbox: BBox,
        color: QColor,
        label: str,
    ) -> None:
        rect = QRectF(0, 0, bbox.w, bbox.h)
        item = BBoxItem(rect, color, label)
        item.setPos(bbox.x, bbox.y)
        item.set_block(block)
        item.set_selectable(True)
        item.set_editable(self._block_is_editable(block))
        item.setZValue(self._block_z_value(block))
        item.setData(0, block.uid)
        item.signals.edit_started.connect(self.block_edit_started_uid.emit)
        item.signals.geometry_changed.connect(self.block_geometry_change_requested.emit)
        self._scene.addItem(item)
        self._block_items.append((item, block))

    def show_readonly_overlays(self, overlays: List[Tuple[str, BBox]]) -> None:
        """Show read-only layout geometry that should not become editable blocks."""
        for item in self._readonly_overlay_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._readonly_overlay_items.clear()
        color = QColor("#d93025")
        for label, bbox in overlays:
            if bbox.w <= 0 or bbox.h <= 0:
                continue
            rect = QGraphicsRectItem(QRectF(0, 0, bbox.w, bbox.h))
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            rect.setPen(pen)
            rect.setBrush(QColor(217, 48, 37, 28))
            rect.setPos(bbox.x, bbox.y)
            rect.setZValue(4)
            rect.setToolTip(f"[{label}] read-only overlay\nx={bbox.x} y={bbox.y} w={bbox.w} h={bbox.h}")
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._scene.addItem(rect)
            self._readonly_overlay_items.append(rect)

    def show_char_boxes(self, chars: List[Char], *, editable: bool = True) -> None:
        """Overlay editable OCR char/token boxes on top of layout blocks."""
        for item, _ in self._char_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._char_items.clear()
        for char in chars:
            if char.bbox is None or char.bbox.w <= 0 or char.bbox.h <= 0:
                continue
            bb = char.bbox
            is_formula_carrier = char.bbox_source == "paddle_inline_formula"
            color = _FORMULA_CHAR_BOX_COLOR if is_formula_carrier else _CHAR_BOX_COLOR
            label = "[formula-token]" if is_formula_carrier else "[char]"
            item = BBoxItem(
                QRectF(0, 0, bb.w, bb.h),
                color,
                f"{label} {char_display_text(char)}",
                pen_width=1.0,
            )
            item.setPos(bb.x, bb.y)
            item.set_block(char)
            item.set_selectable(editable)
            item.set_editable(editable and self._edit_mode and not self._space_pan_active)
            item.setZValue(_CHAR_BOX_Z)
            item.signals.moved.connect(self.char_bbox_moved.emit)
            self._scene.addItem(item)
            self._char_items.append((item, char))

    def delete_selected(self) -> None:
        """删除所有选中的 BBoxItem，并 emit block_deleted_uid 信号。"""
        to_remove = [
            (item, block) for item, block in self._block_items
            if item.isSelected()
        ]
        for item, block in to_remove:
            self._scene.removeItem(item)
            self._block_items.remove((item, block))
            if block.uid:
                self.block_deleted_uid.emit(block.uid)

    def selected_block_uids(self) -> List[str]:
        return [
            block.uid for item, block in self._block_items
            if item.isSelected() and block.uid
        ]

    def set_bbox_snapper(self, snapper: Optional[Callable[[BBox], BBox]]) -> None:
        """Install a layout-level snap callback used while drawing new boxes."""
        self._bbox_snapper = snapper

    def select_block_uid(self, target_uid: str) -> bool:
        """Select one visible, editable layout block after overlays are rebuilt."""
        selected = False
        for item, block in self._block_items:
            should_select = bool(target_uid) and block.uid == target_uid
            item.setSelected(should_select)
            selected = selected or should_select
        return selected

    def highlight_bbox(self, bbox: BBox, *, zoom: bool = False) -> None:
        """高亮某个 BBox（橙色边框），并将其滚动到视野中心。用于纵校定位字符。"""
        # 清除旧的高亮
        if hasattr(self, "_highlight_item") and self._highlight_item is not None:
            if self._highlight_item.scene() is self._scene:
                self._scene.removeItem(self._highlight_item)
            self._highlight_item = None
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPen, QColor
        rect = QGraphicsRectItem(QRectF(bbox.x, bbox.y, bbox.w, bbox.h))
        pen = QPen(QColor("#FF8C00"), 4 if zoom else 2)
        rect.setPen(pen)
        rect.setBrush(QColor(255, 140, 0, 80 if zoom else 40))
        rect.setZValue(80 if zoom else 10)
        self._scene.addItem(rect)
        self._highlight_item = rect
        if zoom:
            pad = max(80, int(max(bbox.w, bbox.h) * 4))
            target = QRectF(
                bbox.x - pad,
                bbox.y - pad,
                bbox.w + pad * 2,
                bbox.h + pad * 2,
            ).intersected(self._scene.sceneRect())
            if target.isValid() and target.width() > 0 and target.height() > 0:
                self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(rect)

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
        """设置 BBox 编辑权限；Space 平移期间会临时关闭。"""
        self._edit_mode = bool(on)
        if self._space_pan_active:
            return
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        self._refresh_item_editability()

    def is_pan_mode_active(self) -> bool:
        return self._space_pan_active

    # ------------------------------------------------------------------ events

    def wheelEvent(self, event):
        mods = event.modifiers()
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()
        delta = dy or dx
        if mods & Qt.KeyboardModifier.ControlModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
        elif mods & Qt.KeyboardModifier.AltModifier:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
        else:
            factor = 1.15 if dy > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._enter_space_pan()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._leave_space_pan()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if (
            self._edit_mode
            and not self._space_pan_active
            and event.button() == Qt.MouseButton.LeftButton
            and self._event_has_shift(event)
        ):
            self._begin_draw(event)
            event.accept()
            return
        if self._edit_mode and not self._space_pan_active and event.button() == Qt.MouseButton.RightButton:
            self._begin_box_selection(event)
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            for item, block in self._block_items:
                if item.contains(item.mapFromScene(pos)) and item.isSelected():
                    if block.uid:
                        self.block_clicked_uid.emit(block.uid)
                    break

    def mouseMoveEvent(self, event):
        if self._draw_start is not None and self._draw_item is not None:
            cur = self._map_event_to_scene(event)
            self._draw_item.setRect(QRectF(self._draw_start, cur).normalized())
            event.accept()
            return
        if self._selection_start is not None and self._selection_item is not None:
            cur = self._map_event_to_scene(event)
            self._selection_item.setRect(QRectF(self._selection_start, cur).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_start is not None and event.button() == Qt.MouseButton.LeftButton:
            cur = self._map_event_to_scene(event)
            rect = self._snap_draw_rect(QRectF(self._draw_start, cur).normalized())
            # 移除临时画框
            if self._draw_item is not None:
                self._scene.removeItem(self._draw_item)
                self._draw_item = None
            self._draw_start = None
            # 最小尺寸过滤
            if rect.width() >= 10 and rect.height() >= 10:
                bbox = BBox(
                    int(rect.x()), int(rect.y()),
                    int(rect.width()), int(rect.height())
                )
                self.block_created.emit(bbox)
            event.accept()
            return
        if self._selection_start is not None and event.button() == Qt.MouseButton.RightButton:
            cur = self._map_event_to_scene(event)
            rect = QRectF(self._selection_start, cur).normalized()
            self._finish_box_selection(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ private

    def _clear_overlays(self) -> None:
        if self._highlight_item is not None and self._highlight_item.scene() is self._scene:
            self._scene.removeItem(self._highlight_item)
        self._highlight_item = None
        for item, _ in self._block_items:
            self._scene.removeItem(item)
        self._block_items.clear()
        for item, _ in self._char_items:
            self._scene.removeItem(item)
        self._char_items.clear()
        for item in self._readonly_overlay_items:
            self._scene.removeItem(item)
        self._readonly_overlay_items.clear()

    def _begin_draw(self, event) -> None:
        self._draw_start = self._map_event_to_scene(event)
        pen = QPen(QColor("#2C2C2C"), 2, Qt.PenStyle.DashLine)
        self._draw_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
        self._draw_item.setPen(pen)
        self._draw_item.setBrush(QColor(44, 44, 44, 28))
        self._draw_item.setZValue(50)
        self._scene.addItem(self._draw_item)

    def _begin_box_selection(self, event) -> None:
        self._selection_start = self._map_event_to_scene(event)
        pen = QPen(QColor("#5C6B58"), 2, Qt.PenStyle.DashLine)
        self._selection_item = QGraphicsRectItem(QRectF(self._selection_start, self._selection_start))
        self._selection_item.setPen(pen)
        self._selection_item.setBrush(QColor(92, 107, 88, 24))
        self._selection_item.setZValue(55)
        self._scene.addItem(self._selection_item)

    def _finish_box_selection(self, rect: QRectF) -> None:
        if self._selection_item is not None and self._selection_item.scene() is self._scene:
            self._scene.removeItem(self._selection_item)
        self._selection_item = None
        self._selection_start = None
        if rect.width() < 4 or rect.height() < 4:
            return
        selected_uids: list[str] = []
        for item, block in self._block_items:
            selected = self._selection_rect_hits_frame(rect, item)
            item.setSelected(selected)
            if selected and block.uid:
                selected_uids.append(block.uid)
        if len(selected_uids) == 1:
            self.block_clicked_uid.emit(selected_uids[0])

    @staticmethod
    def _selection_rect_hits_frame(selection: QRectF, item: BBoxItem) -> bool:
        outer = item.sceneBoundingRect()
        if not selection.intersects(outer):
            return False
        tol = max(_FRAME_HIT_TOLERANCE, item.pen().widthF() * 2)
        if outer.width() <= tol * 2 or outer.height() <= tol * 2:
            return selection.intersects(outer)
        bands = (
            QRectF(outer.left(), outer.top(), outer.width(), tol),
            QRectF(outer.left(), outer.bottom() - tol, outer.width(), tol),
            QRectF(outer.left(), outer.top(), tol, outer.height()),
            QRectF(outer.right() - tol, outer.top(), tol, outer.height()),
        )
        return any(selection.intersects(band) for band in bands)

    def _snap_draw_rect(self, rect: QRectF) -> QRectF:
        if self._bbox_snapper is None or rect.width() <= 0 or rect.height() <= 0:
            return rect
        bbox = BBox(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        snapped = self._bbox_snapper(bbox)
        return QRectF(snapped.x, snapped.y, snapped.w, snapped.h)

    @staticmethod
    def _event_has_shift(event) -> bool:
        event_mods = event.modifiers() if hasattr(event, "modifiers") else Qt.KeyboardModifier.NoModifier
        app_mods = QApplication.keyboardModifiers()
        return bool((event_mods | app_mods) & Qt.KeyboardModifier.ShiftModifier)

    def _map_event_to_scene(self, event) -> QPointF:
        if hasattr(event, "position"):
            return self.mapToScene(event.position().toPoint())
        return self.mapToScene(event.pos())

    def _enter_space_pan(self) -> None:
        self._space_pan_active = True
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._refresh_item_editability(force=False)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    def _leave_space_pan(self) -> None:
        self._space_pan_active = False
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if self._edit_mode
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self._refresh_item_editability()
        self.viewport().unsetCursor()

    def _block_is_editable(self, block: Block) -> bool:
        return self._edit_mode and not self._space_pan_active

    @staticmethod
    def _block_z_value(block: Block) -> int:
        attrs = block_attributes(block)
        if attrs.semantic_block_type == BlockType.EQUATION:
            return 14
        if attrs.semantic_block_type == BlockType.TABLE:
            return 12
        if attrs.semantic_block_type in (BlockType.FIGURE, BlockType.FIGURE_CAPTION, BlockType.TABLE_CAPTION):
            return 10
        return 8

    def _refresh_item_editability(self, *, force: Optional[bool] = None) -> None:
        for item, block in self._block_items:
            item.set_selectable(True)
            item.set_editable(force if force is not None else self._block_is_editable(block))
            item.setZValue(self._block_z_value(block))
        char_editable = force if force is not None else (self._edit_mode and not self._space_pan_active)
        for item, _char in self._char_items:
            item.set_editable(char_editable)
