"""图像查看器：支持缩放、平移，以及 BBox 框叠加显示与编辑。

模式：
- view 模式（默认）：ScrollHandDrag，鼠标拖拽=平移
- edit 模式：RubberBandDrag，BBox 可拖动；移动后发出 block_moved
  - 编辑模式额外功能：选中框时出现 8 个缩放手柄（可拖拽四角/四边缩放）
  - 右键长按拖拽 → 画出新矩形 → 发出 block_created(BBox)
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QCursor
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView,
)

from app.models import BBox, Block, BlockType, Char


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


class _ResizeHandle(QGraphicsRectItem):
    """BBoxItem 的缩放手柄（8 个方向）。

    作为 BBoxItem 的子 Item，坐标系为父 item 的本地坐标（原点=bbox左上角）。
    拖拽时计算新 bbox 并更新父 item。
    """

    def __init__(self, pos: int, parent: "BBoxItem") -> None:
        super().__init__(-_HS, -_HS, _HS * 2, _HS * 2, parent)
        self._pos = pos
        self._bbox_item = parent
        pen = QPen(QColor("#1a73e8"), 1)
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
        bi.prepareGeometryChange()
        bi.setPos(r.topLeft())
        bi.setRect(QRectF(0, 0, r.width(), r.height()))
        for h in bi._handles:
            h.update_position()
        # 同步 block
        if bi._block is not None and hasattr(bi._block, "bbox"):
            bi._block.bbox = BBox(int(r.x()), int(r.y()), int(r.width()), int(r.height()))
            bi.signals.moved.emit(bi._block)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start = None
        self._orig_scene_rect = None
        event.accept()


# ── BBoxItem ──────────────────────────────────────────────────

class _BBoxSignals(QObject):
    """BBoxItem 内部信号代理（QGraphicsRectItem 不能多继承 QObject）。"""
    moved = Signal(object)  # payload = block


class BBoxItem(QGraphicsRectItem):
    """可选中、可拖动、可缩放的包围盒矩形。

    坐标方案：item.rect() 始终为 QRectF(0,0,w,h)，位置由 item.setPos(x,y) 决定。
    sceneBoundingRect() 返回的就是 block.bbox 在场景中的真实位置。
    """

    def __init__(self, rect: QRectF, color: QColor, label: str = "", parent=None):
        super().__init__(rect, parent)
        pen = QPen(color, 2)
        self.setPen(pen)
        self._label = label
        self._color = color
        self._block: Optional[object] = None
        self.signals = _BBoxSignals()
        self._handles: List[_ResizeHandle] = []
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
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        if editable:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()
        # 只在选中且可编辑时显示手柄
        show_handles = editable and self.isSelected()
        for h in self._handles:
            h.setVisible(show_handles)

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
            if self._block is not None and hasattr(self._block, "bbox"):
                pos = self.scenePos()
                r = self.rect()
                bb = self._block.bbox
                self._block.bbox = BBox(
                    int(pos.x() + r.x()),
                    int(pos.y() + r.y()),
                    bb.w,
                    bb.h,
                )
                self.signals.moved.emit(self._block)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # 选中时显示手柄，取消选中时隐藏
            editable = bool(self.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            show = bool(value) and editable
            for h in self._handles:
                h.setVisible(show)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            fill = QColor(self._color)
            fill.setAlpha(60)
            painter.fillRect(self.rect(), fill)
        super().paint(painter, option, widget)


class ImageViewer(QGraphicsView):
    """通用图像查看组件。"""

    block_clicked  = Signal(object)  # Block
    block_moved    = Signal(object)  # Block
    block_created  = Signal(object)  # BBox — 右键拖拽画出新矩形
    block_deleted  = Signal(object)  # Block — Delete 键删除选中框
    char_bbox_moved = Signal(object)  # Char

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._block_items: List[Tuple[BBoxItem, Block]] = []
        self._char_items: List[Tuple[BBoxItem, Char]] = []
        self._highlight_item = None  # highlight_bbox 使用

        # 右键拖拽画框状态
        self._draw_start: Optional[QPointF] = None   # scene 坐标
        self._draw_item: Optional[QGraphicsRectItem] = None

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._char_items.clear()
        pixmap = _pixmap_from_path(image_path)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_image_from_qimage(self, qimage: QImage) -> None:
        self._scene.clear()
        self._block_items.clear()
        self._char_items.clear()
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
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
            item.set_editable(True)  # 始终可编辑
            item.setData(0, block)
            item.signals.moved.connect(self.block_moved.emit)
            self._scene.addItem(item)
            self._block_items.append((item, block))

    def show_char_boxes(self, chars: List[Char]) -> None:
        """Overlay editable OCR char/token boxes on top of layout blocks."""
        for item, _ in self._char_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._char_items.clear()
        color = QColor("#ff8c00")
        for char in chars:
            if char.bbox is None or char.bbox.w <= 0 or char.bbox.h <= 0:
                continue
            bb = char.bbox
            item = BBoxItem(QRectF(0, 0, bb.w, bb.h), color, f"[char] {char.char or char.token_text}")
            item.setPos(bb.x, bb.y)
            item.set_block(char)
            item.set_editable(True)
            item.setZValue(8)
            item.signals.moved.connect(self.char_bbox_moved.emit)
            self._scene.addItem(item)
            self._char_items.append((item, char))

    def delete_selected(self) -> None:
        """删除所有选中的 BBoxItem，并 emit block_deleted 信号。"""
        to_remove = [
            (item, block) for item, block in self._block_items
            if item.isSelected()
        ]
        for item, block in to_remove:
            self._scene.removeItem(item)
            self._block_items.remove((item, block))
            self.block_deleted.emit(block)

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
        """兼容旧调用，始终保持可编辑状态。"""
        pass  # 始终 editable，无需切换

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

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # 右键开始画新框
            self._draw_start = self.mapToScene(event.pos())
            pen = QPen(QColor("#1a73e8"), 2, Qt.PenStyle.DashLine)
            self._draw_item = QGraphicsRectItem(
                QRectF(self._draw_start, self._draw_start)
            )
            self._draw_item.setPen(pen)
            self._draw_item.setBrush(QColor(26, 115, 232, 30))
            self._draw_item.setZValue(50)
            self._scene.addItem(self._draw_item)
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            for item, block in self._block_items:
                if item.sceneBoundingRect().contains(pos) and item.isSelected():
                    self.block_clicked.emit(block)
                    break

    def mouseMoveEvent(self, event):
        if self._draw_start is not None and self._draw_item is not None:
            cur = self.mapToScene(event.pos())
            self._draw_item.setRect(
                QRectF(self._draw_start, cur).normalized()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._draw_start is not None:
            cur = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start, cur).normalized()
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
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ private

    def _clear_overlays(self) -> None:
        for item, _ in self._block_items:
            self._scene.removeItem(item)
        self._block_items.clear()
        for item, _ in self._char_items:
            self._scene.removeItem(item)
        self._char_items.clear()
