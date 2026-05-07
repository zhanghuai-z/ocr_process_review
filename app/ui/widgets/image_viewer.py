"""图像查看器：支持缩放、平移，以及 BBox 框叠加显示。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QRectF, Signal
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
    """通过 cv2 加载图片并转为 QPixmap。

    Qt6 的 QPixmap(path) 会根据 JPEG EXIF 自动旋转，
    但版面分析/OCR 均用 cv2.imread（不处理 EXIF）。
    统一使用 cv2 保证显示与分析坐标系一致，避免 BBox 偏移。
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return QPixmap(image_path)  # 路径无效时退回原生加载
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# 各块类型对应的边框颜色
BLOCK_COLORS: dict[BlockType, QColor] = {
    BlockType.TEXT:           QColor(0x4C, 0xAF, 0x50),  # 绿
    BlockType.TITLE:          QColor(0x21, 0x96, 0xF3),  # 蓝
    BlockType.FIGURE:         QColor(0xFF, 0x98, 0x00),  # 橙
    BlockType.FIGURE_CAPTION: QColor(0xFF, 0xC1, 0x07),  # 黄
    BlockType.TABLE:          QColor(0x9C, 0x27, 0xB0),  # 紫
    BlockType.TABLE_CAPTION:  QColor(0xE0, 0x91, 0xFF),  # 浅紫
    BlockType.REFERENCE:      QColor(0x00, 0xBC, 0xD4),  # 青
    BlockType.EQUATION:       QColor(0xF4, 0x43, 0x36),  # 红
    BlockType.UNKNOWN:        QColor(0x9E, 0x9E, 0x9E),  # 灰
}

_LINE_HIGHLIGHT = QColor(0xFF, 0x57, 0x22, 160)   # 低置信度行高亮（半透明红）
_LINE_OK_COLOR  = QColor(0x4C, 0xAF, 0x50, 100)   # 已确认行（半透明绿）


class BBoxItem(QGraphicsRectItem):
    """可选中的包围盒矩形。"""

    def __init__(self, rect: QRectF, color: QColor, label: str = "", parent=None):
        super().__init__(rect, parent)
        pen = QPen(color, 2)
        self.setPen(pen)
        self.setToolTip(label)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._color = color

    def paint(self, painter: QPainter, option, widget=None):
        # 选中时填充半透明背景
        if self.isSelected():
            fill = QColor(self._color)
            fill.setAlpha(40)
            painter.fillRect(self.rect(), fill)
        super().paint(painter, option, widget)


class ImageViewer(QGraphicsView):
    """
    通用图像查看组件。
    - Ctrl+滚轮 缩放
    - 中键/右键 平移
    - 支持叠加 Block 级别和 Line 级别 BBox
    """

    block_clicked = Signal(object)   # 点击某个 Block 时发出，payload=Block

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._block_items: List[Tuple[BBoxItem, Block]] = []
        self._image_path: str = ""

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # ------------------------------------------------------------------ public API

    def set_image(self, image_path: str) -> None:
        """加载图片到视图（经 cv2 加载，与分析坐标系一致）。"""
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
        """在图像上叠加版面块 BBox。"""
        self._clear_overlays()
        for block in blocks:
            color = BLOCK_COLORS.get(block.block_type, BLOCK_COLORS[BlockType.UNKNOWN])
            bb = block.bbox
            rect = QRectF(bb.x, bb.y, bb.w, bb.h)
            label = f"[{block.block_type.value}] 置信度: {block.avg_confidence:.2f}"
            item = BBoxItem(rect, color, label)
            item.setData(0, block)   # 存入 block 引用
            self._scene.addItem(item)
            self._block_items.append((item, block))
        self._write_viewer_debug(blocks)

    def show_line_highlight(self, bbox: BBox, flagged: bool = False) -> QGraphicsRectItem:
        """高亮单行，返回 item 以便外部移除。"""
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

    # ------------------------------------------------------------------ events

    def wheelEvent(self, event):
        mods = event.modifiers()
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()

        if mods & Qt.KeyboardModifier.ShiftModifier:
            # Shift+滚轮 → 水平滚动
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - (dy or dx))
        else:
            # 直接滚轮缩放（去掉 Ctrl 要求）
            factor = 1.15 if dy > 0 else 1 / 1.15
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            for item, block in self._block_items:
                if item.rect().contains(pos) and item.isSelected():
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
