"""页面图像 LRU 缓存：避免重复磁盘 IO。

用法：
    from app.core.page_image_cache import PageImageCache

    cache = PageImageCache.instance()
    img_bgr = cache.get_page_image(page_path)        # np.ndarray | None
    crop = cache.get_bbox_crop(page_path, bbox)       # np.ndarray | None
    pix = cache.get_char_crop(page_path, bbox, 56)    # QPixmap | None
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

from app.core.coordinate_seam import (
    BBOX_SPACE_PAGE, BBoxSpace, CropCoordinateSeam,
)
from app.core.char_bbox_utils import refine_line_bbox
from app.models import BBox

logger = logging.getLogger(__name__)


class PageImageCache:
    """单例 LRU 页面图像缓存。"""

    _instance: "PageImageCache | None" = None

    def __init__(self, max_pages: int = 8) -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_pages = max_pages

    # ------------------------------------------------------------------ 单例

    @classmethod
    def instance(cls) -> "PageImageCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """测试用：重置单例并清空缓存。"""
        cls._instance = None

    # ------------------------------------------------------------------ 基础图像

    def get_page_image(self, page_path: str) -> Optional[np.ndarray]:
        """返回 BGR ndarray，失败时返回 None。"""
        if not page_path:
            return None
        cached = self._cache.get(page_path)
        if cached is not None:
            self._cache.move_to_end(page_path)
            return cached

        img = cv2.imread(page_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("PageImageCache: cannot read %s", page_path)
            return None

        self._cache[page_path] = img
        self._cache.move_to_end(page_path)
        if len(self._cache) > self._max_pages:
            self._cache.popitem(last=False)
        return img

    def get_image(self, path: str) -> Optional[np.ndarray]:
        """旧接口别名。"""
        return self.get_page_image(path)

    def get_bbox_crop(
        self,
        page_path: str,
        bbox: BBox,
        *,
        source_space: BBoxSpace = BBOX_SPACE_PAGE,
        seam: CropCoordinateSeam | None = None,
    ) -> Optional[np.ndarray]:
        """返回 bbox 对应的 BGR 裁图。

        若提供 seam，则按 seam 将 crop-space bbox 回写到 page-space。
        """
        image = self.get_page_image(page_path)
        if image is None:
            return None

        if seam is not None:
            clamped = seam.to_page_bbox(bbox, source_space=source_space)
        else:
            clamped = bbox.normalize().clamp(image.shape[1], image.shape[0])

        if clamped.w <= 0 or clamped.h <= 0:
            return None
        return image[clamped.y:clamped.y2, clamped.x:clamped.x2].copy()

    def get_char_crop(
        self,
        page_path: str,
        bbox: BBox,
        size: int = 56,
        pad: int = 2,
    ):
        """裁剪字符区域并缩放为 QPixmap（正方形）。"""
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QImage, QPixmap
        except ImportError:
            return None

        img = self.get_page_image(page_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, bbox.x - pad)
        y1 = max(0, bbox.y - pad)
        x2 = min(w, bbox.x + bbox.w + pad)
        y2 = min(h, bbox.y + bbox.h + pad)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = np.ascontiguousarray(img[y1:y2, x1:x2])
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), cw, ch, cw * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def get_line_crop(
        self,
        page_path: str,
        bbox: BBox,
        pad_y: int = 6,
    ) -> Optional[np.ndarray]:
        """裁剪整行 BGR ndarray（含上下额外像素）。"""
        image = self.get_page_image(page_path)
        if image is None:
            return None
        refined = refine_line_bbox(bbox, image)
        h, w = image.shape[:2]
        x1 = max(0, refined.x)
        y1 = max(0, refined.y - pad_y)
        x2 = min(w, refined.x2)
        y2 = min(h, refined.y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    def cached_paths(self) -> list[str]:
        return list(self._cache.keys())

    def invalidate(self, path: str) -> None:
        """手动淘汰某页缓存（页面重新识别后调用）。"""
        self._cache.pop(path, None)

    def clear(self) -> None:
        """清空全部缓存。"""
        self._cache.clear()
