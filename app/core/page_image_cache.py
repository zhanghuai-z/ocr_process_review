"""页面图像 LRU 缓存：避免重复磁盘 IO。

用法：
    from app.core.page_image_cache import PageImageCache

    cache = PageImageCache.instance()
    img_bgr = cache.get_image(page_path)          # np.ndarray | None
    pix = cache.get_char_crop(page_path, bbox, 56) # QPixmap | None
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PageImageCache:
    """单例 LRU 页面图像缓存。

    默认最多缓存 8 张页面的 BGR ndarray；超出则淘汰最旧的。
    """

    _instance: "PageImageCache | None" = None

    def __init__(self, max_pages: int = 8) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max = max_pages

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

    # ------------------------------------------------------------------ API

    def get_image(self, path: str) -> Optional[np.ndarray]:
        """返回 BGR ndarray，失败时返回 None。"""
        if not path:
            return None
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        img = cv2.imread(path)
        if img is None:
            logger.warning("PageImageCache: cannot read %s", path)
            return None
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[path] = img
        return img

    def get_char_crop(
        self,
        page_path: str,
        bbox,  # app.models.BBox
        size: int = 56,
        pad: int = 2,
    ):
        """裁剪字符区域并缩放为 QPixmap（正方形）。

        返回 QPixmap 或 None（需要 Qt，仅在 Qt 初始化后调用）。
        """
        # 延迟导入避免在纯 Python 测试环境中引入 Qt
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QImage, QPixmap
        except ImportError:
            return None

        img = self.get_image(page_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        x1 = max(0, bbox.x - pad)
        y1 = max(0, bbox.y - pad)
        x2 = min(W, bbox.x + bbox.w + pad)
        y2 = min(H, bbox.y + bbox.h + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = np.ascontiguousarray(img[y1:y2, x1:x2])
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def get_line_crop(
        self,
        page_path: str,
        bbox,  # app.models.BBox
        pad_y: int = 6,
    ) -> Optional[np.ndarray]:
        """裁剪整行 BGR ndarray（含上下额外像素）。"""
        img = self.get_image(page_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y - pad_y)
        x2 = min(W, bbox.x + bbox.w)
        y2 = min(H, bbox.y + bbox.h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()

    def invalidate(self, path: str) -> None:
        """手动淘汰某页缓存（页面重新识别后调用）。"""
        self._cache.pop(path, None)

    def clear(self) -> None:
        """清空全部缓存。"""
        self._cache.clear()
