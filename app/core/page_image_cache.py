from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from app.models import BBox


class PageImageCache:
    """页图与字符裁片缓存。"""

    def __init__(self, max_pages: int = 8) -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self._max_pages = max_pages
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_page_image(self, page_path: str) -> np.ndarray:
        cached = self._cache.get(page_path)
        if cached is not None:
            self._cache.move_to_end(page_path)
            return cached

        image = cv2.imread(page_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to load page image: {page_path}")

        self._cache[page_path] = image
        self._cache.move_to_end(page_path)
        if len(self._cache) > self._max_pages:
            self._cache.popitem(last=False)
        return image

    def get_char_crop(self, page_path: str, bbox: BBox) -> np.ndarray:
        image = self.get_page_image(page_path)
        clamped = bbox.normalize().clamp(image.shape[1], image.shape[0])
        if clamped.w <= 0 or clamped.h <= 0:
            raise ValueError("bbox produces an empty crop")
        return image[clamped.y:clamped.y2, clamped.x:clamped.x2].copy()

    def cached_paths(self) -> list[str]:
        return list(self._cache.keys())
