from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from app.core.coordinate_seam import (
    BBOX_SPACE_PAGE, BBoxSpace, CropCoordinateSeam,
)
from app.core.char_bbox_utils import refine_line_bbox
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

    def get_bbox_crop(
        self,
        page_path: str,
        bbox: BBox,
        *,
        source_space: BBoxSpace = BBOX_SPACE_PAGE,
        seam: CropCoordinateSeam | None = None,
    ) -> np.ndarray:
        image = self.get_page_image(page_path)
        if seam is not None:
            clamped = seam.to_page_bbox(bbox, source_space=source_space)
        else:
            clamped = bbox.normalize().clamp(image.shape[1], image.shape[0])
        if clamped.w <= 0 or clamped.h <= 0:
            raise ValueError("bbox produces an empty crop")
        return image[clamped.y:clamped.y2, clamped.x:clamped.x2].copy()

    def get_char_crop(
        self,
        page_path: str,
        bbox: BBox,
        *,
        source_space: BBoxSpace = BBOX_SPACE_PAGE,
        seam: CropCoordinateSeam | None = None,
    ) -> np.ndarray:
        return self.get_bbox_crop(
            page_path,
            bbox,
            source_space=source_space,
            seam=seam,
        )

    def get_line_crop(
        self,
        page_path: str,
        bbox: BBox,
        pad_y: int = 6,
    ) -> np.ndarray:
        image = self.get_page_image(page_path)
        refined = refine_line_bbox(bbox, image)
        h, w = image.shape[:2]
        x1 = max(0, refined.x)
        y1 = max(0, refined.y - pad_y)
        x2 = min(w, refined.x2)
        y2 = min(h, refined.y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("line bbox produces an empty crop")
        return image[y1:y2, x1:x2].copy()

    def cached_paths(self) -> list[str]:
        return list(self._cache.keys())
