from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.models import BBox


BBoxSpace = Literal["auto", "crop", "page"]

BBOX_SPACE_AUTO: BBoxSpace = "auto"
BBOX_SPACE_CROP: BBoxSpace = "crop"
BBOX_SPACE_PAGE: BBoxSpace = "page"


def _contains_bbox(container: BBox, bbox: BBox, tolerance: int) -> bool:
    return (
        bbox.x >= container.x - tolerance
        and bbox.y >= container.y - tolerance
        and bbox.x2 <= container.x2 + tolerance
        and bbox.y2 <= container.y2 + tolerance
    )


@dataclass(frozen=True)
class CropCoordinateSeam:
    """统一 crop-space 与 page-space 的裁切/回写语义。"""

    crop_page_bbox: BBox
    page_w: int
    page_h: int
    tolerance: int = 2

    @classmethod
    def from_page_bbox(
        cls,
        crop_page_bbox: BBox,
        *,
        page_w: int,
        page_h: int,
        tolerance: int = 2,
    ) -> "CropCoordinateSeam":
        return cls(
            crop_page_bbox=crop_page_bbox.normalize().clamp(page_w, page_h),
            page_w=page_w,
            page_h=page_h,
            tolerance=tolerance,
        )

    @property
    def crop_bbox(self) -> BBox:
        return self.crop_page_bbox.normalize().clamp(self.page_w, self.page_h)

    @property
    def crop_origin_x(self) -> int:
        return self.crop_bbox.x

    @property
    def crop_origin_y(self) -> int:
        return self.crop_bbox.y

    @property
    def crop_w(self) -> int:
        return self.crop_bbox.w

    @property
    def crop_h(self) -> int:
        return self.crop_bbox.h

    def crop_image(self, page_image: np.ndarray) -> np.ndarray:
        bbox = self.crop_bbox.clamp(page_image.shape[1], page_image.shape[0])
        if bbox.w <= 0 or bbox.h <= 0:
            raise ValueError("crop seam produces an empty crop")
        return page_image[bbox.y:bbox.y2, bbox.x:bbox.x2].copy()

    def infer_input_space(self, bbox: BBox) -> BBoxSpace:
        normalized = bbox.normalize()
        crop_canvas = BBox(0, 0, self.crop_w, self.crop_h)
        fits_crop = _contains_bbox(crop_canvas, normalized, self.tolerance)
        fits_page = _contains_bbox(self.crop_bbox, normalized, self.tolerance)

        if fits_crop and not fits_page:
            return BBOX_SPACE_CROP
        if fits_page and not fits_crop:
            return BBOX_SPACE_PAGE
        if fits_page and fits_crop:
            # 无法可靠区分时，保守保持 page-space，避免再次平移。
            return BBOX_SPACE_PAGE
        return BBOX_SPACE_CROP

    def to_page_bbox(
        self,
        bbox: BBox,
        *,
        source_space: BBoxSpace = BBOX_SPACE_AUTO,
    ) -> BBox:
        normalized = bbox.normalize()
        resolved_space = (
            self.infer_input_space(normalized)
            if source_space == BBOX_SPACE_AUTO
            else source_space
        )

        if resolved_space == BBOX_SPACE_CROP:
            normalized = normalized.translated(self.crop_origin_x, self.crop_origin_y)
        return normalized.clamp(self.page_w, self.page_h)

    def to_crop_bbox(
        self,
        bbox: BBox,
        *,
        source_space: BBoxSpace = BBOX_SPACE_PAGE,
    ) -> BBox:
        page_bbox = self.to_page_bbox(bbox, source_space=source_space)
        return page_bbox.translated(-self.crop_origin_x, -self.crop_origin_y).clamp(
            self.crop_w, self.crop_h
        )
