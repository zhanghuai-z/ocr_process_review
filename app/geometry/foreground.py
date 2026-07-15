"""Foreground geometry observations shared by OCR routing stages.

Foreground components are evidence about pixels in the page image.  They are
never layout blocks, OCR text, or routing decisions.  Keeping extraction in a
    single stage prevents line normalization and character reconciliation from
quietly developing different pixel ownership rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


XYXY = tuple[int, int, int, int]
_MIN_COMPONENT_AREA = 3


@dataclass(frozen=True)
class ForegroundComponent:
    """One connected foreground component in page-image coordinates."""

    bbox: XYXY
    area: int

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError("foreground component requires positive area")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("foreground component requires non-empty geometry")


@dataclass(frozen=True)
class ForegroundAnalysis:
    """Immutable pixel evidence for one page or one bounded region."""

    region_bbox: XYXY
    components: tuple[ForegroundComponent, ...]


def analyze_foreground_components(
    image_bgr: np.ndarray | None,
    region_bbox: XYXY | None = None,
    *,
    detect_polarity: bool = True,
) -> ForegroundAnalysis:
    """Extract connected foreground components in page coordinates.

    ``detect_polarity`` preserves the two validated callers' contracts. Mixed
    text partitioning detects dark-on-light and light-on-dark crops; physical
    row normalization uses the historical dark-on-light page mask.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ForegroundAnalysis((0, 0, 0, 0), ())

    gray = _grayscale(image_bgr)
    page_height, page_width = gray.shape[:2]
    bounds = (0, 0, page_width, page_height)
    requested = region_bbox or bounds
    x1, y1, x2, y2 = _clip(requested, bounds)
    if x2 <= x1 or y2 <= y1:
        return ForegroundAnalysis((x1, y1, x2, y2), ())

    crop = gray[y1:y2, x1:x2]
    binary = _foreground_mask(crop) if detect_polarity else _dark_on_light_mask(crop)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = tuple(
        ForegroundComponent(
            bbox=(
                x1 + left,
                y1 + top,
                x1 + left + width,
                y1 + top + height,
            ),
            area=area,
        )
        for left, top, width, height, area in (
            tuple(int(value) for value in stats[index])
            for index in range(1, count)
        )
        if area >= _MIN_COMPONENT_AREA
    )
    return ForegroundAnalysis((x1, y1, x2, y2), components)


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape for foreground analysis: {image.shape!r}")


def _dark_on_light_mask(crop: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary


def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    """Return foreground for either dark-on-light or light-on-dark text."""
    if crop.size == 0:
        return np.zeros(crop.shape, dtype=np.uint8)
    threshold, _unused = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = float(np.median(border))
    if background <= threshold:
        return (crop > threshold).astype(np.uint8)
    return (crop <= threshold).astype(np.uint8)


def _clip(bbox: XYXY, bounds: XYXY) -> XYXY:
    return (
        max(bbox[0], bounds[0]),
        max(bbox[1], bounds[1]),
        min(bbox[2], bounds[2]),
        min(bbox[3], bounds[3]),
    )


__all__ = [
    "ForegroundAnalysis",
    "ForegroundComponent",
    "analyze_foreground_components",
]
