from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from app.core.ocr_ir import is_cjk_char
from app.models import BBox


@dataclass(frozen=True)
class TextComponent:
    bbox: BBox
    area: int

    @property
    def aspect_ratio(self) -> float:
        if self.bbox.h <= 0:
            return 0.0
        return float(self.bbox.w) / float(self.bbox.h)


@dataclass(frozen=True)
class ComponentShapeFilter:
    min_width: float
    max_width: float
    min_height: float
    max_height: float
    min_area: float
    max_area: float
    min_aspect_ratio: float
    max_aspect_ratio: float

    def accepts(self, component: TextComponent) -> bool:
        aspect_ratio = component.aspect_ratio
        return (
            self.min_width <= component.bbox.w <= self.max_width
            and self.min_height <= component.bbox.h <= self.max_height
            and self.min_area <= component.area <= self.max_area
            and self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
        )


@dataclass(frozen=True)
class RelativeComponentShapeFilter:
    min_width_ratio: float
    max_width_ratio: float
    min_height_ratio: float
    max_height_ratio: float
    min_area_ratio: float
    max_area_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float

    def accepts(self, component: TextComponent, reference_height: float) -> bool:
        if reference_height <= 0:
            return False
        width_ratio = float(component.bbox.w) / float(reference_height)
        height_ratio = float(component.bbox.h) / float(reference_height)
        area_ratio = float(component.area) / float(reference_height * reference_height)
        aspect_ratio = component.aspect_ratio
        return (
            self.min_width_ratio <= width_ratio <= self.max_width_ratio
            and self.min_height_ratio <= height_ratio <= self.max_height_ratio
            and self.min_area_ratio <= area_ratio <= self.max_area_ratio
            and self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
        )


@dataclass(frozen=True)
class TokenComponentAnalysis:
    text: str
    bbox: BBox
    components: tuple[TextComponent, ...]
    status: str

    @property
    def component_count(self) -> int:
        return len(self.components)


def is_cjk_token(text: str) -> bool:
    return bool(text) and all(is_cjk_char(ch) for ch in text)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return thresh


def extract_text_components(
    image: np.ndarray,
    bbox: BBox,
    *,
    kernel_size: tuple[int, int] | None = (3, 2),
    min_area: int = 20,
    min_width: int = 3,
    min_height: int = 8,
) -> list[TextComponent]:
    """Extract foreground connected components in page-space coordinates."""
    if image.size == 0:
        return []
    region = bbox.normalize().clamp(image.shape[1], image.shape[0])
    if region.w <= 0 or region.h <= 0:
        return []

    crop = image[region.y:region.y2, region.x:region.x2]
    mask = foreground_mask(crop)
    if mask.size == 0:
        return []

    if kernel_size is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components: list[TextComponent] = []
    for idx in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < min_area or w < min_width or h < min_height:
            continue
        components.append(
            TextComponent(
                bbox=BBox(region.x + x, region.y + y, w, h),
                area=area,
            )
        )
    components.sort(key=lambda component: (component.bbox.x, component.bbox.y))
    return components


def analyze_token_components(
    image: np.ndarray,
    text: str,
    bbox: BBox,
    *,
    kernel_size: tuple[int, int] | None = (3, 2),
) -> TokenComponentAnalysis:
    components = tuple(extract_text_components(image, bbox, kernel_size=kernel_size))
    if not is_cjk_token(text):
        status = "not_cjk"
    elif len(text) == 1 and len(components) >= 1:
        status = "single_cjk"
    elif len(text) > 1 and len(components) == len(text):
        status = "split_candidate"
    elif len(text) > 1:
        status = "keep_token"
    else:
        status = "empty"
    return TokenComponentAnalysis(text=text, bbox=bbox, components=components, status=status)


def count_cjk_tokens(tokens: Iterable[str]) -> tuple[int, int, int]:
    single = 0
    multi = 0
    other = 0
    for token in tokens:
        if is_cjk_token(token):
            if len(token) == 1:
                single += 1
            else:
                multi += 1
        else:
            other += 1
    return single, multi, other


def build_component_shape_filter(
    components: Sequence[TextComponent],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    padding_ratio: float = 0.15,
) -> ComponentShapeFilter:
    if not components:
        return ComponentShapeFilter(0, 0, 0, 0, 0, 0, 0, 0)

    def interval(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        low = float(np.quantile(arr, lower_quantile))
        high = float(np.quantile(arr, upper_quantile))
        pad = max((high - low) * padding_ratio, 1.0)
        return max(0.0, low - pad), high + pad

    min_width, max_width = interval([component.bbox.w for component in components])
    min_height, max_height = interval([component.bbox.h for component in components])
    min_area, max_area = interval([component.area for component in components])
    min_aspect, max_aspect = interval([component.aspect_ratio for component in components])
    return ComponentShapeFilter(
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_area=min_area,
        max_area=max_area,
        min_aspect_ratio=min_aspect,
        max_aspect_ratio=max_aspect,
    )


def build_relative_component_shape_filter(
    samples: Sequence[tuple[TextComponent, float]],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    padding_ratio: float = 0.15,
) -> RelativeComponentShapeFilter:
    valid_samples = [
        (component, float(reference_height))
        for component, reference_height in samples
        if reference_height > 0
    ]
    if not valid_samples:
        return RelativeComponentShapeFilter(0, 0, 0, 0, 0, 0, 0, 0)

    def interval(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        low = float(np.quantile(arr, lower_quantile))
        high = float(np.quantile(arr, upper_quantile))
        pad = max((high - low) * padding_ratio, 0.01)
        return max(0.0, low - pad), high + pad

    width_ratios = [component.bbox.w / reference for component, reference in valid_samples]
    height_ratios = [component.bbox.h / reference for component, reference in valid_samples]
    area_ratios = [
        component.area / (reference * reference)
        for component, reference in valid_samples
    ]
    aspects = [component.aspect_ratio for component, _ in valid_samples]
    min_width, max_width = interval(width_ratios)
    min_height, max_height = interval(height_ratios)
    min_area, max_area = interval(area_ratios)
    min_aspect, max_aspect = interval(aspects)
    return RelativeComponentShapeFilter(
        min_width_ratio=min_width,
        max_width_ratio=max_width,
        min_height_ratio=min_height,
        max_height_ratio=max_height,
        min_area_ratio=min_area,
        max_area_ratio=max_area,
        min_aspect_ratio=min_aspect,
        max_aspect_ratio=max_aspect,
    )


def bbox_from_xyxy(values: Sequence[int | float]) -> BBox:
    x1, y1, x2, y2 = (int(round(v)) for v in values[:4])
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()
