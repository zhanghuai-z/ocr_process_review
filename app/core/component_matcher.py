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


def bbox_from_xyxy(values: Sequence[int | float]) -> BBox:
    x1, y1, x2, y2 = (int(round(v)) for v in values[:4])
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()
