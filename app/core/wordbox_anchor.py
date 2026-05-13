"""PP-OCRv5 word-box anchored glyph refinement.

The proof chain trusts Paddle ``text_word + text_word_boxes`` order as the
token anchor, then uses page ink only to tighten CJK token crops.  Non-CJK
tokens stay on the OCR token path and are filtered by VProof by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from app.core.ocr_ir import is_cjk_char, is_formula_token
from app.models import BBox


@dataclass(frozen=True)
class Component:
    bbox: BBox
    area: int
    cx: float
    cy: float


@dataclass(frozen=True)
class RadiationParams:
    name: str = "radiation_v12a_search_minus3"
    search_kernel: tuple[int, int] = (3, 1)
    seed_left_inset_ratio: float = 0.02
    seed_right_inset_ratio: float = 0.30
    seed_y_inset_ratio: float = 0.15
    search_left_extend_ratio: float = 0.09
    search_right_extend_ratio: float = 0.00
    search_y_extend_ratio: float = 0.15
    seed_touch_tolerance_px: int = 1
    weak_area_ratio: float = 0.05
    weak_y_overlap_ratio: float = 0.25
    right_rescue_guard_ratio: float = 0.10
    side_crumb_area_ratio: float = 0.06
    side_crumb_box_area_ratio: float = 0.012
    side_crumb_narrow_ratio: float = 0.22
    max_seed_component_width_ratio: float = 1.45
    crop_x_pad: int = 0
    crop_y_pad: int = 2
    allow_left_overflow: bool = True
    allow_right_overflow: bool = False


@dataclass(frozen=True)
class TokenAnchor:
    token_index: int
    text: str
    kind: str
    source_bbox: BBox
    ownership_x: tuple[float, float]
    raw_cc_count: int
    kept_cc_count: int
    ink_bbox: BBox | None
    crop_bbox: BBox
    status: str
    confidence: float
    notes: tuple[str, ...]


def classify_wordbox_token(text: str) -> str:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if len(compact) == 1 and is_cjk_char(compact):
        return "cjk"
    if compact.isdigit():
        return "number"
    if compact.isascii() and compact.isalpha():
        return "latin"
    if any(is_cjk_char(ch) for ch in compact):
        return "mixed_cjk"
    if is_formula_token(compact) or any(ch.isalpha() or ch.isdigit() for ch in compact):
        return "formula"
    return "punct"


def _xyxy(x1: int | float, y1: int | float, x2: int | float, y2: int | float) -> BBox:
    left = int(round(min(x1, x2)))
    top = int(round(min(y1, y2)))
    right = int(round(max(x1, x2)))
    bottom = int(round(max(y1, y2)))
    return BBox.from_xyxy(left, top, right, bottom)


def _clip(box: BBox, width: int, height: int) -> BBox:
    return box.normalize().clamp(width, height)


def _expand_xy(box: BBox, x_pad: int, y_pad: int) -> BBox:
    return BBox.from_xyxy(
        box.x1 - x_pad,
        box.y1 - y_pad,
        box.x2 + x_pad,
        box.y2 + y_pad,
    )


def _foreground_mask(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _thresh, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _foreground_ratio(image_bgr: np.ndarray, region: BBox) -> float:
    ih, iw = image_bgr.shape[:2]
    box = _clip(region, iw, ih)
    if box.w <= 0 or box.h <= 0:
        return 0.0
    mask = _foreground_mask(image_bgr[box.y1:box.y2, box.x1:box.x2])
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _default_kernel(ref_h: int) -> tuple[int, int]:
    kw = max(5, min(9, round(ref_h * 0.13)))
    kh = max(1, min(2, round(ref_h * 0.035)))
    return kw, kh


def extract_components(
    image_bgr: np.ndarray,
    region: BBox,
    *,
    kernel: tuple[int, int] | None = None,
    min_area_ratio: float = 0.0012,
) -> tuple[Component, ...]:
    ih, iw = image_bgr.shape[:2]
    box = _clip(region, iw, ih)
    if box.w <= 0 or box.h <= 0:
        return ()

    crop = image_bgr[box.y1:box.y2, box.x1:box.x2]
    mask = _foreground_mask(crop)
    if kernel is None:
        kernel = _default_kernel(box.h)
    if kernel and kernel[0] > 0 and kernel[1] > 0:
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(3, int(box.h * box.h * min_area_ratio))
    min_side = max(1, int(box.h * 0.025))

    components: list[Component] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area or w < min_side or h < min_side:
            continue
        components.append(
            Component(
                bbox=BBox(box.x1 + x, box.y1 + y, w, h),
                area=area,
                cx=box.x1 + float(centroids[idx][0]),
                cy=box.y1 + float(centroids[idx][1]),
            )
        )
    return tuple(sorted(components, key=lambda c: (c.bbox.x1, c.bbox.y1)))


def _union_bbox(components: Iterable[Component]) -> BBox | None:
    comps = list(components)
    if not comps:
        return None
    return BBox.from_xyxy(
        min(c.bbox.x1 for c in comps),
        min(c.bbox.y1 for c in comps),
        max(c.bbox.x2 for c in comps),
        max(c.bbox.y2 for c in comps),
    )


def _intersects(a: BBox, b: BBox, tolerance: int = 0) -> bool:
    return (
        min(a.x2 + tolerance, b.x2) > max(a.x1 - tolerance, b.x1)
        and min(a.y2 + tolerance, b.y2) > max(a.y1 - tolerance, b.y1)
    )


def _y_overlap(a: BBox, b: BBox) -> int:
    return max(0, min(a.y2, b.y2) - max(a.y1, b.y1))


def _dedupe_notes(notes: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(notes)))


def _radiation_refine_cjk(
    image_bgr: np.ndarray,
    source_bbox: BBox,
    ownership_x: tuple[float, float],
    params: RadiationParams,
) -> tuple[tuple[Component, ...], tuple[Component, ...], BBox | None, BBox, tuple[str, ...]]:
    ih, iw = image_bgr.shape[:2]
    source_bbox = _clip(source_bbox, iw, ih)
    sw = max(1, source_bbox.w)
    sh = max(1, source_bbox.h)
    notes: list[str] = [params.name]

    if _foreground_ratio(image_bgr, source_bbox) > 0.85:
        return (), (), None, source_bbox, ("saturated_source", params.name)

    search = _xyxy(
        source_bbox.x1 - sw * params.search_left_extend_ratio,
        source_bbox.y1 - sh * params.search_y_extend_ratio,
        source_bbox.x2 + sw * params.search_right_extend_ratio,
        source_bbox.y2 + sh * params.search_y_extend_ratio,
    ).clamp(iw, ih)
    seed = _xyxy(
        source_bbox.x1 + sw * params.seed_left_inset_ratio,
        source_bbox.y1 + sh * params.seed_y_inset_ratio,
        source_bbox.x2 - sw * params.seed_right_inset_ratio,
        source_bbox.y2 - sh * params.seed_y_inset_ratio,
    ).clamp(iw, ih)

    search_components = extract_components(image_bgr, search, kernel=params.search_kernel)
    if not search_components:
        return (), (), None, source_bbox, ("no_foreground", params.name)
    if seed.w <= 0 or seed.h <= 0:
        return search_components, (), None, source_bbox, ("empty_seed", params.name)

    seed_components = [
        comp for comp in search_components
        if _intersects(comp.bbox, seed, params.seed_touch_tolerance_px)
    ]
    if seed_components:
        notes.append(f"seed_hit_{len(seed_components)}")

    main_area = max((comp.area for comp in seed_components), default=0)
    weak_area_threshold = max(8, int(main_area * params.weak_area_ratio))
    largest_area = max(comp.area for comp in search_components)
    side_area_threshold = max(
        10,
        int(largest_area * params.side_crumb_area_ratio),
        int(source_bbox.area * params.side_crumb_box_area_ratio),
    )
    side_margin = max(2, round(sw * 0.035))

    kept: list[Component] = []
    dropped_edge = 0
    dropped_anti = 0
    rescued = 0
    oversized_seed = 0

    for comp in search_components:
        if comp in seed_components:
            kept.append(comp)
            if comp.bbox.w > sw * params.max_seed_component_width_ratio:
                oversized_seed += 1
            continue

        in_ownership = ownership_x[0] <= comp.cx < ownership_x[1]
        right_guard_x = source_bbox.x2 - sw * params.right_rescue_guard_ratio
        source_body_rescue = (
            in_ownership
            and comp.cx <= right_guard_x
            and _y_overlap(comp.bbox, source_bbox) >= sh * params.weak_y_overlap_ratio
            and comp.area >= weak_area_threshold
        )

        touches_side = comp.bbox.x1 <= source_bbox.x1 + side_margin or comp.bbox.x2 >= source_bbox.x2 - side_margin
        small_area = comp.area < side_area_threshold
        narrow = comp.bbox.w < sw * params.side_crumb_narrow_ratio or comp.bbox.h < sh * params.side_crumb_narrow_ratio
        side_crumb = touches_side and small_area and narrow

        if source_body_rescue and not side_crumb:
            kept.append(comp)
            rescued += 1
        elif side_crumb:
            dropped_edge += 1
        else:
            dropped_anti += 1

    if not kept:
        owned = [comp for comp in search_components if ownership_x[0] <= comp.cx < ownership_x[1]]
        fallback = max(owned or list(search_components), key=lambda comp: comp.area)
        kept = [fallback]
        notes.append("fallback_largest_component")

    if rescued:
        notes.append(f"rescue_source_body_{rescued}")
    if dropped_edge:
        notes.append(f"drop_edge_crumb_{dropped_edge}")
    if dropped_anti:
        notes.append(f"drop_anti_{dropped_anti}")
    if oversized_seed:
        notes.append(f"oversized_seed_{oversized_seed}")

    ink_bbox = _union_bbox(kept)
    if ink_bbox is None:
        return search_components, tuple(kept), None, source_bbox, _dedupe_notes(notes)

    seed_ink = _union_bbox(seed_components)
    seed_crosses_left = bool(seed_ink and seed_ink.x1 < source_bbox.x1)
    left_bound = search.x1 if params.allow_left_overflow and seed_crosses_left else source_bbox.x1
    right_bound = search.x2 if params.allow_right_overflow and ink_bbox.x2 > source_bbox.x2 else source_bbox.x2

    crop_bbox = _expand_xy(ink_bbox, params.crop_x_pad, params.crop_y_pad).clamp(iw, ih)
    crop_bbox = _xyxy(
        max(crop_bbox.x1, left_bound),
        max(crop_bbox.y1, source_bbox.y1 - params.crop_y_pad),
        min(crop_bbox.x2, right_bound),
        min(crop_bbox.y2, source_bbox.y2 + params.crop_y_pad),
    ).clamp(iw, ih)

    if crop_bbox.x1 < source_bbox.x1:
        notes.append(f"left_overflow_{source_bbox.x1 - crop_bbox.x1}")
    if ink_bbox.x2 > source_bbox.x2 and not params.allow_right_overflow:
        notes.append(f"right_clamped_{ink_bbox.x2 - source_bbox.x2}")

    return search_components, tuple(kept), ink_bbox, crop_bbox, _dedupe_notes(notes)


def _tight_foreground_bbox(image_bgr: np.ndarray, region: BBox) -> BBox | None:
    ih, iw = image_bgr.shape[:2]
    box = _clip(region, iw, ih)
    if box.w <= 0 or box.h <= 0:
        return None
    mask = _foreground_mask(image_bgr[box.y1:box.y2, box.x1:box.x2])
    if mask.size == 0:
        return None
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1 = int(np.argmax(rows))
    y2 = int(len(rows) - np.argmax(rows[::-1]))
    x1 = int(np.argmax(cols))
    x2 = int(len(cols) - np.argmax(cols[::-1]))
    return BBox.from_xyxy(box.x1 + x1, box.y1 + y1, box.x1 + x2, box.y1 + y2)


def _ownership_intervals(line_bbox: BBox, token_boxes: Sequence[BBox]) -> list[tuple[float, float]]:
    if not token_boxes:
        return []
    centers = [box.x1 + box.w / 2.0 for box in token_boxes]
    intervals: list[tuple[float, float]] = []
    for idx, center in enumerate(centers):
        left = float(line_bbox.x1) if idx == 0 else (centers[idx - 1] + center) / 2.0
        right = float(line_bbox.x2) if idx == len(centers) - 1 else (center + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def refine_wordbox_anchors(
    image_bgr: np.ndarray,
    line_bbox: BBox,
    tokens: Sequence[tuple[str, BBox]],
    *,
    include_non_cjk: bool = True,
    radiation_params: RadiationParams | None = None,
) -> tuple[TokenAnchor, ...]:
    """Return refined anchors for OCR token boxes in image/page coordinates."""
    if image_bgr is None or image_bgr.size == 0:
        return ()
    ih, iw = image_bgr.shape[:2]
    params = radiation_params or RadiationParams()
    token_boxes = [_clip(bbox, iw, ih) for _text, bbox in tokens]
    intervals = _ownership_intervals(_clip(line_bbox, iw, ih), token_boxes)

    anchors: list[TokenAnchor] = []
    for token_index, ((text, source_bbox), ownership_x) in enumerate(zip(tokens, intervals)):
        source_bbox = _clip(source_bbox, iw, ih)
        kind = classify_wordbox_token(text)
        if kind != "cjk" and not include_non_cjk:
            continue

        if kind == "cjk":
            raw_components, kept_components, ink_bbox, crop_bbox, notes = _radiation_refine_cjk(
                image_bgr,
                source_bbox,
                ownership_x,
                params,
            )
            if ink_bbox is None:
                status = "fallback_source_box"
                confidence = 0.45
                crop_bbox = source_bbox
            else:
                status = "single_cc" if len(kept_components) == 1 else "multi_cc_union"
                confidence = 0.82 if "fallback_largest_component" in notes else 0.96
        else:
            raw_components = extract_components(image_bgr, source_bbox, kernel=None)
            kept_components = raw_components
            ink_bbox = _tight_foreground_bbox(image_bgr, source_bbox)
            crop_bbox = (_expand_xy(ink_bbox, params.crop_x_pad, params.crop_y_pad) if ink_bbox else source_bbox).clamp(iw, ih)
            notes = ("preserve_as_token",)
            status = "token_group" if kind in {"number", "latin", "formula", "mixed_cjk"} else "punct_or_empty"
            confidence = 0.9 if ink_bbox else 0.6

        anchors.append(TokenAnchor(
            token_index=token_index,
            text=text,
            kind=kind,
            source_bbox=source_bbox,
            ownership_x=ownership_x,
            raw_cc_count=len(raw_components),
            kept_cc_count=len(kept_components),
            ink_bbox=ink_bbox,
            crop_bbox=crop_bbox,
            status=status,
            confidence=confidence,
            notes=notes,
        ))
    return tuple(anchors)
