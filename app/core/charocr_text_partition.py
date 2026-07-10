"""Build CharOCR text crops from PP-OCRv6 line observations.

PP-OCR word boxes are geometry proposals, not OCR facts.  For a mixed line,
only Latin/digit proposals create EngCut masks; every remaining horizontal
region is dispatched to LineCut.  This keeps punctuation and CJK outside
EngCut without promoting PP-OCR text into OCR truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.models.charocr_routing import RoutingSegment
from app.core.ocr_ir import is_cjk_char


XYXY = tuple[int, int, int, int]
_INK_THRESHOLD = 180
_MIN_COMPONENT_AREA = 3


@dataclass(frozen=True)
class RoutePartitionIssue:
    code: str
    message: str
    bbox: XYXY


@dataclass(frozen=True)
class RoutePartition:
    segments: tuple[RoutingSegment, ...]
    issues: tuple[RoutePartitionIssue, ...] = ()


def partition_charocr_text_region(
    image_bgr: np.ndarray | None,
    prepass_line: PpOcrV6LineHint,
    region_bbox: XYXY,
) -> RoutePartition:
    """Partition one non-structural PP-OCR row region into native OCR crops.

    CJK-only rows remain one LineCut route.  On a Latin/digit row, PP-OCR
    Latin word boxes define EngCut masks.  Component closure may repair a
    word-box edge, but ink anchored by any other PP-OCR token cannot enter a
    Latin mask.  Everything outside those masks remains a LineCut region.
    """
    text = str(prepass_line.text or "")
    if not _has_latin_or_digit(text):
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox, text=text),))
    if not prepass_line.words:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_missing_word_boxes",
                message="PP-OCRv6 Latin/digit line has no word-box proposals",
                bbox=region_bbox,
            ),
        ))
    if image_bgr is None:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_requires_page_image",
                message="PP-OCRv6 Latin/digit routing requires the page image",
                bbox=region_bbox,
            ),
        ))

    tokens = tuple(
        token
        for token in prepass_line.words
        if str(token.text or "").strip() and _intersect(token.bbox, region_bbox) is not None
    )
    if not tokens:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_no_tokens_in_text_region",
                message="PP-OCRv6 Latin/digit line has no word boxes in its text region",
                bbox=region_bbox,
            ),
        ))

    mixed_tokens = [
        token
        for token in tokens
        if _has_latin_or_digit(token.text) and any(is_cjk_char(char) for char in str(token.text or ""))
    ]
    if mixed_tokens:
        token = mixed_tokens[0]
        return RoutePartition((), (
            RoutePartitionIssue(
                code="mixed_word_token",
                message=f"PP-OCRv6 mixed CJK/Latin token cannot form a deterministic mask: {token.text!r}",
                bbox=_clip(token.bbox, region_bbox),
            ),
        ))

    latin_tokens = tuple(token for token in tokens if _token_branch(token.text) == "latin")
    if not latin_tokens:
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox),))

    components = _ink_components(image_bgr, region_bbox)
    masks: list[tuple[PpOcrV6WordBox, XYXY]] = []
    issues: list[RoutePartitionIssue] = []
    for token in latin_tokens:
        mask_components = _latin_mask_components(
            components,
            token,
            tokens,
            region_bbox,
        )
        if not mask_components:
            issues.append(RoutePartitionIssue(
                code="missing_latin_token_ink",
                message=f"PP-OCRv6 Latin/digit token has no unambiguous ink: {token.text!r}",
                bbox=_clip(token.bbox, region_bbox),
            ))
            continue
        masks.append((token, _union([component[:4] for component in mask_components])))

    if issues:
        return RoutePartition((), tuple(issues))
    return _segments_from_latin_masks(region_bbox, masks, components)


def _segments_from_latin_masks(
    region_bbox: XYXY,
    masks: list[tuple[PpOcrV6WordBox, XYXY]],
    components: list[tuple[int, int, int, int, int]],
) -> RoutePartition:
    groups: list[tuple[list[PpOcrV6WordBox], XYXY]] = []
    current_tokens: list[PpOcrV6WordBox] = []
    current_bbox: XYXY | None = None
    rx1, ry1, rx2, ry2 = region_bbox
    for token, bbox in sorted(masks, key=lambda item: (item[1][0], item[0].token_index)):
        if current_bbox is None:
            current_tokens = [token]
            current_bbox = bbox
            continue
        if bbox[0] < current_bbox[2]:
            return RoutePartition((), (
                RoutePartitionIssue(
                    code="overlapping_latin_masks",
                    message="PP-OCRv6 Latin/digit masks overlap after component closure",
                    bbox=(bbox[0], ry1, current_bbox[2], ry2),
                ),
            ))
        gap = (current_bbox[2], ry1, bbox[0], ry2)
        if _has_visible_ink(components, gap):
            groups.append((current_tokens, current_bbox))
            current_tokens = [token]
            current_bbox = bbox
            continue
        current_tokens.append(token)
        current_bbox = _union([current_bbox, bbox])
    if current_bbox is not None:
        groups.append((current_tokens, current_bbox))

    segments: list[RoutingSegment] = []
    cursor = rx1
    for tokens, bbox in groups:
        x1, _y1, x2, _y2 = bbox
        before = (cursor, ry1, x1, ry2)
        if _is_nonempty(before) and _has_visible_ink(components, before):
            segments.append(RoutingSegment(kind="text_other", bbox=before))
        segments.append(RoutingSegment(
            kind="text_latin",
            bbox=bbox,
            text="".join(str(token.text or "") for token in tokens),
        ))
        cursor = x2
    after = (cursor, ry1, rx2, ry2)
    if _is_nonempty(after) and _has_visible_ink(components, after):
        segments.append(RoutingSegment(kind="text_other", bbox=after))
    return RoutePartition(tuple(segments))


def _latin_mask_components(
    components: list[tuple[int, int, int, int, int]],
    token: PpOcrV6WordBox,
    all_tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
) -> list[tuple[int, int, int, int, int]]:
    core_bbox = _clip(token.bbox, region_bbox)
    seed_bbox = _latin_seed_bbox(token, region_bbox)
    selected = [
        component
        for component in components
        if _intersect(component[:4], core_bbox) is not None
        and not _component_is_anchored_by_other_token(component[:4], token, all_tokens, region_bbox)
    ]
    if not selected:
        return selected
    reference_height = _reference_body_height(selected)
    glyph_advance = _latin_glyph_advance(token, region_bbox)
    while True:
        reclaimed_body = [
            component
            for component in components
            if component not in selected
            and _intersect(component[:4], seed_bbox) is not None
            and not _component_is_anchored_by_other_token(component[:4], token, all_tokens, region_bbox)
            and _is_adjacent_body_component(component, selected, reference_height, glyph_advance)
        ]
        if not reclaimed_body:
            break
        selected.extend(reclaimed_body)
    for component in components:
        if component in selected or _intersect(component[:4], seed_bbox) is None:
            continue
        if _component_is_anchored_by_other_token(component[:4], token, all_tokens, region_bbox):
            continue
        if _is_detached_mark(component, selected, reference_height):
            selected.append(component)
    return selected


def _latin_seed_bbox(token: PpOcrV6WordBox, region_bbox: XYXY) -> XYXY:
    """Expand a Latin proposal by one measured glyph advance in its own row."""
    core = _clip(token.bbox, region_bbox)
    glyph_advance = _latin_glyph_advance(token, region_bbox)
    vertical_margin = max(1, ceil((core[3] - core[1]) / 5))
    return _clip(
        (
            core[0] - glyph_advance,
            core[1] - vertical_margin,
            core[2] + glyph_advance,
            core[3] + vertical_margin,
        ),
        region_bbox,
    )


def _component_is_anchored_by_other_token(
    component: XYXY,
    current_token: PpOcrV6WordBox,
    tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
) -> bool:
    for token in tokens:
        if token.token_index == current_token.token_index:
            continue
        token_bbox = _clip(token.bbox, region_bbox)
        if _center_inside(component, token_bbox) or _overlap_ratio(token_bbox, component) >= 0.30:
            return True
    return False


def _latin_glyph_advance(token: PpOcrV6WordBox, region_bbox: XYXY) -> int:
    core = _clip(token.bbox, region_bbox)
    glyph_count = max(1, sum(1 for char in str(token.text or "") if char.isascii() and char.isalnum()))
    return max(1, ceil((core[2] - core[0]) / glyph_count))


def _reference_body_height(
    components: list[tuple[int, int, int, int, int]],
) -> int:
    heights = sorted(component[3] - component[1] for component in components)
    return max(1, heights[len(heights) // 2])


def _is_adjacent_body_component(
    component: tuple[int, int, int, int, int],
    selected: list[tuple[int, int, int, int, int]],
    reference_height: int,
    glyph_advance: int,
) -> bool:
    """Accept a detached body only when it matches and neighbours known word ink."""
    x1, y1, x2, y2, _area = component
    if y2 - y1 < ceil(reference_height * 0.70):
        return False
    for body_x1, body_y1, body_x2, body_y2, _body_area in selected:
        vertical_overlap = min(y2, body_y2) - max(y1, body_y1)
        if vertical_overlap <= 0:
            continue
        if _horizontal_gap((x1, y1, x2, y2), (body_x1, body_y1, body_x2, body_y2)) <= glyph_advance:
            return True
    return False


def _is_detached_mark(
    component: tuple[int, int, int, int, int],
    selected: list[tuple[int, int, int, int, int]],
    reference_height: int,
) -> bool:
    """Accept a detached mark only when it is vertically attached to word ink."""
    x1, y1, x2, y2, _area = component
    if y2 - y1 > ceil(reference_height * 0.55):
        return False
    for body_x1, body_y1, body_x2, _body_y2, _body_area in selected:
        horizontal_overlap = min(x2, body_x2) - max(x1, body_x1)
        if horizontal_overlap > 0 and y2 <= body_y1 and body_y1 - y2 <= reference_height:
            return True
    return False


def _has_visible_ink(
    components: list[tuple[int, int, int, int, int]],
    bbox: XYXY,
) -> bool:
    return any(_intersect(component[:4], bbox) is not None for component in components)


def _ink_components(image_bgr: np.ndarray, bbox: XYXY) -> list[tuple[int, int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    page_height, page_width = gray.shape[:2]
    x1, y1, x2, y2 = _clip(bbox, (0, 0, page_width, page_height))
    if x2 <= x1 or y2 <= y1:
        return []
    binary = (gray[y1:y2, x1:x2] < _INK_THRESHOLD).astype(np.uint8)
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(binary, 8)
    return [
        (x1 + left, y1 + top, x1 + left + width, y1 + top + height, area)
        for left, top, width, height, area in (tuple(int(value) for value in stats[index]) for index in range(1, count))
        if area >= _MIN_COMPONENT_AREA
    ]


def _token_branch(text: str) -> str:
    value = str(text or "")
    if _has_latin_or_digit(value) and not any(is_cjk_char(char) for char in value):
        return "latin"
    if any(is_cjk_char(char) for char in value):
        return "other"
    return "symbol"


def _has_latin_or_digit(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in str(text or ""))


def _clip(bbox: XYXY, bounds: XYXY) -> XYXY:
    return (
        max(bbox[0], bounds[0]),
        max(bbox[1], bounds[1]),
        min(bbox[2], bounds[2]),
        min(bbox[3], bounds[3]),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    bbox = _clip(left, right)
    return bbox if _is_nonempty(bbox) else None


def _is_nonempty(bbox: XYXY) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _center_inside(component: XYXY, bbox: XYXY) -> bool:
    center_x = (component[0] + component[2]) / 2
    center_y = (component[1] + component[3]) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _overlap_ratio(owner: XYXY, component: XYXY) -> float:
    overlap = _intersect(owner, component)
    if overlap is None:
        return 0.0
    area = max(1, (component[2] - component[0]) * (component[3] - component[1]))
    return ((overlap[2] - overlap[0]) * (overlap[3] - overlap[1])) / area


def _horizontal_gap(left: XYXY, right: XYXY) -> int:
    if left[2] < right[0]:
        return right[0] - left[2]
    if right[2] < left[0]:
        return left[0] - right[2]
    return 0


__all__ = ["RoutePartition", "RoutePartitionIssue", "partition_charocr_text_region"]
