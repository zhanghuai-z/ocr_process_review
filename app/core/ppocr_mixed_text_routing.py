"""Image-local partitioning for PP-OCRv6 mixed text rows.

PP-OCRv6 word boxes are proposal geometry only.  This module assigns actual
ink components to those proposals in reading order and emits CharOCR crop
regions.  It never creates character facts and never calls an OCR engine.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.core.layout_routing_contract import RoutingSegment
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


def partition_mixed_text_segment(
    image_bgr: np.ndarray,
    prepass_line: PpOcrV6LineHint,
    segment_bbox: XYXY,
) -> RoutePartition:
    """Split one non-structural line area into Latin, CJK, and symbol crops.

    Pure lines do not use word boxes.  A mixed line requires PP-OCRv6 word-box
    output: Latin/digit masks go to EngCut; all remaining physical ink becomes
    CJK or symbol input for LineCut.
    """
    mixed = _has_cjk(prepass_line.text) and _has_latin_or_digit(prepass_line.text)
    if not mixed:
        return RoutePartition((RoutingSegment(kind=_whole_text_kind(prepass_line.text), bbox=segment_bbox),))
    if not prepass_line.words:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="mixed_line_missing_word_boxes",
                message="mixed PP-OCRv6 line has no word-box proposals",
                bbox=segment_bbox,
            ),
        ))

    tokens = [
        token
        for token in prepass_line.words
        if not _is_space(token.text) and _intersect(token.bbox, segment_bbox) is not None
    ]
    if not tokens:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="mixed_line_no_tokens_in_text_region",
                message="mixed PP-OCRv6 line has no word boxes inside a text route",
                bbox=segment_bbox,
            ),
        ))

    latin_tokens = [token for token in tokens if _token_kind(token.text) == "latin"]
    if not latin_tokens:
        kinds = {_token_kind(token.text) for token in tokens}
        kind = "text_zh" if "cjk" in kinds else "text_symbol" if "symbol" in kinds else "text"
        return RoutePartition((RoutingSegment(kind=kind, bbox=segment_bbox),))

    components = _ink_components(image_bgr, segment_bbox)
    ownership, unowned = _assign_components_in_reading_order(components, tokens)
    issues: list[RoutePartitionIssue] = []
    if unowned:
        issues.extend(
            RoutePartitionIssue(
                code="unowned_mixed_line_ink",
                message="mixed-line ink component has no PP-OCRv6 token owner",
                bbox=component[:4],
            )
            for component in unowned
        )

    owned_masks: list[tuple[PpOcrV6WordBox, XYXY]] = []
    for token in tokens:
        owned = ownership.get(token.token_index, [])
        if not owned:
            issues.append(RoutePartitionIssue(
                code="missing_token_mask",
                message=f"PP-OCRv6 token has no owned ink: {token.text!r}",
                bbox=_clip(token.bbox, segment_bbox),
            ))
            continue
        owned_masks.append((token, _union([component[:4] for component in owned])))

    if issues:
        return RoutePartition((), tuple(issues))
    return RoutePartition(
        segments=tuple(_segments_from_owned_masks(segment_bbox, owned_masks)),
    )


def _segments_from_owned_masks(
    segment_bbox: XYXY,
    masks: list[tuple[PpOcrV6WordBox, XYXY]],
) -> list[RoutingSegment]:
    _left, top, _right, bottom = segment_bbox
    segments: list[RoutingSegment] = []
    grouped: list[tuple[str, str, XYXY, int]] = []
    for token, mask in sorted(masks, key=lambda item: item[0].token_index):
        mask = _clip(mask, segment_bbox)
        kind = _route_kind_for_token(token.text)
        if grouped and kind == "text_zh" and grouped[-1][0] == "text_zh":
            previous_kind, previous_text, previous_bbox, _previous_index = grouped[-1]
            grouped[-1] = (
                previous_kind,
                previous_text + token.text,
                _union([previous_bbox, mask]),
                token.token_index,
            )
            continue
        grouped.append((kind, token.text, mask, token.token_index))
    for kind, text, mask, _token_index in grouped:
        segments.append(RoutingSegment(
            kind=kind,
            bbox=(mask[0], top, mask[2], bottom),
            text=text,
        ))
    return [segment for segment in segments if _area(segment.bbox) > 0]


def _route_kind_for_token(text: str) -> str:
    kind = _token_kind(text)
    if kind == "latin":
        return "text_latin"
    if kind == "cjk":
        return "text_zh"
    return "text_symbol"


def _assign_components_in_reading_order(
    components: list[tuple[int, int, int, int, int]],
    tokens: list[PpOcrV6WordBox],
) -> tuple[dict[int, list[tuple[int, int, int, int, int]]], list[tuple[int, int, int, int, int]]]:
    """Assign components to ordered token owners with punctuation capacity.

    Punctuation proposals may sit flush against a neighbouring word box.  Their
    capacity is limited to the number of physical glyph components required by
    their token, so a comma cannot also claim the leading stem/dot of a word.
    """
    owned: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    punctuation_remaining = {
        token.token_index: _punctuation_component_capacity(token.text)
        for token in tokens
        if _token_kind(token.text) == "symbol"
    }
    unowned: list[tuple[int, int, int, int, int]] = []
    for component in sorted(components, key=lambda item: (item[0], item[1], item[2], item[3])):
        candidates: list[tuple[int, int, int, int, PpOcrV6WordBox]] = []
        for token in tokens:
            if _token_kind(token.text) == "symbol" and punctuation_remaining.get(token.token_index, 0) <= 0:
                continue
            rank, gap, center_delta = _component_distance_rank(component[:4], token.bbox)
            if rank is None:
                continue
            candidates.append((rank, gap, center_delta, token.token_index, token))
        if not candidates:
            unowned.append(component)
            continue
        _rank, _gap, _center, _index, owner = min(candidates)
        owned[owner.token_index].append(component)
        if _token_kind(owner.text) == "symbol":
            punctuation_remaining[owner.token_index] -= 1
    return dict(owned), unowned


def _component_distance_rank(component: XYXY, token_bbox: XYXY) -> tuple[int | None, int, int]:
    vertical_overlap = min(component[3], token_bbox[3]) - max(component[1], token_bbox[1])
    vertical_gap = max(token_bbox[1] - component[3], component[1] - token_bbox[3], 0)
    horizontal_overlap = min(component[2], token_bbox[2]) - max(component[0], token_bbox[0])
    token_height = max(1, token_bbox[3] - token_bbox[1])
    horizontal_gap = _horizontal_gap(component, token_bbox)
    if vertical_overlap <= 0 and not (
        vertical_gap <= max(2, token_height // 4)
        and (horizontal_overlap > 0 or horizontal_gap <= max(2, token_height // 4))
    ):
        return None, 0, 0
    center_inside = token_bbox[0] <= (component[0] + component[2]) / 2 <= token_bbox[2]
    overlap = _area(_intersect(component, token_bbox) or (0, 0, 0, 0))
    component_area = max(1, _area(component))
    if vertical_overlap <= 0:
        rank = 1
    elif center_inside:
        rank = 0
    elif overlap / component_area >= 0.3:
        rank = 1
    else:
        rank = 2
    gap = _horizontal_gap(component, token_bbox)
    center_delta = abs((component[0] + component[2]) - (token_bbox[0] + token_bbox[2]))
    return rank, gap, center_delta


def _ink_components(image_bgr: np.ndarray, bbox: XYXY) -> list[tuple[int, int, int, int, int]]:
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
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


def _token_kind(text: str) -> str:
    if _has_latin_or_digit(text):
        return "latin"
    if _has_cjk(text):
        return "cjk"
    return "symbol"


def _whole_text_kind(text: str) -> str:
    if _has_cjk(text) and not _has_latin_or_digit(text):
        return "text_zh"
    if _has_latin_or_digit(text) and not _has_cjk(text):
        return "text_latin"
    if str(text or "").strip() and not _has_cjk(text) and not _has_latin_or_digit(text):
        return "text_symbol"
    return "text"


def _has_cjk(text: str) -> bool:
    return any(is_cjk_char(char) for char in str(text or ""))


def _has_latin_or_digit(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in str(text or ""))


def _is_space(text: str) -> bool:
    return not str(text or "").strip()


def _punctuation_component_capacity(text: str) -> int:
    count = 0
    for char in str(text or ""):
        if char.isspace() or char.isascii() and char.isalnum() or is_cjk_char(char):
            continue
        count += 2 if char in ":;!：；！" else 1
    return max(1, count)


def _horizontal_gap(left: XYXY, right: XYXY) -> int:
    if left[2] < right[0]:
        return right[0] - left[2]
    if right[2] < left[0]:
        return left[0] - right[2]
    return 0


def _clip(bbox: XYXY, bounds: XYXY) -> XYXY:
    return (
        max(bbox[0], bounds[0]),
        max(bbox[1], bounds[1]),
        min(bbox[2], bounds[2]),
        min(bbox[3], bounds[3]),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    bbox = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return bbox if _area(bbox) > 0 else None


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _area(bbox: XYXY) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


__all__ = ["RoutePartition", "RoutePartitionIssue", "partition_mixed_text_segment"]
