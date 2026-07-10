"""Build exact CharOCR text crops from PP-OCRv6 line observations.

The PP-OCR word boxes are proposals, not OCR facts.  This module assigns each
physical ink component in one layout-clipped row to exactly one proposal, then
emits independent LineCut and EngCut crops.  It is a pure routing boundary:
it does not call native OCR and does not modify layout or OCR observations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

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

    CJK-only rows remain one LineCut route. Rows containing Latin letters or
    digits require PP-OCR word boxes so punctuation can be isolated from
    EngCut. Any visible component without an owner is a page-local routing
    error rather than a guessed crop.
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

    components = _ink_components(image_bgr, region_bbox)
    ownership, unowned = _assign_component_owners(components, tokens, region_bbox)
    issues: list[RoutePartitionIssue] = [
        RoutePartitionIssue(
            code="unowned_text_ink",
            message="visible text ink has no PP-OCRv6 token owner",
            bbox=component[:4],
        )
        for component in unowned
    ]

    owned: list[tuple[PpOcrV6WordBox, XYXY]] = []
    for token in tokens:
        token_components = ownership.get(token.token_index, [])
        branch = _token_branch(token.text)
        if not token_components:
            if branch == "symbol":
                continue
            issues.append(RoutePartitionIssue(
                code="missing_token_mask",
                message=f"PP-OCRv6 token has no owned ink: {token.text!r}",
                bbox=_clip(token.bbox, region_bbox),
            ))
            continue
        owned.append((token, _union([component[:4] for component in token_components])))

    if issues:
        return RoutePartition((), tuple(issues))
    return RoutePartition(tuple(_segments_from_owned_components(owned)))


def _segments_from_owned_components(
    owned: list[tuple[PpOcrV6WordBox, XYXY]],
) -> list[RoutingSegment]:
    segments: list[RoutingSegment] = []
    current_kind = ""
    current_text = ""
    current_boxes: list[XYXY] = []
    for token, bbox in sorted(owned, key=lambda item: item[0].token_index):
        kind = "text_latin" if _token_branch(token.text) == "latin" else "text_other"
        if current_boxes and kind != current_kind:
            segments.append(RoutingSegment(
                kind=current_kind,
                bbox=_union(current_boxes),
                text=current_text,
            ))
            current_text = ""
            current_boxes = []
        current_kind = kind
        current_text += str(token.text or "")
        current_boxes.append(bbox)
    if current_boxes:
        segments.append(RoutingSegment(
            kind=current_kind,
            bbox=_union(current_boxes),
            text=current_text,
        ))
    return segments


def _assign_component_owners(
    components: list[tuple[int, int, int, int, int]],
    tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
) -> tuple[
    dict[int, list[tuple[int, int, int, int, int]]],
    list[tuple[int, int, int, int, int]],
]:
    owned: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    unowned: list[tuple[int, int, int, int, int]] = []
    for component in sorted(components, key=lambda item: (item[0], item[1], item[2], item[3])):
        candidates: list[tuple[int, int, int, int, PpOcrV6WordBox]] = []
        for token in tokens:
            candidate = _ownership_rank(component[:4], token, region_bbox)
            if candidate is None:
                continue
            rank, gap, center_delta = candidate
            candidates.append((rank, gap, center_delta, token.token_index, token))
        if not candidates:
            unowned.append(component)
            continue
        owner = min(candidates)[-1]
        owned[owner.token_index].append(component)
    return dict(owned), unowned


def _ownership_rank(
    component: XYXY,
    token: PpOcrV6WordBox,
    region_bbox: XYXY,
) -> tuple[int, int, int] | None:
    token_box = _clip(token.bbox, region_bbox)
    if not _is_nonempty(token_box):
        return None
    branch = _token_branch(token.text)
    vertical_overlap = min(component[3], token_box[3]) - max(component[1], token_box[1])
    vertical_gap = max(token_box[1] - component[3], component[1] - token_box[3], 0)
    horizontal_overlap = min(component[2], token_box[2]) - max(component[0], token_box[0])
    horizontal_gap = _horizontal_gap(component, token_box)
    token_height = max(1, token_box[3] - token_box[1])

    if branch == "symbol":
        if not (_center_inside(component, token_box) or _overlap_ratio(token_box, component) >= 0.30):
            return None
        return 0, horizontal_gap, _center_delta(component, token_box)

    # Letter dots and narrow leading stems may be just outside a word proposal.
    # They remain eligible only in the row's immediate geometric neighbourhood;
    # punctuation cannot claim them because symbol ownership is exact-only.
    if vertical_overlap <= 0 and not (
        vertical_gap <= max(2, token_height // 4)
        and (horizontal_overlap > 0 or horizontal_gap <= max(2, token_height // 4))
    ):
        return None
    if _center_inside(component, token_box):
        rank = 0
    elif _overlap_ratio(token_box, component) >= 0.30:
        rank = 1
    elif branch == "latin":
        rank = 2
    else:
        return None
    return rank, horizontal_gap, _center_delta(component, token_box)


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


def _center_delta(left: XYXY, right: XYXY) -> int:
    return abs((left[0] + left[2]) - (right[0] + right[2]))


__all__ = ["RoutePartition", "RoutePartitionIssue", "partition_charocr_text_region"]
