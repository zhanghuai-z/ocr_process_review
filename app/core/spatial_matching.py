"""Shared spatial matching rules for OCR token rows and layout containers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Protocol, TypeVar

from app.models import BBox, Block, Line
from app.models.ocr_observation import line_ocr_bbox


class HasOptionalBBox(Protocol):
    bbox: Optional[BBox]


T = TypeVar("T", bound=HasOptionalBBox)


def min_area_overlap_ratio(first: Optional[BBox], second: Optional[BBox]) -> float:
    if first is None or second is None or first.area <= 0 or second.area <= 0:
        return 0.0
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / float(min(first.area, second.area))


def source_area_overlap_ratio(source: BBox, target: BBox) -> float:
    if source.area <= 0 or target.area <= 0:
        return 0.0
    x1 = max(source.x, target.x)
    y1 = max(source.y, target.y)
    x2 = min(source.x2, target.x2)
    y2 = min(source.y2, target.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / float(source.area) if inter > 0 else 0.0


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0


def bbox_contains_point(bbox: BBox, x: float, y: float) -> bool:
    return bbox.x <= x <= bbox.x2 and bbox.y <= y <= bbox.y2


def select_token_row_for_line(token_rows: Sequence[T], line_bbox: BBox) -> Optional[T]:
    scored: list[tuple[float, float, int, int, T]] = []
    _line_center_x, line_center_y = bbox_center(line_bbox)
    for idx, row in enumerate(token_rows):
        row_bbox = row.bbox
        overlap = min_area_overlap_ratio(row_bbox, line_bbox)
        if overlap <= 0:
            continue
        row_center_y = row_bbox.y + row_bbox.h / 2.0 if row_bbox else line_center_y
        distance = abs(row_center_y - line_center_y)
        scored.append((-overlap, distance, row_bbox.x if row_bbox else 0, idx, row))
    if not scored:
        return None
    scored.sort()
    return scored[0][4]


def select_container_block_for_line(
    line: Line,
    blocks: Sequence[Block],
    *,
    min_overlap: float = 0.10,
    center_score: float = 0.10,
) -> Optional[Block]:
    scored: list[tuple[float, int, int, int, Block]] = []
    line_bbox = line_ocr_bbox(line)
    center_x, center_y = bbox_center(line_bbox)
    for idx, block in enumerate(blocks):
        overlap = source_area_overlap_ratio(line_bbox, block.bbox)
        contains_center = bbox_contains_point(block.bbox, center_x, center_y)
        if overlap < min_overlap and not contains_center:
            continue
        score = max(overlap, center_score if contains_center else 0.0)
        scored.append((score, -block.bbox.area, -block.order, -idx, block))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][4]


def merge_bboxes(boxes: Sequence[BBox]) -> Optional[BBox]:
    valid = [bbox for bbox in boxes if bbox.area > 0]
    if not valid:
        return None
    x1 = min(bbox.x for bbox in valid)
    y1 = min(bbox.y for bbox in valid)
    x2 = max(bbox.x2 for bbox in valid)
    y2 = max(bbox.y2 for bbox in valid)
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()
