"""Page-scoped validation for derived PP-OCR CharOCR routes."""
from __future__ import annotations

from app.core.charocr_text_partition import RoutePartitionIssue
from app.models.charocr_routing import (
    PageRoutingPlan,
    RouteValidationIssue,
)
from app.models.enums import BlockType
from app.models.layout_snapshot import LayoutBlockSnapshot


XYXY = tuple[int, int, int, int]


def invalid_prepass_line_bbox_issue(
    line_index: int,
    bbox: XYXY,
) -> RouteValidationIssue:
    return RouteValidationIssue(
        code="invalid_prepass_line_bbox",
        message="PP-OCRv6 returned an empty line bbox",
        line_index=line_index,
        bbox=bbox,
    )


def unmatched_prepass_line_issue(
    line_index: int,
    bbox: XYXY,
) -> RouteValidationIssue:
    return RouteValidationIssue(
        code="unmatched_prepass_line",
        message="PP-OCRv6 line does not belong to a text layout block",
        line_index=line_index,
        bbox=bbox,
    )


def text_container_clip_empty_issue(
    line_index: int,
    bbox: XYXY,
) -> RouteValidationIssue:
    return RouteValidationIssue(
        code="text_container_clip_empty",
        message="text layout block does not overlap its selected PP-OCRv6 line",
        line_index=line_index,
        bbox=bbox,
    )


def overlapping_formula_masks_issue(
    line_index: int,
    bbox: XYXY,
) -> RouteValidationIssue:
    return RouteValidationIssue(
        code="overlapping_formula_masks",
        message="formula masks overlap inside one PP-OCRv6 text row",
        line_index=line_index,
        bbox=bbox,
    )


def partition_issues_to_route_issues(
    issues: tuple[RoutePartitionIssue, ...],
    *,
    line_index: int,
) -> tuple[RouteValidationIssue, ...]:
    return tuple(
        RouteValidationIssue(
            code=issue.code,
            message=issue.message,
            line_index=line_index,
            bbox=issue.bbox,
        )
        for issue in issues
    )


def validate_formula_masks(
    masks: tuple[tuple[LayoutBlockSnapshot, XYXY, XYXY], ...],
) -> XYXY | None:
    """Return the first overlapping formula intersection, if any."""
    formulas = [
        bbox
        for block, bbox, _content_bbox in masks
        if block.block_type == BlockType.EQUATION
    ]
    for index, left in enumerate(formulas):
        for right in formulas[index + 1:]:
            overlap = _intersect(left, right)
            if overlap is not None:
                return overlap
    return None


def validate_compiled_page_routing_plan(
    routing_plan: PageRoutingPlan,
) -> tuple[RouteValidationIssue, ...]:
    """Validate route geometry after compilation without changing the plan."""
    issues: list[RouteValidationIssue] = []
    for block_route in routing_plan.blocks:
        for routing_line in block_route.plan.lines:
            if not _is_nonempty(routing_line.bbox):
                issues.append(RouteValidationIssue(
                    code="compiled_line_empty_bbox",
                    message="compiled routing line has empty geometry",
                    line_index=routing_line.index,
                    bbox=routing_line.bbox,
                ))
            for segment in routing_line.segments:
                if not _is_nonempty(segment.bbox):
                    issues.append(RouteValidationIssue(
                        code="compiled_segment_empty_bbox",
                        message=(
                            "compiled routing segment has empty geometry: "
                            f"kind={segment.kind!r}"
                        ),
                        line_index=routing_line.index,
                        bbox=segment.bbox,
                    ))
                if segment.content_bbox is not None and not _is_nonempty(segment.content_bbox):
                    issues.append(RouteValidationIssue(
                        code="compiled_content_bbox_empty",
                        message="compiled structural route has empty content geometry",
                        line_index=routing_line.index,
                        bbox=segment.content_bbox,
                    ))
    return tuple(issues)


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _is_nonempty(bbox: XYXY) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


__all__ = [
    "invalid_prepass_line_bbox_issue",
    "overlapping_formula_masks_issue",
    "partition_issues_to_route_issues",
    "text_container_clip_empty_issue",
    "unmatched_prepass_line_issue",
    "validate_compiled_page_routing_plan",
    "validate_formula_masks",
]
