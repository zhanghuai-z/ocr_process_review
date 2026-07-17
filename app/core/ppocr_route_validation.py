"""Page-scoped validation for derived PP-OCR CharOCR routes."""
from __future__ import annotations

from app.core.charocr_text_partition import RoutePartitionIssue
from app.models.charocr_routing import (
    PageRoutingPlan,
    RouteValidationIssue,
    is_text_route_segment_kind,
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


def missing_rotated_line_orientation_issue(
    line_index: int,
    bbox: XYXY,
) -> RouteValidationIssue:
    return RouteValidationIssue(
        code="missing_rotated_line_orientation",
        message="rotated PP-OCRv6 line has no textline orientation result",
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
    *,
    page_width: int,
    page_height: int,
) -> tuple[RouteValidationIssue, ...]:
    """Validate the page-local route contract without changing the plan.

    Every line, segment, and text slice must be non-empty and page-bounded.
    Segment geometry must be contained by its routing line. A formula
    ``content_bbox`` is the only intentional exception: it is canonical
    formula geometry that may extend beyond the line intersection, but it must
    remain page-bounded and contain that intersection. Text segments and text
    slices are a bidirectional identity map keyed by
    ``(line_index, segment_index)``. Positive PP-OCR line identities are
    unique across the page; synthetic negative identities are scoped to their
    block because vertical layout routes do not originate in PP-OCR.
    """
    issues: list[RouteValidationIssue] = []
    if page_width <= 0 or page_height <= 0:
        return (
            RouteValidationIssue(
                code="invalid_page_bounds",
                message="routing validation requires positive page dimensions",
                line_index=-1,
                bbox=(0, 0, 0, 0),
            ),
        )

    page_line_owners: dict[int, str] = {}
    page_segment_owners: dict[tuple[int, int], str] = {}
    for block_route in routing_plan.blocks:
        block_plan = block_route.plan
        block_uid = block_route.block_uid
        if block_plan.has_layout_routes != bool(block_plan.lines):
            issues.append(RouteValidationIssue(
                code="compiled_has_layout_routes_mismatch",
                message=(
                    "has_layout_routes must equal whether the block contains "
                    "routing lines"
                ),
                line_index=-1,
                bbox=block_plan.lines[0].bbox if block_plan.lines else (0, 0, 0, 0),
            ))

        lines_by_index = {}
        expected_text_slices: dict[tuple[int, int], tuple[tuple[int, int, int, int], str]] = {}
        seen_segment_identities: set[tuple[int, int]] = set()
        for routing_line in block_plan.lines:
            line_index = routing_line.index
            if line_index in lines_by_index:
                issues.append(RouteValidationIssue(
                    code="compiled_duplicate_line_index",
                    message=f"duplicate routing line index: {line_index}",
                    line_index=line_index,
                    bbox=routing_line.bbox,
                ))
            else:
                lines_by_index[line_index] = routing_line
            if line_index >= 0:
                owner = page_line_owners.get(line_index)
                if owner is not None:
                    issues.append(RouteValidationIssue(
                        code="compiled_duplicate_line_index",
                        message=(
                            f"routing line index {line_index} is owned by both "
                            f"{owner!r} and {block_uid!r}"
                        ),
                        line_index=line_index,
                        bbox=routing_line.bbox,
                    ))
                else:
                    page_line_owners[line_index] = block_uid

            if not _is_nonempty(routing_line.bbox):
                issues.append(RouteValidationIssue(
                    code="compiled_line_empty_bbox",
                    message="compiled routing line has empty geometry",
                    line_index=routing_line.index,
                    bbox=routing_line.bbox,
                ))
            elif not _is_in_page(routing_line.bbox, page_width, page_height):
                issues.append(RouteValidationIssue(
                    code="compiled_line_out_of_page",
                    message="compiled routing line is outside page bounds",
                    line_index=routing_line.index,
                    bbox=routing_line.bbox,
                ))
            if not routing_line.segments:
                issues.append(RouteValidationIssue(
                    code="compiled_line_without_segments",
                    message="compiled routing line has no segments",
                    line_index=routing_line.index,
                    bbox=routing_line.bbox,
                ))
            for segment_index, segment in enumerate(routing_line.segments):
                identity = (line_index, segment_index)
                if identity in seen_segment_identities:
                    issues.append(RouteValidationIssue(
                        code="compiled_duplicate_segment_identity",
                        message=f"duplicate routing segment identity: {identity!r}",
                        line_index=line_index,
                        bbox=segment.bbox,
                    ))
                else:
                    seen_segment_identities.add(identity)
                if line_index >= 0:
                    owner = page_segment_owners.get(identity)
                    if owner is not None:
                        issues.append(RouteValidationIssue(
                            code="compiled_duplicate_segment_identity",
                            message=(
                                f"routing segment identity {identity!r} is owned by "
                                f"both {owner!r} and {block_uid!r}"
                            ),
                            line_index=line_index,
                            bbox=segment.bbox,
                        ))
                    else:
                        page_segment_owners[identity] = block_uid

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
                elif not _is_in_page(segment.bbox, page_width, page_height):
                    issues.append(RouteValidationIssue(
                        code="compiled_segment_out_of_page",
                        message="compiled routing segment is outside page bounds",
                        line_index=routing_line.index,
                        bbox=segment.bbox,
                    ))
                if _is_nonempty(segment.bbox) and _is_nonempty(routing_line.bbox) and not _contains_bbox(
                    routing_line.bbox,
                    segment.bbox,
                ):
                    issues.append(RouteValidationIssue(
                        code="compiled_segment_outside_line",
                        message="compiled routing segment is not contained by its line",
                        line_index=routing_line.index,
                        bbox=segment.bbox,
                    ))
                if segment.content_bbox is not None:
                    content_bbox = segment.content_bbox
                    if not _is_nonempty(content_bbox):
                        issues.append(RouteValidationIssue(
                            code="compiled_content_bbox_empty",
                            message="compiled structural route has empty content geometry",
                            line_index=routing_line.index,
                            bbox=content_bbox,
                        ))
                    elif not _is_in_page(content_bbox, page_width, page_height):
                        issues.append(RouteValidationIssue(
                            code="compiled_content_bbox_out_of_page",
                            message="compiled content geometry is outside page bounds",
                            line_index=routing_line.index,
                            bbox=content_bbox,
                        ))
                    if segment.kind != "formula":
                        issues.append(RouteValidationIssue(
                            code="compiled_content_bbox_non_formula",
                            message="content_bbox is only valid for formula segments",
                            line_index=routing_line.index,
                            bbox=content_bbox,
                        ))
                    elif _is_nonempty(segment.bbox) and _is_nonempty(content_bbox) and not _contains_bbox(
                        content_bbox,
                        segment.bbox,
                    ):
                        issues.append(RouteValidationIssue(
                            code="compiled_segment_outside_content_bbox",
                            message="formula segment is not contained by content_bbox",
                            line_index=routing_line.index,
                            bbox=segment.bbox,
                        ))
                for token in segment.ppocr_latin_tokens:
                    if not _is_nonempty(token.bbox):
                        issues.append(RouteValidationIssue(
                            code="compiled_token_empty_bbox",
                            message="PP-OCR Latin token has empty geometry",
                            line_index=routing_line.index,
                            bbox=token.bbox,
                        ))
                    elif not _is_in_page(token.bbox, page_width, page_height):
                        issues.append(RouteValidationIssue(
                            code="compiled_token_out_of_page",
                            message="PP-OCR Latin token is outside page bounds",
                            line_index=routing_line.index,
                            bbox=token.bbox,
                        ))
                    elif _intersect(segment.bbox, token.bbox) is None:
                        issues.append(RouteValidationIssue(
                            code="compiled_token_disjoint_segment",
                            message="PP-OCR Latin token does not intersect its segment mask",
                            line_index=routing_line.index,
                            bbox=token.bbox,
                        ))
                if is_text_route_segment_kind(segment.kind):
                    expected_text_slices.setdefault(
                        identity,
                        (segment.bbox, segment.kind),
                    )

            for observation in routing_line.vl_marker_observations:
                if not _is_in_page(observation.bbox, page_width, page_height):
                    issues.append(RouteValidationIssue(
                        code="compiled_vl_marker_out_of_page",
                        message="VL semantic marker foreground is outside page bounds",
                        line_index=routing_line.index,
                        bbox=observation.bbox,
                    ))
                elif not _contains_bbox(routing_line.bbox, observation.bbox):
                    issues.append(RouteValidationIssue(
                        code="compiled_vl_marker_outside_line",
                        message="VL semantic marker foreground is outside its routing line",
                        line_index=routing_line.index,
                        bbox=observation.bbox,
                    ))
                if not _contains_bbox(observation.proposal_bbox, observation.bbox):
                    issues.append(RouteValidationIssue(
                        code="compiled_vl_marker_outside_proposal",
                        message="VL semantic marker foreground is outside its aligned proposal",
                        line_index=routing_line.index,
                        bbox=observation.bbox,
                    ))

        seen_slice_identities: set[tuple[int, int]] = set()
        for text_slice in block_plan.text_slices:
            identity = (text_slice.line_index, text_slice.segment_index)
            if identity in seen_slice_identities:
                issues.append(RouteValidationIssue(
                    code="compiled_duplicate_segment_identity",
                    message=f"duplicate text slice identity: {identity!r}",
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))
            else:
                seen_slice_identities.add(identity)
            if text_slice.line_index >= 0:
                owner = page_segment_owners.get(identity)
                if owner is not None and owner != block_uid:
                    issues.append(RouteValidationIssue(
                        code="compiled_duplicate_segment_identity",
                        message=(
                            f"text slice identity {identity!r} is owned by both "
                            f"{owner!r} and {block_uid!r}"
                        ),
                        line_index=text_slice.line_index,
                        bbox=text_slice.bbox,
                    ))
            if not _is_nonempty(text_slice.bbox):
                issues.append(RouteValidationIssue(
                    code="compiled_text_slice_empty_bbox",
                    message="compiled text slice has empty geometry",
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))
            elif not _is_in_page(text_slice.bbox, page_width, page_height):
                issues.append(RouteValidationIssue(
                    code="compiled_text_slice_out_of_page",
                    message="compiled text slice is outside page bounds",
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))
            line = lines_by_index.get(text_slice.line_index)
            if line is None or not (0 <= text_slice.segment_index < len(line.segments)):
                issues.append(RouteValidationIssue(
                    code="compiled_text_slice_orphan",
                    message="compiled text slice does not reference a routing segment",
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))
                continue
            segment = line.segments[text_slice.segment_index]
            if not is_text_route_segment_kind(segment.kind):
                issues.append(RouteValidationIssue(
                    code="compiled_text_slice_non_text_segment",
                    message="compiled text slice references a non-text segment",
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))
                continue
            if (
                text_slice.bbox != segment.bbox
                or text_slice.kind != segment.kind
            ):
                issues.append(RouteValidationIssue(
                    code="compiled_text_slice_mismatch",
                    message=(
                        "compiled text slice must match its segment bbox and kind"
                    ),
                    line_index=text_slice.line_index,
                    bbox=text_slice.bbox,
                ))

        for identity, (bbox, _kind) in expected_text_slices.items():
            if identity not in seen_slice_identities:
                issues.append(RouteValidationIssue(
                    code="compiled_text_segment_missing_slice",
                    message=f"text segment has no matching text slice: {identity!r}",
                    line_index=identity[0],
                    bbox=bbox,
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


def _is_in_page(bbox: XYXY, page_width: int, page_height: int) -> bool:
    return (
        0 <= bbox[0] < bbox[2] <= page_width
        and 0 <= bbox[1] < bbox[3] <= page_height
    )


def _contains_bbox(outer: XYXY, inner: XYXY) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


__all__ = [
    "invalid_prepass_line_bbox_issue",
    "overlapping_formula_masks_issue",
    "partition_issues_to_route_issues",
    "unmatched_prepass_line_issue",
    "validate_compiled_page_routing_plan",
    "validate_formula_masks",
]
