"""Compile page-local CharOCR routes from adopted layout and PP-OCRv6 facts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RouteDiagnostic,
    RouteValidationIssue,
    ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT,
    ROUTING_SOURCE_PPOCR_V6_PREPASS,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
    is_text_route_segment_kind,
)
from app.core.ocr_ir import is_cjk_char
from app.core.charocr_text_partition import partition_charocr_text_region
from app.core.ppocr_layout_ownership import (
    LayoutBlockCandidate,
    LayoutOwnership,
    build_layout_ownership,
    structural_masks_for_line,
)
from app.geometry.physical_line import resolve_physical_lines
from app.geometry.text_decoration import (
    DecorationLine,
    DecorationToken,
    TextDecoration,
    detect_page_text_decorations,
)
from app.models.physical_line_geometry import ResolvedPhysicalLine
from app.core.ppocr_route_validation import (
    invalid_prepass_line_bbox_issue,
    missing_rotated_line_orientation_issue,
    overlapping_formula_masks_issue,
    partition_issues_to_route_issues,
    unmatched_prepass_line_issue,
    validate_compiled_page_routing_plan,
    validate_formula_masks,
)
from app.models.enums import BlockType
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot


XYXY = tuple[int, int, int, int]


def compile_page_routing_plan(
    snapshot: LayoutSnapshot,
    prepass: PpOcrV6PrepassArtifact,
    *,
    page_width: int,
    page_height: int,
    page_image_bgr: np.ndarray | None = None,
) -> PageRoutingPlan:
    """Compile the immutable page-scoped CharOCR execution input.

    The compiler orchestrates explicit stages. It does not adopt external
    observations or persist route records: layout ownership comes from the
    snapshot, line/text geometry comes from the prepass and page image, and
    only the returned ``PageRoutingPlan`` crosses the native boundary.
    """
    if snapshot.page_uid != prepass.page_uid:
        raise ValueError(
            "routing snapshot and PP-OCRv6 prepass belong to different pages: "
            f"{snapshot.page_uid!r} != {prepass.page_uid!r}"
        )

    ownership = build_layout_ownership(
        snapshot,
        prepass.lines,
        page_width=page_width,
        page_height=page_height,
    )

    routes_by_block_uid: dict[str, list[RoutingLine]] = {
        item.block.uid: [] for item in ownership.text_blocks
    }
    for candidate in ownership.text_blocks:
        if _is_vertical_text_block(candidate.block):
            routes_by_block_uid[candidate.block.uid].append(_vertical_text_route(candidate))
    physical_rows = resolve_physical_lines(
        ownership.line_geometry_contexts(),
        page_image_bgr,
    )
    page_decorations = detect_page_text_decorations(
        page_image_bgr,
        tuple(
            DecorationLine(
                line.bbox,
                tuple(DecorationToken(word.text, word.bbox) for word in line.words),
            )
            for line in prepass.lines
            if line.text_axis == "horizontal"
        ),
    )
    issues: list[RouteValidationIssue] = []
    diagnostics: list[RouteDiagnostic] = []
    for raw_prepass_line in prepass.lines:
        normalized_row = physical_rows.row_for_source_index(raw_prepass_line.index)
        if (
            normalized_row is not None
            and normalized_row.representative_index != raw_prepass_line.index
        ):
            continue
        prepass_line = (
            _materialize_resolved_prepass_line(normalized_row, ownership)
            if normalized_row is not None
            else ownership.ownership_for(raw_prepass_line).line
        )
        line_ownership = ownership.ownership_for(prepass_line)
        line_bbox = prepass_line.bbox
        if not _is_nonempty(line_bbox):
            issues.append(invalid_prepass_line_bbox_issue(prepass_line.index, line_bbox))
            continue
        if line_ownership.is_structural_exclusion:
            continue
        target = (
            _text_block_by_uid(ownership, normalized_row.owner_block_uid)
            if normalized_row is not None
            else line_ownership.text_block
        )
        if target is None:
            issues.append(unmatched_prepass_line_issue(prepass_line.index, line_bbox))
            continue
        if line_ownership.is_vertical_text:
            # VL owns vertical text geometry. PP-OCR may return no row or a
            # horizontal fragment for this region; neither should replace or
            # duplicate the explicit vertical route.
            continue
        if prepass_line.text_axis == "vertical" and prepass_line.orientation_angle == -1:
            issues.append(
                missing_rotated_line_orientation_issue(prepass_line.index, line_bbox)
            )
            continue

        structural_masks = structural_masks_for_line(
            line_bbox,
            ownership.structural_blocks,
        )
        formula_overlap = validate_formula_masks(structural_masks)
        if formula_overlap is not None:
            issues.append(overlapping_formula_masks_issue(prepass_line.index, formula_overlap))
            continue
        (
            segments,
            partition_issues,
            partition_symbol_observations,
            partition_diagnostics,
        ) = _segments_for_line(
            prepass_line,
            line_bbox,
            structural_masks,
            page_decorations,
            page_image_bgr=page_image_bgr,
        )
        issues.extend(partition_issues_to_route_issues(
            tuple(partition_issues),
            line_index=prepass_line.index,
        ))
        diagnostics.extend(
            RouteDiagnostic(
                code=item.code,
                message=item.message,
                line_index=prepass_line.index,
                bbox=item.bbox,
            )
            for item in partition_diagnostics
        )
        if partition_issues:
            continue
        text_slices = [segment for segment in segments if is_text_route_segment_kind(segment.kind)]
        if not text_slices:
            # The line is completely contained in a structure nested in a text
            # block (for example a table within a manually drawn text parent).
            continue
        route = RoutingLine(
            index=prepass_line.index,
            bbox=line_bbox,
            segments=tuple(segments),
            source=ROUTING_SOURCE_PPOCR_V6_PREPASS,
            text_axis=prepass_line.text_axis,
            orientation_angle=prepass_line.orientation_angle,
            ppocr_symbol_observations=tuple(partition_symbol_observations),
        )
        routes_by_block_uid[target.block.uid].append(route)

    block_routes: list[BlockRoutingPlan] = []
    for candidate in ownership.text_blocks:
        routes = tuple(sorted(routes_by_block_uid[candidate.block.uid], key=lambda item: (item.bbox[1], item.bbox[0], item.index)))
        text_slices = tuple(
            TextSliceRoute(
                line_index=line.index,
                segment_index=segment_index,
                bbox=segment.bbox,
                carved=len(line.segments) > 1,
                kind=segment.kind,
            )
            for line in routes
            for segment_index, segment in enumerate(line.segments)
            if is_text_route_segment_kind(segment.kind)
        )
        block_routes.append(BlockRoutingPlan(
            block_uid=candidate.block.uid,
            plan=RoutingPlan(
                lines=routes,
                text_slices=text_slices,
                has_layout_routes=bool(routes),
            ),
        ))

    plan = PageRoutingPlan(
        page_uid=snapshot.page_uid,
        prepass_run_id=prepass.run_id,
        blocks=tuple(block_routes),
        validation_issues=tuple(issues),
        diagnostics=tuple(diagnostics),
    )
    return replace(
        plan,
        validation_issues=(
            *plan.validation_issues,
            *validate_compiled_page_routing_plan(
                plan,
                page_width=page_width,
                page_height=page_height,
            ),
        ),
    )


def _is_vertical_text_block(block: LayoutBlockSnapshot) -> bool:
    return str(block.source_label or "").strip().lower() == "vertical_text"


def _vertical_text_route(candidate: LayoutBlockCandidate) -> RoutingLine:
    return RoutingLine(
        index=-1,
        bbox=candidate.bbox,
        segments=(RoutingSegment(kind="text_other", bbox=candidate.bbox),),
        source=ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT,
    )


def _segments_for_line(
    prepass_line: PpOcrV6LineHint,
    line_bbox: XYXY,
    structural_masks: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]],
    page_decorations: tuple[TextDecoration, ...],
    *,
    page_image_bgr: np.ndarray | None,
) -> tuple[list[RoutingSegment], tuple, tuple, tuple]:
    text_kind = _whole_line_text_kind(prepass_line.text)
    decorations = tuple(
        TextDecoration(decoration.kind, overlap)
        for decoration in page_decorations
        if (overlap := _intersect_bbox(decoration.bbox, line_bbox)) is not None
    )
    exclusion_bboxes = (
        *(mask for _block, mask, _content_bbox in structural_masks),
        *(decoration.bbox for decoration in decorations),
    )
    partition_image = _masked_partition_image(page_image_bgr, exclusion_bboxes)
    partition_line = _prepass_line_without_excluded_tokens(prepass_line, exclusion_bboxes)
    if (
        prepass_line.text_axis == "horizontal"
        and _line_requires_text_partition(prepass_line.text)
    ):
        partition = partition_charocr_text_region(
            partition_image,
            partition_line,
            line_bbox,
            excluded_bboxes=exclusion_bboxes,
        )
        text_segments = list(partition.segments)
        issues = list(partition.issues)
        symbol_observations = partition.symbol_observations
        diagnostics = partition.diagnostics
    else:
        text_segments = [RoutingSegment(kind=text_kind, bbox=line_bbox)]
        issues = []
        symbol_observations = ()
        diagnostics = ()

    # Structural regions are exact two-dimensional masks.  They deliberately
    # do not split the physical PP row into left/right crops: the native input
    # layer subtracts these rectangles from one full-line canvas.
    structure_segments = [
        RoutingSegment(
            kind="formula" if block.block_type == BlockType.EQUATION else "skip",
            label=block.source_label,
            bbox=mask,
            content_bbox=content_bbox if block.block_type == BlockType.EQUATION else None,
        )
        for block, mask, content_bbox in structural_masks
    ]
    decoration_segments = [
        RoutingSegment(kind="decoration", label=decoration.kind, bbox=decoration.bbox)
        for decoration in decorations
    ]
    return (
        [*text_segments, *structure_segments, *decoration_segments],
        tuple(issues),
        tuple(symbol_observations),
        tuple(diagnostics),
    )


def _masked_partition_image(
    page_image_bgr: np.ndarray | None,
    excluded_bboxes: tuple[XYXY, ...],
) -> np.ndarray | None:
    if page_image_bgr is None or not excluded_bboxes:
        return page_image_bgr
    masked = page_image_bgr.copy()
    for x1, y1, x2, y2 in excluded_bboxes:
        masked[y1:y2, x1:x2] = 255
    return masked


def _prepass_line_without_excluded_tokens(
    line: PpOcrV6LineHint,
    excluded_bboxes: tuple[XYXY, ...],
) -> PpOcrV6LineHint:
    if not excluded_bboxes or not line.words:
        return line
    words = tuple(
        word
        for word in line.words
        if all(not _bbox_center_inside(word.bbox, mask) for mask in excluded_bboxes)
    )
    return line if words == line.words else replace(line, words=words)


def _bbox_center_inside(inner: XYXY, outer: XYXY) -> bool:
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _intersect_bbox(left: XYXY, right: XYXY) -> XYXY | None:
    bbox = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return bbox if _is_nonempty(bbox) else None


def _whole_line_text_kind(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "text_other"
    has_cjk = any(is_cjk_char(char) for char in value)
    has_latin_or_digit = any(char.isascii() and char.isalnum() for char in value)
    if has_cjk and not has_latin_or_digit:
        return "text_other"
    if has_latin_or_digit and not has_cjk:
        return "text_latin"
    if value and not has_cjk and not has_latin_or_digit:
        return "text_other"
    return "text_other"


def _line_requires_text_partition(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in str(text or ""))


def _materialize_resolved_prepass_line(
    row: ResolvedPhysicalLine,
    ownership: LayoutOwnership,
) -> PpOcrV6LineHint:
    """Reattach immutable PP text metadata after geometry resolution."""
    source_lines = {
        decision.line.index: decision.line
        for decision in ownership.decisions
    }
    ordered = [source_lines[index] for index in row.source_indices]
    words = [
        word
        for line in ordered
        for word in sorted(line.words, key=lambda item: item.token_index)
    ]
    return PpOcrV6LineHint(
        index=row.representative_index,
        text="".join(line.text for line in ordered),
        bbox=row.bbox,
        words=tuple(
            replace(word, line_index=row.representative_index, token_index=index)
            for index, word in enumerate(words)
        ),
        text_axis=row.text_axis,
        orientation_angle=row.orientation_angle,
    )


def _text_block_by_uid(
    ownership: LayoutOwnership,
    block_uid: str,
) -> LayoutBlockCandidate | None:
    return next(
        (candidate for candidate in ownership.text_blocks if candidate.block.uid == block_uid),
        None,
    )


def _is_nonempty(bbox: XYXY) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


__all__ = ["compile_page_routing_plan"]
