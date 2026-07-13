"""Compile page-local CharOCR routes from adopted layout and PP-OCRv6 facts."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RouteValidationIssue,
    ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT,
    ROUTING_SOURCE_PPOCR_V6_PREPASS,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)
from app.core.ocr_ir import is_cjk_char
from app.core.charocr_text_partition import partition_charocr_text_region
from app.core.ppocr_line_geometry import TextBlockLineGroup, derive_complete_text_rows
from app.models.enums import BlockType, OcrPolicy
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot


XYXY = tuple[int, int, int, int]
_STRUCTURAL_BLOCK_TYPES = frozenset({
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.UNKNOWN,
})


@dataclass(frozen=True)
class _BlockCandidate:
    block: LayoutBlockSnapshot
    bbox: XYXY


def compile_page_routing_plan(
    snapshot: LayoutSnapshot,
    prepass: PpOcrV6PrepassArtifact,
    *,
    page_width: int,
    page_height: int,
    page_image_bgr: np.ndarray | None = None,
) -> PageRoutingPlan:
    """Create the only dispatch plan accepted by CharOCR for one page.

    The PP-OCRv6 request sees a full page for line geometry.  This compiler is
    the boundary that prevents its observations inside tables, figures, charts,
    formulas, and other structural regions from becoming CharOCR input.
    """
    if snapshot.page_uid != prepass.page_uid:
        raise ValueError(
            "routing snapshot and PP-OCRv6 prepass belong to different pages: "
            f"{snapshot.page_uid!r} != {prepass.page_uid!r}"
        )

    text_blocks: list[_BlockCandidate] = []
    structural_blocks: list[_BlockCandidate] = []
    for block in snapshot.blocks:
        bbox = _clamp(block.bbox.to_xyxy(), page_width, page_height)
        if _is_text_dispatch_block(block):
            text_blocks.append(_BlockCandidate(block, bbox))
        else:
            structural_blocks.append(_BlockCandidate(block, bbox))

    routes_by_block_uid: dict[str, list[RoutingLine]] = {
        item.block.uid: [] for item in text_blocks
    }
    for candidate in text_blocks:
        if _is_vertical_text_block(candidate.block):
            routes_by_block_uid[candidate.block.uid].append(_vertical_text_route(candidate))
    derived_lines = _derive_text_rows(
        prepass.lines,
        text_blocks,
        structural_blocks,
        page_width=page_width,
        page_height=page_height,
        page_image_bgr=page_image_bgr,
    )
    issues: list[RouteValidationIssue] = []
    for raw_prepass_line in prepass.lines:
        if raw_prepass_line.index in derived_lines and derived_lines[raw_prepass_line.index] is None:
            continue
        prepass_line = derived_lines.get(raw_prepass_line.index, raw_prepass_line)
        if prepass_line is None:
            continue
        line_bbox = _clamp(prepass_line.bbox, page_width, page_height)
        if not _is_nonempty(line_bbox):
            issues.append(RouteValidationIssue(
                code="invalid_prepass_line_bbox",
                message="PP-OCRv6 returned an empty line bbox",
                line_index=prepass_line.index,
                bbox=line_bbox,
            ))
            continue

        target = _select_text_container(line_bbox, text_blocks)
        structural_owner = _select_structural_owner(line_bbox, structural_blocks)
        if structural_owner is not None and (
            target is None
            or _overlap_score(line_bbox, structural_owner.bbox)
            >= _overlap_score(line_bbox, target.bbox)
        ):
            continue
        if target is None:
            issues.append(RouteValidationIssue(
                code="unmatched_prepass_line",
                message="PP-OCRv6 line does not belong to a text layout block",
                line_index=prepass_line.index,
                bbox=line_bbox,
            ))
            continue
        if _is_vertical_text_block(target.block):
            # VL owns vertical text geometry. PP-OCR may return no row or a
            # horizontal fragment for this region; neither should replace or
            # duplicate the explicit vertical route.
            continue

        clipped_line = _clip_line_to_text_block_width(line_bbox, target.bbox)
        if clipped_line is None:
            issues.append(RouteValidationIssue(
                code="text_container_clip_empty",
                message="text layout block does not overlap its selected PP-OCRv6 line",
                line_index=prepass_line.index,
                bbox=line_bbox,
            ))
            continue
        structural_masks = _structural_masks_for_line(clipped_line, structural_blocks)
        formula_overlap = _overlapping_formula_mask(structural_masks)
        if formula_overlap is not None:
            issues.append(RouteValidationIssue(
                code="overlapping_formula_masks",
                message="formula masks overlap inside one PP-OCRv6 text row",
                line_index=prepass_line.index,
                bbox=formula_overlap,
            ))
            continue
        segments, partition_issues = _segments_for_line(
            prepass_line,
            clipped_line,
            structural_masks,
            page_image_bgr=page_image_bgr,
        )
        issues.extend(
            RouteValidationIssue(
                code=issue.code,
                message=issue.message,
                line_index=prepass_line.index,
                bbox=issue.bbox,
            )
            for issue in partition_issues
        )
        if partition_issues:
            continue
        text_slices = [segment for segment in segments if segment.kind.startswith("text")]
        if not text_slices:
            # The line is completely contained in a structure nested in a text
            # block (for example a table within a manually drawn text parent).
            continue
        route = RoutingLine(
            index=prepass_line.index,
            bbox=clipped_line,
            segments=tuple(segments),
            source=ROUTING_SOURCE_PPOCR_V6_PREPASS,
        )
        routes_by_block_uid[target.block.uid].append(route)

    block_routes: list[BlockRoutingPlan] = []
    for candidate in text_blocks:
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
            if segment.kind.startswith("text")
        )
        block_routes.append(BlockRoutingPlan(
            block_uid=candidate.block.uid,
            plan=RoutingPlan(
                lines=routes,
                text_slices=text_slices,
                has_layout_routes=bool(routes),
            ),
        ))

    return PageRoutingPlan(
        page_uid=snapshot.page_uid,
        prepass_run_id=prepass.run_id,
        blocks=tuple(block_routes),
        validation_issues=tuple(issues),
    )


def _derive_text_rows(
    lines: tuple[PpOcrV6LineHint, ...],
    text_blocks: list[_BlockCandidate],
    structural_blocks: list[_BlockCandidate],
    *,
    page_width: int,
    page_height: int,
    page_image_bgr: np.ndarray | None,
) -> dict[int, PpOcrV6LineHint | None]:
    grouped: dict[str, tuple[_BlockCandidate, list[PpOcrV6LineHint]]] = {}
    for line in lines:
        line_bbox = _clamp(line.bbox, page_width, page_height)
        target = _select_text_container(line_bbox, text_blocks)
        structural_owner = _select_structural_owner(line_bbox, structural_blocks)
        if target is None or _is_vertical_text_block(target.block):
            continue
        if structural_owner is not None and _overlap_score(
            line_bbox, structural_owner.bbox
        ) >= _overlap_score(line_bbox, target.bbox):
            continue
        entry = grouped.setdefault(target.block.uid, (target, []))
        entry[1].append(replace(line, bbox=line_bbox))
    return derive_complete_text_rows(
        tuple(
            TextBlockLineGroup(
                block_uid=block_uid,
                block_bbox=target.bbox,
                lines=tuple(block_lines),
                excluded_bboxes=tuple(
                    candidate.bbox
                    for candidate in structural_blocks
                    if _intersect(candidate.bbox, target.bbox) is not None
                ),
            )
            for block_uid, (target, block_lines) in grouped.items()
        ),
        page_image_bgr,
    )


def _is_text_dispatch_block(block: LayoutBlockSnapshot) -> bool:
    return block.ocr_policy == OcrPolicy.TEXT_OCR and block.block_type not in _STRUCTURAL_BLOCK_TYPES


def _is_vertical_text_block(block: LayoutBlockSnapshot) -> bool:
    return str(block.source_label or "").strip().lower() == "vertical_text"


def _vertical_text_route(candidate: _BlockCandidate) -> RoutingLine:
    return RoutingLine(
        index=-1,
        bbox=candidate.bbox,
        segments=(RoutingSegment(kind="text_other", bbox=candidate.bbox),),
        source=ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT,
    )


def _select_text_container(line_bbox: XYXY, candidates: list[_BlockCandidate]) -> _BlockCandidate | None:
    best: tuple[float, int, _BlockCandidate] | None = None
    for candidate in candidates:
        overlap = _intersect(line_bbox, candidate.bbox)
        if overlap is None:
            continue
        line_area = _area(line_bbox)
        score = _area(overlap) / max(1, line_area)
        if score <= 0:
            continue
        ranked = (score, -candidate.block.order, candidate)
        if best is None or ranked[:2] > best[:2]:
            best = ranked
    return best[2] if best is not None and best[0] >= 0.1 else None


def _select_structural_owner(
    line_bbox: XYXY,
    candidates: list[_BlockCandidate],
) -> _BlockCandidate | None:
    """Return the strongest structure containing the PP line center.

    PP detector boxes commonly extend beyond a formula or formula-number
    block.  Center ownership identifies that structural row without weakening
    the unmatched-line gate.  A surrounding text block can still win by
    covering more of the row, which preserves inline-formula carving.
    """
    center_x = (line_bbox[0] + line_bbox[2]) / 2.0
    center_y = (line_bbox[1] + line_bbox[3]) / 2.0
    best: tuple[float, int, _BlockCandidate] | None = None
    for candidate in candidates:
        bbox = candidate.bbox
        if not (
            bbox[0] <= center_x <= bbox[2]
            and bbox[1] <= center_y <= bbox[3]
        ):
            continue
        score = _overlap_score(line_bbox, bbox)
        ranked = (score, -candidate.block.order, candidate)
        if best is None or ranked[:2] > best[:2]:
            best = ranked
    return best[2] if best is not None else None


def _overlap_score(line_bbox: XYXY, candidate_bbox: XYXY) -> float:
    overlap = _intersect(line_bbox, candidate_bbox)
    return _area(overlap) / max(1, _area(line_bbox)) if overlap is not None else 0.0


def _structural_masks_for_line(
    line_bbox: XYXY,
    candidates: list[_BlockCandidate],
) -> list[tuple[LayoutBlockSnapshot, XYXY, XYXY]]:
    masks: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]] = []
    for candidate in candidates:
        overlap = _intersect(line_bbox, candidate.bbox)
        if overlap is not None:
            masks.append((candidate.block, overlap, candidate.bbox))
    return sorted(masks, key=lambda item: (item[1][0], item[1][1], item[0].order))


def _overlapping_formula_mask(
    cuts: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]],
) -> XYXY | None:
    formulas = [bbox for block, bbox, _content_bbox in cuts if block.block_type == BlockType.EQUATION]
    for index, left in enumerate(formulas):
        for right in formulas[index + 1:]:
            overlap = _intersect(left, right)
            if overlap is not None:
                return overlap
    return None


def _segments_for_line(
    prepass_line: PpOcrV6LineHint,
    line_bbox: XYXY,
    structural_masks: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]],
    *,
    page_image_bgr: np.ndarray | None,
) -> tuple[list[RoutingSegment], tuple]:
    text_kind = _whole_line_text_kind(prepass_line.text)
    partition_image = _masked_partition_image(page_image_bgr, structural_masks)
    partition_line = _prepass_line_without_structural_tokens(prepass_line, structural_masks)
    if _line_requires_text_partition(prepass_line.text):
        partition = partition_charocr_text_region(
            partition_image,
            partition_line,
            line_bbox,
            excluded_bboxes=tuple(mask for _block, mask, _content_bbox in structural_masks),
        )
        text_segments = list(partition.segments)
        issues = list(partition.issues)
    else:
        text_segments = [RoutingSegment(kind=text_kind, bbox=line_bbox)]
        issues = []

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
    return [*text_segments, *structure_segments], tuple(issues)


def _masked_partition_image(
    page_image_bgr: np.ndarray | None,
    structural_masks: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]],
) -> np.ndarray | None:
    if page_image_bgr is None or not structural_masks:
        return page_image_bgr
    masked = page_image_bgr.copy()
    for _block, (x1, y1, x2, y2), _content_bbox in structural_masks:
        masked[y1:y2, x1:x2] = 255
    return masked


def _prepass_line_without_structural_tokens(
    line: PpOcrV6LineHint,
    structural_masks: list[tuple[LayoutBlockSnapshot, XYXY, XYXY]],
) -> PpOcrV6LineHint:
    if not structural_masks or not line.words:
        return line
    mask_boxes = [mask for _block, mask, _content_bbox in structural_masks]
    words = tuple(
        word
        for word in line.words
        if all(not _bbox_center_inside(word.bbox, mask) for mask in mask_boxes)
    )
    return line if words == line.words else replace(line, words=words)


def _bbox_center_inside(inner: XYXY, outer: XYXY) -> bool:
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


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


def _clamp(bbox: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _clip_line_to_text_block_width(line_bbox: XYXY, block_bbox: XYXY) -> XYXY | None:
    """Keep block ownership horizontal without clipping complete glyph ink.

    PP word observations own the row's vertical glyph extent. Layout geometry
    selects the text container and limits cross-column spill, but a tight VL
    block must not cut the top or bottom from an otherwise owned glyph.
    """
    x1 = max(line_bbox[0], block_bbox[0])
    x2 = min(line_bbox[2], block_bbox[2])
    return (x1, line_bbox[1], x2, line_bbox[3]) if x2 > x1 else None


def _merged_horizontal_coverage(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _area(bbox: XYXY) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _is_nonempty(bbox: XYXY) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


__all__ = ["compile_page_routing_plan"]
