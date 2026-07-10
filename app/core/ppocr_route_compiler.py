"""Compile page-local CharOCR routes from adopted layout and PP-OCRv6 facts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.core.layout_routing_contract import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RouteValidationIssue,
    ROUTING_SOURCE_PPOCR_V6_PREPASS,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)
from app.core.ocr_ir import is_cjk_char
from app.core.ppocr_mixed_text_routing import partition_mixed_text_segment
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
    issues: list[RouteValidationIssue] = []
    for prepass_line in prepass.lines:
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
        structural_cuts = _structural_cuts_for_line(line_bbox, structural_blocks)
        if target is None:
            if _line_is_structural_only(line_bbox, structural_cuts):
                continue
            issues.append(RouteValidationIssue(
                code="unmatched_prepass_line",
                message="PP-OCRv6 line does not belong to a text layout block",
                line_index=prepass_line.index,
                bbox=line_bbox,
            ))
            continue

        clipped_line = _intersect(line_bbox, target.bbox)
        if clipped_line is None:
            issues.append(RouteValidationIssue(
                code="text_container_clip_empty",
                message="text layout block does not overlap its selected PP-OCRv6 line",
                line_index=prepass_line.index,
                bbox=line_bbox,
            ))
            continue
        segments, partition_issues = _segments_for_line(
            prepass_line,
            clipped_line,
            structural_cuts,
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


def _is_text_dispatch_block(block: LayoutBlockSnapshot) -> bool:
    return block.ocr_policy == OcrPolicy.TEXT_OCR and block.block_type not in _STRUCTURAL_BLOCK_TYPES


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


def _structural_cuts_for_line(line_bbox: XYXY, candidates: list[_BlockCandidate]) -> list[tuple[LayoutBlockSnapshot, XYXY]]:
    cuts: list[tuple[LayoutBlockSnapshot, XYXY]] = []
    for candidate in candidates:
        overlap = _intersect(line_bbox, candidate.bbox)
        if overlap is not None:
            cuts.append((candidate.block, overlap))
    return sorted(cuts, key=lambda item: (item[1][0], item[1][1], item[0].order))


def _line_is_structural_only(line_bbox: XYXY, cuts: list[tuple[LayoutBlockSnapshot, XYXY]]) -> bool:
    if not cuts:
        return False
    covered_width = _merged_horizontal_coverage([(bbox[0], bbox[2]) for _block, bbox in cuts])
    return covered_width >= (line_bbox[2] - line_bbox[0])


def _segments_for_line(
    prepass_line: PpOcrV6LineHint,
    line_bbox: XYXY,
    structural_cuts: list[tuple[LayoutBlockSnapshot, XYXY]],
    *,
    page_image_bgr: np.ndarray | None,
) -> tuple[list[RoutingSegment], tuple]:
    lx1, ly1, lx2, ly2 = line_bbox
    cuts_by_x = _non_overlapping_cuts(structural_cuts, line_bbox)
    text_kind = _whole_line_text_kind(prepass_line.text)
    segments: list[RoutingSegment] = []
    cursor = lx1
    for block, cut in cuts_by_x:
        cx1, _cy1, cx2, _cy2 = cut
        if cursor < cx1:
            segments.append(RoutingSegment(kind=text_kind, bbox=(cursor, ly1, cx1, ly2)))
        kind = "formula" if block.block_type == BlockType.EQUATION else "skip"
        segments.append(RoutingSegment(kind=kind, label=block.source_label, bbox=cut))
        cursor = max(cursor, cx2)
    if cursor < lx2:
        segments.append(RoutingSegment(kind=text_kind, bbox=(cursor, ly1, lx2, ly2)))
    if not cuts_by_x:
        segments = [RoutingSegment(kind=text_kind, bbox=line_bbox)]
    else:
        segments = [segment for segment in segments if _is_nonempty(segment.bbox)]

    output: list[RoutingSegment] = []
    issues = []
    for segment in segments:
        if segment.kind != "text":
            output.append(segment)
            continue
        if _is_mixed_ppocr_text(prepass_line.text) and not prepass_line.words:
            issues.append(RouteValidationIssue(
                code="mixed_line_missing_word_boxes",
                message="mixed PP-OCRv6 line has no word-box proposals",
                line_index=prepass_line.index,
                bbox=segment.bbox,
            ))
            continue
        if page_image_bgr is None:
            issues.append(RouteValidationIssue(
                code="mixed_line_requires_page_image",
                message="mixed PP-OCRv6 line requires page image for component routing",
                line_index=prepass_line.index,
                bbox=segment.bbox,
            ))
            continue
        partition = partition_mixed_text_segment(page_image_bgr, prepass_line, segment.bbox)
        output.extend(partition.segments)
        issues.extend(partition.issues)
    return output, tuple(issues)


def _non_overlapping_cuts(
    cuts: list[tuple[LayoutBlockSnapshot, XYXY]],
    line_bbox: XYXY,
) -> list[tuple[LayoutBlockSnapshot, XYXY]]:
    """Choose a deterministic non-overlapping structural cover on one row."""
    selected: list[tuple[LayoutBlockSnapshot, XYXY]] = []
    cursor = line_bbox[0]
    for block, bbox in cuts:
        x1, y1, x2, y2 = bbox
        x1 = max(x1, cursor)
        if x2 <= x1:
            continue
        selected.append((block, (x1, y1, x2, y2)))
        cursor = x2
    return selected


def _whole_line_text_kind(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "text"
    has_cjk = any(is_cjk_char(char) for char in value)
    has_latin_or_digit = any(char.isascii() and char.isalnum() for char in value)
    if has_cjk and not has_latin_or_digit:
        return "text_zh"
    if has_latin_or_digit and not has_cjk:
        return "text_latin"
    if value and not has_cjk and not has_latin_or_digit:
        return "text_symbol"
    return "text"


def _is_mixed_ppocr_text(text: str) -> bool:
    value = str(text or "")
    return (
        any(is_cjk_char(char) for char in value)
        and any(char.isascii() and char.isalnum() for char in value)
    )


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
