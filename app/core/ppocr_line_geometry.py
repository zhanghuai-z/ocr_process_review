"""Normalize PP-OCRv6 observations into physical text rows."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import (
    PpOcrV6LineHint,
    TEXT_AXIS_HORIZONTAL,
    TEXT_AXIS_VERTICAL,
)
from app.geometry.foreground import analyze_foreground_components


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class TextBlockLineGroup:
    """PP rows assigned to one adopted text layout block."""

    block_uid: str
    block_bbox: XYXY
    lines: tuple[PpOcrV6LineHint, ...]
    excluded_bboxes: tuple[XYXY, ...] = ()


@dataclass(frozen=True)
class PhysicalTextRow:
    """One normalized physical row and the PP-OCR observations it replaces."""

    line: PpOcrV6LineHint
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_indices", tuple(self.source_indices))
        if not self.source_indices:
            raise ValueError("physical text row requires at least one source index")


@dataclass(frozen=True)
class PhysicalLineNormalization:
    """Typed result of physical-line normalization.

    A source index can be the retained representative of a row or a merged
    member.  The distinction is explicit instead of using a nullable mapping
    as a hidden ownership signal.
    """

    rows: tuple[PhysicalTextRow, ...]
    _rows_by_source_index: Mapping[int, PhysicalTextRow] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        object.__setattr__(self, "rows", rows)
        if any(not isinstance(row, PhysicalTextRow) for row in rows):
            raise TypeError("physical line normalization rows require typed values")
        by_source_index: dict[int, PhysicalTextRow] = {}
        for row in rows:
            for source_index in row.source_indices:
                if source_index in by_source_index:
                    raise ValueError(f"duplicate physical source index: {source_index}")
                by_source_index[source_index] = row
        object.__setattr__(self, "_rows_by_source_index", MappingProxyType(by_source_index))

    def row_for_source_index(self, source_index: int) -> PhysicalTextRow | None:
        return self._rows_by_source_index.get(source_index)

    def line_for_source_index(self, source_index: int) -> PpOcrV6LineHint | None:
        row = self.row_for_source_index(source_index)
        if row is None or row.line.index != source_index:
            return None
        return row.line


def normalize_physical_text_rows(
    groups: tuple[TextBlockLineGroup, ...],
    page_image_bgr: np.ndarray | None,
) -> PhysicalLineNormalization:
    """Return immutable derived rows with an O(1) source-index lookup.

    Raw PP observations remain unchanged. Rows are merged only inside one
    adopted text block when their cross-axis spans describe the same physical
    row and their reading-axis spans are disjoint. Unclaimed foreground inside
    the block/row intersection closes either row end. The derived cross-axis
    extent follows foreground components uniquely owned by that physical row
    instead of retaining an oversized or clipped PP proposal.
    """
    components = _foreground_bboxes(page_image_bgr)
    page_bbox = (
        (0, 0, int(page_image_bgr.shape[1]), int(page_image_bgr.shape[0]))
        if page_image_bgr is not None
        else (0, 0, 0, 0)
    )
    rows_by_source: list[PhysicalTextRow] = []
    for group in groups:
        rows = _merge_co_baseline_fragments(group.lines)
        for row in rows:
            completed = _recover_unclaimed_line_ends(
                row.line,
                tuple(item.line for item in rows),
                group.block_bbox,
                components,
                group.excluded_bboxes,
            )
            normalized_line = _fit_cross_extent_to_owned_ink(
                completed,
                components,
                group.excluded_bboxes,
                page_bbox,
                tuple(item.line for item in rows),
            )
            rows_by_source.append(
                PhysicalTextRow(
                    line=normalized_line,
                    source_indices=row.source_indices,
                )
            )
    return PhysicalLineNormalization(tuple(rows_by_source))


def _merge_co_baseline_fragments(
    lines: tuple[PpOcrV6LineHint, ...],
) -> tuple[PhysicalTextRow, ...]:
    pending = list(sorted(lines, key=_row_sort_key))
    rows: list[PhysicalTextRow] = []
    while pending:
        seed = pending.pop(0)
        members = [seed]
        changed = True
        while changed:
            changed = False
            for candidate in pending[:]:
                if any(_same_physical_row(candidate, member) for member in members):
                    members.append(candidate)
                    pending.remove(candidate)
                    changed = True
        rows.append(_merge_line_hints(members))
    return tuple(rows)


def _same_physical_row(left: PpOcrV6LineHint, right: PpOcrV6LineHint) -> bool:
    if (
        left.text_axis != right.text_axis
        or left.orientation_angle != right.orientation_angle
    ):
        return False
    left_cross = _cross_span(left.bbox, left.text_axis)
    right_cross = _cross_span(right.bbox, right.text_axis)
    left_center = sum(left_cross) / 2.0
    right_center = sum(right_cross) / 2.0
    return (
        right_cross[0] <= left_center <= right_cross[1]
        and left_cross[0] <= right_center <= left_cross[1]
    )


def _merge_line_hints(lines: list[PpOcrV6LineHint]) -> PhysicalTextRow:
    ordered = sorted(lines, key=_primary_sort_key)
    if len(ordered) == 1:
        return PhysicalTextRow(ordered[0], (ordered[0].index,))
    line_index = min(hint.index for hint in ordered)
    words = [word for hint in ordered for word in sorted(hint.words, key=lambda item: item.token_index)]
    merged_words = tuple(
        replace(word, line_index=line_index, token_index=token_index)
        for token_index, word in enumerate(words)
    )
    return PhysicalTextRow(
        PpOcrV6LineHint(
            index=line_index,
            text="".join(hint.text for hint in ordered),
            bbox=(
                min(hint.bbox[0] for hint in ordered),
                min(hint.bbox[1] for hint in ordered),
                max(hint.bbox[2] for hint in ordered),
                max(hint.bbox[3] for hint in ordered),
            ),
            words=merged_words,
            text_axis=ordered[0].text_axis,
            orientation_angle=ordered[0].orientation_angle,
        ),
        tuple(hint.index for hint in ordered),
    )


def _recover_unclaimed_line_ends(
    hint: PpOcrV6LineHint,
    block_rows: tuple[PpOcrV6LineHint, ...],
    block_bbox: XYXY,
    components: tuple[XYXY, ...],
    excluded_bboxes: tuple[XYXY, ...],
) -> PpOcrV6LineHint:
    if not components:
        return hint
    line_primary = _primary_span(hint.bbox, hint.text_axis)
    block_primary = _primary_span(block_bbox, hint.text_axis)
    line_cross = _cross_span(hint.bbox, hint.text_axis)
    recovered: list[XYXY] = []
    for component in components:
        center_x = (component[0] + component[2]) / 2.0
        center_y = (component[1] + component[3]) / 2.0
        primary = center_x if hint.text_axis == TEXT_AXIS_HORIZONTAL else center_y
        cross = center_y if hint.text_axis == TEXT_AXIS_HORIZONTAL else center_x
        if not (block_primary[0] <= primary <= block_primary[1]):
            continue
        if not (line_cross[0] <= cross <= line_cross[1]):
            continue
        if any(_contains_point(bbox, center_x, center_y) for bbox in excluded_bboxes):
            continue
        if any(
            other is not hint and _contains_point(other.bbox, center_x, center_y)
            for other in block_rows
        ):
            continue
        recovered.append(component)
    if not recovered:
        return hint
    if hint.text_axis == TEXT_AXIS_HORIZONTAL:
        return replace(hint, bbox=(
            min(hint.bbox[0], *(component[0] for component in recovered)),
            hint.bbox[1],
            max(hint.bbox[2], *(component[2] for component in recovered)),
            hint.bbox[3],
        ))
    return replace(hint, bbox=(
        hint.bbox[0],
        min(hint.bbox[1], *(component[1] for component in recovered)),
        hint.bbox[2],
        max(hint.bbox[3], *(component[3] for component in recovered)),
    ))


def _fit_cross_extent_to_owned_ink(
    hint: PpOcrV6LineHint,
    components: tuple[XYXY, ...],
    excluded_bboxes: tuple[XYXY, ...],
    page_bbox: XYXY,
    block_rows: tuple[PpOcrV6LineHint, ...],
) -> PpOcrV6LineHint:
    if not components or not hint.words:
        return hint
    owned: list[XYXY] = []
    accepted: set[int] = set()
    active_bbox = hint.bbox
    changed = True
    while changed:
        changed = False
        for component_index, component in enumerate(components):
            if component_index in accepted:
                continue
            center_x = (component[0] + component[2]) / 2.0
            center_y = (component[1] + component[3]) / 2.0
            if not _strictly_contains_point(active_bbox, center_x, center_y):
                continue
            if any(
                row.index != hint.index
                and _strictly_contains_point(row.bbox, center_x, center_y)
                for row in block_rows
            ):
                continue
            if any(_contains_point(bbox, center_x, center_y) for bbox in excluded_bboxes):
                continue
            crosses_peer_row = any(
                row.index != hint.index and _intersect(component, row.bbox) is not None
                for row in block_rows
            )
            # Layout blocks determine business ownership, not the last glyph
            # pixel. Page-boundary and multi-row components may be structural
            # rules, so they remain clipped to the observed row.
            if _strictly_inside(component, page_bbox) and not crosses_peer_row:
                accepted_component = component
            else:
                accepted_component = _intersect(component, hint.bbox)
                if accepted_component is None:
                    continue
            accepted.add(component_index)
            owned.append(accepted_component)
            active_bbox = (
                min(active_bbox[0], accepted_component[0])
                if hint.text_axis == TEXT_AXIS_VERTICAL
                else hint.bbox[0],
                min(active_bbox[1], accepted_component[1])
                if hint.text_axis == TEXT_AXIS_HORIZONTAL
                else hint.bbox[1],
                max(active_bbox[2], accepted_component[2])
                if hint.text_axis == TEXT_AXIS_VERTICAL
                else hint.bbox[2],
                max(active_bbox[3], accepted_component[3])
                if hint.text_axis == TEXT_AXIS_HORIZONTAL
                else hint.bbox[3],
            )
            changed = True
    if not owned:
        return hint
    if hint.text_axis == TEXT_AXIS_HORIZONTAL:
        bbox = (
            hint.bbox[0],
            min(component[1] for component in owned),
            hint.bbox[2],
            max(component[3] for component in owned),
        )
    else:
        bbox = (
            min(component[0] for component in owned),
            hint.bbox[1],
            max(component[2] for component in owned),
            hint.bbox[3],
        )
    return replace(hint, bbox=bbox)


def _row_sort_key(hint: PpOcrV6LineHint) -> tuple[int, int, int, int]:
    cross = _cross_span(hint.bbox, hint.text_axis)
    primary = _primary_span(hint.bbox, hint.text_axis)
    return (
        0 if hint.text_axis == TEXT_AXIS_HORIZONTAL else 1,
        cross[0],
        primary[0],
        hint.index,
    )


def _primary_sort_key(hint: PpOcrV6LineHint) -> tuple[int, int]:
    start = _primary_span(hint.bbox, hint.text_axis)[0]
    if hint.orientation_angle == 180:
        start = -start
    return start, hint.index


def _primary_span(bbox: XYXY, axis: str) -> tuple[int, int]:
    return (bbox[0], bbox[2]) if axis == TEXT_AXIS_HORIZONTAL else (bbox[1], bbox[3])


def _cross_span(bbox: XYXY, axis: str) -> tuple[int, int]:
    return (bbox[1], bbox[3]) if axis == TEXT_AXIS_HORIZONTAL else (bbox[0], bbox[2])


def _contains_point(bbox: XYXY, x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _strictly_contains_point(bbox: XYXY, x: float, y: float) -> bool:
    return bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]


def _strictly_inside(inner: XYXY, outer: XYXY) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _foreground_bboxes(image_bgr: np.ndarray | None) -> tuple[XYXY, ...]:
    if image_bgr is None:
        return ()
    return tuple(
        component.bbox
        for component in analyze_foreground_components(
            image_bgr,
            detect_polarity=False,
        ).components
    )


__all__ = [
    "PhysicalTextRow",
    "PhysicalLineNormalization",
    "TextBlockLineGroup",
    "normalize_physical_text_rows",
]
