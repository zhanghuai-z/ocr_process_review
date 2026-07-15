"""Derive complete physical rows from PP-OCRv6 line observations."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class TextBlockLineGroup:
    """PP rows assigned to one adopted text layout block."""

    block_uid: str
    block_bbox: XYXY
    lines: tuple[PpOcrV6LineHint, ...]
    excluded_bboxes: tuple[XYXY, ...] = ()


@dataclass(frozen=True)
class _DerivedRow:
    line: PpOcrV6LineHint
    source_indices: tuple[int, ...]


def derive_complete_text_rows(
    groups: tuple[TextBlockLineGroup, ...],
    page_image_bgr: np.ndarray | None,
) -> dict[int, PpOcrV6LineHint | None]:
    """Return derived rows keyed by original index; merged members map to None.

    Raw PP observations remain unchanged. Rows are merged only inside one
    adopted text block when their vertical spans describe the same physical
    baseline and their horizontal spans are disjoint. If PP omitted a prefix,
    unclaimed foreground inside the block/row intersection extends the derived
    row to the first real ink component. The derived row height follows the
    closure of foreground components uniquely owned by that physical row
    instead of retaining PP's oversized or clipped vertical proposal.
    """
    components = _ink_components(page_image_bgr) if page_image_bgr is not None else ()
    page_bbox = (
        (0, 0, int(page_image_bgr.shape[1]), int(page_image_bgr.shape[0]))
        if page_image_bgr is not None
        else (0, 0, 0, 0)
    )
    derived: dict[int, PpOcrV6LineHint | None] = {}
    for group in groups:
        rows = _merge_co_baseline_fragments(group.lines)
        for row in rows:
            completed = _recover_unclaimed_prefix(
                row.line,
                tuple(item.line for item in rows),
                group.block_bbox,
                components,
                group.excluded_bboxes,
            )
            derived[row.line.index] = _fit_vertical_extent_to_owned_ink(
                completed,
                components,
                group.excluded_bboxes,
                page_bbox,
                tuple(item.line for item in rows),
            )
            for source_index in row.source_indices:
                if source_index != row.line.index:
                    derived[source_index] = None
    return derived


def _merge_co_baseline_fragments(
    lines: tuple[PpOcrV6LineHint, ...],
) -> tuple[_DerivedRow, ...]:
    pending = list(sorted(lines, key=lambda hint: (hint.bbox[1], hint.bbox[0], hint.index)))
    rows: list[_DerivedRow] = []
    while pending:
        seed = pending.pop(0)
        members = [seed]
        changed = True
        while changed:
            changed = False
            for candidate in pending[:]:
                if any(_same_physical_row(candidate.bbox, member.bbox) for member in members):
                    members.append(candidate)
                    pending.remove(candidate)
                    changed = True
        rows.append(_merge_line_hints(members))
    return tuple(rows)


def _same_physical_row(left: XYXY, right: XYXY) -> bool:
    left_center_y = (left[1] + left[3]) / 2.0
    right_center_y = (right[1] + right[3]) / 2.0
    return (
        right[1] <= left_center_y <= right[3]
        and left[1] <= right_center_y <= left[3]
    )


def _merge_line_hints(lines: list[PpOcrV6LineHint]) -> _DerivedRow:
    ordered = sorted(lines, key=lambda hint: (hint.bbox[0], hint.index))
    if len(ordered) == 1:
        return _DerivedRow(ordered[0], (ordered[0].index,))
    line_index = min(hint.index for hint in ordered)
    words = [word for hint in ordered for word in sorted(hint.words, key=lambda item: item.token_index)]
    merged_words = tuple(
        replace(word, line_index=line_index, token_index=token_index)
        for token_index, word in enumerate(words)
    )
    return _DerivedRow(
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
        ),
        tuple(hint.index for hint in ordered),
    )


def _recover_unclaimed_prefix(
    hint: PpOcrV6LineHint,
    block_rows: tuple[PpOcrV6LineHint, ...],
    block_bbox: XYXY,
    components: tuple[XYXY, ...],
    excluded_bboxes: tuple[XYXY, ...],
) -> PpOcrV6LineHint:
    bx1, _by1, _bx2, _by2 = block_bbox
    lx1, ly1, lx2, ly2 = hint.bbox
    if not components or lx1 <= bx1:
        return hint
    prefix_components = []
    for component in components:
        center_x = (component[0] + component[2]) / 2.0
        center_y = (component[1] + component[3]) / 2.0
        if not (bx1 <= center_x < lx1 and ly1 <= center_y <= ly2):
            continue
        if any(_contains_point(bbox, center_x, center_y) for bbox in excluded_bboxes):
            continue
        if any(
            other is not hint and _contains_point(other.bbox, center_x, center_y)
            for other in block_rows
        ):
            continue
        prefix_components.append(component)
    if not prefix_components:
        return hint
    return replace(hint, bbox=(min(component[0] for component in prefix_components), ly1, lx2, ly2))


def _fit_vertical_extent_to_owned_ink(
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
                hint.bbox[0],
                min(active_bbox[1], accepted_component[1]),
                hint.bbox[2],
                max(active_bbox[3], accepted_component[3]),
            )
            changed = True
    if not owned:
        return hint
    return replace(
        hint,
        bbox=(
            hint.bbox[0],
            min(component[1] for component in owned),
            hint.bbox[2],
            max(component[3] for component in owned),
        ),
    )


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


def _ink_components(image_bgr: np.ndarray) -> tuple[XYXY, ...]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    return tuple(
        (int(x), int(y), int(x + width), int(y + height))
        for x, y, width, height, _area in stats[1:count]
    )
