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
    row to the first real ink component.
    """
    components = _ink_components(page_image_bgr) if page_image_bgr is not None else ()
    derived: dict[int, PpOcrV6LineHint | None] = {}
    for group in groups:
        rows = _merge_co_baseline_fragments(group.lines)
        for row in rows:
            derived[row.line.index] = _recover_unclaimed_prefix(
                row.line,
                tuple(item.line for item in rows),
                group.block_bbox,
                components,
                group.excluded_bboxes,
            )
            for source_index in row.source_indices:
                if source_index != row.line.index:
                    derived[source_index] = None
    return derived


def _merge_co_baseline_fragments(
    lines: tuple[PpOcrV6LineHint, ...],
) -> tuple[_DerivedRow, ...]:
    pending = list(sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0], line.index)))
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
    horizontal_overlap = min(left[2], right[2]) - max(left[0], right[0])
    left_center_y = (left[1] + left[3]) / 2.0
    right_center_y = (right[1] + right[3]) / 2.0
    centers_share_row = (
        right[1] <= left_center_y <= right[3]
        and left[1] <= right_center_y <= left[3]
    )
    return horizontal_overlap <= 0 and centers_share_row


def _merge_line_hints(lines: list[PpOcrV6LineHint]) -> _DerivedRow:
    ordered = sorted(lines, key=lambda line: (line.bbox[0], line.index))
    if len(ordered) == 1:
        return _DerivedRow(ordered[0], (ordered[0].index,))
    line_index = min(line.index for line in ordered)
    words = [word for line in ordered for word in sorted(line.words, key=lambda item: item.token_index)]
    merged_words = tuple(
        replace(word, line_index=line_index, token_index=token_index)
        for token_index, word in enumerate(words)
    )
    return _DerivedRow(
        PpOcrV6LineHint(
            index=line_index,
            text="".join(line.text for line in ordered),
            bbox=(
                min(line.bbox[0] for line in ordered),
                min(line.bbox[1] for line in ordered),
                max(line.bbox[2] for line in ordered),
                max(line.bbox[3] for line in ordered),
            ),
            words=merged_words,
        ),
        tuple(line.index for line in ordered),
    )


def _recover_unclaimed_prefix(
    line: PpOcrV6LineHint,
    block_rows: tuple[PpOcrV6LineHint, ...],
    block_bbox: XYXY,
    components: tuple[XYXY, ...],
    excluded_bboxes: tuple[XYXY, ...],
) -> PpOcrV6LineHint:
    bx1, _by1, _bx2, _by2 = block_bbox
    lx1, ly1, lx2, ly2 = line.bbox
    if not components or lx1 <= bx1:
        return line
    prefix_components = []
    for component in components:
        center_x = (component[0] + component[2]) / 2.0
        center_y = (component[1] + component[3]) / 2.0
        if not (bx1 <= center_x < lx1 and ly1 <= center_y <= ly2):
            continue
        if any(_contains_point(bbox, center_x, center_y) for bbox in excluded_bboxes):
            continue
        if any(
            other is not line and _contains_point(other.bbox, center_x, center_y)
            for other in block_rows
        ):
            continue
        prefix_components.append(component)
    if not prefix_components:
        return line
    return replace(line, bbox=(min(component[0] for component in prefix_components), ly1, lx2, ly2))


def _contains_point(bbox: XYXY, x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _ink_components(image_bgr: np.ndarray) -> tuple[XYXY, ...]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    return tuple(
        (int(x), int(y), int(x + width), int(y + height))
        for x, y, width, height, _area in stats[1:count]
    )
