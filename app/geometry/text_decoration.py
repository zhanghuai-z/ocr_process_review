"""Detect non-text line decorations from vendor geometry and page pixels."""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import numpy as np

from app.geometry.foreground import ForegroundComponent, analyze_foreground_components


XYXY = tuple[int, int, int, int]
DOT_LEADER = "dot_leader"
_MIN_REPEATED_COMPONENTS = 8
_MIN_LEADER_LINE_HEIGHTS = 4


@dataclass(frozen=True)
class DecorationToken:
    """Vendor token geometry used only as a decoration candidate boundary."""

    text: str
    bbox: XYXY


@dataclass(frozen=True)
class DecorationLine:
    """One vendor line used to locate page-level decoration rows."""

    bbox: XYXY
    tokens: tuple[DecorationToken, ...]


@dataclass(frozen=True)
class TextDecoration:
    """One derived non-text decoration mask inside a physical line."""

    kind: str
    bbox: XYXY


def detect_text_decorations(
    image_bgr: np.ndarray | None,
    line_bbox: XYXY,
    tokens: tuple[DecorationToken, ...],
) -> tuple[TextDecoration, ...]:
    return detect_page_text_decorations(
        image_bgr,
        (DecorationLine(line_bbox, tokens),),
    )


def detect_page_text_decorations(
    image_bgr: np.ndarray | None,
    lines: tuple[DecorationLine, ...],
) -> tuple[TextDecoration, ...]:
    """Return proven long punctuation leaders, never ordinary ellipses.

    PP may split one directory row into left and right observations, leaving
    the leader itself unboxed. Co-baseline observations therefore form one
    detection envelope without becoming one OCR/layout owner. Candidate pixels
    are limited to punctuation-token boxes and gaps between separate
    co-baseline observations. A candidate must contain at least eight aligned
    small components spanning four row heights. The returned mask covers those
    components only; tall punctuation such as a trailing parenthesis remains
    text input.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ()
    decorations: list[TextDecoration] = []
    for group in _co_baseline_groups(lines):
        line_height = max(1, max(source.bbox[3] - source.bbox[1] for source in group))
        for candidate_bbox in _candidate_envelopes(group, line_height):
            analysis = analyze_foreground_components(image_bgr, candidate_bbox)
            components = tuple(
                component
                for component in analysis.components
                if _is_small_repeated_mark(component, line_height)
            )
            run = _dominant_aligned_run(components, line_height)
            if len(run) < _MIN_REPEATED_COMPONENTS:
                continue
            bbox = _union(tuple(component.bbox for component in run))
            if bbox[2] - bbox[0] < _MIN_LEADER_LINE_HEIGHTS * line_height:
                continue
            decorations.append(TextDecoration(DOT_LEADER, bbox))
    return tuple(_merge_overlapping(decorations))


def _candidate_envelopes(
    group: tuple[DecorationLine, ...],
    line_height: int,
) -> tuple[XYXY, ...]:
    candidates = [
        token.bbox
        for source in group
        for token in source.tokens
        if _is_punctuation_token(token.text)
    ]
    ordered_lines = sorted(group, key=lambda source: source.bbox[0])
    for left, right in zip(ordered_lines, ordered_lines[1:]):
        if left.bbox[2] < right.bbox[0]:
            candidates.append((
                max(left.bbox[0], left.bbox[2] - line_height),
                min(left.bbox[1], right.bbox[1]),
                min(right.bbox[2], right.bbox[0] + line_height),
                max(left.bbox[3], right.bbox[3]),
            ))
        left_token = _boundary_punctuation_token(left, at_end=True, tolerance=line_height)
        right_token = _boundary_punctuation_token(right, at_end=False, tolerance=line_height)
        if left_token is None or right_token is None:
            continue
        if left_token.bbox[2] >= right_token.bbox[0]:
            continue
        candidates.append(_union((left_token.bbox, right_token.bbox)))
    return tuple(_merge_boxes(candidates))


def _boundary_punctuation_token(
    source: DecorationLine,
    *,
    at_end: bool,
    tolerance: int,
) -> DecorationToken | None:
    tokens = [token for token in source.tokens if _is_punctuation_token(token.text)]
    if not tokens:
        return None
    token = max(tokens, key=lambda item: item.bbox[2]) if at_end else min(
        tokens,
        key=lambda item: item.bbox[0],
    )
    distance = source.bbox[2] - token.bbox[2] if at_end else token.bbox[0] - source.bbox[0]
    return token if distance <= tolerance else None


def _merge_boxes(boxes: list[XYXY]) -> list[XYXY]:
    merged: list[XYXY] = []
    for bbox in sorted(boxes, key=lambda item: (item[0], item[1])):
        if merged and _intersect(merged[-1], bbox) is not None:
            previous = merged.pop()
            merged.append(_union((previous, bbox)))
        else:
            merged.append(bbox)
    return merged


def _is_punctuation_token(text: str) -> bool:
    visible = [char for char in str(text or "") if not char.isspace()]
    return bool(visible) and all(_is_punctuation_or_symbol(char) for char in visible)


def _is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def _is_small_repeated_mark(component: ForegroundComponent, line_height: int) -> bool:
    width = component.bbox[2] - component.bbox[0]
    height = component.bbox[3] - component.bbox[1]
    maximum = max(2, int(round(line_height * 0.45)))
    return width <= maximum and height <= maximum


def _co_baseline_groups(lines: tuple[DecorationLine, ...]) -> tuple[tuple[DecorationLine, ...], ...]:
    pending = sorted(lines, key=lambda source: (source.bbox[1], source.bbox[0]))
    groups: list[tuple[DecorationLine, ...]] = []
    while pending:
        members = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in pending[:]:
                if any(_same_horizontal_row(candidate.bbox, member.bbox) for member in members):
                    members.append(candidate)
                    pending.remove(candidate)
                    changed = True
        groups.append(tuple(members))
    return tuple(groups)


def _same_horizontal_row(left: XYXY, right: XYXY) -> bool:
    left_center = (left[1] + left[3]) / 2.0
    right_center = (right[1] + right[3]) / 2.0
    return right[1] <= left_center <= right[3] and left[1] <= right_center <= left[3]


def _dominant_aligned_run(
    components: tuple[ForegroundComponent, ...],
    line_height: int,
) -> tuple[ForegroundComponent, ...]:
    if not components:
        return ()
    tolerance = max(2.0, line_height * 0.18)
    groups: list[list[ForegroundComponent]] = []
    for component in sorted(components, key=lambda item: (_center_y(item.bbox), item.bbox[0])):
        center = _center_y(component.bbox)
        for group in groups:
            group_center = sum(_center_y(item.bbox) for item in group) / len(group)
            if abs(center - group_center) <= tolerance:
                group.append(component)
                break
        else:
            groups.append([component])
    return tuple(max(groups, key=lambda group: (_span_width(group), len(group))))


def _span_width(components: list[ForegroundComponent]) -> int:
    return max(item.bbox[2] for item in components) - min(item.bbox[0] for item in components)


def _merge_overlapping(decorations: list[TextDecoration]) -> list[TextDecoration]:
    merged: list[TextDecoration] = []
    for decoration in sorted(decorations, key=lambda item: (item.bbox[0], item.bbox[1])):
        if merged and _intersect(merged[-1].bbox, decoration.bbox) is not None:
            previous = merged.pop()
            merged.append(TextDecoration(DOT_LEADER, _union((previous.bbox, decoration.bbox))))
        else:
            merged.append(decoration)
    return merged


def _center_y(bbox: XYXY) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    bbox = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _union(boxes: tuple[XYXY, ...]) -> XYXY:
    return (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )


__all__ = [
    "DOT_LEADER",
    "DecorationLine",
    "DecorationToken",
    "TextDecoration",
    "detect_text_decorations",
    "detect_page_text_decorations",
]
