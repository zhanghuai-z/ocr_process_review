"""Typed read model for layout routing plans.

The current route generator still emits dictionaries because Hanwang routing
has many established tests. This module is the typed boundary for consumers:
new code should read ``RoutingPlan`` instead of reaching into
``_layout_line_routes`` or ``_route_subblocks`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.paddle_line_routing import (
    LAYOUT_ROUTE_SOURCE_FIELD,
    has_layout_line_routes,
    line_routes_for_block,
    text_slice_routes_for_block,
)


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class RoutingSegment:
    kind: str
    bbox: XYXY
    label: str = ""
    text: str = ""


@dataclass(frozen=True)
class RoutingLine:
    index: int
    bbox: XYXY
    segments: tuple[RoutingSegment, ...]
    source: str = ""

    @property
    def has_formula(self) -> bool:
        return any(segment.kind == "formula" for segment in self.segments)


@dataclass(frozen=True)
class TextSliceRoute:
    line_index: int
    segment_index: int
    bbox: XYXY
    carved: bool


@dataclass(frozen=True)
class RoutingPlan:
    lines: tuple[RoutingLine, ...]
    text_slices: tuple[TextSliceRoute, ...]
    has_layout_routes: bool


def routing_plan_for_block_record(
    block: dict[str, Any],
    width: int,
    height: int,
) -> RoutingPlan:
    lines = tuple(
        _routing_line_from_dict(index, route)
        for index, route in enumerate(line_routes_for_block(block, width, height))
    )
    text_slices = tuple(
        _text_slice_from_dict(route)
        for route in text_slice_routes_for_block(block, width, height)
    )
    return RoutingPlan(
        lines=lines,
        text_slices=text_slices,
        has_layout_routes=has_layout_line_routes(block, width, height),
    )


def _routing_line_from_dict(index: int, route: dict[str, Any]) -> RoutingLine:
    return RoutingLine(
        index=index,
        bbox=_xyxy(route.get("bbox")),
        segments=tuple(
            _routing_segment_from_dict(segment)
            for segment in route.get("segments", [])
            if isinstance(segment, dict)
        ),
        source=str(route.get(LAYOUT_ROUTE_SOURCE_FIELD) or ""),
    )


def _routing_segment_from_dict(segment: dict[str, Any]) -> RoutingSegment:
    return RoutingSegment(
        kind=str(segment.get("kind") or "text"),
        label=str(segment.get("label") or ""),
        bbox=_xyxy(segment.get("bbox")),
        text=str(segment.get("text") or ""),
    )


def routing_line_to_record(line: RoutingLine) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bbox": list(line.bbox),
        "segments": [
            {
                "kind": segment.kind,
                "label": segment.label,
                "bbox": list(segment.bbox),
                "text": segment.text,
            }
            for segment in line.segments
        ],
    }
    if line.source:
        record[LAYOUT_ROUTE_SOURCE_FIELD] = line.source
    return record


def _text_slice_from_dict(route: dict[str, Any]) -> TextSliceRoute:
    return TextSliceRoute(
        line_index=_int_or_default(route.get("line_idx"), -1),
        segment_index=_int_or_default(route.get("segment_idx"), 0),
        bbox=_xyxy(route.get("bbox")),
        carved=bool(route.get("carved", False)),
    )


def _xyxy(value: object) -> XYXY:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return (0, 0, 0, 0)
    try:
        x1, y1, x2, y2 = (int(item) for item in value)
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RoutingLine",
    "RoutingPlan",
    "RoutingSegment",
    "TextSliceRoute",
    "routing_line_to_record",
    "routing_plan_for_block_record",
]
