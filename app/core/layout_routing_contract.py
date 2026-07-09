"""Typed layout routing contract shared by routing producers and consumers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


XYXY = tuple[int, int, int, int]
ROUTE_SEGMENT_TEXT = "text"
ROUTE_SEGMENT_TEXT_ZH = "text_zh"
ROUTE_SEGMENT_TEXT_LATIN = "text_latin"
ROUTE_SEGMENT_FORMULA = "formula"
ROUTE_SEGMENT_SKIP = "skip"
TEXT_ROUTE_SEGMENT_KINDS = frozenset({
    ROUTE_SEGMENT_TEXT,
    ROUTE_SEGMENT_TEXT_ZH,
    ROUTE_SEGMENT_TEXT_LATIN,
})
VALID_ROUTE_SEGMENT_KINDS = frozenset({
    *TEXT_ROUTE_SEGMENT_KINDS,
    ROUTE_SEGMENT_FORMULA,
    ROUTE_SEGMENT_SKIP,
})


@dataclass(frozen=True)
class RoutingSegment:
    kind: str
    bbox: XYXY
    label: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalize_route_segment_kind(self.kind))


@dataclass(frozen=True)
class RoutingLine:
    index: int
    bbox: XYXY
    segments: tuple[RoutingSegment, ...]
    source: str = ""

    @property
    def has_formula(self) -> bool:
        return any(segment.kind == ROUTE_SEGMENT_FORMULA for segment in self.segments)


def is_text_route_segment_kind(kind: str) -> bool:
    return kind in TEXT_ROUTE_SEGMENT_KINDS


def normalize_route_segment_kind(value: object) -> str:
    kind = str(value or ROUTE_SEGMENT_TEXT).strip() or ROUTE_SEGMENT_TEXT
    if kind not in VALID_ROUTE_SEGMENT_KINDS:
        raise ValueError(f"unsupported layout route segment kind: {kind!r}")
    return kind


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


def routing_line_from_record(
    index: int,
    route: dict[str, Any],
    *,
    source_field: str,
) -> RoutingLine:
    return RoutingLine(
        index=index,
        bbox=xyxy(route.get("bbox")),
        segments=tuple(
            routing_segment_from_record(segment)
            for segment in route.get("segments", [])
            if isinstance(segment, dict)
        ),
        source=str(route.get(source_field) or ""),
    )


def routing_segment_from_record(segment: dict[str, Any]) -> RoutingSegment:
    return RoutingSegment(
        kind=normalize_route_segment_kind(segment.get("kind")),
        label=str(segment.get("label") or ""),
        bbox=xyxy(segment.get("bbox")),
        text=str(segment.get("text") or ""),
    )


def routing_line_to_record(
    line: RoutingLine,
    *,
    source_field: str,
) -> dict[str, Any]:
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
        record[source_field] = line.source
    return record


def text_slice_from_record(route: dict[str, Any]) -> TextSliceRoute:
    return TextSliceRoute(
        line_index=int_or_default(route.get("line_idx"), -1),
        segment_index=int_or_default(route.get("segment_idx"), 0),
        bbox=xyxy(route.get("bbox")),
        carved=bool(route.get("carved", False)),
    )


def text_slice_to_record(route: TextSliceRoute) -> dict[str, Any]:
    return {
        "line_idx": route.line_index,
        "segment_idx": route.segment_index,
        "bbox": list(route.bbox),
        "carved": route.carved,
    }


def xyxy(value: object) -> XYXY:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return (0, 0, 0, 0)
    try:
        x1, y1, x2, y2 = (int(item) for item in value)
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RoutingLine",
    "RoutingPlan",
    "RoutingSegment",
    "ROUTE_SEGMENT_FORMULA",
    "ROUTE_SEGMENT_SKIP",
    "ROUTE_SEGMENT_TEXT",
    "ROUTE_SEGMENT_TEXT_LATIN",
    "ROUTE_SEGMENT_TEXT_ZH",
    "TEXT_ROUTE_SEGMENT_KINDS",
    "TextSliceRoute",
    "VALID_ROUTE_SEGMENT_KINDS",
    "is_text_route_segment_kind",
    "normalize_route_segment_kind",
    "routing_line_from_record",
    "routing_line_to_record",
    "routing_segment_from_record",
    "text_slice_from_record",
    "text_slice_to_record",
    "xyxy",
]
