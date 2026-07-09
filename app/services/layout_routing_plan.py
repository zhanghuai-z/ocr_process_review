"""Typed read model for layout routing plans.

This module is the stable service boundary for consumers. The contract types
live in ``app.core.layout_routing_contract`` so both producers and consumers can
share the same typed model without depending on UI or service internals.
"""
from __future__ import annotations

from typing import Any

from app.core.paddle_line_routing import (
    LAYOUT_ROUTE_SOURCE_FIELD,
    layout_routing_plan_for_block,
)
from app.core.layout_routing_contract import (
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
    is_text_route_segment_kind,
    routing_line_from_record,
    routing_line_to_record as _routing_line_to_record,
    text_slice_from_record,
)


def routing_plan_for_block_record(
    block: dict[str, Any],
    width: int,
    height: int,
) -> RoutingPlan:
    return layout_routing_plan_for_block(block, width, height)


def routing_line_to_record(line: RoutingLine) -> dict[str, Any]:
    return _routing_line_to_record(line, source_field=LAYOUT_ROUTE_SOURCE_FIELD)


__all__ = [
    "RoutingLine",
    "RoutingPlan",
    "RoutingSegment",
    "TextSliceRoute",
    "is_text_route_segment_kind",
    "routing_line_from_record",
    "routing_line_to_record",
    "routing_plan_for_block_record",
    "text_slice_from_record",
]
