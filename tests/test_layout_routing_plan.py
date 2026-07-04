from __future__ import annotations

from app.services.layout_routing_plan import (
    RoutingLine,
    RoutingSegment,
    routing_line_to_record,
    routing_plan_for_block_record,
)
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    LAYOUT_ROUTE_SOURCE_FIELD,
    LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
    ROUTE_SUBBLOCKS_FIELD,
)


def test_routing_plan_exposes_lines_segments_and_text_slices():
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        "block_content": "甲 $ A $ 乙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [60, 10, 100, 45]},
        ],
    }

    plan = routing_plan_for_block_record(block, 240, 120)

    assert plan.has_layout_routes is True
    assert len(plan.lines) >= 1
    formula_segments = [
        segment
        for line in plan.lines
        for segment in line.segments
        if segment.kind == "formula"
    ]
    assert [(segment.label, segment.bbox, segment.text) for segment in formula_segments] == [
        ("inline_formula", (60, 10, 100, 45), "$ A $")
    ]
    assert any(route.carved for route in plan.text_slices)
    assert all(len(route.bbox) == 4 for route in plan.text_slices)


def test_routing_plan_preserves_ppocr_runtime_route_source():
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        LAYOUT_LINE_ROUTES_FIELD: [
            {
                "bbox": [0, 0, 120, 30],
                LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
                "segments": [{"kind": "text", "bbox": [0, 0, 120, 30], "text": ""}],
            }
        ],
    }

    plan = routing_plan_for_block_record(block, 240, 120)

    assert len(plan.lines) == 1
    assert plan.lines[0].source == LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS
    assert plan.lines[0].bbox == (0, 0, 120, 30)
    assert plan.text_slices[0].bbox == (0, 0, 120, 30)


def test_routing_line_to_record_serializes_runtime_cache_shape():
    line = RoutingLine(
        index=2,
        bbox=(10, 20, 90, 60),
        source=LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
        segments=(
            RoutingSegment(kind="text", bbox=(10, 20, 40, 60)),
            RoutingSegment(kind="formula", label="inline_formula", bbox=(40, 20, 70, 60), text="$ A $"),
            RoutingSegment(kind="text", bbox=(70, 20, 90, 60)),
        ),
    )

    record = routing_line_to_record(line)

    assert record == {
        "bbox": [10, 20, 90, 60],
        "segments": [
            {"kind": "text", "label": "", "bbox": [10, 20, 40, 60], "text": ""},
            {"kind": "formula", "label": "inline_formula", "bbox": [40, 20, 70, 60], "text": "$ A $"},
            {"kind": "text", "label": "", "bbox": [70, 20, 90, 60], "text": ""},
        ],
        LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
    }
