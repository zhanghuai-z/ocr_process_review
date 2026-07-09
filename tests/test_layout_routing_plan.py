from __future__ import annotations

import pytest

from app.core.layout_routing_contract import routing_segment_from_record
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
    PageOcrLineHint,
    ROUTE_SUBBLOCKS_FIELD,
    apply_page_ocr_line_route_attachment,
    build_layout_line_routes,
    build_layout_routing_plan,
    build_page_ocr_line_route_attachment,
    layout_routing_plan_for_block,
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


def test_routing_contract_rejects_catch_all_segment_kinds():
    for kind in ("text_mixed", "unknown"):
        with pytest.raises(ValueError):
            routing_segment_from_record({"kind": kind, "bbox": [0, 0, 10, 10]})


def test_explicit_text_segment_kinds_are_text_slices():
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        LAYOUT_LINE_ROUTES_FIELD: [
            {
                "bbox": [0, 0, 160, 30],
                LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
                "segments": [
                    {"kind": "text_zh", "bbox": [0, 0, 60, 30], "text": ""},
                    {"kind": "formula", "bbox": [60, 0, 100, 30], "text": "$ A $"},
                    {"kind": "text_latin", "bbox": [100, 0, 160, 30], "text": ""},
                ],
            }
        ],
    }

    plan = routing_plan_for_block_record(block, 240, 120)

    assert [segment.kind for segment in plan.lines[0].segments] == [
        "text_zh",
        "formula",
        "text_latin",
    ]
    assert [(route.segment_index, route.bbox) for route in plan.text_slices] == [
        (0, (0, 0, 60, 30)),
        (2, (100, 0, 160, 30)),
    ]


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


def test_hanwang_assembles_explicit_text_segment_kinds():
    import app.engines.hanwang.micro_recblock as micro_module

    route = RoutingLine(
        index=0,
        bbox=(0, 0, 160, 30),
        segments=(
            RoutingSegment(kind="text_zh", bbox=(0, 0, 60, 30)),
            RoutingSegment(kind="formula", label="inline_formula", bbox=(60, 0, 100, 30), text="$ A $"),
            RoutingSegment(kind="text_latin", bbox=(100, 0, 160, 30)),
        ),
    )
    grouped = {
        (0, 0, 0): [
            micro_module.LineResult(
                text="甲",
                bbox=(0, 0, 60, 30),
                confidence=0.9,
                chars=[micro_module.CharResult(text="甲", confidence=0.9, bbox=(0, 0, 20, 30))],
            )
        ],
        (0, 0, 2): [
            micro_module.LineResult(
                text="abc",
                bbox=(100, 0, 160, 30),
                confidence=0.9,
                chars=[
                    micro_module.CharResult(text="a", confidence=0.9, bbox=(100, 0, 112, 30)),
                    micro_module.CharResult(text="b", confidence=0.9, bbox=(116, 0, 128, 30)),
                    micro_module.CharResult(text="c", confidence=0.9, bbox=(132, 0, 144, 30)),
                ],
            )
        ],
    }

    lines = micro_module._assemble_layout_route_line(
        block_idx=0,
        line_idx=0,
        route=route,
        grouped_lines=grouped,
    )

    assert len(lines) == 1
    assert lines[0].text == "甲$ A $abc"
    assert [char.text for char in lines[0].chars] == ["甲", "$ A $", "a", "b", "c"]


def test_paddle_routing_producer_builds_typed_plan_before_legacy_records():
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        "block_content": "甲 $ A $ 乙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [60, 10, 100, 45]},
        ],
    }

    plan = build_layout_routing_plan(block, 240, 120)
    records = build_layout_line_routes(block, 240, 120)

    assert plan.has_layout_routes is True
    assert [routing_line_to_record(line) for line in plan.lines] == records
    assert any(segment.kind == "formula" for line in plan.lines for segment in line.segments)


def test_layout_routing_plan_ignores_stale_cache_without_mutating_block():
    stale_cache = [
        {
            "bbox": [0, 0, 300, 40],
            LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
            "segments": [
                {"kind": "text", "bbox": [0, 0, 70, 40]},
                {"kind": "formula", "bbox": [70, 0, 100, 40], "text": "$ ^{②} $"},
                {"kind": "text", "bbox": [100, 0, 300, 40]},
            ],
        }
    ]
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 300, 40],
        "block_content": "甲 $ ^{②} $ 乙 $ B $ 丙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [70, 0, 100, 30]},
            {"block_label": "inline_formula", "block_bbox": [170, 0, 200, 30]},
        ],
        LAYOUT_LINE_ROUTES_FIELD: stale_cache,
    }

    plan = layout_routing_plan_for_block(block, 320, 60)

    assert block[LAYOUT_LINE_ROUTES_FIELD] is stale_cache
    assert [
        segment.text
        for line in plan.lines
        for segment in line.segments
        if segment.kind == "formula"
    ] == ["$ B $"]


def test_page_ocr_line_route_attachment_builds_without_mutating_blocks():
    stale_cache = [
        {
            "bbox": [0, 0, 90, 30],
            LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
            "segments": [{"kind": "text", "bbox": [0, 0, 90, 30], "text": ""}],
        }
    ]
    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 180, 50],
        "block_content": "甲 $ A $ 乙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [55, 0, 95, 35]},
        ],
        LAYOUT_LINE_ROUTES_FIELD: stale_cache,
    }

    attachment = build_page_ocr_line_route_attachment(
        [parent],
        [PageOcrLineHint(text="甲 A 乙", bbox=(0, 0, 180, 40))],
        200,
        80,
    )

    assert parent[LAYOUT_LINE_ROUTES_FIELD] is stale_cache
    assert 0 in attachment.route_records_by_block_index
    apply_page_ocr_line_route_attachment([parent], attachment)
    assert parent[LAYOUT_LINE_ROUTES_FIELD] is attachment.route_records_by_block_index[0]


def test_page_ocr_line_route_attachment_clears_stale_routes_explicitly():
    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 100, 40],
        LAYOUT_LINE_ROUTES_FIELD: [{"bbox": [0, 0, 100, 40], "segments": []}],
    }

    attachment = build_page_ocr_line_route_attachment(
        [parent],
        [PageOcrLineHint(text="unrelated", bbox=(200, 200, 260, 230))],
        300,
        300,
    )

    assert 0 in attachment.clear_block_indices
    assert LAYOUT_LINE_ROUTES_FIELD in parent
    apply_page_ocr_line_route_attachment([parent], attachment)
    assert LAYOUT_LINE_ROUTES_FIELD not in parent
