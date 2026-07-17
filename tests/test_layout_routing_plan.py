from __future__ import annotations

import pytest

from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    PpOcrLatinTokenObservation,
    PpOcrSymbolObservation,
    RouteValidationIssue,
    RoutingPlan,
    TextSliceRoute,
    routing_line_from_record,
    routing_segment_from_record,
)
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


def test_page_routing_plan_freezes_nested_route_collections():
    line = RoutingLine(
        index=0,
        bbox=(0, 0, 20, 20),
        segments=[RoutingSegment(kind="text_other", bbox=(0, 0, 20, 20))],
    )
    plan = PageRoutingPlan(
        page_uid="page-1",
        prepass_run_id="run-1",
        blocks=[
            BlockRoutingPlan(
                block_uid="text-1",
                plan=RoutingPlan(lines=[line], text_slices=[], has_layout_routes=True),
            ),
        ],
    )

    assert isinstance(plan.blocks, tuple)
    assert isinstance(plan.blocks[0].plan.lines, tuple)
    assert isinstance(plan.blocks[0].plan.lines[0].segments, tuple)


def test_routing_dto_bboxes_copy_external_lists_to_immutable_tuples():
    token_bbox = [12, 10, 30, 28]
    segment_bbox = [10, 8, 40, 32]
    content_bbox = [10, 4, 40, 36]
    line_bbox = [0, 0, 50, 40]
    slice_bbox = [10, 8, 40, 32]
    issue_bbox = [50, 50, 60, 60]
    token = PpOcrLatinTokenObservation("AB", token_bbox)
    segment = RoutingSegment(
        kind="text_latin",
        bbox=segment_bbox,
        ppocr_latin_tokens=(token,),
    )
    formula = RoutingSegment(kind="formula", bbox=segment_bbox, content_bbox=content_bbox)
    line = RoutingLine(index=0, bbox=line_bbox, segments=[segment])
    text_slice = TextSliceRoute(0, 0, slice_bbox, True, kind="text_latin")
    issue = RouteValidationIssue("test", "test", 0, issue_bbox)

    token_bbox[0] = 99
    segment_bbox[0] = 99
    content_bbox[0] = 99
    line_bbox[0] = 99
    slice_bbox[0] = 99
    issue_bbox[0] = 99

    assert token.bbox == (12, 10, 30, 28)
    assert segment.bbox == (10, 8, 40, 32)
    assert formula.content_bbox == (10, 4, 40, 36)
    assert line.bbox == (0, 0, 50, 40)
    assert text_slice.bbox == (10, 8, 40, 32)
    assert issue.bbox == (50, 50, 60, 60)
    assert all(isinstance(value, int) for value in (*token.bbox, *segment.bbox, *line.bbox))

    with pytest.raises((TypeError, ValueError)):
        RoutingSegment(kind="text_other", bbox=(0, 0, 10))


def test_routing_plan_preserves_ppocr_runtime_route_source():
    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        LAYOUT_LINE_ROUTES_FIELD: [
            {
                "bbox": [0, 0, 120, 30],
                LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
                "segments": [{"kind": "text_other", "bbox": [0, 0, 120, 30], "text": ""}],
            }
        ],
    }

    plan = routing_plan_for_block_record(block, 240, 120)

    assert len(plan.lines) == 1
    assert plan.lines[0].source == LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS
    assert plan.lines[0].bbox == (0, 0, 120, 30)
    assert plan.text_slices[0].bbox == (0, 0, 120, 30)


def test_routing_contract_rejects_retired_and_catch_all_segment_kinds():
    for kind in ("text", "text_zh", "text_mixed", "unknown"):
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
                    {"kind": "text_other", "bbox": [0, 0, 60, 30], "text": ""},
                    {"kind": "formula", "bbox": [60, 0, 100, 30], "text": "$ A $"},
                    {"kind": "text_latin", "bbox": [100, 0, 160, 30], "text": ""},
                ],
            }
        ],
    }

    plan = routing_plan_for_block_record(block, 240, 120)

    assert [segment.kind for segment in plan.lines[0].segments] == [
        "text_other",
        "formula",
        "text_latin",
    ]
    assert [(route.segment_index, route.bbox) for route in plan.text_slices] == [
        (0, (0, 0, 60, 30)),
        (2, (100, 0, 160, 30)),
    ]
    assert [route.kind for route in plan.text_slices] == ["text_other", "text_latin"]


def test_text_slice_route_rejects_non_text_kind():
    with pytest.raises(ValueError):
        TextSliceRoute(
            line_index=0,
            segment_index=1,
            bbox=(0, 0, 10, 10),
            carved=True,
            kind="formula",
        )


def test_routing_line_to_record_serializes_runtime_cache_shape():
    line = RoutingLine(
        index=2,
        bbox=(10, 20, 90, 60),
        source=LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
        segments=(
            RoutingSegment(kind="text_other", bbox=(10, 20, 40, 60)),
            RoutingSegment(kind="formula", label="inline_formula", bbox=(40, 20, 70, 60), text="$ A $"),
            RoutingSegment(kind="text_other", bbox=(70, 20, 90, 60)),
        ),
    )

    record = routing_line_to_record(line)

    assert record == {
        "bbox": [10, 20, 90, 60],
        "segments": [
            {"kind": "text_other", "label": "", "bbox": [10, 20, 40, 60], "text": ""},
            {"kind": "formula", "label": "inline_formula", "bbox": [40, 20, 70, 60], "text": "$ A $"},
            {"kind": "text_other", "label": "", "bbox": [70, 20, 90, 60], "text": ""},
        ],
        LAYOUT_ROUTE_SOURCE_FIELD: LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
    }


def test_rotated_routing_line_round_trips_axis_and_orientation():
    line = RoutingLine(
        index=0,
        bbox=(20, 10, 60, 180),
        segments=(RoutingSegment(kind="text_other", bbox=(20, 10, 60, 180)),),
        source=LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
        text_axis="vertical",
        orientation_angle=180,
    )

    record = routing_line_to_record(line)
    restored = routing_line_from_record(
        0,
        record,
        source_field=LAYOUT_ROUTE_SOURCE_FIELD,
    )

    assert record["text_axis"] == "vertical"
    assert record["orientation_angle"] == 180
    assert restored == line


def test_formula_segment_round_trips_distinct_mask_and_content_geometry():
    line = RoutingLine(
        index=0,
        bbox=(0, 30, 180, 60),
        segments=(
            RoutingSegment(
                kind="formula",
                label="inline_formula",
                bbox=(70, 30, 110, 60),
                content_bbox=(70, 10, 110, 82),
                text="$ A $",
            ),
        ),
    )

    record = routing_line_to_record(line)
    restored = routing_segment_from_record(record["segments"][0])

    assert record["segments"][0]["bbox"] == [70, 30, 110, 60]
    assert record["segments"][0]["content_bbox"] == [70, 10, 110, 82]
    assert restored.bbox == (70, 30, 110, 60)
    assert restored.content_bbox == (70, 10, 110, 82)


def test_latin_token_observations_round_trip_only_on_latin_routes():
    segment = RoutingSegment(
        kind="text_latin",
        bbox=(20, 10, 90, 50),
        text="UrbanCrisis",
        ppocr_latin_tokens=(
            PpOcrLatinTokenObservation("Urban", (20, 10, 52, 50)),
            PpOcrLatinTokenObservation("Crisis", (58, 10, 90, 50)),
        ),
    )
    line = RoutingLine(index=0, bbox=(0, 0, 100, 60), segments=(segment,))

    record = routing_line_to_record(line)
    restored = routing_segment_from_record(record["segments"][0])

    assert record["segments"][0]["ppocr_latin_tokens"] == [
        {"text": "Urban", "bbox": [20, 10, 52, 50]},
        {"text": "Crisis", "bbox": [58, 10, 90, 50]},
    ]
    assert restored.ppocr_latin_tokens == segment.ppocr_latin_tokens


def test_symbol_observations_round_trip_on_physical_routing_line():
    observation = PpOcrSymbolObservation(
        text="“",
        bbox=(24, 12, 32, 28),
        proposal_bbox=(27, 8, 35, 32),
    )
    line = RoutingLine(
        index=0,
        bbox=(0, 0, 100, 40),
        segments=(RoutingSegment(kind="text_other", bbox=(0, 0, 100, 40)),),
        ppocr_symbol_observations=(observation,),
    )

    record = routing_line_to_record(line)
    restored = routing_line_from_record(
        0,
        record,
        source_field=LAYOUT_ROUTE_SOURCE_FIELD,
    )

    assert record["ppocr_symbol_observations"] == [{
        "text": "“",
        "bbox": [24, 12, 32, 28],
        "proposal_bbox": [27, 8, 35, 32],
    }]
    assert restored.ppocr_symbol_observations == (observation,)


def test_hanwang_assembles_explicit_text_segment_kinds():
    import app.engines.hanwang.micro_recblock as micro_module

    route = RoutingLine(
        index=0,
        bbox=(0, 0, 160, 30),
        segments=(
            RoutingSegment(kind="text_other", bbox=(0, 0, 60, 30)),
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


def test_hanwang_merges_low_sitting_latin_slice_into_its_physical_routing_line():
    import app.engines.hanwang.micro_recblock as micro_module

    route = RoutingLine(
        index=0,
        bbox=(0, 0, 160, 50),
        segments=(
            RoutingSegment(kind="text_latin", bbox=(0, 0, 100, 45), text="China"),
            RoutingSegment(kind="text_other", bbox=(100, 0, 120, 45)),
            RoutingSegment(kind="text_latin", bbox=(120, 20, 140, 45), text="s"),
        ),
    )
    grouped = {
        (0, 0, 0): [micro_module.LineResult(
            text="China",
            bbox=(0, 2, 98, 42),
            chars=[micro_module.CharResult(text="China", bbox=(0, 2, 98, 42))],
        )],
        (0, 0, 1): [micro_module.LineResult(
            text="’",
            bbox=(102, 4, 112, 18),
            chars=[micro_module.CharResult(text="’", bbox=(102, 4, 112, 18))],
        )],
        (0, 0, 2): [micro_module.LineResult(
            text="s",
            bbox=(122, 25, 138, 43),
            chars=[micro_module.CharResult(text="s", bbox=(122, 25, 138, 43))],
        )],
    }

    lines = micro_module._assemble_layout_route_line(
        block_idx=0,
        line_idx=0,
        route=route,
        grouped_lines=grouped,
    )

    assert len(lines) == 1
    assert lines[0].text == "China’s"
    assert [char.text for char in lines[0].chars] == ["China", "’", "s"]
    assert lines[0].bbox == route.bbox
    assert lines[0].bbox_source == "ppocrv6_physical_routing_line"


def test_hanwang_keeps_ppocr_physical_row_when_native_group_is_partial():
    import app.engines.hanwang.micro_recblock as micro_module

    route = RoutingLine(
        index=0,
        bbox=(100, 200, 700, 290),
        segments=(RoutingSegment(kind="text_other", bbox=(100, 200, 700, 290)),),
    )
    grouped = {
        (0, 0, 0): [micro_module.LineResult(
            text="第二章",
            bbox=(108, 205, 350, 250),
            chars=[
                micro_module.CharResult(text="第", bbox=(108, 205, 170, 250)),
                micro_module.CharResult(text="二", bbox=(180, 220, 250, 235)),
                micro_module.CharResult(text="章", bbox=(270, 205, 350, 250)),
            ],
        )],
    }

    lines = micro_module._assemble_layout_route_line(
        block_idx=0,
        line_idx=0,
        route=route,
        grouped_lines=grouped,
    )

    assert len(lines) == 1
    assert lines[0].bbox == route.bbox
    assert lines[0].bbox_source == "ppocrv6_physical_routing_line"
    assert [char.bbox for char in lines[0].chars] == [
        (108, 205, 170, 250),
        (180, 220, 250, 235),
        (270, 205, 350, 250),
    ]


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
                    {"kind": "text_other", "bbox": [0, 0, 70, 40]},
                {"kind": "formula", "bbox": [70, 0, 100, 40], "text": "$ ^{②} $"},
                    {"kind": "text_other", "bbox": [100, 0, 300, 40]},
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
            "segments": [{"kind": "text_other", "bbox": [0, 0, 90, 30], "text": ""}],
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


def test_ppocr_latin_line_hint_marks_text_latin_segment():
    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 50],
        "block_content": "Urban Crisis 2026",
    }

    attachment = build_page_ocr_line_route_attachment(
        [parent],
        [PageOcrLineHint(text="Urban Crisis 2026", bbox=(0, 0, 220, 40))],
        240,
        80,
    )

    route = attachment.route_records_by_block_index[0][0]
    assert [segment["kind"] for segment in route["segments"]] == ["text_latin"]


def test_ppocr_cjk_line_hint_marks_text_other_segment():
    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 50],
        "block_content": "城市生产率2026",
    }

    attachment = build_page_ocr_line_route_attachment(
        [parent],
        [PageOcrLineHint(text="城市生产率2026", bbox=(0, 0, 220, 40))],
        240,
        80,
    )

    route = attachment.route_records_by_block_index[0][0]
    assert [segment["kind"] for segment in route["segments"]] == ["text_other"]


def test_ppocr_mixed_line_hint_stays_other_until_splitter_runs():
    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 50],
        "block_content": "城市 Urban Crisis",
    }

    attachment = build_page_ocr_line_route_attachment(
        [parent],
        [PageOcrLineHint(text="城市 Urban Crisis", bbox=(0, 0, 220, 40))],
        240,
        80,
    )

    route = attachment.route_records_by_block_index[0][0]
    assert [segment["kind"] for segment in route["segments"]] == ["text_other"]


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
