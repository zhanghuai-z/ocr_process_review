from __future__ import annotations

import numpy as np
import pytest

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.engines.hanwang.micro_recblock import (
    BlockResult,
    CharResult,
    HanwangMicroRecBlockEngine,
    LineResult,
    RunStats,
    _LineCutMaskedLineRoute,
    _TextRoute,
    _compile_native_route_map,
    _materialize_linecut_masked_page,
    _ppocr_symbol_route_observation,
)


def test_linecut_canvas_whitens_only_exact_formula_intersection():
    text = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(0, 40, 100, 80),
        kind="text_other",
    )
    formula = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=1,
        bbox=(20, 40, 50, 48),
        kind="formula",
        content_bbox=(20, 10, 50, 48),
    )
    route = _LineCutMaskedLineRoute(
        block_idx=0,
        line_idx=0,
        bbox=(0, 40, 100, 80),
        linecut_segments=(text,),
        excluded_segments=(formula,),
    )
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    canvas = _materialize_linecut_masked_page(image, [route])

    assert np.all(canvas[44, 25] == 255)
    assert np.all(canvas[60, 25] == 0)
    assert np.all(canvas[60, 75] == 0)
from app.models import BBox, Block, BlockType, OcrPolicy, Page
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection
from app.models.charocr_routing import COMPONENT_GROUPING_SINGLE_GLYPH
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)


def test_single_glyph_route_uses_explicit_ppocr_text_and_component_geometry():
    route = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(100, 20, 150, 90),
        kind="text_symbol",
        content_bbox=(108, 28, 132, 79),
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
        ppocr_punctuation_candidate="?",
    )

    line = _ppocr_symbol_route_observation(route)

    assert line.text == "?"
    assert line.bbox == (108, 28, 132, 79)
    assert line.chars[0].bbox == (108, 28, 132, 79)
    assert line.chars[0].source == "ppocrv6:single_glyph_punctuation"


def test_single_glyph_route_rejects_missing_canonical_geometry():
    route = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(100, 20, 150, 90),
        kind="text_symbol",
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
        ppocr_punctuation_candidate="’",
    )
    with pytest.raises(RuntimeError, match="missing text or canonical"):
        _ppocr_symbol_route_observation(route)


def test_native_runner_honors_explicit_skip_policy_for_unknown_label():
    import app.engines.hanwang.micro_recblock as micro_module

    figure = Block(
        block_type=BlockType.FIGURE,
        bbox=BBox.from_xyxy(10, 10, 90, 50),
        source_label="header_image",
        ocr_policy=OcrPolicy.SKIP,
    )
    page = Page(image_path="", width=100, height=60, blocks=[figure])
    sync_page_layout_snapshot_from_projection(page, source_engine="test")
    plan = PageRoutingPlan(
        page_uid=page.uid,
        prepass_run_id="test-skip-policy",
        blocks=(),
    )

    rows, stats = micro_module.run_micro_recblock(
        np.full((60, 100, 3), 255, dtype=np.uint8),
        [{
            "block_label": "header_image",
            "block_bbox": [10, 10, 90, 50],
            "block_content": "",
            micro_module.ROUTE_ROW_LAYOUT_BLOCK_UID_KEY: figure.uid,
            micro_module.ROUTE_ROW_OCR_POLICY_KEY: OcrPolicy.SKIP.value,
        }],
        routing_plan=plan,
        page=page,
        include_chars=True,
    )

    assert stats.n_blocks_hanwang == 0
    assert stats.n_blocks_ppvl == 1
    assert rows[0].source == "ppvl"


def test_hanwang_receives_only_explicit_page_routing_plan_for_text_blocks():
    text = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(0, 0, 280, 60),
        source_label="text",
        ocr_policy=OcrPolicy.TEXT_OCR,
    )
    table = Block(
        block_type=BlockType.TABLE,
        bbox=BBox.from_xyxy(80, 0, 180, 60),
        source_label="table",
        ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
    )
    page = Page(image_path="/tmp/route-boundary.png", width=300, height=100, blocks=[text, table])
    snapshot = sync_page_layout_snapshot_from_projection(page, source_engine="test")
    routing_plan = compile_page_routing_plan(
        snapshot,
        PpOcrV6PrepassArtifact(
            page_uid=page.uid,
            run_id="ppocr-job-1",
            lines=(PpOcrV6LineHint(index=0, text="正文", bbox=(0, 0, 280, 60), words=()),),
        ),
        page_width=page.width,
        page_height=page.height,
    )
    captured: dict = {}

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        captured["blocks"] = ppvl_blocks
        captured["kwargs"] = kwargs
        rows = []
        for index, row in enumerate(ppvl_blocks):
            label = row["block_label"]
            bbox = tuple(row["block_bbox"])
            lines = []
            if label == "text":
                lines = [LineResult(text="正文", bbox=bbox, confidence=0.9)]
            rows.append(BlockResult(
                block_idx=index,
                block_label=label,
                block_bbox=bbox,
                source="hanwang" if label == "text" else "ppvl",
                text="正文" if label == "text" else "",
                ppvl_text="",
                lines=lines,
                raw_block=dict(row),
            ))
        return rows, RunStats(n_blocks_total=len(rows), n_blocks_hanwang=1, n_blocks_ppvl=1)

    HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((100, 300, 3), dtype=np.uint8),
        page,
        routing_plan=routing_plan,
    )

    assert "page_ocr_lines" not in captured["kwargs"]
    assert captured["kwargs"]["routing_plan"] is routing_plan
    # The engine hands an immutable plan to the native-runner boundary.  The
    # runner alone projects it to its temporary row dictionaries, so an
    # injected runner never receives stale PP-OCR line attachments.
    assert all("_layout_line_routes" not in row for row in captured["blocks"])


def test_native_route_map_keeps_typed_routes_out_of_ppvl_rows():
    text = Block(BlockType.TEXT, BBox.from_xyxy(0, 0, 280, 60), source_label="text")
    table = Block(
        BlockType.TABLE,
        BBox.from_xyxy(80, 0, 180, 60),
        source_label="table",
        ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
    )
    page = Page(image_path="/tmp/route-projection.png", width=300, height=100, blocks=[text, table])
    snapshot = sync_page_layout_snapshot_from_projection(page, source_engine="test")
    plan = compile_page_routing_plan(
        snapshot,
        PpOcrV6PrepassArtifact(
            page_uid=page.uid,
            run_id="ppocr-job-2",
            lines=(PpOcrV6LineHint(index=0, text="正文", bbox=(0, 0, 280, 60), words=()),),
        ),
        page_width=page.width,
        page_height=page.height,
    )
    native_rows = [
        {
            "block_label": "text",
            "_layout_block_uid": text.uid,
            "_layout_block_ocr_policy": OcrPolicy.TEXT_OCR.value,
        },
        {
            "block_label": "table",
            "_layout_block_uid": table.uid,
            "_layout_block_ocr_policy": OcrPolicy.PRESERVE_AS_TABLE.value,
        },
    ]

    route_map = _compile_native_route_map(native_rows, plan, page)

    assert route_map[0][0].segments[1].kind == "skip"
    assert "_layout_line_routes" not in native_rows[0]
    assert "_layout_line_routes" not in native_rows[1]


def test_native_route_map_rejects_text_row_missing_from_page_plan():
    text = Block(BlockType.TEXT, BBox.from_xyxy(0, 0, 120, 40), source_label="text")
    page = Page(image_path="/tmp/route-missing-row.png", width=180, height=80, blocks=[text])
    snapshot = sync_page_layout_snapshot_from_projection(page, source_engine="test")
    plan = compile_page_routing_plan(
        snapshot,
        PpOcrV6PrepassArtifact(
            page_uid=page.uid,
            run_id="ppocr-job-3",
            lines=(PpOcrV6LineHint(index=0, text="正文", bbox=(0, 0, 120, 40), words=()),),
        ),
        page_width=page.width,
        page_height=page.height,
    )

    with pytest.raises(RuntimeError, match="outside the explicit page routing plan"):
        _compile_native_route_map(
            [
                {
                    "block_label": "text",
                    "_layout_block_uid": text.uid,
                    "_layout_block_ocr_policy": OcrPolicy.TEXT_OCR.value,
                },
                {
                    "block_label": "text",
                    "_layout_block_uid": "unexpected-text-block",
                    "_layout_block_ocr_policy": OcrPolicy.TEXT_OCR.value,
                },
            ],
            plan,
            page,
        )
