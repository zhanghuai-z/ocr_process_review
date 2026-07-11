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
    _TextRoute,
    _compile_native_route_map,
    _apply_single_glyph_route_contract,
)
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


def test_single_glyph_route_unions_detached_native_component_geometry():
    route = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(100, 20, 150, 90),
        kind="text_other",
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
    )
    lines = [LineResult(
        text="?",
        bbox=(108, 28, 132, 64),
        chars=[CharResult(text="?", bbox=(108, 28, 132, 64))],
    )]

    _apply_single_glyph_route_contract(
        route,
        lines,
        [
            {"segimg_group_bbox": [108, 28, 132, 64]},
            {"segimg_group_bbox": [114, 70, 123, 79]},
        ],
    )

    assert lines[0].bbox == (108, 28, 132, 79)
    assert lines[0].chars[0].bbox == (108, 28, 132, 79)
    assert lines[0].chars[0].source.endswith(":native_component_union")


def test_single_glyph_route_selects_only_an_existing_native_symbol_candidate():
    route = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(100, 20, 150, 90),
        kind="text_other",
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
        ppocr_punctuation_candidate="’",
    )
    lines = [LineResult(
        text="，",
        bbox=(108, 28, 132, 64),
        chars=[CharResult(text="，", bbox=(108, 28, 132, 64), candidates=["，", "’", "'"])],
    )]

    _apply_single_glyph_route_contract(route, lines, [])

    assert lines[0].text == "’"
    assert lines[0].chars[0].text == "’"
    assert lines[0].chars[0].source.endswith(":native_candidate_selected_by_ppocr_punctuation")


def test_single_glyph_route_does_not_inject_a_symbol_missing_from_native_candidates():
    route = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(100, 20, 150, 90),
        kind="text_other",
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
        ppocr_punctuation_candidate="’",
    )
    lines = [LineResult(
        text="，",
        bbox=(108, 28, 132, 64),
        chars=[CharResult(text="，", bbox=(108, 28, 132, 64), candidates=["，", "。"])],
    )]

    _apply_single_glyph_route_contract(route, lines, [])

    assert lines[0].text == "，"
    assert lines[0].chars[0].text == "，"


def test_single_glyph_route_without_segimg_group_uses_route_context(monkeypatch):
    import app.engines.hanwang.micro_recblock as micro_module

    def code(char: str) -> int:
        return int.from_bytes(char.encode("gbk"), "little")

    def fake_segimg(_image_bgr, *, recblocks_xyxy=None, timeout=0):
        assert recblocks_xyxy == [(40, 10, 60, 50)]
        return {"lines": [{"groups": []}]}

    def fake_recog(image_bgr, **_kwargs):
        assert tuple(image_bgr.shape[:2]) == (60, 36)
        return {
            "lines": [{
                "groups": [{
                    "bbox": {"left": 0, "top": 0, "right": 36, "bottom": 60},
                    "chars": [{
                        "codes": [code("．"), ord(".")],
                        "scores": [8, 9],
                        "bbox": {"left": 14, "top": 41, "right": 21, "bottom": 48},
                    }],
                }],
            }],
        }

    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_segimg", fake_segimg)
    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_recog", fake_recog)
    monkeypatch.setattr(micro_module, "_BATCH_DISABLED_FOR_SESSION", True)

    page = Page(image_path="", width=100, height=60)
    segment = RoutingSegment(
        kind="text_other",
        bbox=(40, 10, 60, 50),
        component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
        ppocr_punctuation_candidate=".",
    )
    line = RoutingLine(index=0, bbox=(40, 10, 60, 50), segments=(segment,), source="test")
    plan = PageRoutingPlan(
        page_uid=page.uid,
        prepass_run_id="test-missing-symbol-group",
        blocks=(BlockRoutingPlan(
            block_uid="block-1",
            plan=RoutingPlan(
                lines=(line,),
                text_slices=(TextSliceRoute(
                    line_index=0,
                    segment_index=0,
                    bbox=segment.bbox,
                    carved=False,
                    kind=segment.kind,
                ),),
                has_layout_routes=True,
            ),
        ),),
    )

    rows, stats = micro_module.run_micro_recblock(
        np.full((60, 100, 3), 255, dtype=np.uint8),
        [{
            "block_label": "text",
            "block_bbox": [40, 10, 60, 50],
            "block_content": ".",
            micro_module.ROUTE_ROW_LAYOUT_BLOCK_UID_KEY: "block-1",
        }],
        routing_plan=plan,
        page=page,
        include_chars=True,
    )

    assert rows[0].text == "."
    assert rows[0].lines[0].chars[0].bbox == (46, 41, 53, 48)
    assert stats.n_groups == 0
    assert stats.single_glyph_route_context_recogs == 1
    assert rows[0].segimg_group_audits == [{
        "route_text_slice_bbox": [40, 10, 60, 50],
        "segimg_group_bbox": None,
        "recog_group_bbox": [32, 0, 68, 60],
        "recog_group_bbox_before_padding": [40, 10, 60, 50],
        "recog_group_bbox_padded": True,
        "clipped": False,
        "dropped": False,
        "segimg_group_missing": True,
        "recog_strategy": "routing_segment_context",
    }]


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
        {"block_label": "text", "_layout_block_uid": text.uid},
        {"block_label": "table", "_layout_block_uid": table.uid},
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
                {"block_label": "text", "_layout_block_uid": text.uid},
                {"block_label": "text", "_layout_block_uid": "unexpected-text-block"},
            ],
            plan,
            page,
        )
