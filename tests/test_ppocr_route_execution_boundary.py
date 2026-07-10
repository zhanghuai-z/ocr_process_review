from __future__ import annotations

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.engines.hanwang.micro_recblock import (
    BlockResult,
    HanwangMicroRecBlockEngine,
    LineResult,
    RunStats,
    _apply_page_routing_plan_to_native_rows,
)
from app.models import BBox, Block, BlockType, OcrPolicy, Page
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection


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

    assert captured["kwargs"]["page_ocr_lines"] is None
    assert captured["kwargs"]["routing_plan"] is routing_plan
    # The engine hands an immutable plan to the native-runner boundary.  The
    # runner alone projects it to its temporary row dictionaries, so an
    # injected runner never receives stale PP-OCR line attachments.
    assert all("_layout_line_routes" not in row for row in captured["blocks"])


def test_native_row_projection_carries_routes_only_for_text_block_uid():
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

    _apply_page_routing_plan_to_native_rows(native_rows, plan, page)

    assert native_rows[0]["_layout_line_routes"][0]["segments"][1]["kind"] == "skip"
    assert "_layout_line_routes" not in native_rows[1]
