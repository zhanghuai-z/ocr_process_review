from __future__ import annotations

from app.engines.hanwang.micro_recblock import (
    _inline_formula_crop_ocr_targets,
    _page_blocks_from_layout,
)
from app.models import BBox, Block, BlockType, Page
from app.models.enums import OcrPolicy
from app.models.layout_block_state import (
    set_layout_block_bbox,
    set_layout_block_ocr_policy,
    set_layout_block_source_label,
    set_layout_block_type,
)
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection


def test_hanwang_layout_rows_use_snapshot_when_runtime_projection_drifts():
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(10, 20, 100, 30),
        source_label="text",
        ocr_policy=OcrPolicy.TEXT_OCR,
        note="snapshot text",
    )
    page = Page(image_path="/tmp/page.png", width=240, height=160, blocks=[block])
    sync_page_layout_snapshot_from_projection(page, source_engine="test")

    set_layout_block_type(block, BlockType.FIGURE)
    set_layout_block_bbox(block, BBox(180, 120, 20, 20))
    set_layout_block_source_label(block, "figure")
    set_layout_block_ocr_policy(block, OcrPolicy.SKIP)

    rows = _page_blocks_from_layout(page)

    assert rows[0]["block_label"] == "text"
    assert rows[0]["source_label"] == "text"
    assert rows[0]["block_bbox"] == [10, 20, 110, 50]
    assert rows[0]["_layout_block_ocr_policy"] == OcrPolicy.TEXT_OCR.value


def test_hanwang_formula_crop_targets_use_snapshot_label_when_runtime_projection_drifts():
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(10, 20, 100, 30),
        source_label="inline_formula",
        ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
    )
    page = Page(image_path="/tmp/page.png", width=240, height=160, blocks=[block])
    sync_page_layout_snapshot_from_projection(page, source_engine="test")

    set_layout_block_type(block, BlockType.TEXT)
    set_layout_block_source_label(block, "text")
    set_layout_block_ocr_policy(block, OcrPolicy.TEXT_OCR)

    assert _inline_formula_crop_ocr_targets(page) == [block]
