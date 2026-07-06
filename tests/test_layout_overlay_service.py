from __future__ import annotations

from app.core.inline_formula_edit_state import mark_inline_formula_origin_handled
from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BBox, Block, BlockOrigin, BlockType, OcrPolicy, Page
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.services.layout_overlay_service import LayoutOverlayService


def test_layout_overlay_service_promotes_inline_formula_overlays():
    page = Page(image_path="", width=200, height=80)
    set_paddle_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [0, 0, 180, 40],
            "block_content": "甲 $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
            ],
        }
    ])
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 180, 40),
            origin=BlockOrigin(source_label="text", raw_index=0),
        )
    ]

    created = LayoutOverlayService().ensure_inline_formula_blocks(page)

    assert created == 1
    inline = page.blocks[-1]
    assert inline.block_type == BlockType.EQUATION
    assert inline.source_label == "inline_formula"
    assert inline.ocr_policy == OcrPolicy.PRESERVE_AS_FORMULA
    assert inline.origin is not None
    assert inline.origin.raw_index == 0
    assert inline.origin.original_bbox == BBox.from_xyxy(40, 0, 70, 30)
    assert inline.origin.raw_artifact_uid == page.raw_layout_artifact.uid
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.blocks[-1].uid == inline.uid
    assert snapshot.blocks[-1].source_label == "inline_formula"
    assert snapshot.blocks[-1].ocr_policy == OcrPolicy.PRESERVE_AS_FORMULA


def test_layout_overlay_service_skips_handled_inline_formula_origin():
    page = Page(image_path="", width=200, height=80)
    set_paddle_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [0, 0, 180, 40],
            "block_content": "甲 $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
            ],
        }
    ])
    handled = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(40, 0, 70, 30),
        source_label="inline_formula",
        origin=BlockOrigin(source_label="inline_formula", original_bbox=BBox.from_xyxy(40, 0, 70, 30), raw_index=0),
    )
    mark_inline_formula_origin_handled(page, handled, op="delete_inline_formula")

    created = LayoutOverlayService().ensure_inline_formula_blocks(page)

    assert created == 0
    assert page.blocks == []


def test_layout_overlay_service_does_not_treat_legacy_type_as_inline_formula():
    page = Page(image_path="", width=200, height=80)
    set_paddle_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [0, 0, 180, 40],
            "block_content": "甲 $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"type": "inline_formula", "block_bbox": [40, 0, 70, 30]},
            ],
        }
    ])
    handled = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(40, 0, 70, 30),
        source_label="inline_formula",
        origin=BlockOrigin(source_label="inline_formula", original_bbox=BBox.from_xyxy(40, 0, 70, 30), raw_index=0),
    )
    mark_inline_formula_origin_handled(page, handled, op="delete_inline_formula")

    created = LayoutOverlayService().ensure_inline_formula_blocks(page)

    assert created == 0
    assert page.blocks == []


def test_layout_overlay_service_readonly_overlays_skip_inline_formula_and_deduplicate():
    page = Page(image_path="", width=200, height=100)
    set_paddle_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [0, 0, 180, 60],
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "table_region", "block_bbox": [10, 10, 80, 40]},
                {"block_label": "table_region", "block_bbox": [10, 10, 80, 40]},
                {"block_label": "inline_formula", "block_bbox": [100, 10, 120, 35]},
            ],
        }
    ])

    overlays = LayoutOverlayService().readonly_layout_overlays(page)

    assert overlays == [("table_region", BBox.from_xyxy(10, 10, 80, 40))]
