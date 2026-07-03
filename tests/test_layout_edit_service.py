from __future__ import annotations

from app.core.paddle_artifact_index import BINDING_EMPTY_REVIEW
from app.models import BBox, Block, BlockSource, BlockType, Line, Page
from app.models.ocr_observation import block_ocr_lines
from app.services.layout_edit_service import LayoutEditCommand, LayoutEditService


def test_layout_edit_service_create_block_records_event_and_binding():
    page = Page(image_path="", width=200, height=100)
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.create_block(
        page,
        BBox.from_xyxy(10, 20, 60, 40),
        BlockType.EQUATION,
        "inline_formula",
    ))

    assert result.op == "create_block"
    assert result.block is page.blocks[0]
    assert result.binding_status == BINDING_EMPTY_REVIEW
    block = result.block
    assert block is not None
    assert block.block_type == BlockType.EQUATION
    assert block.source == BlockSource.MANUAL_DRAW
    assert block.source_label == "inline_formula"
    assert block.paddle_binding is not None
    assert block.paddle_binding.manual_bbox == [10, 20, 60, 40]
    assert page.layout_edit_events[-1].op == "create_block"
    assert page.layout_edit_events[-1].target_uid == block.uid


def test_layout_edit_service_delete_block_records_event_and_removes_block():
    block = Block(block_type=BlockType.TABLE, bbox=BBox(10, 10, 30, 20))
    page = Page(image_path="", width=200, height=100, blocks=[block])
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.delete_block(page, block))

    assert result.op == "delete_block"
    assert page.blocks == []
    assert page.layout_edit_events[-1].op == "delete_block"
    assert page.layout_edit_events[-1].before["block"]["uid"] == block.uid


def test_layout_edit_service_change_block_kind_updates_policy_and_event():
    block = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 30, 20), source_label="text")
    page = Page(image_path="", width=200, height=100, blocks=[block])
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.change_kind(
        page,
        block,
        block_type=BlockType.TABLE,
        source_label="table",
    ))

    assert result.op == "change_kind"
    assert block.block_type == BlockType.TABLE
    assert block.source == BlockSource.USER_EDITED
    assert block.source_label == "table"
    assert block.paddle_binding is not None
    assert block.paddle_binding.manual_bbox == [10, 10, 40, 30]
    assert page.layout_edit_events[-1].op == "change_kind"
    assert page.layout_edit_events[-1].before["block"]["block_type"] == "text"
    assert page.layout_edit_events[-1].after["block"]["block_type"] == "table"


def test_layout_edit_service_preserves_explicit_structural_subtype_label():
    block = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 30, 20), source_label="text")
    page = Page(image_path="", width=200, height=100, blocks=[block])
    service = LayoutEditService()

    service.apply(LayoutEditCommand.change_kind(
        page,
        block,
        block_type=BlockType.FIGURE,
        source_label="chart",
    ))

    assert block.block_type == BlockType.FIGURE
    assert block.source_label == "chart"
    assert block.paddle_binding is not None
    assert block.paddle_binding.source_label == "figure"


def test_layout_edit_service_merge_blocks_invalidates_primary_and_clears_ocr_lines():
    line = Line(text="x", confidence=0.9, bbox=BBox(20, 20, 10, 10))
    primary = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(20, 20, 40, 30),
        lines=[line],
        order=0,
    )
    secondary = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(60, 20, 80, 30),
        lines=[Line(text="(1)", confidence=0.9, bbox=BBox(60, 20, 20, 10))],
        order=1,
    )
    page = Page(image_path="", width=200, height=100, blocks=[primary, secondary])
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.merge_blocks(
        page,
        [secondary, primary],
        BBox.from_xyxy(10, 10, 90, 40),
        block_type=BlockType.EQUATION,
        source_label="display_formula",
    ))

    assert result.op == "merge_blocks"
    assert result.block is primary
    assert page.blocks == [primary]
    assert primary.bbox == BBox.from_xyxy(10, 10, 90, 40)
    assert block_ocr_lines(primary) == []
    assert primary.source == BlockSource.USER_EDITED
    assert primary.source_label == "display_formula"
    assert primary.note == "manual_geometry_empty_formula_review"
    assert primary.ocr_invalidated_reason == "manual_draw_merge"
    assert primary.paddle_binding is not None
    assert primary.paddle_binding.manual_bbox == [10, 10, 90, 40]
    assert page.layout_edit_events[-1].op == "merge_blocks"
    assert [item["uid"] for item in page.layout_edit_events[-1].before["blocks"]] == [
        primary.uid,
        secondary.uid,
    ]


def test_layout_edit_service_geometry_update_preserves_existing_manual_binding_row():
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(10, 10, 40, 30),
        source_label="inline_formula",
    )
    page = Page(image_path="", width=200, height=100, blocks=[block])
    service = LayoutEditService()
    service.apply(LayoutEditCommand.create_block(page, block.bbox, block.block_type, block.source_label))
    bound = page.blocks[-1]
    bound.bbox = BBox.from_xyxy(12, 10, 42, 30)

    result = service.apply(LayoutEditCommand.update_geometry(
        page,
        bound,
        before=LayoutEditService.block_state(bound),
    ))

    assert result.op == "resize_block"
    assert bound.paddle_binding is not None
    assert bound.paddle_binding.manual_bbox == [12, 10, 42, 30]
    assert bound.source == BlockSource.USER_EDITED
    assert bound.ocr_invalidated_reason == "block_geometry_changed"
