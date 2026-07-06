from __future__ import annotations

from app.core.paddle_artifact_index import BINDING_EMPTY_REVIEW
from app.models import (
    BBox,
    Block,
    BlockOrigin,
    BlockSource,
    BlockType,
    LayoutBlockSnapshot,
    LayoutSnapshot,
    Line,
    OcrPolicy,
    Page,
)
from app.models.layout_snapshot_store import layout_snapshot_for_page, set_layout_snapshot_for_page
from app.models.ocr_observation import block_ocr_line_observations
from app.services.layout_edit_service import LayoutEditCommand, LayoutEditService


def _seed_layout_snapshot(page: Page) -> None:
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="test-artifact",
            source_engine="test_seed",
            source_run_id="test-seed",
            blocks=tuple(
                LayoutBlockSnapshot(
                    block_type=block.block_type,
                    bbox=block.bbox,
                    order=block.order,
                    source_label=block.source_label,
                    origin=block.origin or BlockOrigin(
                        source_engine="test",
                        source_label=block.source_label or block.block_type.value,
                        original_bbox=block.bbox,
                        original_kind=block.block_type,
                    ),
                    ocr_policy=block.ocr_policy,
                    uid=block.uid,
                )
                for block in page.blocks
            ),
        ),
    )


def test_layout_edit_service_create_block_records_event_and_binding():
    page = Page(image_path="", width=200, height=100)
    _seed_layout_snapshot(page)
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
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.source_engine == "layout_edit"
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid
    assert snapshot.blocks[0].bbox == block.bbox
    assert snapshot.blocks[0].source_label == "inline_formula"


def test_layout_edit_service_create_block_appends_to_snapshot_without_projection_drift():
    existing = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(90, 80, 140, 95),
        order=7,
        source_label="text",
    )
    existing_snapshot_bbox = BBox.from_xyxy(10, 10, 60, 30)
    page = Page(image_path="", width=200, height=100, blocks=[existing])
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=existing_snapshot_bbox,
                    order=0,
                    source_label="text",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    uid=existing.uid,
                ),
            ),
        ),
    )
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.create_block(
        page,
        BBox.from_xyxy(70, 20, 110, 40),
        BlockType.EQUATION,
        "inline_formula",
    ))

    assert result.block is not None
    assert page.blocks == [existing, result.block]
    assert existing.bbox == existing_snapshot_bbox
    assert result.block.order == 1
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [existing.uid, result.block.uid]
    assert snapshot.blocks[0].bbox == existing_snapshot_bbox
    assert snapshot.blocks[1].bbox == BBox.from_xyxy(70, 20, 110, 40)
    assert snapshot.blocks[1].order == 1
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid


def test_layout_edit_service_delete_block_records_event_and_removes_block():
    block = Block(block_type=BlockType.TABLE, bbox=BBox(10, 10, 30, 20))
    page = Page(image_path="", width=200, height=100, blocks=[block])
    _seed_layout_snapshot(page)
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.delete_block(page, block.uid))

    assert result.op == "delete_block"
    assert page.blocks == []
    assert page.layout_edit_events[-1].op == "delete_block"
    assert page.layout_edit_events[-1].before["block"]["uid"] == block.uid
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.blocks == ()


def test_layout_edit_service_delete_block_keeps_remaining_snapshot_projection():
    deleted = Block(block_type=BlockType.TABLE, bbox=BBox.from_xyxy(10, 10, 40, 30), order=4)
    kept = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(90, 90, 130, 120), order=9)
    page = Page(image_path="", width=200, height=100, blocks=[deleted, kept])
    kept_snapshot_bbox = BBox.from_xyxy(50, 20, 100, 40)
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TABLE,
                    bbox=BBox.from_xyxy(10, 10, 40, 30),
                    order=4,
                    source_label="table",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="table"),
                    ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
                    uid=deleted.uid,
                ),
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=kept_snapshot_bbox,
                    order=1,
                    source_label="text",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    uid=kept.uid,
                ),
            ),
        ),
    )
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.delete_block(page, deleted.uid))

    assert result.op == "delete_block"
    assert page.blocks == [kept]
    assert kept.bbox == kept_snapshot_bbox
    assert kept.order == 1
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [kept.uid]
    assert snapshot.blocks[0].bbox == kept_snapshot_bbox
    assert snapshot.blocks[0].order == 1
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid


def test_layout_edit_service_restore_blocks_records_event_and_reorders_blocks():
    old_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 10, 10), order=9)
    restored_a = LayoutBlockSnapshot(
        block_type=BlockType.TABLE,
        bbox=BBox(10, 10, 30, 20),
        order=4,
        source_label="table",
        origin=BlockOrigin(source_engine="test", source_label="table"),
        ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
    )
    restored_b = LayoutBlockSnapshot(
        block_type=BlockType.FIGURE,
        bbox=BBox(40, 10, 30, 20),
        order=2,
        source_label="figure",
        origin=BlockOrigin(source_engine="test", source_label="figure"),
        ocr_policy=OcrPolicy.SKIP,
    )
    page = Page(image_path="", width=200, height=100, blocks=[old_block])
    _seed_layout_snapshot(page)
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.restore_blocks(
        page,
        [restored_a, restored_b],
        before={"blocks": [LayoutEditService.block_state(old_block)]},
    ))

    assert result.op == "restore_blocks"
    assert [block.uid for block in page.blocks] == [restored_a.uid, restored_b.uid]
    assert page.blocks[0].lines == []
    assert page.blocks[0].block_type == BlockType.TABLE
    assert page.blocks[1].block_type == BlockType.FIGURE
    assert [block.order for block in page.blocks] == [0, 1]
    assert page.layout_edit_events[-1].op == "restore_blocks"
    assert page.layout_edit_events[-1].before["blocks"][0]["uid"] == old_block.uid
    assert [item["uid"] for item in page.layout_edit_events[-1].after["blocks"]] == [
        restored_a.uid,
        restored_b.uid,
    ]
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [restored_a.uid, restored_b.uid]
    assert [block.order for block in snapshot.blocks] == [0, 1]
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid


def test_layout_edit_service_change_block_kind_updates_policy_and_event():
    block = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 30, 20), source_label="text")
    page = Page(image_path="", width=200, height=100, blocks=[block])
    _seed_layout_snapshot(page)
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.change_kind(
        page,
        block.uid,
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
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.blocks[0].block_type == BlockType.TABLE
    assert snapshot.blocks[0].source_label == "table"


def test_layout_edit_service_change_block_kind_uses_snapshot_geometry_when_projection_drifts():
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(90, 90, 130, 120),
        order=9,
        source_label="text",
    )
    page = Page(image_path="", width=200, height=100, blocks=[block])
    snapshot_bbox = BBox.from_xyxy(10, 20, 60, 40)
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=snapshot_bbox,
                    order=2,
                    source_label="text",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    uid=block.uid,
                ),
            ),
        ),
    )
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.change_kind(
        page,
        block.uid,
        block_type=BlockType.TITLE,
        source_label="heading_1",
    ))

    assert result.op == "change_kind"
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.source_engine == "layout_edit"
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid
    assert snapshot.blocks[0].block_type == BlockType.TITLE
    assert snapshot.blocks[0].source_label == "heading_1"
    assert snapshot.blocks[0].bbox == snapshot_bbox
    assert snapshot.blocks[0].order == 2
    assert snapshot.blocks[0].ocr_policy == OcrPolicy.TEXT_OCR
    assert block.block_type == BlockType.TITLE
    assert block.source == BlockSource.USER_EDITED
    assert block.source_label == "heading_1"
    assert block.bbox == snapshot_bbox
    assert block.order == 2


def test_layout_edit_service_preserves_explicit_structural_subtype_label():
    block = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 30, 20), source_label="text")
    page = Page(image_path="", width=200, height=100, blocks=[block])
    _seed_layout_snapshot(page)
    service = LayoutEditService()

    service.apply(LayoutEditCommand.change_kind(
        page,
        block.uid,
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
    _seed_layout_snapshot(page)
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.merge_blocks(
        page,
        [secondary.uid, primary.uid],
        BBox.from_xyxy(10, 10, 90, 40),
        block_type=BlockType.EQUATION,
        source_label="display_formula",
    ))

    assert result.op == "merge_blocks"
    assert result.block is primary
    assert page.blocks == [primary]
    assert primary.bbox == BBox.from_xyxy(10, 10, 90, 40)
    assert block_ocr_line_observations(primary) == []
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


def test_layout_edit_service_merge_blocks_uses_snapshot_order_and_geometry_when_projection_drifts():
    primary = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(120, 80, 180, 95),
        lines=[Line(text="old", confidence=0.9, bbox=BBox.from_xyxy(120, 80, 180, 95))],
        order=9,
        source_label="inline_formula",
    )
    secondary = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(0, 0, 180, 90),
        order=0,
        source_label="inline_formula",
    )
    kept = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(130, 80, 190, 98),
        order=5,
        source_label="text",
    )
    kept_snapshot_bbox = BBox.from_xyxy(100, 50, 160, 70)
    page = Page(image_path="", width=200, height=100, blocks=[secondary, primary, kept])
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.EQUATION,
                    bbox=BBox.from_xyxy(20, 20, 40, 30),
                    order=0,
                    source_label="inline_formula",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="inline_formula"),
                    ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                    uid=primary.uid,
                ),
                LayoutBlockSnapshot(
                    block_type=BlockType.EQUATION,
                    bbox=BBox.from_xyxy(60, 20, 80, 30),
                    order=1,
                    source_label="inline_formula",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="inline_formula"),
                    ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                    uid=secondary.uid,
                ),
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=kept_snapshot_bbox,
                    order=2,
                    source_label="text",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    uid=kept.uid,
                ),
            ),
        ),
    )
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.merge_blocks(
        page,
        [secondary.uid, primary.uid],
        BBox.from_xyxy(10, 10, 90, 40),
        block_type=BlockType.EQUATION,
        source_label="display_formula",
    ))

    assert result.op == "merge_blocks"
    assert result.block is primary
    assert page.blocks == [primary, kept]
    assert primary.bbox == BBox.from_xyxy(10, 10, 90, 40)
    assert block_ocr_line_observations(primary) == []
    assert kept.bbox == kept_snapshot_bbox
    assert [block.order for block in page.blocks] == [0, 1]
    assert [item["uid"] for item in page.layout_edit_events[-1].before["blocks"]] == [
        primary.uid,
        secondary.uid,
    ]
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [primary.uid, kept.uid]
    assert snapshot.blocks[0].bbox == BBox.from_xyxy(10, 10, 90, 40)
    assert snapshot.blocks[1].bbox == kept_snapshot_bbox
    assert [block.order for block in snapshot.blocks] == [0, 1]
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid


def test_layout_edit_service_geometry_update_preserves_existing_manual_binding_row():
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(10, 10, 40, 30),
        source_label="inline_formula",
    )
    page = Page(image_path="", width=200, height=100, blocks=[block])
    _seed_layout_snapshot(page)
    service = LayoutEditService()
    service.apply(LayoutEditCommand.create_block(page, block.bbox, block.block_type, block.source_label))
    bound = page.blocks[-1]
    bound.bbox = BBox.from_xyxy(12, 10, 42, 30)

    result = service.apply(LayoutEditCommand.update_geometry(
        page,
        bound.uid,
        bbox=bound.bbox,
        before=LayoutEditService.block_state(bound),
    ))

    assert result.op == "resize_block"
    assert bound.paddle_binding is not None
    assert bound.paddle_binding.manual_bbox == [12, 10, 42, 30]
    assert bound.source == BlockSource.USER_EDITED
    assert bound.ocr_invalidated_reason == "block_geometry_changed"


def test_layout_edit_service_geometry_update_uses_command_bbox_when_projection_drifts():
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(90, 80, 150, 95),
        order=7,
        source_label="inline_formula",
    )
    snapshot_bbox = BBox.from_xyxy(10, 10, 40, 30)
    next_bbox = BBox.from_xyxy(12, 14, 42, 34)
    page = Page(image_path="", width=200, height=100, blocks=[block])
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.EQUATION,
                    bbox=snapshot_bbox,
                    order=1,
                    source_label="inline_formula",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="inline_formula"),
                    ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                    uid=block.uid,
                ),
            ),
        ),
    )
    service = LayoutEditService()

    result = service.apply(LayoutEditCommand.update_geometry(page, block.uid, bbox=next_bbox))

    assert result.op == "resize_block"
    assert block.bbox == next_bbox
    assert block.order == 1
    assert block.source == BlockSource.USER_EDITED
    assert block.paddle_binding is not None
    assert block.paddle_binding.manual_bbox == [12, 14, 42, 34]
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.source_run_id == page.layout_edit_events[-1].uid
    assert snapshot.blocks[0].bbox == next_bbox
    assert snapshot.blocks[0].order == 1
