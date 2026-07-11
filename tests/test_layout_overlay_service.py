from __future__ import annotations

from app.core.inline_formula_edit_state import mark_inline_formula_origin_handled
from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BBox, Block, BlockOrigin, BlockType, OcrPolicy, OcrProject, Page
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.services.inline_formula_layout_service import InlineFormulaLayoutService
from app.services.layout_overlay_service import LayoutOverlayService
from app.services.layout_snapshot import adopt_page_layout_snapshot, layout_snapshot_from_normalized_artifact


def _adopt_raw_layout(page: Page) -> None:
    adopt_page_layout_snapshot(
        page,
        layout_snapshot_from_normalized_artifact(normalized_layout_artifact_from_page(page)),
    )


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
    _adopt_raw_layout(page)

    created = InlineFormulaLayoutService().adopt_page(page)

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
    _adopt_raw_layout(page)

    created = InlineFormulaLayoutService().adopt_page(page)

    assert created == 0
    assert [block.source_label for block in page.blocks] == ["text"]


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
    _adopt_raw_layout(page)

    created = InlineFormulaLayoutService().adopt_page(page)

    assert created == 0
    assert [block.source_label for block in page.blocks] == ["text"]


def test_workflow_controller_adopts_inline_formulas_before_layout_finished():
    from app.controllers.workflow_controller import WorkflowController

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
    _adopt_raw_layout(page)
    controller = WorkflowController()
    controller._project = OcrProject(name="layout adoption", pages=[page])
    controller._auto_start_ocr_after_layout = False
    emitted_labels: list[list[str]] = []
    controller.layout_finished.connect(
        lambda pages: emitted_labels.append([block.source_label for block in pages[0].blocks])
    )

    controller.on_layout_done([page])

    assert emitted_labels == [["text", "inline_formula"]]


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
