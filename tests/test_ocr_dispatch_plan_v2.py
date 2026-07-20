from dataclasses import FrozenInstanceError, fields

import pytest

from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.project_session import PageRecord
from app.services.ocr_dispatch_plan import (
    DispatchBlock,
    build_text_ocr_dispatch_plan,
    count_text_ocr_blocks,
    iter_text_ocr_blocks,
)


def _page(uid: str = "page-1") -> PageRecord:
    return PageRecord(
        project_uid="project-1",
        uid=uid,
        image_path=f"/images/{uid}.png",
        source_path="/sources/input.pdf",
        cache_image_path=f"/cache/{uid}.png",
        thumbnail_path=f"/thumbs/{uid}.png",
        width=320,
        height=240,
        page_number=1,
        source_page_index=0,
        status="imported",
        error="",
        image_hash=f"hash-{uid}",
        image_revision=1,
    )


def _block(
    uid: str,
    order: int,
    block_type: BlockType,
    policy: OcrPolicy,
    source_label: str,
) -> LayoutBlockSnapshot:
    bbox = BBox(10 + order * 20, 10, 30, 20)
    return LayoutBlockSnapshot(
        uid=uid,
        block_type=block_type,
        bbox=bbox,
        order=order,
        source_label=source_label,
        origin=BlockOrigin(
            created_by=BlockSource.AUTO_LAYOUT.value,
            vendor_label=source_label,
            original_bbox=bbox,
            original_kind=block_type,
        ),
        ocr_policy=policy,
        authorship=BlockSource.AUTO_LAYOUT,
    )


def _layout(page_uid: str, blocks: tuple[LayoutBlockSnapshot, ...]) -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid=page_uid,
        revision=4,
        artifact_uid="layout-artifact-1",
        source_engine="test-layout",
        source_run_id="layout-run-1",
        blocks=blocks,
    )


def test_dispatch_plan_contains_immutable_facts_and_excludes_structural_blocks():
    page = _page()
    layout = _layout(
        page.uid,
        (
            _block("text-1", 0, BlockType.TEXT, OcrPolicy.TEXT_OCR, "text"),
            _block(
                "formula-1",
                1,
                BlockType.EQUATION,
                OcrPolicy.PRESERVE_AS_FORMULA,
                "formula",
            ),
            _block(
                "table-1",
                2,
                BlockType.TABLE,
                OcrPolicy.PRESERVE_AS_TABLE,
                "table",
            ),
            _block("figure-1", 3, BlockType.FIGURE, OcrPolicy.SKIP, "figure"),
            _block("title-1", 4, BlockType.TITLE, OcrPolicy.TEXT_OCR, "title"),
        ),
    )

    plan = build_text_ocr_dispatch_plan(page, layout)

    assert plan.page is page
    assert plan.layout is layout
    assert plan.page_uid == page.uid
    assert plan.layout_revision == layout.revision
    assert [record.block_uid for record in plan.text_blocks] == ["text-1", "title-1"]
    assert [record.block_uid for record in plan.blocked_blocks] == [
        "formula-1",
        "table-1",
        "figure-1",
    ]
    assert plan.text_blocks[0].order == 0
    assert plan.text_blocks[0].bbox == layout.blocks[0].bbox
    assert plan.text_blocks[0].block_type is BlockType.TEXT
    assert plan.text_blocks[0].ocr_policy is OcrPolicy.TEXT_OCR
    assert plan.text_blocks[0].source_label == "text"
    assert plan.text_blocks[0].reason == "policy:text_ocr"
    assert plan.blocked_blocks[0].reason == "policy:preserve_as_formula"
    assert plan.total_text_blocks == 2
    assert plan.text_block_uids == ("text-1", "title-1")
    assert plan.blocked_block_uids == ("formula-1", "table-1", "figure-1")

    dispatch_fields = {field.name for field in fields(DispatchBlock)}
    assert dispatch_fields == {
        "block_uid",
        "order",
        "bbox",
        "block_type",
        "ocr_policy",
        "source_label",
        "reason",
    }
    with pytest.raises(FrozenInstanceError):
        plan.text_blocks[0].order = 99


def test_dispatch_plan_rejects_layout_from_another_page():
    page = _page("page-1")
    layout = _layout(
        "page-2",
        (_block("text-1", 0, BlockType.TEXT, OcrPolicy.TEXT_OCR, "text"),),
    )

    with pytest.raises(ValueError, match="same page UID"):
        build_text_ocr_dispatch_plan(page, layout)


def test_dispatch_iteration_and_count_return_records_in_stable_order():
    page = _page()
    layout = _layout(
        page.uid,
        (
            _block("text-3", 0, BlockType.TEXT, OcrPolicy.TEXT_OCR, "text"),
            _block("formula-1", 1, BlockType.EQUATION, OcrPolicy.SKIP, "formula"),
            _block("text-1", 2, BlockType.TEXT, OcrPolicy.TEXT_OCR, "text"),
            _block("text-2", 3, BlockType.REFERENCE, OcrPolicy.TEXT_OCR, "reference"),
        ),
    )

    records = list(iter_text_ocr_blocks(((page, layout),)))

    assert all(isinstance(record, DispatchBlock) for record in records)
    assert [record.block_uid for record in records] == ["text-3", "text-1", "text-2"]
    assert [record.order for record in records] == [0, 2, 3]
    assert count_text_ocr_blocks(((page, layout),)) == 3
