from __future__ import annotations

from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BlockType, OcrPolicy, Page
from app.models.layout_snapshot import LayoutSnapshot
from app.services.layout_snapshot import (
    layout_snapshot_from_normalized_artifact,
    project_layout_snapshot_to_blocks,
)


def test_layout_snapshot_projects_normalized_artifact_to_legacy_blocks():
    page = Page(image_path="", width=300, height=220)
    artifact = set_paddle_raw_layout_records(
        page,
        [
            {
                "block_label": "text",
                "block_bbox": [20, 30, 180, 70],
                "block_content": "正文内容",
                "score": 0.9876,
            },
            {
                "block_label": "page_number",
                "block_bbox": [240, 190, 280, 210],
                "block_content": "12",
            },
        ],
        run_id="job-1",
    )

    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page),
        source_run_id="job-1",
    )
    blocks = project_layout_snapshot_to_blocks(snapshot)

    assert isinstance(snapshot, LayoutSnapshot)
    assert snapshot.page_uid == page.uid
    assert snapshot.artifact_uid == artifact.uid
    assert [block.source_label for block in blocks] == ["text", "page_number"]
    assert [block.block_type for block in blocks] == [BlockType.TEXT, BlockType.TEXT]
    assert blocks[0].note == "正文内容 | score=0.988"
    assert blocks[1].note == "12 | source_label=page_number"
    assert blocks[0].ocr_policy == OcrPolicy.TEXT_OCR
    assert blocks[0].origin is not None
    assert blocks[0].origin.source_engine == "paddleocr-vl"
    assert blocks[0].origin.source_run_id == "job-1"
    assert blocks[0].origin.raw_artifact_uid == artifact.uid
    assert blocks[0].origin.raw_index == 0
    assert blocks[0].origin.source_confidence == 0.9876


def test_layout_snapshot_deduplicates_regions_by_label_and_bbox():
    page = Page(image_path="", width=300, height=220)
    set_paddle_raw_layout_records(page, [
        {"block_label": "text", "block_bbox": [20, 30, 180, 70], "block_content": "A"},
        {"block_label": "text", "block_bbox": [20, 30, 180, 70], "block_content": "A"},
    ])

    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page)
    )

    assert len(snapshot.blocks) == 1
    assert snapshot.blocks[0].order == 0
