from __future__ import annotations

from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.core.project_store import ProjectStore
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BBox, Block, BlockType, Char, Line, OcrPolicy, OcrProject, Page
from app.models.layout_snapshot import LayoutSnapshot
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.models.ocr_character_observation import line_ocr_chars
from app.models.ocr_observation import block_ocr_lines
from app.services.layout_snapshot import (
    adopt_page_layout_snapshot,
    layout_snapshot_from_normalized_artifact,
    project_layout_snapshot_to_blocks,
    sync_page_layout_snapshot_from_projection,
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


def test_adopt_layout_snapshot_stores_truth_and_projects_blocks():
    page = Page(image_path="", width=300, height=220)
    set_paddle_raw_layout_records(page, [
        {"block_label": "text", "block_bbox": [20, 30, 180, 70], "block_content": "A"},
    ])
    snapshot = layout_snapshot_from_normalized_artifact(normalized_layout_artifact_from_page(page))

    blocks = adopt_page_layout_snapshot(page, snapshot)

    assert page.blocks == blocks
    assert layout_snapshot_for_page(page) is snapshot
    assert page.blocks[0].source_label == "text"


def test_sync_layout_snapshot_from_projection_tracks_current_blocks():
    block = Block(block_type=BlockType.TABLE, bbox=BBox.from_xyxy(10, 20, 80, 60), source_label="table")
    page = Page(image_path="", width=300, height=220, blocks=[block])

    snapshot = sync_page_layout_snapshot_from_projection(page, source_engine="layout_edit")

    assert layout_snapshot_for_page(page) is snapshot
    assert snapshot.source_engine == "layout_edit"
    assert snapshot.blocks[0].bbox == block.bbox
    assert snapshot.blocks[0].source_label == "table"
    assert snapshot.blocks[0].origin.original_bbox == block.bbox


def test_project_store_load_rebuilds_layout_and_ocr_runtime_stores(tmp_path):
    line = Line(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 2, 30, 12),
        chars=[Char(char="甲", confidence=0.9, bbox=BBox(1, 2, 10, 12))],
    )
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 30),
        lines=[line],
        source_label="text",
    )
    project = OcrProject(
        name="snapshot load",
        pages=[Page(image_path="/tmp/snapshot-load.png", width=100, height=80, blocks=[block])],
    )
    db_path = str(tmp_path / "snapshot-load.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
    with ProjectStore(db_path) as store:
        loaded = store.load_project(saved.id)

    page = loaded.pages[0]
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.source_engine == "project_store"
    assert snapshot.blocks[0].source_label == "text"

    loaded_block = page.blocks[0]
    loaded_line = block_ocr_lines(loaded_block)[0]
    assert block_ocr_lines(loaded_block) is loaded_block.lines
    assert line_ocr_chars(loaded_line) is loaded_line.chars
    assert line_ocr_chars(loaded_line)[0].char == "甲"
