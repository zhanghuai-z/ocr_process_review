from __future__ import annotations

import json

import pytest

from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.core.project_store import ProjectDataError, ProjectStore
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BBox, Block, BlockType, Char, Line, OcrPolicy, OcrProject, Page
from app.models.layout_block_view import LayoutSnapshotRequiredError, current_layout_snapshot
from app.models.layout_snapshot import LayoutSnapshot
from app.models.layout_snapshot_projection import (
    project_layout_snapshot_to_blocks,
    sync_page_layout_snapshot_from_projection,
)
from app.models.layout_snapshot_store import clear_layout_snapshot_for_page, layout_snapshot_for_page
from app.models.ocr_character_observation import line_ocr_chars
from app.models.ocr_observation import (
    block_ocr_line_observations_by_uid,
    replace_block_ocr_line_observations,
)
from app.services.layout_snapshot import (
    adopt_page_layout_snapshot,
    layout_snapshot_from_normalized_artifact,
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
    assert snapshot.blocks[0].uid.startswith("block_")
    assert blocks[0].uid == snapshot.blocks[0].uid
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
    stored_snapshot = layout_snapshot_for_page(page)
    assert stored_snapshot is not None
    assert stored_snapshot.page_uid == snapshot.page_uid
    assert stored_snapshot.blocks[0].uid == page.blocks[0].uid
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


def test_current_layout_snapshot_requires_adopted_snapshot():
    page = Page(
        image_path="/tmp/no-snapshot.png",
        width=100,
        height=80,
    )
    page.blocks.append(Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(1, 2, 30, 20)))
    clear_layout_snapshot_for_page(page)

    with pytest.raises(LayoutSnapshotRequiredError, match="has no layout snapshot"):
        current_layout_snapshot(page)


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
    replace_block_ocr_line_observations(block.uid, [line])
    sync_page_layout_snapshot_from_projection(project.pages[0], source_engine="test_seed")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
    with ProjectStore(db_path) as store:
        loaded = store.load_project(saved.id)

    page = loaded.pages[0]
    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert snapshot.source_engine == "test_seed"
    assert snapshot.blocks[0].source_label == "text"

    loaded_block = page.blocks[0]
    loaded_line = block_ocr_line_observations_by_uid(loaded_block.uid)[0]
    assert loaded_block.lines == []
    assert loaded_line.chars == []
    assert line_ocr_chars(loaded_line)[0].char == "甲"


def test_project_store_persists_layout_snapshot_independently_from_block_projection(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot.png", width=300, height=220)
    artifact = set_paddle_raw_layout_records(
        page,
        [
            {
                "block_label": "text",
                "block_bbox": [20, 30, 180, 70],
                "block_content": "正文内容",
                "score": 0.9,
            },
            {
                "block_label": "formula",
                "block_bbox": [40, 100, 160, 130],
                "block_content": "$x$",
            },
        ],
        run_id="job-layout",
    )
    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page),
        source_run_id="job-layout",
    )
    adopt_page_layout_snapshot(page, snapshot)
    projected_uid = page.blocks[0].uid
    project = OcrProject(name="snapshot persist", pages=[page])
    db_path = str(tmp_path / "snapshot-persist.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
    with ProjectStore(db_path) as store:
        loaded = store.load_project(saved.id)

    loaded_page = loaded.pages[0]
    loaded_snapshot = layout_snapshot_for_page(loaded_page)
    assert loaded_snapshot is not None
    assert loaded_snapshot.source_engine == "paddleocr-vl"
    assert loaded_snapshot.source_run_id == "job-layout"
    assert loaded_snapshot.artifact_uid == artifact.uid
    assert loaded_snapshot.blocks[0].source_label == "text"
    assert loaded_snapshot.blocks[0].uid == projected_uid
    assert loaded_page.blocks[0].uid == projected_uid


def test_project_store_keeps_snapshot_authoritative_when_saving_projection_drift(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot-save-projection.png", width=300, height=220)
    set_paddle_raw_layout_records(
        page,
        [
            {
                "block_label": "text",
                "block_bbox": [20, 30, 180, 70],
                "block_content": "正文内容",
                "score": 0.9,
            },
        ],
        run_id="job-layout",
    )
    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page),
        source_run_id="job-layout",
    )
    adopt_page_layout_snapshot(page, snapshot)
    projected_uid = page.blocks[0].uid
    page.blocks[0].block_type = BlockType.FIGURE
    page.blocks[0].bbox = BBox(200, 180, 40, 20)
    page.blocks[0].source_label = "figure"
    project = OcrProject(name="snapshot save projection", pages=[page])
    db_path = str(tmp_path / "snapshot-save-projection.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
    with ProjectStore(db_path) as store:
        loaded = store.load_project(saved.id)

    loaded_page = loaded.pages[0]
    loaded_snapshot = layout_snapshot_for_page(loaded_page)
    assert loaded_snapshot is not None
    assert loaded_snapshot.blocks[0].uid == projected_uid
    assert loaded_snapshot.blocks[0].block_type == BlockType.TEXT
    assert loaded_snapshot.blocks[0].bbox == BBox.from_xyxy(20, 30, 180, 70)
    assert loaded_page.blocks[0].uid == projected_uid
    assert loaded_page.blocks[0].block_type == BlockType.TEXT
    assert loaded_page.blocks[0].bbox == BBox.from_xyxy(20, 30, 180, 70)
    assert loaded_page.blocks[0].source_label == "text"


def test_project_store_load_projects_block_projection_from_layout_snapshot(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot-load-projection.png", width=300, height=220)
    set_paddle_raw_layout_records(
        page,
        [
            {
                "block_label": "text",
                "block_bbox": [20, 30, 180, 70],
                "block_content": "正文内容",
                "score": 0.9,
            },
        ],
        run_id="job-layout",
    )
    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page),
        source_run_id="job-layout",
    )
    adopt_page_layout_snapshot(page, snapshot)
    projected_uid = page.blocks[0].uid
    project = OcrProject(name="snapshot load projection", pages=[page])
    db_path = str(tmp_path / "snapshot-load-projection.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
        store.conn.execute(
            "UPDATE block SET block_type=?, x=?, y=?, w=?, h=?, source_label=? WHERE uid=?",
            ("figure", 200, 180, 40, 20, "figure", projected_uid),
        )
        store.conn.commit()

    with ProjectStore(db_path) as store:
        loaded = store.load_project(saved.id)

    loaded_page = loaded.pages[0]
    loaded_snapshot = layout_snapshot_for_page(loaded_page)
    assert loaded_snapshot is not None
    assert loaded_snapshot.blocks[0].uid == projected_uid
    assert loaded_page.blocks[0].uid == projected_uid
    assert loaded_page.blocks[0].block_type == BlockType.TEXT
    assert loaded_page.blocks[0].bbox == BBox.from_xyxy(20, 30, 180, 70)
    assert loaded_page.blocks[0].source_label == "text"


def test_project_store_rejects_project_without_layout_snapshot(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot-required.png", width=300, height=220)
    set_paddle_raw_layout_records(
        page,
        [
            {
                "block_label": "text",
                "block_bbox": [20, 30, 180, 70],
                "block_content": "正文内容",
                "score": 0.9,
            },
        ],
        run_id="job-layout",
    )
    snapshot = layout_snapshot_from_normalized_artifact(
        normalized_layout_artifact_from_page(page),
        source_run_id="job-layout",
    )
    adopt_page_layout_snapshot(page, snapshot)
    project = OcrProject(name="snapshot required", pages=[page])
    db_path = str(tmp_path / "snapshot-required.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
        store.conn.execute(
            "DELETE FROM layout_snapshot WHERE project_id=?",
            (saved.id,),
        )
        store.conn.commit()

    with ProjectStore(db_path) as store:
        with pytest.raises(ProjectDataError, match="has no persisted layout snapshot"):
            store.load_project(saved.id)


def test_project_store_rejects_malformed_persisted_layout_snapshot(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot-bad.png", width=100, height=80)
    snapshot = sync_page_layout_snapshot_from_projection(page, source_engine="layout_edit")
    project = OcrProject(name="snapshot malformed", pages=[page])
    db_path = str(tmp_path / "snapshot-malformed.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
        store.conn.execute(
            "UPDATE layout_snapshot SET blocks_json=? WHERE project_id=? AND page_uid=?",
            ("{}", saved.id, snapshot.page_uid),
        )
        store.conn.commit()

    with ProjectStore(db_path) as store:
        with pytest.raises(ProjectDataError):
            store.load_project(saved.id)


def test_project_store_rejects_empty_layout_snapshot_block_uid(tmp_path):
    page = Page(image_path="/tmp/layout-snapshot-empty-uid.png", width=120, height=80)
    set_paddle_raw_layout_records(
        page,
        [{"block_label": "text", "block_bbox": [1, 2, 30, 20], "block_content": "A"}],
    )
    snapshot = layout_snapshot_from_normalized_artifact(normalized_layout_artifact_from_page(page))
    adopt_page_layout_snapshot(page, snapshot)
    project = OcrProject(name="snapshot empty uid", pages=[page])
    db_path = str(tmp_path / "snapshot-empty-uid.ocrproj")

    with ProjectStore(db_path) as store:
        saved = store.save_project(project)
        payload = [
            {
                "uid": "",
                "block_type": "text",
                "bbox": {"x": 1, "y": 2, "w": 29, "h": 18},
                "order": 0,
                "source_label": "text",
                "origin": {
                    "created_by": "auto_layout",
                    "source_engine": "paddleocr-vl",
                    "source_run_id": "",
                    "source_label": "text",
                    "source_confidence": None,
                    "original_bbox": {"x": 1, "y": 2, "w": 29, "h": 18},
                    "original_kind": "text",
                    "raw_artifact_uid": "",
                    "raw_json_path": "",
                    "raw_index": 0,
                },
                "ocr_policy": "text_ocr",
                "note": "",
            }
        ]
        store.conn.execute(
            "UPDATE layout_snapshot SET blocks_json=? WHERE project_id=? AND page_uid=?",
            (json.dumps(payload), saved.id, page.uid),
        )
        store.conn.commit()

    with ProjectStore(db_path) as store:
        with pytest.raises(ProjectDataError, match="uid is empty"):
            store.load_project(saved.id)
