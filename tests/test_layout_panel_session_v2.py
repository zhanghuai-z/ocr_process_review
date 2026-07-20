from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from app.core.block_attributes import block_attributes
from app.core.paddle_artifact_index import BINDING_GEOMETRY_HIT, PaddleArtifactIndex
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.services.inline_formula_layout_service import InlineFormulaLayoutService
from app.services.layout_edit_service import LayoutEditCommand
from app.services.layout_overlay_service import LayoutOverlayService
from app.ui.recognize.layout_panel import LayoutPanel, PageLayoutInput
from app.ui.widgets.image_viewer import ImageViewer, OcrAtomBox


def _qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page(session: ProjectSession, path: Path) -> PageRecord:
    from PIL import Image

    Image.new("RGB", (160, 100), "white").save(path)
    return session.page_repository.put(
        PageRecord(
            project_uid=session.project_uid,
            uid="page-1",
            image_path=str(path),
            source_path=str(path),
            cache_image_path=str(path),
            thumbnail_path=str(path),
            width=160,
            height=100,
            page_number=1,
            source_page_index=0,
            status="imported",
            error="",
            image_hash="image-hash",
            image_revision=1,
        ),
        expected_revision=0,
    )


def _artifact() -> PaddleArtifact:
    payload = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_bbox": [0, 0, 120, 50],
                                "block_content": "before $x$ after",
                                "_route_subblocks": [
                                    {
                                        "block_label": "inline_formula",
                                        "block_bbox": [40, 10, 65, 35],
                                    },
                                    {
                                        "block_label": "figure",
                                        "block_bbox": [75, 10, 110, 40],
                                    },
                                ],
                            }
                        ]
                    }
                }
            ]
        }
    }
    return PaddleArtifact(
        project_uid="project-1",
        uid="artifact-1",
        page_uid="page-1",
        source_engine="paddle-vl",
        source_run_id="run-1",
        image_hash="image-hash",
        payload_json=json.dumps(payload),
    )


def _snapshot(artifact: PaddleArtifact) -> LayoutSnapshot:
    block = LayoutBlockSnapshot(
        uid="block-1",
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(0, 0, 120, 50),
        order=0,
        source_label="text",
        origin=BlockOrigin(
            source_engine=artifact.source_engine,
            raw_artifact_uid=artifact.uid,
            original_bbox=BBox.from_xyxy(0, 0, 120, 50),
            original_kind=BlockType.TEXT,
        ),
        ocr_policy=OcrPolicy.TEXT_OCR,
        authorship=BlockSource.AUTO_LAYOUT,
    )
    return LayoutSnapshot(
        page_uid=artifact.page_uid,
        revision=1,
        artifact_uid=artifact.uid,
        source_engine=artifact.source_engine,
        source_run_id=artifact.source_run_id,
        blocks=(block,),
    )


def test_panel_session_edits_snapshot_with_cas_and_stable_uid(tmp_path: Path) -> None:
    _qt_app()
    session = ProjectSession(ProjectRecord("project-1"))
    page = _page(session, tmp_path / "page.png")
    artifact = _artifact()
    session.paddle_artifact_repository.append(artifact)
    snapshot = _snapshot(artifact)
    session.layout_repository.put(snapshot, expected_revision=0)

    panel = LayoutPanel(session)
    applied = []
    contracts = []
    panel.layout_edit_applied.connect(applied.append)
    panel.block_contract_changed.connect(lambda page_uid, reason: contracts.append((page_uid, reason)))

    panel._on_block_created(BBox.from_xyxy(130, 10, 155, 35))

    current = session.layout_repository.get(page.uid)
    assert current.revision == 2
    assert len(current.blocks) == 2
    assert current.blocks[0].uid == "block-1"
    assert current.blocks[1].bbox.x == 130
    assert current.blocks[1].bbox.y == 10
    assert current.blocks[1].bbox.w > 0 and current.blocks[1].bbox.h > 0
    assert session.page_repository.get(page.uid) == page
    assert applied and applied[-1].snapshot is current
    assert contracts == [("page-1", "layout_block_created")]
    assert all(not hasattr(block, "chars") for block in current.blocks)


def test_panel_input_mode_emits_edit_intent_without_mutating_input() -> None:
    _qt_app()
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    page = PageRecord(
        project_uid="project-1",
        uid="page-1",
        image_path="page.png",
        source_path="source.pdf",
        cache_image_path="",
        thumbnail_path="",
        width=160,
        height=100,
        page_number=1,
        source_page_index=0,
        status="layout_done",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )
    panel = LayoutPanel()
    panel.set_page_layout_inputs([PageLayoutInput(page, snapshot, artifact)])
    requested = []
    applied = []
    panel.layout_edit_requested.connect(requested.append)
    panel.layout_edit_applied.connect(applied.append)

    command = LayoutEditCommand.create_block(
        page.uid,
        snapshot.revision,
        BBox.from_xyxy(130, 10, 155, 35),
        BlockType.TEXT,
        "text",
        new_block_uid="input-only-block",
    )
    result = panel._apply_layout_edit(panel._pages[0], command)

    assert requested == [command]
    assert applied and applied[0].snapshot == result.snapshot
    assert result is not None
    assert result.snapshot.revision == 2
    assert snapshot.revision == 1
    assert len(snapshot.blocks) == 1


def test_image_viewer_keeps_snapshot_blocks_and_atom_boxes_immutable() -> None:
    _qt_app()
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    atom = OcrAtomBox(
        uid="atom-1",
        bbox=BBox.from_xyxy(42, 14, 56, 30),
        text="x",
        confidence=0.91,
        line_uid="line-1",
        block_uid="block-1",
    )
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(160, 100, QImage.Format.Format_RGB32))
    viewer.show_layout_snapshot(snapshot, atom_boxes=(atom,))
    viewer.show_atom_boxes((atom,))

    assert viewer._block_items[0][1] is snapshot.blocks[0]
    assert viewer._atom_items[0][1] is atom
    assert viewer._block_items[0][0]._layout_block is snapshot.blocks[0]
    assert viewer._atom_items[0][0]._atom_box is atom
    assert viewer._atom_items[0][0].is_editable() is False


def test_inline_formula_service_returns_snapshot_edits_from_artifact_only() -> None:
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    service = InlineFormulaLayoutService()

    regions = service.iter_regions(artifact, page_width=160, page_height=100)
    results = service.apply_snapshot_edits(
        snapshot,
        artifact,
        page_width=160,
        page_height=100,
    )

    assert [(region.label, region.bbox) for region in regions] == [
        ("inline_formula", BBox.from_xyxy(40, 10, 65, 35))
    ]
    assert len(results) == 1
    assert results[0].snapshot.revision == 2
    created = results[0].snapshot.blocks[-1]
    assert created.block_type is BlockType.EQUATION
    assert created.origin.raw_artifact_uid == artifact.uid
    assert created.ocr_policy is OcrPolicy.PRESERVE_AS_FORMULA
    assert snapshot.revision == 1
    assert len(snapshot.blocks) == 1


def test_snapshot_attributes_overlay_and_artifact_index_use_new_boundaries() -> None:
    artifact = _artifact()
    snapshot = _snapshot(artifact)

    attributes = block_attributes(snapshot.blocks[0])
    assert attributes.current_label == "text"
    assert attributes.semantic_block_type is BlockType.TEXT

    overlays = LayoutOverlayService().readonly_layout_overlays(
        artifact,
        page_width=160,
        page_height=100,
    )
    assert overlays == [("figure", BBox.from_xyxy(75, 10, 110, 40))]

    index = PaddleArtifactIndex.from_artifact(
        artifact,
        page_width=160,
        page_height=100,
    )
    assert index.artifact is artifact
    assert index.parents[0].label == "text"
    assert index.formula_geometry[0].bbox == (40, 10, 65, 35)
    assert index.bind_manual_bbox(
        BBox.from_xyxy(40, 10, 65, 35),
        BlockType.EQUATION,
    ).status == BINDING_GEOMETRY_HIT


def test_new_contracts_reject_legacy_layout_values() -> None:
    _qt_app()
    with pytest.raises(TypeError):
        LayoutPanel().set_page_layout_inputs([object()])
    with pytest.raises(TypeError):
        InlineFormulaLayoutService().iter_regions(object(), page_width=160, page_height=100)
