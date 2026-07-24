from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from app.application.contracts import (
    LayoutEditCommand,
    LayoutWorkspaceView,
    PageView,
)
from app.application.ocr_workspace import OcrPageView, OcrWorkspaceView
from app.core.block_attributes import block_attributes
from app.core.paddle_artifact_index import BINDING_GEOMETRY_HIT, PaddleArtifactIndex
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord
from app.services.layout_overlay_service import LayoutOverlayService
from app.ui.recognize.layout_panel import LayoutPanel
from app.ui.recognize.ocr_panel import OcrPanel
from app.ui.widgets.image_viewer import BBoxItem, ImageViewer, OcrAtomBox


def _qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page(path: Path) -> PageRecord:
    from PIL import Image

    Image.new("RGB", (160, 100), "white").save(path)
    return PageRecord(
        project_uid="project-1",
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
                                        "block_label": "figure",
                                        "block_bbox": [75, 10, 110, 40],
                                    },
                                ],
                            }
                        ],
                        "layout_det_res": {
                            "boxes": [{
                                "label": "inline_formula",
                                "coordinate": [40, 10, 65, 35],
                                "score": 0.9,
                            }]
                        },
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


def test_panel_emits_draw_intent_from_workspace_view(tmp_path: Path) -> None:
    _qt_app()
    page = _page(tmp_path / "page.png")
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    page_view = PageView.from_records(page, snapshot)
    workspace = LayoutWorkspaceView("project-1", "Book", (page_view,))

    panel = LayoutPanel(workspace)
    requested = []
    panel.layout_edit_requested.connect(requested.append)

    panel._on_block_created(BBox.from_xyxy(130, 10, 155, 35))

    assert len(requested) == 1
    command = requested[0]
    assert isinstance(command, LayoutEditCommand)
    assert command.op == "draw"
    assert command.page_uid == page.uid
    assert command.expected_revision == snapshot.revision
    assert command.bbox == BBox.from_xyxy(130, 10, 160, 35)
    assert command.new_block_uid.startswith("block-")
    assert page_view.layout_revision == 1
    assert page_view.block_uids == ("block-1",)


def test_panel_skips_identical_workspace_and_current_page_redraw(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _qt_app()
    page = _page(tmp_path / "page.png")
    page_view = PageView.from_records(page, _snapshot(_artifact()))
    workspace = LayoutWorkspaceView("project-1", "Book", (page_view,))
    panel = LayoutPanel(workspace)
    redraws: list[int] = []
    monkeypatch.setattr(panel, "_update_viewer", redraws.append)

    panel.set_workspace(workspace)
    panel.set_current_page_uid(page.uid)

    assert redraws == []


def test_panel_emits_geometry_intent_without_mutating_workspace_view(tmp_path: Path) -> None:
    _qt_app()
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    page = _page(tmp_path / "page.png")
    page_view = PageView.from_records(page, snapshot)
    panel = LayoutPanel(LayoutWorkspaceView("project-1", "Book", (page_view,)))
    requested = []
    panel.layout_edit_requested.connect(requested.append)

    panel._on_block_geometry_change_requested(
        "block-1",
        BBox.from_xyxy(10, 20, 140, 80),
    )

    assert len(requested) == 1
    command = requested[0]
    assert isinstance(command, LayoutEditCommand)
    assert command.op == "resize"
    assert command.block_uid == "block-1"
    assert command.expected_revision == snapshot.revision
    assert page_view.layout_revision == 1
    assert page_view.blocks[0].bbox == snapshot.blocks[0].bbox


def test_workspace_refresh_preserves_zoom_page_and_multi_selection(tmp_path: Path) -> None:
    _qt_app()
    artifact = _artifact()
    base = _snapshot(artifact)
    second = replace(
        base.blocks[0],
        uid="block-2",
        bbox=BBox.from_xyxy(0, 55, 120, 95),
        order=1,
    )
    snapshot = replace(base, blocks=(base.blocks[0], second))
    page = _page(tmp_path / "page.png")
    page_view = PageView.from_records(page, snapshot)
    panel = LayoutPanel(LayoutWorkspaceView("project-1", "Book", (page_view,)))
    panel._viewer.scale(1.7, 1.7)
    panel._viewer.select_block_uids(("block-1", "block-2"))
    transform = panel._viewer.transform()

    changed_snapshot = replace(
        snapshot,
        revision=2,
        blocks=(replace(snapshot.blocks[0], source_label="paragraph_title"), second),
    )
    changed_page = PageView.from_records(page, changed_snapshot)
    panel.set_workspace(LayoutWorkspaceView("project-1", "Book", (changed_page,)))

    assert panel._current_page_idx == 0
    assert panel._viewer.transform() == transform
    assert tuple(panel._viewer.selected_block_uids()) == ("block-1", "block-2")
    assert panel._selected_block_uids == ("block-1", "block-2")


def test_multi_selection_emits_one_atomic_type_change(tmp_path: Path) -> None:
    _qt_app()
    artifact = _artifact()
    base = _snapshot(artifact)
    second = replace(
        base.blocks[0],
        uid="block-2",
        bbox=BBox.from_xyxy(0, 55, 120, 95),
        order=1,
    )
    snapshot = replace(base, blocks=(base.blocks[0], second))
    page_view = PageView.from_records(_page(tmp_path / "page.png"), snapshot)
    panel = LayoutPanel(LayoutWorkspaceView("project-1", "Book", (page_view,)))
    requested: list[LayoutEditCommand] = []
    panel.layout_edit_requested.connect(requested.append)
    panel._viewer.select_block_uids(("block-1", "block-2"))

    panel._on_selected_type_button_clicked(BlockType.TITLE)

    assert len(requested) == 1
    assert requested[0].op == "change_types"
    assert requested[0].block_uids == ("block-1", "block-2")
    assert requested[0].block_type is BlockType.TITLE
    assert requested[0].expected_revision == snapshot.revision


def test_panel_emits_all_layout_edit_intents_as_application_commands(tmp_path: Path) -> None:
    _qt_app()
    artifact = _artifact()
    base_snapshot = _snapshot(artifact)
    second_block = replace(
        base_snapshot.blocks[0],
        uid="block-2",
        bbox=BBox.from_xyxy(0, 55, 120, 95),
        order=1,
    )
    snapshot = replace(base_snapshot, blocks=(base_snapshot.blocks[0], second_block))
    page_view = PageView.from_records(_page(tmp_path / "page.png"), snapshot)
    panel = LayoutPanel(LayoutWorkspaceView("project-1", "Book", (page_view,)))
    requested: list[LayoutEditCommand] = []
    panel.layout_edit_requested.connect(requested.append)

    panel._on_block_geometry_change_requested("block-1", BBox.from_xyxy(5, 5, 125, 55))
    panel._on_block_geometry_change_requested("block-1", BBox.from_xyxy(5, 5, 130, 60))
    panel._on_block_clicked_uid("block-1")
    panel._on_selected_type_button_clicked(BlockType.TITLE)
    panel._on_block_created(BBox.from_xyxy(0, 0, 120, 50))
    panel._on_block_deleted_uid("block-1")
    panel._on_block_created(BBox.from_xyxy(130, 10, 155, 35))

    assert [command.op for command in requested] == [
        "move",
        "resize",
        "change_type",
        "merge",
        "delete",
        "draw",
    ]
    assert all(isinstance(command, LayoutEditCommand) for command in requested)
    assert all(command.page_uid == "page-1" for command in requested)
    assert page_view.block_uids == ("block-1", "block-2")


def test_image_viewer_consumes_view_blocks_and_atom_boxes_immutably(tmp_path: Path) -> None:
    _qt_app()
    artifact = _artifact()
    snapshot = _snapshot(artifact)
    page_view = PageView.from_records(_page(tmp_path / "page.png"), snapshot)
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
    with pytest.raises(TypeError):
        viewer.show_layout_blocks(snapshot.blocks)
    viewer.show_layout_blocks(page_view.blocks, atom_boxes=(atom,))
    viewer.show_atom_boxes((atom,))

    assert viewer._block_items[0][1] is page_view.blocks[0]
    assert viewer._atom_items[0][1] is atom
    assert viewer._block_items[0][0]._block_view is page_view.blocks[0]
    assert viewer._atom_items[0][0]._atom_box is atom
    assert viewer._atom_items[0][0].is_editable() is False


def test_bbox_item_coalesces_drag_geometry_until_release(tmp_path: Path) -> None:
    _qt_app()
    page_view = PageView.from_records(_page(tmp_path / "page.png"), _snapshot(_artifact()))
    item = BBoxItem(
        QRectF(0, 0, page_view.blocks[0].bbox.w, page_view.blocks[0].bbox.h),
        color=QColor("#ff0000"),
    )
    item.set_block(page_view.blocks[0])
    emitted: list[tuple[str, BBox]] = []
    item.signals.geometry_changed.connect(lambda uid, bbox: emitted.append((uid, bbox)))

    first = BBox.from_xyxy(10, 10, 130, 60)
    final = BBox.from_xyxy(20, 20, 150, 80)
    item._queue_geometry_changed(first)
    item._queue_geometry_changed(final)

    assert emitted == []
    item._flush_geometry_changed()
    assert emitted == [("block-1", final)]


def test_ocr_panel_consumes_ocr_workspace_without_legacy_entrypoints(tmp_path: Path) -> None:
    _qt_app()
    page = _page(tmp_path / "page.png")
    page_view = PageView.from_records(page, _snapshot(_artifact()))
    panel = OcrPanel()

    workspace = OcrWorkspaceView(
        project_uid=page_view.project_uid,
        project_name="Book",
        pages=(OcrPageView(page=page_view, batch_uid=None, regions=()),),
    )
    panel.set_workspace(workspace)

    assert panel._pages == workspace.pages
    assert panel._tree.topLevelItemCount() == 1
    assert panel._tree.topLevelItem(0).childCount() == 1
    with pytest.raises(TypeError):
        panel.set_workspace(page)
    assert not hasattr(panel, "set_pages")
    assert not hasattr(panel, "set_snapshot")


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
        LayoutPanel().set_workspace(object())
