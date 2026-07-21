from __future__ import annotations

from app.application import LayoutEditCommand, WorkbenchApplication
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.services.import_service import ImportService


def _workbench() -> WorkbenchApplication:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = PageRecord(
        project_uid=session.project_uid,
        uid="page-1",
        image_path="page.png",
        source_path="page.png",
        cache_image_path="",
        thumbnail_path="",
        width=100,
        height=80,
        page_number=1,
        source_page_index=0,
        status="imported",
        error="",
        image_hash="hash",
        image_revision=1,
    )
    session.page_repository.put(page, expected_revision=0)
    artifact = PaddleArtifact(
        project_uid=session.project_uid,
        uid="artifact-1",
        page_uid=page.uid,
        source_engine="test",
        source_run_id="run-1",
        image_hash=page.image_hash,
        payload_json="{}",
    )
    session.paddle_artifact_repository.append(artifact)
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid=artifact.uid,
            source_engine="test",
            source_run_id="run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=BBox(1, 2, 30, 20),
                    order=0,
                    source_label="text",
                    origin=BlockOrigin(created_by="auto_layout"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    authorship=BlockSource.AUTO_LAYOUT,
                    uid="block-1",
                ),
            ),
        ),
        expected_revision=0,
    )
    return WorkbenchApplication(session=session, import_service=ImportService())


def test_workbench_is_the_layout_write_boundary() -> None:
    workbench = _workbench()

    result = workbench.apply_layout_edit(
        LayoutEditCommand.resize(
            "page-1",
            1,
            "block-1",
            BBox(5, 6, 40, 22),
        )
    )

    assert result.revision_before == 1
    assert result.revision_after == 2
    assert result.page_view is not None
    assert result.page_view.blocks[0].bbox == BBox(5, 6, 40, 22)
    assert workbench.is_dirty is True


def test_workbench_queries_return_detached_immutable_values() -> None:
    workbench = _workbench()

    view = workbench.layout_workspace()

    assert view.project_uid == "project-1"
    assert view.pages[0].blocks[0].block_uid == "block-1"
    assert not hasattr(view, "layout_repository")
    assert not hasattr(view.pages[0], "session")
