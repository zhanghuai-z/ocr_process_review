from __future__ import annotations

from dataclasses import replace

from app.application import LayoutEditCommand, WorkbenchApplication
from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_records import OcrActivePointer, OcrBatch, OcrRun
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.services.import_service import ImportService


def _workbench_session() -> tuple[WorkbenchApplication, ProjectSession]:
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
    return WorkbenchApplication(session=session, import_service=ImportService()), session


def _workbench() -> WorkbenchApplication:
    return _workbench_session()[0]


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


def test_layout_edit_makes_previous_ocr_batch_non_current() -> None:
    workbench, session = _workbench_session()
    layout = session.layout_repository.get("page-1")
    layout_fingerprint = layout_snapshot_fingerprint(layout)
    run = OcrRun(
        project_uid="project-1",
        uid="run-ocr-1",
        engine="test",
        layout_fingerprint=layout_fingerprint,
        input_fingerprint="input-1",
        metadata=(("page_fingerprint", session.page_repository.get("page-1").fingerprint),),
    )
    batch = OcrBatch(
        project_uid="project-1",
        uid="batch-1",
        run_uid=run.uid,
        scope_uid="page-1",
        input_fingerprint="input-1",
        layout_fingerprint=layout_fingerprint,
    )
    session.ocr_observation_repository.append_observation_batch(
        run=run,
        regions=(),
        lines=(),
        atoms=(),
        candidates=(),
        batch=batch,
    )
    session.ocr_observation_repository.switch_active_pointer(
        OcrActivePointer(
            project_uid="project-1",
            uid="active-page-1",
            scope_uid="page-1",
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )

    assert workbench.has_ocr("page-1") is True
    assert workbench.ocr_workspace().pages[0].batch_uid == "batch-1"

    workbench.apply_layout_edit(
        LayoutEditCommand.resize("page-1", 1, "block-1", BBox(2, 3, 32, 21))
    )

    assert workbench.has_ocr("page-1") is False
    assert workbench.ocr_workspace().pages[0].batch_uid is None
    assert workbench.pending_ocr_page_uids() == ("page-1",)


def test_page_image_change_makes_previous_ocr_batch_non_current() -> None:
    workbench, session = _workbench_session()
    page = session.page_repository.get("page-1")
    layout_fingerprint = layout_snapshot_fingerprint(
        session.layout_repository.get("page-1")
    )
    run = OcrRun(
        project_uid="project-1",
        uid="run-ocr-image-1",
        engine="test",
        layout_fingerprint=layout_fingerprint,
        input_fingerprint="input-image-1",
        metadata=(("page_fingerprint", page.fingerprint),),
    )
    batch = OcrBatch(
        project_uid="project-1",
        uid="batch-image-1",
        run_uid=run.uid,
        scope_uid="page-1",
        input_fingerprint=run.input_fingerprint,
        layout_fingerprint=layout_fingerprint,
    )
    session.ocr_observation_repository.append_observation_batch(
        run=run,
        regions=(),
        lines=(),
        atoms=(),
        candidates=(),
        batch=batch,
    )
    session.ocr_observation_repository.switch_active_pointer(
        OcrActivePointer(
            project_uid="project-1",
            uid="active-page-image-1",
            scope_uid="page-1",
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )

    assert workbench.has_ocr("page-1") is True

    session.page_repository.put(
        replace(page, image_hash="new-hash", image_revision=2),
        expected_revision=1,
    )

    assert workbench.has_ocr("page-1") is False
    assert workbench.ocr_workspace().pages[0].batch_uid is None
    assert workbench.pending_ocr_page_uids() == ("page-1",)
