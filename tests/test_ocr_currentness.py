from dataclasses import replace

from app.core.layout_scope import layout_snapshot_fingerprint
from app.core.ocr_currentness import current_ocr_observation
from app.models.layout_snapshot import LayoutSnapshot
from app.models.ocr_records import OcrActivePointer, OcrBatch, OcrRun
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession


def _session() -> ProjectSession:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = session.page_repository.put(
        PageRecord(
            project_uid=session.project_uid,
            uid="page-1",
            image_path="/tmp/import/page.png",
            source_path="source.pdf",
            cache_image_path="/tmp/import/page.png",
            thumbnail_path="/tmp/import/thumb.png",
            width=100,
            height=200,
            page_number=1,
            source_page_index=0,
            status="imported",
            error="",
            image_hash="image-hash",
            image_revision=1,
        ),
        expected_revision=0,
    )
    artifact = session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=session.project_uid,
            uid="artifact-1",
            page_uid=page.uid,
            source_engine="paddle",
            source_run_id="layout-run-1",
            image_hash=page.image_hash,
            payload_json="{}",
        )
    )
    layout = session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid=artifact.uid,
            source_engine="paddle",
            source_run_id="layout-run-1",
            blocks=(),
        ),
        expected_revision=0,
    )
    layout_fingerprint = layout_snapshot_fingerprint(layout)
    run = session.ocr_observation_repository.append_run(
        OcrRun(
            project_uid=session.project_uid,
            uid="run-1",
            engine="charocr",
            layout_fingerprint=layout_fingerprint,
            input_fingerprint="input-1",
            metadata=(("page_fingerprint", page.fingerprint),),
        )
    )
    batch = session.ocr_observation_repository.append_batch(
        OcrBatch(
            project_uid=session.project_uid,
            uid="batch-1",
            run_uid=run.uid,
            scope_uid=page.uid,
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=layout_fingerprint,
        )
    )
    session.ocr_observation_repository.switch_active_pointer(
        OcrActivePointer(
            project_uid=session.project_uid,
            uid="pointer-1",
            scope_uid=page.uid,
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )
    return session


def _with_page(session: ProjectSession, page: PageRecord) -> ProjectSession:
    ocr = session.ocr_observation_repository
    return ProjectSession.from_records(
        session.project_record,
        pages=(page,),
        paddle_artifacts=session.paddle_artifact_repository.all(),
        layouts=session.layout_repository.all(),
        ocr_runs=ocr.all_runs(),
        ocr_batches=ocr.all_batches(),
        ocr_active_pointers=ocr.all_active_pointers(),
    )


def test_current_ocr_survives_project_asset_path_rebinding() -> None:
    session = _session()
    page = session.page_repository.get("page-1")
    rebound = _with_page(
        session,
        replace(
            page,
            image_path="/saved/book.assets/images/page-1.png",
            cache_image_path="/saved/book.assets/images/page-1.png",
            thumbnail_path="/saved/book.assets/thumbnails/page-1.png",
        ),
    )

    assert rebound.page_repository.get("page-1").fingerprint != page.fingerprint
    assert current_ocr_observation(rebound, "page-1") is not None


def test_current_ocr_rejects_a_different_page_image() -> None:
    session = _session()
    page = session.page_repository.get("page-1")
    changed = _with_page(
        session,
        replace(page, image_hash="different-image", image_revision=2),
    )

    assert current_ocr_observation(changed, "page-1") is None
