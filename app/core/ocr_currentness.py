"""Single currentness rule for active OCR observations."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.ocr_records import OcrActivePointer, OcrBatch, OcrRun
from app.models.project_session import ProjectSession, RecordNotFoundError


@dataclass(frozen=True, slots=True)
class CurrentOcrObservation:
    pointer: OcrActivePointer
    run: OcrRun
    batch: OcrBatch


@dataclass(frozen=True, slots=True)
class CurrentOcrFailure:
    pointer: OcrActivePointer
    run: OcrRun
    batch: OcrBatch
    message: str


def current_ocr_observation(
    session: ProjectSession,
    page_uid: str,
) -> CurrentOcrObservation | None:
    """Return the active observation only when its immutable inputs are current."""

    try:
        page = session.page_repository.get(page_uid)
        layout = session.layout_repository.get(page_uid)
        artifact = session.paddle_artifact_repository.get(layout.artifact_uid)
        pointer = session.ocr_observation_repository.get_active_pointer(page_uid)
        batch = session.ocr_observation_repository.get_batch(
            pointer.batch_uid,
            fingerprint=pointer.batch_fingerprint,
        )
        run = session.ocr_observation_repository.get_run(batch.run_uid)
    except RecordNotFoundError:
        return None
    if run.status != "completed" or batch.status != "complete":
        return None
    if batch.layout_fingerprint != layout_snapshot_fingerprint(layout):
        return None
    if artifact.page_uid != page.uid or artifact.image_hash != page.image_hash:
        return None
    return CurrentOcrObservation(pointer=pointer, run=run, batch=batch)


def current_ocr_failure(
    session: ProjectSession,
    page_uid: str,
) -> CurrentOcrFailure | None:
    """Return the active failed attempt only while its page and layout remain current."""
    try:
        page = session.page_repository.get(page_uid)
        layout = session.layout_repository.get(page_uid)
        artifact = session.paddle_artifact_repository.get(layout.artifact_uid)
        pointer = session.ocr_observation_repository.get_active_pointer(page_uid)
        batch = session.ocr_observation_repository.get_batch(
            pointer.batch_uid,
            fingerprint=pointer.batch_fingerprint,
        )
        run = session.ocr_observation_repository.get_run(batch.run_uid)
    except RecordNotFoundError:
        return None
    if run.status != "failed" or batch.status != "failed":
        return None
    if batch.layout_fingerprint != layout_snapshot_fingerprint(layout):
        return None
    metadata = dict(run.metadata)
    if (
        artifact.page_uid != page.uid
        or artifact.image_hash != page.image_hash
        or metadata.get("image_hash") != page.image_hash
    ):
        return None
    message = metadata.get("error", "OCR execution failed").strip()
    return CurrentOcrFailure(
        pointer=pointer,
        run=run,
        batch=batch,
        message=message or "OCR execution failed",
    )


def has_active_ocr_pointer(session: ProjectSession, page_uid: str) -> bool:
    """Return whether the page has ever adopted an OCR observation pointer."""
    try:
        session.ocr_observation_repository.get_active_pointer(page_uid)
    except RecordNotFoundError:
        return False
    return True


__all__ = [
    "CurrentOcrFailure",
    "CurrentOcrObservation",
    "current_ocr_failure",
    "current_ocr_observation",
    "has_active_ocr_pointer",
]
