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


def current_ocr_observation(
    session: ProjectSession,
    page_uid: str,
) -> CurrentOcrObservation | None:
    """Return the active observation only when its immutable inputs are current."""

    try:
        page = session.page_repository.get(page_uid)
        layout = session.layout_repository.get(page_uid)
        pointer = session.ocr_observation_repository.get_active_pointer(page_uid)
        batch = session.ocr_observation_repository.get_batch(
            pointer.batch_uid,
            fingerprint=pointer.batch_fingerprint,
        )
        run = session.ocr_observation_repository.get_run(batch.run_uid)
    except RecordNotFoundError:
        return None
    metadata = dict(run.metadata)
    if batch.layout_fingerprint != layout_snapshot_fingerprint(layout):
        return None
    if metadata.get("page_fingerprint") != page.fingerprint:
        return None
    return CurrentOcrObservation(pointer=pointer, run=run, batch=batch)


__all__ = ["CurrentOcrObservation", "current_ocr_observation"]
