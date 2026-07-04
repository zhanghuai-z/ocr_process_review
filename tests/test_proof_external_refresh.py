from __future__ import annotations

from app.core.proof_state import ProofUpdateRequest
from app.models import BBox, Line
from app.services.proof_external_refresh import (
    ProofExternalLineRef,
    ProofExternalRefreshQueue,
)


def test_proof_external_line_ref_prefers_uid_over_rowid():
    line = Line(text="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10), id=7)
    ref = ProofExternalLineRef(line_uid=line.uid, line_id=999)

    assert ref.matches_line(line) is True
    assert ref.matches_line(Line(text="乙", confidence=0.9, bbox=BBox(0, 0, 10, 10), id=7)) is False


def test_proof_external_refresh_queue_rejects_invalid_request():
    queue = ProofExternalRefreshQueue()

    assert queue.queue_request(ProofUpdateRequest(page_id=None, line_id=None, status="MODIFIED")) is False
    assert queue.has_pending is False
    assert queue.consume().has_work is False


def test_proof_external_refresh_queue_batches_line_refs_and_page_keys():
    queue = ProofExternalRefreshQueue()

    assert queue.queue_line_key("line-a", page_keys=[("page", 1)]) is True
    assert queue.queue_line_key(42, page_keys=[("page", 2)]) is True

    batch = queue.consume()

    assert batch.has_work is True
    assert [ref.line_uid for ref in batch.line_refs] == ["line-a", ""]
    assert [ref.line_id for ref in batch.line_refs] == [None, 42]
    assert set(batch.page_keys) == {("page", 1), ("page", 2)}
    assert queue.has_pending is False
