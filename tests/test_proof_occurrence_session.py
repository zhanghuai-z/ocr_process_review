from __future__ import annotations

from app.core.proof_occurrence import proof_page_identity_key
from app.models import Page
from app.services.proof_occurrence_session import VProofOccurrenceSession


def _page(number: int) -> Page:
    return Page(
        image_path=f"/tmp/vproof-session-{number}.png",
        width=10,
        height=10,
        page_number=number,
    )


def test_vproof_occurrence_session_consumes_external_refresh_for_current_page():
    page1 = _page(1)
    page2 = _page(2)
    session = VProofOccurrenceSession()
    session.set_pages([page1, page2])

    session.queue_external_refresh(
        line_key="l1",
        page_keys=[proof_page_identity_key(page1)],
    )
    plan = session.consume_external_refresh_plan()

    assert plan.has_work is True
    assert plan.affected_page_keys == (proof_page_identity_key(page1),)
    assert plan.reload_current_page is True
    assert session.has_pending_external_refresh is False


def test_vproof_occurrence_session_consumes_external_refresh_for_offscreen_page():
    page1 = _page(1)
    page2 = _page(2)
    session = VProofOccurrenceSession()
    session.set_pages([page1, page2])

    session.queue_external_refresh(
        line_key="l2",
        page_keys=[proof_page_identity_key(page2)],
    )
    plan = session.consume_external_refresh_plan()

    assert plan.has_work is True
    assert plan.affected_page_keys == (proof_page_identity_key(page2),)
    assert plan.reload_current_page is False


def test_vproof_occurrence_session_clears_pending_refresh_without_pages():
    session = VProofOccurrenceSession()
    session.queue_external_refresh(line_key="l1", page_keys=[("page", "missing")])

    plan = session.consume_external_refresh_plan()

    assert plan.has_work is False
    assert plan.reload_current_page is False
    assert session.has_pending_external_refresh is False


def test_vproof_occurrence_session_legacy_line_only_refresh_targets_current_page():
    page1 = _page(1)
    session = VProofOccurrenceSession()
    session.set_pages([page1])
    session.queue_external_refresh(line_key="legacy-line-only", page_keys=[])

    plan = session.consume_external_refresh_plan()

    assert plan.has_work is True
    assert plan.affected_page_keys == (proof_page_identity_key(page1),)
    assert plan.reload_current_page is True
