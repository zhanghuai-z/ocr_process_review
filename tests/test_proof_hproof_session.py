from __future__ import annotations

from app.core.proof_state import ProofUpdateRequest
from app.models import Page
from app.services.proof_hproof_session import (
    HProofLineEditSession,
    HProofRuntimeSession,
)


def test_hproof_session_keeps_valid_page_selection_and_clears_invalid_one():
    session = HProofRuntimeSession()
    page1 = Page(image_path="/tmp/hproof-session-p1.png", width=10, height=10, page_number=1)
    page2 = Page(image_path="/tmp/hproof-session-p2.png", width=10, height=10, page_number=2)

    session.selected_page_number = 2
    session.set_pages([page1, page2], page_has_lines=lambda page: page.page_number == 2)

    assert session.selected_page_number == 2
    assert session.filtered_pages() == [page2]

    session.set_pages([page1], page_has_lines=lambda _page: True)

    assert session.selected_page_number is None
    assert session.filtered_pages() == [page1]


def test_hproof_session_debug_flags_and_pending_external_queue_are_explicit():
    session = HProofRuntimeSession()
    request = ProofUpdateRequest(page_id=None, line_id=None, page_uid="p1", line_uid="l1", status="MODIFIED")

    session.set_debug_flags(formula=True, table=False)
    session.add_pending_external(request)

    assert session.debug_enabled is True
    assert session.debug_label == "公式"
    assert session.pending_external_requests == [request]
    assert session.pop_pending_external() == [request]
    assert session.pending_external_requests == []

    session.set_debug_flags(formula=False, table=True)
    assert session.debug_label == "表格"

    session.reset()
    assert session.debug_enabled is False
    assert session.debug_label == ""
    assert session.pending_external_requests == []


def test_hproof_line_edit_session_detects_dirty_conflict_and_rebinds():
    session = HProofLineEditSession(
        editable=True,
        loaded_display_text="AAAA",
        loaded_line_signature="sig-a",
    )

    assert session.is_dirty("CCCC")
    assert session.dirty_snapshot("CCCC") == ("CCCC", "AAAA")

    was_dirty = session.rebind_to_model(
        editor_text="CCCC",
        model_display_text="DDDD",
        model_line_signature="sig-d",
    )

    assert was_dirty is True
    assert session.loaded_display_text == "DDDD"
    assert session.loaded_line_signature == "sig-d"
    assert session.external_conflict is True

    assert session.clear_conflict_if_editor_matches_model(
        editor_text="CCCC",
        model_display_text="DDDD",
        model_line_signature="sig-d2",
    ) is False
    assert session.clear_conflict_if_editor_matches_model(
        editor_text="DDDD",
        model_display_text="DDDD",
        model_line_signature="sig-d2",
    ) is True
    assert session.external_conflict is False
    assert session.loaded_line_signature == "sig-d2"
