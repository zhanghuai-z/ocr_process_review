from __future__ import annotations

from app.services.proof_edit_service import ProofEditStatus
from app.services.proof_rebuild_gate import (
    ProofEditorRebuildState,
    ProofRebuildDecision,
    allow_proof_rebuild,
    block_proof_rebuild,
    proof_rebuild_gate_for_editor_state,
    proof_rebuild_gate_for_reference_context,
    proof_rebuild_gate_for_save_status,
)


def test_proof_rebuild_gate_allows_non_blocking_save_statuses():
    for status in (
        ProofEditStatus.NOOP,
        ProofEditStatus.SAVED,
        ProofEditStatus.READONLY,
    ):
        result = proof_rebuild_gate_for_save_status(status)
        assert result.allow_rebuild is True
        assert result.decision == ProofRebuildDecision.ALLOW
        assert result.save_status == status


def test_proof_rebuild_gate_blocks_conflict_and_invalid_target():
    for status in (
        ProofEditStatus.CONFLICT,
        ProofEditStatus.INVALID_TARGET,
    ):
        result = proof_rebuild_gate_for_save_status(status, conflict_message="blocked")
        assert result.allow_rebuild is False
        assert result.decision == ProofRebuildDecision.BLOCK
        assert result.save_status == status
        assert result.message == "blocked"


def test_explicit_proof_rebuild_gate_helpers():
    assert allow_proof_rebuild().allow_rebuild is True

    blocked = block_proof_rebuild("stop")
    assert blocked.allow_rebuild is False
    assert blocked.message == "stop"


def test_editor_rebuild_state_blocks_external_conflict_before_save():
    state = ProofEditorRebuildState(editable=True, external_conflict=True, dirty=True)

    result = proof_rebuild_gate_for_editor_state(
        state,
        save_status=ProofEditStatus.SAVED,
        conflict_message="conflict",
    )

    assert state.needs_save is False
    assert result.allow_rebuild is False
    assert result.save_status == ProofEditStatus.CONFLICT
    assert result.message == "conflict"


def test_editor_rebuild_state_allows_clean_and_readonly_rows_without_save():
    clean = proof_rebuild_gate_for_editor_state(
        ProofEditorRebuildState(editable=True, external_conflict=False, dirty=False),
        save_status=ProofEditStatus.CONFLICT,
    )
    readonly = proof_rebuild_gate_for_editor_state(
        ProofEditorRebuildState(editable=False, external_conflict=False, dirty=True),
        save_status=ProofEditStatus.CONFLICT,
    )

    assert clean.allow_rebuild is True
    assert clean.save_status == ProofEditStatus.NOOP
    assert readonly.allow_rebuild is True
    assert readonly.save_status == ProofEditStatus.READONLY


def test_editor_rebuild_state_delegates_dirty_save_status():
    state = ProofEditorRebuildState(editable=True, external_conflict=False, dirty=True)

    blocked = proof_rebuild_gate_for_editor_state(
        state,
        save_status=ProofEditStatus.INVALID_TARGET,
        conflict_message="invalid",
    )
    saved = proof_rebuild_gate_for_editor_state(
        state,
        save_status=ProofEditStatus.SAVED,
    )

    assert state.needs_save is True
    assert blocked.allow_rebuild is False
    assert blocked.message == "invalid"
    assert saved.allow_rebuild is True
    assert saved.save_status == ProofEditStatus.SAVED


def test_reference_context_rebuild_treats_dirty_text_as_readonly_projection():
    result = proof_rebuild_gate_for_reference_context(dirty=True)

    assert result.allow_rebuild is True
    assert result.save_status == ProofEditStatus.READONLY
