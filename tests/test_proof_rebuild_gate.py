from __future__ import annotations

from app.services.proof_edit_service import ProofEditStatus
from app.services.proof_rebuild_gate import (
    ProofRebuildDecision,
    allow_proof_rebuild,
    block_proof_rebuild,
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
