"""Shared proof-view rebuild gate.

Proof panels may need to destroy/recreate widgets after page switches, debug
filter changes, or external refreshes. This module keeps the non-UI decision
about whether a proof edit state allows that rebuild in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.proof_edit_service import ProofEditStatus


class ProofRebuildDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class ProofRebuildGateResult:
    decision: ProofRebuildDecision
    save_status: ProofEditStatus = ProofEditStatus.NOOP
    message: str = ""

    @property
    def allow_rebuild(self) -> bool:
        return self.decision == ProofRebuildDecision.ALLOW


@dataclass(frozen=True)
class ProofEditorRebuildState:
    editable: bool = True
    external_conflict: bool = False
    dirty: bool = False

    @property
    def needs_save(self) -> bool:
        return self.editable and self.dirty and not self.external_conflict


def allow_proof_rebuild(
    *,
    save_status: ProofEditStatus = ProofEditStatus.NOOP,
) -> ProofRebuildGateResult:
    return ProofRebuildGateResult(
        decision=ProofRebuildDecision.ALLOW,
        save_status=save_status,
    )


def block_proof_rebuild(
    message: str,
    *,
    save_status: ProofEditStatus = ProofEditStatus.CONFLICT,
) -> ProofRebuildGateResult:
    return ProofRebuildGateResult(
        decision=ProofRebuildDecision.BLOCK,
        save_status=save_status,
        message=message,
    )


def proof_rebuild_gate_for_save_status(
    status: ProofEditStatus,
    *,
    conflict_message: str = "当前校对内容存在保存冲突，处理后再切换视图",
) -> ProofRebuildGateResult:
    if status in {ProofEditStatus.CONFLICT, ProofEditStatus.INVALID_TARGET}:
        return block_proof_rebuild(conflict_message, save_status=status)
    return allow_proof_rebuild(save_status=status)


def proof_rebuild_gate_for_editor_state(
    state: ProofEditorRebuildState,
    *,
    save_status: ProofEditStatus = ProofEditStatus.NOOP,
    conflict_message: str = "当前校对内容存在保存冲突，处理后再切换视图",
) -> ProofRebuildGateResult:
    if state.external_conflict:
        return block_proof_rebuild(conflict_message, save_status=ProofEditStatus.CONFLICT)
    if not state.editable:
        return allow_proof_rebuild(save_status=ProofEditStatus.READONLY)
    if not state.dirty:
        return allow_proof_rebuild(save_status=ProofEditStatus.NOOP)
    return proof_rebuild_gate_for_save_status(
        save_status,
        conflict_message=conflict_message,
    )


def proof_rebuild_gate_for_reference_context(
    *,
    dirty: bool = False,
) -> ProofRebuildGateResult:
    """Gate rebuilds for read-only reference text surfaces.

    VProof's page text area is a reference projection, not the write surface for
    proof edits. Direct text differences are therefore reloadable display state,
    while actual writes must go through ProofEditService.
    """
    return proof_rebuild_gate_for_editor_state(
        ProofEditorRebuildState(editable=False, dirty=dirty),
    )


__all__ = [
    "ProofEditorRebuildState",
    "ProofRebuildDecision",
    "ProofRebuildGateResult",
    "allow_proof_rebuild",
    "block_proof_rebuild",
    "proof_rebuild_gate_for_editor_state",
    "proof_rebuild_gate_for_reference_context",
    "proof_rebuild_gate_for_save_status",
]
