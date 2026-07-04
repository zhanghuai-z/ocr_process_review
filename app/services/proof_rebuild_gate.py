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


__all__ = [
    "ProofRebuildDecision",
    "ProofRebuildGateResult",
    "allow_proof_rebuild",
    "block_proof_rebuild",
    "proof_rebuild_gate_for_save_status",
]
