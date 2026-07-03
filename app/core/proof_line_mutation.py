"""Explicit write helpers for line proof state."""
from __future__ import annotations

from app.models import Line, ProofLineState, ProofStatus
from app.models.proof_line_state_store import proof_state_for_line, set_proof_state_for_line


def set_line_proof_text(
    line: Line,
    new_text: str,
    *,
    status: ProofStatus = ProofStatus.MODIFIED,
) -> None:
    state = _proof_state_for(line)
    state.final_text = new_text
    state.final_text_set = True
    state.proof_status = status
    apply_line_proof_state(line, state)


def set_line_proof_status(line: Line, status: ProofStatus) -> None:
    state = _proof_state_for(line)
    state.proof_status = status
    apply_line_proof_state(line, state)


def apply_line_proof_state(line: Line, state: ProofLineState) -> None:
    set_proof_state_for_line(line, state)


def _proof_state_for(line: Line) -> ProofLineState:
    return proof_state_for_line(line)
