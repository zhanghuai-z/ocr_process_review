"""Explicit write helpers for line proof state."""
from __future__ import annotations

from app.models import Line, ProofLineState, ProofStatus


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
    state.line_uid = line.uid
    object.__setattr__(line, "proof_state", state)


def _proof_state_for(line: Line) -> ProofLineState:
    state = getattr(line, "proof_state", None)
    if isinstance(state, ProofLineState):
        return state
    return ProofLineState(line_uid=line.uid)
