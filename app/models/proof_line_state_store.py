"""External runtime store for per-line proof state."""
from __future__ import annotations

from .proof_line_state import ProofLineState


_STATE_BY_LINE_UID: dict[str, ProofLineState] = {}


def proof_state_for_line(line: object) -> ProofLineState:
    """Return the runtime proof state for ``line``, creating one if needed."""
    _ensure_line_like(line)
    line_uid = _line_uid(line)
    if line_uid and line_uid in _STATE_BY_LINE_UID:
        state = _STATE_BY_LINE_UID[line_uid]
        _ensure_state_uid(line, state)
        return state

    state = ProofLineState(line_uid=_line_uid(line))
    set_proof_state_for_line(line, state)
    return state


def set_proof_state_for_line(line: object, state: ProofLineState) -> None:
    """Store proof state outside the active Line model."""
    _ensure_line_like(line)
    _ensure_state_uid(line, state)
    if state.line_uid:
        _STATE_BY_LINE_UID[state.line_uid] = state


def _ensure_state_uid(line: object, state: ProofLineState) -> None:
    state.line_uid = _line_uid(line)


def _line_uid(line: object) -> str:
    return str(getattr(line, "uid", "") or "")


def _ensure_line_like(line: object) -> None:
    values = _line_dict(line)
    if "proof_state" in values:
        raise TypeError("Line proof_state field is retired")
    if "text" in values and "confidence" in values and "bbox" in values:
        return
    raise TypeError("proof_state store requires a Line object")


def _line_dict(line: object) -> dict[str, object]:
    values = getattr(line, "__dict__", None)
    return values if isinstance(values, dict) else {}
