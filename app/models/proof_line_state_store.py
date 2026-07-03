"""External runtime store for per-line proof state."""
from __future__ import annotations

import weakref
from typing import Any

from .proof_line_state import ProofLineState


_StateEntry = tuple[weakref.ReferenceType[object], ProofLineState]
_STATE_BY_OBJECT_ID: dict[int, _StateEntry] = {}


def proof_state_for_line(line: object) -> ProofLineState:
    """Return the runtime proof state for ``line``, creating one if needed."""
    _ensure_line_like(line)
    key = id(line)
    entry = _STATE_BY_OBJECT_ID.get(key)
    if entry is not None:
        ref, state = entry
        if ref() is line:
            _ensure_state_uid(line, state)
            return state
        _STATE_BY_OBJECT_ID.pop(key, None)

    state = _consume_legacy_state_attr(line) or ProofLineState(line_uid=_line_uid(line))
    set_proof_state_for_line(line, state)
    return state


def set_proof_state_for_line(line: object, state: ProofLineState) -> None:
    """Store proof state outside the active Line model."""
    _ensure_line_like(line)
    _ensure_state_uid(line, state)
    key = id(line)

    def _cleanup(_ref: weakref.ReferenceType[object], *, object_id: int = key) -> None:
        entry = _STATE_BY_OBJECT_ID.get(object_id)
        if entry is not None and entry[0] is _ref:
            _STATE_BY_OBJECT_ID.pop(object_id, None)

    try:
        line_ref = weakref.ref(line, _cleanup)
    except TypeError as exc:
        raise TypeError("proof_state store requires a weak-referenceable Line object") from exc
    _STATE_BY_OBJECT_ID[key] = (line_ref, state)
    _remove_legacy_state_attr(line)


def clear_proof_state_for_line(line: object) -> None:
    _STATE_BY_OBJECT_ID.pop(id(line), None)
    _remove_legacy_state_attr(line)


def _ensure_state_uid(line: object, state: ProofLineState) -> None:
    state.line_uid = _line_uid(line)


def _line_uid(line: object) -> str:
    return str(getattr(line, "uid", "") or "")


def _ensure_line_like(line: object) -> None:
    values = _line_dict(line)
    if "text" in values and "confidence" in values and "bbox" in values:
        return
    raise TypeError("proof_state store requires a Line object")


def _consume_legacy_state_attr(line: object) -> ProofLineState | None:
    values = _line_dict(line)
    state = values.pop("proof_state", None)
    return state if isinstance(state, ProofLineState) else None


def _remove_legacy_state_attr(line: object) -> None:
    _line_dict(line).pop("proof_state", None)


def _line_dict(line: object) -> dict[str, Any]:
    values = getattr(line, "__dict__", None)
    return values if isinstance(values, dict) else {}
