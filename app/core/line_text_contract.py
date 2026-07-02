"""Normalize OCR line text fields at explicit workflow boundaries."""
from __future__ import annotations

from app.models.proof_line_state import ProofLineState


def ensure_line_text_contract(line: object) -> None:
    """Ensure OCR text fields and proof state are internally consistent.

    This is intentionally a workflow helper, not a ``Line`` method. It keeps
    model objects passive while storage/OCR boundaries remain responsible for
    normalizing legacy or partially constructed line data.
    """
    uid = str(getattr(line, "uid", "") or "")
    state = getattr(line, "proof_state", None)
    if isinstance(state, ProofLineState):
        state.line_uid = uid
    else:
        setattr(line, "proof_state", ProofLineState(line_uid=uid))

    text = str(getattr(line, "text", "") or "")
    ocr_text = str(getattr(line, "ocr_text", "") or text)
    if not text:
        setattr(line, "text", ocr_text)
    setattr(line, "ocr_text", ocr_text)
