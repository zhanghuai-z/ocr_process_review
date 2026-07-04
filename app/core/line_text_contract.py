"""Normalize OCR line text fields at explicit workflow boundaries."""
from __future__ import annotations

from dataclasses import dataclass, replace

from app.models.proof_line_state import ProofLineState
from app.models.proof_line_state_store import proof_state_for_line, set_proof_state_for_line
from app.models.ocr_text_observation import (
    line_ocr_text_observation,
    set_line_ocr_text_observation,
)


@dataclass(frozen=True)
class LineTextContract:
    text: str
    ocr_text: str
    confidence: float
    proof_state: ProofLineState


def line_text_contract(line: object) -> LineTextContract:
    """Return normalized OCR/proof line facts without mutating ``line``."""
    uid = str(getattr(line, "uid", "") or "")
    state = proof_state_for_line(line)
    observation = line_ocr_text_observation(line)
    text = observation.text
    ocr_text = observation.ocr_text or text
    normalized_text = text or ocr_text

    proof_state = state if state.line_uid == uid else replace(state, line_uid=uid)
    return LineTextContract(
        text=normalized_text,
        ocr_text=ocr_text,
        confidence=float(observation.confidence or 0.0),
        proof_state=proof_state,
    )


def ensure_line_text_contract(line: object) -> None:
    """Ensure OCR text fields and proof state are internally consistent.

    This is intentionally a workflow helper, not a ``Line`` method. It keeps
    model objects passive while storage/OCR boundaries remain responsible for
    normalizing partially constructed line data.
    """
    contract = line_text_contract(line)
    set_proof_state_for_line(line, contract.proof_state)
    observation = line_ocr_text_observation(line)
    set_line_ocr_text_observation(
        line,
        replace(observation, text=contract.text, ocr_text=contract.ocr_text),
    )
