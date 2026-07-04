"""Read-only proof facts derived from the current project model.

``Line`` carries OCR observation fields only. Code outside proof editing/storage
should read proof text/status through this module so the proof state remains an
external runtime fact keyed by the line object.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.line_text_contract import line_text_contract
from app.models import Line, ProofLineState, ProofStatus
from app.models.ocr_observation import block_ocr_lines
from app.models.ocr_text_observation import line_ocr_review_flags


@dataclass(frozen=True)
class ProofLineFacts:
    text: str
    ocr_text: str
    status: ProofStatus
    confidence: float
    review_flags: tuple[str, ...]
    char_count: int

    @property
    def status_value(self) -> str:
        return self.status.value if hasattr(self.status, "value") else str(self.status)

    @property
    def is_auto_flagged(self) -> bool:
        return self.status == ProofStatus.AUTO_FLAGGED or bool(self.review_flags)

    @property
    def is_modified(self) -> bool:
        return self.status == ProofStatus.MODIFIED

    @property
    def is_confirmed(self) -> bool:
        return self.status == ProofStatus.OK


def proof_line_facts(line: object) -> ProofLineFacts:
    contract = line_text_contract(line)
    state = contract.proof_state
    text = _display_text_from_state(contract.text, state)
    return ProofLineFacts(
        text=text,
        ocr_text=contract.ocr_text,
        status=state.proof_status,
        confidence=contract.confidence,
        review_flags=line_ocr_review_flags(line),
        char_count=len(text),
    )


def proof_runtime_state(line: object) -> ProofLineState:
    return line_text_contract(line).proof_state


def proof_final_text(line: Line) -> str:
    return proof_runtime_state(line).final_text


def proof_final_text_set(line: Line) -> bool:
    return bool(proof_runtime_state(line).final_text_set)


def proof_display_text(line: Line) -> str:
    return proof_line_facts(line).text


def proof_ocr_text(line: Line) -> str:
    return proof_line_facts(line).ocr_text


def proof_block_text(block) -> str:
    return "\n".join(proof_display_text(line) for line in block_ocr_lines(block))


def proof_status(line: Line) -> ProofStatus:
    return proof_line_facts(line).status


def proof_status_value(line: Line) -> str:
    return proof_line_facts(line).status_value


def proof_search_texts(line: Line) -> list[str]:
    facts = proof_line_facts(line)
    values = [facts.text, facts.ocr_text]
    return [value for value in values if value]


def _display_text_from_state(base_text: str, state: ProofLineState) -> str:
    if state.final_text_set:
        return state.final_text
    return state.final_text or base_text
