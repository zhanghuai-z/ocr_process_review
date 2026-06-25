from __future__ import annotations

from dataclasses import dataclass, field
import time

from app.models.enums import ProofStatus


@dataclass
class ProofLineState:
    """Human proofreading truth for one OCR line."""

    line_uid: str
    final_text: str = ""
    final_text_set: bool = False
    proof_status: ProofStatus = ProofStatus.UNCHECKED
    alignment_state: str = "aligned"
    updated_at: float = field(default_factory=time.time)
