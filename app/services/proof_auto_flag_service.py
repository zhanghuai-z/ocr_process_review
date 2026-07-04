"""Auto-flag proof lines through the shared proof write boundary."""
from __future__ import annotations

from app.core.proof_line_facts import proof_status
from app.models import Page, ProofStatus
from app.models.ocr_observation import iter_page_ocr_line_occurrences
from app.models.ocr_text_observation import line_ocr_confidence
from app.services.proof_edit_service import ProofEditService

LOW_CONFIDENCE = 0.80


class ProofAutoFlagService:
    """Mark low-confidence proof lines without bypassing ProofEditService."""

    def __init__(self, threshold: float = LOW_CONFIDENCE) -> None:
        self.threshold = threshold

    def auto_flag(self, pages: list[Page]) -> int:
        """Auto-flag suspicious lines, returning the number of changed lines."""
        count = 0
        for page in pages:
            for occurrence in iter_page_ocr_line_occurrences(page):
                line = occurrence.line
                if proof_status(line) in (ProofStatus.MODIFIED, ProofStatus.OK):
                    continue
                if line_ocr_confidence(line) >= self.threshold:
                    continue
                result = ProofEditService.set_line_status(
                    page,
                    occurrence.block,
                    line,
                    ProofStatus.AUTO_FLAGGED,
                )
                if result.changed:
                    count += 1
        return count
