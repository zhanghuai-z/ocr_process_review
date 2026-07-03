"""Auto-flag proof lines through the shared proof write boundary."""
from __future__ import annotations

from app.core.proof_line_facts import proof_status
from app.models import Page, ProofStatus
from app.models.ocr_observation import block_ocr_lines
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
            for block in page.blocks:
                for line in block_ocr_lines(block):
                    if proof_status(line) in (ProofStatus.MODIFIED, ProofStatus.OK):
                        continue
                    if line.confidence >= self.threshold:
                        continue
                    result = ProofEditService.set_line_status(
                        page,
                        block,
                        line,
                        ProofStatus.AUTO_FLAGGED,
                    )
                    if result.changed:
                        count += 1
        return count
