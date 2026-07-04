from __future__ import annotations

from dataclasses import asdict, dataclass

from app.core.proof_line_facts import proof_line_facts
from app.models import OcrProject, ProofStatus
from app.models.ocr_observation import iter_project_ocr_line_occurrences


@dataclass(frozen=True)
class ProofStats:
    total_lines: int = 0
    confirmed_lines: int = 0
    modified_lines: int = 0
    flagged_lines: int = 0
    pending_lines: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ProofStatsService:
    """项目级校对统计。"""

    def summarize(self, project: OcrProject) -> ProofStats:
        total = confirmed = modified = flagged = pending = 0

        for occurrence in iter_project_ocr_line_occurrences(project):
            facts = proof_line_facts(occurrence.line)
            total += 1
            if facts.status == ProofStatus.OK:
                confirmed += 1
            elif facts.status == ProofStatus.MODIFIED:
                modified += 1
            elif facts.is_auto_flagged:
                flagged += 1
            else:
                pending += 1

        return ProofStats(
            total_lines=total,
            confirmed_lines=confirmed,
            modified_lines=modified,
            flagged_lines=flagged,
            pending_lines=pending,
        )
