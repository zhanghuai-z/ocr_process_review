"""校对引擎：自动标记低置信度行，提供切图辅助。"""
from __future__ import annotations
from typing import List, Tuple

from app.models import Line, Page, ProofStatus

LOW_CONFIDENCE = 0.80


class ProofEngine:
    """
    自动扫描所有行，将 confidence < 阈值 的行标记为 AUTO_FLAGGED。
    不修改已经 MODIFIED 或 OK 的行。
    """

    def __init__(self, threshold: float = LOW_CONFIDENCE) -> None:
        self.threshold = threshold

    def auto_flag(self, pages: List[Page]) -> int:
        """自动标记可疑行，返回标记数量。"""
        count = 0
        for page in pages:
            for block in page.blocks:
                for line in block.lines:
                    if line.proof_status in (ProofStatus.MODIFIED, ProofStatus.OK):
                        continue
                    if line.confidence < self.threshold:
                        line.proof_status = ProofStatus.AUTO_FLAGGED
                        count += 1
        return count

    def get_flagged_lines(self, pages: List[Page]) -> List[Tuple[Line, Page]]:
        """返回所有被标记的行及其所属页面。"""
        result = []
        for page in pages:
            for block in page.blocks:
                for line in block.lines:
                    if line.proof_status == ProofStatus.AUTO_FLAGGED:
                        result.append((line, page))
        return result

    def proof_summary(self, pages: List[Page]) -> dict:
        total = flagged = modified = ok = unchecked = 0
        for page in pages:
            for block in page.blocks:
                for line in block.lines:
                    total += 1
                    match line.proof_status:
                        case ProofStatus.AUTO_FLAGGED:
                            flagged += 1
                        case ProofStatus.MODIFIED:
                            modified += 1
                        case ProofStatus.OK:
                            ok += 1
                        case _:
                            unchecked += 1
        return {
            "total": total,
            "flagged": flagged,
            "modified": modified,
            "ok": ok,
            "unchecked": unchecked,
        }
