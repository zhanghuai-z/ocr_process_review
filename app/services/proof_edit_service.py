"""Single write boundary for proof edits."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from app.core.proof_change import ProofChangeSet
from app.core.proof_line_facts import proof_display_text, proof_status
from app.core.proof_line_mutation import set_line_proof_status
from app.core.proof_occurrence import ProofOccurrence, line_signature
from app.models import Block, Line, Page, ProofStatus
from app.services.proof_probe_text_service import save_displayed_edit_result


class ProofEditStatus(str, Enum):
    NOOP = "noop"
    SAVED = "saved"
    CONFLICT = "conflict"
    INVALID_TARGET = "invalid_target"
    READONLY = "readonly"


@dataclass(frozen=True)
class ProofEditResult:
    status: ProofEditStatus
    change: ProofChangeSet = ProofChangeSet()
    message: str = ""

    @property
    def changed(self) -> bool:
        return self.change.changed

    @property
    def blocked(self) -> bool:
        return self.status in {
            ProofEditStatus.CONFLICT,
            ProofEditStatus.INVALID_TARGET,
            ProofEditStatus.READONLY,
        }


@dataclass(frozen=True)
class ProofSpanReplacement:
    start: int
    end: int
    text: str


class ProofEditService:
    """Apply proof edits to the shared Page/Block/Line model.

    UI panels should submit intent here instead of mutating line text/status
    directly.  The service verifies that the target still belongs to the page,
    optionally checks the caller's line signature, then returns a scoped
    ProofChangeSet for persistence.
    """

    @staticmethod
    def replace_line_text(
        page: Page,
        block: Block,
        line: Line,
        displayed_text: str,
        *,
        expected_signature: str = "",
        write_chars: bool | None = None,
    ) -> ProofEditResult:
        validation = ProofEditService._validate_target(
            page, block, line, expected_signature=expected_signature,
        )
        if validation is not None:
            return validation
        change = save_displayed_edit_result(line, page, block, displayed_text)
        if not change.changed:
            return ProofEditResult(ProofEditStatus.NOOP, change)
        return ProofEditResult(
            ProofEditStatus.SAVED,
            change.scoped_to_line(
                page,
                block,
                line,
                write_chars=change.text_changed if write_chars is None else write_chars,
            ),
        )

    @staticmethod
    def replace_spans(
        page: Page,
        block: Block,
        line: Line,
        replacements: Iterable[ProofSpanReplacement],
        *,
        expected_signature: str = "",
    ) -> ProofEditResult:
        validation = ProofEditService._validate_target(
            page, block, line, expected_signature=expected_signature,
        )
        if validation is not None:
            return validation
        current = proof_display_text(line)
        ranges = sorted(
            (
                ProofSpanReplacement(
                    max(0, item.start),
                    min(len(current), max(item.start, item.end)),
                    item.text,
                )
                for item in replacements
            ),
            key=lambda item: item.start,
        )
        if not ranges:
            return ProofEditResult(ProofEditStatus.NOOP)
        prev_end = -1
        for item in ranges:
            if item.start < prev_end or item.start > item.end:
                return ProofEditResult(
                    ProofEditStatus.INVALID_TARGET,
                    ProofChangeSet(cancelled=True),
                    "replacement spans overlap or are invalid",
                )
            prev_end = item.end
        edited = current
        for item in reversed(ranges):
            edited = edited[:item.start] + item.text + edited[item.end:]
        return ProofEditService.replace_line_text(
            page,
            block,
            line,
            edited,
            write_chars=True,
        )

    @staticmethod
    def replace_occurrence_text(
        page: Page,
        block: Block,
        line: Line,
        occurrence: ProofOccurrence,
        text: str,
    ) -> ProofEditResult:
        return ProofEditService.replace_spans(
            page,
            block,
            line,
            [ProofSpanReplacement(occurrence.span_start, occurrence.span_end, text)],
            expected_signature=occurrence.line_signature,
        )

    @staticmethod
    def set_line_status(
        page: Page,
        block: Block,
        line: Line,
        status: ProofStatus,
        *,
        expected_signature: str = "",
    ) -> ProofEditResult:
        validation = ProofEditService._validate_target(
            page, block, line, expected_signature=expected_signature,
        )
        if validation is not None:
            return validation
        if proof_status(line) == status:
            return ProofEditResult(ProofEditStatus.NOOP)
        set_line_proof_status(line, status)
        change = ProofChangeSet(status_changed=True).scoped_to_line(
            page,
            block,
            line,
            write_chars=False,
        )
        return ProofEditResult(ProofEditStatus.SAVED, change)

    @staticmethod
    def _validate_target(
        page: Page,
        block: Block,
        line: Line,
        *,
        expected_signature: str = "",
    ) -> ProofEditResult | None:
        if block not in page.blocks or line not in block.lines:
            return ProofEditResult(
                ProofEditStatus.INVALID_TARGET,
                ProofChangeSet(cancelled=True),
                "line is no longer owned by the supplied page/block",
            )
        if expected_signature and expected_signature != line_signature(line):
            return ProofEditResult(
                ProofEditStatus.CONFLICT,
                ProofChangeSet(conflict=True),
                "line changed after the proof occurrence was built",
            )
        return None
