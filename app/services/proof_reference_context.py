"""Reference text context used by vertical proof views."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.proof_line_utils import iter_unique_page_text_lines
from app.models import Block, Line, Page
from app.services.proof_probe_text_service import displayed_text as proof_displayed_text


@dataclass(frozen=True)
class ProofTextSlot:
    """One visible character slot in the read-only proof reference text."""

    line: Line
    char_idx: int
    start: int
    end: int

    @property
    def lookup_key(self) -> tuple[int, int]:
        return id(self.line), self.char_idx


@dataclass(frozen=True)
class ProofReferenceContext:
    """Read-only OCR context text for a page plus slot ownership."""

    page: Page
    text: str
    slots: tuple[ProofTextSlot, ...]

    def position_for(self, line: Line, char_idx: int) -> Optional[int]:
        key = (id(line), int(char_idx))
        for slot in self.slots:
            if slot.lookup_key == key:
                return slot.start
        return None

    def first_slot_for_block(self, block: Block) -> Optional[ProofTextSlot]:
        for slot in self.slots:
            if any(slot.line is line for line in block.lines):
                return slot
        return None


def build_proof_reference_context(page: Page) -> ProofReferenceContext:
    parts: list[str] = []
    slots: list[ProofTextSlot] = []
    pos = 0
    last_block: Optional[Block] = None
    for block, line, _line_idx in iter_unique_page_text_lines(page):
        if last_block is not None and block is not last_block:
            parts.append("\n")
            pos += 1
        last_block = block
        display_text = proof_displayed_text(line, page, block)
        for char_idx, _char in enumerate(display_text):
            slots.append(ProofTextSlot(line=line, char_idx=char_idx, start=pos, end=pos + 1))
            pos += 1
        parts.append(display_text)
        parts.append("\n")
        pos += 1
    return ProofReferenceContext(page=page, text="".join(parts), slots=tuple(slots))
