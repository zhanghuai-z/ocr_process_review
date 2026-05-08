from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict

from app.core.char_bbox_utils import ensure_line_char_bboxes
from app.models import BBox, Line, OcrProject


@dataclass(frozen=True)
class CharIndexEntry:
    char: str
    page_idx: int
    block_order: int
    line_idx: int
    char_idx: int
    bbox: BBox
    confidence: float


class CharIndexService:
    """全文字符索引。"""

    def __init__(self) -> None:
        self._entries_by_char: DefaultDict[str, list[CharIndexEntry]] = defaultdict(list)
        self._frequency: Counter[str] = Counter()

    def build_index(self, project: OcrProject) -> "CharIndexService":
        self._entries_by_char.clear()
        self._frequency.clear()

        for page_idx, page in enumerate(project.pages):
            for block in page.blocks:
                for line_idx, line in enumerate(block.lines):
                    for entry in self._iter_line_entries(page_idx, block.order, line_idx, line):
                        self._entries_by_char[entry.char].append(entry)
                        self._frequency[entry.char] += 1
        return self

    def query(self, char: str) -> list[CharIndexEntry]:
        if not char:
            return []
        return list(self._entries_by_char.get(char[0], ()))

    def char_frequency(self) -> list[tuple[str, int]]:
        return sorted(self._frequency.items(), key=lambda item: (-item[1], item[0]))

    def _iter_line_entries(
        self, page_idx: int, block_order: int, line_idx: int, line: Line,
    ) -> list[CharIndexEntry]:
        ensure_line_char_bboxes(line)
        if line.chars:
            entries = []
            for char_idx, char in enumerate(line.chars):
                glyph = char.char[:1]
                if not glyph:
                    continue
                entries.append(
                    CharIndexEntry(
                        char=glyph,
                        page_idx=page_idx,
                        block_order=block_order,
                        line_idx=line_idx,
                        char_idx=char_idx,
                        bbox=char.bbox or line.bbox,
                        confidence=float(char.confidence),
                    )
                )
            if entries:
                return entries

        return [
            CharIndexEntry(
                char=glyph,
                page_idx=page_idx,
                block_order=block_order,
                line_idx=line_idx,
                char_idx=char_idx,
                bbox=line.bbox,
                confidence=float(line.confidence),
            )
            for char_idx, glyph in enumerate(line.text)
            if glyph
        ]
