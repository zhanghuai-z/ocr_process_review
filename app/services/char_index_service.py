from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Optional

import cv2

from app.core.char_bbox_utils import ensure_line_char_bboxes
from app.models import BBox, Line, OcrProject, Page


@dataclass
class CharEntry:
    """单个字符索引记录。

    同时兼容旧纵校 UI 依赖的字段，以及新链路使用的页/块/行定位信息。
    """

    char: str
    page_path: str
    page_number: int
    line: Line
    char_idx: int
    bbox: BBox
    page_idx: int = 0
    block_order: int = 0
    line_idx: int = 0
    confidence: float = 0.0


CharIndexEntry = CharEntry


class CharIndexService:
    """全文字符索引。"""

    def __init__(self) -> None:
        self._entries_by_char: DefaultDict[str, list[CharEntry]] = defaultdict(list)
        self._frequency: Counter[str] = Counter()

    def build(self, pages: list[Page]) -> "CharIndexService":
        self._entries_by_char.clear()
        self._frequency.clear()

        for page_idx, page in enumerate(pages):
            page_image = cv2.imread(page.display_image_path, cv2.IMREAD_COLOR)
            for block in page.text_blocks:
                for line_idx, line in enumerate(block.lines):
                    for entry in self._iter_line_entries(
                        page_idx=page_idx,
                        page=page,
                        page_image=page_image,
                        block_order=block.order,
                        line_idx=line_idx,
                        line=line,
                    ):
                        self._entries_by_char[entry.char].append(entry)
                        self._frequency[entry.char] += 1
        return self

    def build_index(self, project: OcrProject) -> "CharIndexService":
        return self.build(project.pages)

    def query(self, char: str) -> list[CharEntry]:
        if not char:
            return []
        return list(self._entries_by_char.get(char[0], ()))

    def first_entry(self, char: str) -> Optional[CharEntry]:
        entries = self.query(char)
        return entries[0] if entries else None

    def unique_chars(self) -> int:
        return len(self._entries_by_char)

    def total_chars(self) -> int:
        return sum(self._frequency.values())

    def char_frequency(self) -> list[tuple[str, int]]:
        return sorted(self._frequency.items(), key=lambda item: (-item[1], item[0]))

    def _iter_line_entries(
        self,
        page_idx: int,
        page: Page,
        page_image,
        block_order: int,
        line_idx: int,
        line: Line,
    ) -> list[CharEntry]:
        ensure_line_char_bboxes(line, page_image=page_image)
        if line.chars:
            entries = []
            for char_idx, char in enumerate(line.chars):
                glyph = char.char[:1]
                if not glyph or glyph.isspace():
                    continue
                entries.append(
                    CharEntry(
                        char=glyph,
                        page_path=page.display_image_path,
                        page_number=page.page_number,
                        line=line,
                        char_idx=char_idx,
                        bbox=char.bbox or line.bbox,
                        page_idx=page_idx,
                        block_order=block_order,
                        line_idx=line_idx,
                        confidence=float(char.confidence),
                    )
                )
            if entries:
                return entries

        return [
            CharEntry(
                char=glyph,
                page_path=page.display_image_path,
                page_number=page.page_number,
                line=line,
                char_idx=char_idx,
                bbox=line.bbox,
                page_idx=page_idx,
                block_order=block_order,
                line_idx=line_idx,
                confidence=float(line.confidence),
            )
            for char_idx, glyph in enumerate(line.text)
            if glyph and not glyph.isspace()
        ]
