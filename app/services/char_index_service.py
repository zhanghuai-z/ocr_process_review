"""字符索引服务：为纵校提供频次统计、稳定排序与全书同字检索。"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2

from app.core.char_bbox_utils import ensure_line_char_bboxes
from app.models import BBox, Line, OcrProject, Page

try:
    from pypinyin import Style, lazy_pinyin  # type: ignore

    _HAS_PYPINYIN = True
except ImportError:  # pragma: no cover
    _HAS_PYPINYIN = False


_KIND_LETTER = 0
_KIND_DIGIT = 1
_KIND_PUNCT = 2
_KIND_SYMBOL = 3
_KIND_OTHER = 4


def _char_kind(char: str) -> int:
    if not char:
        return _KIND_OTHER
    cat = unicodedata.category(char)
    if cat.startswith("L"):
        return _KIND_LETTER
    if cat.startswith("N"):
        return _KIND_DIGIT
    if cat.startswith("P"):
        return _KIND_PUNCT
    if cat.startswith("S"):
        return _KIND_SYMBOL
    return _KIND_OTHER


def _sort_key(char: str) -> Tuple[int, str, str]:
    kind = _char_kind(char)
    if kind == _KIND_LETTER:
        if _HAS_PYPINYIN and ord(char) > 0x2E80:
            py = lazy_pinyin(char, style=Style.NORMAL)
            label = py[0] if py else char.lower()
        else:
            label = char.lower()
    else:
        label = char
    return kind, label, char


def _is_vertical_line(bbox: BBox) -> bool:
    if bbox.w <= 0:
        return True
    return bbox.h >= bbox.w * 1.5


def _estimate_char_bbox(line: Line, idx: int, total: int) -> Optional[BBox]:
    """兼容旧测试与旧纵校路径的字符框估算。"""
    if total <= 0:
        return None
    bbox = line.bbox
    if _is_vertical_line(bbox):
        char_h = max(bbox.h / total, 1)
        return BBox(bbox.x, int(bbox.y + idx * char_h), bbox.w, int(char_h))
    char_w = max(bbox.w / total, 1)
    return BBox(int(bbox.x + idx * char_w), bbox.y, int(char_w), bbox.h)


@dataclass
class CharEntry:
    """单个字符索引记录，兼容旧纵校 UI 与新 proof 链路。"""

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

    @property
    def _entry_sort_key(self) -> Tuple[int, int, int, int]:
        return (
            self.page_number,
            self.line.bbox.y,
            self.line.bbox.x,
            self.char_idx,
        )


CharIndexEntry = CharEntry


class CharIndexService:
    """全文字符索引。"""

    def __init__(self) -> None:
        self._index: Dict[str, List[CharEntry]] = {}
        self._freq: Counter[str] = Counter()

    def build(self, pages: List[Page]) -> "CharIndexService":
        self._index = {}
        self._freq = Counter()
        seen: Set[Tuple[int, int]] = set()

        for page_idx, page in enumerate(pages):
            page_image = cv2.imread(page.display_image_path, cv2.IMREAD_COLOR)
            for block in page.text_blocks:
                for line_idx, line in enumerate(block.lines):
                    self._index_line(
                        page_idx=page_idx,
                        page=page,
                        page_image=page_image,
                        block_order=block.order,
                        line_idx=line_idx,
                        line=line,
                        seen=seen,
                    )

        for entries in self._index.values():
            entries.sort(key=lambda entry: entry._entry_sort_key)
        return self

    def build_index(self, project: OcrProject) -> "CharIndexService":
        return self.build(project.pages)

    def _index_line(
        self,
        *,
        page_idx: int,
        page: Page,
        page_image,
        block_order: int,
        line_idx: int,
        line: Line,
        seen: Set[Tuple[int, int]],
    ) -> None:
        ensure_line_char_bboxes(line, page_image=page_image)

        if line.chars:
            for char_idx, char_obj in enumerate(line.chars):
                self._maybe_add(
                    char_obj.char,
                    line=line,
                    char_idx=char_idx,
                    page=page,
                    page_idx=page_idx,
                    block_order=block_order,
                    line_idx=line_idx,
                    explicit_bbox=char_obj.bbox,
                    confidence=float(char_obj.confidence),
                    seen=seen,
                )
            return

        text = line.text or ""
        for char_idx, glyph in enumerate(text):
            self._maybe_add(
                glyph,
                line=line,
                char_idx=char_idx,
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                line_idx=line_idx,
                explicit_bbox=line.bbox,
                confidence=float(line.confidence),
                seen=seen,
            )

    def _maybe_add(
        self,
        glyph: str,
        *,
        line: Line,
        char_idx: int,
        page: Page,
        page_idx: int,
        block_order: int,
        line_idx: int,
        explicit_bbox: Optional[BBox],
        confidence: float,
        seen: Set[Tuple[int, int]],
    ) -> None:
        if not glyph or glyph.isspace():
            return
        key = (id(line), char_idx)
        if key in seen:
            return
        bbox = explicit_bbox or line.bbox
        if bbox is None:
            return
        seen.add(key)
        entry = CharEntry(
            char=glyph,
            page_path=page.display_image_path,
            page_number=page.page_number,
            line=line,
            char_idx=char_idx,
            bbox=bbox,
            page_idx=page_idx,
            block_order=block_order,
            line_idx=line_idx,
            confidence=confidence,
        )
        self._index.setdefault(glyph, []).append(entry)
        self._freq[glyph] += 1

    def query(self, char: str) -> List[CharEntry]:
        if not char:
            return []
        return list(self._index.get(char[0], ()))

    def first_entry(self, char: str) -> Optional[CharEntry]:
        entries = self.query(char)
        return entries[0] if entries else None

    def unique_chars(self) -> int:
        return len(self._index)

    def total_chars(self) -> int:
        return sum(self._freq.values())

    def char_frequency(self) -> List[Tuple[str, int]]:
        return sorted(self._freq.items(), key=lambda item: (-item[1], _sort_key(item[0])))

    def sorted_chars(self) -> List[Tuple[str, int]]:
        return sorted(self._freq.items(), key=lambda item: _sort_key(item[0]))
