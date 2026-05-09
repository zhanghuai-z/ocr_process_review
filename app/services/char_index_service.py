"""字符索引服务：为纵校提供频次统计、稳定排序与全书同字检索。"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2

from app.core.char_bbox_utils import (
    MISSING_LINE_BBOX_FLAG,
    ensure_line_char_bboxes,
    is_meaningful_text_bbox,
)
from app.models import BBox, Char, Line, OcrProject, Page

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


def _token_kind(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return _KIND_OTHER
    if normalized.isdigit():
        return _KIND_DIGIT
    if all(_char_kind(ch) == _KIND_PUNCT for ch in normalized):
        return _KIND_PUNCT
    if all(_char_kind(ch) == _KIND_SYMBOL for ch in normalized):
        return _KIND_SYMBOL
    if any(_char_kind(ch) == _KIND_LETTER for ch in normalized):
        return _KIND_LETTER
    if any(_char_kind(ch) == _KIND_DIGIT for ch in normalized):
        return _KIND_DIGIT
    return _char_kind(normalized[0])


def _sort_key(char: str) -> Tuple[int, str, str]:
    kind = _token_kind(char)
    if kind == _KIND_DIGIT and char.strip().isdigit():
        return kind, f"{len(char.strip()):04d}:{char.strip()}", char
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
    bbox_source: str = ""
    bbox_granularity: str = ""
    token_text: str = ""
    collection_kind: str = "char"

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
        seen: Set[Tuple[int, int, str]] = set()

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
        seen: Set[Tuple[int, int, str]],
    ) -> None:
        ensure_line_char_bboxes(line, page_image=page_image)
        if MISSING_LINE_BBOX_FLAG in line.review_flags:
            return

        if not line.chars:
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
                    bbox_source="fallback",
                    bbox_granularity="fallback",
                    token_text=glyph,
                    collection_kind="char",
                    page_image=page_image,
                )
            return

        for unit in self._iter_index_units(line):
            self._maybe_add(
                unit["key"],
                line=line,
                char_idx=unit["char_idx"],
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                line_idx=line_idx,
                explicit_bbox=unit["bbox"],
                confidence=float(unit["confidence"]),
                seen=seen,
                bbox_source=unit["bbox_source"],
                bbox_granularity=unit["bbox_granularity"],
                token_text=unit["token_text"],
                collection_kind=unit["collection_kind"],
                page_image=page_image,
            )

    def _iter_index_units(self, line: Line) -> List[dict]:
        units: List[dict] = []
        chars = line.chars
        idx = 0
        while idx < len(chars):
            char_obj = chars[idx]
            glyph = char_obj.char or ""
            if not glyph or glyph.isspace():
                idx += 1
                continue

            if char_obj.bbox_granularity == "word" and char_obj.token_text:
                end = self._word_span_end(chars, idx)
                span_chars = chars[idx:end]
                units.extend(self._build_word_units(span_chars, idx, line))
                idx = end
                continue

            if glyph.isdigit():
                end = idx + 1
                while end < len(chars) and (chars[end].char or "").isdigit() and chars[end].bbox_granularity != "word":
                    end += 1
                units.append(self._build_digit_unit(chars[idx:end], idx, line))
                idx = end
                continue

            bbox = char_obj.bbox or _estimate_char_bbox(line, idx, len(chars)) or line.bbox
            units.append({
                "key": glyph,
                "char_idx": idx,
                "bbox": bbox,
                "confidence": float(char_obj.confidence),
                "bbox_source": char_obj.bbox_source or "fallback",
                "bbox_granularity": char_obj.bbox_granularity or "fallback",
                "token_text": char_obj.token_text or glyph,
                "collection_kind": "char",
            })
            idx += 1
        return units

    def _word_span_end(self, chars: List[Char], start_idx: int) -> int:
        current = chars[start_idx]
        bbox_dict = current.bbox.to_dict() if current.bbox is not None else None
        end = start_idx + 1
        while end < len(chars):
            other = chars[end]
            other_bbox = other.bbox.to_dict() if other.bbox is not None else None
            if other.bbox_granularity != current.bbox_granularity:
                break
            if (other.token_text or "") != (current.token_text or ""):
                break
            if other_bbox != bbox_dict:
                break
            end += 1
        return end

    def _merge_bboxes(self, chars: List[Char], fallback_bbox: BBox) -> BBox:
        boxes = [char.bbox.normalize() for char in chars if char.bbox is not None and char.bbox.area > 0]
        if not boxes:
            return fallback_bbox
        x1 = min(box.x for box in boxes)
        y1 = min(box.y for box in boxes)
        x2 = max(box.x2 for box in boxes)
        y2 = max(box.y2 for box in boxes)
        return BBox.from_xyxy(x1, y1, x2, y2).normalize()

    def _build_digit_unit(self, chars: List[Char], start_idx: int, line: Line) -> dict:
        token = "".join(char.char for char in chars if char.char and not char.char.isspace())
        bbox = self._merge_bboxes(chars, line.bbox)
        confidence = sum(float(char.confidence) for char in chars) / max(1, len(chars))
        return {
            "key": token,
            "char_idx": start_idx,
            "bbox": bbox,
            "confidence": confidence,
            "bbox_source": chars[0].bbox_source or "fallback",
            "bbox_granularity": chars[0].bbox_granularity or "fallback",
            "token_text": token,
            "collection_kind": "token" if len(token) > 1 else "char",
        }

    def _build_word_units(self, chars: List[Char], start_idx: int, line: Line) -> List[dict]:
        bbox = self._merge_bboxes(chars, line.bbox)
        content = [
            (offset, char)
            for offset, char in enumerate(chars)
            if char.char and not char.char.isspace() and _char_kind(char.char) not in (_KIND_PUNCT, _KIND_SYMBOL)
        ]
        if not content:
            return []

        has_digit = any(char.char.isdigit() for _, char in content)
        has_non_digit = any(not char.char.isdigit() for _, char in content)
        if has_digit and has_non_digit:
            units: List[dict] = []
            pos = 0
            while pos < len(content):
                offset, char = content[pos]
                if not char.char.isdigit():
                    pos += 1
                    continue
                end = pos + 1
                while end < len(content) and content[end][1].char.isdigit():
                    end += 1
                digit_chars = [item[1] for item in content[pos:end]]
                units.append(self._build_digit_unit(digit_chars, start_idx + offset, line))
                pos = end
            return units

        core_text = "".join(char.char for _, char in content)
        confidence = sum(float(char.confidence) for _, char in content) / max(1, len(content))
        return [{
            "key": core_text,
            "char_idx": start_idx + content[0][0],
            "bbox": bbox,
            "confidence": confidence,
            "bbox_source": chars[0].bbox_source or "fallback",
            "bbox_granularity": chars[0].bbox_granularity or "fallback",
            "token_text": chars[0].token_text or core_text,
            "collection_kind": "char" if len(core_text) == 1 else "token",
        }]

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
        seen: Set[Tuple[int, int, str]],
        bbox_source: str,
        bbox_granularity: str,
        token_text: str,
        collection_kind: str,
        page_image,
    ) -> None:
        if not glyph or glyph.isspace():
            return
        key = (id(line), char_idx, glyph)
        if key in seen:
            return
        bbox = explicit_bbox or line.bbox
        if bbox is None:
            return
        if not is_meaningful_text_bbox(page_image, bbox, token_text or glyph):
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
            bbox_source=bbox_source,
            bbox_granularity=bbox_granularity,
            token_text=token_text,
            collection_kind=collection_kind,
        )
        self._index.setdefault(glyph, []).append(entry)
        self._freq[glyph] += 1

    def query(self, char: str) -> List[CharEntry]:
        if not char:
            return []
        return list(self._index.get(char, ()))

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
