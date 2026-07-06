"""字符索引服务：为纵校提供频次统计、稳定排序与全书同字检索。"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2

from app.core.char_bbox_utils import (
    MISSING_LINE_BBOX_FLAG,
    is_meaningful_text_bbox,
    refine_line_char_bboxes,
    split_line_bbox_into_char_bboxes,
)
from app.core.ocr_ir import is_cjk_char, is_formula_char, is_formula_token
from app.core.proof_char_text import chars_display_text, is_display_carrier
from app.core.proof_geometry_quality import is_char_index_hidden_geometry
from app.core.proof_line_facts import proof_display_text
from app.core.proof_line_utils import iter_unique_page_text_line_views, iter_unique_page_text_lines
from app.core.proof_occurrence import (
    line_signature,
    proof_entry_page_identity_key,
    proof_page_identity_key,
)
from app.models import BBox, Char, Line, OcrProject, Page
from app.models.ocr_character_observation import line_ocr_char_at, line_ocr_chars_by_uid
from app.models.ocr_observation import line_ocr_bbox
from app.models.ocr_text_observation import line_has_ocr_review_flag, line_ocr_confidence

try:
    from pypinyin import Style, lazy_pinyin  # type: ignore

    _HAS_PYPINYIN = True
except ImportError:  # pragma: no cover
    _HAS_PYPINYIN = False


_KIND_LETTER = 0
_KIND_DIGIT = 1
_KIND_FORMULA = 2
_KIND_PUNCT = 3
_KIND_SYMBOL = 4
_KIND_OTHER = 5


def _char_kind(char: str) -> int:
    if not char:
        return _KIND_OTHER
    if len(char) != 1:
        return _token_kind(char)
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
    if is_formula_token(normalized):
        return _KIND_FORMULA
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
    if kind == _KIND_FORMULA:
        return kind, char.strip().lower(), char
    if kind == _KIND_LETTER:
        if _HAS_PYPINYIN and ord(char) > 0x2E80:
            py = lazy_pinyin(char, style=Style.NORMAL)
            label = py[0] if py else char.lower()
        else:
            label = char.lower()
    else:
        label = char
    return kind, label, char


def _bbox_granularity_for_index(char: Char) -> str:
    if char.bbox_granularity:
        return char.bbox_granularity
    return "char" if char.bbox is not None and char.bbox.area > 0 else "fallback"


def _display_width(char: Char) -> int:
    raw = char.char or ""
    return max(1, len(raw)) if is_display_carrier(char) else 1


def _span_display_width(chars: List[Char]) -> int:
    return sum(_display_width(char) for char in chars)


def _should_group_formula_char(char: Char, glyph: str) -> bool:
    if not glyph or glyph.isspace():
        return False
    # Latin letters and digits now have reliable EngCut/Hanwang char boxes in
    # the main path, so VProof should expose them as a-z/0-9 buckets instead of
    # re-aggregating them into opaque formula-like tokens.
    if glyph.isascii() and (glyph.isalpha() or glyph.isdigit()):
        return False
    return is_formula_char(glyph)


def _word_content_should_index_as_chars(content: List[tuple[int, Char]]) -> bool:
    if not content:
        return False
    for _offset, char in content:
        glyph = char.char or ""
        if len(glyph) != 1 or is_cjk_char(glyph):
            return False
        kind = _char_kind(glyph)
        if kind not in (_KIND_LETTER, _KIND_DIGIT):
            return False
    return True


def _is_vertical_line(bbox: BBox) -> bool:
    if bbox.w <= 0:
        return True
    return bbox.h >= bbox.w * 1.5


def _estimate_char_bbox(line: Line, idx: int, total: int) -> Optional[BBox]:
    """Fallback character bbox estimate when OCR did not provide char boxes."""
    if total <= 0:
        return None
    bbox = line_ocr_bbox(line)
    if _is_vertical_line(bbox):
        char_h = max(bbox.h / total, 1)
        return BBox(bbox.x, int(bbox.y + idx * char_h), bbox.w, int(char_h))
    char_w = max(bbox.w / total, 1)
    return BBox(int(bbox.x + idx * char_w), bbox.y, int(char_w), bbox.h)


@dataclass
class CharEntry:
    """Single character index record shared by vertical proof and proof state."""

    char: str
    page_path: str
    page_number: int
    line: Line
    char_idx: int
    bbox: BBox
    page_id: Optional[int] = None
    page_uid: str = ""
    page_idx: int = 0
    block_order: int = 0
    line_idx: int = 0
    confidence: float = 0.0
    bbox_source: str = ""
    bbox_granularity: str = ""
    token_text: str = ""
    collection_kind: str = "char"
    block_uid: str = ""
    line_uid: str = ""
    line_signature: str = ""

    @property
    def _entry_sort_key(self) -> Tuple[int, int, int, int]:
        return (
            self.page_number,
            line_ocr_bbox(self.line).y,
            line_ocr_bbox(self.line).x,
            self.char_idx,
        )


CharIndexEntry = CharEntry


def char_entry_display_text(entry: CharEntry) -> str:
    """Return the proof-visible text represented by a char index entry."""
    return str(entry.token_text or entry.char or "")


class CharIndexService:
    """全文字符索引。"""

    def __init__(self, *, include_fallback: bool = False, include_non_cjk: bool = False) -> None:
        self._include_fallback = include_fallback
        self._include_non_cjk = include_non_cjk
        self._index: Dict[str, List[CharEntry]] = {}
        self._freq: Counter[str] = Counter()
        self._image_cache: Dict[str, object] = {}

    def build(self, pages: List[Page]) -> "CharIndexService":
        self._index = {}
        self._freq = Counter()
        self._image_cache = {}
        seen: Set[Tuple[int, int, str]] = set()

        for page_idx, page in enumerate(pages):
            page_image = self._load_page_image(page.display_image_path) if self._page_needs_image(page) else None
            for view, block, line, line_idx in iter_unique_page_text_line_views(page):
                self._index_line(
                    page_idx=page_idx,
                    page=page,
                    page_image=page_image,
                    block_order=view.order,
                    block_uid=view.uid,
                    line_idx=line_idx,
                    line=line,
                    seen=seen,
                )

        for entries in self._index.values():
            entries.sort(key=lambda entry: entry._entry_sort_key)
        return self

    def build_index(self, project: OcrProject) -> "CharIndexService":
        return self.build(project.pages)

    def _load_page_image(self, path: str):
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            self._image_cache[path] = image
        return image

    def _line_needs_image(self, line: Line) -> bool:
        text = proof_display_text(line)
        if not text:
            return False
        chars = line_ocr_chars_by_uid(line.uid)
        has_tokenized_chars = any(is_display_carrier(char) for char in chars)
        if not chars or (len(chars) != len(text) and not has_tokenized_chars):
            return True
        return any(self._explicit_bbox_needs_validation(char.bbox) for char in chars)

    def _page_needs_image(self, page: Page) -> bool:
        return any(
            self._line_needs_image(line)
            for _block, line, _line_idx in iter_unique_page_text_lines(page)
        )

    @staticmethod
    def _explicit_bbox_needs_validation(bbox: Optional[BBox]) -> bool:
        if bbox is None:
            return False
        normalized = bbox.normalize()
        return normalized.w <= 2 or normalized.h <= 2 or normalized.area <= 4

    @staticmethod
    def _page_key(page: Page) -> Tuple[object, ...]:
        return proof_page_identity_key(page)

    @staticmethod
    def _entry_page_key(entry: CharEntry) -> Tuple[object, ...]:
        return proof_entry_page_identity_key(entry)

    def replace_pages(self, pages: List[Page]) -> "CharIndexService":
        """Rebuild index entries only for the given pages.

        VProof edits usually affect one page at a time. Rebuilding the whole
        book on every local edit scales poorly, so this method removes stale
        entries for the changed pages and merges freshly built page-local
        entries back into the existing full-project index.
        """
        if not pages:
            return self

        page_keys = {self._page_key(page) for page in pages}
        for glyph in list(self._index.keys()):
            kept: List[CharEntry] = []
            removed_count = 0
            for entry in self._index[glyph]:
                if self._entry_page_key(entry) in page_keys:
                    removed_count += 1
                else:
                    kept.append(entry)
            if removed_count:
                self._freq[glyph] -= removed_count
                if self._freq[glyph] <= 0:
                    self._freq.pop(glyph, None)
            if kept:
                self._index[glyph] = kept
            else:
                self._index.pop(glyph, None)

        page_service = CharIndexService(
            include_fallback=self._include_fallback,
            include_non_cjk=self._include_non_cjk,
        ).build(pages)
        for glyph, entries in page_service._index.items():
            self._index.setdefault(glyph, []).extend(entries)
            self._freq[glyph] += len(entries)
        for entries in self._index.values():
            entries.sort(key=lambda entry: entry._entry_sort_key)
        return self

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
        block_uid: str = "",
    ) -> None:
        if line_has_ocr_review_flag(line, MISSING_LINE_BBOX_FLAG):
            return

        text = proof_display_text(line)
        if not text:
            return

        chars = line_ocr_chars_by_uid(line.uid)
        has_tokenized_chars = any(is_display_carrier(char) for char in chars)
        if not chars:
            self._index_fallback_line(
                text=text,
                line=line,
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                block_uid=block_uid,
                line_idx=line_idx,
                seen=seen,
                page_image=page_image,
            )
            return
        if has_tokenized_chars and chars_display_text(chars) != text:
            return
        if len(chars) != len(text) and not has_tokenized_chars:
            return

        if not has_tokenized_chars and any((char.char or "") != text[idx] for idx, char in enumerate(chars)):
            if self._mismatched_line_geometry_is_untrusted(line, text):
                return
            self._index_positional_line(
                text=text,
                line=line,
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                block_uid=block_uid,
                line_idx=line_idx,
                seen=seen,
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
                block_uid=block_uid,
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

    def _mismatched_line_geometry_is_untrusted(self, line: Line, text: str) -> bool:
        """Return True when line geometry no longer proves the display text.

        A small number of mismatches is a normal proofing edit: the glyph text
        changes but the original char bbox is still the correct image location.
        When most of a line is mismatched, or multiple mismatches sit on
        fallback boxes, the line's text and geometry may belong to different
        physical rows.  VProof should skip that line instead of showing
        thumbnails from unrelated positions.
        """
        chars = line_ocr_chars_by_uid(line.uid)
        if not chars or len(chars) != len(text):
            return True
        mismatches = [
            char for idx, char in enumerate(chars)
            if (char.char or "") != text[idx]
        ]
        if not mismatches:
            return False
        if len(text) <= 2 and len(mismatches) == len(text):
            return True
        mismatch_threshold = max(3, int(len(text) * 0.35))
        if len(mismatches) >= mismatch_threshold:
            return True
        if len(mismatches) >= 2 and any(
            self._is_fallback_unit(char.bbox_source, char.bbox_granularity)
            for char in mismatches
        ):
            return True
        return False

    def _index_positional_line(
        self,
        *,
        text: str,
        line: Line,
        page: Page,
        page_idx: int,
        block_order: int,
        block_uid: str,
        line_idx: int,
        seen: Set[Tuple[int, int, str]],
        page_image,
    ) -> None:
        for char_idx, glyph in enumerate(text):
            char_obj = line_ocr_char_at(line, char_idx)
            explicit_bbox = char_obj.bbox or _estimate_char_bbox(line, char_idx, len(text)) or line_ocr_bbox(line)
            self._maybe_add(
                glyph,
                line=line,
                char_idx=char_idx,
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                block_uid=block_uid,
                line_idx=line_idx,
                explicit_bbox=explicit_bbox,
                confidence=float(char_obj.confidence),
                seen=seen,
                bbox_source=char_obj.bbox_source or "fallback",
                bbox_granularity=_bbox_granularity_for_index(char_obj),
                token_text=glyph,
                collection_kind="char",
                page_image=page_image,
            )

    def _index_fallback_line(
        self,
        *,
        text: str,
        line: Line,
        page: Page,
        page_idx: int,
        block_order: int,
        block_uid: str,
        line_idx: int,
        seen: Set[Tuple[int, int, str]],
        page_image,
    ) -> None:
        boxes = (
            refine_line_char_bboxes(line_ocr_bbox(line), text, page_image)
            if page_image is not None
            else split_line_bbox_into_char_bboxes(line_ocr_bbox(line), text)
        )
        for char_idx, glyph in enumerate(text):
            self._maybe_add(
                glyph,
                line=line,
                char_idx=char_idx,
                page=page,
                page_idx=page_idx,
                block_order=block_order,
                block_uid=block_uid,
                line_idx=line_idx,
                explicit_bbox=boxes[char_idx] if char_idx < len(boxes) else (_estimate_char_bbox(line, char_idx, len(text)) or line_ocr_bbox(line)),
                confidence=line_ocr_confidence(line),
                seen=seen,
                bbox_source="fallback",
                bbox_granularity="fallback",
                token_text=glyph,
                collection_kind="char",
                page_image=page_image,
            )

    def _iter_index_units(self, line: Line) -> List[dict]:
        units: List[dict] = []
        chars = line_ocr_chars_by_uid(line.uid)
        text = proof_display_text(line)
        idx = 0
        display_idx = 0
        while idx < len(chars):
            char_obj = chars[idx]
            raw_glyph = char_obj.char or ""
            glyph = raw_glyph if is_display_carrier(char_obj) else (text[display_idx] if display_idx < len(text) else raw_glyph)
            if not glyph or glyph.isspace():
                display_idx += _display_width(char_obj)
                idx += 1
                continue

            if char_obj.bbox_granularity == "word" and char_obj.token_text:
                end = self._word_span_end(chars, idx)
                span_chars = chars[idx:end]
                units.extend(self._build_word_units(span_chars, display_idx, line))
                display_idx += _span_display_width(span_chars)
                idx = end
                continue

            if _should_group_formula_char(char_obj, glyph):
                end = idx + 1
                while (
                    end < len(chars)
                    and chars[end].bbox_granularity != "word"
                    and _should_group_formula_char(chars[end], chars[end].char or "")
                ):
                    end += 1
                span_chars = chars[idx:end]
                units.append(self._build_formula_unit(span_chars, display_idx, line))
                display_idx += _span_display_width(span_chars)
                idx = end
                continue

            bbox = char_obj.bbox or _estimate_char_bbox(line, idx, len(chars)) or line_ocr_bbox(line)
            units.append({
                "key": glyph,
                "char_idx": display_idx,
                "bbox": bbox,
                "confidence": float(char_obj.confidence),
                "bbox_source": char_obj.bbox_source or "fallback",
                "bbox_granularity": _bbox_granularity_for_index(char_obj),
                "token_text": glyph,
                "collection_kind": "char",
            })
            display_idx += _display_width(char_obj)
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
        bbox = self._merge_bboxes(chars, line_ocr_bbox(line))
        confidence = sum(float(char.confidence) for char in chars) / max(1, len(chars))
        return {
            "key": token,
            "char_idx": start_idx,
            "bbox": bbox,
            "confidence": confidence,
            "bbox_source": chars[0].bbox_source or "fallback",
            "bbox_granularity": _bbox_granularity_for_index(chars[0]),
            "token_text": token,
            "collection_kind": "token" if len(token) > 1 else "char",
        }

    def _build_formula_unit(self, chars: List[Char], start_idx: int, line: Line) -> dict:
        token = "".join(char.char for char in chars if char.char and not char.char.isspace())
        bbox = self._merge_bboxes(chars, line_ocr_bbox(line))
        confidence = sum(float(char.confidence) for char in chars) / max(1, len(chars))
        return {
            "key": token,
            "char_idx": start_idx,
            "bbox": bbox,
            "confidence": confidence,
            "bbox_source": chars[0].bbox_source or "fallback",
            "bbox_granularity": _bbox_granularity_for_index(chars[0]),
            "token_text": token,
            "collection_kind": "token",
        }

    def _build_word_units(self, chars: List[Char], start_idx: int, line: Line) -> List[dict]:
        bbox = self._merge_bboxes(chars, line_ocr_bbox(line))
        raw_content = [
            (offset, char)
            for offset, char in enumerate(chars)
            if char.char and not char.char.isspace()
        ]
        raw_text = "".join(char.char for _, char in raw_content)
        if is_formula_token(raw_text):
            return [self._build_formula_unit([char for _, char in raw_content], start_idx + raw_content[0][0], line)]

        content = [
            (offset, char)
            for offset, char in enumerate(chars)
            if char.char and not char.char.isspace() and _char_kind(char.char) not in (_KIND_PUNCT, _KIND_SYMBOL)
        ]
        if not content:
            return []

        if _word_content_should_index_as_chars(content):
            units: List[dict] = []
            for offset, char in content:
                glyph = char.char
                units.append({
                    "key": glyph,
                    "char_idx": start_idx + offset,
                    "bbox": (
                        char.bbox
                        or _estimate_char_bbox(line, start_idx + offset, len(line_ocr_chars_by_uid(line.uid)))
                        or line_ocr_bbox(line)
                    ),
                    "confidence": float(char.confidence),
                    "bbox_source": char.bbox_source or "fallback",
                    "bbox_granularity": _bbox_granularity_for_index(char),
                    "token_text": glyph,
                    "collection_kind": "char",
                })
            return units

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
            "bbox_granularity": _bbox_granularity_for_index(chars[0]),
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
        block_uid: str,
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
        if not self._include_fallback and self._is_fallback_unit(bbox_source, bbox_granularity):
            return
        if not self._include_non_cjk and not self._is_cjk_index_key(glyph):
            return
        key = (id(line), char_idx, glyph)
        if key in seen:
            return
        bbox = explicit_bbox or line_ocr_bbox(line)
        if bbox is None:
            return
        validation_image = (
            page_image
            if (
                self._is_fallback_unit(bbox_source, bbox_granularity)
                or self._explicit_bbox_needs_validation(bbox)
            )
            else None
        )
        if not is_meaningful_text_bbox(validation_image, bbox, token_text or glyph):
            return
        seen.add(key)
        entry = CharEntry(
            char=glyph,
            page_path=page.display_image_path,
            page_number=page.page_number,
            line=line,
            char_idx=char_idx,
            bbox=bbox,
            page_id=page.id,
            page_uid=page.uid,
            page_idx=page_idx,
            block_order=block_order,
            line_idx=line_idx,
            confidence=confidence,
            bbox_source=bbox_source,
            bbox_granularity=bbox_granularity,
            token_text=token_text,
            collection_kind=collection_kind,
            block_uid=block_uid,
            line_uid=line.uid,
            line_signature=line_signature(line),
        )
        self._index.setdefault(glyph, []).append(entry)
        self._freq[glyph] += 1

    def _is_fallback_unit(self, bbox_source: str, bbox_granularity: str) -> bool:
        return is_char_index_hidden_geometry(bbox_source, bbox_granularity)

    def _is_cjk_index_key(self, glyph: str) -> bool:
        compact = "".join(ch for ch in str(glyph) if not ch.isspace())
        return bool(compact) and all(is_cjk_char(ch) for ch in compact)

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
