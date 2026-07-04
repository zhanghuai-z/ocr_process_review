"""Proof atom construction for proofreading views.

The proof UI should not infer whether a visual object is a single character,
word, formula, or table region. This module compiles OCR/layout observations
into a stable atom sequence that UI renderers can consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Iterable

from app.core.block_attributes import normalize_source_label, semantic_block_type
from app.core.proof_line_facts import proof_display_text
from app.core.proof_char_text import char_display_text, chars_display_text
from app.models import BBox, Block, BlockType, Char, Line
from app.models.ocr_character_observation import line_ocr_chars


class ProofAtomKind(str, Enum):
    CHAR = "char"
    PUNCT = "punct"
    NUMBER = "number"
    WORD = "word"
    TOKEN = "token"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProofAtom:
    kind: ProofAtomKind
    text: str
    edit_text: str
    bbox: BBox | None
    char_indices: tuple[int, ...] = ()
    source: str = ""
    confidence: float = 0.0
    reliable: bool = True
    reason: str = ""

    @property
    def is_textual(self) -> bool:
        return self.kind in {
            ProofAtomKind.CHAR,
            ProofAtomKind.PUNCT,
            ProofAtomKind.NUMBER,
            ProofAtomKind.WORD,
            ProofAtomKind.TOKEN,
            ProofAtomKind.FORMULA,
        }


_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")


def build_line_proof_atoms(block: Block | None, line: Line) -> list[ProofAtom]:
    """Build a visual atom sequence for a proof line.

    The builder keeps reliable single-character geometry as char-like atoms and
    downgrades unstable or multi-character observations to token/word atoms.
    """
    if block is not None:
        block_kind = semantic_block_type(block)
        if block_kind == BlockType.TABLE:
            return [_region_atom(ProofAtomKind.TABLE, line)]
        if block_kind == BlockType.FIGURE:
            return [_region_atom(ProofAtomKind.IMAGE, line)]
        if block_kind == BlockType.EQUATION:
            return [_region_atom(ProofAtomKind.FORMULA, line)]

    chars = list(line_ocr_chars(line))
    if not chars:
        return [_region_atom(ProofAtomKind.UNKNOWN, line, reliable=False, reason="line_without_chars")]
    reconstructed_text = chars_display_text(chars)
    if reconstructed_text != proof_display_text(line):
        return [_region_atom(
            ProofAtomKind.TOKEN,
            line,
            reliable=False,
            reason="chars_do_not_match_display_text",
        )]

    atoms: list[ProofAtom] = []
    pending: list[tuple[int, Char, str, str]] = []

    def flush_pending(reason: str = "") -> None:
        nonlocal pending
        if not pending:
            return
        indices = tuple(idx for idx, _char, _text, _source in pending)
        text = "".join(text for _idx, _char, text, _source in pending)
        bbox = _union_bbox(char.bbox for _idx, char, _text, _source in pending)
        confidence = _avg(char.confidence for _idx, char, _text, _source in pending)
        source = _join_sources(source for _idx, _char, _text, source in pending)
        atoms.append(
            ProofAtom(
                kind=_kind_for_token(text),
                text=text,
                edit_text=text,
                bbox=bbox,
                char_indices=indices,
                source=source,
                confidence=confidence,
                reliable=False,
                reason=reason or "grouped_unstable_chars",
            )
        )
        pending = []

    for idx, char in enumerate(chars):
        text = _char_text(char)
        source = normalize_source_label(char.bbox_source)
        if _is_formula_observation(char, source):
            flush_pending()
            atoms.append(
                ProofAtom(
                    kind=ProofAtomKind.FORMULA,
                    text=text,
                    edit_text=text,
                    bbox=char.bbox,
                    char_indices=(idx,),
                    source=source or char.bbox_source,
                    confidence=float(char.confidence or 0.0),
                    reliable=bool(char.bbox),
                    reason="" if char.bbox else "formula_without_bbox",
                )
            )
            continue

        if _is_stable_single_char(chars, idx):
            flush_pending()
            atoms.append(
                ProofAtom(
                    kind=_kind_for_single_char(text),
                    text=text,
                    edit_text=text,
                    bbox=char.bbox,
                    char_indices=(idx,),
                    source=source or char.bbox_source,
                    confidence=float(char.confidence or 0.0),
                    reliable=True,
                )
            )
            continue

        if _is_low_confidence_number_or_punct_slot(char, text):
            flush_pending()
            atoms.append(
                ProofAtom(
                    kind=_kind_for_single_char(text),
                    text=text,
                    edit_text=text,
                    bbox=char.bbox,
                    char_indices=(idx,),
                    source=source or char.bbox_source,
                    confidence=float(char.confidence or 0.0),
                    reliable=False,
                    reason="low_confidence_single_char",
                )
            )
            continue

        pending.append((idx, char, text, source or char.bbox_source))

    flush_pending()
    return atoms


def has_non_char_atoms(atoms: Iterable[ProofAtom]) -> bool:
    return any(atom.kind not in {ProofAtomKind.CHAR, ProofAtomKind.PUNCT, ProofAtomKind.NUMBER} for atom in atoms)


def _region_atom(
    kind: ProofAtomKind,
    line: Line,
    *,
    reliable: bool = True,
    reason: str = "",
) -> ProofAtom:
    text = proof_display_text(line)
    return ProofAtom(
        kind=kind,
        text=text,
        edit_text=text,
        bbox=line.bbox,
        source="layout_region",
        confidence=float(line.confidence or 0.0),
        reliable=reliable,
        reason=reason,
    )


def _char_text(char: Char) -> str:
    return char_display_text(char)


def _is_formula_observation(char: Char, source: str) -> bool:
    return "formula" in source or "equation" in source or normalize_source_label(char.bbox_granularity) == "formula"


def _is_stable_single_char(chars: list[Char], idx: int) -> bool:
    char = chars[idx]
    text = _char_text(char)
    if len(text) != 1:
        return False
    if char.bbox is None:
        return False
    if normalize_source_label(char.bbox_granularity) != "char":
        return False
    if float(char.confidence or 0.0) < 0.25:
        return False
    if _overlaps_neighbor_too_much(chars, idx):
        return False
    if _breaks_x_order(chars, idx):
        return False
    return True


def _is_low_confidence_number_or_punct_slot(char: Char, text: str) -> bool:
    if len(text) != 1:
        return False
    if char.bbox is None:
        return False
    if normalize_source_label(char.bbox_granularity) != "char":
        return False
    return _kind_for_single_char(text) in {ProofAtomKind.NUMBER, ProofAtomKind.PUNCT}


def _overlaps_neighbor_too_much(chars: list[Char], idx: int) -> bool:
    bbox = chars[idx].bbox
    if bbox is None:
        return False
    for n_idx in (idx - 1, idx + 1):
        if n_idx < 0 or n_idx >= len(chars):
            continue
        other = chars[n_idx].bbox
        if other is None:
            continue
        overlap = _horizontal_overlap_ratio(bbox, other)
        if overlap >= 0.45:
            return True
    return False


def _breaks_x_order(chars: list[Char], idx: int) -> bool:
    bbox = chars[idx].bbox
    if bbox is None:
        return False
    cx = (bbox.x1 + bbox.x2) / 2.0
    if idx > 0 and chars[idx - 1].bbox is not None:
        prev = chars[idx - 1].bbox
        prev_cx = (prev.x1 + prev.x2) / 2.0
        if cx < prev_cx - 1.0:
            return True
    if idx + 1 < len(chars) and chars[idx + 1].bbox is not None:
        nxt = chars[idx + 1].bbox
        next_cx = (nxt.x1 + nxt.x2) / 2.0
        if cx > next_cx + 1.0:
            return True
    return False


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    denom = max(1, min(abs(a.w), abs(b.w)))
    return overlap / denom


def _kind_for_single_char(text: str) -> ProofAtomKind:
    if _DIGIT_RE.fullmatch(text):
        return ProofAtomKind.NUMBER
    if _is_punctuation_token(text):
        return ProofAtomKind.PUNCT
    return ProofAtomKind.CHAR


def _kind_for_token(text: str) -> ProofAtomKind:
    stripped = text.strip()
    if not stripped:
        return ProofAtomKind.TOKEN
    if _DIGIT_RE.fullmatch(stripped):
        return ProofAtomKind.NUMBER
    if _LATIN_RE.search(stripped):
        return ProofAtomKind.WORD
    if _is_punctuation_token(stripped):
        return ProofAtomKind.PUNCT
    return ProofAtomKind.TOKEN


def _is_punctuation_token(text: str) -> bool:
    if not text:
        return False
    return all(ch.isspace() or unicodedata.category(ch).startswith("P") for ch in text)


def _union_bbox(boxes: Iterable[BBox | None]) -> BBox | None:
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    x1 = min(box.x1 for box in present)
    y1 = min(box.y1 for box in present)
    x2 = max(box.x2 for box in present)
    y2 = max(box.y2 for box in present)
    return BBox.from_xyxy(x1, y1, x2, y2)


def _avg(values: Iterable[float]) -> float:
    nums = [float(value or 0.0) for value in values]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def _join_sources(values: Iterable[str]) -> str:
    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return "+".join(unique)
