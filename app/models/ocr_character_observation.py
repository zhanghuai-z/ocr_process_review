"""Access boundary for OCR character observations.

Application code reads character, word, and formula carriers through this
boundary so mutable lines do not own OCR character state.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .project import BBox, Char, Line
from .ocr_character_observation_store import (
    ocr_chars_for_line,
    ocr_chars_for_line_uid,
    set_ocr_chars_for_line_uid,
)


@dataclass(frozen=True)
class OcrCharOccurrence:
    line: Line
    char: Char
    char_index: int


def line_ocr_chars(line: Line) -> list[Char]:
    return ocr_chars_for_line(line)


def line_ocr_chars_by_uid(line_uid: str) -> list[Char]:
    return ocr_chars_for_line_uid(line_uid)


def line_ocr_char_count(line: Line) -> int:
    return len(line_ocr_chars(line))


def line_has_ocr_chars(line: Line) -> bool:
    return bool(line_ocr_chars(line))


def replace_line_ocr_char_observations(line_uid: str, chars: Iterable[Char]) -> None:
    set_ocr_chars_for_line_uid(line_uid, list(chars))


def set_ocr_char_bbox(char: Char, bbox: BBox) -> None:
    char.bbox = bbox


def line_ocr_char_at(line: Line, index: int) -> Char:
    return line_ocr_chars(line)[index]


def iter_line_ocr_char_occurrences(line: Line) -> Iterator[OcrCharOccurrence]:
    for char_index, char in enumerate(line_ocr_chars(line)):
        yield OcrCharOccurrence(
            line=line,
            char=char,
            char_index=char_index,
        )
