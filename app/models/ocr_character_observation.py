"""Access boundary for OCR character observations.

``Line.chars`` is a current runtime projection. Application code should use this
module so character, word, and formula carriers live behind one boundary instead
of being owned directly by the mutable line model.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .project import BBox, Char, Line
from .ocr_character_observation_store import ocr_chars_for_line, set_ocr_chars_for_line


@dataclass(frozen=True)
class OcrCharOccurrence:
    line: Line
    char: Char
    char_index: int


def line_ocr_chars(line: Line) -> list[Char]:
    return ocr_chars_for_line(line, line.chars)


def line_ocr_char_count(line: Line) -> int:
    return len(line_ocr_chars(line))


def line_has_ocr_chars(line: Line) -> bool:
    return bool(line_ocr_chars(line))


def replace_line_ocr_chars(line: Line, chars: Iterable[Char]) -> None:
    projected = list(chars)
    line.chars = projected
    set_ocr_chars_for_line(line, projected)


def replace_line_ocr_char_span(line: Line, start: int, end: int, chars: Iterable[Char]) -> None:
    line_ocr_chars(line)[start:end] = list(chars)


def set_ocr_char_bbox(char: Char, bbox: BBox) -> None:
    char.bbox = bbox


def clear_line_ocr_chars(line: Line) -> None:
    replace_line_ocr_chars(line, [])


def line_ocr_char_at(line: Line, index: int) -> Char:
    return line_ocr_chars(line)[index]


def iter_line_ocr_char_occurrences(line: Line) -> Iterator[OcrCharOccurrence]:
    for char_index, char in enumerate(line_ocr_chars(line)):
        yield OcrCharOccurrence(
            line=line,
            char=char,
            char_index=char_index,
        )
