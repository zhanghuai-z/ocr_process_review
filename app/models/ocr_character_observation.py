"""Access boundary for OCR character observations currently attached to lines.

``Line.chars`` is still the physical runtime storage for character, word, and
formula carriers.  Application code should use this module so character
observations can later move out of the mutable line model without another broad
rewrite.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .project import BBox, Char, Line


@dataclass(frozen=True)
class OcrCharOccurrence:
    line: Line
    char: Char
    char_index: int


def line_ocr_chars(line: Line) -> list[Char]:
    return line.chars


def line_ocr_char_count(line: Line) -> int:
    return len(line_ocr_chars(line))


def line_has_ocr_chars(line: Line) -> bool:
    return bool(line_ocr_chars(line))


def replace_line_ocr_chars(line: Line, chars: Iterable[Char]) -> None:
    line.chars = list(chars)


def replace_line_ocr_char_span(line: Line, start: int, end: int, chars: Iterable[Char]) -> None:
    line.chars[start:end] = list(chars)


def set_ocr_char_bbox(char: Char, bbox: BBox) -> None:
    char.bbox = bbox


def clear_line_ocr_chars(line: Line) -> None:
    line.chars = []


def line_ocr_char_at(line: Line, index: int) -> Char:
    return line_ocr_chars(line)[index]


def iter_line_ocr_char_occurrences(line: Line) -> Iterator[OcrCharOccurrence]:
    for char_index, char in enumerate(line_ocr_chars(line)):
        yield OcrCharOccurrence(
            line=line,
            char=char,
            char_index=char_index,
        )
