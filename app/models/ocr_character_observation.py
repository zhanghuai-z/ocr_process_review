"""Access boundary for OCR character observations.

Application code reads character, word, and formula carriers through this
boundary so mutable lines do not own OCR character state.
"""
from __future__ import annotations

from collections.abc import Iterable

from .project import BBox, Char
from .ocr_character_observation_store import (
    ocr_chars_for_line_uid,
    set_ocr_chars_for_line_uid,
)


def line_ocr_chars_by_uid(line_uid: str) -> list[Char]:
    return ocr_chars_for_line_uid(line_uid)


def replace_line_ocr_char_observations(line_uid: str, chars: Iterable[Char]) -> None:
    set_ocr_chars_for_line_uid(line_uid, list(chars))


def set_ocr_char_bbox(char: Char, bbox: BBox) -> None:
    char.bbox = bbox
