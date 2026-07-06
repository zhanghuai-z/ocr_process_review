"""Runtime store for OCR character observations.

Active code should go through ``ocr_character_observation`` helpers, which use
stable line uid as the current character observation boundary.
"""
from __future__ import annotations

from .project import Char


_CHARS_BY_LINE_UID: dict[str, list[Char]] = {}


def ocr_chars_for_line_uid(line_uid: str) -> list[Char]:
    return _CHARS_BY_LINE_UID.get(str(line_uid or ""), [])


def set_ocr_chars_for_line_uid(line_uid: str, chars: list[Char]) -> None:
    uid = str(line_uid or "")
    if not uid:
        return
    _CHARS_BY_LINE_UID[uid] = chars


__all__ = [
    "ocr_chars_for_line_uid",
    "set_ocr_chars_for_line_uid",
]
