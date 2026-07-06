"""Runtime store for OCR character observations.

Active code should go through ``ocr_character_observation`` helpers, which use
stable line uid as the current character observation boundary.
"""
from __future__ import annotations

from .project import Char


_CHARS_BY_LINE_UID: dict[str, list[Char]] = {}


def _object_uid(line: object) -> str:
    return str(getattr(line, "uid", "") or "")


def ocr_chars_for_line(line: object) -> list[Char]:
    line_uid = _object_uid(line)
    return ocr_chars_for_line_uid(line_uid)


def set_ocr_chars_for_line(line: object, chars: list[Char]) -> None:
    line_uid = _object_uid(line)
    if line_uid:
        _CHARS_BY_LINE_UID[line_uid] = chars


def ocr_chars_for_line_uid(line_uid: str) -> list[Char]:
    return _CHARS_BY_LINE_UID.get(str(line_uid or ""), [])


def set_ocr_chars_for_line_uid(line_uid: str, chars: list[Char]) -> None:
    uid = str(line_uid or "")
    if not uid:
        return
    _CHARS_BY_LINE_UID[uid] = chars


__all__ = [
    "ocr_chars_for_line_uid",
    "ocr_chars_for_line",
    "set_ocr_chars_for_line",
    "set_ocr_chars_for_line_uid",
]
