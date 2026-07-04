"""Runtime store for OCR character observations.

``Line.chars`` remains as a compatibility projection. Active code should go
through ``ocr_character_observation`` helpers, which use this store as the
current character observation boundary and keep the old field in sync.
"""
from __future__ import annotations

import weakref

from .project import Char


_CharEntry = tuple[weakref.ReferenceType[object], list[Char]]
_CHARS_BY_LINE_OBJECT: dict[int, _CharEntry] = {}


def ocr_chars_for_line(line: object, projection: list[Char] | None = None) -> list[Char]:
    line_id = id(line)
    entry = _CHARS_BY_LINE_OBJECT.get(line_id)
    if entry is not None:
        line_ref, chars = entry
        if line_ref() is line:
            if projection is not None and projection is not chars:
                set_ocr_chars_for_line(line, projection)
                return projection
            return chars
        _CHARS_BY_LINE_OBJECT.pop(line_id, None)
    chars = projection if projection is not None else []
    set_ocr_chars_for_line(line, chars)
    return chars


def set_ocr_chars_for_line(line: object, chars: list[Char]) -> None:
    line_id = id(line)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _CHARS_BY_LINE_OBJECT.get(line_id)
        if entry is not None and entry[0] is _ref:
            _CHARS_BY_LINE_OBJECT.pop(line_id, None)

    _CHARS_BY_LINE_OBJECT[line_id] = (weakref.ref(line, _cleanup), chars)


def clear_ocr_chars_for_line(line: object) -> None:
    _CHARS_BY_LINE_OBJECT.pop(id(line), None)


__all__ = [
    "clear_ocr_chars_for_line",
    "ocr_chars_for_line",
    "set_ocr_chars_for_line",
]
