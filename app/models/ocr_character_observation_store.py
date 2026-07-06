"""Runtime store for OCR character observations.

Active code should go through ``ocr_character_observation`` helpers, which use
stable line uid as the current character observation boundary.
"""
from __future__ import annotations

import weakref

from .project import Char


_CharEntry = tuple[weakref.ReferenceType[object], list[Char]]
_CHARS_BY_LINE_OBJECT: dict[int, _CharEntry] = {}
_CHARS_BY_LINE_UID: dict[str, list[Char]] = {}


def _object_uid(line: object) -> str:
    return str(getattr(line, "uid", "") or "")


def _drop_object_entries_for_uid(line_uid: str, *, keep: object | None = None) -> None:
    if not line_uid:
        return
    keep_id = id(keep) if keep is not None else None
    for line_id, entry in list(_CHARS_BY_LINE_OBJECT.items()):
        if keep_id is not None and line_id == keep_id:
            continue
        line = entry[0]()
        if line is None or _object_uid(line) == line_uid:
            _CHARS_BY_LINE_OBJECT.pop(line_id, None)


def ocr_chars_for_line(line: object) -> list[Char]:
    line_id = id(line)
    line_uid = _object_uid(line)
    entry = _CHARS_BY_LINE_OBJECT.get(line_id)
    if entry is not None:
        line_ref, chars = entry
        if line_ref() is line:
            return chars
        _CHARS_BY_LINE_OBJECT.pop(line_id, None)
    if line_uid and line_uid in _CHARS_BY_LINE_UID:
        chars = _CHARS_BY_LINE_UID[line_uid]
        _set_ocr_chars_for_line_object(line, chars)
        return chars
    return []


def _set_ocr_chars_for_line_object(line: object, chars: list[Char]) -> None:
    line_id = id(line)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _CHARS_BY_LINE_OBJECT.get(line_id)
        if entry is not None and entry[0] is _ref:
            _CHARS_BY_LINE_OBJECT.pop(line_id, None)

    _CHARS_BY_LINE_OBJECT[line_id] = (weakref.ref(line, _cleanup), chars)


def set_ocr_chars_for_line(line: object, chars: list[Char]) -> None:
    line_uid = _object_uid(line)
    if line_uid:
        _CHARS_BY_LINE_UID[line_uid] = chars
    _set_ocr_chars_for_line_object(line, chars)


def ocr_chars_for_line_uid(line_uid: str) -> list[Char]:
    return _CHARS_BY_LINE_UID.get(str(line_uid or ""), [])


def set_ocr_chars_for_line_uid(line_uid: str, chars: list[Char]) -> None:
    uid = str(line_uid or "")
    if not uid:
        return
    _CHARS_BY_LINE_UID[uid] = chars
    _drop_object_entries_for_uid(uid)


def clear_ocr_chars_for_line(line: object) -> None:
    _CHARS_BY_LINE_OBJECT.pop(id(line), None)
    line_uid = _object_uid(line)
    if line_uid:
        _CHARS_BY_LINE_UID.pop(line_uid, None)


__all__ = [
    "clear_ocr_chars_for_line",
    "ocr_chars_for_line_uid",
    "ocr_chars_for_line",
    "set_ocr_chars_for_line",
    "set_ocr_chars_for_line_uid",
]
