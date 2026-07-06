"""Runtime store for OCR line observations.

This store keeps active OCR line observations outside layout block objects so
OCR facts are keyed by stable block uid instead of runtime projections.
"""
from __future__ import annotations

import weakref

from .project import Line


_LineEntry = tuple[weakref.ReferenceType[object], list[Line]]
_LINES_BY_BLOCK_OBJECT: dict[int, _LineEntry] = {}
_LINES_BY_BLOCK_UID: dict[str, list[Line]] = {}


def _object_uid(block: object) -> str:
    return str(getattr(block, "uid", "") or "")


def _drop_object_entries_for_uid(block_uid: str, *, keep: object | None = None) -> None:
    if not block_uid:
        return
    keep_id = id(keep) if keep is not None else None
    for block_id, entry in list(_LINES_BY_BLOCK_OBJECT.items()):
        if keep_id is not None and block_id == keep_id:
            continue
        block = entry[0]()
        if block is None or _object_uid(block) == block_uid:
            _LINES_BY_BLOCK_OBJECT.pop(block_id, None)


def ocr_lines_for_block(block: object) -> list[Line]:
    """Return the current OCR line observations for ``block``."""
    block_id = id(block)
    block_uid = _object_uid(block)
    entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
    if entry is not None:
        block_ref, lines = entry
        if block_ref() is block:
            return lines
        _LINES_BY_BLOCK_OBJECT.pop(block_id, None)
    if block_uid and block_uid in _LINES_BY_BLOCK_UID:
        lines = _LINES_BY_BLOCK_UID[block_uid]
        _set_ocr_lines_for_block_object(block, lines)
        return lines
    return []


def _set_ocr_lines_for_block_object(block: object, lines: list[Line]) -> None:
    block_id = id(block)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
        if entry is not None and entry[0] is _ref:
            _LINES_BY_BLOCK_OBJECT.pop(block_id, None)

    _LINES_BY_BLOCK_OBJECT[block_id] = (weakref.ref(block, _cleanup), lines)


def ocr_lines_for_block_uid(block_uid: str) -> list[Line]:
    return _LINES_BY_BLOCK_UID.get(str(block_uid or ""), [])


def set_ocr_lines_for_block_uid(block_uid: str, lines: list[Line]) -> None:
    uid = str(block_uid or "")
    if not uid:
        return
    _LINES_BY_BLOCK_UID[uid] = lines
    _drop_object_entries_for_uid(uid)


__all__ = [
    "ocr_lines_for_block_uid",
    "ocr_lines_for_block",
    "set_ocr_lines_for_block_uid",
]
