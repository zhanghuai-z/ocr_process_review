"""Runtime store for OCR line observations.

``Block.lines`` still exists as the runtime projection consumed by current UI
and storage code. This store keeps the active OCR line observation list outside
the layout block object so OCR facts can move without adding another field to
``Block``.
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


def ocr_lines_for_block(block: object, projection: list[Line] | None = None) -> list[Line]:
    """Return the current OCR line observations for ``block``."""
    block_id = id(block)
    block_uid = _object_uid(block)
    entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
    if entry is not None:
        block_ref, lines = entry
        if block_ref() is block:
            if projection is not None and projection is not lines:
                if block_uid and block_uid in _LINES_BY_BLOCK_UID:
                    uid_lines = _LINES_BY_BLOCK_UID[block_uid]
                    _set_ocr_lines_for_block_object(block, uid_lines)
                    return uid_lines
                set_ocr_lines_for_block(block, projection)
                return projection
            return lines
        _LINES_BY_BLOCK_OBJECT.pop(block_id, None)
    if block_uid and block_uid in _LINES_BY_BLOCK_UID:
        lines = _LINES_BY_BLOCK_UID[block_uid]
        _set_ocr_lines_for_block_object(block, lines)
        return lines
    lines = projection if projection is not None else []
    set_ocr_lines_for_block(block, lines)
    return lines


def _set_ocr_lines_for_block_object(block: object, lines: list[Line]) -> None:
    block_id = id(block)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
        if entry is not None and entry[0] is _ref:
            _LINES_BY_BLOCK_OBJECT.pop(block_id, None)

    _LINES_BY_BLOCK_OBJECT[block_id] = (weakref.ref(block, _cleanup), lines)


def set_ocr_lines_for_block(block: object, lines: list[Line]) -> None:
    block_uid = _object_uid(block)
    if block_uid:
        _LINES_BY_BLOCK_UID[block_uid] = lines
        _drop_object_entries_for_uid(block_uid, keep=block)
    _set_ocr_lines_for_block_object(block, lines)


def ocr_lines_for_block_uid(block_uid: str) -> list[Line]:
    return _LINES_BY_BLOCK_UID.get(str(block_uid or ""), [])


def set_ocr_lines_for_block_uid(block_uid: str, lines: list[Line]) -> None:
    uid = str(block_uid or "")
    if not uid:
        return
    _LINES_BY_BLOCK_UID[uid] = lines
    _drop_object_entries_for_uid(uid)


def clear_ocr_lines_for_block(block: object) -> None:
    _LINES_BY_BLOCK_OBJECT.pop(id(block), None)
    block_uid = _object_uid(block)
    if block_uid:
        _LINES_BY_BLOCK_UID.pop(block_uid, None)


__all__ = [
    "clear_ocr_lines_for_block",
    "ocr_lines_for_block_uid",
    "ocr_lines_for_block",
    "set_ocr_lines_for_block",
    "set_ocr_lines_for_block_uid",
]
