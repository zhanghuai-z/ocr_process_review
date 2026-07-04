"""Runtime store for OCR line observations.

``Block.lines`` still exists as a compatibility projection for legacy UI and
storage code. This store keeps the active OCR line observation list outside the
layout block object so OCR facts can move without adding another field to
``Block``.
"""
from __future__ import annotations

import weakref

from .project import Line


_LineEntry = tuple[weakref.ReferenceType[object], list[Line]]
_LINES_BY_BLOCK_OBJECT: dict[int, _LineEntry] = {}


def ocr_lines_for_block(block: object, projection: list[Line] | None = None) -> list[Line]:
    """Return the current OCR line observations for ``block``."""
    block_id = id(block)
    entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
    if entry is not None:
        block_ref, lines = entry
        if block_ref() is block:
            if projection is not None and projection is not lines:
                set_ocr_lines_for_block(block, projection)
                return projection
            return lines
        _LINES_BY_BLOCK_OBJECT.pop(block_id, None)
    lines = projection if projection is not None else []
    set_ocr_lines_for_block(block, lines)
    return lines


def set_ocr_lines_for_block(block: object, lines: list[Line]) -> None:
    block_id = id(block)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _LINES_BY_BLOCK_OBJECT.get(block_id)
        if entry is not None and entry[0] is _ref:
            _LINES_BY_BLOCK_OBJECT.pop(block_id, None)

    _LINES_BY_BLOCK_OBJECT[block_id] = (weakref.ref(block, _cleanup), lines)


def clear_ocr_lines_for_block(block: object) -> None:
    _LINES_BY_BLOCK_OBJECT.pop(id(block), None)


__all__ = [
    "clear_ocr_lines_for_block",
    "ocr_lines_for_block",
    "set_ocr_lines_for_block",
]
