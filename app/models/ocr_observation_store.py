"""Runtime store for OCR line observations.

This store keeps active OCR line observations outside layout block objects so
OCR facts are keyed by stable block uid instead of runtime projections.
"""
from __future__ import annotations

from .project import Line


_LINES_BY_BLOCK_UID: dict[str, list[Line]] = {}


def ocr_lines_for_block_uid(block_uid: str) -> list[Line]:
    return _LINES_BY_BLOCK_UID.get(str(block_uid or ""), [])


def set_ocr_lines_for_block_uid(block_uid: str, lines: list[Line]) -> None:
    uid = str(block_uid or "")
    if not uid:
        return
    _LINES_BY_BLOCK_UID[uid] = lines


__all__ = [
    "ocr_lines_for_block_uid",
    "set_ocr_lines_for_block_uid",
]
