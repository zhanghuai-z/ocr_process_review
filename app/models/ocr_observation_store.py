"""Runtime store for OCR line observations.

This store keeps active OCR line observations outside layout block objects so
OCR facts are keyed by stable block uid instead of runtime projections.
"""
from __future__ import annotations

from .project import Line


_LINES_BY_BLOCK_UID: dict[str, list[Line]] = {}


def _object_uid(block: object) -> str:
    return str(getattr(block, "uid", "") or "")


def ocr_lines_for_block(block: object) -> list[Line]:
    """Return the current OCR line observations for ``block``."""
    block_uid = _object_uid(block)
    return ocr_lines_for_block_uid(block_uid)


def ocr_lines_for_block_uid(block_uid: str) -> list[Line]:
    return _LINES_BY_BLOCK_UID.get(str(block_uid or ""), [])


def set_ocr_lines_for_block_uid(block_uid: str, lines: list[Line]) -> None:
    uid = str(block_uid or "")
    if not uid:
        return
    _LINES_BY_BLOCK_UID[uid] = lines


__all__ = [
    "ocr_lines_for_block_uid",
    "ocr_lines_for_block",
    "set_ocr_lines_for_block_uid",
]
