"""Helpers for transient OCR line hints used by hybrid OCR routing."""
from __future__ import annotations

from app.models import Line


PPOCR_PAGE_LINE_HINT_FLAG = "ppocrv5_page_line_hint"


def mark_ppocr_page_line_hint(line: Line) -> None:
    if PPOCR_PAGE_LINE_HINT_FLAG not in line.review_flags:
        line.review_flags.append(PPOCR_PAGE_LINE_HINT_FLAG)


def is_ppocr_page_line_hint(line: Line) -> bool:
    return PPOCR_PAGE_LINE_HINT_FLAG in getattr(line, "review_flags", [])


__all__ = [
    "PPOCR_PAGE_LINE_HINT_FLAG",
    "is_ppocr_page_line_hint",
    "mark_ppocr_page_line_hint",
]
