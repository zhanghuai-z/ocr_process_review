"""Helpers for transient OCR line hints used by hybrid OCR routing."""
from __future__ import annotations

from app.models import Line
from app.models.ocr_text_observation import append_line_ocr_review_flag_once, line_has_ocr_review_flag


PPOCR_PAGE_LINE_HINT_FLAG = "ppocrv5_page_line_hint"


def mark_ppocr_page_line_hint(line: Line) -> None:
    append_line_ocr_review_flag_once(line, PPOCR_PAGE_LINE_HINT_FLAG)


def is_ppocr_page_line_hint(line: Line) -> bool:
    return line_has_ocr_review_flag(line, PPOCR_PAGE_LINE_HINT_FLAG)


__all__ = [
    "PPOCR_PAGE_LINE_HINT_FLAG",
    "is_ppocr_page_line_hint",
    "mark_ppocr_page_line_hint",
]
