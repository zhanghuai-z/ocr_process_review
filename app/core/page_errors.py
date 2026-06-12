"""Page-level error classification helpers."""
from __future__ import annotations


def is_ocr_error_message(message: object) -> bool:
    """Return whether a page error was produced by the OCR stage."""
    return str(message or "").startswith("OCR ")
