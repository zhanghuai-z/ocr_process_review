"""Construction boundary for OCR text observations attached to lines."""
from __future__ import annotations

from collections.abc import Iterable

from .project import BBox, Line


def create_ocr_text_line(
    *,
    text: str,
    confidence: float,
    bbox: BBox,
    source_text: str | None = None,
    review_flags: Iterable[str] | None = None,
) -> Line:
    """Create the current proof-facing line object from OCR text facts."""
    normalized_text = str(text or "")
    normalized_source = str(source_text if source_text is not None else normalized_text)
    return Line(
        text=normalized_text,
        confidence=float(confidence),
        bbox=bbox,
        ocr_text=normalized_source,
        review_flags=list(review_flags or []),
    )


def line_ocr_text(line: object) -> str:
    """Return the OCR source text attached to a proof line."""
    text = str(getattr(line, "text", "") or "")
    return str(getattr(line, "ocr_text", "") or text)


def line_ocr_review_flags(line: object) -> tuple[str, ...]:
    """Return OCR/route review flags without exposing the physical field."""
    values = getattr(line, "review_flags", ()) or ()
    return tuple(str(value) for value in values if str(value))


def line_has_ocr_review_flag(line: object, flag: str) -> bool:
    return str(flag) in line_ocr_review_flags(line)


def set_line_ocr_review_flags(line: Line, flags: Iterable[str]) -> None:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in flags:
        flag = str(value or "")
        if not flag or flag in seen:
            continue
        seen.add(flag)
        normalized.append(flag)
    line.review_flags = normalized


def append_line_ocr_review_flag_once(line: Line, flag: str) -> None:
    current = list(line_ocr_review_flags(line))
    value = str(flag or "")
    if not value or value in current:
        return
    set_line_ocr_review_flags(line, [*current, value])


__all__ = [
    "append_line_ocr_review_flag_once",
    "create_ocr_text_line",
    "line_has_ocr_review_flag",
    "line_ocr_review_flags",
    "line_ocr_text",
    "set_line_ocr_review_flags",
]
