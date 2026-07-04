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
