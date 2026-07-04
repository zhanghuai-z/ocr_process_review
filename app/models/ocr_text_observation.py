"""Construction boundary for OCR text observations attached to lines."""
from __future__ import annotations

from collections.abc import Iterable

from .project import BBox, Line
from .ocr_text_observation_store import (
    OcrTextObservation,
    ocr_text_observation_for_line,
    set_ocr_text_observation_for_line,
)


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
    line = Line(
        text=normalized_text,
        confidence=float(confidence),
        bbox=bbox,
        ocr_text=normalized_source,
        review_flags=list(review_flags or []),
    )
    set_line_ocr_text_observation(
        line,
        OcrTextObservation(
            text=normalized_text,
            ocr_text=normalized_source,
            confidence=float(confidence),
            review_flags=_normalized_review_flags(review_flags or ()),
        ),
    )
    return line


def line_ocr_text_observation(line: object) -> OcrTextObservation:
    return ocr_text_observation_for_line(line, _projection_observation(line))


def line_ocr_text(line: object) -> str:
    """Return the OCR source text attached to a proof line."""
    observation = line_ocr_text_observation(line)
    return observation.ocr_text or observation.text


def line_ocr_review_flags(line: object) -> tuple[str, ...]:
    """Return OCR/route review flags without exposing the physical field."""
    return line_ocr_text_observation(line).review_flags


def line_has_ocr_review_flag(line: object, flag: str) -> bool:
    return str(flag) in line_ocr_review_flags(line)


def set_line_ocr_review_flags(line: Line, flags: Iterable[str]) -> None:
    observation = line_ocr_text_observation(line)
    set_line_ocr_text_observation(
        line,
        OcrTextObservation(
            text=observation.text,
            ocr_text=observation.ocr_text,
            confidence=observation.confidence,
            review_flags=_normalized_review_flags(flags),
        ),
    )


def append_line_ocr_review_flag_once(line: Line, flag: str) -> None:
    current = list(line_ocr_review_flags(line))
    value = str(flag or "")
    if not value or value in current:
        return
    set_line_ocr_review_flags(line, [*current, value])


def set_line_ocr_text_observation(line: Line, observation: OcrTextObservation) -> None:
    normalized = OcrTextObservation(
        text=str(observation.text or ""),
        ocr_text=str(observation.ocr_text or observation.text or ""),
        confidence=float(observation.confidence or 0.0),
        review_flags=_normalized_review_flags(observation.review_flags),
    )
    line.text = normalized.text
    line.ocr_text = normalized.ocr_text
    line.confidence = normalized.confidence
    line.review_flags = list(normalized.review_flags)
    set_ocr_text_observation_for_line(line, normalized)


def _projection_observation(line: object) -> OcrTextObservation:
    text = str(getattr(line, "text", "") or "")
    ocr_text = str(getattr(line, "ocr_text", "") or text)
    return OcrTextObservation(
        text=text or ocr_text,
        ocr_text=ocr_text,
        confidence=float(getattr(line, "confidence", 0.0) or 0.0),
        review_flags=_normalized_review_flags(getattr(line, "review_flags", ()) or ()),
    )


def _normalized_review_flags(flags: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in flags:
        flag = str(value or "")
        if not flag or flag in seen:
            continue
        seen.add(flag)
        normalized.append(flag)
    return tuple(normalized)


__all__ = [
    "append_line_ocr_review_flag_once",
    "create_ocr_text_line",
    "line_has_ocr_review_flag",
    "line_ocr_review_flags",
    "line_ocr_text",
    "line_ocr_text_observation",
    "set_line_ocr_review_flags",
    "set_line_ocr_text_observation",
]
