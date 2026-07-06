"""Runtime store for OCR text observations.

``Line.text``, ``Line.ocr_text`` and ``Line.review_flags`` remain as the
storage/UI projection. Active code should go through ``ocr_text_observation``
helpers so OCR source text and route review flags have one boundary.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrTextObservation:
    text: str
    ocr_text: str
    confidence: float
    review_flags: tuple[str, ...] = tuple()


_TEXT_BY_LINE_UID: dict[str, OcrTextObservation] = {}


def _line_uid(line: object) -> str:
    return str(getattr(line, "uid", "") or "")


def ocr_text_observation_for_line(
    line: object,
    projection: OcrTextObservation | None = None,
) -> OcrTextObservation:
    line_uid = _line_uid(line)
    if line_uid and line_uid in _TEXT_BY_LINE_UID:
        return _TEXT_BY_LINE_UID[line_uid]
    observation = projection if projection is not None else OcrTextObservation("", "", 0.0)
    set_ocr_text_observation_for_line(line, observation)
    return observation


def set_ocr_text_observation_for_line(
    line: object,
    observation: OcrTextObservation,
) -> None:
    line_uid = _line_uid(line)
    if line_uid:
        _TEXT_BY_LINE_UID[line_uid] = observation


def clear_ocr_text_observation_for_line(line: object) -> None:
    line_uid = _line_uid(line)
    if line_uid:
        _TEXT_BY_LINE_UID.pop(line_uid, None)


__all__ = [
    "OcrTextObservation",
    "clear_ocr_text_observation_for_line",
    "ocr_text_observation_for_line",
    "set_ocr_text_observation_for_line",
]
