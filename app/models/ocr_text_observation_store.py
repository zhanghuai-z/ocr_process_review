"""Runtime store for OCR text observations.

``Line.text``, ``Line.ocr_text`` and ``Line.review_flags`` remain as the
storage/UI projection. Active code should go through ``ocr_text_observation``
helpers so OCR source text and route review flags have one boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref


@dataclass(frozen=True)
class OcrTextObservation:
    text: str
    ocr_text: str
    confidence: float
    review_flags: tuple[str, ...] = tuple()


_TextEntry = tuple[weakref.ReferenceType[object], OcrTextObservation]
_TEXT_BY_LINE_OBJECT: dict[int, _TextEntry] = {}


def ocr_text_observation_for_line(
    line: object,
    projection: OcrTextObservation | None = None,
) -> OcrTextObservation:
    line_id = id(line)
    entry = _TEXT_BY_LINE_OBJECT.get(line_id)
    if entry is not None:
        line_ref, observation = entry
        if line_ref() is line:
            if projection is not None and projection != observation:
                set_ocr_text_observation_for_line(line, projection)
                return projection
            return observation
        _TEXT_BY_LINE_OBJECT.pop(line_id, None)
    observation = projection if projection is not None else OcrTextObservation("", "", 0.0)
    set_ocr_text_observation_for_line(line, observation)
    return observation


def set_ocr_text_observation_for_line(
    line: object,
    observation: OcrTextObservation,
) -> None:
    line_id = id(line)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _TEXT_BY_LINE_OBJECT.get(line_id)
        if entry is not None and entry[0] is _ref:
            _TEXT_BY_LINE_OBJECT.pop(line_id, None)

    _TEXT_BY_LINE_OBJECT[line_id] = (weakref.ref(line, _cleanup), observation)


def clear_ocr_text_observation_for_line(line: object) -> None:
    _TEXT_BY_LINE_OBJECT.pop(id(line), None)


__all__ = [
    "OcrTextObservation",
    "clear_ocr_text_observation_for_line",
    "ocr_text_observation_for_line",
    "set_ocr_text_observation_for_line",
]
