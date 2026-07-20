"""Evidence-based verdicts for OCR atoms shown by the proof UI."""
from __future__ import annotations

from dataclasses import dataclass


COLOR_LIKELY_OK = "#2e7d32"
COLOR_UNVERIFIED = "#222222"
COLOR_SUSPECT = "#e8801f"
COLOR_ERROR = "#c62828"

SEVERITY_LIKELY_OK = "likely_ok"
SEVERITY_UNVERIFIED = "unverified"
SEVERITY_SUSPECT = "suspect"
SEVERITY_ERROR = "error"


@dataclass(frozen=True, slots=True)
class CharVerdict:
    """One display verdict and the evidence used to derive it."""

    color: str
    severity: str
    evidence: str
    user_modified: bool = False


def classify_char(
    *,
    confidence: float | None,
    text_char: str,
    ocr_char: str | None,
) -> CharVerdict:
    """Classify a proof character using only the OCR score and source text."""

    user_modified = ocr_char is not None and text_char != ocr_char
    if confidence is None:
        return CharVerdict(
            COLOR_UNVERIFIED,
            SEVERITY_UNVERIFIED,
            "OCR confidence is unavailable",
            user_modified,
        )
    if confidence < 0.50:
        return CharVerdict(
            COLOR_ERROR,
            SEVERITY_ERROR,
            f"OCR confidence {confidence:.2f} is below 0.50",
            user_modified,
        )
    if confidence < 0.80:
        return CharVerdict(
            COLOR_SUSPECT,
            SEVERITY_SUSPECT,
            f"OCR confidence {confidence:.2f} is below 0.80",
            user_modified,
        )
    if confidence < 0.95:
        return CharVerdict(
            COLOR_UNVERIFIED,
            SEVERITY_UNVERIFIED,
            f"OCR confidence {confidence:.2f}; no proof of human verification",
            user_modified,
        )
    return CharVerdict(
        COLOR_LIKELY_OK,
        SEVERITY_LIKELY_OK,
        f"OCR confidence {confidence:.2f} is at least 0.95",
        user_modified,
    )


__all__ = [
    "COLOR_ERROR",
    "COLOR_LIKELY_OK",
    "COLOR_SUSPECT",
    "COLOR_UNVERIFIED",
    "CharVerdict",
    "SEVERITY_ERROR",
    "SEVERITY_LIKELY_OK",
    "SEVERITY_SUSPECT",
    "SEVERITY_UNVERIFIED",
    "classify_char",
]
