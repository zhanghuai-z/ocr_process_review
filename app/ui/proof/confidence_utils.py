"""Proof UI confidence sourcing and normalization helpers."""
from __future__ import annotations

from typing import Optional

from app.models import Line


def normalize_confidence(raw) -> Optional[float]:
    """Return confidence in 0..1, or None when the value is absent/default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    if value > 1.0:
        return None
    return max(0.0, value)


def line_confidence(line: Line) -> Optional[float]:
    """Prefer real char confidence, then fall back to a non-zero line score."""
    char_scores = [
        score
        for ch in (line.chars or [])
        if (score := normalize_confidence(getattr(ch, "confidence", None))) is not None
    ]
    if char_scores:
        return sum(char_scores) / len(char_scores)
    return normalize_confidence(getattr(line, "confidence", None))


def char_confidence(line: Line, idx: int) -> Optional[float]:
    chars = line.chars or []
    if 0 <= idx < len(chars):
        score = normalize_confidence(getattr(chars[idx], "confidence", None))
        if score is not None:
            return score
    return line_confidence(line)
