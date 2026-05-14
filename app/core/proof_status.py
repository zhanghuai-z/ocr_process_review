"""Shared proof confidence and status rules."""
from __future__ import annotations

from collections.abc import Iterable

from app.models import Line, ProofStatus

AUTO_FLAG_THRESHOLD = 0.80


def normalize_confidence(score) -> float:
    """Normalize OCR confidence values to the 0..1 range."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0

    if 1.0 < value <= 100.0:
        value = value / 100.0
    return max(0.0, min(value, 1.0))


def should_auto_flag(confidence, review_flags: Iterable | None = None) -> bool:
    if review_flags:
        return True
    return normalize_confidence(confidence) < AUTO_FLAG_THRESHOLD


def proof_status_for(confidence, review_flags: Iterable | None = None) -> ProofStatus:
    return (
        ProofStatus.AUTO_FLAGGED
        if should_auto_flag(confidence, review_flags)
        else ProofStatus.UNCHECKED
    )


def apply_auto_flag(line: Line, review_flags: Iterable | None = None) -> None:
    """Promote only unchecked lines to auto-flagged when shared rules require it."""
    if line.proof_status != ProofStatus.UNCHECKED:
        return
    if should_auto_flag(line.confidence, review_flags):
        line.proof_status = ProofStatus.AUTO_FLAGGED
