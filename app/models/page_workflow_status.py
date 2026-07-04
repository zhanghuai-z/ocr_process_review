"""Shared page workflow status groups."""
from __future__ import annotations

from .enums import PageStatus


OCR_AVAILABLE_PAGE_STATUSES = frozenset({
    PageStatus.OCR_DONE,
    PageStatus.PROOFING,
    PageStatus.PROOF_DONE,
})
