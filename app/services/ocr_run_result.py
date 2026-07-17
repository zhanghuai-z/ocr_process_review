"""Typed OCR run result and progress contracts."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models import Page
from app.models.ocr_routing_run_audit import OcrRoutingRunAuditDraft


@dataclass
class OcrProgress:
    """OCR progress event emitted by pipeline workers."""

    current_page: int = 0
    total_pages: int = 0
    current_block: int = 0
    total_blocks: int = 0
    completed_pages: int = 0
    message: str = ""


@dataclass
class OcrRunResult:
    """Project-level OCR run result."""

    pages: list[Page] = field(default_factory=list)
    failed_blocks: list[tuple[int, int, str]] = field(default_factory=list)
    routing_audits: list[OcrRoutingRunAuditDraft] = field(default_factory=list)


@dataclass
class PageOcrRunResult:
    """Single-page OCR worker result used by parallel page OCR."""

    page_idx: int
    page: Page
    total_blocks: int = 0
    failed_blocks: list[tuple[int, int, str]] = field(default_factory=list)
    completion_message: str = ""
    routing_audits: list[OcrRoutingRunAuditDraft] = field(default_factory=list)


__all__ = [
    "OcrProgress",
    "OcrRunResult",
    "PageOcrRunResult",
]
