"""Typed workflow state contracts shared by controller and UI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowViewState:
    """Controller-owned view state for workflow UI consumers."""

    max_step: int
    current_step: int
    current_page_number: int
    layout_run_enabled: bool
    has_project: bool = False
    total_pages: int = 0
    total_lines: int = 0


@dataclass(frozen=True)
class WorkflowProgressState:
    """Typed progress event emitted by workflow for layout/OCR progress."""

    phase: str
    current: int = 0
    total: int = 0
    completed_pages: int = 0
    total_pages: int = 0
    message: str = ""

