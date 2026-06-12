"""Typed workflow state contracts shared by controller and UI."""
from __future__ import annotations

from dataclasses import dataclass

from app.models import OcrProject, Page

# 步骤索引（与 stacked widget 顺序一致）
STEP_IMPORT = 0
STEP_LAYOUT = 1
STEP_OCR = 2
STEP_HPROOF = 3
STEP_VPROOF = 4


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


@dataclass(frozen=True)
class PageGateInfo:
    """Single-page OCR gate state for Hanwang layout-confirm workflow."""

    page_state: str
    is_pending: bool
    reason_code: str
    reason_text: str
    action_key: str
    action_label: str
    action_enabled: bool


def compute_max_step(project: OcrProject | None) -> int:
    """Return the maximum globally enterable workflow step."""
    if project is None or not project.pages:
        return STEP_IMPORT

    has_blocks = any(page.is_analyzed for page in project.pages)
    if project.has_any_ocr_done_page:
        return STEP_VPROOF
    if has_blocks:
        return STEP_OCR
    return STEP_LAYOUT


def page_gate_info(page: Page) -> PageGateInfo:
    """Return the OCR entry gate for one page."""
    if not page.is_analyzed:
        return PageGateInfo(
            page_state="layout_pending",
            is_pending=True,
            reason_code="layout_not_done",
            reason_text="请先完成版面分析",
            action_key="enter_ocr",
            action_label="提交并进入 OCR",
            action_enabled=False,
        )
    if page.needs_ocr_rerun:
        return PageGateInfo(
            page_state="ocr_invalidated",
            is_pending=True,
            reason_code="invalidated_after_edit",
            reason_text="当前页版面已变更，需要重新进入 OCR",
            action_key="rerun_ocr",
            action_label="重新进入 OCR",
            action_enabled=True,
        )
    if page.is_ocr_done:
        return PageGateInfo(
            page_state="ocr_complete",
            is_pending=False,
            reason_code="ocr_complete",
            reason_text="当前页 OCR 已完成",
            action_key="enter_ocr",
            action_label="提交并进入 OCR",
            action_enabled=False,
        )
    return PageGateInfo(
        page_state="ocr_ready",
        is_pending=True,
        reason_code="ready_for_ocr",
        reason_text="当前页版面已确认，可进入 OCR",
        action_key="enter_ocr",
        action_label="提交并进入 OCR",
        action_enabled=True,
    )


def pending_ocr_pages(project: OcrProject | None) -> list[Page]:
    if project is None:
        return []
    return [page for page in project.pages if page_gate_info(page).is_pending]
