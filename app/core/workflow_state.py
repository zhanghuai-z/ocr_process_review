"""Typed workflow contracts over the project-session repositories."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.ocr_currentness import (
    current_ocr_failure,
    current_ocr_observation,
    has_active_ocr_pointer,
)
from app.models.project_session import PageRecord, ProjectSession, RecordNotFoundError


# Step indexes are part of the stacked-workspace UI contract.
STEP_IMPORT = 0
STEP_LAYOUT = 1
STEP_OCR = 2
STEP_HPROOF = 3
STEP_VPROOF = 4


@dataclass(frozen=True, slots=True)
class WorkflowViewState:
    """Controller-owned state consumed by the UI shell."""

    max_step: int
    current_step: int
    current_page_uid: str
    layout_run_enabled: bool
    has_project: bool = False
    total_pages: int = 0
    total_lines: int = 0


@dataclass(frozen=True, slots=True)
class WorkflowProgressState:
    """Immutable progress event for a layout or OCR service run."""

    phase: str
    current: int = 0
    total: int = 0
    completed_pages: int = 0
    total_pages: int = 0
    message: str = ""
    page_uid: str = ""


@dataclass(frozen=True, slots=True)
class PageGateInfo:
    """OCR entry state for one stable page UID."""

    page_state: str
    is_pending: bool
    reason_code: str
    reason_text: str
    action_key: str
    action_label: str
    action_enabled: bool


def _require_session(session: ProjectSession) -> ProjectSession:
    if not isinstance(session, ProjectSession):
        raise TypeError("workflow state requires ProjectSession")
    return session


def _page(session: ProjectSession, page_uid: str) -> PageRecord:
    return _require_session(session).page_repository.get(page_uid)


def _has_layout(session: ProjectSession, page_uid: str) -> bool:
    try:
        session.layout_repository.get(page_uid)
    except RecordNotFoundError:
        return False
    return True


def _has_ocr(session: ProjectSession, page_uid: str) -> bool:
    return current_ocr_observation(session, page_uid) is not None


def active_line_count(session: ProjectSession) -> int:
    """Count lines in the active OCR batches without projecting runtime objects."""
    _require_session(session)
    total = 0
    for page in session.page_repository.all():
        observation = current_ocr_observation(session, page.uid)
        if observation is not None:
            total += len(observation.batch.line_uids)
    return total


def compute_max_step(session: ProjectSession | None) -> int:
    """Return the maximum globally enterable workflow step."""
    if session is None:
        return STEP_IMPORT
    _require_session(session)
    pages = session.page_repository.all()
    if not pages:
        return STEP_IMPORT
    if any(_has_ocr(session, page.uid) for page in pages):
        return STEP_VPROOF
    if any(_has_layout(session, page.uid) for page in pages):
        return STEP_OCR
    return STEP_LAYOUT


def page_gate_info(session: ProjectSession, page_uid: str) -> PageGateInfo:
    """Return the OCR gate for one page, using only adopted session facts."""
    page = _page(session, page_uid)
    if page.error:
        return PageGateInfo(
            page_state="error",
            is_pending=False,
            reason_code="page_error",
            reason_text=f"当前页处理失败：{page.error}",
            action_key="enter_ocr",
            action_label="提交并进入 OCR",
            action_enabled=False,
        )
    if not _has_layout(session, page.uid):
        return PageGateInfo(
            page_state="layout_pending",
            is_pending=True,
            reason_code="layout_not_done",
            reason_text="请先完成版面分析",
            action_key="enter_ocr",
            action_label="提交并进入 OCR",
            action_enabled=False,
        )
    failure = current_ocr_failure(session, page.uid)
    if failure is not None:
        return PageGateInfo(
            page_state="ocr_error",
            is_pending=True,
            reason_code="ocr_failed",
            reason_text=f"当前页 OCR 失败：{failure.message}",
            action_key="rerun_ocr",
            action_label="重新进入 OCR",
            action_enabled=True,
        )
    if _has_ocr(session, page.uid):
        return PageGateInfo(
            page_state="ocr_complete",
            is_pending=False,
            reason_code="ocr_complete",
            reason_text="当前页 OCR 已完成",
            action_key="enter_ocr",
            action_label="提交并进入 OCR",
            action_enabled=False,
        )
    if has_active_ocr_pointer(session, page.uid):
        return PageGateInfo(
            page_state="ocr_invalidated",
            is_pending=True,
            reason_code="invalidated_after_input_change",
            reason_text="当前页图像或版面已变更，需要重新进入 OCR",
            action_key="rerun_ocr",
            action_label="重新进入 OCR",
            action_enabled=True,
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


def pending_ocr_page_uids(session: ProjectSession | None) -> tuple[str, ...]:
    """Return pending page UIDs in adopted page order."""
    if session is None:
        return ()
    _require_session(session)
    pending: list[str] = []
    for page in session.page_repository.all():
        gate = page_gate_info(session, page.uid)
        if gate.is_pending and gate.action_enabled:
            pending.append(page.uid)
    return tuple(pending)


def pending_ocr_pages(session: ProjectSession | None) -> tuple[PageRecord, ...]:
    """Return immutable page records pending OCR for UI projections."""
    if session is None:
        return ()
    _require_session(session)
    pending = set(pending_ocr_page_uids(session))
    return tuple(page for page in session.page_repository.all() if page.uid in pending)


__all__ = [
    "PageGateInfo",
    "STEP_HPROOF",
    "STEP_IMPORT",
    "STEP_LAYOUT",
    "STEP_OCR",
    "STEP_VPROOF",
    "WorkflowProgressState",
    "WorkflowViewState",
    "active_line_count",
    "compute_max_step",
    "page_gate_info",
    "pending_ocr_page_uids",
    "pending_ocr_pages",
]
