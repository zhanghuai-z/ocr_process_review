"""Explicit mutation helpers for page workflow state."""
from __future__ import annotations

from app.models import Page, PageStatus
from app.models.ocr_observation import page_has_ocr_result


def mark_page_imported(page: Page) -> None:
    page.status = PageStatus.IMPORTED
    page.error_message = ""
    page.ocr_invalidated_reason = ""


def mark_page_layout_done(page: Page) -> None:
    page.status = PageStatus.LAYOUT_DONE
    page.error_message = ""


def mark_page_layout_failed(page: Page, message: str | None = None) -> None:
    if message is not None:
        page.error_message = str(message)
    page.status = PageStatus.ERROR


def invalidate_page_ocr(page: Page, reason: str) -> None:
    page.ocr_invalidated_reason = str(reason or "layout_changed")


def clear_page_ocr_invalidation(page: Page) -> None:
    page.ocr_invalidated_reason = ""


def mark_page_ocr_done(page: Page) -> None:
    page.status = PageStatus.OCR_DONE
    page.error_message = ""
    clear_page_ocr_invalidation(page)


def mark_page_ocr_failed(page: Page, message: str | None = None) -> None:
    if message is not None:
        page.error_message = str(message)
    page.status = PageStatus.ERROR


def reconcile_page_ocr_done_from_result(page: Page) -> None:
    """Promote loaded OCR content into explicit page state."""
    if (
        page_has_ocr_result(page)
        and not page.is_ocr_done
        and not page.needs_ocr_rerun
        and not page.error_message
    ):
        mark_page_ocr_done(page)
