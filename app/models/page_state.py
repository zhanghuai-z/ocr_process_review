"""Explicit mutation helpers for page workflow state."""
from __future__ import annotations

from app.models import Page, PageStatus
from app.models.layout_projection import page_has_layout_blocks
from app.models.ocr_observation import page_has_ocr_result


OCR_AVAILABLE_PAGE_STATUSES = {
    PageStatus.OCR_DONE,
    PageStatus.PROOFING,
    PageStatus.PROOF_DONE,
}


def page_is_layout_analyzed(page: Page) -> bool:
    return page_has_layout_blocks(page)


def page_is_ocr_done(page: Page) -> bool:
    return page.status in OCR_AVAILABLE_PAGE_STATUSES


def page_needs_ocr_rerun(page: Page) -> bool:
    return bool(page.ocr_invalidated_reason)


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


def clear_page_error_message(page: Page) -> None:
    page.error_message = ""


def mark_page_ocr_done(page: Page) -> None:
    page.status = PageStatus.OCR_DONE
    clear_page_error_message(page)
    clear_page_ocr_invalidation(page)


def mark_page_ocr_failed(page: Page, message: str | None = None) -> None:
    if message is not None:
        page.error_message = str(message)
    page.status = PageStatus.ERROR


def reconcile_page_ocr_done_from_result(page: Page) -> None:
    """Promote loaded OCR content into explicit page state."""
    if (
        page_has_ocr_result(page)
        and not page_is_ocr_done(page)
        and not page_needs_ocr_rerun(page)
        and not page.error_message
    ):
        mark_page_ocr_done(page)
