"""Accessors for external OCR artifact evidence.

Artifacts are evidence from OCR/layout engines. Layout records stored here are
normalized to the current page image coordinate space before application code
consumes them. Application logic should depend on layout snapshots first; when
external evidence is needed, access it through this module instead of treating
Page as the owner of vendor JSON.
"""
from __future__ import annotations

from typing import Any

from app.models import Page, RawOcrArtifact


def raw_layout_records(page: Page) -> list[dict[str, Any]]:
    artifact = page.raw_layout_artifact
    if artifact is None:
        return []
    return artifact.records


def set_paddle_raw_layout_records(
    page: Page,
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
) -> RawOcrArtifact:
    artifact = RawOcrArtifact.from_paddle_layout_records(
        records,
        page_uid=page.uid,
        run_id=run_id,
    )
    page.raw_layout_artifact = artifact
    return artifact


def _raw_payload_from_origin(page: Page | None, block: object) -> dict[str, Any]:
    if page is None:
        return {}
    origin = getattr(block, "origin", None)
    try:
        raw_index = int(getattr(origin, "raw_index", -1))
    except (TypeError, ValueError):
        return {}
    records = raw_layout_records(page)
    if raw_index < 0 or raw_index >= len(records):
        return {}
    record = records[raw_index]
    return dict(record) if isinstance(record, dict) else {}


def raw_block_payload(block: object, page: Page | None = None) -> dict[str, Any]:
    """Return a copy of a block's vendor payload.

    Raw payload is external evidence. Callers that still need it should read a
    copy through this module instead of treating ``Block`` as a vendor JSON
    container.
    """
    return _raw_payload_from_origin(page, block)


def raw_block_text_values(
    block: object,
    keys: tuple[str, ...],
    page: Page | None = None,
) -> list[str]:
    payload = raw_block_payload(block, page)
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                values.append(text)
    return values
