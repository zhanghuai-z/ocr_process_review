"""Accessors for external OCR raw artifacts.

Raw artifacts are evidence from OCR/layout engines. Application logic should
depend on current layout blocks first; when raw evidence is needed, access it
through this module instead of treating Page as the owner of vendor JSON.
"""
from __future__ import annotations

from typing import Any

from app.models import Page, RawOcrArtifact


def raw_layout_records(page: Page) -> list[dict[str, Any]]:
    artifact = page.raw_layout_artifact
    if artifact is None:
        return []
    return artifact.records


def set_paddle_raw_layout_records(page: Page, records: list[dict[str, Any]]) -> RawOcrArtifact:
    artifact = RawOcrArtifact.from_paddle_layout_records(records, page_uid=page.uid)
    page.raw_layout_artifact = artifact
    return artifact


def raw_block_payload(block: object) -> dict[str, Any]:
    """Return a copy of a block's vendor payload.

    Raw payload is external evidence. Callers that still need it should read a
    copy through this module instead of treating ``Block`` as a mutable vendor
    JSON container.
    """
    payload = getattr(block, "raw_payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def raw_block_text_values(block: object, keys: tuple[str, ...]) -> list[str]:
    payload = raw_block_payload(block)
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                values.append(text)
    return values
