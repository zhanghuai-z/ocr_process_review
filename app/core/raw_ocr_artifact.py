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
