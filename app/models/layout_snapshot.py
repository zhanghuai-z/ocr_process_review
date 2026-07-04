"""Application layout snapshot contract.

``LayoutSnapshot`` is the adopted layout state for a page. It is separate from
the current ``Page.blocks`` projection so importers, editors, OCR, and export
can converge on a stable model before the legacy block tree is physically
removed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .enums import BlockType, OcrPolicy
from .project import BBox, BlockOrigin


@dataclass(frozen=True)
class LayoutBlockSnapshot:
    block_type: BlockType
    bbox: BBox
    order: int
    source_label: str
    origin: BlockOrigin
    ocr_policy: OcrPolicy
    note: str = ""


@dataclass(frozen=True)
class LayoutSnapshot:
    page_uid: str
    artifact_uid: str
    source_engine: str
    source_run_id: str
    blocks: tuple[LayoutBlockSnapshot, ...]


__all__ = [
    "LayoutBlockSnapshot",
    "LayoutSnapshot",
]
