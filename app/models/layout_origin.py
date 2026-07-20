"""Immutable provenance attached to adopted layout regions."""
from __future__ import annotations

from dataclasses import dataclass

from .enums import BlockSource, BlockType
from .geometry import BBox


@dataclass(frozen=True, slots=True)
class BlockOrigin:
    """Vendor/import provenance; never the current semantic layout truth."""

    created_by: str = BlockSource.AUTO_LAYOUT.value
    source_engine: str = ""
    source_run_id: str = ""
    vendor_label: str = ""
    source_confidence: float | None = None
    original_bbox: BBox | None = None
    original_kind: BlockType | None = None
    raw_artifact_uid: str = ""
    raw_json_path: str = ""
    raw_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "created_by": self.created_by,
            "source_engine": self.source_engine,
            "source_run_id": self.source_run_id,
            "vendor_label": self.vendor_label,
            "source_confidence": self.source_confidence,
            "original_bbox": self.original_bbox.to_dict() if self.original_bbox else None,
            "original_kind": self.original_kind.value if self.original_kind else "",
            "raw_artifact_uid": self.raw_artifact_uid,
            "raw_json_path": self.raw_json_path,
            "raw_index": self.raw_index,
        }


__all__ = ["BlockOrigin"]
