"""Authoritative, immutable page layout records."""
from __future__ import annotations

from dataclasses import dataclass

from .entity_id import ensure_entity_uid
from .enums import BlockSource, BlockType, OcrPolicy
from .geometry import BBox
from .layout_origin import BlockOrigin


@dataclass(frozen=True, slots=True)
class LayoutBlockSnapshot:
    block_type: BlockType
    bbox: BBox
    order: int
    source_label: str
    origin: BlockOrigin
    ocr_policy: OcrPolicy
    authorship: BlockSource = BlockSource.AUTO_LAYOUT
    uid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.block_type, BlockType):
            raise TypeError("block_type must be BlockType")
        if not isinstance(self.bbox, BBox) or self.bbox.w <= 0 or self.bbox.h <= 0:
            raise ValueError("layout block bbox must be a non-empty BBox")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("layout block order must be a non-negative integer")
        if not isinstance(self.source_label, str):
            raise TypeError("source_label must be str")
        if not isinstance(self.origin, BlockOrigin):
            raise TypeError("origin must be BlockOrigin")
        if not isinstance(self.ocr_policy, OcrPolicy):
            raise TypeError("ocr_policy must be OcrPolicy")
        if not isinstance(self.authorship, BlockSource):
            raise TypeError("authorship must be BlockSource")
        object.__setattr__(self, "uid", ensure_entity_uid(self.uid, "block"))


@dataclass(frozen=True, slots=True)
class LayoutSnapshot:
    page_uid: str
    revision: int
    artifact_uid: str
    source_engine: str
    source_run_id: str
    blocks: tuple[LayoutBlockSnapshot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.page_uid, str) or not self.page_uid.strip():
            raise ValueError("page_uid must be a non-empty stable UID")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("layout revision must be a positive integer")
        for field_name in ("artifact_uid", "source_engine", "source_run_id"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be str")
        blocks = tuple(self.blocks)
        if any(not isinstance(block, LayoutBlockSnapshot) for block in blocks):
            raise TypeError("blocks must contain LayoutBlockSnapshot values")
        if len({block.uid for block in blocks}) != len(blocks):
            raise ValueError("layout snapshot contains duplicate block UIDs")
        if tuple(block.order for block in blocks) != tuple(range(len(blocks))):
            raise ValueError("layout block order must match snapshot sequence")
        object.__setattr__(self, "blocks", blocks)


__all__ = [
    "LayoutBlockSnapshot",
    "LayoutSnapshot",
]
