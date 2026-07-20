"""Immutable OCR dispatch records derived from page and layout snapshots."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.core.ocr_dispatch_policy import should_dispatch_to_text_ocr
from app.models.enums import BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.project_session import PageRecord


@dataclass(frozen=True, slots=True)
class DispatchBlock:
    """Immutable facts needed to dispatch one layout block to OCR."""

    block_uid: str
    order: int
    bbox: BBox
    block_type: BlockType
    ocr_policy: OcrPolicy
    source_label: str
    reason: str


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """Immutable OCR dispatch decision for one page layout snapshot."""

    page: PageRecord
    layout: LayoutSnapshot
    text_blocks: tuple[DispatchBlock, ...]
    blocked_blocks: tuple[DispatchBlock, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.page, PageRecord):
            raise TypeError("dispatch plan page must be a PageRecord")
        if not isinstance(self.layout, LayoutSnapshot):
            raise TypeError("dispatch plan layout must be a LayoutSnapshot")
        if self.page.uid != self.layout.page_uid:
            raise ValueError("dispatch plan page and layout snapshot must have the same page UID")
        text_blocks = tuple(self.text_blocks)
        blocked_blocks = tuple(self.blocked_blocks)
        if any(not isinstance(block, DispatchBlock) for block in (*text_blocks, *blocked_blocks)):
            raise TypeError("dispatch plan entries must be DispatchBlock records")
        object.__setattr__(self, "text_blocks", text_blocks)
        object.__setattr__(self, "blocked_blocks", blocked_blocks)

    @property
    def page_uid(self) -> str:
        return self.page.uid

    @property
    def layout_revision(self) -> int:
        return self.layout.revision

    @property
    def total_text_blocks(self) -> int:
        return len(self.text_blocks)

    @property
    def has_text_work(self) -> bool:
        return bool(self.text_blocks)

    @property
    def text_block_uids(self) -> tuple[str, ...]:
        return tuple(block.block_uid for block in self.text_blocks)

    @property
    def blocked_block_uids(self) -> tuple[str, ...]:
        return tuple(block.block_uid for block in self.blocked_blocks)


def _dispatch_record(block: LayoutBlockSnapshot, *, reason: str) -> DispatchBlock:
    return DispatchBlock(
        block_uid=block.uid,
        order=block.order,
        bbox=block.bbox,
        block_type=block.block_type,
        ocr_policy=block.ocr_policy,
        source_label=block.source_label,
        reason=reason,
    )


def build_text_ocr_dispatch_plan(
    page: PageRecord,
    layout: LayoutSnapshot,
) -> DispatchPlan:
    """Build the text OCR dispatch plan from immutable page/layout facts."""
    if not isinstance(page, PageRecord):
        raise TypeError("dispatch plan page must be a PageRecord")
    if not isinstance(layout, LayoutSnapshot):
        raise TypeError("dispatch plan layout must be a LayoutSnapshot")
    if page.uid != layout.page_uid:
        raise ValueError("dispatch plan page and layout snapshot must have the same page UID")

    text_blocks: list[DispatchBlock] = []
    blocked_blocks: list[DispatchBlock] = []
    for block in layout.blocks:
        if should_dispatch_to_text_ocr(block):
            text_blocks.append(_dispatch_record(block, reason="policy:text_ocr"))
        else:
            blocked_blocks.append(
                _dispatch_record(block, reason=f"policy:{block.ocr_policy.value}")
            )
    return DispatchPlan(
        page=page,
        layout=layout,
        text_blocks=tuple(text_blocks),
        blocked_blocks=tuple(blocked_blocks),
    )


PageLayoutInput = tuple[PageRecord, LayoutSnapshot]


def iter_text_ocr_blocks(
    page_layouts: Iterable[PageLayoutInput],
) -> Iterator[DispatchBlock]:
    """Yield immutable text OCR dispatch records in page/layout order."""
    for page, layout in page_layouts:
        yield from build_text_ocr_dispatch_plan(page, layout).text_blocks


def count_text_ocr_blocks(page_layouts: Iterable[PageLayoutInput]) -> int:
    """Count text OCR dispatch records without exposing mutable models."""
    return sum(1 for _ in iter_text_ocr_blocks(page_layouts))


__all__ = [
    "DispatchBlock",
    "DispatchPlan",
    "PageLayoutInput",
    "build_text_ocr_dispatch_plan",
    "count_text_ocr_blocks",
    "iter_text_ocr_blocks",
]
