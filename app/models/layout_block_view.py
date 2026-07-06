"""Read views over adopted layout snapshots and runtime block projection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .layout_projection import page_layout_blocks
from .layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from .layout_snapshot_projection import layout_snapshot_from_blocks
from .layout_snapshot_store import layout_snapshot_for_page
from .project import BBox, Block, BlockOrigin, Page
from .enums import BlockType, OcrPolicy


@dataclass(frozen=True)
class LayoutBlockView:
    """A layout-truth block paired with its current runtime projection.

    Layout fields come from ``LayoutSnapshot``.  ``runtime_block`` is present
    only while existing OCR/proof/export code still stores side facts on
    ``Block``.
    """

    page: Page
    snapshot_block: LayoutBlockSnapshot
    snapshot_index: int
    runtime_block: Block | None
    runtime_block_index: int | None

    @property
    def uid(self) -> str:
        return self.snapshot_block.uid

    @property
    def block_type(self) -> BlockType:
        return self.snapshot_block.block_type

    @property
    def bbox(self) -> BBox:
        return self.snapshot_block.bbox

    @property
    def order(self) -> int:
        return self.snapshot_block.order

    @property
    def source_label(self) -> str:
        return self.snapshot_block.source_label

    @property
    def origin(self) -> BlockOrigin:
        return self.snapshot_block.origin

    @property
    def ocr_policy(self) -> OcrPolicy:
        return self.snapshot_block.ocr_policy

    @property
    def note(self) -> str:
        return self.snapshot_block.note


def current_layout_snapshot(page: Page) -> LayoutSnapshot:
    snapshot = layout_snapshot_for_page(page)
    if snapshot is not None:
        return snapshot
    return layout_snapshot_from_blocks(page, source_engine="layout_projection_view")


def iter_page_layout_block_views(page: Page) -> Iterator[LayoutBlockView]:
    runtime_blocks = page_layout_blocks(page)
    runtime_by_uid: dict[str, tuple[int, Block]] = {}
    for block_index, block in enumerate(runtime_blocks):
        if block.uid and block.uid not in runtime_by_uid:
            runtime_by_uid[block.uid] = (block_index, block)
    snapshot = current_layout_snapshot(page)
    for snapshot_index, snapshot_block in enumerate(snapshot.blocks):
        runtime_pair = runtime_by_uid.get(snapshot_block.uid)
        runtime_block_index: int | None = None
        runtime_block: Block | None = None
        if runtime_pair is not None:
            runtime_block_index, runtime_block = runtime_pair
        yield LayoutBlockView(
            page=page,
            snapshot_block=snapshot_block,
            snapshot_index=snapshot_index,
            runtime_block=runtime_block,
            runtime_block_index=runtime_block_index,
        )


__all__ = [
    "LayoutBlockView",
    "current_layout_snapshot",
    "iter_page_layout_block_views",
]
