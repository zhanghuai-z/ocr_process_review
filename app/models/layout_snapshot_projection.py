"""Projection helpers between ``LayoutSnapshot`` and current ``Page.blocks``."""
from __future__ import annotations

from typing import Iterable

from .layout_projection import page_layout_blocks, replace_page_layout_blocks
from .layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from .layout_snapshot_store import set_layout_snapshot_for_page
from .layout_block_state import (
    set_layout_block_bbox,
    set_layout_block_note,
    set_layout_block_ocr_policy,
    set_layout_block_order,
    set_layout_block_source_label,
    set_layout_block_type,
)
from .project import Block, BlockOrigin, BlockSource, Page


def project_layout_snapshot_to_blocks(snapshot: LayoutSnapshot) -> list[Block]:
    return [
        Block(
            block_type=block.block_type,
            bbox=block.bbox,
            order=block.order,
            note=block.note,
            source_label=block.source_label,
            origin=block.origin,
            ocr_policy=block.ocr_policy,
            uid=block.uid,
        )
        for block in snapshot.blocks
    ]


def layout_snapshot_from_blocks(
    page: Page,
    *,
    artifact_uid: str = "",
    source_engine: str = "layout_projection",
    source_run_id: str = "",
) -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid=artifact_uid,
        source_engine=source_engine,
        source_run_id=source_run_id,
        blocks=tuple(
            _snapshot_from_block(block)
            for block in page_layout_blocks(page)
        ),
    )


def sync_page_layout_snapshot_from_projection(
    page: Page,
    *,
    source_engine: str = "layout_projection",
    source_run_id: str = "",
) -> LayoutSnapshot:
    snapshot = layout_snapshot_from_blocks(
        page,
        artifact_uid=page.raw_layout_artifact.uid if page.raw_layout_artifact else "",
        source_engine=source_engine,
        source_run_id=source_run_id,
    )
    set_layout_snapshot_for_page(page, snapshot)
    return snapshot


def adopt_page_layout_snapshot(page: Page, snapshot: LayoutSnapshot) -> list[Block]:
    blocks = project_layout_snapshot_to_blocks(snapshot)
    replace_page_layout_blocks(page, blocks)
    adopted_snapshot = _snapshot_with_projected_block_uids(snapshot, blocks)
    set_layout_snapshot_for_page(page, adopted_snapshot)
    return blocks


def replace_page_layout_projection_from_snapshot(
    page: Page,
    snapshot: LayoutSnapshot,
    *,
    candidate_blocks: Iterable[Block] = (),
) -> list[Block]:
    """Refresh the temporary ``Page.blocks`` projection from layout truth.

    Existing blocks are reused by uid so side facts attached to the current
    runtime projection are not discarded while the rest of the app migrates away
    from ``Page.blocks``.
    """

    runtime_by_uid = {
        block.uid: block
        for block in page_layout_blocks(page)
        if block.uid
    }
    for block in candidate_blocks:
        if block.uid:
            runtime_by_uid[block.uid] = block
    next_blocks: list[Block] = []
    for snapshot_block in snapshot.blocks:
        block = runtime_by_uid.get(snapshot_block.uid)
        if block is None:
            block = block_from_layout_snapshot(snapshot_block)
        apply_layout_snapshot_block_to_projection(block, snapshot_block)
        next_blocks.append(block)
    replace_page_layout_blocks(page, next_blocks)
    return next_blocks


def _snapshot_from_block(block: Block) -> LayoutBlockSnapshot:
    return layout_block_snapshot_from_projection_block(block)


def layout_block_snapshot_from_projection_block(
    block: Block,
    *,
    order: int | None = None,
) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        block_type=block.block_type,
        bbox=block.bbox,
        order=block.order if order is None else order,
        source_label=block.source_label,
        origin=block.origin or _origin_from_block(block),
        ocr_policy=block.ocr_policy,
        note=block.note,
        uid=block.uid,
    )


def block_from_layout_snapshot(snapshot_block: LayoutBlockSnapshot) -> Block:
    return Block(
        block_type=snapshot_block.block_type,
        bbox=snapshot_block.bbox,
        order=snapshot_block.order,
        note=snapshot_block.note,
        source_label=snapshot_block.source_label,
        origin=snapshot_block.origin,
        ocr_policy=snapshot_block.ocr_policy,
        uid=snapshot_block.uid,
    )


def apply_layout_snapshot_block_to_projection(
    block: Block,
    snapshot_block: LayoutBlockSnapshot,
) -> None:
    set_layout_block_type(block, snapshot_block.block_type)
    set_layout_block_bbox(block, snapshot_block.bbox)
    set_layout_block_order(block, snapshot_block.order)
    set_layout_block_source_label(block, snapshot_block.source_label)
    set_layout_block_ocr_policy(block, snapshot_block.ocr_policy)
    set_layout_block_note(block, snapshot_block.note)
    block.origin = snapshot_block.origin


def _snapshot_with_projected_block_uids(
    snapshot: LayoutSnapshot,
    blocks: list[Block],
) -> LayoutSnapshot:
    if len(snapshot.blocks) != len(blocks):
        return snapshot
    projected_blocks = tuple(
        LayoutBlockSnapshot(
            block_type=block_snapshot.block_type,
            bbox=block_snapshot.bbox,
            order=block_snapshot.order,
            source_label=block_snapshot.source_label,
            origin=block_snapshot.origin,
            ocr_policy=block_snapshot.ocr_policy,
            note=block_snapshot.note,
            uid=block.uid,
        )
        for block_snapshot, block in zip(snapshot.blocks, blocks)
    )
    if projected_blocks == snapshot.blocks:
        return snapshot
    return LayoutSnapshot(
        page_uid=snapshot.page_uid,
        artifact_uid=snapshot.artifact_uid,
        source_engine=snapshot.source_engine,
        source_run_id=snapshot.source_run_id,
        blocks=projected_blocks,
    )


def _origin_from_block(block: Block) -> BlockOrigin:
    return BlockOrigin(
        created_by=getattr(block.source, "value", str(block.source or BlockSource.AUTO_LAYOUT.value)),
        source_label=str(block.source_label or ""),
        original_bbox=block.bbox,
        original_kind=block.block_type,
    )


__all__ = [
    "adopt_page_layout_snapshot",
    "apply_layout_snapshot_block_to_projection",
    "block_from_layout_snapshot",
    "layout_block_snapshot_from_projection_block",
    "layout_snapshot_from_blocks",
    "project_layout_snapshot_to_blocks",
    "replace_page_layout_projection_from_snapshot",
    "sync_page_layout_snapshot_from_projection",
]
