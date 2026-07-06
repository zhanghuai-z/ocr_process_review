"""Projection helpers between ``LayoutSnapshot`` and current ``Page.blocks``."""
from __future__ import annotations

from .layout_projection import page_layout_blocks, replace_page_layout_blocks
from .layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from .layout_snapshot_store import set_layout_snapshot_for_page
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


def _snapshot_from_block(block: Block) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        block_type=block.block_type,
        bbox=block.bbox,
        order=block.order,
        source_label=block.source_label,
        origin=block.origin or _origin_from_block(block),
        ocr_policy=block.ocr_policy,
        note=block.note,
        uid=block.uid,
    )


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
    "layout_snapshot_from_blocks",
    "project_layout_snapshot_to_blocks",
    "sync_page_layout_snapshot_from_projection",
]
