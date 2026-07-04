"""Compile layout snapshots and project them to the legacy block tree.

``LayoutSnapshot`` is the application-level layout adopted for a page. External
artifacts such as PaddleOCR-VL or vector PDF extraction are normalized first.
``Page.blocks`` remains a compatibility projection for existing UI/OCR code.
"""
from __future__ import annotations

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.normalized_layout_artifact import NormalizedLayoutArtifact
from app.core.ocr_dispatch_policy import default_ocr_policy_for_block
from app.core.paddle_labels import normalize_paddle_label
from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Page
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.layout_snapshot_store import set_layout_snapshot_for_page
from app.models.layout_projection import page_layout_blocks, replace_page_layout_blocks


_POSITION_SOURCE_LABELS = {
    "page_number",
    "number",
    "formula_number",
    "header",
    "footer",
    "footnote",
    "sidebar_text",
}


def layout_snapshot_from_normalized_artifact(
    artifact: NormalizedLayoutArtifact,
    *,
    source_run_id: str = "",
) -> LayoutSnapshot:
    blocks: list[LayoutBlockSnapshot] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    order = 0
    for region in artifact.regions:
        source_label = normalize_paddle_label(region.label)
        signature = (region.label, *region.bbox)
        if signature in seen:
            continue
        seen.add(signature)
        block_type = map_paddle_label_to_block_type(region.label)
        bbox = BBox.from_xyxy(*region.bbox)
        origin = BlockOrigin(
            created_by=BlockSource.AUTO_LAYOUT.value,
            source_engine=artifact.engine,
            source_run_id=source_run_id,
            source_label=source_label,
            source_confidence=region.confidence,
            original_bbox=bbox,
            original_kind=block_type,
            raw_artifact_uid=artifact.artifact_uid,
            raw_index=region.index,
        )
        probe = Block(
            block_type=block_type,
            bbox=bbox,
            order=order,
            source_label=source_label,
            origin=origin,
        )
        ocr_policy = default_ocr_policy_for_block(probe)
        blocks.append(
            LayoutBlockSnapshot(
                block_type=block_type,
                bbox=bbox,
                order=order,
                source_label=source_label,
                origin=origin,
                ocr_policy=ocr_policy,
                note=_note_for_region(region.text, region.confidence, source_label),
            )
        )
        order += 1
    return LayoutSnapshot(
        page_uid=artifact.page_uid,
        artifact_uid=artifact.artifact_uid,
        source_engine=artifact.engine,
        source_run_id=source_run_id,
        blocks=tuple(blocks),
    )


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
            _snapshot_from_block(block, order=order)
            for order, block in enumerate(page_layout_blocks(page))
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
    set_layout_snapshot_for_page(page, snapshot)
    return blocks


def _snapshot_from_block(block: Block, *, order: int) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        block_type=block.block_type,
        bbox=block.bbox,
        order=order,
        source_label=block.source_label,
        origin=block.origin or _origin_from_block(block),
        ocr_policy=block.ocr_policy,
        note=block.note,
    )


def _origin_from_block(block: Block) -> BlockOrigin:
    return BlockOrigin(
        created_by=getattr(block.source, "value", str(block.source or BlockSource.AUTO_LAYOUT.value)),
        source_label=str(block.source_label or ""),
        original_bbox=block.bbox,
        original_kind=block.block_type,
    )


def _note_for_region(text: str, confidence: float | None, source_label: str) -> str:
    note_parts: list[str] = []
    preview = str(text or "")
    if preview:
        note_parts.append(preview[:120])
    if confidence is not None:
        note_parts.append(f"score={confidence:.3f}")
    if source_label in _POSITION_SOURCE_LABELS:
        note_parts.append(f"source_label={source_label}")
    return " | ".join(note_parts)


__all__ = [
    "LayoutBlockSnapshot",
    "LayoutSnapshot",
    "adopt_page_layout_snapshot",
    "layout_snapshot_from_blocks",
    "layout_snapshot_from_normalized_artifact",
    "project_layout_snapshot_to_blocks",
    "sync_page_layout_snapshot_from_projection",
]
