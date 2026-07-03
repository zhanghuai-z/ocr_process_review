"""Current layout truth model and legacy block projection.

``LayoutSnapshot`` is the application-level layout adopted for a page. External
artifacts such as PaddleOCR-VL or vector PDF extraction are normalized first,
then compiled into this snapshot. ``Page.blocks`` remains a compatibility
projection for existing UI/OCR code.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.normalized_layout_artifact import NormalizedLayoutArtifact
from app.core.ocr_dispatch_policy import default_ocr_policy_for_block
from app.core.paddle_labels import normalize_paddle_label
from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, OcrPolicy


_POSITION_SOURCE_LABELS = {
    "page_number",
    "number",
    "formula_number",
    "header",
    "footer",
    "footnote",
    "sidebar_text",
}


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
    "layout_snapshot_from_normalized_artifact",
    "project_layout_snapshot_to_blocks",
]
