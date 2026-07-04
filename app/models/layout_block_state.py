"""Helpers for interpreting layout block lifecycle state."""
from __future__ import annotations

from .enums import BlockSource, BlockType, OcrPolicy
from .project import BBox, Block


USER_AUTHORED_LAYOUT_SOURCES = frozenset({
    BlockSource.MANUAL_DRAW,
    BlockSource.USER_EDITED,
})


def block_source_value(block: Block) -> str:
    source = getattr(block, "source", BlockSource.AUTO_LAYOUT)
    return getattr(source, "value", str(source or ""))


def is_user_authored_layout_block(block: Block) -> bool:
    return getattr(block, "source", None) in USER_AUTHORED_LAYOUT_SOURCES


def is_user_authored_layout_source(value: object) -> bool:
    text = str(value or "")
    return text in {source.value for source in USER_AUTHORED_LAYOUT_SOURCES}


def mark_layout_block_manual_draw(block: Block) -> None:
    block.source = BlockSource.MANUAL_DRAW


def mark_layout_block_user_edited(block: Block) -> None:
    block.source = BlockSource.USER_EDITED


def set_layout_block_ocr_policy(block: Block, policy: OcrPolicy) -> None:
    block.ocr_policy = policy


def set_layout_block_source_label(block: Block, source_label: str) -> None:
    block.source_label = str(source_label or "")


def set_layout_block_type(block: Block, block_type: BlockType) -> None:
    block.block_type = block_type


def set_layout_block_note(block: Block, note: str) -> None:
    block.note = str(note or "")


def set_layout_block_bbox(block: Block, bbox: BBox) -> None:
    block.bbox = bbox


def set_layout_block_order(block: Block, order: int) -> None:
    block.order = int(order)


def append_layout_block_note_once(block: Block, note: str) -> None:
    text = str(note or "")
    if not text:
        return
    if not block.note:
        set_layout_block_note(block, text)
        return
    if text not in block.note:
        set_layout_block_note(block, f"{block.note}\n{text}")


def is_auto_tightened_layout_block(block: Block) -> bool:
    return getattr(block, "source", None) == BlockSource.AUTO_TIGHTENED


def export_origin_for_block(block: Block) -> str:
    if is_auto_tightened_layout_block(block):
        return "derived"
    return "project_model"
