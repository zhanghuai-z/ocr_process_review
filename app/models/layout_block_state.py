"""Helpers for interpreting layout block lifecycle state."""
from __future__ import annotations

from .enums import BlockSource, OcrPolicy
from .project import Block


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


def is_auto_tightened_layout_block(block: Block) -> bool:
    return getattr(block, "source", None) == BlockSource.AUTO_TIGHTENED


def export_origin_for_block(block: Block) -> str:
    if is_auto_tightened_layout_block(block):
        return "derived"
    return "project_model"
