"""Shared policy for dispatching layout blocks to text OCR."""
from __future__ import annotations

from typing import Any

from app.core.paddle_labels import authoritative_paddle_label, is_hanwang_skip_label
from app.models import Block, BlockType


TEXT_OCR_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.REFERENCE,
}

PRESERVE_BLOCK_TYPES = {
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.UNKNOWN,
}


def _payload_label(payload: dict[str, Any]) -> str:
    binding = payload.get("paddle_binding")
    if isinstance(binding, dict):
        label = binding.get("source_label") or binding.get("block_type")
        if label:
            return str(label)
    return authoritative_paddle_label(payload, default="")


def authoritative_block_label(block: Block) -> str:
    """Return the best available vendor/source label for routing decisions."""
    if block.source_label:
        return str(block.source_label)
    if isinstance(block.raw_payload, dict):
        label = _payload_label(block.raw_payload)
        if label:
            return label
    return block.block_type.value


def is_text_ocr_candidate(block: Block) -> bool:
    """Whether the block's semantic type/source label may enter text OCR.

    This ignores ``block.recognizable`` so UI code can use it when refreshing
    recognizability after type changes. Runtime dispatch should call
    ``should_dispatch_to_text_ocr`` instead.
    """
    if block.block_type in PRESERVE_BLOCK_TYPES:
        return False
    label = authoritative_block_label(block)
    if is_hanwang_skip_label(label):
        return False
    return block.block_type in TEXT_OCR_BLOCK_TYPES


def should_dispatch_to_text_ocr(block: Block) -> bool:
    """Whether this block should be sent to a text OCR engine now."""
    return bool(block.recognizable) and is_text_ocr_candidate(block)


def should_block_page_ocr_line(block: Block) -> bool:
    """Whether page-level OCR lines inside this block should be ignored.

    Page OCR runs on the full image, so formula/table/figure text may still be
    observed by the OCR engine. Those lines must not be assigned to text proof
    blocks or synthesized as unmatched text blocks.
    """
    return not should_dispatch_to_text_ocr(block)
