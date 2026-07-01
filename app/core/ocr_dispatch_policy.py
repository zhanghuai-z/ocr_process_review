"""Shared policy for dispatching layout blocks to text OCR."""
from __future__ import annotations

from typing import Protocol

from app.core.block_attributes import block_attributes
from app.core.paddle_labels import is_hanwang_skip_label, normalize_paddle_label
from app.models.enums import BlockType, OcrPolicy


class OcrDispatchBlock(Protocol):
    block_type: BlockType
    ocr_policy: OcrPolicy


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


def authoritative_block_label(block: OcrDispatchBlock) -> str:
    """Return the normalized source label used by routing decisions."""
    attrs = block_attributes(block)  # type: ignore[arg-type]
    return attrs.normalized_semantic_label or attrs.source_label or block.block_type.value


def is_text_ocr_candidate(block: OcrDispatchBlock) -> bool:
    """Whether the block's semantic type/source label may enter text OCR.

    This ignores ``block.ocr_policy`` so UI code can use it when deriving the
    default policy after type changes. Runtime dispatch should call
    ``should_dispatch_to_text_ocr`` instead.
    """
    if block.block_type in PRESERVE_BLOCK_TYPES:
        return False
    label = authoritative_block_label(block)
    if is_hanwang_skip_label(label):
        return False
    return block.block_type in TEXT_OCR_BLOCK_TYPES


def should_dispatch_to_text_ocr(block: OcrDispatchBlock) -> bool:
    """Whether this block should be sent to a text OCR engine now."""
    return block.ocr_policy == OcrPolicy.TEXT_OCR


def default_ocr_policy_for_block(block: OcrDispatchBlock) -> OcrPolicy:
    label = normalize_paddle_label(authoritative_block_label(block))
    if block.block_type == BlockType.EQUATION or any(token in label for token in ("equation", "formula", "math")):
        return OcrPolicy.PRESERVE_AS_FORMULA
    if block.block_type == BlockType.TABLE or "table" in label:
        return OcrPolicy.PRESERVE_AS_TABLE
    if is_text_ocr_candidate(block):
        return OcrPolicy.TEXT_OCR
    return OcrPolicy.SKIP


def should_block_page_ocr_line(block: OcrDispatchBlock) -> bool:
    """Whether page-level OCR lines inside this block should be ignored.

    Page OCR runs on the full image, so formula/table/figure text may still be
    observed by the OCR engine. Those lines must not be assigned to text proof
    blocks or synthesized as unmatched text blocks.
    """
    return not should_dispatch_to_text_ocr(block)
