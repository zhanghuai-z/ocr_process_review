from __future__ import annotations

from collections.abc import Iterator

from app.core.block_attributes import is_position_only_block, semantic_block_type
from app.models import BBox, Block, BlockType, Line, Page


PROOF_LINE_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.REFERENCE,
}

HPROOF_LINE_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.REFERENCE,
}

PROOF_SKIP_LINE_FLAGS = {
    "hanwang_route_table",
}

def _is_duplicate_line(line: Line, seen: list[tuple[str, BBox]]) -> bool:
    text = line.text or ""
    bbox = line.bbox.normalize()
    for seen_text, seen_bbox in seen:
        if text == seen_text and bbox.iou(seen_bbox) >= 0.85:
            return True
    seen.append((text, bbox))
    return False


def iter_unique_page_text_lines(page: Page) -> Iterator[tuple[Block, Line, int]]:
    """Yield text lines once per page, suppressing duplicate OCR rows.

    Paddle/layout retries can leave the same text line in overlapping blocks.
    Proof panels and vertical collections should consume each physical line once,
    while preserving repeated text at different page positions.
    """
    seen: list[tuple[str, BBox]] = []
    for block in page.blocks:
        if block.block_type == BlockType.EQUATION:
            continue
        if semantic_block_type(block) not in PROOF_LINE_BLOCK_TYPES:
            continue
        for line_idx, line in enumerate(block.lines):
            if any(flag in PROOF_SKIP_LINE_FLAGS for flag in line.review_flags):
                continue
            if _is_duplicate_line(line, seen):
                continue
            yield block, line, line_idx


def iter_unique_page_hproof_lines(page: Page) -> Iterator[tuple[Block, Line, int]]:
    """Yield only text-like lines for HProof, excluding captions/equations."""
    seen: list[tuple[str, BBox]] = []
    for block in page.blocks:
        if block.block_type == BlockType.EQUATION:
            continue
        if semantic_block_type(block) not in HPROOF_LINE_BLOCK_TYPES:
            continue
        if is_position_only_block(block):
            continue
        for line_idx, line in enumerate(block.lines):
            if any(flag in PROOF_SKIP_LINE_FLAGS for flag in line.review_flags):
                continue
            if _is_duplicate_line(line, seen):
                continue
            yield block, line, line_idx
