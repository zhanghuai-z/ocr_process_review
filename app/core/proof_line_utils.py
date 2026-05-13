from __future__ import annotations

from collections.abc import Iterator

from app.models import BBox, Block, BlockType, Line, Page


PROOF_LINE_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.REFERENCE,
    BlockType.EQUATION,
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
        if block.block_type not in PROOF_LINE_BLOCK_TYPES:
            continue
        for line_idx, line in enumerate(block.lines):
            if _is_duplicate_line(line, seen):
                continue
            yield block, line, line_idx
