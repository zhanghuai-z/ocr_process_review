from __future__ import annotations

from collections.abc import Iterator

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.block_attributes import normalize_source_label
from app.core.proof_line_facts import proof_display_text
from app.models import BBox, Block, BlockType, Line, Page
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_observation import block_ocr_line_observations_by_uid, line_ocr_bbox
from app.models.ocr_text_observation import line_ocr_review_flags


PROOF_LINE_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.REFERENCE,
}

PROOF_SKIP_LINE_FLAGS = {
    "hanwang_route_table",
}

POSITION_ONLY_LABELS = {
    "page_number",
    "number",
    "formula_number",
    "header",
    "footer",
    "sidebar_text",
}


def _semantic_block_type_from_view(view: LayoutBlockView) -> BlockType:
    label = normalize_source_label(view.origin.source_label or view.source_label or view.block_type.value)
    semantic = map_paddle_label_to_block_type(label)
    return view.block_type if semantic == BlockType.UNKNOWN else semantic


def _is_position_only_view(view: LayoutBlockView) -> bool:
    label = normalize_source_label(view.origin.source_label or view.source_label)
    return label in POSITION_ONLY_LABELS


def _is_duplicate_line(line: Line, seen: list[tuple[str, BBox]]) -> bool:
    text = proof_display_text(line)
    bbox = line_ocr_bbox(line).normalize()
    for seen_text, seen_bbox in seen:
        if text == seen_text and bbox.iou(seen_bbox) >= 0.85:
            return True
    seen.append((text, bbox))
    return False


def iter_unique_page_text_line_views(page: Page) -> Iterator[tuple[LayoutBlockView, Block, Line, int]]:
    """Yield text lines once per page, suppressing duplicate OCR rows.

    Paddle/layout retries can leave the same text line in overlapping blocks.
    Proof panels and vertical collections should consume each physical line once,
    while preserving repeated text at different page positions.
    """
    seen: list[tuple[str, BBox]] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        if view.block_type == BlockType.EQUATION:
            continue
        if _semantic_block_type_from_view(view) not in PROOF_LINE_BLOCK_TYPES:
            continue
        for line_idx, line in enumerate(block_ocr_line_observations_by_uid(view.uid)):
            if any(flag in PROOF_SKIP_LINE_FLAGS for flag in line_ocr_review_flags(line)):
                continue
            if _is_duplicate_line(line, seen):
                continue
            yield view, block, line, line_idx


def iter_unique_page_text_lines(page: Page) -> Iterator[tuple[Block, Line, int]]:
    """Yield text lines once per page, suppressing duplicate OCR rows."""
    for _view, block, line, line_idx in iter_unique_page_text_line_views(page):
        yield block, line, line_idx


def iter_unique_page_hproof_lines(page: Page) -> Iterator[tuple[Block, Line, int]]:
    """Yield proof text lines for HProof, excluding position-only blocks.

    HProof and VProof should be two views over the same proofable text facts.
    HProof keeps one extra UI filter: page numbers/headers and other
    position-only blocks are not useful in row-by-row proofreading.
    """
    seen: list[tuple[str, BBox]] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        if view.block_type == BlockType.EQUATION:
            continue
        if _semantic_block_type_from_view(view) not in PROOF_LINE_BLOCK_TYPES:
            continue
        if _is_position_only_view(view):
            continue
        for line_idx, line in enumerate(block_ocr_line_observations_by_uid(view.uid)):
            if any(flag in PROOF_SKIP_LINE_FLAGS for flag in line_ocr_review_flags(line)):
                continue
            if _is_duplicate_line(line, seen):
                continue
            yield block, line, line_idx
