"""Paddle layout label normalization for application block types."""
from __future__ import annotations

from app.core.paddle_labels import normalize_paddle_label
from app.models import BlockType


_PADDLE_LABEL_TO_BLOCK_TYPE: dict[str, BlockType] = {
    "text": BlockType.TEXT,
    "paragraph_text": BlockType.TEXT,
    "text_block": BlockType.TEXT,
    "text_box": BlockType.TEXT,
    "body": BlockType.TEXT,
    "title": BlockType.TITLE,
    "doc_title": BlockType.TITLE,
    "doc_heading": BlockType.TITLE,
    "section_title": BlockType.TITLE,
    "chapter_title": BlockType.TITLE,
    "paragraph_title": BlockType.TITLE,
    "text_title": BlockType.TITLE,
    "heading": BlockType.TITLE,
    "headline": BlockType.TITLE,
    "heading_1": BlockType.TITLE,
    "heading_2": BlockType.TITLE,
    "heading_3": BlockType.TITLE,
    "heading_4": BlockType.TITLE,
    "heading_5": BlockType.TITLE,
    "heading_6": BlockType.TITLE,
    "paragraph": BlockType.TEXT,
    "plain_text": BlockType.TEXT,
    "doc_text": BlockType.TEXT,
    "text_region": BlockType.TEXT,
    "body_text": BlockType.TEXT,
    "content": BlockType.TEXT,
    "abstract": BlockType.TEXT,
    "table_of_contents": BlockType.TEXT,
    "toc": BlockType.TEXT,
    "page_number": BlockType.TEXT,
    "number": BlockType.TEXT,
    "header": BlockType.TEXT,
    "footer": BlockType.TEXT,
    "footnote": BlockType.TEXT,
    "vision_footnote": BlockType.TEXT,
    "sidebar_text": BlockType.TEXT,
    "algorithm": BlockType.TEXT,
    "figure": BlockType.FIGURE,
    "graphic": BlockType.FIGURE,
    "photo": BlockType.FIGURE,
    "logo": BlockType.FIGURE,
    "image": BlockType.FIGURE,
    "picture": BlockType.FIGURE,
    "illustration": BlockType.FIGURE,
    "chart": BlockType.FIGURE,
    "seal": BlockType.FIGURE,
    "header_image": BlockType.FIGURE,
    "footer_image": BlockType.FIGURE,
    "figure_caption": BlockType.FIGURE_CAPTION,
    "figure_title": BlockType.FIGURE_CAPTION,
    "caption": BlockType.FIGURE_CAPTION,
    "figure_note": BlockType.FIGURE_CAPTION,
    "image_caption": BlockType.FIGURE_CAPTION,
    "table": BlockType.TABLE,
    "table_region": BlockType.TABLE,
    "table_block": BlockType.TABLE,
    "table_cell": BlockType.TABLE,
    "table_body": BlockType.TABLE,
    "table_caption": BlockType.TABLE_CAPTION,
    "table_caption_text": BlockType.TABLE_CAPTION,
    "table_note": BlockType.TABLE_CAPTION,
    "table_title": BlockType.TABLE_CAPTION,
    "reference": BlockType.REFERENCE,
    "reference_content": BlockType.REFERENCE,
    "references": BlockType.REFERENCE,
    "reference_list": BlockType.REFERENCE,
    "reference_text": BlockType.REFERENCE,
    "bibliography": BlockType.REFERENCE,
    "equation": BlockType.EQUATION,
    "equation_block": BlockType.EQUATION,
    "isolated_formula": BlockType.EQUATION,
    "inline_formula": BlockType.EQUATION,
    "formula": BlockType.EQUATION,
    "formula_number": BlockType.EQUATION,
}


def map_paddle_label_to_block_type(label: object) -> BlockType:
    """Map vendor layout labels into the application's coarse block taxonomy."""
    normalized = normalize_paddle_label(label)
    if not normalized:
        return BlockType.UNKNOWN
    if normalized in _PADDLE_LABEL_TO_BLOCK_TYPE:
        return _PADDLE_LABEL_TO_BLOCK_TYPE[normalized]

    if "title" in normalized or normalized.startswith("heading_"):
        return BlockType.TITLE
    if "caption" in normalized and "table" in normalized:
        return BlockType.TABLE_CAPTION
    if "note" in normalized and "table" in normalized:
        return BlockType.TABLE_CAPTION
    if "caption" in normalized:
        return BlockType.FIGURE_CAPTION
    if "table" in normalized:
        return BlockType.TABLE
    if any(token in normalized for token in ("reference", "bibliography")):
        return BlockType.REFERENCE
    if any(token in normalized for token in ("paragraph", "text", "body", "content")):
        return BlockType.TEXT
    if any(token in normalized for token in ("figure", "image", "picture", "illustration", "graphic", "logo", "photo")):
        return BlockType.FIGURE
    if any(token in normalized for token in ("equation", "formula", "math")):
        return BlockType.EQUATION
    return BlockType.UNKNOWN


__all__ = ["map_paddle_label_to_block_type"]
