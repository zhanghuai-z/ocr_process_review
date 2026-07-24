"""Authoritative Paddle/PP-VL layout label extraction."""
from __future__ import annotations

from typing import Any

PADDLE_LABEL_AUTHORITY_KEYS = (
    "block_label",
    "label",
)

PADDLE_HANWANG_TEXT_LABELS = {
    "text",
    "paragraph",
    "paragraph_text",
    "paragraph_title",
    "plain_text",
    "body",
    "body_text",
    "abstract",
    "title",
    "header",
    "footer",
    "footnote",
    "number",
    "page_number",
    "reference",
    "reference_content",
    "references",
    "reference_list",
    "bibliography",
    "caption",
    "figure_caption",
    "figure_title",
    "table_caption",
    "table_title",
    "table_note",
}

PADDLE_LABEL_ALIASES = {
    # Legacy Paddle/VL wording. The product/UI concept is simply "footnote".
    "vision_footnote": "footnote",
}

PADDLE_HANWANG_SKIP_LABELS = {
    "display_formula",
    "inline_formula",
    "isolated_formula",
    "formula",
    "formula_number",
    "equation",
    "table",
    "table_region",
    "table_block",
    "table_body",
    "figure",
    "chart",
    "graphic",
    "image",
    "picture",
    "photo",
    "seal",
    "stamp",
}


def authoritative_paddle_label(record: dict[str, Any] | None, default: str = "") -> str:
    """Return Paddle layout label using the shared field-truth precedence."""
    if not isinstance(record, dict):
        return default
    for key in PADDLE_LABEL_AUTHORITY_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return default


def normalize_paddle_label(label: object) -> str:
    normalized = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return PADDLE_LABEL_ALIASES.get(normalized, normalized)


def is_hanwang_text_label(label: object) -> bool:
    normalized = normalize_paddle_label(label)
    return normalized in PADDLE_HANWANG_TEXT_LABELS


def is_hanwang_skip_label(label: object) -> bool:
    normalized = normalize_paddle_label(label)
    if normalized in PADDLE_HANWANG_TEXT_LABELS:
        return False
    return normalized in PADDLE_HANWANG_SKIP_LABELS
