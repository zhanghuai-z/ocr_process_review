"""Authoritative Paddle/PP-VL layout label extraction."""
from __future__ import annotations

from typing import Any

PADDLE_LABEL_AUTHORITY_KEYS = (
    "block_label",
    "label",
    "type",
    "category",
    "category_name",
    "cls_name",
    "layout_label",
)


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
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
