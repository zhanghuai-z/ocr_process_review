"""Payload boundary constants and typed-state helpers for layout blocks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models import PaddleBinding

PADDLE_BINDING_KEY = "paddle_binding"
TABLE_TEXT_LAYER_CELLS_KEY = "table_text_layer_cells"

RUNTIME_LAYOUT_PAYLOAD_KEYS = frozenset({
    "_layout_line_routes",
    "_route_subblocks",
    "_layout_block_source",
    "_layout_block_ocr_policy",
    "_layout_paddle_parent_index",
    "_layout_manual_route_subblock",
})

OCR_TEXT_INVALIDATED_KEY = "ocr_text_invalidated"
OCR_INVALIDATION_KIND_KEY = "ocr_invalidation_kind"

MANUAL_MERGE_FROM_KEY = "manual_merge_from"
MANUAL_DRAW_BBOX_KEY = "manual_draw_bbox"

UI_GENERATED_INLINE_FORMULA_BLOCK_KEY = "ui_generated_inline_formula_block"
UI_INLINE_FORMULA_ORIGIN_BBOX_KEY = "ui_inline_formula_origin_bbox"
UI_INLINE_FORMULA_PARENT_LABEL_KEY = "ui_inline_formula_parent_label"
UI_DELETED_INLINE_FORMULA_KEY = "_ui_deleted"

HANWANG_BBOX_AUDIT_KEY = "_hanwang_bbox_audit"

APP_PAYLOAD_KEYS = frozenset()

APP_OWNED_PAYLOAD_KEYS = frozenset({
    PADDLE_BINDING_KEY,
    OCR_TEXT_INVALIDATED_KEY,
    OCR_INVALIDATION_KIND_KEY,
    MANUAL_MERGE_FROM_KEY,
    MANUAL_DRAW_BBOX_KEY,
    UI_GENERATED_INLINE_FORMULA_BLOCK_KEY,
    UI_INLINE_FORMULA_ORIGIN_BBOX_KEY,
    UI_INLINE_FORMULA_PARENT_LABEL_KEY,
    UI_DELETED_INLINE_FORMULA_KEY,
    HANWANG_BBOX_AUDIT_KEY,
    TABLE_TEXT_LAYER_CELLS_KEY,
})

RAW_PAYLOAD_FORBIDDEN_APP_KEYS = APP_OWNED_PAYLOAD_KEYS


def paddle_binding_dict(block: object) -> dict[str, Any]:
    binding = getattr(block, "paddle_binding", None)
    if isinstance(binding, PaddleBinding):
        return binding.to_dict()
    return {}


def set_paddle_binding(block: object, binding: Mapping[str, Any] | PaddleBinding | None) -> None:
    if isinstance(binding, PaddleBinding):
        setattr(block, "paddle_binding", binding)
        return
    setattr(block, "paddle_binding", PaddleBinding.from_dict(dict(binding or {})))


def clear_paddle_binding(block: object) -> None:
    setattr(block, "paddle_binding", None)


def is_ocr_text_invalidated(block: object) -> bool:
    reason = str(getattr(block, "ocr_invalidated_reason", "") or "")
    return bool(reason)


def ocr_invalidation_reason(block: object) -> str:
    return str(getattr(block, "ocr_invalidated_reason", "") or "")


def unknown_app_payload_keys(payload: Mapping[str, Any] | None) -> set[str]:
    return set(dict(payload or {})) - APP_PAYLOAD_KEYS


def validate_app_payload_keys(
    payload: Mapping[str, Any] | None,
    *,
    field: str = "block.app_payload",
) -> None:
    unknown = sorted(unknown_app_payload_keys(payload))
    if unknown:
        raise ValueError(f"{field} contains unregistered app_payload keys: {unknown}")


def strip_runtime_layout_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {})
    for key in RUNTIME_LAYOUT_PAYLOAD_KEYS:
        value.pop(key, None)
    return value


def mark_ocr_text_invalidated(block: object, kind: str | None = None) -> dict[str, Any]:
    reason = str(kind or "layout_changed")
    setattr(block, "ocr_invalidated_reason", reason)
    return {}


def clear_ocr_text_invalidation(block: object) -> dict[str, Any]:
    setattr(block, "ocr_invalidated_reason", "")
    return {}
