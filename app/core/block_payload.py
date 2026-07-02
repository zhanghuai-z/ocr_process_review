"""Shared keys/helpers for app-owned block.app_payload entries."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models import PaddleBinding

PADDLE_BINDING_KEY = "paddle_binding"
PADDLE_BLOCK_LABEL_KEY = "block_label"
PADDLE_BLOCK_BBOX_KEY = "block_bbox"
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

APP_PAYLOAD_KEYS = frozenset({
    PADDLE_BINDING_KEY,
    PADDLE_BLOCK_LABEL_KEY,
    PADDLE_BLOCK_BBOX_KEY,
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

RAW_PAYLOAD_FORBIDDEN_APP_KEYS = APP_PAYLOAD_KEYS - {
    PADDLE_BLOCK_LABEL_KEY,
    PADDLE_BLOCK_BBOX_KEY,
}


def app_payload_dict(block: object) -> dict[str, Any]:
    payload = getattr(block, "app_payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def payload_get(block: object, key: str, default: Any = None) -> Any:
    return app_payload_dict(block).get(key, default)


def payload_bool(block: object, key: str) -> bool:
    return bool(payload_get(block, key))


def paddle_binding_dict(block: object) -> dict[str, Any]:
    binding = getattr(block, "paddle_binding", None)
    if isinstance(binding, PaddleBinding):
        return binding.to_dict()
    legacy = app_payload_dict(block).get(PADDLE_BINDING_KEY)
    return dict(legacy) if isinstance(legacy, dict) else {}


def set_paddle_binding(block: object, binding: Mapping[str, Any] | PaddleBinding | None) -> None:
    if isinstance(binding, PaddleBinding):
        setattr(block, "paddle_binding", binding)
        return
    setattr(block, "paddle_binding", PaddleBinding.from_dict(dict(binding or {})))


def clear_paddle_binding(block: object) -> None:
    setattr(block, "paddle_binding", None)


def set_payload_entries(block: object, entries: Mapping[str, Any]) -> dict[str, Any]:
    payload = app_payload_dict(block)
    payload.update(dict(entries))
    validate_app_payload_keys(payload)
    setattr(block, "app_payload", payload)
    return payload


def clear_payload_entries(block: object, keys: object) -> dict[str, Any]:
    if isinstance(keys, str):
        remove_keys = {keys}
    else:
        remove_keys = set(keys)
    payload = app_payload_dict(block)
    for key in remove_keys:
        payload.pop(key, None)
    validate_app_payload_keys(payload)
    setattr(block, "app_payload", payload)
    return payload


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
    entries: dict[str, Any] = {OCR_TEXT_INVALIDATED_KEY: True}
    if kind is not None:
        entries[OCR_INVALIDATION_KIND_KEY] = kind
    return set_payload_entries(block, entries)
