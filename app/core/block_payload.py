"""Shared keys/helpers for app-owned block.app_payload entries."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PADDLE_BINDING_KEY = "paddle_binding"
PADDLE_BLOCK_LABEL_KEY = "block_label"
PADDLE_BLOCK_BBOX_KEY = "block_bbox"

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
    OCR_TEXT_INVALIDATED_KEY,
    OCR_INVALIDATION_KIND_KEY,
    MANUAL_MERGE_FROM_KEY,
    MANUAL_DRAW_BBOX_KEY,
    UI_GENERATED_INLINE_FORMULA_BLOCK_KEY,
    UI_INLINE_FORMULA_ORIGIN_BBOX_KEY,
    UI_INLINE_FORMULA_PARENT_LABEL_KEY,
    UI_DELETED_INLINE_FORMULA_KEY,
    HANWANG_BBOX_AUDIT_KEY,
    "_layout_block_source",
    "_layout_block_recognizable",
    "_layout_paddle_parent_index",
    "_layout_manual_route_subblock",
    "_layout_line_routes",
    "_route_subblocks",
})


def app_payload_dict(block: object) -> dict[str, Any]:
    payload = getattr(block, "app_payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def payload_get(block: object, key: str, default: Any = None) -> Any:
    return app_payload_dict(block).get(key, default)


def payload_bool(block: object, key: str) -> bool:
    return bool(payload_get(block, key))


def set_payload_entries(block: object, entries: Mapping[str, Any]) -> dict[str, Any]:
    payload = app_payload_dict(block)
    payload.update(dict(entries))
    setattr(block, "app_payload", payload)
    return payload


def mark_ocr_text_invalidated(block: object, kind: str | None = None) -> dict[str, Any]:
    entries: dict[str, Any] = {OCR_TEXT_INVALIDATED_KEY: True}
    if kind is not None:
        entries[OCR_INVALIDATION_KIND_KEY] = kind
    return set_payload_entries(block, entries)


def split_legacy_raw_payload(
    raw_payload: Mapping[str, Any] | None,
    app_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move app-owned keys out of legacy raw payloads.

    Existing project files stored routing/UI state in ``block.raw_payload``.
    New code keeps vendor facts in ``raw_payload`` and app state in
    ``app_payload``. Explicit ``app_payload`` values win over legacy values.
    """
    raw = dict(raw_payload or {})
    app = dict(app_payload or {})
    for key in list(raw):
        if key in APP_PAYLOAD_KEYS:
            app.setdefault(key, raw.pop(key))
    return raw, app
