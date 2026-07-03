"""Payload boundary constants and typed-state helpers for layout blocks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models import PaddleBinding

RUNTIME_LAYOUT_PAYLOAD_KEYS = frozenset({
    "_layout_line_routes",
    "_route_subblocks",
    "_layout_block_source",
    "_layout_block_ocr_policy",
    "_layout_paddle_parent_index",
    "_layout_manual_route_subblock",
})

APP_PAYLOAD_KEYS = frozenset()

RAW_PAYLOAD_FORBIDDEN_APP_KEYS = frozenset({
    "paddle_binding",
    "ocr_text_invalidated",
    "ocr_invalidation_kind",
    "manual_merge_from",
    "manual_draw_bbox",
    "ui_generated_inline_formula_block",
    "ui_inline_formula_origin_bbox",
    "ui_inline_formula_parent_label",
    "_ui_deleted",
    "_hanwang_bbox_audit",
    "table_text_layer_cells",
})


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
