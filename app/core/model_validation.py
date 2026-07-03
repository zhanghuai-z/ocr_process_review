"""Runtime model boundary validation."""
from __future__ import annotations

from typing import Any, Mapping

from app.core.block_payload import (
    RAW_PAYLOAD_FORBIDDEN_APP_KEYS,
    RUNTIME_LAYOUT_PAYLOAD_KEYS,
    validate_app_payload_keys,
)
from app.models import Block, BlockOrigin, OcrPolicy, Page


class ModelValidationError(ValueError):
    """Current in-memory model violates application boundaries."""


def validate_persistent_block_payloads(
    raw_payload: Mapping[str, Any] | None,
    app_payload: Mapping[str, Any] | None,
    *,
    raw_field: str = "block.raw_payload",
    app_field: str = "block.app_payload",
) -> None:
    raw = dict(raw_payload or {})
    app = dict(app_payload or {})
    for runtime_field in RUNTIME_LAYOUT_PAYLOAD_KEYS:
        if runtime_field in raw:
            raise ModelValidationError(
                f"{raw_field} contains runtime routing data: {runtime_field}"
            )
        if runtime_field in app:
            raise ModelValidationError(
                f"{app_field} contains runtime routing data: {runtime_field}"
            )
    try:
        validate_app_payload_keys(app, field=app_field)
    except ValueError as exc:
        raise ModelValidationError(str(exc)) from exc
    forbidden_app_keys = sorted(set(raw) & RAW_PAYLOAD_FORBIDDEN_APP_KEYS)
    if forbidden_app_keys:
        raise ModelValidationError(
            f"{raw_field} contains app-owned payload keys: {forbidden_app_keys}"
        )


def validate_block_model(block: Block) -> None:
    if not isinstance(block.ocr_policy, OcrPolicy):
        raise ModelValidationError("block.ocr_policy must be OcrPolicy")
    if block.origin is not None and not isinstance(block.origin, BlockOrigin):
        raise ModelValidationError("block.origin must be BlockOrigin")
    validate_persistent_block_payloads(block.raw_payload, {})


def validate_page_model(page: Page) -> None:
    if hasattr(page, "ppvl_parsing_res_list"):
        raise ModelValidationError("Page active model must not expose ppvl_parsing_res_list")
    for block in page.blocks:
        validate_block_model(block)


__all__ = [
    "ModelValidationError",
    "validate_block_model",
    "validate_page_model",
    "validate_persistent_block_payloads",
]
