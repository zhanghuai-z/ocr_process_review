"""Runtime model boundary validation."""
from __future__ import annotations

from typing import Any, Mapping

from app.models import Block, BlockOrigin, OcrPolicy, Page


class ModelValidationError(ValueError):
    """Current in-memory model violates application boundaries."""


RUNTIME_LAYOUT_PAYLOAD_KEYS = frozenset({
    "_layout_line_routes",
    "_route_subblocks",
    "_layout_block_source",
    "_layout_block_ocr_policy",
    "_layout_paddle_parent_index",
    "_layout_manual_route_subblock",
})

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
    if app:
        unknown = sorted(app)
        raise ModelValidationError(f"{app_field} contains retired app payload: {unknown}")
    if raw:
        raise ModelValidationError(f"{raw_field} contains retired raw payload")


def validate_block_model(block: Block) -> None:
    if not isinstance(block.ocr_policy, OcrPolicy):
        raise ModelValidationError("block.ocr_policy must be OcrPolicy")
    if block.origin is not None and not isinstance(block.origin, BlockOrigin):
        raise ModelValidationError("block.origin must be BlockOrigin")
    if hasattr(block, "raw_payload"):
        raise ModelValidationError("Block active model must not expose raw_payload")


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
