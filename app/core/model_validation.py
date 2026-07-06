"""Runtime model boundary validation."""
from __future__ import annotations

from app.models import Block, BlockOrigin, OcrPolicy, Page
from app.models.layout_block_view import iter_page_layout_block_views


class ModelValidationError(ValueError):
    """Current in-memory model violates application boundaries."""


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
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        validate_block_model(block)


__all__ = [
    "ModelValidationError",
    "validate_block_model",
    "validate_page_model",
]
