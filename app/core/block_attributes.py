"""Structured accessors for Paddle/PP-VL block attributes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.paddle_labels import authoritative_paddle_label, normalize_paddle_label
from app.models import Block, BlockSource, BlockType


def normalize_source_label(label: object) -> str:
    return normalize_paddle_label(label)


@dataclass(frozen=True)
class BlockAttributes:
    """Canonical view of layout attributes for UI/proof/export consumers."""

    block_type: BlockType
    source_label: str = ""
    raw_label: str = ""
    semantic_label: str = ""
    semantic_block_type: BlockType = BlockType.UNKNOWN
    raw_payload: dict[str, Any] = field(default_factory=dict)
    app_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_source_label(self) -> str:
        return normalize_source_label(self.source_label)

    @property
    def normalized_semantic_label(self) -> str:
        return normalize_source_label(self.semantic_label)

    @property
    def display_label(self) -> str:
        semantic = self.normalized_semantic_label
        if semantic and semantic != self.block_type.value:
            return f"{self.block_type.value} · {semantic}"
        return self.block_type.value

    @property
    def is_position_only(self) -> bool:
        return self.normalized_semantic_label in {
            "page_number",
            "number",
            "formula_number",
            "header",
            "footer",
            "sidebar_text",
        }

    def to_export_dict(self) -> dict[str, Any]:
        payload = {
            "block_type": self.block_type.value,
            "source_label": self.source_label,
            "raw_label": self.raw_label,
            "semantic_label": self.semantic_label,
            "semantic_block_type": self.semantic_block_type.value,
        }
        if self.raw_payload:
            payload["raw_payload"] = dict(self.raw_payload)
        return payload


def block_attributes(block: Block) -> BlockAttributes:
    raw_payload = dict(block.raw_payload)
    app_payload = dict(block.app_payload)
    raw_label = authoritative_paddle_label(raw_payload)
    binding = app_payload.get("paddle_binding")
    binding_label = ""
    if isinstance(binding, dict):
        binding_label = str(binding.get("source_label") or binding.get("block_type") or "")
    source_label = normalize_source_label(block.source_label or binding_label or raw_label)
    user_authored_label = block.source in {BlockSource.MANUAL_DRAW, BlockSource.USER_EDITED}
    semantic_label = normalize_source_label(
        (source_label if user_authored_label else raw_label) or source_label or block.block_type.value
    )
    semantic_block_type = BlockType.from_paddle(semantic_label)
    if semantic_block_type == BlockType.UNKNOWN:
        semantic_block_type = block.block_type
    return BlockAttributes(
        block_type=block.block_type,
        source_label=source_label,
        raw_label=raw_label,
        semantic_label=semantic_label,
        semantic_block_type=semantic_block_type,
        raw_payload=raw_payload,
        app_payload=app_payload,
    )


def semantic_block_type(block: Block) -> BlockType:
    return block_attributes(block).semantic_block_type


def semantic_label(block: Block) -> str:
    return block_attributes(block).normalized_semantic_label


def block_display_label(block: Block) -> str:
    return block_attributes(block).display_label


def is_position_only_block(block: Block) -> bool:
    return block_attributes(block).is_position_only
