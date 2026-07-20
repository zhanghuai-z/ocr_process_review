"""Structured attributes for immutable layout blocks.

Layout semantics are derived from the adopted snapshot.  Vendor payloads and
runtime projections are intentionally outside this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.paddle_labels import normalize_paddle_label
from app.models.enums import BlockType
from app.models.layout_snapshot import LayoutBlockSnapshot


def normalize_source_label(label: object) -> str:
    return normalize_paddle_label(label)


@dataclass(frozen=True, slots=True)
class BlockAttributes:
    """Canonical, read-only attributes of one adopted layout block."""

    block_type: BlockType
    source_label: str = ""
    current_label: str = ""
    origin_label: str = ""
    semantic_label: str = ""
    semantic_block_type: BlockType = BlockType.UNKNOWN

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
        return {
            "block_type": self.block_type.value,
            "source_label": self.source_label,
            "current_label": self.current_label,
            "origin_label": self.origin_label,
            "semantic_label": self.semantic_label,
            "semantic_block_type": self.semantic_block_type.value,
        }


def block_attributes(block: LayoutBlockSnapshot) -> BlockAttributes:
    """Return attributes for an immutable ``LayoutBlockSnapshot`` only."""

    if not isinstance(block, LayoutBlockSnapshot):
        raise TypeError("block attributes require LayoutBlockSnapshot")
    current_label = normalize_source_label(block.source_label or block.block_type.value)
    origin_label = normalize_source_label(block.origin.vendor_label)
    semantic_label = current_label or origin_label or block.block_type.value
    semantic_block_type = map_paddle_label_to_block_type(semantic_label)
    if semantic_block_type is BlockType.UNKNOWN:
        semantic_block_type = block.block_type
    return BlockAttributes(
        block_type=block.block_type,
        source_label=current_label,
        current_label=current_label,
        origin_label=origin_label,
        semantic_label=normalize_source_label(semantic_label),
        semantic_block_type=semantic_block_type,
    )


def semantic_block_type(block: LayoutBlockSnapshot) -> BlockType:
    return block_attributes(block).semantic_block_type


def semantic_label(block: LayoutBlockSnapshot) -> str:
    return block_attributes(block).normalized_semantic_label


def current_source_label(
    block: LayoutBlockSnapshot,
    *,
    fallback_to_type: bool = True,
) -> str:
    if not isinstance(block, LayoutBlockSnapshot):
        raise TypeError("current source label requires LayoutBlockSnapshot")
    label = normalize_source_label(block.source_label)
    if label or not fallback_to_type:
        return label
    return normalize_source_label(block.block_type.value)


def origin_source_label(block: LayoutBlockSnapshot) -> str:
    if not isinstance(block, LayoutBlockSnapshot):
        raise TypeError("origin source label requires LayoutBlockSnapshot")
    return normalize_source_label(block.origin.vendor_label)


def route_source_label(block: LayoutBlockSnapshot) -> str:
    """Return the explicit route label carried by the adopted snapshot."""

    attrs = block_attributes(block)
    return (
        current_source_label(block, fallback_to_type=False)
        or attrs.origin_label
        or attrs.semantic_label
        or normalize_source_label(block.block_type.value)
    )


def block_display_label(block: LayoutBlockSnapshot) -> str:
    return block_attributes(block).display_label


def is_position_only_block(block: LayoutBlockSnapshot) -> bool:
    return block_attributes(block).is_position_only


__all__ = [
    "BlockAttributes",
    "block_attributes",
    "block_display_label",
    "current_source_label",
    "is_position_only_block",
    "normalize_source_label",
    "origin_source_label",
    "route_source_label",
    "semantic_block_type",
    "semantic_label",
]
