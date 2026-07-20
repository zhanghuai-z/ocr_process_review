"""Derive read-only layout overlays from an immutable Paddle artifact."""
from __future__ import annotations

import json
from typing import Any

from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import route_subblocks_for_block
from app.core.paddle_response import parsing_records_from_item, result_items
from app.models.geometry import BBox
from app.models.paddle_artifact import PaddleArtifact


class LayoutOverlayService:
    """Build non-editable geometry without consulting adopted layout state."""

    def readonly_layout_overlays(
        self,
        artifact: PaddleArtifact,
        *,
        page_width: int,
        page_height: int,
    ) -> list[tuple[str, BBox]]:
        if not isinstance(artifact, PaddleArtifact):
            raise TypeError("layout overlays require PaddleArtifact")
        if isinstance(page_width, bool) or not isinstance(page_width, int) or page_width <= 0:
            raise ValueError("page_width must be positive")
        if isinstance(page_height, bool) or not isinstance(page_height, int) or page_height <= 0:
            raise ValueError("page_height must be positive")

        overlays: list[tuple[str, BBox]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for record in _artifact_layout_records(artifact):
            for subblock in route_subblocks_for_block(record, page_width, page_height):
                label = str(subblock.get("label") or "")
                if normalize_paddle_label(label) == "inline_formula":
                    continue
                bbox = BBox.from_xyxy(*subblock["bbox"]).clamp(page_width, page_height)
                if bbox.w <= 0 or bbox.h <= 0:
                    continue
                key = (label, bbox.to_xyxy())
                if key in seen:
                    continue
                seen.add(key)
                overlays.append((label or "layout_overlay", bbox))
        return overlays


def _artifact_layout_records(artifact: PaddleArtifact) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(artifact.payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Paddle artifact payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Paddle layout artifact payload must be an object")
    records: list[dict[str, Any]] = []
    for item in result_items(payload, "layoutParsingResults"):
        records.extend(dict(record) for record in parsing_records_from_item(item))
    return tuple(records)


__all__ = ["LayoutOverlayService"]
