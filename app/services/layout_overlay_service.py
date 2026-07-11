"""Build read-only layout overlay view data from raw layout artifacts."""
from __future__ import annotations

from app.core.normalized_layout_artifact import normalized_layout_regions
from app.core.paddle_labels import normalize_paddle_label
from app.models import BBox, Page


class LayoutOverlayService:
    """Convert non-editable Paddle subregions into UI overlay objects."""

    def readonly_layout_overlays(self, page: Page) -> list[tuple[str, BBox]]:
        overlays: list[tuple[str, BBox]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for parent in normalized_layout_regions(page):
            for subblock in parent.subregions:
                label = str(subblock.label or "")
                if normalize_paddle_label(label) == "inline_formula":
                    continue
                bbox = BBox.from_xyxy(*subblock.bbox)
                bbox = bbox.clamp(page.width, page.height)
                key = (label, bbox.to_xyxy())
                if key in seen:
                    continue
                seen.add(key)
                overlays.append((label or "inline_formula", bbox))
        return overlays
