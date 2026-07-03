"""Build layout overlay view data from raw layout artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.core.inline_formula_edit_state import (
    handled_inline_formula_origin_bboxes,
    inline_formula_origin_bbox,
)
from app.core.normalized_layout_artifact import LayoutRegion, LayoutSubregion, normalized_layout_regions
from app.core.ocr_ir import is_formula_marker_token
from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import (
    block_text,
    formula_texts_by_subblock_bbox,
    line_routes_for_block,
)
from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Page


INLINE_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class InlineFormulaOverlay:
    parent_index: int
    parent: LayoutRegion
    subblock: LayoutSubregion
    bbox: BBox


class LayoutOverlayService:
    """Convert raw Paddle layout evidence into UI overlay objects."""

    def ensure_inline_formula_blocks(self, page: Page) -> int:
        """Promote raw inline-formula overlays to editable equation blocks."""
        created = 0
        handled_origins = handled_inline_formula_origin_bboxes(page)
        for overlay in self.iter_inline_formula_overlays(page):
            origin_tuple = overlay.bbox.to_xyxy()
            if origin_tuple in handled_origins:
                continue
            origin = list(origin_tuple)
            if self.has_inline_formula_origin_block(page, origin):
                continue
            page.blocks.append(Block(
                block_type=BlockType.EQUATION,
                bbox=overlay.bbox,
                order=len(page.blocks),
                source=BlockSource.AUTO_LAYOUT,
                source_label="inline_formula",
                origin=BlockOrigin(
                    created_by=BlockSource.AUTO_LAYOUT.value,
                    source_engine="paddleocr-vl",
                    source_label="inline_formula",
                    original_bbox=overlay.bbox,
                    original_kind=BlockType.EQUATION,
                    raw_artifact_uid=page.raw_layout_artifact.uid if page.raw_layout_artifact else "",
                    raw_index=overlay.parent_index,
                ),
            ))
            created += 1
        return created

    def iter_inline_formula_overlays(self, page: Page) -> Iterable[InlineFormulaOverlay]:
        for parent in normalized_layout_regions(page):
            for subblock in parent.subregions:
                label = str(subblock.label or "")
                if normalize_paddle_label(label) != "inline_formula":
                    continue
                bbox = BBox.from_xyxy(*subblock.bbox)
                bbox = bbox.clamp(page.width, page.height)
                formula_text = self.inline_formula_subblock_text(page, parent, bbox)
                if formula_text and is_formula_marker_token(formula_text):
                    continue
                yield InlineFormulaOverlay(parent.index, parent, subblock, bbox)

    def inline_formula_subblock_text(self, page: Page, parent: LayoutRegion, bbox: BBox) -> str:
        target = bbox.clamp(page.width, page.height).to_xyxy()
        marker_text = self.inline_formula_marker_text_from_parent_order(page, parent, target)
        if marker_text:
            return marker_text
        parent_raw = dict(parent.raw or {})
        for route in line_routes_for_block(parent_raw, page.width, page.height):
            for segment in route.get("segments", []):
                if segment.get("kind") != "formula":
                    continue
                segment_bbox = self.bbox_from_route_segment(segment.get("bbox"))
                if segment_bbox is None:
                    continue
                if self.same_inline_formula_route_span(segment_bbox.to_xyxy(), target):
                    return str(segment.get("text") or "")
        for formula_bbox, text in formula_texts_by_subblock_bbox(parent_raw, page.width, page.height).items():
            if self.same_inline_formula_route_span(formula_bbox, target):
                return text
        return ""

    def inline_formula_marker_text_from_parent_order(
        self,
        page: Page,
        parent: LayoutRegion,
        target: tuple[int, int, int, int],
    ) -> str:
        spans = [match.group(0) for match in INLINE_FORMULA_SPAN_RE.finditer(parent.text or block_text(dict(parent.raw or {})))]
        if not spans:
            return ""
        formula_bboxes: list[tuple[int, int, int, int]] = []
        for subblock in parent.subregions:
            label = str(subblock.label or "")
            if normalize_paddle_label(label) != "inline_formula":
                continue
            bbox = BBox.from_xyxy(*subblock.bbox)
            formula_bboxes.append(bbox.clamp(page.width, page.height).to_xyxy())
        formula_bboxes.sort(key=lambda item: (item[1], item[0]))
        for index, formula_bbox in enumerate(formula_bboxes):
            if index >= len(spans):
                break
            if not self.same_inline_formula_route_span(formula_bbox, target):
                continue
            span = spans[index]
            if is_formula_marker_token(span):
                return span
            return ""
        return ""

    @staticmethod
    def same_inline_formula_route_span(
        route_bbox: tuple[int, int, int, int],
        raw_bbox: tuple[int, int, int, int],
    ) -> bool:
        if route_bbox == raw_bbox:
            return True
        if route_bbox[0] != raw_bbox[0] or route_bbox[2] != raw_bbox[2]:
            return False
        overlap = max(0, min(route_bbox[3], raw_bbox[3]) - max(route_bbox[1], raw_bbox[1]))
        denom = max(1, min(route_bbox[3] - route_bbox[1], raw_bbox[3] - raw_bbox[1]))
        return overlap / denom >= 0.5

    @staticmethod
    def has_inline_formula_origin_block(page: Page, origin_bbox: list[int]) -> bool:
        origin_tuple = tuple(origin_bbox)
        for block in page.blocks:
            if inline_formula_origin_bbox(block) == origin_tuple:
                return True
            if normalize_paddle_label(getattr(block, "source_label", "")) != "inline_formula":
                continue
            if block.bbox.to_xyxy() == origin_tuple:
                return True
        return False

    @staticmethod
    def bbox_from_route_segment(value: object) -> BBox | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x1, y1, x2, y2 = (int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox.from_xyxy(x1, y1, x2, y2)

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
