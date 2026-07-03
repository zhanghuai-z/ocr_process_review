"""Build layout overlay view data from raw layout artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.core.bbox_extraction import bbox_from_variant
from app.core.inline_formula_edit_state import (
    handled_inline_formula_origin_bboxes,
    inline_formula_origin_bbox,
)
from app.core.ocr_ir import is_formula_marker_token
from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import (
    ROUTE_SUBBLOCKS_FIELD,
    block_text,
    formula_texts_by_subblock_bbox,
    line_routes_for_block,
)
from app.core.raw_ocr_artifact import raw_layout_records
from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Page


INLINE_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class InlineFormulaOverlay:
    parent_index: int
    parent: dict
    subblock: dict
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
        for parent_index, parent in enumerate(raw_layout_records(page)):
            subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
            if not isinstance(subblocks, list):
                continue
            for subblock in subblocks:
                if not isinstance(subblock, dict):
                    continue
                label = str(
                    subblock.get("block_label")
                    or subblock.get("label")
                    or subblock.get("type")
                    or ""
                )
                if normalize_paddle_label(label) != "inline_formula":
                    continue
                bbox = bbox_from_variant(
                    subblock.get("block_bbox") or subblock.get("bbox") or subblock.get("coordinate"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if bbox is None or bbox.area <= 0:
                    continue
                bbox = bbox.clamp(page.width, page.height)
                formula_text = self.inline_formula_subblock_text(page, parent, bbox)
                if formula_text and is_formula_marker_token(formula_text):
                    continue
                yield InlineFormulaOverlay(parent_index, parent, subblock, bbox)

    def inline_formula_subblock_text(self, page: Page, parent: dict, bbox: BBox) -> str:
        target = bbox.clamp(page.width, page.height).to_xyxy()
        marker_text = self.inline_formula_marker_text_from_parent_order(page, parent, target)
        if marker_text:
            return marker_text
        for route in line_routes_for_block(parent, page.width, page.height):
            for segment in route.get("segments", []):
                if segment.get("kind") != "formula":
                    continue
                segment_bbox = bbox_from_variant(
                    segment.get("bbox"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if segment_bbox is None:
                    continue
                if self.same_inline_formula_route_span(segment_bbox.to_xyxy(), target):
                    return str(segment.get("text") or "")
        for formula_bbox, text in formula_texts_by_subblock_bbox(parent, page.width, page.height).items():
            if self.same_inline_formula_route_span(formula_bbox, target):
                return text
        return ""

    def inline_formula_marker_text_from_parent_order(
        self,
        page: Page,
        parent: dict,
        target: tuple[int, int, int, int],
    ) -> str:
        spans = [match.group(0) for match in INLINE_FORMULA_SPAN_RE.finditer(block_text(parent))]
        if not spans:
            return ""
        subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
        if not isinstance(subblocks, list):
            return ""
        formula_bboxes: list[tuple[int, int, int, int]] = []
        for subblock in subblocks:
            if not isinstance(subblock, dict):
                continue
            label = str(
                subblock.get("block_label")
                or subblock.get("label")
                or subblock.get("type")
                or ""
            )
            if normalize_paddle_label(label) != "inline_formula":
                continue
            bbox = bbox_from_variant(
                subblock.get("block_bbox") or subblock.get("bbox") or subblock.get("coordinate"),
                max_w=page.width,
                max_h=page.height,
            )
            if bbox is None or bbox.area <= 0:
                continue
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

    def readonly_layout_overlays(self, page: Page) -> list[tuple[str, BBox]]:
        overlays: list[tuple[str, BBox]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for parent in raw_layout_records(page):
            subblocks = parent.get(ROUTE_SUBBLOCKS_FIELD)
            if not isinstance(subblocks, list):
                continue
            for subblock in subblocks:
                if not isinstance(subblock, dict):
                    continue
                label = str(
                    subblock.get("block_label")
                    or subblock.get("label")
                    or subblock.get("type")
                    or ""
                )
                if normalize_paddle_label(label) == "inline_formula":
                    continue
                bbox = bbox_from_variant(
                    subblock.get("block_bbox") or subblock.get("bbox") or subblock.get("coordinate"),
                    max_w=page.width,
                    max_h=page.height,
                )
                if bbox is None or bbox.area <= 0:
                    continue
                bbox = bbox.clamp(page.width, page.height)
                key = (label, bbox.to_xyxy())
                if key in seen:
                    continue
                seen.add(key)
                overlays.append((label or "inline_formula", bbox))
        return overlays
