"""Build layout overlay view data from raw layout artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.core.inline_formula_edit_state import (
    handled_inline_formula_origin_bboxes,
    inline_formula_origin_bbox,
)
from app.core.normalized_layout_artifact import (
    LayoutRegion,
    LayoutSubregion,
    layout_region_route_record,
    normalized_layout_regions,
)
from app.core.ocr_ir import is_formula_marker_token
from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import (
    block_text,
    formula_texts_by_subblock_bbox,
)
from app.models import (
    BBox,
    BlockOrigin,
    BlockSource,
    BlockType,
    LayoutBlockSnapshot,
    LayoutSnapshot,
    OcrPolicy,
    Page,
)
from app.models.layout_block_view import current_layout_snapshot, iter_page_layout_block_views
from app.models.layout_snapshot_projection import replace_page_layout_projection_from_snapshot
from app.models.layout_snapshot_store import set_layout_snapshot_for_page
from app.services.layout_routing_plan import routing_plan_for_block_record


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
        created_origins: set[tuple[int, int, int, int]] = set()
        snapshot = current_layout_snapshot(page)
        next_blocks = list(snapshot.blocks)
        for overlay in self.iter_inline_formula_overlays(page):
            origin_tuple = overlay.bbox.to_xyxy()
            if origin_tuple in handled_origins:
                continue
            if origin_tuple in created_origins:
                continue
            origin = list(origin_tuple)
            if self.has_inline_formula_origin_block(page, origin):
                continue
            next_blocks.append(self._inline_formula_snapshot_block(page, overlay, order=len(next_blocks)))
            created_origins.add(origin_tuple)
            created += 1
        if created:
            next_snapshot = LayoutSnapshot(
                page_uid=snapshot.page_uid,
                artifact_uid=snapshot.artifact_uid,
                source_engine="layout_overlay_service",
                source_run_id="",
                blocks=tuple(next_blocks),
            )
            set_layout_snapshot_for_page(page, next_snapshot)
            replace_page_layout_projection_from_snapshot(page, next_snapshot)
        return created

    @staticmethod
    def _inline_formula_snapshot_block(
        page: Page,
        overlay: InlineFormulaOverlay,
        *,
        order: int,
    ) -> LayoutBlockSnapshot:
        return LayoutBlockSnapshot(
            block_type=BlockType.EQUATION,
            bbox=overlay.bbox,
            order=order,
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
            ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
        )

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
        parent_raw = layout_region_route_record(parent)
        plan = routing_plan_for_block_record(parent_raw, page.width, page.height)
        for route in plan.lines:
            for segment in route.segments:
                if segment.kind != "formula":
                    continue
                segment_bbox = BBox.from_xyxy(*segment.bbox)
                if self.same_inline_formula_route_span(segment_bbox.to_xyxy(), target):
                    return segment.text
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
        for view in iter_page_layout_block_views(page):
            block = view.runtime_block
            if block is not None and inline_formula_origin_bbox(block) == origin_tuple:
                return True
            if normalize_paddle_label(view.source_label) != "inline_formula":
                continue
            if view.bbox.to_xyxy() == origin_tuple:
                return True
        return False

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
