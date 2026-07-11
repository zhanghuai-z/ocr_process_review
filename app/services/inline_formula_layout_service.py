"""Adopt Paddle inline-formula subregions into layout truth."""
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
from app.core.paddle_line_routing import block_text, formula_texts_by_subblock_bbox
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
class InlineFormulaRegion:
    parent_index: int
    parent: LayoutRegion
    subregion: LayoutSubregion
    bbox: BBox


class InlineFormulaLayoutService:
    """Promote normalized inline formulas at the layout-adoption boundary."""

    def adopt_page(self, page: Page) -> int:
        if page.raw_layout_artifact is None:
            return 0
        snapshot = current_layout_snapshot(page)
        handled_origins = handled_inline_formula_origin_bboxes(page)
        existing_origins = self._existing_origins(page)
        next_blocks = list(snapshot.blocks)
        created = 0
        for region in self.iter_regions(page):
            origin = region.bbox.to_xyxy()
            if origin in handled_origins or origin in existing_origins:
                continue
            next_blocks.append(self._snapshot_block(page, region, order=len(next_blocks)))
            existing_origins.add(origin)
            created += 1
        if not created:
            return 0
        next_snapshot = LayoutSnapshot(
            page_uid=snapshot.page_uid,
            artifact_uid=snapshot.artifact_uid,
            source_engine=snapshot.source_engine,
            source_run_id=snapshot.source_run_id,
            blocks=tuple(next_blocks),
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        replace_page_layout_projection_from_snapshot(page, next_snapshot)
        return created

    def iter_regions(self, page: Page) -> Iterable[InlineFormulaRegion]:
        for parent in normalized_layout_regions(page):
            for subregion in parent.subregions:
                if normalize_paddle_label(subregion.label) != "inline_formula":
                    continue
                bbox = BBox.from_xyxy(*subregion.bbox).clamp(page.width, page.height)
                formula_text = self._subregion_text(page, parent, bbox)
                if formula_text and is_formula_marker_token(formula_text):
                    continue
                yield InlineFormulaRegion(parent.index, parent, subregion, bbox)

    @staticmethod
    def _snapshot_block(
        page: Page,
        region: InlineFormulaRegion,
        *,
        order: int,
    ) -> LayoutBlockSnapshot:
        return LayoutBlockSnapshot(
            block_type=BlockType.EQUATION,
            bbox=region.bbox,
            order=order,
            source_label="inline_formula",
            origin=BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                source_engine="paddleocr-vl",
                source_label="inline_formula",
                original_bbox=region.bbox,
                original_kind=BlockType.EQUATION,
                raw_artifact_uid=page.raw_layout_artifact.uid if page.raw_layout_artifact else "",
                raw_index=region.parent_index,
            ),
            ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
        )

    def _subregion_text(self, page: Page, parent: LayoutRegion, bbox: BBox) -> str:
        target = bbox.to_xyxy()
        marker_text = self._marker_text_from_parent_order(page, parent, target)
        if marker_text:
            return marker_text
        parent_raw = layout_region_route_record(parent)
        plan = routing_plan_for_block_record(parent_raw, page.width, page.height)
        for route in plan.lines:
            for segment in route.segments:
                if segment.kind != "formula":
                    continue
                if self._same_route_span(segment.bbox, target):
                    return segment.text
        for formula_bbox, text in formula_texts_by_subblock_bbox(
            parent_raw,
            page.width,
            page.height,
        ).items():
            if self._same_route_span(formula_bbox, target):
                return text
        return ""

    def _marker_text_from_parent_order(
        self,
        page: Page,
        parent: LayoutRegion,
        target: tuple[int, int, int, int],
    ) -> str:
        source_text = parent.text or block_text(dict(parent.raw or {}))
        spans = [match.group(0) for match in INLINE_FORMULA_SPAN_RE.finditer(source_text)]
        formula_bboxes = sorted(
            (
                BBox.from_xyxy(*subregion.bbox).clamp(page.width, page.height).to_xyxy()
                for subregion in parent.subregions
                if normalize_paddle_label(subregion.label) == "inline_formula"
            ),
            key=lambda item: (item[1], item[0]),
        )
        for index, formula_bbox in enumerate(formula_bboxes):
            if index >= len(spans) or not self._same_route_span(formula_bbox, target):
                continue
            return spans[index] if is_formula_marker_token(spans[index]) else ""
        return ""

    @staticmethod
    def _same_route_span(
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
    def _existing_origins(page: Page) -> set[tuple[int, int, int, int]]:
        origins: set[tuple[int, int, int, int]] = set()
        for view in iter_page_layout_block_views(page):
            block = view.runtime_block
            if block is not None:
                origin = inline_formula_origin_bbox(block)
                if origin is not None:
                    origins.add(origin)
            if normalize_paddle_label(view.source_label) == "inline_formula":
                origins.add(view.bbox.to_xyxy())
        return origins


__all__ = ["InlineFormulaLayoutService", "InlineFormulaRegion"]
