"""Build hidden table text-layer geometry for export."""
from __future__ import annotations

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6PrepassArtifact
from app.core.proof_line_facts import proof_ocr_text
from app.core.raw_ocr_artifact import raw_block_text_values
from app.core.table_text_layer import build_table_text_layer_cells
from app.models import Block, BlockType, Page
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_observation import block_ocr_line_observations_by_uid


class TableTextLayerService:
    """Attach export-only table cell geometry to table blocks."""

    def enrich_page(
        self,
        page: Page,
        *,
        prepass: PpOcrV6PrepassArtifact | None = None,
    ) -> int:
        updated = 0
        for view in iter_page_layout_block_views(page):
            if view.block_type != BlockType.TABLE or view.runtime_block is None:
                continue
            block = view.runtime_block
            prepass_items = self._prepass_table_text_items(prepass, view.bbox.to_xyxy())
            if prepass_items:
                block.table_text_layer_cells = prepass_items
                updated += 1
                continue
            html = self._table_html(page, view, block)
            if not html:
                block.table_text_layer_cells = []
                continue
            cells = build_table_text_layer_cells(
                image_path=page.display_image_path,
                table_bbox=view.bbox.to_dict(),
                html=html,
                page_width=page.width,
                page_height=page.height,
            )
            if cells:
                block.table_text_layer_cells = cells
                updated += 1
            else:
                block.table_text_layer_cells = []
        return updated

    @staticmethod
    def _prepass_table_text_items(
        prepass: PpOcrV6PrepassArtifact | None,
        table_bbox: tuple[int, int, int, int],
    ) -> list[dict]:
        """Project PP-OCR physical rows inside a table into export geometry.

        PP-OCR text remains an OCR observation.  It does not replace the VL
        table HTML or become proof text; the projection is persisted only as the
        hidden PDF text-layer aid already owned by the table block.
        """
        if prepass is None:
            return []
        rows: list[tuple[tuple[int, int, int, int], str]] = []
        for prepass_line in prepass.lines:
            clipped = _intersect_xyxy(prepass_line.bbox, table_bbox)
            if clipped is None:
                continue
            center_x = (prepass_line.bbox[0] + prepass_line.bbox[2]) / 2.0
            center_y = (prepass_line.bbox[1] + prepass_line.bbox[3]) / 2.0
            if not (
                table_bbox[0] <= center_x <= table_bbox[2]
                and table_bbox[1] <= center_y <= table_bbox[3]
            ):
                continue
            text = str(prepass_line.text or "").strip()
            if text:
                rows.append((clipped, text))
        if not rows:
            return []

        grouped: list[list[tuple[tuple[int, int, int, int], str]]] = []
        for item in sorted(rows, key=lambda value: (value[0][1], value[0][0])):
            for group in grouped:
                if _same_physical_row(item[0], group[0][0]):
                    group.append(item)
                    break
            else:
                grouped.append([item])

        result: list[dict] = []
        for row_index, group in enumerate(grouped):
            for col_index, (bbox, text) in enumerate(sorted(group, key=lambda value: value[0][0])):
                x1, y1, x2, y2 = bbox
                result.append({
                    "text": text,
                    "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "row": row_index,
                    "col": col_index,
                    "row_span": 1,
                    "col_span": 1,
                    "bbox_source": "ppocrv6_prepass",
                    "bbox_granularity": "table_text_line",
                })
        return result

    @staticmethod
    def _table_html(page: Page, view: LayoutBlockView, block: Block) -> str:
        candidates: list[str] = []
        lines = block_ocr_line_observations_by_uid(view.uid)
        for line in lines:
            ocr_text = proof_ocr_text(line)
            if ocr_text:
                candidates.append(ocr_text)
        candidates.extend(raw_block_text_values(block, ("html", "table_html", "block_content"), page))
        for candidate in candidates:
            text = candidate.strip()
            if "<table" in text.lower() or "<tr" in text.lower():
                return text
        return ""


def _intersect_xyxy(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    bbox = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _same_physical_row(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    overlap = min(left[3], right[3]) - max(left[1], right[1])
    return overlap > 0 and overlap / max(1, min(left[3] - left[1], right[3] - right[1])) >= 0.5
