"""Build hidden table text-layer geometry for export."""
from __future__ import annotations

from app.core.proof_line_facts import proof_ocr_text
from app.core.raw_ocr_artifact import raw_block_text_values
from app.core.table_text_layer import build_table_text_layer_cells
from app.models import Block, BlockType, Page
from app.models.layout_projection import page_layout_blocks
from app.models.ocr_observation import block_ocr_lines


class TableTextLayerService:
    """Attach export-only table cell geometry to table blocks."""

    def enrich_page(self, page: Page) -> int:
        updated = 0
        for block in page_layout_blocks(page):
            if block.block_type != BlockType.TABLE:
                continue
            html = self._table_html(page, block)
            if not html:
                block.table_text_layer_cells = []
                continue
            cells = build_table_text_layer_cells(
                image_path=page.display_image_path,
                table_bbox=block.bbox.to_dict(),
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
    def _table_html(page: Page, block: Block) -> str:
        candidates: list[str] = []
        lines = block_ocr_lines(block)
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
