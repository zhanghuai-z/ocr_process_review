"""Build hidden table text-layer geometry for export."""
from __future__ import annotations

from app.core.block_payload import set_payload_entries
from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY, build_table_text_layer_cells
from app.models import Block, BlockType, Page


class TableTextLayerService:
    """Attach export-only table cell geometry to table blocks."""

    def enrich_page(self, page: Page) -> int:
        updated = 0
        for block in page.blocks:
            if block.block_type != BlockType.TABLE:
                continue
            html = self._table_html(block)
            if not html:
                block.app_payload.pop(TABLE_TEXT_LAYER_CELLS_KEY, None)
                continue
            cells = build_table_text_layer_cells(
                image_path=page.display_image_path,
                table_bbox=block.bbox.to_dict(),
                html=html,
                page_width=page.width,
                page_height=page.height,
            )
            if cells:
                set_payload_entries(block, {TABLE_TEXT_LAYER_CELLS_KEY: cells})
                updated += 1
            else:
                block.app_payload.pop(TABLE_TEXT_LAYER_CELLS_KEY, None)
        return updated

    @staticmethod
    def _table_html(block: Block) -> str:
        candidates: list[str] = []
        candidates.extend(line.text for line in block.lines if line.text)
        candidates.extend(line.ocr_text for line in block.lines if line.ocr_text)
        if isinstance(block.raw_payload, dict):
            candidates.extend(
                str(value)
                for key in ("html", "table_html", "block_content")
                if (value := block.raw_payload.get(key))
            )
        for candidate in candidates:
            text = candidate.strip()
            if "<table" in text.lower() or "<tr" in text.lower():
                return text
        return ""
