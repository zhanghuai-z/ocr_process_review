from __future__ import annotations

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.models import BBox, Block, BlockType, Page
from app.services.table_text_layer_service import TableTextLayerService


def test_table_text_layer_prefers_ppocr_physical_rows_over_equal_grid():
    table = Block(block_type=BlockType.TABLE, bbox=BBox(100, 100, 300, 160), order=0)
    page = Page(image_path="/tmp/table-prepass.png", width=600, height=400, blocks=[table])
    prepass = PpOcrV6PrepassArtifact(
        page_uid=page.uid,
        run_id="ppocr-table",
        lines=(
            PpOcrV6LineHint(index=0, text="A", bbox=(120, 120, 150, 145), words=()),
            PpOcrV6LineHint(index=1, text="B", bbox=(300, 118, 345, 146), words=()),
            PpOcrV6LineHint(index=2, text="10.2%", bbox=(285, 190, 360, 220), words=()),
            PpOcrV6LineHint(index=3, text="outside", bbox=(450, 300, 550, 330), words=()),
        ),
    )

    updated = TableTextLayerService().enrich_page(page, prepass=prepass)

    assert updated == 1
    assert [item["text"] for item in table.table_text_layer_cells] == ["A", "B", "10.2%"]
    assert [item["bbox"] for item in table.table_text_layer_cells] == [
        {"x": 120, "y": 120, "w": 30, "h": 25},
        {"x": 300, "y": 118, "w": 45, "h": 28},
        {"x": 285, "y": 190, "w": 75, "h": 30},
    ]
    assert [(item["row"], item["col"]) for item in table.table_text_layer_cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert all(item["bbox_source"] == "ppocrv6_prepass" for item in table.table_text_layer_cells)
    assert all(item["bbox_granularity"] == "table_text_line" for item in table.table_text_layer_cells)
