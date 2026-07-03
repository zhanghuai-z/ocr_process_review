from __future__ import annotations

from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import Page


def test_normalized_layout_artifact_exposes_region_and_subregion_facts():
    page = Page(image_path="", width=300, height=180)
    artifact = set_paddle_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [10, 20, 260, 80],
            "block_content": "甲 $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "inline_formula", "block_bbox": [60, 25, 95, 60]},
                {"block_label": "table_region", "block_bbox": [120, 25, 180, 60]},
            ],
        }
    ])

    normalized = normalized_layout_artifact_from_page(page)

    assert normalized.artifact_uid == artifact.uid
    assert normalized.page_uid == page.uid
    assert normalized.engine == "paddleocr-vl"
    assert normalized.engine_version == "1.6"
    assert normalized.page_width == 300
    assert normalized.page_height == 180
    assert len(normalized.regions) == 1

    region = normalized.regions[0]
    assert region.index == 0
    assert region.label == "text"
    assert region.bbox == (10, 20, 260, 80)
    assert region.text == "甲 $ A $ 乙"
    assert len(region.subregions) == 2
    assert [(item.label, item.bbox) for item in region.subregions] == [
        ("inline_formula", (60, 25, 95, 60)),
        ("table_region", (120, 25, 180, 60)),
    ]
