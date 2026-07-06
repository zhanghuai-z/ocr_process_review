from __future__ import annotations

from app.models import BBox, Block, BlockType, Line, OcrProject, Page
from app.models.layout_projection import replace_page_layout_blocks
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection
from app.models.ocr_observation import (
    find_block_ocr_line_index,
    iter_page_ocr_line_observation_occurrences,
    iter_project_ocr_line_observation_occurrences,
)


def test_ocr_line_occurrences_use_layout_snapshot_order_when_runtime_projection_drifts():
    first_line = Line(text="first", confidence=0.9, bbox=BBox(0, 0, 20, 10))
    second_line = Line(text="second", confidence=0.9, bbox=BBox(0, 30, 20, 10))
    first = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 20),
        order=0,
        lines=[first_line],
    )
    second = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 30, 80, 20),
        order=1,
        lines=[second_line],
    )
    page = Page(image_path="/tmp/page.png", width=120, height=80, blocks=[first, second])
    sync_page_layout_snapshot_from_projection(page, source_engine="test")
    replace_page_layout_blocks(page, [second, first])

    occurrences = list(iter_page_ocr_line_observation_occurrences(page))

    assert [(occ.block, occ.line, occ.block_index) for occ in occurrences] == [
        (first, first_line, 0),
        (second, second_line, 1),
    ]
    assert find_block_ocr_line_index(page, first, first_line) == (0, 0)
    assert find_block_ocr_line_index(page, second, second_line) == (1, 0)


def test_project_ocr_line_occurrences_delegate_to_snapshot_order():
    line = Line(text="text", confidence=0.9, bbox=BBox(0, 0, 20, 10))
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/page.png", width=120, height=80, blocks=[block])
    sync_page_layout_snapshot_from_projection(page, source_engine="test")
    project = OcrProject(name="project", pages=[page])

    occurrences = list(iter_project_ocr_line_observation_occurrences(project))

    assert len(occurrences) == 1
    assert occurrences[0].page is page
    assert occurrences[0].block is block
    assert occurrences[0].line is line
    assert occurrences[0].block_index == 0
