"""Access boundary for OCR line observations.

``Block.lines`` is a compatibility projection. Application code should use
these helpers so OCR observations live behind one boundary instead of being
owned by the layout block model.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from collections.abc import Iterator

from .layout_projection import (
    find_page_layout_block_index,
    iter_page_layout_block_occurrences,
    iter_project_layout_block_occurrences,
    page_layout_blocks,
)
from .ocr_observation_store import ocr_lines_for_block, set_ocr_lines_for_block
from .project import BBox, Block, Line, OcrProject, Page
from .ocr_text_observation import line_ocr_confidence


@dataclass(frozen=True)
class OcrLineOccurrence:
    page: Page
    block: Block
    line: Line
    block_index: int
    line_index: int


def block_ocr_lines(block: Block) -> list[Line]:
    return ocr_lines_for_block(block, block.lines)


def block_ocr_line_count(block: Block) -> int:
    return len(block_ocr_lines(block))


def block_has_ocr_lines(block: Block) -> bool:
    return bool(block_ocr_lines(block))


def block_avg_confidence(block: Block) -> float:
    lines = block_ocr_lines(block)
    if not lines:
        return 0.0
    return sum(line_ocr_confidence(line) for line in lines) / len(lines)


def replace_block_ocr_lines(block: Block, lines: Iterable[Line]) -> None:
    projected = list(lines)
    block.lines = projected
    set_ocr_lines_for_block(block, projected)


def set_ocr_line_bbox(line: Line, bbox: BBox) -> None:
    line.bbox = bbox


def clear_block_ocr_lines(block: Block) -> None:
    replace_block_ocr_lines(block, [])


def append_block_ocr_line(block: Block, line: Line) -> None:
    block_ocr_lines(block).append(line)


def block_ocr_line_at(block: Block, index: int) -> Line:
    return block_ocr_lines(block)[index]


def line_belongs_to_block(block: Block, line: Line) -> bool:
    return any(candidate is line for candidate in block_ocr_lines(block))


def find_block_ocr_line_index(
    page: Page,
    block: Block,
    line: Line,
) -> tuple[int, int] | None:
    block_index = find_page_layout_block_index(page, block)
    if block_index is None:
        return None
    try:
        line_index = block_ocr_lines(block).index(line)
    except ValueError:
        return None
    return block_index, line_index


def find_block_ocr_line_occurrence(
    page: Page,
    block: Block,
    line: Line,
) -> OcrLineOccurrence | None:
    index = find_block_ocr_line_index(page, block, line)
    if index is None:
        return None
    block_index, line_index = index
    return OcrLineOccurrence(
        page=page,
        block=block,
        line=line,
        block_index=block_index,
        line_index=line_index,
    )


def iter_page_ocr_line_occurrences(page: Page) -> Iterator[OcrLineOccurrence]:
    for block_occurrence in iter_page_layout_block_occurrences(page):
        block = block_occurrence.block
        for line_index, line in enumerate(block_ocr_lines(block)):
            yield OcrLineOccurrence(
                page=page,
                block=block,
                line=line,
                block_index=block_occurrence.block_index,
                line_index=line_index,
            )


def iter_project_ocr_line_occurrences(
    project: OcrProject,
) -> Iterator[OcrLineOccurrence]:
    for block_occurrence in iter_project_layout_block_occurrences(project):
        block = block_occurrence.block
        for line_index, line in enumerate(block_ocr_lines(block)):
            yield OcrLineOccurrence(
                page=block_occurrence.page,
                block=block,
                line=line,
                block_index=block_occurrence.block_index,
                line_index=line_index,
            )


def page_ocr_line_count(page: Page) -> int:
    return sum(block_ocr_line_count(block) for block in page_layout_blocks(page))


def page_has_ocr_result(page: Page) -> bool:
    return page_ocr_line_count(page) > 0


def project_ocr_line_count(project: OcrProject) -> int:
    return sum(page_ocr_line_count(page) for page in project.pages)


def project_has_any_ocr_result(project: OcrProject) -> bool:
    return any(page_has_ocr_result(page) for page in project.pages)


def project_all_pages_ocr_done(project: OcrProject) -> bool:
    return bool(project.pages) and all(page.is_ocr_done for page in project.pages)


def project_has_any_ocr_done_page(project: OcrProject) -> bool:
    return any(page.is_ocr_done for page in project.pages)
