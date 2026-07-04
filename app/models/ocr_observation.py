"""Access boundary for OCR lines currently attached to layout blocks.

Block.lines is a transitional storage detail.  Application code should use
these helpers so OCR observations can move out of the layout block model
without another broad rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from collections.abc import Iterator

from .project import Block, Line, OcrProject, Page


@dataclass(frozen=True)
class OcrLineOccurrence:
    page: Page
    block: Block
    line: Line
    block_index: int
    line_index: int


def block_ocr_lines(block: Block) -> list[Line]:
    return block.lines


def block_ocr_line_count(block: Block) -> int:
    return len(block_ocr_lines(block))


def block_has_ocr_lines(block: Block) -> bool:
    return bool(block_ocr_lines(block))


def block_avg_confidence(block: Block) -> float:
    lines = block_ocr_lines(block)
    if not lines:
        return 0.0
    return sum(line.confidence for line in lines) / len(lines)


def replace_block_ocr_lines(block: Block, lines: Iterable[Line]) -> None:
    block.lines = list(lines)


def clear_block_ocr_lines(block: Block) -> None:
    block.lines = []


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
    try:
        block_index = page.blocks.index(block)
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
    for block_index, block in enumerate(page.blocks):
        for line_index, line in enumerate(block_ocr_lines(block)):
            yield OcrLineOccurrence(
                page=page,
                block=block,
                line=line,
                block_index=block_index,
                line_index=line_index,
            )


def iter_project_ocr_line_occurrences(
    project: OcrProject,
) -> Iterator[OcrLineOccurrence]:
    for page in project.pages:
        yield from iter_page_ocr_line_occurrences(page)


def page_ocr_line_count(page: Page) -> int:
    return sum(block_ocr_line_count(block) for block in page.blocks)


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
