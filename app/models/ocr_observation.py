"""Access boundary for OCR line observations.

Application code reads OCR line facts through this boundary so layout blocks do
not own OCR text state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from collections.abc import Iterator

from .layout_block_view import iter_page_layout_block_views
from .ocr_observation_store import (
    ocr_lines_for_block,
    ocr_lines_for_block_uid,
    set_ocr_lines_for_block_uid,
)
from .ocr_text_observation import refresh_line_ocr_text_observation_from_projection
from .page_workflow_status import OCR_AVAILABLE_PAGE_STATUSES
from .project import BBox, Block, Line, OcrProject, Page


@dataclass(frozen=True)
class OcrLineOccurrence:
    page: Page
    block: Block
    line: Line
    block_index: int
    line_index: int


def block_ocr_line_observations(block: Block) -> list[Line]:
    """Return OCR line observations for ``block``."""
    return ocr_lines_for_block(block)


def block_ocr_line_observations_by_uid(block_uid: str) -> list[Line]:
    """Return OCR line observations keyed by stable layout block uid."""
    return ocr_lines_for_block_uid(block_uid)


def block_has_ocr_line_observations(block: Block) -> bool:
    """Return whether OCR observations exist without reading ``Block.lines``."""
    return bool(block_ocr_line_observations(block))


def replace_block_ocr_line_observations(block_uid: str, lines: Iterable[Line]) -> None:
    line_list = list(lines)
    for line in line_list:
        refresh_line_ocr_text_observation_from_projection(line)
    set_ocr_lines_for_block_uid(block_uid, line_list)


def discard_block_ocr_line_projection(block: Block) -> None:
    """Drop the legacy runtime projection without changing uid observations."""
    block.lines = []


def set_ocr_line_bbox(line: Line, bbox: BBox) -> None:
    line.bbox = bbox


def line_ocr_bbox(line: Line) -> BBox:
    return line.bbox


def clear_block_ocr_line_observations(block_uid: str) -> None:
    replace_block_ocr_line_observations(block_uid, [])


def line_belongs_to_block(block: Block, line: Line) -> bool:
    return any(candidate is line for candidate in block_ocr_line_observations(block))


def find_block_ocr_line_index(
    page: Page,
    block: Block,
    line: Line,
) -> tuple[int, int] | None:
    block_index: int | None = None
    for view in iter_page_layout_block_views(page):
        if view.runtime_block is block:
            block_index = view.snapshot_index
            break
    if block_index is None:
        return None
    try:
        line_index = block_ocr_line_observations(block).index(line)
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


def iter_page_ocr_line_observation_occurrences(page: Page) -> Iterator[OcrLineOccurrence]:
    """Yield OCR line observations in layout snapshot order."""
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        for line_index, line in enumerate(block_ocr_line_observations(block)):
            yield OcrLineOccurrence(
                page=page,
                block=block,
                line=line,
                block_index=view.snapshot_index,
                line_index=line_index,
            )


def iter_project_ocr_line_observation_occurrences(
    project: OcrProject,
) -> Iterator[OcrLineOccurrence]:
    """Yield project OCR line observations in layout snapshot order."""
    for page in project.pages:
        yield from iter_page_ocr_line_observation_occurrences(page)


def page_ocr_line_count(page: Page) -> int:
    return sum(1 for _occurrence in iter_page_ocr_line_observation_occurrences(page))


def page_has_ocr_result(page: Page) -> bool:
    return page_ocr_line_count(page) > 0


def project_ocr_line_count(project: OcrProject) -> int:
    return sum(1 for _occurrence in iter_project_ocr_line_observation_occurrences(project))


def project_has_any_ocr_result(project: OcrProject) -> bool:
    return any(page_has_ocr_result(page) for page in project.pages)


def project_all_pages_ocr_done(project: OcrProject) -> bool:
    return bool(project.pages) and all(
        page.status in OCR_AVAILABLE_PAGE_STATUSES
        for page in project.pages
    )


def project_has_any_ocr_done_page(project: OcrProject) -> bool:
    return any(page.status in OCR_AVAILABLE_PAGE_STATUSES for page in project.pages)
