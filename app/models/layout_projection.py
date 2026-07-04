"""Access boundary for the current layout block projection.

``Page.blocks`` is still the physical runtime storage used by existing UI,
OCR, and export code.  Application services should go through this module so
the current projection can later move behind ``LayoutSnapshot`` without another
broad rewrite.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .project import Block, OcrProject, Page


@dataclass(frozen=True)
class LayoutBlockOccurrence:
    page: Page
    block: Block
    block_index: int


def page_layout_blocks(page: Page) -> list[Block]:
    return page.blocks


def page_layout_block_count(page: Page) -> int:
    return len(page_layout_blocks(page))


def page_has_layout_blocks(page: Page) -> bool:
    return bool(page_layout_blocks(page))


def replace_page_layout_blocks(page: Page, blocks: Iterable[Block]) -> None:
    page.blocks = list(blocks)


def append_page_layout_block(page: Page, block: Block) -> None:
    page_layout_blocks(page).append(block)


def find_page_layout_block_index(page: Page, block: Block) -> int | None:
    try:
        return page_layout_blocks(page).index(block)
    except ValueError:
        return None


def block_belongs_to_page(page: Page, block: Block) -> bool:
    return find_page_layout_block_index(page, block) is not None


def iter_page_layout_block_occurrences(page: Page) -> Iterator[LayoutBlockOccurrence]:
    for block_index, block in enumerate(page_layout_blocks(page)):
        yield LayoutBlockOccurrence(
            page=page,
            block=block,
            block_index=block_index,
        )


def iter_project_layout_block_occurrences(
    project: OcrProject,
) -> Iterator[LayoutBlockOccurrence]:
    for page in project.pages:
        yield from iter_page_layout_block_occurrences(page)
