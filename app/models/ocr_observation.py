"""Access boundary for OCR lines currently attached to layout blocks.

Block.lines is a transitional storage detail.  Application code should use
these helpers so OCR observations can move out of the layout block model
without another broad rewrite.
"""
from __future__ import annotations

from collections.abc import Iterable

from .project import Block, Line


def block_ocr_lines(block: Block) -> list[Line]:
    return block.lines


def block_ocr_line_count(block: Block) -> int:
    return len(block_ocr_lines(block))


def block_has_ocr_lines(block: Block) -> bool:
    return bool(block_ocr_lines(block))


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
