"""Typed OCR dispatch plan for page text-recognition work."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.core.ocr_dispatch_policy import should_dispatch_to_text_ocr
from app.models import BBox, Block, Page
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views


@dataclass(frozen=True)
class DispatchBlock:
    """A layout block plus its stable page-local position in the dispatch plan."""

    index: int
    block: Block
    view: LayoutBlockView
    reason: str

    @property
    def bbox(self) -> BBox:
        return self.view.bbox

    @property
    def order(self) -> int:
        return self.view.order


@dataclass(frozen=True)
class DispatchPlan:
    """Read model for deciding which page blocks enter text OCR."""

    page: Page
    text_blocks: tuple[DispatchBlock, ...]
    blocked_blocks: tuple[DispatchBlock, ...]

    @property
    def total_text_blocks(self) -> int:
        return len(self.text_blocks)

    @property
    def has_text_work(self) -> bool:
        return bool(self.text_blocks)

    @property
    def text_block_models(self) -> tuple[Block, ...]:
        return tuple(target.block for target in self.text_blocks)

    @property
    def blocker_block_models(self) -> tuple[Block, ...]:
        return tuple(target.block for target in self.blocked_blocks)


def build_text_ocr_dispatch_plan(page: Page) -> DispatchPlan:
    """Build the authoritative text OCR dispatch view for a page."""
    text_blocks: list[DispatchBlock] = []
    blocked_blocks: list[DispatchBlock] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        if should_dispatch_to_text_ocr(view):
            text_blocks.append(
                DispatchBlock(
                    index=view.snapshot_index,
                    block=block,
                    view=view,
                    reason="policy:text_ocr",
                )
            )
        else:
            blocked_blocks.append(
                DispatchBlock(
                    index=view.snapshot_index,
                    block=block,
                    view=view,
                    reason=f"policy:{view.ocr_policy.value}",
                )
            )
    return DispatchPlan(
        page=page,
        text_blocks=tuple(text_blocks),
        blocked_blocks=tuple(blocked_blocks),
    )


def iter_text_ocr_blocks(pages: Iterable[Page]) -> Iterator[Block]:
    """Yield text OCR blocks for pages using the dispatch plan boundary."""
    for page in pages:
        yield from build_text_ocr_dispatch_plan(page).text_block_models


def count_text_ocr_blocks(pages: Iterable[Page]) -> int:
    """Count text OCR blocks for pages using the dispatch plan boundary."""
    return sum(1 for _ in iter_text_ocr_blocks(pages))


__all__ = [
    "DispatchBlock",
    "DispatchPlan",
    "build_text_ocr_dispatch_plan",
    "count_text_ocr_blocks",
    "iter_text_ocr_blocks",
]
