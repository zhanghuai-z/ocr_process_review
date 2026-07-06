"""Shared test fixtures for model-boundary migrations."""
from __future__ import annotations

import pytest

from app.models import Block, Line, Page
from app.models.ocr_character_observation import replace_line_ocr_char_observations
from app.models.ocr_observation_store import set_ocr_lines_for_block_uid


@pytest.fixture(autouse=True)
def _seed_block_constructor_lines_into_ocr_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat legacy ``Block(lines=[...])`` test factories as OCR observations.

    Production readers are moving away from the ``Block.lines`` projection. Many
    older tests still build their pages with ``Block(lines=[...])`` directly;
    this test-only bridge preserves their intended fixture facts without adding
    a production fallback path.
    """

    original_post_init = Block.__post_init__

    def patched_post_init(self: Block) -> None:
        original_post_init(self)
        if self.lines:
            set_ocr_lines_for_block_uid(self.uid, list(self.lines))

    monkeypatch.setattr(Block, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def _seed_line_constructor_chars_into_ocr_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat ``Line(chars=[...])`` test factories as OCR char observations."""

    original_post_init = Line.__post_init__

    def patched_post_init(self: Line) -> None:
        original_post_init(self)
        if self.chars:
            replace_line_ocr_char_observations(self.uid, list(self.chars))

    monkeypatch.setattr(Line, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def _seed_page_constructor_blocks_into_layout_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat ``Page(blocks=[...])`` test factories as adopted layout snapshots.

    Production code now requires an explicit ``LayoutSnapshot`` before layout
    consumers run. Older tests still use direct Page/Block constructors to state
    their fixture layout; this test-only bridge keeps those fixtures concise
    without adding a product fallback from ``Page.blocks`` back to layout truth.
    """

    original_post_init = Page.__post_init__
    original_setattr = Page.__setattr__

    def sync_page_snapshot(page: Page) -> None:
        from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection

        sync_page_layout_snapshot_from_projection(page, source_engine="test_fixture")

    def patched_post_init(self: Page) -> None:
        original_post_init(self)
        sync_page_snapshot(self)

    def patched_setattr(self: Page, name: str, value: object) -> None:
        original_setattr(self, name, value)
        if name != "blocks":
            return
        if not getattr(self, "uid", ""):
            return
        from app.models.layout_snapshot_store import layout_snapshot_for_page

        snapshot = layout_snapshot_for_page(self)
        if snapshot is not None and snapshot.source_engine != "test_fixture":
            return
        sync_page_snapshot(self)

    monkeypatch.setattr(Page, "__post_init__", patched_post_init)
    monkeypatch.setattr(Page, "__setattr__", patched_setattr)
