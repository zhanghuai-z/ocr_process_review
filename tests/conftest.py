"""Shared test fixtures for model-boundary migrations."""
from __future__ import annotations

import pytest

from app.models import Block, Line
from app.models.ocr_character_observation import replace_line_ocr_chars
from app.models.ocr_observation import replace_block_ocr_line_observations


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
            replace_block_ocr_line_observations(self.uid, list(self.lines))

    monkeypatch.setattr(Block, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def _seed_line_constructor_chars_into_ocr_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat ``Line(chars=[...])`` test factories as OCR char observations."""

    original_post_init = Line.__post_init__

    def patched_post_init(self: Line) -> None:
        original_post_init(self)
        if self.chars:
            replace_line_ocr_chars(self, list(self.chars))

    monkeypatch.setattr(Line, "__post_init__", patched_post_init)
