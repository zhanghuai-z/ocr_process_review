"""Shared test fixtures for model-boundary migrations."""
from __future__ import annotations

import pytest

from app.models import Block
from app.models.ocr_observation import replace_block_ocr_lines


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
            replace_block_ocr_lines(self, list(self.lines))

    monkeypatch.setattr(Block, "__post_init__", patched_post_init)
