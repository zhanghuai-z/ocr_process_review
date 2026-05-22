"""Round 16/18 — coord 评审：probe 身份不得通过 gallery 自曝；
mark-observed 路径在 Round 18 已被"文本即锚点"取代，此处只保留 token 隔离断言。"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core import quality_probe as qp
from app.models import BBox, Block, BlockType, Char, Line, Page


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    qp.reset_active_store()


def _build_pages_with_planted_probe():
    line = Line(text="己已", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    line.chars = [
        Char(char="己", confidence=0.9, bbox=BBox(0, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="己"),
        Char(char="已", confidence=0.9, bbox=BBox(20, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="已"),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block],
                image_path="/tmp/probe-blocker.png", width=100, height=100)
    # Round 18：probe.true_char 必须 == line.text[char_index]；fake_char 是近形字
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page_number=1, block_index=0, line_index=0, char_index=1),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    return [page], line, probe


def test_blocker1_extras_are_disabled_in_round18():
    """Round 18：gallery extras 完全移除——probe 通过 displayed_text 在 OCR
    文本窗口里体现，gallery 不应再接收任何 probe extra 入口。"""
    from app.ui.proof.v_proof import VProofPanel

    pages, _line, _probe = _build_pages_with_planted_probe()
    panel = VProofPanel()
    panel.load_pages(pages)

    extras = panel._extras_for_tokens(["已"])
    assert extras == [], f"_extras_for_tokens 应永远返回 []，实际 {extras!r}"
    panel.close()
