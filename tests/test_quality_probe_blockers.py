"""Round 16 — coord 评审两个 blocker 的回归覆盖。

Blocker 1: probe 不得通过可见的 gallery 文本标签自曝身份
Blocker 2: 必须存在一条非破坏性识别路径（不动 line.text）
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
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
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page_number=1, block_index=0, line_index=0, char_index=1),
        true_char="己",
        fake_char="已",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    return [page], line, probe


def test_blocker1_extras_token_text_hides_probe_identity():
    from app.ui.proof.v_proof import VProofPanel

    pages, _line, probe = _build_pages_with_planted_probe()
    panel = VProofPanel()
    panel.load_pages(pages)

    extras = panel._extras_for_tokens(["己"])
    assert len(extras) == 1
    entry = extras[0]
    assert entry.char_idx == probe.key.char_index
    assert entry.token_text == "己", (
        f"probe 身份泄露：token_text 应为 true_char '己'，实际 {entry.token_text!r}"
    )
    assert entry.char == "己"
    panel.close()


def test_blocker2_mark_observed_is_nondestructive():
    from app.ui.proof.v_proof import VProofPanel

    pages, line, probe = _build_pages_with_planted_probe()
    original_text = line.text
    original_chars = [c.char for c in line.chars]

    panel = VProofPanel()
    panel.load_pages(pages)
    panel._selected_char = "己"
    entries = list(panel._char_svc.query("己"))
    entries.extend(panel._extras_for_tokens(["己"]))
    panel._gallery_model.set_entries(entries)
    QApplication.processEvents()

    model = panel._gallery_model
    target_row = None
    for row in range(model.rowCount()):
        idx = model.index(row, 0)
        e = idx.data(Qt.ItemDataRole.UserRole)
        if e is not None and e.char_idx == probe.key.char_index:
            target_row = row
            break
    assert target_row is not None, "probe extra entry 应出现在 gallery 中"

    sel = panel._gallery_view.selectionModel()
    sel.select(
        model.index(target_row, 0),
        QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
    )
    QApplication.processEvents()

    panel._mark_selected_observed()
    QApplication.processEvents()

    assert probe.observation == "corrected", (
        f"probe.observation 应为 'corrected'，实际 {probe.observation!r}"
    )
    assert line.text == original_text, (
        f"line.text 不应被修改：期望 {original_text!r}，实际 {line.text!r}"
    )
    assert [c.char for c in line.chars] == original_chars
    panel.close()


def test_blocker2_mark_observed_no_store_is_safe():
    from app.ui.proof.v_proof import VProofPanel

    pages, line, _probe = _build_pages_with_planted_probe()
    qp.reset_active_store()
    original_text = line.text

    panel = VProofPanel()
    panel.load_pages(pages)
    panel._selected_char = "己"
    entries = list(panel._char_svc.query("己"))
    entries.extend(panel._extras_for_tokens(["己"]))
    panel._gallery_model.set_entries(entries)
    QApplication.processEvents()
    panel._mark_selected_observed()
    assert line.text == original_text
    panel.close()
