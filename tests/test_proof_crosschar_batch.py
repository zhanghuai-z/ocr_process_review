"""proof-crosschar-batch (round 8): char_list 多选 → 跨字 gallery 合并 + 批量改字。

覆盖：
- _selected_char_tokens / _on_char_selection_changed：char_list 选中 N(>=2) 字时，
  gallery_model 装入的是合并后的 entry 列表，按 (page_number, char_idx) 排序。
- _select_all_visible_chars：Ctrl+A 选中可见全部，触发合并。
- _clear_char_list_multi：Esc 收回到单选。
- _apply_replacement_to_selected 在跨字模式下，按 entry 自己的 token 长度替换，
  不会用 anchor.token_len 一刀切。
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp_mod
from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    qp_mod.reset_active_store()
    ProofStateBus.reset()


def _make_project_with_char_crops(text: str) -> OcrProject:
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, len(text) * 10, 20))
    line.id = 7001
    line.chars = [
        Char(
            char=ch,
            confidence=0.9,
            bbox=BBox(i * 10, 0, 10, 20),
            bbox_source="ocr",
            bbox_granularity="char",
            token_text=ch,
        )
        for i, ch in enumerate(text)
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, len(text) * 10, 200), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=300, height=200)
    page.id = 9001
    return OcrProject(name="t", pages=[page])


def _load(text: str):
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project_with_char_crops(text)
    v = VProofPanel()
    v.load_pages(proj.pages)
    return v, proj


def _select_chars_in_list(v, tokens):
    """在 _char_list 里多选指定 token 的 item。"""
    v._char_list.clearSelection()
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        tok = it.data(Qt.ItemDataRole.UserRole)
        if tok in tokens:
            it.setSelected(True)
    # itemSelectionChanged 信号同步触发；为稳起见手动 invoke
    v._on_char_selection_changed()


# CharIndexService 默认 include_non_cjk=False，因此测试必须用 CJK 字符。
TEXT = "甲乙丙甲乙丙"  # 6 字；甲/乙/丙 各 2 次


# ── 跨字 gallery 合并 ─────────────────────────────────────────

def test_cross_char_merges_entries_in_gallery():
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    entries = v._gallery_model._entries  # type: ignore[attr-defined]
    # 应包含所有 甲 和 乙 位置：char_idx 0,1,3,4
    assert [e.char_idx for e in entries] == [0, 1, 3, 4]
    assert {e.char for e in entries} == {"甲", "乙"}


def test_cross_char_header_shows_multi_format():
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    txt = v._gallery_hdr.text()
    assert "跨字" in txt
    assert "2 字" in txt
    assert "4 处" in txt


def test_cross_char_gallery_selection_keeps_cross_char_header():
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    sel = v._gallery_view.selectionModel()
    idx0 = v._gallery_model.index(0, 0)
    idx1 = v._gallery_model.index(1, 0)
    sel.select(idx0, sel.SelectionFlag.Select)
    sel.select(idx1, sel.SelectionFlag.Select)
    v._on_gallery_selection_changed()
    txt = v._gallery_hdr.text()
    assert "跨字" in txt
    assert "2 字" in txt
    assert "已选 2 个" in txt


def test_single_select_keeps_legacy_header():
    v, _ = _load(TEXT)
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == "甲":
            v._on_char_clicked(it)
            break
    assert v._gallery_hdr.text().startswith('"甲"')
    assert "跨字" not in v._gallery_hdr.text()


# ── Ctrl+A / Esc ─────────────────────────────────────────────

def test_select_all_visible_chars_merges_all():
    v, _ = _load(TEXT)
    v._select_all_visible_chars()
    # 应该全选 3 个字 → gallery 有 6 个 entry
    assert len(v._gallery_model._entries) == 6  # type: ignore[attr-defined]
    assert "3 字" in v._gallery_hdr.text()


def test_clear_char_list_multi_returns_to_single():
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    assert "跨字" in v._gallery_hdr.text()
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == "甲":
            v._char_list.setCurrentItem(it)
            break
    v._clear_char_list_multi()
    assert v._gallery_hdr.text().startswith('"甲"')
    assert "跨字" not in v._gallery_hdr.text()


# ── batch apply：按 entry 各自 token 长度替换 ────────────────────

def test_batch_apply_uses_per_entry_token_len():
    """跨字 batch：甲/乙/丙 三处各自被替换为 'X'，互不影响。"""
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    sel = v._gallery_view.selectionModel()
    # 选前 3 个 entry：甲@0, 乙@1, 甲@3
    for row in range(3):
        idx = v._gallery_model.index(row, 0)
        sel.select(idx, sel.SelectionFlag.Select)
    applied = v._apply_replacement_to_selected("X")
    assert applied == 3
    # 原 "甲乙丙甲乙丙" → "XX丙X乙丙"
    assert v._text_edit.toPlainText().rstrip("\n") == "XX丙X乙丙"


def test_cross_char_sync_gallery_entry_highlights_exact_entry_token():
    v, _ = _load(TEXT)
    _select_chars_in_list(v, ["甲", "乙"])
    idx = v._gallery_model.index(1, 0)  # 乙@1
    v._sync_gallery_entry(idx)
    cursor = v._text_edit.textCursor()
    assert cursor.hasSelection()
    assert cursor.selectedText() == "乙"
