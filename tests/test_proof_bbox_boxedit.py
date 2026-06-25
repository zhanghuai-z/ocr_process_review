"""proof-bbox-boxedit (round 9): bbox / 单字位编辑。

覆盖：
- OCR 文本框是只读参考上下文，Backspace / Delete 不改写模型。
- 单槽位 inline 编辑：gallery 直输 / 右键气泡改当前 entry。
- 单槽位清空：Backspace/Delete 把当前 entry 改成 " " 而非删除。
- modifier + 方向键：Ctrl+Alt+→ 步进 current；Shift+Alt+→ 扩展选择；
  Alt+↓ 跨 wrap 行步进。
- gallery thumb 尺寸被 round 9 抬到 ≥30。
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
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


def _make_project(text: str) -> OcrProject:
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, len(text) * 10, 20))
    line.id = 7101
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
    page.id = 9101
    return OcrProject(name="t", pages=[page])


def _load(text: str):
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project(text)
    v = VProofPanel()
    v.load_pages(proj.pages)
    return v, proj


TEXT = "甲乙丙丁戊"


# ───── thumb size ─────────────────────────────────────────────

def test_gallery_thumb_bumped_round9():
    from app.ui.proof import v_proof
    assert v_proof.GALLERY_THUMB >= 30


# ───── OCR 文本参考框只读 ───────────────────────────────────

def _press(widget, key, mod=Qt.KeyboardModifier.NoModifier):
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, key, mod, "")
    QApplication.sendEvent(widget, ev)


def test_backspace_does_not_edit_reference_text():
    v, _ = _load(TEXT)
    te = v._text_edit
    before = te.toPlainText()
    assert te.isReadOnly()
    cur = te.textCursor()
    cur.setPosition(1)  # 光标在"甲"之后
    te.setTextCursor(cur)
    _press(te, Qt.Key.Key_Backspace)
    after = te.toPlainText()
    assert after == before


def test_delete_forward_does_not_edit_reference_text():
    v, _ = _load(TEXT)
    te = v._text_edit
    before = te.toPlainText()
    assert te.isReadOnly()
    cur = te.textCursor()
    cur.setPosition(0)
    te.setTextCursor(cur)
    _press(te, Qt.Key.Key_Delete)
    after = te.toPlainText()
    assert after == before


def test_selection_delete_does_not_edit_reference_text():
    v, _ = _load(TEXT)
    te = v._text_edit
    before = te.toPlainText()
    assert te.isReadOnly()
    cur = te.textCursor()
    cur.setPosition(0)
    cur.setPosition(3, cur.MoveMode.KeepAnchor)
    te.setTextCursor(cur)
    _press(te, Qt.Key.Key_Delete)
    after = te.toPlainText()
    assert after == before


def _select_char_in_list(v, char):
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == char:
            v._char_list.setCurrentRow(i)
            v._on_char_selection_changed()
            return
    raise AssertionError(f"char_list 中找不到 {char!r}")


# ───── inline 单槽位编辑 ────────────────────────────────────

def test_inline_slot_apply_replaces_current_entry():
    v, _ = _load(TEXT)
    _select_char_in_list(v, "甲")
    first = v._gallery_model.index(0, 0)
    v._sync_gallery_entry(first)
    assert v._current_candidate_entry is not None
    assert v._gallery_direct_overwrite("替") is True
    txt = v._text_edit.toPlainText()
    assert txt.startswith("替"), f"首槽未被替换：{txt!r}"
    assert len(txt.rstrip("\n")) == len(TEXT)


def test_inline_slot_blank_button_fills_space():
    v, _ = _load(TEXT)
    _select_char_in_list(v, "甲")
    first = v._gallery_model.index(0, 0)
    v._sync_gallery_entry(first)
    assert v._gallery_direct_blank() is True
    txt = v._text_edit.toPlainText()
    assert txt[0] == " "
    assert len(txt.rstrip("\n")) == len(TEXT)


# ───── modifier + 方向键 ─────────────────────────────────────

def test_modifier_arrow_steps_current_entry():
    v, _ = _load("甲乙甲乙甲乙")  # "甲" 有 3 处
    _select_char_in_list(v, "甲")
    sel = v._gallery_view.selectionModel()
    sel.setCurrentIndex(v._gallery_model.index(0, 0), sel.SelectionFlag.ClearAndSelect)
    v._step_gallery_singleton(+1)
    assert sel.currentIndex().row() == 1
    v._step_gallery_singleton(-1)
    assert sel.currentIndex().row() == 0


def test_shift_alt_extends_selection_without_moving_current():
    v, _ = _load("甲乙甲乙甲乙")
    _select_char_in_list(v, "甲")
    sel = v._gallery_view.selectionModel()
    sel.setCurrentIndex(v._gallery_model.index(0, 0), sel.SelectionFlag.ClearAndSelect)
    v._extend_gallery_selection(+1)
    rows = sorted(i.row() for i in sel.selectedIndexes())
    assert 1 in rows
    assert sel.currentIndex().row() == 0, "current 不应被扩选移动"


def test_alt_up_down_steps_by_row():
    v, _ = _load("甲" * 12)  # 12 处
    _select_char_in_list(v, "甲")
    sel = v._gallery_view.selectionModel()
    sel.setCurrentIndex(v._gallery_model.index(0, 0), sel.SelectionFlag.ClearAndSelect)
    per_row = v._gallery_items_per_row()
    v._step_gallery_row(+1)
    assert sel.currentIndex().row() == min(11, per_row)


# ───── _refresh_slot_info ───────────────────────────────────

def test_slot_info_label_updates_on_gallery_sync():
    v, _ = _load(TEXT)
    _select_char_in_list(v, "甲")
    first = v._gallery_model.index(0, 0)
    v._sync_gallery_entry(first)
    assert not hasattr(v, "_slot_info_lbl")
    assert v._current_candidate_entry is not None
    assert v._current_candidate_entry.page_number == 1
    assert (v._current_candidate_entry.token_text or v._current_candidate_entry.char) == "甲"
    assert "第 1 页" in v._gallery_hdr.text()
