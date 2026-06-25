"""vproof-direct-overwrite-residual (round 11) tests.

覆盖：
1. _sync_gallery_entry 会绑定当前 gallery entry，键盘直输继续作用于该 entry。
2. _GalleryListView.keyPressEvent 真的把单字按键路由到 _gallery_direct_overwrite，
   并真正覆盖当前 entry。
3. Backspace 把当前 entry 填空白。
4. Ctrl+A / Alt+方向键 不被吃。
5. _on_external_line_changed 改为 QTimer debounce：N 条事件折叠成一次重建。
"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from app.core.proof_state import ProofUpdateRequest
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
    line.id = 8401
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
    page.id = 9401
    return OcrProject(name="t", pages=[page])


def _load_vproof(text: str):
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project(text)
    v = VProofPanel()
    v.load_pages(proj.pages)
    return v, proj


def _make_page(text: str, page_number: int, page_id: int, line_id: int) -> Page:
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, len(text) * 10, 20))
    line.id = line_id
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
    page = Page(
        page_number=page_number,
        blocks=[block],
        image_path=f"/tmp/vproof-refresh-{page_number}.png",
        width=300,
        height=200,
    )
    page.id = page_id
    return page


def _select_char(v, ch):
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == ch:
            v._char_list.setCurrentRow(i)
            v._on_char_selection_changed()
            return
    raise AssertionError(f"未找到 char {ch!r}")


# ───── 任务 1：gallery 同步绑定当前 entry ─────────────────────

def test_sync_gallery_entry_binds_current_candidate_entry():
    v, _ = _load_vproof("甲乙丙")
    _select_char(v, "甲")
    first = v._gallery_model.index(0, 0)
    v._sync_gallery_entry(first)
    assert v._current_candidate_entry is not None
    assert (v._current_candidate_entry.token_text or v._current_candidate_entry.char) == "甲"


def test_sync_gallery_entry_records_selected_occurrence_in_session():
    v, _ = _load_vproof("甲乙甲")
    _select_char(v, "甲")
    first = v._gallery_model.index(0, 0)

    v._sync_gallery_entry(first)

    assert v._session.selected_occurrence_key is not None
    assert len(v._session.selected_occurrence_keys) == 1


def test_vproof_undo_resolves_occurrence_after_line_moves_between_blocks():
    v, proj = _load_vproof("甲甲")
    page = proj.pages[0]
    original_block = page.blocks[0]
    moved_block = Block(
        block_type=BlockType.TEXT,
        order=1,
        bbox=BBox(0, 40, 40, 20),
        lines=[],
    )
    page.blocks.append(moved_block)
    line = original_block.lines[0]

    _select_char(v, "甲")
    assert v._gallery_direct_overwrite("丙")
    assert proof_display_text(line) == "丙甲"
    assert v._vproof_undo_stack
    assert v._vproof_undo_stack[-1].edits[0].occurrence_keys

    original_block.lines.remove(line)
    moved_block.lines.append(line)

    assert v._undo_vproof_edit() is True
    assert moved_block.lines[0] is line
    assert proof_display_text(line) == "甲甲"


def test_merge_pages_restores_selected_occurrence_to_new_line_object():
    from app.ui.proof.v_proof import VProofPanel

    page = _make_page("甲甲", page_number=1, page_id=9401, line_id=8401)
    v = VProofPanel()
    v.load_pages([page])
    _select_char(v, "甲")
    before_key = v._session.selected_occurrence_key
    assert before_key is not None

    new_page = _make_page("甲甲", page_number=1, page_id=9401, line_id=8401)
    new_page.uid = page.uid
    new_page.blocks[0].uid = page.blocks[0].uid
    new_page.blocks[0].lines[0].uid = page.blocks[0].lines[0].uid

    v.merge_pages([new_page])

    assert v._session.selected_occurrence_key == before_key
    assert v._current_candidate_entry is not None
    assert v._current_candidate_entry.line is new_page.blocks[0].lines[0]
    assert v._text_edit.toPlainText() == "甲甲\n"


# ───── 任务 1：gallery 直输覆盖当前 entry ──────────────────────

def test_gallery_keystroke_overwrites_current_entry():
    v, _ = _load_vproof("甲乙丙")
    _select_char(v, "甲")
    v._sync_gallery_entry(v._gallery_model.index(0, 0))
    # 直接走 view 的 keyPressEvent
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier, "X")
    v._gallery_view.keyPressEvent(ev)
    txt = v._text_edit.toPlainText().rstrip("\n")
    assert txt.startswith("X"), f"keystroke 未覆盖第一槽位：{txt!r}"


def test_gallery_keystroke_backspace_blanks_current_entry():
    v, _ = _load_vproof("甲乙丙")
    _select_char(v, "甲")
    v._sync_gallery_entry(v._gallery_model.index(0, 0))
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier, "")
    v._gallery_view.keyPressEvent(ev)
    txt = v._text_edit.toPlainText()
    assert txt.startswith(" "), f"Backspace 未把首槽位填空白：{txt!r}"


def test_gallery_keystroke_with_ctrl_is_not_consumed():
    v, _ = _load_vproof("甲乙丙")
    _select_char(v, "甲")
    v._sync_gallery_entry(v._gallery_model.index(0, 0))
    before = v._text_edit.toPlainText()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier, "\x01")
    v._gallery_view.keyPressEvent(ev)
    assert v._text_edit.toPlainText() == before, "Ctrl+A 不应该改文本"


def test_gallery_keystroke_with_alt_arrow_is_not_consumed():
    v, _ = _load_vproof("甲乙丙")
    _select_char(v, "甲")
    v._sync_gallery_entry(v._gallery_model.index(0, 0))
    before = v._text_edit.toPlainText()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier, "")
    v._gallery_view.keyPressEvent(ev)
    assert v._text_edit.toPlainText() == before


def test_gallery_keystroke_without_current_entry_is_noop():
    v, _ = _load_vproof("甲乙丙")
    v._current_candidate_entry = None
    before = v._text_edit.toPlainText()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier, "X")
    v._gallery_view.keyPressEvent(ev)
    assert v._text_edit.toPlainText() == before


# ───── 任务 2：debounced 外部刷新 ─────────────────────────────

def test_on_external_line_changed_does_not_reload_synchronously():
    """N 次 external 事件应只触发 _do_external_refresh 一次（debounce）。"""
    v, proj = _load_vproof("甲乙丙")
    calls = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (calls.__setitem__("load", calls["load"] + 1), _o(i))[1]  # type: ignore

    page = proj.pages[0]
    line_id = page.blocks[0].lines[0].id
    # 一次性发 5 条
    for _ in range(5):
        v._on_external_line_changed(
            ProofUpdateRequest(
                page_id=page.id,
                line_id=line_id,
                status="MODIFIED",
                origin=99999,
            )
        )
    # 同步阶段 _load_page 不应被调用
    assert calls["load"] == 0, \
        f"外部事件应 debounce，不应同步调 _load_page；当前 {calls['load']}"
    # 触发一次 timer 回调（手动 flush）
    v._do_external_refresh()
    assert calls["load"] == 1, f"flush 后应只调一次 _load_page，当前 {calls['load']}"


def test_external_refresh_timer_is_single_shot():
    v, _ = _load_vproof("甲乙丙")
    assert v._external_refresh_timer.isSingleShot()


def test_external_refresh_clears_pending_when_no_pages():
    v, _ = _load_vproof("甲乙丙")
    v._session.pages = []
    v._session.pending_external_lines.add(123)
    v._session.pending_external_page_keys.add(("page", 1))
    v._do_external_refresh()
    assert v._session.pending_external_lines == set()
    assert v._session.pending_external_page_keys == set()


def test_external_refresh_does_not_reload_new_current_page_after_page_switch():
    from app.ui.proof.v_proof import VProofPanel

    page1 = _make_page("甲甲", page_number=1, page_id=9401, line_id=8401)
    page2 = _make_page("乙乙", page_number=2, page_id=9402, line_id=8402)
    v = VProofPanel()
    v.load_pages([page1, page2])

    calls = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (calls.__setitem__("load", calls["load"] + 1), _o(i))[1]  # type: ignore

    v._on_external_line_changed(
        ProofUpdateRequest(
            page_uid=page1.uid,
            page_id=page1.id,
            line_uid=page1.blocks[0].lines[0].uid,
            line_id=page1.blocks[0].lines[0].id,
            status="MODIFIED",
            origin=99999,
        )
    )
    assert v._session.pending_external_lines

    assert v._safe_load_page(1) is True
    assert calls["load"] == 1
    v._do_external_refresh()

    assert calls["load"] == 1
    assert v._session.current_page_index() == 1
    assert v._text_edit.toPlainText() == "乙乙\n"
    assert v._session.pending_external_lines == set()
    assert v._session.pending_external_page_keys == set()
