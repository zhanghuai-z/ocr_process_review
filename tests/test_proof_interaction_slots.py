"""proof-interaction-slots round 6: 固定槽位编辑 + gallery 批量交互。

只覆盖纯单元层面的语义：_RowEditor 在固定模式下的删/插/粘语义，
以及 v_proof 的字符索引文本（无图、无前缀）。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from app.ui.proof.h_proof import _RowEditor


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _make_fixed_editor(qapp, text: str = "abcde") -> _RowEditor:
    ed = _RowEditor()
    ed.setPlainText(text)
    ed.set_fixed_length(len(text))
    return ed


def _press(ed: _RowEditor, key: Qt.Key, *, mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier, text: str = "") -> None:
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, int(key), mods, text)
    ed.keyPressEvent(ev)


def test_backspace_fills_blank_not_remove(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    cur = ed.textCursor()
    cur.setPosition(3)  # 光标在 c|d
    ed.setTextCursor(cur)
    _press(ed, Qt.Key.Key_Backspace)
    # 长度不变；前一个槽位被空白填充
    assert len(ed.toPlainText()) == 5
    assert ed.toPlainText() == "ab de"


def test_delete_fills_blank_not_remove(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    cur = ed.textCursor()
    cur.setPosition(2)  # 光标在 b|c
    ed.setTextCursor(cur)
    _press(ed, Qt.Key.Key_Delete)
    assert len(ed.toPlainText()) == 5
    assert ed.toPlainText() == "ab de"


def test_printable_replaces_next_slot(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    cur = ed.textCursor()
    cur.setPosition(0)
    ed.setTextCursor(cur)
    _press(ed, Qt.Key.Key_X, text="X")
    # 长度保持；首槽被替换
    assert len(ed.toPlainText()) == 5
    assert ed.toPlainText() == "Xbcde"


def test_paste_truncates_when_too_long(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    cur = ed.textCursor()
    cur.setPosition(0)
    ed.setTextCursor(cur)
    md = QMimeData()
    md.setText("XYZWVU_extra")  # 远超 5 槽
    ed.insertFromMimeData(md)
    assert len(ed.toPlainText()) == 5


def test_paste_blank_pads_when_too_short(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    cur = ed.textCursor()
    cur.setPosition(0)
    cur.setPosition(3, cur.MoveMode.KeepAnchor)  # 选中 abc
    ed.setTextCursor(cur)
    md = QMimeData()
    md.setText("X")  # 只有 1 字符
    ed.insertFromMimeData(md)
    assert len(ed.toPlainText()) == 5
    # X + 2 空格 + de
    assert ed.toPlainText() == "X  de"


def test_fixed_editor_has_no_tooltip(qapp):
    ed = _make_fixed_editor(qapp, "abcde")
    # 第 1 任务：固定模式不再设 tooltip，避免空白悬浮框
    assert ed.toolTip() == ""
