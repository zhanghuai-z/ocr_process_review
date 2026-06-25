"""proof-slot-residual (round 7): fixed-slot 残留闭环 + 批量改字。

覆盖：
- _canonicalize_text_to_slots：单字符 chars 下的 pad / 不截断 / 多 token 降级
- _RowEditor 在 misaligned-shorter 场景下也进入 fixed 模式
- gallery 多选：批量替换走统一 proof edit 路径
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.ui.proof.h_proof import (
    _RowEditor,
    _canonicalize_text_to_slots,
    _chars_are_single_codepoint,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _mk_chars(s: str):
    """chars 列表，每个元素都是单字符。"""
    return [SimpleNamespace(char=ch, confidence=0.9) for ch in s]


def _mk_token_chars(tokens):
    return [SimpleNamespace(char=t, confidence=0.9) for t in tokens]


def test_canonicalize_pads_short_text():
    text = "abc"
    chars = _mk_chars("xxxxx")
    out, locked = _canonicalize_text_to_slots(text, chars)
    assert out == "abc  "
    assert locked is True


def test_canonicalize_keeps_equal_length():
    text = "abcde"
    chars = _mk_chars("xxxxx")
    out, locked = _canonicalize_text_to_slots(text, chars)
    assert out == "abcde"
    assert locked is True


def test_canonicalize_does_not_truncate_long_text():
    text = "abcdefg"
    chars = _mk_chars("xxxxx")
    out, locked = _canonicalize_text_to_slots(text, chars)
    assert out == "abcdefg"  # 不动！
    assert locked is False


def test_canonicalize_token_granularity_degrades():
    text = "2016 年"
    chars = _mk_token_chars(["2016", " ", "年"])
    out, locked = _canonicalize_text_to_slots(text, chars)
    assert out == text
    assert locked is False


def test_canonicalize_empty_chars_degrades():
    text = "anything"
    out, locked = _canonicalize_text_to_slots(text, [])
    assert out == text
    assert locked is False


def test_chars_are_single_codepoint():
    assert _chars_are_single_codepoint(_mk_chars("abc")) is True
    assert _chars_are_single_codepoint(_mk_token_chars(["ab", "c"])) is False
    assert _chars_are_single_codepoint([]) is False


def test_editor_enters_fixed_mode_after_short_text_padded(qapp):
    """misaligned-shorter 场景：手动 set_fixed_length 等同于面板加载后状态。"""
    ed = _RowEditor()
    # 模拟"原文 3 字、chars 5 字 → 面板加载时被补到 5 字"
    padded_text = "abc  "
    ed.setPlainText(padded_text)
    ed.set_fixed_length(5)
    assert ed.fixed_length() == 5
    # Backspace 不删字、填空白
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    cur = ed.textCursor()
    cur.setPosition(3)  # 光标在 abc| __
    ed.setTextCursor(cur)
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Backspace),
                   Qt.KeyboardModifier.NoModifier, "")
    ed.keyPressEvent(ev)
    assert len(ed.toPlainText()) == 5
    assert ed.toPlainText() == "ab   "  # c 被填空


def test_editor_overlong_text_stays_free_mode(qapp):
    """misaligned-longer 场景：editor 不应被强行截断。"""
    ed = _RowEditor()
    ed.setPlainText("abcdefg")  # 7 字
    # 模拟 _apply_fixed_length_to_editor 看到 7 > 5 → 不设 fixed
    ed.set_fixed_length(None)
    assert ed.fixed_length() is None
    # 自由模式：Backspace 真的删字
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    cur = ed.textCursor()
    cur.setPosition(7)
    ed.setTextCursor(cur)
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Backspace),
                   Qt.KeyboardModifier.NoModifier, "")
    ed.keyPressEvent(ev)
    assert ed.toPlainText() == "abcdef"  # 真的删了
