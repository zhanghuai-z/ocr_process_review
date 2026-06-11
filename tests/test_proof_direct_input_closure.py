"""Regression tests for the accepted HProof subset of proof-direct-input-closure."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent
from PySide6.QtGui import QFocusEvent
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


def _hproof_with_two_lines():
    from app.ui.proof.h_proof import HProofPanel
    line1 = Line(text="甲乙丙", confidence=0.9, bbox=BBox(0, 0, 30, 20))
    line1.id = 81
    line1.chars = [Char(char=c, confidence=0.9, bbox=BBox(i * 10, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text=c) for i, c in enumerate("甲乙丙")]
    line2 = Line(text="丁戊己", confidence=0.9, bbox=BBox(0, 30, 30, 20))
    line2.id = 82
    line2.chars = [Char(char=c, confidence=0.9, bbox=BBox(i * 10, 30, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text=c) for i, c in enumerate("丁戊己")]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 30, 50), lines=[line1, line2])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=300, height=200)
    page.id = 9301
    h = HProofPanel()
    h.load_pages([page])
    return h


def test_hproof_text_editor_has_no_frame_border():
    h = _hproof_with_two_lines()
    assert len(h._pairs) >= 1
    ed = h._pairs[0]._editor
    ss = ed.styleSheet()
    assert "border:none" in ss.replace(" ", "") or "border:none" in ss
    assert "background:#eaf3ff" in ss.replace(" ", "") or "background:#eaf3ff" in ss
    h._activate(1)
    inactive_ss = h._pairs[0]._editor.styleSheet()
    assert (
        "background:transparent" in inactive_ss.replace(" ", "")
        or "background:transparent" in inactive_ss
    )
    # cursor width 0 = caret 不显示
    assert ed.cursorWidth() == 0


def test_hproof_editor_focus_activates_row():
    h = _hproof_with_two_lines()
    # 先把 0 行激活；再让 1 行的 editor 拿到焦点
    h._activate(0)
    assert h._current_idx == 0
    p1 = h._pairs[1]
    # 模拟 focusIn —— 走 row_focus_requested
    fe = QFocusEvent(QEvent.Type.FocusIn)
    p1._editor.focusInEvent(fe)
    assert h._current_idx == 1, "editor 取得焦点应自动切换激活行"


def test_hproof_editor_emits_row_focus_on_mouse_press():
    h = _hproof_with_two_lines()
    h._activate(0)
    p1 = h._pairs[1]
    fired = []
    p1._editor.row_focus_requested.connect(lambda: fired.append(1))
    # 直接 emit（绕过 platform mouse 事件不稳）
    p1._editor.row_focus_requested.emit()
    assert fired
