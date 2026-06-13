"""hproof-yaxis-quiet-load 本轮测试。

任务 1：图字 y 轴对应 —— _RowEditor 接受 _LinePair 推入的 x_centers，
        paintEvent 走自绘路径，槽位 x 中心直接来自 image bbox。
任务 2：加载页面时的成串小弹窗 —— page_directory.row 不再设 fname tooltip，
        confidence_badge 也不再 setToolTip。
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True, scope="module")
def _qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_pair_with_chars(text: str = "abc"):
    from app.models import Block, BBox, Char, Line, Page, ProofStatus
    from app.models.project import BlockType
    from app.core.page_image_cache import PageImageCache
    from app.ui.proof.h_proof import _LinePair

    chars = [
        Char(char=c, confidence=0.9, bbox=BBox(10 + i * 30, 0, 35 + i * 30, 30))
        for i, c in enumerate(text)
    ]
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, 400, 32),
                chars=chars, proof_status=ProofStatus.UNCHECKED, ocr_text=text)
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 400, 32),
                  lines=[line])
    page = Page(image_path="", width=400, height=32, blocks=[block])
    return _LinePair(0, block, line, page, 1, PageImageCache.instance())


def test_editor_exposes_slot_geometry_api():
    """_RowEditor 暴露 set_slot_geometry / has_slot_geometry，是图字 y 轴对应的
    主接入点。"""
    from app.ui.proof.h_proof import _RowEditor

    ed = _RowEditor()
    assert hasattr(ed, "set_slot_geometry")
    assert hasattr(ed, "has_slot_geometry")
    assert ed.has_slot_geometry() is False

    ed.set_slot_geometry([10.0, 30.0, 50.0], [16.0, 16.0, 16.0])
    assert ed.has_slot_geometry() is True

    ed.set_slot_geometry(None, None)
    assert ed.has_slot_geometry() is False
    ed.deleteLater()


def test_line_pair_pushes_image_aligned_xcenters_to_editor():
    """_LinePair._sync_editor_slot_geometry 应按 char.bbox * render_scale
    把每字 x 中心送进 editor —— 这就是"图字真正 y 轴对应"的证据。

    fixture: char i 的 BBox(10+i*30, 0, 35+i*30, 30) →
    x=10+i*30, x2=45+2*i*30, center=(10+i*30 + 45+2*i*30)/2=27.5+1.5*i*30。
    origin=0, scale=2.0：
      i=0: center=27.5 → 55.0
      i=1: center=72.5 → 145.0
      i=2: center=117.5 → 235.0
    """
    pair = _make_pair_with_chars("abc")
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 2.0
    pair._sync_editor_slot_geometry()

    assert pair._editor.has_slot_geometry() is True
    xs = pair._editor._slot_x_centers
    assert xs is not None and len(xs) == 3
    expected = [55.0, 145.0, 235.0]
    for got, want in zip(xs, expected):
        assert got is not None
        assert abs(got - want) < 0.5
    pair.deleteLater()


def test_line_pair_uses_slot_line_editor_for_visible_hproof_text():
    """横校可见文本层应是 slot editor，而不是 QPlainTextEdit 原生布局。"""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from app.ui.proof.h_proof import _SlotLineEditor

    pair = _make_pair_with_chars("abc")
    assert isinstance(pair._editor, _SlotLineEditor)
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0
    pair._sync_editor_slot_geometry()
    pair._editor._select_slot_index(1)
    ev = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        int(Qt.Key.Key_X),
        Qt.KeyboardModifier.NoModifier,
        "X",
    )
    pair._editor.keyPressEvent(ev)

    assert pair._editor.toPlainText() == "aXc"
    assert pair._editor.has_slot_geometry() is True
    pair.deleteLater()


def test_slot_line_editor_accepts_chinese_ime_commit():
    """中文输入法提交走 QInputMethodEvent.commitString，不走 keyPressEvent。"""
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QInputMethodEvent
    from app.ui.proof.h_proof import _SlotLineEditor

    pair = _make_pair_with_chars("abc")
    editor = pair._editor
    assert isinstance(editor, _SlotLineEditor)
    assert editor.testAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled)
    assert editor.inputMethodQuery(Qt.InputMethodQuery.ImEnabled) is True

    editor._select_slot_index(1)
    ev = QInputMethodEvent("", [])
    ev.setCommitString("好")
    QCoreApplication.sendEvent(editor, ev)

    assert editor.toPlainText() == "a好c"
    pair.deleteLater()


def test_line_pair_degrades_when_chars_misaligned():
    """chars 长度与文本不等 → _chars_aligned False → editor 清空 slot 几何，
    走原生渲染降级。"""
    from app.models import BBox, Char, Line, Page, Block, ProofStatus
    from app.models.project import BlockType
    from app.core.page_image_cache import PageImageCache
    from app.ui.proof.h_proof import _LinePair

    # text='ab' 但 chars 只给一个
    chars = [Char(char="a", confidence=0.9, bbox=BBox(0, 0, 20, 20))]
    line = Line(text="ab", confidence=0.9, bbox=BBox(0, 0, 80, 20),
                chars=chars, proof_status=ProofStatus.UNCHECKED)
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20),
                  lines=[line])
    page = Page(image_path="", width=80, height=20, blocks=[block])
    pair = _LinePair(0, block, line, page, 1, PageImageCache.instance())

    pair._render_scale = 2.0
    pair._sync_editor_slot_geometry()
    assert pair._editor.has_slot_geometry() is False
    pair.deleteLater()


def test_page_directory_row_has_no_filename_tooltip():
    """任务 2：directory 列表的文件名行不再 setToolTip，
    扫过列表时不会冒一串小气泡。"""
    src = open("app/ui/widgets/page_directory.py", encoding="utf-8").read()
    # 不再存在 fname_lbl.setToolTip(fname) 这条
    assert "fname_lbl.setToolTip(fname)" not in src


def test_confidence_badge_has_no_tooltip_calls():
    """任务 2：confidence badge 也走静默 —— 百分比文字已可视，
    不再用 hover tooltip 重复。"""
    src = open("app/ui/widgets/confidence_badge.py", encoding="utf-8").read()
    assert "self.setToolTip(" not in src


def test_hproof_status_label_has_no_tooltip_calls():
    """任务 2：行级状态标签也必须静默，不能残留 setToolTip("") 这类空气泡。"""
    src = open("app/ui/proof/h_proof.py", encoding="utf-8").read()
    assert "self._status_lbl.setToolTip(" not in src


def test_layout_still_two_rows_after_yaxis_refit():
    """任务 1 的"红线"：保持 image+editor 两层结构，不能借机回到 3 行
    stacked elements。"""
    pair = _make_pair_with_chars("ab")
    layout = pair._content.layout()
    assert layout.count() == 2
    assert layout.itemAt(0).widget() is pair._img_lbl
    assert layout.itemAt(1).widget() is pair._editor
    assert not hasattr(pair, "_ribbon")
    pair.deleteLater()
