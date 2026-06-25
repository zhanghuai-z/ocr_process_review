"""vproof-ime-persist-visibility (round 12) tests.

四项硬验收：
1. 相同字索引窗口"挤到看不见" → 缩略图 / 列数 / 行数 提到真实可读尺寸。
2. VProof 输不了中文 → _GalleryListView 接 QInputMethodEvent.commitString。
3. 改字后切换字符再回来不应回退 → 目标编辑直接提交到共享模型。
4. 改字后 char index 没迁移 → _char_svc 在 commit 后重建。

注意：测试 IME 我们直接构造 QInputMethodEvent 投递到 _GalleryListView，
这是 Qt 在 IME commit 时走的同一条 dispatch；不是"测试里手工触发自定
义信号"——这是真实 Qt 输入路径。
"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtGui import QInputMethodEvent
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
    line.id = 9501
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
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, len(text) * 10, 200),
        lines=[line],
    )
    page = Page(
        page_number=1,
        blocks=[block],
        image_path="/tmp/none.png",
        width=300,
        height=200,
    )
    page.id = 9601
    return OcrProject(name="t", pages=[page])


def _load_vproof(text: str):
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project(text)
    v = VProofPanel()
    v.load_pages(proj.pages)
    return v, proj


def _select_char(v, ch: str) -> bool:
    for i in range(v._char_list.count()):
        it = v._char_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == ch:
            v._char_list.setCurrentRow(i)
            v._on_char_clicked(it)
            return True
    return False


# ─────────────────────── 任务 1：相同字索引尺寸 ───────────────────────


def test_gallery_thumb_truly_readable_size():
    """缩略图必须 >= 50px。56 是这轮的目标值。"""
    from app.ui.proof import v_proof
    assert v_proof.GALLERY_THUMB >= 50, (
        f"GALLERY_THUMB={v_proof.GALLERY_THUMB} 仍然偏小，CJK 字看不清"
    )


def test_gallery_items_per_row_reduced():
    """每行 item 数下调到 <=6，给单字更多像素。"""
    from app.ui.proof import v_proof
    assert v_proof.GALLERY_ITEMS_PER_ROW <= 6
    assert v_proof.GALLERY_MAX_ROWS <= 3


def test_gallery_delegate_size_follows_thumb():
    """delegate 报告的 sizeHint 必须跟着 GALLERY_THUMB 抬。"""
    from app.ui.proof.v_proof import _GalleryDelegate, GALLERY_THUMB
    assert _GalleryDelegate.SIZE >= GALLERY_THUMB + 12


# ─────────────────────── 任务 2：IME 真实闭环 ───────────────────────


def test_gallery_view_has_ime_enabled():
    v, _ = _load_vproof("也草")
    assert v._gallery_view.testAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled), (
        "WA_InputMethodEnabled 没开，IME 不会把 commit string 投到 gallery"
    )
    # 同时 inputMethodQuery 必须自报 enabled=True
    assert v._gallery_view.inputMethodQuery(Qt.InputMethodQuery.ImEnabled) is True


def test_ime_commit_string_routes_to_overwrite():
    """走真实 Qt 输入法 commit 路径：构造 QInputMethodEvent 直接派给 gallery。

    这是 Qt 在中文 IME 用户敲下空格 / 选词时走的同一条 dispatch；不是
    单元测试里假装触发的自定义信号。"""
    v, _ = _load_vproof("也草也")
    assert _select_char(v, "也")
    # 选 gallery 第 0 个 entry
    idx = v._gallery_model.index(0, 0)
    v._gallery_view.setCurrentIndex(idx)
    v._sync_gallery_entry(idx)

    ev = QInputMethodEvent("", [])
    ev.setCommitString("好")
    QCoreApplication.sendEvent(v._gallery_view, ev)

    # text_edit 应该立即反映新字
    assert "好" in v._text_edit.toPlainText(), (
        f"IME commit 后 _text_edit 没拿到'好'：{v._text_edit.toPlainText()!r}"
    )


# ─────────────────────── 任务 3+4：编辑落到模型 + 索引重建 ───────────────────────


def test_overwrite_persists_after_char_switch():
    """改字后切换字再回来不应回退。"""
    v, _ = _load_vproof("也草也")
    assert _select_char(v, "也")
    idx = v._gallery_model.index(0, 0)
    v._gallery_view.setCurrentIndex(idx)
    v._sync_gallery_entry(idx)

    assert v._gallery_direct_overwrite("好")
    # 目标字编辑已经同步落到模型；刷新钩子只负责重建当前视图。
    v._refresh_current_selection_context()

    # 切到"草"再切回原位置（"也"应该还剩一个）
    assert _select_char(v, "草")
    assert _select_char(v, "也")
    # 当前"也"只剩 1 个 entry（第二处仍是"也"），第一处已经变"好"
    txt = proof_display_text(v._session.pages[0].blocks[0].lines[0])
    assert txt == "好草也", f"模型未落盘：{txt!r}"


def test_char_index_migrates_after_overwrite():
    """改完字后 _char_svc 必须按新 final_text 重建：'也' 频次降，'好' 出现。"""
    v, _ = _load_vproof("也也也草")
    assert _select_char(v, "也")
    idx = v._gallery_model.index(0, 0)
    v._gallery_view.setCurrentIndex(idx)
    v._sync_gallery_entry(idx)

    # 改前：'也' 出现 3 次
    before = {c: len(v._char_svc.query(c)) for c in ["也", "好", "草"]}
    assert before["也"] == 3
    assert before["好"] == 0
    assert before["草"] == 1

    assert v._gallery_direct_overwrite("好")
    v._refresh_current_selection_context()

    after = {c: len(v._char_svc.query(c)) for c in ["也", "好", "草"]}
    assert after["也"] == 2, f"'也' 数量没下降：{after}"
    assert after["好"] == 1, f"'好' 没进 collection：{after}"
    assert after["草"] == 1, f"'草' 不应受影响：{after}"


def test_blank_via_backspace_also_persists():
    """Backspace 清空槽位也必须落到模型 + 重建。"""
    v, _ = _load_vproof("也草也")
    assert _select_char(v, "也")
    idx = v._gallery_model.index(0, 0)
    v._gallery_view.setCurrentIndex(idx)
    v._sync_gallery_entry(idx)

    assert v._gallery_direct_blank()
    v._refresh_current_selection_context()

    txt = proof_display_text(v._session.pages[0].blocks[0].lines[0])
    # 首字被替成空白；保留长度
    assert txt.startswith(" ") or txt[0] != "也", f"未清空：{txt!r}"
    # _char_svc 重建后"也"少一个
    assert len(v._char_svc.query("也")) == 1


def test_overwrite_emits_scoped_proof_change():
    """纵校目标编辑必须发出带 line scope 的 ProofChangeSet。"""
    v, _ = _load_vproof("也")
    changes = []
    v.proof_changed.connect(changes.append)
    assert _select_char(v, "也")
    idx = v._gallery_model.index(0, 0)
    v._gallery_view.setCurrentIndex(idx)
    v._sync_gallery_entry(idx)

    assert v._gallery_direct_overwrite("好")

    assert len(changes) == 1
    change = changes[0]
    assert change.text_changed is True
    assert change.needs_persist is True
    assert len(change.line_refs) == 1
    assert change.line_refs[0].line is v._session.pages[0].blocks[0].lines[0]
    assert change.line_refs[0].write_chars is True
