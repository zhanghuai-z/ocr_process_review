"""Phase 11: 横/纵校对联动 + 正确率统计入口 回归。"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status
from app.core.proof_line_mutation import set_line_proof_status, set_line_proof_text

import os

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core.proof_state import TOPIC_LINE_PROOF_CHANGED, ProofUpdateRequest
from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp_mod
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.models import BBox, Block, BlockOrigin, BlockType, Char, Line, OcrProject, Page
from app.models.ocr_character_observation import line_ocr_chars, replace_line_ocr_chars
from app.models.ocr_observation import replace_block_ocr_line_observations


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    qp_mod.reset_active_store()
    ProofStateBus.reset()


def _make_project(text="abcdef") -> OcrProject:
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, 100, 20))
    line.id = 1001  # ID 平时由 DB 分配，测试里手动指定才能让 line_id 匹配跑通
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 200), lines=[line])
    page = Page(
        page_number=1,
        blocks=[block],
        image_path="/tmp/none.png",
        width=100,
        height=200,
    )
    page.id = 9001
    return OcrProject(name="t", pages=[page])


def _make_project_with_char_crops(text: str, *, n_pages: int = 1, lines_per_page: int = 1) -> OcrProject:
    pages = []
    for page_no in range(1, n_pages + 1):
        lines = []
        for line_no in range(lines_per_page):
            line = Line(text=text, confidence=0.9, bbox=BBox(0, line_no * 24, len(text) * 10, 20))
            replace_line_ocr_chars(line, [
                Char(
                    char=ch,
                    confidence=0.9,
                    bbox=BBox(i * 10, line_no * 24, 10, 20),
                    bbox_source="ocr",
                    bbox_granularity="char",
                    token_text=ch,
                )
                for i, ch in enumerate(text)
            ])
            lines.append(line)
        block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, max(100, len(text) * 10), 200), lines=lines)
        pages.append(Page(page_number=page_no, blocks=[block], image_path="/tmp/none.png", width=300, height=200))
    return OcrProject(name="t", pages=pages)


# ── H ↔ V 双向同步 ──────────────────────────────────────────────

def test_v_proof_subscribes_to_bus_and_h_proof_publishes_with_origin():
    """两个 panel 实例化后都应订阅 line.proof_changed；origin 字段已带在 publish 中。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("hello")
    h = HProofPanel(); h.load_pages(proj.pages)
    v = VProofPanel(); v.load_pages(proj.pages)
    bus = ProofStateBus.instance()
    # h 和 v 各自订阅 一次
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) >= 2

    bus.publish_line_update(ProofUpdateRequest(
        page_id=proj.pages[0].id,
        line_id=proj.pages[0].blocks[0].lines[0].id,
        status="MODIFIED",
        origin=id(v),
    ))
    h.deleteLater(); v.deleteLater()


def test_h_proof_external_handler_skips_self_origin():
    """h_proof 收到 origin == id(self) 的事件应直接 return，不刷新 stats。"""
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("xy")
    h = HProofPanel(); h.load_pages(proj.pages)

    called = {"refresh": 0, "stats": 0}
    for p in h._pairs:
        orig_refresh = p.refresh_text
        p.refresh_text = lambda *a, **kw: (called.__setitem__("refresh", called["refresh"] + 1), orig_refresh(*a, **kw))[1]  # type: ignore
    orig_update = h._update_stats
    h._update_stats = lambda: (called.__setitem__("stats", called["stats"] + 1), orig_update())[1]  # type: ignore

    # origin == id(self) → 完全跳过
    h._on_external_line_changed(
        ProofUpdateRequest(
            page_id=proj.pages[0].id,
            line_id=proj.pages[0].blocks[0].lines[0].id,
            status="OK",
            origin=id(h),
        )
    )
    assert called["refresh"] == 0
    assert called["stats"] == 0
    h.deleteLater()


def test_h_proof_external_handler_refreshes_matching_line():
    """origin != id(self) 且 line.id 匹配时，对应 pair.refresh_text + _update_stats 被调用。"""
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("xy")
    h = HProofPanel(); h.load_pages(proj.pages)
    target_line = proj.pages[0].blocks[0].lines[0]

    flags = {"refresh": 0, "stats": 0}
    for p in h._pairs:
        orig = p.refresh_text
        p.refresh_text = lambda *a, _orig=orig, **kw: (flags.__setitem__("refresh", flags["refresh"] + 1), _orig(*a, **kw))[1]  # type: ignore
    orig_us = h._update_stats
    h._update_stats = lambda _o=orig_us: (flags.__setitem__("stats", flags["stats"] + 1), _o())[1]  # type: ignore

    h._on_external_line_changed(
        ProofUpdateRequest(
            page_id=proj.pages[0].id,
            line_id=target_line.id,
            line_uid="",
            status="MODIFIED",
            origin=999,  # 假装别人发的
        )
    )
    assert flags["refresh"] == 0
    h._do_external_refresh()
    assert flags["refresh"] >= 1
    assert flags["stats"] == 1
    h.deleteLater()


def test_h_proof_external_refresh_does_not_overwrite_dirty_current_line():
    """HProof dirty editor must not be silently replaced by an external same-line edit."""
    from app.models import ProofStatus
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("AAAA")
    h = HProofPanel(); h.load_pages(proj.pages)
    page = proj.pages[0]
    line = page.blocks[0].lines[0]
    pair = h._pairs[0]

    pair._editor.setPlainText("CCCC")
    set_line_proof_text(line, "DDDD")

    h._on_external_line_changed(
        ProofUpdateRequest(
            page_id=page.id,
            line_id=line.id,
            line_uid=line.uid,
            status=proof_status(line).value,
            origin=999,
        )
    )
    h._do_external_refresh()

    assert pair._editor.toPlainText() == "CCCC"
    assert proof_display_text(line) == "DDDD"
    assert pair.has_external_conflict()
    assert "冲突" in pair._status_lbl.text()

    h._on_confirmed(0)

    assert pair._editor.toPlainText() == "CCCC"
    assert proof_display_text(line) == "DDDD"
    assert proof_status(line) == ProofStatus.MODIFIED
    h.deleteLater()


def test_h_proof_save_current_rejects_stale_line_signature_before_external_refresh():
    """Even before the bus refresh arrives, HProof must not overwrite a changed line."""
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("AAAA")
    h = HProofPanel(); h.load_pages(proj.pages)
    line = proj.pages[0].blocks[0].lines[0]
    pair = h._pairs[0]
    changes: list = []
    h.proof_changed.connect(changes.append)

    pair._editor.setPlainText("CCCC")
    set_line_proof_text(line, "DDDD")

    result = h._save_current(silent=True)

    assert result.name == "CONFLICT"
    assert pair._editor.toPlainText() == "CCCC"
    assert proof_display_text(line) == "DDDD"
    assert pair.has_external_conflict()
    assert changes == []
    h.deleteLater()


def test_h_proof_flag_rejects_stale_line_signature_before_external_refresh():
    """Status-only actions must share the same stale-line guard as text edits."""
    from app.models import ProofStatus
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("AAAA")
    h = HProofPanel(); h.load_pages(proj.pages)
    line = proj.pages[0].blocks[0].lines[0]
    pair = h._pairs[0]
    changes: list = []
    h.proof_changed.connect(changes.append)

    set_line_proof_text(line, "DDDD")

    h._toggle_flag()

    assert proof_display_text(line) == "DDDD"
    assert proof_status(line) == ProofStatus.MODIFIED
    assert pair.has_external_conflict()
    assert changes == []
    h.deleteLater()


def test_h_proof_external_handler_matches_line_uid_before_rowid():
    """line rowid stale 时，横校应优先用稳定 line_uid 同步刷新。"""
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("xy")
    h = HProofPanel(); h.load_pages(proj.pages)
    target_line = proj.pages[0].blocks[0].lines[0]

    flags = {"refresh": 0, "stats": 0}
    for p in h._pairs:
        orig = p.refresh_text
        p.refresh_text = lambda *a, _orig=orig, **kw: (flags.__setitem__("refresh", flags["refresh"] + 1), _orig(*a, **kw))[1]  # type: ignore
    orig_us = h._update_stats
    h._update_stats = lambda _o=orig_us: (flags.__setitem__("stats", flags["stats"] + 1), _o())[1]  # type: ignore

    h._on_external_line_changed(
        ProofUpdateRequest(
            page_uid=proj.pages[0].uid,
            page_id=-1,
            line_uid=target_line.uid,
            line_id=-1,
            status="MODIFIED",
            origin=999,
        )
    )
    assert flags["refresh"] == 0
    h._do_external_refresh()
    assert flags["refresh"] >= 1
    assert flags["stats"] == 1
    h.deleteLater()


def test_h_proof_external_handler_debounces_duplicate_line_updates():
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("xy")
    h = HProofPanel(); h.load_pages(proj.pages)
    target_line = proj.pages[0].blocks[0].lines[0]

    flags = {"refresh": 0, "stats": 0}
    pair = h._pairs[0]
    orig_refresh = pair.refresh_text
    pair.refresh_text = lambda *a, _orig=orig_refresh, **kw: (flags.__setitem__("refresh", flags["refresh"] + 1), _orig(*a, **kw))[1]  # type: ignore
    orig_stats = h._update_stats
    h._update_stats = lambda _o=orig_stats: (flags.__setitem__("stats", flags["stats"] + 1), _o())[1]  # type: ignore

    for _ in range(5):
        h._on_external_line_changed(
            ProofUpdateRequest(
                page_id=proj.pages[0].id,
                line_id=target_line.id,
                line_uid=target_line.uid,
                status="MODIFIED",
                origin=999,
            )
        )

    assert flags == {"refresh": 0, "stats": 0}
    h._do_external_refresh()
    assert flags["refresh"] == 1
    assert flags["stats"] == 1
    h.deleteLater()


def test_h_proof_render_clears_pending_external_refresh_before_rebuild():
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project("xy")
    h = HProofPanel(); h.load_pages(proj.pages)
    target_line = proj.pages[0].blocks[0].lines[0]

    h._on_external_line_changed(
        ProofUpdateRequest(
            page_id=proj.pages[0].id,
            line_id=target_line.id,
            line_uid="",
            status="MODIFIED",
            origin=999,
        )
    )
    assert h._session.has_pending_external_refresh is True

    # Simulate a full view rebuild before the debounce timer flushes. The new
    # page deliberately reuses the same rowid-only identity so a stale request
    # would refresh the wrong row if the queue survived the rebuild.
    next_proj = _make_project("ab")
    h.load_pages(next_proj.pages)
    assert h._session.has_pending_external_refresh is False

    flags = {"refresh": 0}
    pair = h._pairs[0]
    orig_refresh = pair.refresh_text
    pair.refresh_text = lambda *a, _orig=orig_refresh, **kw: (flags.__setitem__("refresh", flags["refresh"] + 1), _orig(*a, **kw))[1]  # type: ignore
    h._do_external_refresh()
    assert flags["refresh"] == 0
    h.deleteLater()


def test_v_proof_external_handler_skips_self_origin():
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("xy")
    v = VProofPanel(); v.load_pages(proj.pages)

    called = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (called.__setitem__("load", called["load"] + 1), _o(i))[1]  # type: ignore

    v._on_external_line_changed(
        ProofUpdateRequest(
            page_id=proj.pages[0].id,
            line_id=proj.pages[0].blocks[0].lines[0].id,
            status="MODIFIED",
            origin=id(v),
        )
    )
    assert called["load"] == 0
    v.deleteLater()


def test_v_proof_external_handler_reloads_current_page_when_line_matches():
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("abc")
    v = VProofPanel(); v.load_pages(proj.pages)

    called = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (called.__setitem__("load", called["load"] + 1), _o(i))[1]  # type: ignore

    v._on_external_line_changed(
        ProofUpdateRequest(
            page_id=proj.pages[0].id,
            line_id=proj.pages[0].blocks[0].lines[0].id,
            line_uid="",
            status="MODIFIED",
            origin=99999,
        )
    )
    # vproof-direct-overwrite-residual round 11: external refresh is debounced
    # via QTimer now; flush it synchronously to keep the assertion semantics.
    v._do_external_refresh()
    assert called["load"] == 1
    v.deleteLater()


def test_v_proof_external_handler_matches_line_uid_before_rowid():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("abc")
    v = VProofPanel(); v.load_pages(proj.pages)

    called = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (called.__setitem__("load", called["load"] + 1), _o(i))[1]  # type: ignore

    page = proj.pages[0]
    line = page.blocks[0].lines[0]
    v._on_external_line_changed(
        ProofUpdateRequest(
            page_uid=page.uid,
            page_id=-1,
            line_uid=line.uid,
            line_id=-1,
            status="MODIFIED",
            origin=99999,
        )
    )
    v._do_external_refresh()
    assert called["load"] == 1
    v.deleteLater()


def test_v_proof_external_refresh_updates_offscreen_page_char_index_without_reloading_current():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project_with_char_crops("AA", n_pages=2)
    page1, page2 = proj.pages
    page1.id = 9101
    page2.id = 9102
    page2.page_number = 2
    line2 = page2.blocks[0].lines[0]
    set_line_proof_text(line2, "BB")
    for char in line_ocr_chars(line2):
        char.char = "B"
        char.token_text = "B"

    v = VProofPanel()
    v.load_pages(proj.pages)
    assert v._session.current_page_index() == 0
    assert any(entry.page_uid == page2.uid for entry in v._char_svc.query("B"))

    calls = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (calls.__setitem__("load", calls["load"] + 1), _o(i))[1]  # type: ignore

    set_line_proof_text(line2, "CC")
    for char in line_ocr_chars(line2):
        char.char = "C"
        char.token_text = "C"

    v._on_external_line_changed(
        ProofUpdateRequest(
            page_uid=page2.uid,
            page_id=page2.id,
            line_uid=line2.uid,
            line_id=line2.id,
            status=proof_status(line2).value,
            origin=99999,
        )
    )
    v._do_external_refresh()

    assert calls["load"] == 0
    assert v._session.current_page_index() == 0
    assert not any(entry.page_uid == page2.uid for entry in v._char_svc.query("B"))
    assert any(entry.page_uid == page2.uid for entry in v._char_svc.query("C"))
    v.deleteLater()


# ── 正确率统计 对话框 ──────────────────────────────────────────

def test_quality_stats_dialog_opens_and_shows_no_active_store():
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    proj = _make_project("hello")
    refresh_calls = []
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: refresh_calls.append(True),
    )
    # 默认 store 未启用
    assert dlg._switch_status.text().startswith("当前：未启用")
    assert dlg.detail_table().rowCount() == 0
    assert dlg.quality_state.enabled is False
    dlg.deleteLater()


def test_quality_stats_dialog_toggle_on_then_off(monkeypatch):
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    from PySide6.QtWidgets import QMessageBox
    from app.models import Char
    # 防止 dialog 在零样本场景弹出阻塞模态
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    # 文本中刻意包含已知互为近形的字（己/已/巳、体/休、拼/并）以保证 sampler 能投放
    line = Line(text="今天已学己事拼并体休巳过本身", confidence=0.9, bbox=BBox(0, 0, 240, 20))
    replace_line_ocr_chars(line, [
        Char(
            char=ch,
            confidence=0.9,
            bbox=BBox(i * 8, 0, 8, 20),
            bbox_source="ocr",
            bbox_granularity="char",
            token_text=ch,
        )
        for i, ch in enumerate(line.text)
    ])
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 200), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png",
                width=100, height=200)
    proj = OcrProject(name="t", pages=[page] * 3)

    refresh_calls = []
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: refresh_calls.append(True),
    )
    dlg._on_toggle(True)
    store = qp_mod.get_active_store()
    if store is None or len(store) == 0:
        # 采样可能因配置而 0 — 此场景下 dialog 应自动 reset_active_store
        assert qp_mod.get_active_store() is None or len(qp_mod.get_active_store()) == 0
    else:
        assert dlg._switch_status.text().startswith("当前：已启用")
        assert dlg.detail_table().rowCount() == len(store)
        # 关闭
        dlg._on_toggle(False)
        assert qp_mod.get_active_store() is None
        assert dlg._switch_status.text().startswith("当前：未启用")
    assert refresh_calls, "refresh_panels_cb 应至少被调用一次"
    dlg.deleteLater()


# ── source check：h_proof 工具栏不再有 评测 按钮 ─────────────

def test_h_proof_no_longer_has_quality_toolbar_buttons():
    from pathlib import Path
    src = Path("app/ui/proof/h_proof.py").read_text(encoding="utf-8")
    bad = [ln for ln in src.splitlines()
           if "_btn_quality_toggle" in ln or "_btn_quality_report" in ln
           or "_on_toggle_quality_probe" in ln
           or "_on_show_quality_report" in ln]
    assert bad == [], (
        f"h_proof 不应再有评测按钮残留：{bad}"
    )


def test_main_window_has_quality_stats_menu_action():
    from pathlib import Path
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    assert "正确率统计" in src
    assert "_show_quality_stats" in src
    assert "QualityStatsDialog" in src


def test_h_proof_bus_unsubscribes_on_destroy():
    """blocker 3: HProofPanel 销毁后 ProofStateBus 应不再持有其订阅。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.core.proof_state_bus import ProofStateBus
    bus = ProofStateBus.instance()
    before = bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED)
    h = HProofPanel()
    after_sub = bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED)
    assert after_sub == before + 1
    # 显式调用 teardown（destroyed 信号在 deleteLater 后异步发出，offscreen 测试更稳的做法是直接调 _teardown_bus）
    h._teardown_bus()
    after_unsub = bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED)
    assert after_unsub == before, f"unsubscribe 后应回到 {before}，实得 {after_unsub}"
    # 幂等
    h._teardown_bus()
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == before
    h.deleteLater()


def test_v_proof_bus_unsubscribes_on_teardown():
    """blocker 3: VProofPanel 同样支持 unsubscribe。"""
    from app.ui.proof.v_proof import VProofPanel
    from app.core.proof_state_bus import ProofStateBus
    bus = ProofStateBus.instance()
    before = bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED)
    v = VProofPanel()
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == before + 1
    v._teardown_bus()
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == before
    v._teardown_bus()  # 幂等
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == before
    v.deleteLater()


# ── editor in-flight 保活 + 显示空间保存 ────────────────────────

def test_save_button_in_normal_mode_still_persists():
    """Phase 21：普通模式（editor 可见）原有保存语义不能回退 ——
    点"保存"仍然走 ProofEditService 把 editor 文本落到 final_text。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    pair0 = h._pairs[0]
    assert not pair0._editor.isHidden()
    pair0._editor.setPlainText("WW")
    h._btn_save.click()
    assert proof_final_text(pair0._line) == "WW", \
        f"普通模式按钮保存失败：final_text={proof_final_text(pair0._line)!r}"
    h.deleteLater()


def test_v_proof_ocr_text_is_read_only_reference():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("hello")
    v = VProofPanel()
    v.load_pages(proj.pages)

    assert v._text_edit.isReadOnly()
    assert v._text_edit.toPlainText() == "hello\n"
    v.deleteLater()


def test_v_proof_no_save_when_text_edit_unchanged():
    """Phase 22 blocker 1：未改 _text_edit 时不应触发伪保存。"""
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("hi")
    v = VProofPanel()
    v.load_pages(proj.pages)
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    changes: list = []
    v.proof_changed.connect(changes.append)
    # _text_edit 未改
    v._bus.publish_line_update(ProofUpdateRequest(
        page_id=page.id, line_id=line0.id,
        status=proof_status(line0).value if hasattr(proof_status(line0), "value") else proof_status(line0),
        origin=999,
    ))
    assert changes == [], f"未 dirty 时不应 emit proof_changed，got {changes}"
    v.deleteLater()


def test_v_proof_refresh_reference_context_without_model_write():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("hi")
    line0 = proj.pages[0].blocks[0].lines[0]
    v = VProofPanel()
    v.load_pages(proj.pages)
    v._text_edit.setPlainText("HI_EDITED")
    result = v._refresh_reference_context()

    assert result is False
    assert proof_display_text(line0) == "hi"
    assert v._text_edit.toPlainText() == "hi\n"
    assert v._session.loaded_text == "hi\n"
    v.deleteLater()


def test_v_proof_no_double_flush_after_save():
    """VProof 参考文本刷新不应制造 proof_changed。"""
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("hi")
    v = VProofPanel()
    v.load_pages(proj.pages)
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    v._text_edit.setPlainText("HI_EDITED")
    v._refresh_reference_context()
    changes: list = []
    v.proof_changed.connect(changes.append)
    v._bus.publish_line_update(ProofUpdateRequest(
        page_id=page.id, line_id=line0.id,
        status=proof_status(line0).value if hasattr(proof_status(line0), "value") else proof_status(line0),
        origin=999,
    ))
    assert changes == [], \
        f"刷新后基线已同步，外部同步不应触发持久化: {changes}"
    v.deleteLater()


def test_v_proof_external_refresh_does_not_write_reference_text_to_model():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("L0_orig")
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    line1 = Line(text="L1_orig", confidence=0.9, bbox=BBox(0, 30, 80, 20))
    line1.id = 2002
    page.blocks[0].lines.append(line1)
    replace_block_ocr_line_observations(page.blocks[0].uid, list(page.blocks[0].lines))
    v = VProofPanel()
    v.load_pages(proj.pages)
    v._text_edit.setPlainText("STALE_PAGE_TEXT\n")

    set_line_proof_text(line1, "L1_HEDIT")
    v._bus.publish_line_update(ProofUpdateRequest(
        page_id=page.id,
        line_id=line1.id,
        status=proof_status(line1).value if hasattr(proof_status(line1), "value") else proof_status(line1),
        origin=999,
    ))
    v._do_external_refresh()

    assert proof_display_text(line0) == "L0_orig"
    assert proof_final_text(line1) == "L1_HEDIT"
    assert v._text_edit.toPlainText().startswith("L0_orig\nL1_HEDIT")
    v.deleteLater()


def test_v_proof_external_refresh_updates_reference_from_model():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("AAAA")
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    replace_line_ocr_chars(line0, [
        Char(char="A", confidence=0.9, bbox=BBox(i * 10, 0, 10, 20))
        for i in range(4)
    ])

    v = VProofPanel()
    v.load_pages(proj.pages)
    changes: list = []
    v.proof_changed.connect(changes.append)

    set_line_proof_text(line0, "DDDD")
    for char in line_ocr_chars(line0):
        char.char = "D"
    v._bus.publish_line_update(ProofUpdateRequest(
        page_id=page.id,
        line_id=line0.id,
        status=proof_status(line0).value if hasattr(proof_status(line0), "value") else proof_status(line0),
        origin=999,
    ))
    v._do_external_refresh()

    assert proof_final_text(line0) == "DDDD"
    assert proof_display_text(line0) == "DDDD"
    assert [char.char for char in line_ocr_chars(line0)] == ["D", "D", "D", "D"]
    assert v._text_edit.toPlainText().startswith("DDDD")
    assert v._session.loaded_text == v._text_edit.toPlainText()
    assert changes == []
    v.deleteLater()


def test_v_proof_cross_page_gallery_switch_ignores_reference_text_mutation():
    from app.ui.proof.v_proof import VProofPanel

    page1 = _make_project_with_char_crops("AAAA").pages[0]
    page1.id = 9101
    page2 = _make_project_with_char_crops("BBBB").pages[0]
    page2.id = 9102
    page2.page_number = 2
    page2.image_path = "/tmp/vproof-cross-page-2.png"
    page2.blocks[0].lines[0].id = 91002

    v = VProofPanel()
    v.load_pages([page1, page2])
    line1 = page1.blocks[0].lines[0]
    v._text_edit.setPlainText("CCCC\n")

    entry = v._char_svc.query("B")[0]
    v._highlight_char_in_viewer(entry)

    assert proof_display_text(line1) == "AAAA"
    assert v._session.current_page_index() == 1
    assert v._text_edit.toPlainText().startswith("BBBB")
    v.deleteLater()


def test_v_proof_refresh_context_ignores_structural_reference_mutation():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("AAAA")
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    line1 = Line(text="BBBB", confidence=0.9, bbox=BBox(0, 30, 80, 20))
    line1.id = 93002
    page.blocks[0].lines.append(line1)
    replace_block_ocr_line_observations(page.blocks[0].uid, list(page.blocks[0].lines))

    v = VProofPanel()
    v.load_pages(proj.pages)
    v._text_edit.setPlainText("CC\nCC\nBBBB\n")

    assert v._refresh_reference_context() is False
    assert proof_display_text(line0) == "AAAA"
    assert proof_display_text(line1) == "BBBB"
    assert v._text_edit.toPlainText().startswith("AAAA\nBBBB")
    v.deleteLater()


def test_v_proof_refresh_context_restores_block_separators_from_model():
    from app.ui.proof.v_proof import VProofPanel

    line0 = Line(text="AAAA", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    line0.id = 94001
    line1 = Line(text="BBBB", confidence=0.9, bbox=BBox(0, 40, 80, 20))
    line1.id = 94002
    block0 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 30), lines=[line0])
    block1 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 40, 100, 30), lines=[line1])
    page = Page(
        page_number=1,
        blocks=[block0, block1],
        image_path="/tmp/vproof-separator-slot.png",
        width=120,
        height=90,
    )
    page.id = 9401

    v = VProofPanel()
    v.load_pages([page])
    assert v._text_edit.toPlainText() == "AAAA\n\nBBBB\n"

    v._text_edit.setPlainText("AAAA\nBBBB\nCCCC\n")

    assert v._refresh_reference_context() is False
    assert proof_display_text(line0) == "AAAA"
    assert proof_display_text(line1) == "BBBB"
    assert v._text_edit.toPlainText() == "AAAA\n\nBBBB\n"
    v.deleteLater()


def test_v_proof_refresh_context_does_not_partially_write_stale_slots():
    from app.ui.proof.v_proof import VProofPanel

    line0 = Line(text="AAAA", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    line0.id = 95001
    line1 = Line(text="BBBB", confidence=0.9, bbox=BBox(0, 40, 80, 20))
    line1.id = 95002
    line2 = Line(text="EEEE", confidence=0.9, bbox=BBox(0, 80, 80, 20))
    line2.id = 95003
    block0 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 30), lines=[line0])
    block1 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 40, 100, 30), lines=[line1])
    block2 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 80, 100, 30), lines=[line2])
    page = Page(
        page_number=1,
        blocks=[block0, block1, block2],
        image_path="/tmp/vproof-stale-slot-owner.png",
        width=120,
        height=120,
    )
    page.id = 9501

    v = VProofPanel()
    v.load_pages([page])
    assert v._text_edit.toPlainText() == "AAAA\n\nBBBB\n\nEEEE\n"

    # 保持 line id 序列不变，但让最后一个 session slot 的 block owner 失效。
    block2.lines.remove(line2)
    block1.lines.append(line2)
    replace_block_ocr_line_observations(block1.uid, [line1, line2])
    replace_block_ocr_line_observations(block2.uid, [])
    v._text_edit.setPlainText("CCCC\n\nDDDD\n\nFFFF\n")
    changes: list = []
    v.proof_changed.connect(changes.append)

    assert v._refresh_reference_context() is False
    assert proof_display_text(line0) == "AAAA"
    assert proof_display_text(line1) == "BBBB"
    assert proof_display_text(line2) == "EEEE"
    assert changes == []
    assert v._text_edit.toPlainText() == "AAAA\n\nBBBB\nEEEE\n"
    v.deleteLater()


def test_v_proof_undo_action_cancels_atomically_when_line_owner_is_stale():
    from app.ui.proof.v_proof import VProofPanel

    line0 = Line(text="AAAA", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    line0.id = 96001
    line1 = Line(text="BBBB", confidence=0.9, bbox=BBox(0, 40, 80, 20))
    line1.id = 96002
    line2 = Line(text="EEEE", confidence=0.9, bbox=BBox(0, 80, 80, 20))
    line2.id = 96003
    block0 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 30), lines=[line0])
    block1 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 40, 100, 30), lines=[line1])
    block2 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 80, 100, 30), lines=[line2])
    page = Page(
        page_number=1,
        blocks=[block0, block1, block2],
        image_path="/tmp/vproof-undo-stale-slot-owner.png",
        width=120,
        height=120,
    )
    page.id = 9601

    v = VProofPanel()
    v.load_pages([page])
    entry = v._char_svc.query("A")[0]
    assert v._apply_replacement_to_selected("C", fallback_entry=entry) == 1
    assert proof_display_text(line0) == "CAAA"
    block0.lines.remove(line0)
    replace_block_ocr_line_observations(block0.uid, [])
    changes: list = []
    v.proof_changed.connect(changes.append)

    assert v._undo_vproof_edit() is True
    assert proof_final_text(line0) == "CAAA"
    assert line0 not in block0.lines
    assert line0 not in block1.lines
    assert proof_display_text(line1) == "BBBB"
    assert proof_display_text(line2) == "EEEE"
    assert changes == []
    assert "撤销已取消" in v._status_lbl.text()
    v.deleteLater()


def test_v_proof_undo_action_failure_preserves_current_page_and_editor_state():
    from app.ui.proof.v_proof import VProofPanel

    page1 = _make_project_with_char_crops("AAAA").pages[0]
    page1.id = 9701
    page1.image_path = "/tmp/vproof-restore-fail-page-1.png"
    page1.blocks[0].lines[0].id = 97001
    page2 = _make_project_with_char_crops("BBBB").pages[0]
    page2.id = 9702
    page2.page_number = 2
    page2.image_path = "/tmp/vproof-restore-fail-page-2.png"
    page2.blocks[0].lines[0].id = 97002

    v = VProofPanel()
    v.load_pages([page1, page2])
    entry = v._char_svc.query("A")[0]
    assert v._apply_replacement_to_selected("C", fallback_entry=entry) == 1
    assert v._vproof_undo_stack
    replacement = Line(text="CCCC", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    replacement.id = 97003
    replace_line_ocr_chars(replacement, [
        Char(char="C", confidence=0.9, bbox=BBox(i * 10, 0, 10, 20))
        for i in range(4)
    ])
    page1.blocks[0].lines[0] = replacement

    assert v._safe_load_page(1) is True
    before_idx = v._session.current_page_index()
    before_text = v._text_edit.toPlainText()
    before_loaded_key = v._session.current_page_key

    assert v._undo_vproof_edit() is True

    assert v._session.current_page_index() == before_idx == 1
    assert v._text_edit.toPlainText() == before_text == "BBBB\n"
    assert v._session.current_page_key == before_loaded_key
    assert "撤销已取消" in v._status_lbl.text()
    v.deleteLater()


def test_v_proof_text_edit_is_not_a_user_edit_surface():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("AAAA")
    v = VProofPanel()
    v.load_pages(proj.pages)

    assert v._text_edit.isReadOnly()
    assert v._text_edit.toPlainText() == "AAAA\n"
    v.deleteLater()


def test_v_proof_edit_bubble_ctrl_z_stays_native_and_does_not_model_undo():
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("AAAA")
    v = VProofPanel()
    v.load_pages(proj.pages)
    entry = v._char_svc.query("A")[0]
    assert v._apply_replacement_to_selected("B", fallback_entry=entry) == 1
    assert len(v._vproof_undo_stack) == 1
    v._edit_bubble_input.setText("AB")

    consumed = v.eventFilter(v._edit_bubble_input, QKeyEvent(
        QEvent.Type.KeyPress,
        int(Qt.Key.Key_Z),
        Qt.KeyboardModifier.ControlModifier,
        "z",
    ))

    assert consumed is False
    assert len(v._vproof_undo_stack) == 1
    assert v._vproof_redo_stack == []
    v.deleteLater()


def test_v_proof_refresh_quality_probe_state_reloads_reference_without_model_write():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project("AAAA")
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]

    v = VProofPanel()
    v.load_pages(proj.pages)
    v._text_edit.setPlainText("CCCC\n")

    v.refresh_quality_probe_state()

    assert proof_display_text(line0) == "AAAA"
    assert v._text_edit.toPlainText().startswith("AAAA")
    assert v._session.loaded_text == v._text_edit.toPlainText()
    v.deleteLater()


# ── Phase 24: UI/UX 重构 (上图下字 / 共享页面目录 / 右侧工具栏 / 纯功能正确率) ──

def test_phase24_h_proof_pair_uses_vertical_image_above_text():
    """Phase 24 blocker 1：_LinePair 必须把图像放在文本上方。
    Phase 25：取消 _text_lbl / _txt_row（editor 始终可见，直接放在 _content 第 1 项）。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project("hi")
    h = HProofPanel()
    h.load_pages(proj.pages)
    pair0 = h._pairs[0]
    assert hasattr(pair0, "_content"), "_LinePair 应有 _content 容器"
    from PySide6.QtWidgets import QVBoxLayout
    content_layout = pair0._content.layout()
    assert isinstance(content_layout, QVBoxLayout), \
        "_content 必须是 QVBoxLayout（上图下字）"
    # proof-layout-collections 第 1 任务：第三行文本（AlignmentRibbon）已去掉，
    # _content 现在只有：第 0 项 = 行图像，第 1 项 = editor。
    assert content_layout.itemAt(0).widget() is pair0._img_lbl, \
        "_img_lbl 必须位于上方"
    assert content_layout.itemAt(1).widget() is pair0._editor, \
        "_editor 必须紧贴行图下方（无第三行解释层）"
    assert content_layout.count() == 2, \
        "_content 仅含 image + editor 两项"
    h.deleteLater()


def test_phase24_h_proof_left_uses_shared_page_directory_list():
    """Phase 24 blocker 2：左侧页面目录复用 PageDirectoryList，
    与版面分析 LayoutPanel._page_list 同类。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.ui.recognize.layout_panel import LayoutPanel
    from app.ui.widgets.page_directory import PageDirectoryList

    h = HProofPanel()
    lp = LayoutPanel()
    assert isinstance(h._page_dir, PageDirectoryList)
    assert isinstance(lp._page_list, PageDirectoryList)
    # 共用同一个类（类对象 identity）
    assert type(h._page_dir) is type(lp._page_list)
    h.deleteLater()
    lp.deleteLater()


def test_phase24_quality_stats_dialog_table_has_corrected_column():
    """Phase 24 blocker 4：弹窗表格新增"是否修正"列（真字/假字/是否修正/观察）。"""
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    proj = _make_project("hi")
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: None,
    )
    headers = [
        dlg.detail_table().horizontalHeaderItem(i).text()
        for i in range(dlg.detail_table().columnCount())
    ]
    assert "真字" in headers, f"缺失真字列：{headers}"
    assert "假字" in headers, f"缺失假字列：{headers}"
    assert "是否修正" in headers, f"缺失是否修正列：{headers}"
    # 按钮文案是"开始统计"（不再叫"启用"）
    assert dlg._btn_toggle.text() == "开始统计"
    # 不再有大段"公式"说明（Phase 11 旧 GroupBox 残留）
    from PySide6.QtWidgets import QGroupBox
    assert not dlg.findChildren(QGroupBox), \
        "Phase 24：纯功能化对话框不应再有 GroupBox 包装的说明区"
    dlg.deleteLater()


def test_phase24_quality_stats_dialog_exposes_typed_disabled_state():
    """未启用时通过 quality_state 暴露 typed 状态。"""
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    proj = _make_project("hi")
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: None,
    )
    assert dlg.quality_state.enabled is False
    assert dlg.quality_state.density_text.startswith("当前密度：")
    dlg.deleteLater()


def test_quality_stats_dialog_replaces_explanatory_note_with_status_feedback():
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    proj = _make_project("今天我们来学习已经发生过的历史事件本身")
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: None,
    )
    note = dlg._scope_note.text()
    assert "状态：未启用" in note
    assert "自动保存" in note
    assert "不是全量字符错误率" not in note
    assert dlg._density_status.text().startswith("当前密度：")
    assert dlg.windowTitle() == "抽样字符校对观察"
    dlg.deleteLater()


def test_quality_stats_dialog_exposes_sand_density_controls():
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    dlg = QualityStatsDialog(
        project_provider=lambda: _make_project("今天我们来学习已经发生过的历史事件本身"),
        refresh_panels_cb=lambda: None,
    )

    dlg._sand_count_spin.setValue(3)
    idx = dlg._sand_unit_combo.findData(10000)
    dlg._sand_unit_combo.setCurrentIndex(idx)

    assert dlg._sand_count_spin.value() == 3
    assert dlg._sand_unit_combo.currentData() == 10000
    assert dlg._sand_unit_combo.currentText() == "每万字"
    dlg.deleteLater()


def test_quality_stats_dialog_saves_density_immediately_and_resamples_live(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox
    from app.core.app_config import AppConfig
    from app.core import quality_probe as qp_mod
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    AppConfig._instance = None
    cfg_store = AppConfig.instance()
    cfg_store.reset_to_defaults()
    cfg_store.set("quality_probe_sand_count", 1)
    cfg_store.set("quality_probe_sand_unit_chars", 1000)

    refreshes = {"count": 0}
    project = _make_project_with_char_crops(
        "今天已学己事拼并体休巳过本身免兔末未鸟乌",
        n_pages=3,
        lines_per_page=8,
    )
    dlg = QualityStatsDialog(
        project_provider=lambda: project,
        refresh_panels_cb=lambda: refreshes.update(count=refreshes["count"] + 1),
    )
    try:
        dlg._on_toggle(True)
        initial_store = qp_mod.get_active_store()
        assert initial_store is not None
        initial_count = len(initial_store)

        dlg._sand_count_spin.setValue(50)

        assert int(cfg_store.get("quality_probe_sand_count")) == 50
        live_store = qp_mod.get_active_store()
        assert live_store is not None
        assert len(live_store) > initial_count
        assert "已保存并生效" in dlg._density_status.text()
        assert "候选池" in dlg._scope_note.text()
        assert refreshes["count"] >= 2
    finally:
        dlg.deleteLater()
        qp_mod.reset_active_store()
        cfg_store.reset_to_defaults()
        AppConfig._instance = None



# ── Phase 25 ─────────────────────────────────────────────────────────────────


def test_phase25_no_cell_mode_button():
    """Phase 25：字格模式按钮已彻底移除。"""
    from app.ui.proof.h_proof import HProofPanel
    h = HProofPanel()
    assert not hasattr(h, "_btn_cell_mode")
    assert not hasattr(h, "_cell_mode_enabled")
    assert not hasattr(h, "_on_toggle_cell_mode")
    h.deleteLater()


def test_phase25_no_prev_next_buttons():
    """Phase 25：右栏不再有冗余的"上一行/下一行"按钮（已有快捷键 + 滚动 + 回车确认覆盖）。"""
    from app.ui.proof.h_proof import HProofPanel
    h = HProofPanel()
    assert not hasattr(h, "_btn_prev")
    assert not hasattr(h, "_btn_next")
    h.deleteLater()


def test_phase25_no_page_combo():
    """Phase 25：右栏页面下拉框已移除，页面选择由左侧页面目录唯一负责。"""
    from app.ui.proof.h_proof import HProofPanel
    h = HProofPanel()
    assert not hasattr(h, "_page_combo")
    assert not hasattr(h, "_progress_lbl")
    assert not hasattr(h, "_shortcuts_lbl")
    h.deleteLater()


def test_phase25_save_button_text_is_just_save():
    """Phase 25：保存按钮文案改为简洁"保存"。"""
    from app.ui.proof.h_proof import HProofPanel
    h = HProofPanel()
    assert h._btn_save.text() == "保存"
    h.deleteLater()


def test_hproof_builds_text_proof_units_aligned_with_model_items():
    from app.core.proof_atom import ProofAtomKind as AtomKind
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    proj = _make_project_with_char_crops("甲乙", lines_per_page=2)
    h = HProofPanel()
    h.load_pages(proj.pages)

    units = [pair._unit for pair in h._pairs]
    assert [unit.kind for unit in units] == [
        ProofUnitKind.TEXT,
        ProofUnitKind.TEXT,
    ]
    assert [
        (unit.block, unit.line, unit.page, unit.line_index)
        for unit in units
    ] == [
        (projection.block, projection.line, projection.page, projection.line_index)
        for projection in h._session.projections
    ]
    assert h._pairs[0]._unit is units[0]
    assert h._session.projections[0].view_model(source="test").text == units[0].display_text
    assert [atom.kind for atom in units[0].atoms] == [AtomKind.CHAR, AtomKind.CHAR]
    h.deleteLater()


def test_hproof_proof_unit_carries_word_atom_for_multichar_token():
    from app.core.proof_atom import ProofAtomKind as AtomKind
    from app.ui.proof.h_proof import HProofPanel

    line = Line(
        text="PE/VC",
        confidence=0.9,
        bbox=BBox(0, 0, 60, 20),
        chars=[
            Char(
                char="PE/VC",
                confidence=0.9,
                bbox=BBox(0, 0, 60, 20),
                bbox_source="engcut",
                bbox_granularity="word",
                token_text="PE/VC",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 30), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=100, height=40)

    h = HProofPanel()
    h.load_pages([page])

    units = [pair._unit for pair in h._pairs]
    assert len(units) == 1
    assert [atom.kind for atom in units[0].atoms] == [AtomKind.WORD]
    assert units[0].atoms[0].text == "PE/VC"
    h.deleteLater()


def test_hproof_inline_formula_atom_visual_overlay_selects_raw_formula():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from app.core.proof_atom import ProofAtomKind as AtomKind
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    formula = "$ E=mc^2 $"
    line = Line(
        text=f"含{formula}",
        confidence=0.9,
        bbox=BBox(0, 0, 120, 24),
        chars=[
            Char(
                char="含",
                confidence=0.9,
                bbox=BBox(0, 0, 10, 24),
                bbox_source="ocr",
                bbox_granularity="char",
                token_text="含",
            ),
            Char(
                char=formula,
                confidence=1.0,
                bbox=BBox(12, 0, 80, 24),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text=formula,
            ),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 30), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=140, height=40)

    h = HProofPanel()
    h.load_pages([page])

    pair = h._pairs[0]
    assert pair._unit.kind == ProofUnitKind.TEXT
    assert [atom.kind for atom in pair._unit.atoms] == [AtomKind.CHAR, AtomKind.FORMULA]
    assert not pair._editor.has_visual_text_override()
    assert pair._editor.has_atom_visual_overlays()
    overlay = pair._editor.atom_visual_overlays()[0]
    assert pair._editor.toPlainText() == f"含{formula}"

    x = (overlay.left + overlay.right) / 2.0
    pair._editor.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, 10),
        QPointF(x, 10),
        QPointF(x, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    ))

    assert pair._editor.textCursor().selectedText() == formula
    assert pair._editor.toPlainText() == f"含{formula}"
    h.deleteLater()


def test_hproof_right_dock_updates_progress_and_status_counts():
    from app.models import ProofStatus
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project_with_char_crops("甲乙", lines_per_page=4)
    lines = proj.pages[0].blocks[0].lines
    set_line_proof_status(lines[0], ProofStatus.OK)
    set_line_proof_status(lines[1], ProofStatus.MODIFIED)
    set_line_proof_status(lines[2], ProofStatus.AUTO_FLAGGED)
    set_line_proof_status(lines[3], ProofStatus.UNCHECKED)

    h = HProofPanel()
    h.load_pages(proj.pages)

    assert h._current_scope_lbl.text() == "全部页面 · 正文"
    assert h._current_line_lbl.text() == "当前行 1 / 4"
    assert h._proof_progress_bar.value() == 2
    assert h._handled_lbl.text() == "已处理 2 / 4"
    assert h._confirmed_lbl.text() == "已确认 1"
    assert h._modified_lbl.text() == "已修改 1"
    assert h._flagged_lbl.text() == "疑点 1"
    assert h._pending_lbl.text() == "待确认 1"

    h._activate(2)
    assert h._current_line_lbl.text() == "当前行 3 / 4"
    h.deleteLater()


def test_hproof_debug_mode_banner_tracks_formula_filter():
    from app.ui.proof.h_proof import HProofPanel

    h = HProofPanel()
    assert h._mode_banner.isHidden()

    h._btn_debug_formula.setChecked(True)

    assert not h._mode_banner.isHidden()
    assert "公式调试视图" in h._mode_banner.text()
    assert h._current_scope_lbl.text() == "全部页面 · 公式调试"
    h.deleteLater()


def test_hproof_active_pair_exposes_stronger_visual_state():
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project_with_char_crops("甲乙", lines_per_page=2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    first, second = h._pairs

    assert first.property("active") is True
    assert "border:1px solid #7A7368" in first.styleSheet()
    assert "待" not in first._status_lbl.text()
    assert "#AFC0D8" in first._active_bar.styleSheet()

    h._activate(1)

    assert first.property("active") is False
    assert "border-bottom:1px solid #E7E2D8" in first.styleSheet()
    assert second.property("active") is True
    assert "border:1px solid #7A7368" in second.styleSheet()
    h.deleteLater()


def test_hproof_focus_depth_compresses_context_rows():
    from app.ui.proof.h_proof import (
        FAR_LINE_PAIR_H,
        HProofPanel,
        LINE_PAIR_H,
        NEAR_LINE_PAIR_H,
    )

    proj = _make_project_with_char_crops("甲乙", lines_per_page=4)
    h = HProofPanel()
    h.load_pages(proj.pages)

    assert [pair._focus_depth for pair in h._pairs] == [
        "active",
        "near",
        "far",
        "far",
    ]
    assert h._pairs[0].height() == LINE_PAIR_H
    assert h._pairs[1].height() == NEAR_LINE_PAIR_H
    assert h._pairs[2].height() == FAR_LINE_PAIR_H

    h._activate(2)

    assert [pair._focus_depth for pair in h._pairs] == [
        "far",
        "near",
        "active",
        "near",
    ]
    assert h._pairs[2].height() == LINE_PAIR_H
    h.deleteLater()


def test_hproof_debug_buttons_filter_formula_and_table_lines():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    normal_line = Line(text="正文", confidence=0.9, bbox=BBox(0, 0, 40, 12))
    inline_formula_line = Line(
        text="含$ A $公式",
        confidence=0.9,
        bbox=BBox(0, 16, 80, 12),
        chars=[
            Char(char="含", confidence=0.9, bbox=BBox(0, 16, 10, 12)),
            Char(
                char="$ A $",
                confidence=1.0,
                bbox=BBox(12, 16, 30, 12),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text="$ A $",
            ),
            Char(char="公", confidence=0.9, bbox=BBox(46, 16, 10, 12)),
            Char(char="式", confidence=0.9, bbox=BBox(58, 16, 10, 12)),
        ],
    )
    formula_number_line = Line(text="(1)", confidence=1.0, bbox=BBox(86, 32, 14, 12))
    formula_line = Line(text="$$ E=mc^2 $$", confidence=1.0, bbox=BBox(0, 48, 80, 12))
    line_less_formula = r"$$ \\frac{a+b}{c+d} = \\sum_{i=1}^{n} x_i $$"
    table_line = Line(text="表格OCR", confidence=0.8, bbox=BBox(0, 64, 80, 12))
    route_table_line = Line(
        text="表格子区",
        confidence=0.0,
        bbox=BBox(0, 80, 80, 12),
        review_flags=["hanwang_route_table"],
    )
    page = Page(image_path="/tmp/hproof-debug.png", width=120, height=120, page_number=1)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 0, 100, 30),
            lines=[normal_line, inline_formula_line],
        ),
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(86, 32, 14, 12),
            lines=[formula_number_line],
            source_label="formula_number",
        ),
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 48, 80, 12),
            lines=[formula_line],
            source_label="display_formula",
        ),
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 58, 100, 12),
            lines=[],
            source_label="display_formula",
            origin=BlockOrigin(source_label="display_formula", raw_index=0),
        ),
        Block(
            block_type=BlockType.TABLE,
            bbox=BBox(0, 64, 80, 12),
            lines=[table_line],
            source_label="table_region",
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 80, 80, 12),
            lines=[route_table_line],
        ),
    ]
    set_paddle_raw_layout_records(page, [
        {
            "block_label": "display_formula",
            "block_bbox": [0, 58, 100, 70],
            "block_content": line_less_formula,
        }
    ])

    h = HProofPanel()
    h.load_pages([page])
    assert [projection.line.text for projection in h._session.projections] == [
        "正文",
        "含$ A $公式",
    ]

    h._btn_debug_formula.setChecked(True)
    assert [projection.line.text for projection in h._session.projections] == [
        "含$ A $公式",
        "$$ E=mc^2 $$",
        line_less_formula,
    ]
    assert [pair._unit.kind for pair in h._pairs] == [
        ProofUnitKind.TEXT,
        ProofUnitKind.FORMULA,
        ProofUnitKind.FORMULA,
    ]
    assert [pair._debug_badge for pair in h._pairs] == ["公式", "公式", "公式"]
    assert not h._pairs[0]._editor.has_visual_text_override()
    assert h._pairs[0]._editor.has_atom_visual_overlays()
    assert h._pairs[1]._editor.has_visual_text_override()
    assert h._pairs[1]._editor.toPlainText() == "$$ E=mc^2 $$"
    assert h._pairs[2]._editor.has_visual_text_override()
    assert h._pairs[2].minimumWidth() > h._pairs[2]._line.bbox.w
    formula_pair = h._pairs[1]
    left_click = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(12, 10),
        QPointF(12, 10),
        QPointF(12, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    formula_pair._editor.mousePressEvent(left_click)
    assert formula_pair._editor.has_visual_text_override()
    formula_pair._img_clicked_lookup(left_click)
    assert formula_pair._editor.has_visual_text_override()
    formula_pair._editor.mousePressEvent(left_click)
    assert formula_pair._editor.has_visual_text_override()
    formula_pair._editor.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(12, 10),
        QPointF(12, 10),
        QPointF(12, 10),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert formula_pair._editor.has_visual_text_override()
    assert formula_pair._formula_source_panel is not None
    assert not formula_pair._formula_source_panel.isHidden()
    assert formula_pair._formula_source_popup is None

    h._btn_debug_formula.setChecked(False)
    h._btn_debug_table.setChecked(True)
    assert [projection.line.text for projection in h._session.projections] == [
        "表格OCR",
        "表格子区",
    ]
    assert [pair._unit.kind for pair in h._pairs] == [
        ProofUnitKind.TABLE,
        ProofUnitKind.TABLE,
    ]
    assert [pair._debug_badge for pair in h._pairs] == ["表格", "表格"]

    h._btn_debug_formula.setChecked(True)
    assert [projection.line.text for projection in h._session.projections] == [
        "含$ A $公式",
        "$$ E=mc^2 $$",
        line_less_formula,
        "表格OCR",
        "表格子区",
    ]
    h.deleteLater()


def test_hproof_formula_visual_uses_experimental_pixmap(monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from app.experimental import formula_rendering
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    calls: list[tuple[str, int]] = []

    def fake_render_formula_pixmap(text: str, *, target_height: int, **_kwargs):
        calls.append((text, target_height))
        pixmap = QPixmap(64, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        pixmap.setDevicePixelRatio(1.0)
        return SimpleNamespace(
            pixmap=pixmap,
            logical_width=64,
            logical_height=18,
            device_pixel_ratio=1.0,
        )

    monkeypatch.setattr(formula_rendering, "render_formula_pixmap", fake_render_formula_pixmap)
    line = Line(text="$$ E=mc^2 $$", confidence=0.9, bbox=BBox(0, 0, 90, 20))
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 90, 20),
        lines=[line],
        source_label="display_formula",
    )
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=120, height=40)

    h = HProofPanel()
    h.load_pages([page])
    h._btn_debug_formula.setChecked(True)

    pair = h._pairs[0]
    assert pair._unit.kind == ProofUnitKind.FORMULA
    assert calls == [("$$ E=mc^2 $$", 28)]
    assert pair._editor.has_visual_text_override()
    assert pair._editor._visual_pixmap_override is not None
    assert pair._editor.visual_text_content_width() == 80
    h.deleteLater()


def test_hproof_display_formula_source_editor_updates_rendered_source(monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from app.experimental import formula_rendering
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    rendered: list[str] = []

    def fake_render_formula_pixmap(text: str, *, target_height: int, **_kwargs):
        rendered.append(text)
        pixmap = QPixmap(80, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        pixmap.setDevicePixelRatio(1.0)
        return SimpleNamespace(
            pixmap=pixmap,
            logical_width=80,
            logical_height=18,
            device_pixel_ratio=1.0,
        )

    monkeypatch.setattr(formula_rendering, "render_formula_pixmap", fake_render_formula_pixmap)
    line = Line(text="$$ E=mc^2 $$", confidence=0.9, bbox=BBox(0, 0, 90, 20))
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(0, 0, 90, 20),
        lines=[line],
        source_label="display_formula",
    )
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=120, height=40)

    h = HProofPanel()
    h.load_pages([page])
    h._btn_debug_formula.setChecked(True)

    pair = h._pairs[0]
    assert pair._unit.kind == ProofUnitKind.FORMULA
    base_height = pair.height()
    pair._open_formula_source_editor(0, len(pair._editor.toPlainText()), QPoint(0, 0))
    assert pair._formula_source_panel is not None
    assert not pair._formula_source_panel.isHidden()
    assert pair._formula_source_popup is None
    assert pair.height() > base_height
    pair._apply_formula_source_text("$$ F=ma $$")

    assert pair._editor.toPlainText() == "$$ F=ma $$"
    assert pair._editor.has_visual_text_override()
    assert rendered[-1] == "$$ F=ma $$"
    h.deleteLater()


def test_hproof_inline_formula_source_editor_replaces_formula_span(monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from app.experimental import formula_rendering
    from app.ui.proof.h_proof import HProofPanel

    def fake_render_formula_pixmap(text: str, *, target_height: int, **_kwargs):
        pixmap = QPixmap(64, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        pixmap.setDevicePixelRatio(1.0)
        return SimpleNamespace(
            pixmap=pixmap,
            logical_width=64,
            logical_height=18,
            device_pixel_ratio=1.0,
        )

    monkeypatch.setattr(formula_rendering, "render_formula_pixmap", fake_render_formula_pixmap)
    formula = "$ E=mc^2 $"
    line = Line(
        text=f"含{formula}公式",
        confidence=0.9,
        bbox=BBox(0, 0, 140, 24),
        chars=[
            Char(char="含", confidence=0.9, bbox=BBox(0, 0, 10, 24), bbox_granularity="char"),
            Char(
                char=formula,
                confidence=1.0,
                bbox=BBox(12, 0, 80, 24),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text=formula,
            ),
            Char(char="公", confidence=0.9, bbox=BBox(94, 0, 10, 24), bbox_granularity="char"),
            Char(char="式", confidence=0.9, bbox=BBox(106, 0, 10, 24), bbox_granularity="char"),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 140, 30), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=160, height=40)

    h = HProofPanel()
    h.load_pages([page])

    pair = h._pairs[0]
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0
    pair._sync_editor_slot_geometry()
    before_overlay = pair._editor.atom_visual_overlays()[0]
    start = 1
    end = start + len(formula)
    pair._open_formula_source_editor(start, end, QPoint(0, 0))
    assert pair._formula_source_popup is not None
    assert pair._formula_source_panel is None
    pair._apply_formula_source_text(r"$ \\frac{a+b}{c+d}=F_{it} $")

    assert pair._editor.toPlainText() == r"含$ \\frac{a+b}{c+d}=F_{it} $公式"
    overlays = pair._editor.atom_visual_overlays()
    assert len(overlays) == 1
    assert overlays[0].kind == "formula"
    assert overlays[0].start == 1
    assert overlays[0].end == len(pair._editor.toPlainText()) - 2
    assert overlays[0].left == before_overlay.left
    assert overlays[0].right == before_overlay.right
    assert pair._editor.has_slot_geometry()
    h.deleteLater()


def test_hproof_formula_fallback_keeps_cjk_text_upright(monkeypatch):
    from app.ui.proof.h_proof import _render_formula_visual

    monkeypatch.setenv("OCR_EXPERIMENTAL_FORMULA_RENDER", "0")

    visual = _render_formula_visual("含$ A_t $公式", target_height=28)

    assert visual is not None
    assert visual.text == "含Aₜ公式"
    assert visual.kind == ""


def test_hproof_display_formula_cjk_fallback_can_open_source_panel(monkeypatch):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from app.ui.proof.h_proof import HProofPanel, ProofUnitKind

    monkeypatch.setenv("OCR_EXPERIMENTAL_FORMULA_RENDER", "0")
    line = Line(text="$$ 中文公式 A_t $$", confidence=0.9, bbox=BBox(0, 0, 120, 24))
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(0, 0, 120, 30),
        lines=[line],
        source_label="display_formula",
    )
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=160, height=40)

    h = HProofPanel()
    h.load_pages([page])
    h._btn_debug_formula.setChecked(True)

    pair = h._pairs[0]
    assert pair._unit.kind == ProofUnitKind.FORMULA
    assert pair._editor.has_visual_text_override()
    assert pair._editor._visual_text_kind == ""

    pair._editor.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(12, 10),
        QPointF(12, 10),
        QPointF(12, 10),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))

    assert pair._formula_source_panel is not None
    assert not pair._formula_source_panel.isHidden()
    pair._apply_formula_source_text("$$ 中文公式 B_t $$")
    assert pair._editor.toPlainText() == "$$ 中文公式 B_t $$"
    assert pair._editor.has_visual_text_override()
    h.deleteLater()


def test_hproof_formula_debug_ignores_superscript_marker_inline_formula():
    from app.ui.proof.h_proof import iter_unique_page_hproof_debug_lines

    marker_line = Line(
        text="注：$ ^{*} $说明",
        confidence=0.9,
        bbox=BBox(0, 0, 100, 12),
        chars=[
            Char(
                char="$ ^{*} $",
                confidence=1.0,
                bbox=BBox(20, 0, 24, 12),
                bbox_source="paddle_inline_formula",
                token_text="$ ^{*} $",
            )
        ],
        review_flags=["hanwang_route_inline_formula"],
    )
    true_formula_line = Line(
        text="其中 $ \\beta_t $ 显著",
        confidence=0.9,
        bbox=BBox(0, 16, 120, 12),
        chars=[
            Char(
                char="$ \\beta_t $",
                confidence=1.0,
                bbox=BBox(30, 16, 30, 12),
                bbox_source="paddle_inline_formula",
                token_text="$ \\beta_t $",
            )
        ],
        review_flags=["hanwang_route_inline_formula"],
    )
    marker_block_line = Line(
        text="$ ^{②} $",
        confidence=1.0,
        bbox=BBox(0, 32, 20, 12),
    )
    page = Page(image_path="/tmp/hproof-debug-marker.png", width=140, height=80, page_number=1)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 30), lines=[marker_line, true_formula_line]),
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 32, 20, 12),
            lines=[marker_block_line],
            source_label="inline_formula",
        ),
    ]

    rows = list(iter_unique_page_hproof_debug_lines(page, formulas=True))

    assert [line.text for _block, line, _idx in rows] == ["其中 $ \\beta_t $ 显著"]


def test_hproof_formula_debug_uses_layout_snapshot_type_over_runtime_projection():
    from app.models import LayoutBlockSnapshot, LayoutSnapshot, OcrPolicy
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.proof.h_proof import iter_unique_page_hproof_debug_lines

    line = Line(
        text="E=mc2",
        confidence=0.9,
        bbox=BBox(0, 0, 80, 12),
    )
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 12),
        lines=[line],
        source_label="text",
    )
    page = Page(image_path="/tmp/hproof-debug-snapshot.png", width=100, height=50, page_number=1)
    page.blocks = [block]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-hproof-debug",
        source_engine="test",
        source_run_id="run-hproof-debug",
        blocks=(
            LayoutBlockSnapshot(
                uid=block.uid,
                block_type=BlockType.EQUATION,
                bbox=block.bbox,
                order=0,
                source_label="display_formula",
                origin=BlockOrigin(source_label="display_formula"),
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
            ),
        ),
    ))

    rows = list(iter_unique_page_hproof_debug_lines(page, formulas=True))

    assert [(line.text, idx) for _block, line, idx in rows] == [("E=mc2", 0)]


def test_vproof_entry_lookup_uses_stable_block_uid_when_runtime_order_drifts():
    from app.models import LayoutBlockSnapshot, LayoutSnapshot, OcrPolicy
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.services.char_index_service import CharEntry
    from app.ui.proof.v_proof import VProofPanel

    line = Line(
        text="甲乙",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 12),
        chars=[
            Char("甲", 0.9, BBox(0, 0, 20, 12)),
            Char("乙", 0.9, BBox(20, 0, 20, 12)),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 12), lines=[line], order=9)
    page = Page(image_path="/tmp/vproof-entry-owner.png", width=100, height=50, page_number=1)
    page.blocks = [block]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-vproof-entry",
        source_engine="test",
        source_run_id="run-vproof-entry",
        blocks=(
            LayoutBlockSnapshot(
                uid=block.uid,
                block_type=BlockType.TEXT,
                bbox=block.bbox,
                order=2,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
        ),
    ))
    entry = CharEntry(
        char="乙",
        page_path=page.display_image_path,
        page_number=page.page_number,
        line=line,
        char_idx=1,
        bbox=BBox(20, 0, 20, 12),
        page_uid=page.uid,
        block_uid=block.uid,
        line_uid=line.uid,
        block_order=2,
        line_idx=0,
    )

    panel = VProofPanel()
    panel.load_pages([page])

    assert panel._block_for_entry(entry) is block
    assert panel._lookup_token_at_entry_position(entry) == "乙"
    assert panel._char_index_page_signature(page)[1][3] == 2
    panel.close()


def test_phase25_only_active_editor_visible_and_weak_cursor():
    """横校只在当前行显示文本编辑层；上下文行只显示图像层。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_char_crops("AB", lines_per_page=3)
    h = HProofPanel()
    h.load_pages(proj.pages)
    assert not h._pairs[0]._editor.isHidden()
    assert h._pairs[1]._editor.isHidden()
    assert h._pairs[2]._editor.isHidden()
    for pair in h._pairs:
        assert pair._editor.cursorWidth() == 0

    h._activate(1)

    assert h._pairs[0]._editor.isHidden()
    assert not h._pairs[1]._editor.isHidden()
    assert h._pairs[2]._editor.isHidden()
    h.deleteLater()


def test_slot_line_editor_ctrl_z_and_redo_restore_text():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from app.ui.proof.h_proof import _SlotLineEditor

    editor = _SlotLineEditor()
    editor.setPlainText("abc")
    editor._select_slot_index(1)

    editor.keyPressEvent(QKeyEvent(
        QKeyEvent.Type.KeyPress,
        int(Qt.Key.Key_X),
        Qt.KeyboardModifier.NoModifier,
        "X",
    ))
    assert editor.toPlainText() == "aXc"

    editor.keyPressEvent(QKeyEvent(
        QKeyEvent.Type.KeyPress,
        int(Qt.Key.Key_Z),
        Qt.KeyboardModifier.ControlModifier,
        "z",
    ))
    assert editor.toPlainText() == "abc"

    editor.keyPressEvent(QKeyEvent(
        QKeyEvent.Type.KeyPress,
        int(Qt.Key.Key_Y),
        Qt.KeyboardModifier.ControlModifier,
        "y",
    ))
    assert editor.toPlainText() == "aXc"
    editor.deleteLater()


def test_phase25_no_image_text_headers():
    """Phase 25：取消每行左侧的"图像 N / 文本 N" hdr 标签。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project("hi")
    h = HProofPanel()
    h.load_pages(proj.pages)
    pair0 = h._pairs[0]
    assert not hasattr(pair0, "_lbl_img_hdr")
    assert not hasattr(pair0, "_lbl_txt_hdr")
    assert not hasattr(pair0, "_text_lbl")
    h.deleteLater()


def test_phase25_low_conf_chars_get_extra_selection():
    """Phase 25：低置信度字符通过 ExtraSelection 高亮（非编辑器自身 setCharFormat）。"""
    from app.models import BBox, Char, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel
    page = Page(image_path="/tmp/p25-conf.png", width=100, height=100, page_number=1)
    line = Line(text="低高", confidence=0.5, bbox=BBox(0, 0, 40, 20), chars=[
        Char(char="低", confidence=0.3, bbox=BBox(0, 0, 20, 20)),
        Char(char="高", confidence=0.95, bbox=BBox(20, 0, 20, 20)),
    ])
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 20), lines=[line])]
    h = HProofPanel()
    h.load_pages([page])
    pair = h._pairs[0]
    sels = pair._editor.extraSelections()
    # 至少包含 1 条低置信度高亮（"低" 字 conf=0.3 < LOW_CONF）
    assert len(sels) >= 1
    h.deleteLater()


def test_hproof_confidence_verdict_falls_back_from_zero_char_to_line_score():
    from app.models import Char
    from app.ui.proof import char_verdict as cv
    from app.ui.proof.h_proof import HProofPanel

    line = Line(text="甲", confidence=92, bbox=BBox(0, 0, 20, 20), chars=[
        Char(char="甲", confidence=0.0, bbox=BBox(0, 0, 20, 20)),
    ])
    page = Page(page_number=1, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line])
    ], image_path="/tmp/hproof-conf.png", width=20, height=20)
    h = HProofPanel()
    h.load_pages([page])
    verdict = h._pairs[0]._classify_char_verdict(0)
    assert verdict is not None
    assert verdict.severity == cv.SEVERITY_UNVERIFIED
    assert "0.92" in verdict.evidence
    h.deleteLater()


def test_vproof_page_badge_uses_char_confidence_when_line_score_is_zero():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project_with_char_crops("甲乙丙")
    line = proj.pages[0].blocks[0].lines[0]
    line.confidence = 0.0
    for ch in line_ocr_chars(line):
        ch.confidence = 87

    v = VProofPanel()
    v.load_pages(proj.pages)
    assert v._conf_badge.text() == "87%"
    v.deleteLater()


def test_vproof_page_badge_marks_missing_confidence_unavailable():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project_with_char_crops("甲乙丙")
    line = proj.pages[0].blocks[0].lines[0]
    line.confidence = 0.0
    for ch in line_ocr_chars(line):
        ch.confidence = 0.0

    v = VProofPanel()
    v.load_pages(proj.pages)
    assert v._conf_badge.text() == "无置信度"
    v.deleteLater()


def test_vproof_entry_diagnostics_do_not_report_fake_zero_confidence():
    from app.ui.proof.v_proof import VProofPanel

    proj = _make_project_with_char_crops("甲乙丙")
    line = proj.pages[0].blocks[0].lines[0]
    line.confidence = 91
    for ch in line_ocr_chars(line):
        ch.confidence = 0.0

    v = VProofPanel()
    v.load_pages(proj.pages)
    entry = v._char_svc.query("甲")[0]
    assert v._format_entry_confidence(entry) == "0.91"
    line.confidence = 0.0
    assert v._format_entry_confidence(entry) == "缺失"
    v.deleteLater()


def test_phase25_horizontal_scroll_as_needed():
    """Phase 25：横校面板横向滚动条改为 AsNeeded 以支持自适应宽度。"""
    from PySide6.QtCore import Qt
    from app.ui.proof.h_proof import HProofPanel
    h = HProofPanel()
    assert h._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    h.deleteLater()


def test_hproof_active_row_scroll_keeps_text_left_aligned():
    """激活横校行时只做纵向可见，水平滚动保持左对齐。"""
    from app.ui.proof.h_proof import HProofPanel

    proj = _make_project_with_char_crops("abcdef", lines_per_page=3)
    h = HProofPanel()
    h.load_pages(proj.pages)
    hbar = h._scroll.horizontalScrollBar()
    hbar.setRange(0, 100)
    hbar.setValue(73)

    h._ensure_pair_visible_left_aligned(h._pairs[1])

    assert hbar.value() == 0
    h.deleteLater()


def test_hproof_slot_editor_hit_testing_uses_painted_slot_geometry():
    """横校文本自绘后，鼠标命中必须按可见槽位算，不能按 Qt 原生文本布局算。"""
    from app.ui.proof.h_proof import _RowEditor

    editor = _RowEditor()
    editor.setPlainText("abc")
    editor.set_slot_geometry([100.0, 160.0, 220.0], [20.0, 20.0, 20.0])

    assert editor._slot_index_for_x(160.0, nearest=False) == 1
    assert editor._slot_index_for_x(140.0, nearest=True) == 1

    editor._select_slot_index(1)
    cursor = editor.textCursor()
    assert cursor.selectionStart() == 1
    assert cursor.selectionEnd() == 2
    assert cursor.selectedText() == "b"
    editor.deleteLater()


def test_hproof_slot_editor_formula_visual_keeps_raw_text():
    """公式渲染只影响横校显示层，不把可读公式回写进 editor 原始文本。"""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from app.ui.proof.h_proof import (
        _SlotLineEditor,
        _is_punctuation_slot_text,
        _render_formula_display,
    )

    assert _render_formula_display(r"$$ E=mc^2 + \alpha_t $$") == "E = mc² + αₜ"
    assert _render_formula_display(r"I n c e n t i v e_t + P o s t_t") == "Incentiveₜ + Postₜ"
    assert _render_formula_display(r"$$ \frac{a+b}{c+d} $$") == "(a + b)⁄(c + d)"
    assert _render_formula_display(r"X _ t") == "Xₜ"
    assert _render_formula_display(r"X _ { t + 1 }") == "Xₜ₊₁"
    assert _is_punctuation_slot_text("，")
    assert not _is_punctuation_slot_text("甲")

    raw = "$$ E=mc^2 $$"
    editor = _SlotLineEditor()
    editor.setPlainText(raw)
    editor.set_visual_text_override(_render_formula_display(raw), kind="formula")

    assert editor.has_visual_text_override()
    assert editor.toPlainText() == raw

    editor.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(12, 10),
        QPointF(12, 10),
        QPointF(12, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    ))

    assert editor.has_visual_text_override()
    assert editor.textCursor().selectedText()
    editor.deleteLater()


def test_hproof_slot_editor_hit_testing_falls_back_without_bbox_slots():
    """chars 不可对齐时仍应能按可见文本点击选中，不能失焦成不可编辑。"""
    from app.ui.proof.h_proof import _SlotLineEditor

    editor = _SlotLineEditor()
    editor.setPlainText("abc")
    assert editor._slot_index_for_x(20.0, nearest=True) >= 0

    editor._select_slot_index(editor._slot_index_for_x(20.0, nearest=True))
    assert editor.textCursor().selectedText()
    editor.deleteLater()


def test_hproof_slot_geometry_clears_when_edit_breaks_alignment():
    """文本长度变动后必须立刻停用旧槽位几何，避免旧 bbox 映射继续误导高亮。"""
    from app.ui.proof.h_proof import HProofPanel

    project = _make_project_with_char_crops("abc")
    panel = HProofPanel()
    panel.load_pages(project.pages)
    pair = panel._pairs[0]

    pair._line_crop = object()
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0
    pair._sync_editor_slot_geometry()
    assert pair._editor.has_slot_geometry()

    pair._editor.setPlainText("abcd")
    assert not pair._editor.has_slot_geometry()
    panel.deleteLater()


def test_hproof_formula_slot_geometry_keeps_single_char_word_atom_visible():
    from app.ui.proof.h_proof import HProofPanel

    formula = "$ F $"
    text = f"取o；{formula}在"
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, 120, 24))
    replace_line_ocr_chars(line, [
        Char(char="取", confidence=0.9, bbox=BBox(0, 0, 18, 22), bbox_source="hanwang:micro_recblock", bbox_granularity="char", token_text="取"),
        Char(char="o", confidence=0.19, bbox=BBox(22, 0, 10, 22), bbox_source="hanwang:micro_recblock", bbox_granularity="char", token_text="o"),
        Char(char="；", confidence=0.4, bbox=BBox(36, 0, 8, 22), bbox_source="hanwang:micro_recblock", bbox_granularity="char", token_text="；"),
        Char(char=formula, confidence=0.0, bbox=BBox(48, 0, 40, 22), bbox_source="paddle_inline_formula", bbox_granularity="word", token_text=formula),
        Char(char="在", confidence=0.9, bbox=BBox(92, 0, 18, 22), bbox_source="hanwang:micro_recblock", bbox_granularity="char", token_text="在"),
    ])
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 140, 30), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png", width=160, height=40)

    panel = HProofPanel()
    panel.load_pages([page])
    pair = panel._pairs[0]
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0
    _overlays, centers, widths = pair._formula_atom_visual_data(text)

    assert centers is not None
    assert centers[text.index("o")] is not None
    assert widths is not None
    assert widths[text.index("；")] <= 8.0
    panel.deleteLater()


def test_phase25_page_dir_thumbnails_loaded():
    """Phase 25：左侧页面目录每项带有缩略图空间。"""
    from app.ui.proof.h_proof import HProofPanel
    from PySide6.QtWidgets import QLabel
    proj = _make_project("hi")
    h = HProofPanel()
    h.load_pages(proj.pages)
    item = h._page_dir.item(0)
    assert item is not None
    row = h._page_dir.itemWidget(item)
    assert row is not None
    thumb = row.findChild(QLabel, "pageThumb")
    assert thumb is not None
    sz = thumb.size()
    assert sz.width() >= 30 and sz.height() >= 40, \
        f"页面目录缩略图控件尺寸异常，实际 {sz}"
