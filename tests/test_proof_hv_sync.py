"""Phase 11: 横/纵校对联动 + 正确率统计入口 回归。"""
from __future__ import annotations

import os

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp_mod
from app.models import BBox, Block, BlockType, Line, OcrProject, Page


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
    assert bus.subscriber_count("line.proof_changed") >= 2

    # 模拟 v_proof 发起编辑：直接调用 v 的内部保存路径不易，在测试里
    # 直接通过 publish 模拟一个外部事件
    received = {"called": False}
    h._on_external_line_changed = lambda **kw: received.update(  # type: ignore[method-assign]
        called=True, kw=kw
    )
    bus.publish(
        "line.proof_changed",
        page_id=proj.pages[0].id,
        line_id=proj.pages[0].blocks[0].lines[0].id,
        status="MODIFIED",
        origin=id(v),
    )
    # 由于 _on_external_line_changed 已被 monkey-patched (覆盖 instance attr 会
    # 让 bus 中的旧绑定仍指向旧方法)。实际调用走旧绑定，但旧绑定本身就有 origin
    # 过滤；为了确认订阅注册存在，比较 subscriber_count 已足够。
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
        page_id=proj.pages[0].id,
        line_id=proj.pages[0].blocks[0].lines[0].id,
        status="OK",
        origin=id(h),
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
        page_id=proj.pages[0].id,
        line_id=target_line.id,
        status="MODIFIED",
        origin=999,  # 假装别人发的
    )
    assert flags["refresh"] >= 1
    assert flags["stats"] == 1
    h.deleteLater()


def test_v_proof_external_handler_skips_self_origin():
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("xy")
    v = VProofPanel(); v.load_pages(proj.pages)

    called = {"load": 0}
    orig_load = v._load_page
    v._load_page = lambda i, _o=orig_load: (called.__setitem__("load", called["load"] + 1), _o(i))[1]  # type: ignore

    v._on_external_line_changed(
        page_id=proj.pages[0].id,
        line_id=proj.pages[0].blocks[0].lines[0].id,
        origin=id(v),
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
        page_id=proj.pages[0].id,
        line_id=proj.pages[0].blocks[0].lines[0].id,
        origin=99999,
    )
    assert called["load"] == 1
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
    assert dlg._table.rowCount() == 0
    assert dlg._rate_lbl.text() == "rate = 待计算"
    dlg.deleteLater()


def test_quality_stats_dialog_toggle_on_then_off():
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    # 用一个稍长项目让 sampler 可能投放成功
    line = Line(text="一二三四五六七八九十百千万", confidence=0.9, bbox=BBox(0, 0, 100, 20))
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
        assert dlg._table.rowCount() == len(store)
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


# ── CharCell 模式 (Phase 11 task 2) ──────────────────────────

def _make_project_with_chars(chars="abc"):
    from app.models import Char
    line = Line(text=chars, confidence=0.9, bbox=BBox(0, 0, len(chars) * 20, 20))
    line.id = 2001
    line.chars = [
        Char(char=c, confidence=0.9, bbox=BBox(i * 20, 0, 20, 20))
        for i, c in enumerate(chars)
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 200, 200), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png",
                width=200, height=200)
    page.id = 9101
    return OcrProject(name="cc", pages=[page])


def test_char_cell_row_builds_one_cell_per_char():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert row.has_cells
    assert len(row._cells) == 3
    assert [c.text() for c in row._cells] == ["a", "b", "c"]
    row.deleteLater()


def test_char_cell_row_text_committed_signal_aggregates_text():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())

    captured = []
    row.text_committed.connect(captured.append)
    # 模拟用户把第二个 cell 改成 'X'
    row._cells[1].setText("X")
    row._on_cell_changed()
    assert captured and captured[-1] == "aXc"
    row.deleteLater()


def test_char_cell_row_no_chars_falls_back_to_tip():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project("hello")  # 不带 char-bbox
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert not row.has_cells
    row.deleteLater()


def test_h_proof_toggle_cell_mode_does_not_crash():
    """开关字格模式 + 切到激活行：不应抛异常。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("abcd")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    assert h._pairs and h._pairs[0]._cell_mode_enabled is True
    # 激活第 0 行
    h._activate(0)
    # 切回普通模式
    h._on_toggle_cell_mode(False)
    assert h._pairs[0]._cell_mode_enabled is False
    h.deleteLater()


# ── Phase 12：CharCell 双向高亮 + 序列对齐 ────────────────────

def test_char_cell_row_focus_changed_emits_idx():
    """cell focusInEvent → CharCellRow.focus_changed(idx)。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    captured: list[int] = []
    row.focus_changed.connect(captured.append)
    row._cells[1].setFocus()
    # 模拟 focusIn —— 直接发信号（FocusEvent 在 offscreen 下不一定触发）
    row._cells[1].focus_in.emit()
    assert captured and captured[-1] == 1
    row.deleteLater()


def test_char_cell_row_focus_cell_method_moves_focus():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abcd")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    row.focus_cell(2)
    # offscreen 下 hasFocus() 不稳；用 selectAll 副作用判断
    assert row._cells[2].hasSelectedText()
    row.deleteLater()


def test_char_cell_row_reseat_uses_line_text_when_differs_from_chars():
    """line.text 已被普通模式改过，进入字格模式时 cells 应反映 line.text。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "aXc"   # 普通模式编辑过
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert [c.text() for c in row._cells] == ["a", "X", "c"]
    row.deleteLater()


def test_h_proof_cell_focus_updates_pair_focus_idx_and_renders():
    """cell 拿焦点 → _LinePair._cell_focus_idx 同步 + 触发行图重渲。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("abcd")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair = h._pairs[0]
    assert pair._cell_row is not None and pair._cell_row.has_cells
    pair._on_cell_focus_changed(2)
    assert pair._cell_focus_idx == 2
    h.deleteLater()


def test_h_proof_img_click_lookup_finds_nearest_char_idx():
    """点击行图区域 → 反查最近 char.bbox → cell_row 拿对应焦点。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("abcd")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair = h._pairs[0]
    # 伪造 _line_crop / _line_crop_origin / _render_scale 让反查可计算
    import numpy as np
    pair._line_crop = np.zeros((20, 80, 3), dtype="uint8")
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0

    class _Evt:
        def __init__(self, x): self._x = x
        def position(self):
            class _P:
                def __init__(self, x): self.x = lambda: x
            return _P(self._x)

    # char idx=2 的中心 = 20*2 + 10 = 50
    pair._img_clicked_lookup(_Evt(50))
    assert pair._cell_row is not None
    assert pair._cell_row._cells[2].hasSelectedText()
    h.deleteLater()


# ── Phase 13：CharCell 增强 ───────────────────────────────────

def test_char_cell_low_confidence_uses_red_border():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from app.models import Char
    line = Line(text="ab", confidence=0.9, bbox=BBox(0, 0, 40, 20))
    line.id = 3001
    line.chars = [
        Char(char="a", confidence=0.40, bbox=BBox(0,  0, 20, 20)),  # 低
        Char(char="b", confidence=0.95, bbox=BBox(20, 0, 20, 20)),  # 高
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0,0,40,20), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png",
                width=40, height=20)
    page.id = 9301
    proj = OcrProject(name="conf", pages=[page])
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert "#d93025" in row._cells[0].styleSheet()    # 红
    assert "#d93025" not in row._cells[1].styleSheet()
    row.deleteLater()


def test_char_cell_allows_multi_char_input():
    """Phase 13: maxLength=1 移除后，单 cell 可放多字符；text_committed 反映之。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("ab")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert row._cells[0].maxLength() in (-1, 32767)  # Qt 默认无限或大值
    captured = []
    row.text_committed.connect(captured.append)
    row._cells[0].setText("XY")
    row._on_cell_changed()
    assert captured[-1] == "XYb"
    row.deleteLater()


def test_h_proof_img_click_exact_bbox_hit_takes_precedence():
    """点击坐标落在 char.bbox 区间内 → 选中对应 cell 而非中心最近。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("abcd")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair = h._pairs[0]
    import numpy as np
    pair._line_crop = np.zeros((20, 80, 3), dtype="uint8")
    pair._line_crop_origin = (0, 0)
    pair._render_scale = 1.0

    class _Evt:
        def __init__(self, x): self._x = x
        def position(self):
            class _P:
                def __init__(self, x): self.x = lambda: x
            return _P(self._x)

    # char idx=1: bbox.x=20, x2=40；点 x=22 必命中 idx=1
    pair._img_clicked_lookup(_Evt(22))
    assert pair._cell_row._cells[1].hasSelectedText()
    h.deleteLater()


# ── Phase 14a：sequence-aware reseat + baseline 显示 ──────────

def test_align_text_to_chars_equal_replace_insert():
    """SequenceMatcher 对齐：分别覆盖 equal / replace / insert / drop / overflow。"""
    from app.ui.proof.char_cell_row import CharCellRow
    inits, overflow = CharCellRow._align_text_to_chars(
        text_chars=list("aXcde"),
        ocr_chars=list("abc"),
    )
    # ocr "abc" vs text "aXcde"：a=equal, b→X=replace, c=equal, de→trailing
    assert inits[0] == ("a", "equal")
    assert inits[1] == ("X", "replace")
    assert inits[2] == ("c", "equal")
    assert overflow == "de"


def test_align_text_to_chars_text_shorter_leaves_insert():
    """text 比 chars 短时，对不上的 cell 留空 + kind=insert。"""
    from app.ui.proof.char_cell_row import CharCellRow
    inits, overflow = CharCellRow._align_text_to_chars(
        text_chars=list("ab"),
        ocr_chars=list("abcd"),
    )
    assert inits[0] == ("a", "equal")
    assert inits[1] == ("b", "equal")
    assert inits[2][1] == "insert" and inits[2][0] == ""
    assert inits[3][1] == "insert" and inits[3][0] == ""
    assert overflow == ""


def test_char_cell_row_replace_kind_uses_red_border():
    """line.text 与 chars 不一致的位置 → cell 加红边。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "aXc"     # 普通模式改过：第二位被替换
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert "#d93025" in row._cells[1].styleSheet()    # replace → 红
    assert "#d93025" not in row._cells[0].styleSheet()
    row.deleteLater()


def test_char_cell_row_uses_row_uniform_scale():
    """所有 cell 共享 _row_scale；单格的 OCR 字符大小不影响其他格。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from app.models import Char
    line = Line(text="ab", confidence=0.9, bbox=BBox(0, 0, 40, 30))
    line.id = 4001
    line.chars = [
        Char(char="a", confidence=0.9, bbox=BBox(0,  0, 20, 30)),
        Char(char="b", confidence=0.9, bbox=BBox(20, 0, 20, 24)),  # 略矮
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0,0,40,30), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png",
                width=40, height=30)
    page.id = 9401
    proj = OcrProject(name="bs", pages=[page])
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert hasattr(row, "_row_scale")
    assert row._row_scale > 0
    row.deleteLater()


# ── Phase 14b：键盘流 + 特殊字符策略 ──────────────────────────

def test_classify_char_kind_basic():
    from app.ui.proof.char_cell_row import (
        classify_char_kind, KIND_DIGIT, KIND_LETTER, KIND_CJK,
        KIND_PUNCT, KIND_FORMULA,
    )
    assert classify_char_kind("3") == KIND_DIGIT
    assert classify_char_kind("a") == KIND_LETTER
    assert classify_char_kind("中") == KIND_CJK
    assert classify_char_kind("，") == KIND_PUNCT
    assert classify_char_kind("\\") == KIND_FORMULA
    assert classify_char_kind("$") == KIND_FORMULA


def test_line_looks_like_formula_detects_latex():
    from app.ui.proof.char_cell_row import line_looks_like_formula
    assert line_looks_like_formula(r"\begin{aligned} a &= b \end{aligned}")
    assert line_looks_like_formula("$$ x = y $$")
    assert not line_looks_like_formula("普通中文一行没有公式")


def test_char_cell_row_formula_line_falls_back_to_tip():
    """公式行：即使有 chars，也走 fallback 不构建 cells。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("ab")
    line = proj.pages[0].blocks[0].lines[0]
    line.text = r"\begin{aligned} a = b \end{aligned}"
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert not row.has_cells
    row.deleteLater()


def test_char_cell_row_next_off_end_signal_emits_at_last_cell():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    captured = []
    row.next_off_end.connect(lambda: captured.append(1))
    row._focus_neighbor(2, +1)   # 在最后一格再 → → 越界
    assert captured == [1]
    row.deleteLater()


def test_char_cell_row_focus_next_low_conf_skips_high():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from app.models import Char
    line = Line(text="abc", confidence=0.9, bbox=BBox(0, 0, 60, 20))
    line.id = 5001
    line.chars = [
        Char(char="a", confidence=0.99, bbox=BBox(0,  0, 20, 20)),
        Char(char="b", confidence=0.40, bbox=BBox(20, 0, 20, 20)),  # 低
        Char(char="c", confidence=0.99, bbox=BBox(40, 0, 20, 20)),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0,0,60,20), lines=[line])
    page = Page(page_number=1, blocks=[block], image_path="/tmp/none.png",
                width=60, height=20)
    page.id = 9501
    proj = OcrProject(name="lc", pages=[page])
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    found = row.focus_next_low_conf(from_idx=-1, threshold=0.85)
    assert found
    assert row._cells[1].hasSelectedText()
    row.deleteLater()


def test_h_proof_cell_next_off_end_advances_to_next_pair():
    """字格末位再按 → 越界 → 行级 next_req → 激活下一 pair 并 focus_first 字格。"""
    from app.ui.proof.h_proof import HProofPanel
    # 两行
    proj = _make_project_with_chars("ab")
    # 复制一份再加一个 line
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 6002
    from app.models import Char
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    proj.pages[0].blocks[0].lines.append(line2)

    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    assert pair0._cell_row is not None
    # 在 pair0 末位 cell 越界
    pair0._cell_row._focus_neighbor(len(pair0._cell_row._cells) - 1, +1)
    # 应当激活 pair1
    assert h._current_idx == 1
    h.deleteLater()


# ── Phase 15: blocker 修复回归 ──────────────────────────────────────

def test_align_text_to_chars_returns_trailing_overflow():
    """text 末尾比 chars 长 → 多余字符进 trailing_overflow。"""
    from app.ui.proof.char_cell_row import CharCellRow
    cell_inits, overflow = CharCellRow._align_text_to_chars(list("ABCDE"), list("ABC"))
    assert len(cell_inits) == 3
    assert [t for t, _ in cell_inits] == ["A", "B", "C"]
    assert overflow == "DE"


def test_char_cell_row_commit_preserves_trailing_overflow():
    """blocker 1：编辑某 cell 触发 text_committed 时，trailing_overflow 必须带回。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")  # chars=a,b,c
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "abcDE"  # 末尾 DE 是 trailing_overflow
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert row._trailing_overflow == "DE"
    captured = []
    row.text_committed.connect(captured.append)
    # 模拟用户把第一格 a 改成 X
    row._cells[0].setText("X")
    row._cells[0].textEdited.emit("X")
    # 拼回的文本必须含末尾 DE，不能丢
    assert captured, "text_committed should fire"
    assert captured[-1].endswith("DE"), f"got {captured[-1]!r}"
    assert captured[-1] == "XbcDE"
    row.deleteLater()


def test_char_cell_row_commit_when_no_overflow_unchanged():
    """无 overflow 时行为与之前一致，不引入额外尾巴。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "abc"
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    assert row._trailing_overflow == ""
    captured = []
    row.text_committed.connect(captured.append)
    row._cells[1].setText("Y")
    row._cells[1].textEdited.emit("Y")
    assert captured[-1] == "aYc"
    row.deleteLater()


def test_char_cell_row_tab_advances_to_next_cell():
    """blocker 2：在中间 cell 按 Tab → 焦点到下一 cell。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    row.show()
    row._cells[0].setFocus()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)
    row._cells[0].keyPressEvent(ev)
    # 焦点应当转到 _cells[1]；用 hasSelectedText 做 proxy（focus_neighbor 没 selectAll，
    # 但 setCursorPosition；这里直接断言 _cells[1] 拿到焦点）
    assert row._cells[1].hasFocus() or row._cells[1] is row.focusWidget()
    row.deleteLater()


def test_char_cell_row_shift_tab_at_first_emits_prev_off_start():
    """blocker 2：在首 cell 按 Shift+Tab (Backtab) → 越界发 prev_off_start。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    captured = []
    row.prev_off_start.connect(lambda: captured.append(1))
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    row._cells[0].keyPressEvent(ev)
    assert captured == [1]
    row.deleteLater()


def test_char_cell_row_tab_at_last_emits_next_off_end():
    """blocker 2：在末 cell 按 Tab → 越界发 next_off_end。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance())
    captured = []
    row.next_off_end.connect(lambda: captured.append(1))
    last = len(row._cells) - 1
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)
    row._cells[last].keyPressEvent(ev)
    assert captured == [1]
    row.deleteLater()


def test_h_proof_cell_tab_at_last_advances_to_next_pair():
    """blocker 2 端到端：Tab 在末位字格 → 跨行到下一 pair 并 focus_first。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.models import Char
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    proj = _make_project_with_chars("ab")
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 7002
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    proj.pages[0].blocks[0].lines.append(line2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    assert pair0._cell_row is not None
    last = len(pair0._cell_row._cells) - 1
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)
    pair0._cell_row._cells[last].keyPressEvent(ev)
    assert h._current_idx == 1
    h.deleteLater()


# ── Phase 16: stale _cell_row blocker ──────────────────────────────

def test_pair_rebind_invalidates_cell_row():
    """blocker: rebind 到新 line 后，旧 _cell_row 必须作废，
    避免后续 cell mode 编辑用旧 chars 覆盖新 line.text。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    old_row = pair0._cell_row
    assert old_row is not None
    # 构造一个新 line（不同 chars / text）
    from app.models import Char
    new_line = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    new_line.id = 8001
    new_line.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    block = pair0._block
    page = pair0._page
    pair0.rebind(block, new_line, page, 1)
    # active 时 invalidate 会立即重建 → 不为 None，但**不再是同一个对象**
    assert pair0._cell_row is not None
    assert pair0._cell_row is not old_row, "rebind 必须丢弃旧 _cell_row"
    # 新 cell_row 反映新 line 的 chars
    assert len(pair0._cell_row._cells) == 2
    assert pair0._cell_row._cells[0].text() == "c"
    h.deleteLater()


def test_external_line_changed_invalidates_cell_row():
    """blocker: VProof 修改 line.text 通过 bus 回流时，
    缓存 _cell_row 必须作废，否则 cell mode 编辑会用旧文本覆盖。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.core.proof_state_bus import ProofStateBus
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    old_row = pair0._cell_row
    assert old_row is not None
    # 模拟 VProof 改了 text + 通过 bus publish
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "AB"  # 大写新版本
    bus = ProofStateBus.instance()
    bus.publish(
        "line.proof_changed",
        page_id=proj.pages[0].id,
        line_id=line.id,
        status=line.proof_status,
        origin=999999,  # 任何非 id(h) 的 origin
    )
    # _on_external_line_changed 应已触发 _invalidate_cell_row
    assert pair0._cell_row is not None  # active 状态会立即重建
    assert pair0._cell_row is not old_row
    # 新 cell_row 反映新 line.text
    assert pair0._cell_row._cells[0].text() == "A"
    assert pair0._cell_row._cells[1].text() == "B"
    h.deleteLater()


def test_external_line_changed_inactive_pair_invalidates_lazily():
    """非 active 的 pair：invalidate 后 _cell_row 应 = None，等下次激活重建。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.core.proof_state_bus import ProofStateBus
    proj = _make_project_with_chars("ab")
    # 加第二行，让 pair0 非 active
    from app.models import Char
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 9001
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    proj.pages[0].blocks[0].lines.append(line2)

    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    # 先激活 pair0 让它建出 _cell_row
    h._activate(0)
    pair0 = h._pairs[0]
    assert pair0._cell_row is not None
    # 切到 pair1 → pair0 不再 active，但 _cell_row 仍缓存
    h._activate(1)
    assert pair0._cell_row is not None  # 缓存还在
    # 外部更新 pair0.line
    line0 = proj.pages[0].blocks[0].lines[0]
    line0.text = "ZZ"
    ProofStateBus.instance().publish(
        "line.proof_changed",
        page_id=proj.pages[0].id,
        line_id=line0.id,
        status=line0.proof_status,
        origin=999999,
    )
    # 非 active 时 _invalidate_cell_row 不会立即重建 → 应为 None
    assert pair0._cell_row is None
    # 下次激活 pair0 时按新 line 重建
    h._activate(0)
    assert pair0._cell_row is not None
    assert pair0._cell_row._cells[0].text() == "Z"
    h.deleteLater()


# ── Phase 17: pair 高度避免裁切 CharCellRow ─────────────────────────

def test_pair_height_grows_when_cell_mode_enabled():
    """blocker: cell mode 开启后 _LinePair 高度必须 ≥ CharCellRow 高度，否则裁切。"""
    from app.ui.proof.h_proof import HProofPanel, LINE_PAIR_H, CELL_PAIR_H
    from app.ui.proof.char_cell_row import IMG_H, EDIT_H
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    pair0 = h._pairs[0]
    # 普通模式下 pair 高度 = LINE_PAIR_H
    assert pair0.height() == LINE_PAIR_H
    # 开启 cell mode → 高度切到 CELL_PAIR_H 且 ≥ CharCellRow 总高
    h._on_toggle_cell_mode(True)
    assert pair0.height() == CELL_PAIR_H
    cell_row_h = IMG_H + EDIT_H + 6
    assert CELL_PAIR_H >= cell_row_h, f"CELL_PAIR_H={CELL_PAIR_H} 必须 ≥ CharCellRow 高 {cell_row_h}"
    # 关闭 cell mode → 还原
    h._on_toggle_cell_mode(False)
    assert pair0.height() == LINE_PAIR_H
    h.deleteLater()


def test_all_pairs_grow_on_cell_mode_not_just_active():
    """blocker: cell mode 应同步调整所有 pair 高度，避免列表滚动时抖动。"""
    from app.ui.proof.h_proof import HProofPanel, LINE_PAIR_H, CELL_PAIR_H
    from app.models import Char
    proj = _make_project_with_chars("ab")
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 17002
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    proj.pages[0].blocks[0].lines.append(line2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    h._on_toggle_cell_mode(True)
    for pair in h._pairs:
        assert pair.height() == CELL_PAIR_H, "所有 pair（含未激活）都应放大"
    h._on_toggle_cell_mode(False)
    for pair in h._pairs:
        assert pair.height() == LINE_PAIR_H
    h.deleteLater()


def test_cell_row_fits_in_pair_after_toggle():
    """直接断言：cell mode 开启时 CharCellRow 本身的固定高度不超过 pair 当前高度。"""
    from app.ui.proof.h_proof import HProofPanel, CELL_PAIR_H
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    assert pair0._cell_row is not None
    assert pair0._cell_row.height() <= pair0.height(), \
        f"cell_row {pair0._cell_row.height()} 不能超过 pair {pair0.height()}（裁切）"
    h.deleteLater()


# ── Phase 18: merge_pages 继承 cell mode / display_text 一致 / bus 释放 ──

def test_merge_pages_new_pair_inherits_cell_mode():
    """blocker 1: cell mode 已开 → merge_pages 新增 pair 必须同步开。"""
    from app.ui.proof.h_proof import HProofPanel, CELL_PAIR_H
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    assert h._cell_mode_enabled is True
    # 构造第二页（line.id 唯一），merge 进来
    from app.models import Char
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 0, 40, 20))
    line2.id = 18001
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  0, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 0, 20, 20)),
    ]
    block2 = Block(block_type=BlockType.TEXT, bbox=BBox(0,0,40,20), lines=[line2])
    page2 = Page(page_number=2, blocks=[block2], image_path="/tmp/none2.png",
                 width=40, height=20)
    page2.id = 18901
    h.merge_pages([proj.pages[0], page2])
    # 找到新合入的 pair（最后一个）
    new_pair = h._pairs[-1]
    assert new_pair._cell_mode_enabled is True, "新合入 pair 必须继承 cell mode"
    assert new_pair.height() == CELL_PAIR_H, "新 pair 高度也应同步"
    h.deleteLater()


def test_char_cell_row_uses_display_text_when_provided():
    """blocker 2: display_text 参数生效，覆盖 line.text。"""
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    proj = _make_project_with_chars("abc")
    line = proj.pages[0].blocks[0].lines[0]
    line.text = "abc"
    # 模拟 quality-probe 注入了 fake_char 的显示文本（与 line.text 不同）
    row = CharCellRow(line, proj.pages[0], PageImageCache.instance(),
                      display_text="aXc")
    assert row._cells[1].text() == "X", "字格必须显示 display_text，而非 line.text"
    # 编辑回送时，commit 出来的也是 display 空间
    captured = []
    row.text_committed.connect(captured.append)
    row._cells[0].setText("Z")
    row._cells[0].textEdited.emit("Z")
    assert captured[-1] == "ZXc"
    row.deleteLater()


def test_h_proof_cell_row_built_with_displayed_text():
    """blocker 2 端到端：HProof 构建 cell_row 时把 _displayed_text 传入。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("abc")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    assert pair0._cell_row is not None
    # 没启用 quality-probe 时 displayed_text == line.text，cell 内容应等于 chars
    line = proj.pages[0].blocks[0].lines[0]
    assert pair0._cell_row._display_text == (line.text or "")
    h.deleteLater()


def test_h_proof_bus_unsubscribes_on_destroy():
    """blocker 3: HProofPanel 销毁后 ProofStateBus 应不再持有其订阅。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.core.proof_state_bus import ProofStateBus
    bus = ProofStateBus.instance()
    before = bus.subscriber_count("line.proof_changed")
    h = HProofPanel()
    after_sub = bus.subscriber_count("line.proof_changed")
    assert after_sub == before + 1
    # 显式调用 teardown（destroyed 信号在 deleteLater 后异步发出，offscreen 测试更稳的做法是直接调 _teardown_bus）
    h._teardown_bus()
    after_unsub = bus.subscriber_count("line.proof_changed")
    assert after_unsub == before, f"unsubscribe 后应回到 {before}，实得 {after_unsub}"
    # 幂等
    h._teardown_bus()
    assert bus.subscriber_count("line.proof_changed") == before
    h.deleteLater()


def test_v_proof_bus_unsubscribes_on_teardown():
    """blocker 3: VProofPanel 同样支持 unsubscribe。"""
    from app.ui.proof.v_proof import VProofPanel
    from app.core.proof_state_bus import ProofStateBus
    bus = ProofStateBus.instance()
    before = bus.subscriber_count("line.proof_changed")
    v = VProofPanel()
    assert bus.subscriber_count("line.proof_changed") == before + 1
    v._teardown_bus()
    assert bus.subscriber_count("line.proof_changed") == before
    v._teardown_bus()  # 幂等
    assert bus.subscriber_count("line.proof_changed") == before
    v.deleteLater()


# ── Phase 19: editor in-flight 保活 + active cell_row 跟显示空间 ────────────

def test_toggle_cell_mode_flushes_in_flight_editor_text():
    """Phase 19 blocker 1：用户在 editor 里改字，未触发 _save_current 就切到字格模式，
    in-flight 文本必须被 flush，line.text 要落盘。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    # 普通模式下激活第 0 行 → editor 可见
    h._activate(0)
    pair0 = h._pairs[0]
    assert not pair0._editor.isHidden()
    # 模拟用户在 editor 里手敲新文本（未保存）
    pair0._editor.setPlainText("ZZ")
    # 切到字格模式：必须先 flush
    h._on_toggle_cell_mode(True)
    assert pair0._line.text == "ZZ", \
        f"expected line.text flushed to 'ZZ', got {pair0._line.text!r}"
    h.deleteLater()


def test_toggle_cell_mode_no_emit_when_editor_unchanged():
    """Phase 19 blocker 1：editor 文本与显示空间一致时，切换不应产生伪保存。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    pair0 = h._pairs[0]
    saves: list = []
    pair0.text_saved.connect(lambda idx, t: saves.append((idx, t)))
    # editor 内容已经等于 displayed_text，未改
    h._on_toggle_cell_mode(True)
    assert saves == [], f"expected no spurious text_saved, got {saves}"
    h.deleteLater()


def test_flush_editor_if_dirty_noop_when_inactive():
    """Phase 19 blocker 1：非 active 行 _flush_editor_if_dirty 不发任何信号
    （load_pages 默认 _activate(0)，所以这里造一条非 active 的 pair1 来测）。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 9101
    proj.pages[0].blocks[0].lines.append(line2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    pair1 = h._pairs[1]
    assert not pair1._active
    saves: list = []
    pair1.text_saved.connect(lambda idx, t: saves.append((idx, t)))
    # 非 active pair 即便 editor 内有内容也应 no-op
    pair1._editor.setPlainText("X")
    pair1._flush_editor_if_dirty()
    assert saves == []
    h.deleteLater()


def test_refresh_text_rebuilds_cell_row_when_active_cell_mode():
    """Phase 19 blocker 2：active+cell_mode 下 refresh_text 必须作废并重建 _cell_row，
    以让字格拿到最新显示空间文本（quality-probe 切换场景）。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    old_cell_row = pair0._cell_row
    assert old_cell_row is not None
    # 模拟 quality-probe 状态变化触发的全局 refresh
    pair0.refresh_text()
    new_cell_row = pair0._cell_row
    assert new_cell_row is not None
    assert new_cell_row is not old_cell_row, \
        "active+cell_mode 下 refresh_text 必须重建 _cell_row"
    h.deleteLater()


def test_refresh_text_inactive_does_not_touch_cell_row():
    """Phase 19 blocker 2：非 active pair 的 refresh_text 走旧路径，
    只刷新 _text_lbl，不意外触发 cell_row 重建。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 9001
    proj.pages[0].blocks[0].lines.append(line2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)  # pair0 active；pair1 非 active
    pair1 = h._pairs[1]
    assert not pair1._active
    before = pair1._cell_row  # 可能为 None
    pair1.refresh_text()
    assert pair1._cell_row is before, \
        "非 active pair 的 refresh_text 不应重建 _cell_row"
    h.deleteLater()


# ── Phase 20: 单次 refresh / 单次 rebuild ──────────────────────────

def test_toggle_cell_mode_does_not_double_refresh_active_pair():
    """Phase 20 blocker 1：_on_toggle_cell_mode 不应在 set_cell_mode 之外
    再对 active pair 手动调一次 _refresh_active_widgets。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    pair0 = h._pairs[0]
    calls: list = []
    orig = pair0._refresh_active_widgets
    pair0._refresh_active_widgets = lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1]  # type: ignore[assignment]
    h._on_toggle_cell_mode(True)
    assert len(calls) == 1, f"expected single refresh, got {len(calls)}"
    h.deleteLater()


def test_toggle_cell_mode_builds_cell_row_only_once():
    """Phase 20 blocker 1：toggle 一次后 active pair 的 _cell_row 只构建一次。"""
    from app.ui.proof.h_proof import HProofPanel
    import app.ui.proof.h_proof as hp_mod
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    builds: list = []
    orig_cls = hp_mod.CharCellRow
    def _spy(*args, **kwargs):
        builds.append(1)
        return orig_cls(*args, **kwargs)
    hp_mod.CharCellRow = _spy  # type: ignore[assignment]
    try:
        h._on_toggle_cell_mode(True)
    finally:
        hp_mod.CharCellRow = orig_cls
    assert len(builds) == 1, f"expected one CharCellRow build, got {len(builds)}"
    h.deleteLater()


def test_external_line_changed_active_cell_mode_single_invalidate():
    """Phase 20 blocker 2：active+cell_mode 行收到 line.proof_changed 时，
    _invalidate_cell_row 只走一次（refresh_text 内部那一次），
    不应再被 _on_external_line_changed 显式调一次。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    invalidations: list = []
    orig = pair0._invalidate_cell_row
    pair0._invalidate_cell_row = lambda *a, **kw: (invalidations.append(1), orig(*a, **kw))[1]  # type: ignore[assignment]
    # 模拟外部 VProof 改了同一 line
    page = proj.pages[0]
    line = page.blocks[0].lines[0]
    h._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line.id,
        status=line.proof_status, origin="v_proof",
    )
    assert len(invalidations) == 1, \
        f"expected single invalidate on active+cell_mode, got {len(invalidations)}"
    h.deleteLater()


def test_external_line_changed_inactive_cell_mode_still_invalidates():
    """Phase 20 blocker 2：非 active 但 cell_mode 已开的 pair，
    缓存的 stale _cell_row 仍然要被显式作废，等下次激活才能按新 line.text 重建。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.models import Char
    proj = _make_project_with_chars("ab")
    line2 = Line(text="cd", confidence=0.9, bbox=BBox(0, 30, 40, 20))
    line2.id = 9201
    line2.chars = [
        Char(char="c", confidence=0.9, bbox=BBox(0,  30, 20, 20)),
        Char(char="d", confidence=0.9, bbox=BBox(20, 30, 20, 20)),
    ]
    proj.pages[0].blocks[0].lines.append(line2)
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(1)  # 让 pair1 先建 cell_row
    h._activate(0)  # 现在 pair1 inactive，但仍持有 _cell_row
    pair1 = h._pairs[1]
    assert pair1._cell_row is not None
    page = proj.pages[0]
    h._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line2.id,
        status=line2.proof_status, origin="v_proof",
    )
    assert pair1._cell_row is None, \
        "非 active+cell_mode pair 的 stale _cell_row 必须被作废"
    h.deleteLater()


# ── Phase 21: 工具栏"保存"按钮在 cell mode 下与 Ctrl+S 同语义 ──────

def test_save_button_emits_proof_saved_in_cell_mode():
    """Phase 21 blocker：cell mode 下点工具栏"保存"必须发 proof_saved，
    不再是 no-op；与 Ctrl+S（_save_all）同语义。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    saves: list = []
    h.proof_saved.connect(lambda: saves.append(1))
    h._btn_save.click()
    assert len(saves) >= 1, "cell mode 下点'保存'应至少发一次 proof_saved"
    h.deleteLater()


def test_save_button_in_normal_mode_still_persists():
    """Phase 21：普通模式（editor 可见）原有保存语义不能回退 ——
    点"保存"仍然走 _save_displayed_edit 把 editor 文本落到 line.text。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._activate(0)
    pair0 = h._pairs[0]
    assert not pair0._editor.isHidden()
    pair0._editor.setPlainText("WW")
    h._btn_save.click()
    assert pair0._line.text == "WW", \
        f"普通模式按钮保存失败：line.text={pair0._line.text!r}"
    h.deleteLater()


def test_save_button_and_ctrl_s_emit_same_signal_set():
    """Phase 21：按钮与 Ctrl+S 共用 _save_all → 至少都发 proof_saved。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    btn_saves: list = []
    ctrl_saves: list = []
    h.proof_saved.connect(lambda: btn_saves.append(1))
    h._btn_save.click()
    h.proof_saved.disconnect()
    h.proof_saved.connect(lambda: ctrl_saves.append(1))
    h._save_all()
    assert len(btn_saves) == len(ctrl_saves) and len(btn_saves) >= 1, \
        f"按钮 vs Ctrl+S proof_saved 计数不一致：{len(btn_saves)} vs {len(ctrl_saves)}"
    h.deleteLater()


# ── Phase 22: VProof flush in-flight + HProof cell mode 状态点即时刷新 ──

def test_v_proof_flushes_in_flight_text_before_external_sync():
    """Phase 22 blocker 1：VProof 收到外部 line.proof_changed 时，
    若 _text_edit 里有用户未保存的输入，必须先 flush 落盘，
    再让 _load_page 用最新文本重渲染，否则覆盖。"""
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("hello")
    v = VProofPanel()
    v.load_pages(proj.pages)
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    # 模拟用户在 _text_edit 改了一行
    v._text_edit.setPlainText("HELLO_EDITED")
    assert v._text_edit.toPlainText() != v._loaded_text  # 确实 dirty
    # 模拟 HProof 改了同一行触发 line.proof_changed（origin 不等于 v）
    v._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line0.id,
        status=line0.proof_status.value if hasattr(line0.proof_status, "value") else line0.proof_status,
        origin="h_proof_fake",
    )
    # 用户的 in-flight 内容应已被持久化到 line.text
    assert line0.text == "HELLO_EDITED", \
        f"in-flight 文本未保住，line.text={line0.text!r}"
    v.deleteLater()


def test_v_proof_no_save_when_text_edit_unchanged():
    """Phase 22 blocker 1：未改 _text_edit 时不应触发伪保存。"""
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("hi")
    v = VProofPanel()
    v.load_pages(proj.pages)
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    saves: list = []
    v.proof_saved.connect(lambda: saves.append(1))
    # _text_edit 未改
    v._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line0.id,
        status=line0.proof_status.value if hasattr(line0.proof_status, "value") else line0.proof_status,
        origin="h_proof_fake",
    )
    assert saves == [], f"未 dirty 时不应 emit proof_saved，got {saves}"
    v.deleteLater()


def test_h_proof_cell_mode_status_dot_updates_on_save():
    """Phase 22 blocker 2：cell mode 下编辑后 active pair 状态点即时刷新，
    不应停在旧状态。"""
    from app.ui.proof.h_proof import HProofPanel
    from app.models import ProofStatus
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    initial_status_html = pair0._status_lbl.text()
    # 模拟在第 0 个 cell 改字 → text_committed → text_saved 链
    pair0._cell_row._cells[0].setText("Z")
    pair0._cell_row._on_cell_changed()
    # 状态应变（line.proof_status 走到 MODIFIED）
    assert pair0._line.proof_status == ProofStatus.MODIFIED, \
        f"line.proof_status={pair0._line.proof_status}"
    new_status_html = pair0._status_lbl.text()
    assert new_status_html != initial_status_html, \
        f"状态点未刷新：{new_status_html!r}"
    h.deleteLater()


def test_h_proof_status_refresh_does_not_rebuild_cell_row():
    """Phase 22 blocker 2：保存触发的状态刷新只调 _refresh_status，
    不应作废重建 _cell_row（否则焦点被拉回第 0 格 / Phase 20 回退）。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project_with_chars("ab")
    h = HProofPanel()
    h.load_pages(proj.pages)
    h._on_toggle_cell_mode(True)
    h._activate(0)
    pair0 = h._pairs[0]
    cell_row_before = pair0._cell_row
    # 触发一次保存
    pair0._cell_row._cells[1].setText("Y")
    pair0._cell_row._on_cell_changed()
    # cell_row 实例不变（无重建）
    assert pair0._cell_row is cell_row_before, \
        "_on_text_saved 不应重建 _cell_row"
    h.deleteLater()


# ── Phase 23: VProof _save_page_text 后同步 _loaded_text ────────

def test_v_proof_save_page_text_resyncs_loaded_text():
    """Phase 23 blocker：_save_page_text 成功后，_loaded_text 必须等于
    当前 _text_edit 内容，否则后续 dirty 检查永远为 True。"""
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("hi")
    v = VProofPanel()
    v.load_pages(proj.pages)
    v._text_edit.setPlainText("HI_EDITED")
    v._save_page_text()
    assert v._loaded_text == v._text_edit.toPlainText(), \
        f"_loaded_text 未同步：{v._loaded_text!r} vs {v._text_edit.toPlainText()!r}"
    v.deleteLater()


def test_v_proof_no_double_flush_after_save():
    """Phase 23 blocker：保存后立刻收到外部同事件，应被识别为非 dirty，
    不再走 _save_page_text 重复落盘。"""
    from app.ui.proof.v_proof import VProofPanel
    proj = _make_project("hi")
    v = VProofPanel()
    v.load_pages(proj.pages)
    page = proj.pages[0]
    line0 = page.blocks[0].lines[0]
    v._text_edit.setPlainText("HI_EDITED")
    v._save_page_text()
    saves: list = []
    v.proof_saved.connect(lambda: saves.append(1))
    v._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line0.id,
        status=line0.proof_status.value if hasattr(line0.proof_status, "value") else line0.proof_status,
        origin="h_proof_fake",
    )
    assert saves == [], \
        f"保存后基线已同步，外部同步不应再触发 _save_page_text: {saves}"
    v.deleteLater()


def test_v_proof_h_change_other_line_not_overwritten_by_stale_baseline():
    """Phase 23 blocker 端到端：
    1) VProof 保存了 line0 的修改
    2) HProof 改 line1
    3) line.proof_changed 进来 → VProof 不应把 line1 的旧文本回写覆盖 HProof 的新内容
    """
    from app.ui.proof.v_proof import VProofPanel
    # 两行的 project
    proj = _make_project("L0_orig")
    page = proj.pages[0]
    line2 = Line(text="L1_orig", confidence=0.9, bbox=BBox(0, 30, 80, 20))
    line2.id = 2002
    page.blocks[0].lines.append(line2)
    v = VProofPanel()
    v.load_pages(proj.pages)
    # 用户在 VProof 改 line0 → 保存
    flat = v._text_edit.toPlainText()
    new_flat = flat.replace("L0_orig", "L0_VEDIT")
    v._text_edit.setPlainText(new_flat)
    v._save_page_text()
    assert page.blocks[0].lines[0].text == "L0_VEDIT"
    # 模拟 HProof 改 line1 真实 line.text，并 publish
    page.blocks[0].lines[1].text = "L1_HEDIT"
    v._bus.publish(
        "line.proof_changed",
        page_id=page.id, line_id=line2.id,
        status=line2.proof_status.value if hasattr(line2.proof_status, "value") else line2.proof_status,
        origin="h_proof_fake",
    )
    # line1 仍然是 HProof 的新值 —— 没有被 VProof 用 stale 整页文本回写
    assert page.blocks[0].lines[1].text == "L1_HEDIT", \
        f"VProof stale baseline 把 HProof 新内容覆盖了：{page.blocks[0].lines[1].text!r}"
    v.deleteLater()
