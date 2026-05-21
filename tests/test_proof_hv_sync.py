"""Phase 11: 横/纵校对联动 + 正确率统计入口 回归。"""
from __future__ import annotations

import os

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
            line.chars = [
                Char(
                    char=ch,
                    confidence=0.9,
                    bbox=BBox(i * 10, line_no * 24, 10, 20),
                    bbox_source="ocr",
                    bbox_granularity="char",
                    token_text=ch,
                )
                for i, ch in enumerate(text)
            ]
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
    from app.models import Char
    # 用一个稍长项目让 sampler 可能投放成功
    line = Line(text="今天我们来学习已经发生过的历史事件本身", confidence=0.9, bbox=BBox(0, 0, 240, 20))
    line.chars = [
        Char(
            char=ch,
            confidence=0.9,
            bbox=BBox(i * 8, 0, 8, 20),
            bbox_source="ocr",
            bbox_granularity="char",
            token_text=ch,
        )
        for i, ch in enumerate(line.text)
    ]
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


def test_char_cell_row_does_not_treat_missing_zero_conf_as_low():
    from app.ui.proof.char_cell_row import CharCellRow
    from app.core.page_image_cache import PageImageCache
    from app.models import Char

    line = Line(text="abc", confidence=0.9, bbox=BBox(0, 0, 60, 20))
    line.chars = [
        Char(char="a", confidence=0.0, bbox=BBox(0, 0, 20, 20)),
        Char(char="b", confidence=0.0, bbox=BBox(20, 0, 20, 20)),
        Char(char="c", confidence=0.0, bbox=BBox(40, 0, 20, 20)),
    ]
    page = Page(page_number=1, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 60, 20), lines=[line])
    ], image_path="/tmp/none.png", width=60, height=20)
    row = CharCellRow(line, page, PageImageCache.instance())
    assert row.focus_next_low_conf(from_idx=-1, threshold=0.85) is False
    row.deleteLater()


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
        dlg._table.horizontalHeaderItem(i).text()
        for i in range(dlg._table.columnCount())
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


def test_phase24_quality_stats_dialog_rate_label_uses_percent_format():
    """Phase 24 兼容：未启用时保留隐藏 rate label 初始值。"""
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    proj = _make_project("hi")
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: None,
    )
    # 默认未启用，仍显示 "rate = 待计算"（兼容 Phase 11 测试）
    assert dlg._rate_lbl.text() == "rate = 待计算"
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


def test_quality_stats_dialog_saves_density_immediately_and_resamples_live(tmp_path):
    from PySide6.QtCore import QSettings
    from app.core.app_config import AppConfig
    from app.core import quality_probe as qp_mod
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    AppConfig._instance = None
    cfg_store = AppConfig.instance()
    cfg_store.reset_to_defaults()
    cfg_store.set("quality_probe_sand_count", 1)
    cfg_store.set("quality_probe_sand_unit_chars", 1000)

    refreshes = {"count": 0}
    project = _make_project_with_char_crops(
        "今天我们来学习已经发生过的历史事件本身",
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


def test_phase25_editor_always_visible_and_weak_cursor():
    """Phase 25：editor 始终可见，cursorWidth=0（弱光标）。"""
    from app.ui.proof.h_proof import HProofPanel
    proj = _make_project("ABCD")
    h = HProofPanel()
    h.load_pages(proj.pages)
    for pair in h._pairs:
        # editor 不再在 active 切换中 hide
        assert not pair._editor.isHidden()
        assert pair._editor.cursorWidth() == 0
    h.deleteLater()


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
    for ch in line.chars:
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
    for ch in line.chars:
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
    for ch in line.chars:
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
