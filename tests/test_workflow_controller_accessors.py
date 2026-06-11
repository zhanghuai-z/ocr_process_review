"""Regression tests for the narrow accessors and proof-sync ownership added to
``WorkflowController`` so that ``MainWindow`` no longer pokes
``self._controller.project.*`` or ``self._controller._project.*`` directly.

每个 case 都对应一处原 MainWindow 的直读，以保证抽出后语义 1:1 等价。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.controllers.workflow_controller import (
    WorkflowController, STEP_IMPORT, STEP_HPROOF, STEP_LAYOUT,
)
from app.models import BBox, Block, BlockType, Line, OcrProject, Page


# ── 辅助 ───────────────────────────────────────────────────────

def _line(text="abc"):
    return Line(text=text, confidence=0.95, bbox=BBox(0, 0, 100, 20))


def _block(lines, btype=BlockType.TEXT):
    return Block(block_type=btype, bbox=BBox(0, 0, 100, 200), lines=list(lines))


def _page(page_no, blocks, *, is_analyzed=False):
    p = Page(image_path=f"/tmp/p{page_no}.png", width=100, height=200,
             blocks=list(blocks), page_number=page_no)
    if is_analyzed:
        p.status = p.status  # placeholder; is_analyzed is a property
    return p


@pytest.fixture
def ctrl():
    c = WorkflowController()
    yield c
    c.close()


# ── 无项目 None-safe 退化路径 ───────────────────────────────────

def test_pages_returns_empty_list_when_no_project(ctrl):
    assert ctrl.pages == []


def test_has_pages_false_when_no_project(ctrl):
    assert ctrl.has_pages is False


def test_total_line_count_zero_when_no_project(ctrl):
    assert ctrl.total_line_count == 0


def test_is_fully_analyzed_false_when_no_project(ctrl):
    assert ctrl.is_fully_analyzed is False


def test_has_any_ocr_result_false_when_no_project(ctrl):
    assert ctrl.has_any_ocr_result is False


def test_cache_dir_falls_back_to_dot_when_no_project(ctrl):
    out = ctrl.cache_dir
    assert isinstance(out, Path)
    assert out.name == ".cache"


def test_page_number_at_returns_none_when_no_project(ctrl):
    assert ctrl.page_number_at(0) is None


# ── 有项目时 1:1 与原 inline 等价 ────────────────────────────────

def test_pages_returns_underlying_list(ctrl):
    p1 = _page(1, [_block([_line()])])
    p2 = _page(2, [_block([_line(), _line()])])
    ctrl._project = OcrProject(name="t", pages=[p1, p2])
    assert ctrl.pages == [p1, p2]
    assert ctrl.has_pages is True


def test_total_line_count_sums_pages(ctrl):
    p1 = _page(1, [_block([_line()])])              # 1 line
    p2 = _page(2, [_block([_line(), _line()])])     # 2 lines
    ctrl._project = OcrProject(name="t", pages=[p1, p2])
    # 与 sum(p.total_lines) 等价
    assert ctrl.total_line_count == sum(p.total_lines for p in [p1, p2])


def test_page_number_at_returns_correct_value(ctrl):
    p1 = _page(7, [_block([_line()])])
    p2 = _page(11, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1, p2])
    assert ctrl.page_number_at(0) == 7
    assert ctrl.page_number_at(1) == 11
    assert ctrl.page_number_at(2) is None
    assert ctrl.page_number_at(-1) is None


def test_cache_dir_uses_db_path_parent(ctrl, tmp_path):
    db = tmp_path / "demo.ocrproj"
    ctrl._project = OcrProject(name="t", pages=[], db_path=str(db))
    assert ctrl.cache_dir == tmp_path / ".cache"


def test_has_any_ocr_result_matches_project_property(ctrl):
    p1 = _page(1, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])
    assert ctrl.has_any_ocr_result is True
    ctrl._project = OcrProject(name="t2",
                               pages=[_page(1, [_block([], )])])
    assert ctrl.has_any_ocr_result is False


# ── proof-sync ownership ────────────────────────────────────────

class _StubPanel:
    """两个面板的最小 stub：只记录 load_pages / merge_pages 被怎么调用过。"""

    def __init__(self):
        self.load_calls = []
        self.merge_calls = []
        self.refresh_calls = 0

    def load_pages(self, pages):
        self.load_calls.append(list(pages))

    def merge_pages(self, pages):
        self.merge_calls.append(list(pages))

    def refresh_quality_probe_state(self):
        self.refresh_calls += 1


def test_sync_proof_panels_does_nothing_without_pages(ctrl):
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    ctrl.sync_proof_panels()
    assert h.load_calls == [] and h.merge_calls == []
    assert v.load_calls == [] and v.merge_calls == []


def test_sync_proof_panels_first_call_loads(ctrl):
    """从 0 → >0 行：首次 sync 应走 load_pages，并记录 line_count。"""
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    p1 = _page(1, [_block([_line(), _line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])

    ctrl.sync_proof_panels()
    assert len(h.load_calls) == 1
    assert len(v.load_calls) == 1
    assert h.merge_calls == [] and v.merge_calls == []
    assert ctrl._proof_loaded_line_count == ctrl.total_line_count


def test_sync_proof_panels_second_call_merges(ctrl):
    """已加载过后再增加行数：应走 merge_pages 而非 load_pages。"""
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    p1 = _page(1, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])
    ctrl.sync_proof_panels()      # 首次 load
    # 增加一行
    p1.blocks[0].lines.append(_line())
    ctrl.sync_proof_panels()      # 第二次：merge
    assert len(h.load_calls) == 1
    assert len(h.merge_calls) == 1
    assert len(v.merge_calls) == 1


def test_sync_proof_panels_skips_when_unchanged(ctrl):
    """行数未变化时 sync 应该 no-op（避免 OCR 进度回调里频繁刷新）。"""
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    p1 = _page(1, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])
    ctrl.sync_proof_panels()
    h.load_calls.clear(); v.load_calls.clear()
    ctrl.sync_proof_panels()
    assert h.load_calls == [] and h.merge_calls == []


def test_sync_proof_panels_force_load_always_loads(ctrl):
    """force_load=True：即便已有 line_count 记录，也走 load 而非 merge（用于
    打开项目这种全量初始化场景）。"""
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    p1 = _page(1, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])
    ctrl.sync_proof_panels()
    h.load_calls.clear(); v.load_calls.clear()
    ctrl.sync_proof_panels(force_load=True)
    assert len(h.load_calls) == 1
    assert h.merge_calls == []


def test_reset_proof_sync_state_resets_counter(ctrl):
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    p1 = _page(1, [_block([_line()])])
    ctrl._project = OcrProject(name="t", pages=[p1])
    ctrl.sync_proof_panels()
    assert ctrl._proof_loaded_line_count > 0
    ctrl.reset_proof_sync_state()
    assert ctrl._proof_loaded_line_count == 0


# refresh_proof_quality_probe_state：原行为 = 进横校只刷横校，进纵校只刷纵校。
# 上一轮把这一段抽进 controller 时一度写成"两个面板都刷"，会让隐藏的 panel
# 也被 reload。这一轮把它修回成只刷激活那一个，并加 4 条精准回归。
from app.controllers.workflow_controller import (
    STEP_HPROOF as _STEP_HPROOF,
    STEP_VPROOF as _STEP_VPROOF,
    STEP_LAYOUT as _STEP_LAYOUT,
)


def test_refresh_proof_quality_probe_state_hproof_only_touches_h(ctrl):
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    ctrl.refresh_proof_quality_probe_state(_STEP_HPROOF)
    assert h.refresh_calls == 1, "横校 step 必须刷横校"
    assert v.refresh_calls == 0, "横校 step 不能刷到纵校（隐藏 panel 不应被 reload）"


def test_refresh_proof_quality_probe_state_vproof_only_touches_v(ctrl):
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    ctrl.refresh_proof_quality_probe_state(_STEP_VPROOF)
    assert v.refresh_calls == 1, "纵校 step 必须刷纵校"
    assert h.refresh_calls == 0, "纵校 step 不能刷到横校（隐藏 panel 不应被 reload）"


def test_refresh_proof_quality_probe_state_other_step_is_noop(ctrl):
    """非 proof step（比如 LAYOUT）传进来时两个 panel 都不应被刷。"""
    h = _StubPanel(); v = _StubPanel()
    ctrl.register_proof_panels(h, v)
    ctrl.refresh_proof_quality_probe_state(_STEP_LAYOUT)
    assert h.refresh_calls == 0 and v.refresh_calls == 0


def test_refresh_proof_quality_probe_state_safe_when_panel_lacks_method(ctrl):
    class _NoMethod:
        pass
    ctrl.register_proof_panels(_NoMethod(), _NoMethod())
    # 不应抛
    ctrl.refresh_proof_quality_probe_state(_STEP_HPROOF)
    ctrl.refresh_proof_quality_probe_state(_STEP_VPROOF)


def test_refresh_proof_quality_probe_state_safe_when_panels_not_registered(ctrl):
    # 默认状态：register 没调过，两个 attr 都是 None
    ctrl.refresh_proof_quality_probe_state(_STEP_HPROOF)
    ctrl.refresh_proof_quality_probe_state(_STEP_VPROOF)


# ── MainWindow 不应再有泄漏的 source 检查 ──────────────────────

def test_main_window_no_longer_reads_private_project_attr():
    """防止任何人写回 self._controller._project.* 的私有访问。"""
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    # 注释里允许提到（用反引号包裹），代码里禁止
    code_lines = [ln for ln in src.splitlines()
                  if "self._controller._project" in ln and not ln.lstrip().startswith("#")]
    assert code_lines == [], (
        f"MainWindow 仍存在对 controller 私有 _project 的访问：{code_lines}"
    )


def test_main_window_no_longer_owns_proof_loaded_line_count():
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    code_lines = [ln for ln in src.splitlines()
                  if "_proof_loaded_line_count" in ln and not ln.lstrip().startswith("#")]
    assert code_lines == [], (
        f"_proof_loaded_line_count 应只属于 controller：{code_lines}"
    )



# ── view-state ownership: current_step / current_page_number / layout_run_enabled ──
# Phase 10：之前 MainWindow 持有 self._current_step / self._current_page_number
# 并散落 6 处 self._top_nav.set_layout_run_enabled(...) 调用；现在 controller
# 是唯一 ownership 持有者，通过 signal 外播。

def test_current_step_default_is_step_import(ctrl):
    assert ctrl.current_step == STEP_IMPORT


def test_set_current_step_updates_value(ctrl):
    ctrl.set_current_step(STEP_HPROOF)
    assert ctrl.current_step == STEP_HPROOF


def test_set_current_step_emits_signal_on_change(ctrl):
    received = []
    ctrl.current_step_changed.connect(lambda s: received.append(s))
    ctrl.set_current_step(STEP_LAYOUT)
    assert received == [STEP_LAYOUT]


def test_set_current_step_no_signal_when_unchanged(ctrl):
    ctrl.set_current_step(STEP_LAYOUT)
    received = []
    ctrl.current_step_changed.connect(lambda s: received.append(s))
    ctrl.set_current_step(STEP_LAYOUT)  # 同值
    assert received == []


def test_current_page_number_default_is_one(ctrl):
    assert ctrl.current_page_number == 1


def test_set_current_page_number_updates_and_emits(ctrl):
    received = []
    ctrl.current_page_number_changed.connect(lambda n: received.append(n))
    ctrl.set_current_page_number(7)
    assert ctrl.current_page_number == 7
    assert received == [7]


def test_set_current_page_number_no_signal_when_unchanged(ctrl):
    ctrl.set_current_page_number(3)
    received = []
    ctrl.current_page_number_changed.connect(lambda n: received.append(n))
    ctrl.set_current_page_number(3)
    assert received == []


def test_layout_run_enabled_default_false(ctrl):
    assert ctrl.layout_run_enabled is False


def test_set_layout_run_enabled_updates_and_emits(ctrl):
    received = []
    ctrl.layout_run_enabled_changed.connect(lambda b: received.append(b))
    ctrl.set_layout_run_enabled(True)
    assert ctrl.layout_run_enabled is True
    assert received == [True]
    ctrl.set_layout_run_enabled(False)
    assert received == [True, False]


def test_set_layout_run_enabled_no_signal_when_unchanged(ctrl):
    ctrl.set_layout_run_enabled(True)
    received = []
    ctrl.layout_run_enabled_changed.connect(lambda b: received.append(b))
    ctrl.set_layout_run_enabled(True)
    assert received == []


# ── source check：MainWindow 不再持有 view-state ─────────────

def test_main_window_no_longer_owns_current_step():
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    # 允许注释里出现 _current_step（用作说明），禁止代码读写 self._current_step
    code_lines = [
        ln for ln in src.splitlines()
        if "self._current_step" in ln and not ln.lstrip().startswith("#")
    ]
    assert code_lines == [], (
        f"MainWindow 仍存在 self._current_step 读写：{code_lines}"
    )


def test_main_window_no_longer_owns_current_page_number():
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    code_lines = [
        ln for ln in src.splitlines()
        if "self._current_page_number" in ln and not ln.lstrip().startswith("#")
    ]
    assert code_lines == [], (
        f"MainWindow 仍存在 self._current_page_number 读写：{code_lines}"
    )


def test_main_window_routes_layout_run_enabled_via_controller():
    """所有 set_layout_run_enabled 调用必须走 controller，不能再直接写 _top_nav。"""
    src = Path("app/ui/main_window.py").read_text(encoding="utf-8")
    bad = [
        ln for ln in src.splitlines()
        if "self._top_nav.set_layout_run_enabled(" in ln
        and ".connect(" not in ln
        and not ln.lstrip().startswith("#")
    ]
    assert bad == [], (
        f"MainWindow 不应直接调用 _top_nav.set_layout_run_enabled，"
        f"应通过 controller.set_layout_run_enabled：{bad}"
    )
