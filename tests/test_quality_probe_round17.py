"""Round 17 — density 真生效 + 更正后立刻进入正确集合 + 实时刷新 + 去 '确认本页'。"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

from app.core import quality_probe as qp
from app.core.proof_state_bus import ProofStateBus
from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    qp.reset_active_store()
    ProofStateBus.reset()


# ─── 1. 密度真实生效（不再被硬上限挤成 2~4 个） ──────────────────

def _make_long_project_with_pairs() -> OcrProject:
    """构造 ~1000 字、含密集近形对的项目。"""
    # 使用多组近形对："己已", "体休", "拼并", "免兔", "末未", "鸟乌", "土士"
    seg = "己已体休拼并免兔末未鸟乌"  # 12 字, 6 pair
    text = (seg * 84)[:1000]  # ≈ 1000 字
    pages = []
    # 切成 10 页，每页 1 block × 1 line，便于撞 max_per_page 默认值
    per_page = 100
    for pi in range(10):
        chunk = text[pi * per_page:(pi + 1) * per_page]
        line = Line(text=chunk, confidence=0.9, bbox=BBox(0, 0, 800, 20))
        line.chars = [
            Char(char=ch, confidence=0.9, bbox=BBox(i * 8, 0, 8, 20),
                 bbox_source="ocr", bbox_granularity="char", token_text=ch)
            for i, ch in enumerate(chunk)
        ]
        block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 800, 20),
                      lines=[line])
        block.order = 0
        pages.append(Page(page_number=pi + 1, blocks=[block],
                          image_path=f"/tmp/r17-{pi}.png",
                          width=800, height=20))
    return OcrProject(name="r17", pages=pages)


def test_round17_density_actually_drives_placement():
    """目标 ~20/1000：实际投放应明显接近目标，而不是被硬上限卡在 2~4 个。"""
    proj = _make_long_project_with_pairs()
    cfg = qp.SamplerConfig(sand_count=20, sand_unit_chars=1000, seed=7)
    store = qp.ProbeSampler(cfg).sample(proj)
    # 期望 >= 15（接近 target=20；放宽一点以应对 confusable 交集波动），
    # 关键反例：不应该只有 2~4 个。
    assert len(store) >= 15, (
        f"密度配置 20/1000 但只投放了 {len(store)} 个 — 硬上限仍在挤压目标"
    )


def test_round17_default_max_per_true_char_is_relaxed():
    """默认 max_per_true_char 不再卡 1，应该 >= 5。"""
    cfg = qp.SamplerConfig()
    assert cfg.max_per_true_char >= 5
    assert cfg.max_per_page >= 100
    assert cfg.max_total >= 100


# ─── 2. observe_slot_edit 通过事件总线广播 probe.observed ────────

def test_round17_observe_slot_edit_publishes_event():
    line = Line(text="己已", confidence=0.9, bbox=BBox(0, 0, 40, 20))
    line.chars = [
        Char(char="己", confidence=0.9, bbox=BBox(0, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="己"),
        Char(char="已", confidence=0.9, bbox=BBox(20, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="已"),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 20), lines=[line])
    block.order = 0
    Page(page_number=1, blocks=[block], image_path="/tmp/x.png",
         width=100, height=100)
    store = qp.ProbeStore()
    store.add(qp.Probe(
        key=qp.ProbeKey(1, 0, 0, 1), true_char="己", fake_char="已"))
    qp.set_active_store(store)

    received: list = []
    ProofStateBus.instance().subscribe(
        qp.TOPIC_PROBE_OBSERVED, received.append,
    )
    ok = qp.observe_slot_edit(store, 1, 0, 0, 1)
    assert ok is True
    assert len(received) == 1
    assert received[0].true_char == "己"
    assert received[0].fake_char == "已"
    # 重复调用不再二次广播（已 corrected）
    qp.observe_slot_edit(store, 1, 0, 0, 1)
    assert len(received) == 1


# ─── 3. corrected probe 只通过 CharIndex 正常查询体现 ────

def test_round17_corrected_probe_uses_char_index_without_gallery_extras():
    from app.ui.proof.v_proof import VProofPanel

    line = Line(text="己已", confidence=0.9, bbox=BBox(0, 0, 40, 20))
    line.chars = [
        Char(char="己", confidence=0.9, bbox=BBox(0, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="己"),
        Char(char="已", confidence=0.9, bbox=BBox(20, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="已"),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block], image_path="/tmp/y.png",
                width=100, height=100)
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(1, 0, 0, 1), true_char="己", fake_char="已")
    store.add(probe)
    qp.set_active_store(store)

    panel = VProofPanel()
    panel.load_pages([page])

    # Round 18：gallery extras 已删除。corrected 通过 line.text 锚点反映；
    # gallery 集合归位由 CharIndexService.query(line.text[i]) 自然完成。
    entries = list(panel._char_svc.query("己"))
    assert [entry.char_idx for entry in entries] == [0]
    probe.observation = "corrected"
    entries = list(panel._char_svc.query("己"))
    assert [entry.char_idx for entry in entries] == [0]
    panel.close()


# ─── 4. QualityStatsDialog 订阅 probe.observed 实时刷新 ──────────

def test_round17_dialog_realtime_refresh_on_probe_observed(monkeypatch):
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    # 构造 active store + 假项目
    line = Line(text="己已", confidence=0.9, bbox=BBox(0, 0, 40, 20))
    line.chars = [
        Char(char="己", confidence=0.9, bbox=BBox(0, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="己"),
        Char(char="已", confidence=0.9, bbox=BBox(20, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="已"),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block], image_path="/tmp/z.png",
                width=100, height=100)
    proj = OcrProject(name="r17-dlg", pages=[page])
    store = qp.ProbeStore()
    store.add(qp.Probe(
        key=qp.ProbeKey(1, 0, 0, 1), true_char="己", fake_char="已"))
    qp.set_active_store(store)

    refresh_calls: list[bool] = []
    dlg = QualityStatsDialog(
        project_provider=lambda: proj,
        refresh_panels_cb=lambda: refresh_calls.append(True),
    )
    # 模拟评测窗已启动，按钮置 checked 以触发 _refresh_view 走"已启用"分支
    dlg._btn_toggle.setChecked(True)
    dlg._refresh_view()
    base_text = dlg._switch_status.text()
    assert "已启用" in base_text

    # 触发一次 probe 观察 → dialog 应自动 refresh + 通知 panel
    qp.observe_slot_edit(store, 1, 0, 0, 1)
    QApplication.processEvents()

    assert refresh_calls, "probe.observed 后应至少触发一次 refresh_panels_cb"
    state = dlg.quality_state
    assert state.corrected == 1
    assert state.total_probes == 1
    dlg.deleteLater()


# ─── 5. VProof 不再有 "确认本页" ─────────────────────────────────

def test_round17_vproof_drops_confirm_page_button():
    from app.ui.proof.v_proof import VProofPanel
    panel = VProofPanel()
    assert not hasattr(panel, "_btn_ok"), "VProof 不应再有 _btn_ok（'确认本页'）"
    assert not hasattr(panel, "_mark_page_ok"), (
        "VProof 不应再有 _mark_page_ok 方法（页级 OK 语义已删除）"
    )
    panel.close()


def test_round17_vproof_source_has_no_confirm_page_strings():
    from pathlib import Path
    src = Path("app/ui/proof/v_proof.py").read_text(encoding="utf-8")
    # 允许出现 "确认本页" 这四个字作为注释/说明，但不允许 _btn_ok / _mark_page_ok 引用
    assert "_btn_ok" not in src
    assert "_mark_page_ok" not in src
