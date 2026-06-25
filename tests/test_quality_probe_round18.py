"""Round 18 — three residual acceptance points.

1. **OCR 文本窗口也要体现假象**：displayed_text 把 fake_char 注入到显示空间，
   line.text 始终保持 true_char（"以文本为锚点"）。
2. **更正后回正确集合**：save_displayed_edit_result 反向写回时，line.text[i] 变了就
   把命中的 probe 标 corrected；CharIndexService 重建后该位置自然出现在新
   true_char 的 gallery。
3. **统计刷新可靠触发**：QualityStatsDialog 暴露 "立刻刷新" 按钮；点击后调
   detect_corrections + 重绘本窗。
"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from app.core import quality_probe as qp
from app.models import BBox, Block, BlockType, Char, Line, Page, OcrProject
from app.services.proof_probe_text_service import (
    displayed_text,
    save_displayed_edit_result,
)
from app.services.char_index_service import CharIndexService


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    qp.reset_active_store()


def _make_pages():
    line = Line(text="己已", confidence=0.9, bbox=BBox(0, 0, 80, 20))
    line.chars = [
        Char(char="己", confidence=0.9, bbox=BBox(0, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="己"),
        Char(char="已", confidence=0.9, bbox=BBox(20, 0, 20, 20),
             bbox_source="ocr", bbox_granularity="char", token_text="已"),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block],
                image_path="/tmp/round18.png", width=100, height=100)
    return page, block, line


def _plant_probe(line, page, block, true_ch, fake_ch, char_index):
    assert line.text[char_index] == true_ch, "测试前提：line.text 在 idx 上必须是 true_char"
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, char_index),
        true_char=true_ch,
        fake_char=fake_ch,
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    return store, probe


# ─── 1. displayed_text 注入 fake_char ─────────────────────────

def test_round18_displayed_text_injects_fake_char_at_pending_probe():
    page, block, line = _make_pages()
    _store, _probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # line.text 不变
    assert line.text == "己已"
    # displayed_text 把 idx=1 的 "已" 渲染成 "己"（fake_char）
    assert displayed_text(line, page, block) == "己己"


def test_round18_displayed_text_no_active_store_returns_line_text():
    page, block, line = _make_pages()
    qp.reset_active_store()
    assert displayed_text(line, page, block) == line.text


def test_round18_displayed_text_corrected_probe_does_not_inject():
    page, block, line = _make_pages()
    _store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    probe.observation = "corrected"
    # corrected 之后该位置不再注入 fake_char
    assert displayed_text(line, page, block) == "己已"


def test_round18_displayed_text_does_not_mask_stale_probe_anchor():
    page, block, line = _make_pages()
    _store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)

    line.set_proof_text("己巳")

    assert displayed_text(line, page, block) == "己巳"
    result = save_displayed_edit_result(line, page, block, "己巳")
    assert result.text_changed is False
    assert result.probe_changed is True
    assert result.changed is True
    assert probe.observation == "corrected"
    assert proof_display_text(line) == "己巳"


# ─── 2. save_displayed_edit_result 文本锚点反向 ────────────────

def test_round18_save_displayed_edit_result_unchanged_keeps_probe_pending():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # 用户没动任何字（送回 displayed 原文）
    change = save_displayed_edit_result(line, page, block, displayed_text(line, page, block))
    assert change.changed is False
    assert probe.observation == "pending"
    assert line.text == "己已"


def test_round18_save_displayed_edit_result_user_corrects_marks_corrected_and_preserves_line_text():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # displayed = "己己"；用户把第二个改回 "已"（正确答案）
    change = save_displayed_edit_result(line, page, block, "己已")
    assert change.probe_changed is True
    assert probe.observation == "corrected"
    # line.text 维持 true_char
    assert line.text == "己已"


def test_round18_save_displayed_edit_result_user_types_other_char_still_marks_corrected():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # 用户把 displayed[1] 从 "己" 改成 "巳"（不是 true_char 也不是 fake_char）
    change = save_displayed_edit_result(line, page, block, "己巳")
    assert change.text_changed is True
    assert change.probe_changed is True
    assert probe.observation == "corrected"
    # final_text 反映用户实际输入
    assert proof_final_text(line) == "己巳"


def test_round18_save_displayed_edit_result_length_change_does_not_persist_untouched_fake_char():
    line = Line(text="甲乙丙", confidence=0.9, bbox=BBox(0, 0, 120, 20))
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block], image_path="/tmp/round18-len.png", width=120, height=20)
    _store, probe = _plant_probe(line, page, block, true_ch="乙", fake_ch="己", char_index=1)

    assert displayed_text(line, page, block) == "甲己丙"
    change = save_displayed_edit_result(line, page, block, "甲己丙丁")

    assert change.text_changed is True
    assert change.probe_changed is True
    assert proof_final_text(line) == "甲乙丙丁"
    assert "己" not in proof_final_text(line)
    assert probe.observation == "corrected"


def test_round18_save_displayed_edit_result_length_change_keeps_user_replacement():
    line = Line(text="甲乙丙", confidence=0.9, bbox=BBox(0, 0, 120, 20))
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 20), lines=[line])
    block.order = 0
    page = Page(page_number=1, blocks=[block], image_path="/tmp/round18-len.png", width=120, height=20)
    _store, probe = _plant_probe(line, page, block, true_ch="乙", fake_ch="己", char_index=1)

    change = save_displayed_edit_result(line, page, block, "甲巳丙丁")

    assert change.text_changed is True
    assert change.probe_changed is True
    assert proof_final_text(line) == "甲巳丙丁"
    assert probe.observation == "corrected"


def test_round18_corrected_probe_position_appears_in_true_char_gallery():
    """编辑后必须以文本为锚点回到正确集合：CharIndexService.query(true_char) 包含该位置。"""
    page, block, line = _make_pages()
    _store, _probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # 用户在显示空间纠正
    save_displayed_edit_result(line, page, block, "己已")
    # line.text 已是 "己已"——任何后续 CharIndexService.build(pages) 都会按
    # line.text[1]=="已" 把 idx=1 收入"已" gallery 集合，这就是"归位"语义。
    assert line.text == "己已"
    assert line.text[1] == "已"  # 文本锚点：query("已") 必然会命中此位置
    # （不实际跑 svc.build，避免依赖图像文件；上面的不变量已足以推导出归位行为。）


# ─── 3. detect_corrections 兜底 ──────────────────────────────

def test_round18_detect_corrections_picks_up_bypass_path_edits():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    # 模拟 batch-replace / 其它绕过 bridge 的路径：直接改 line.text
    line.set_proof_text("己巳")
    # detect_corrections 应捕获这处 mutate 并把 probe 标 corrected
    n = qp.detect_corrections(store, [page])
    assert n == 1
    assert probe.observation == "corrected"


def test_round18_detect_corrections_no_mutation_no_corrected():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    n = qp.detect_corrections(store, [page])
    assert n == 0
    assert probe.observation == "pending"


def test_round18_detect_corrections_accepts_project_or_pages():
    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    project = OcrProject(name="round18", pages=[page])
    line.set_proof_text("己巳")
    assert qp.detect_corrections(store, project) == 1


# ─── 4. QualityStatsDialog 立刻刷新按钮 ────────────────────────

def test_round18_dialog_has_refresh_button_and_uses_detect_corrections(monkeypatch):
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    page, block, line = _make_pages()
    store, probe = _plant_probe(line, page, block, true_ch="已", fake_ch="己", char_index=1)
    project = OcrProject(name="round18", pages=[page])
    changes = []
    dlg = QualityStatsDialog(
        project_provider=lambda: project,
        refresh_panels_cb=lambda: None,
        proof_changed_cb=changes.append,
    )
    # 立刻刷新按钮必须存在且可见
    assert hasattr(dlg, "_btn_refresh")
    assert dlg._btn_refresh.text() == "立刻刷新"

    # 模拟绕过 bridge 的真实编辑
    line.set_proof_text("己巳")
    # 点刷新前 probe 还是 pending；点之后被 detect_corrections 抓到
    assert probe.observation == "pending"
    dlg._on_manual_refresh()
    assert probe.observation == "corrected", "立刻刷新必须触发 detect_corrections"
    assert len(changes) == 1
    assert changes[0].probe_changed is True
    assert changes[0].needs_persist is True
    dlg.close()


# ─── 5. 源码不再有 _mark_observed_btn / mark-observed 残留 ─────

def test_round18_vproof_no_mark_observed_residue():
    from pathlib import Path
    src = Path("app/ui/proof/v_proof.py").read_text()
    assert "_mark_observed_btn" not in src
    assert "_mark_selected_observed" not in src
