"""hproof-yaxis-verdicts 新组件直接覆盖测试。

覆盖三个本轮新增的关注点：

1. ``char_verdict.classify_char``：颜色/severity/证据/不洗白语义。
2. ``AlignmentRibbon`` 的 set_aligned / set_degraded 行为与是否处于降级态。
3. ``_LinePair._refresh_ribbon`` 在 chars 缺失 / 长度不匹配时进入降级，
   在等长 + render_scale 就绪后进入 aligned。
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


def test_classify_char_low_conf_red():
    from app.ui.proof import char_verdict as cv

    v = cv.classify_char(confidence=0.30, text_char="字",
                         ocr_char="字", llm_char=None)
    assert v.severity == cv.SEVERITY_ERROR
    assert v.color == cv.COLOR_ERROR
    assert "0.30" in v.evidence
    assert v.user_modified is False


def test_classify_char_user_modified_does_not_whitewash_color():
    """用户改过字 → user_modified=True，但颜色仍保持原 severity（不洗白）。"""
    from app.ui.proof import char_verdict as cv

    v = cv.classify_char(confidence=0.30, text_char="改",
                         ocr_char="原", llm_char=None)
    assert v.user_modified is True
    # 关键 invariant：改过 ≠ 已通过校对 → 颜色仍是红
    assert v.severity == cv.SEVERITY_ERROR
    assert v.color == cv.COLOR_ERROR


def test_classify_char_high_conf_green():
    from app.ui.proof import char_verdict as cv

    v = cv.classify_char(confidence=0.96, text_char="字",
                         ocr_char="字", llm_char=None)
    assert v.severity == cv.SEVERITY_LIKELY_OK
    assert v.color == cv.COLOR_LIKELY_OK


def test_classify_char_mid_conf_default_unverified():
    """0.80 ≤ conf < 0.95 在当前 codebase 没有"绝对正确"证据链 → unverified。"""
    from app.ui.proof import char_verdict as cv

    v = cv.classify_char(confidence=0.88, text_char="字",
                         ocr_char="字", llm_char=None)
    assert v.severity == cv.SEVERITY_UNVERIFIED
    assert v.color == cv.COLOR_UNVERIFIED


def test_classify_char_llm_disagrees_upgrades():
    from app.ui.proof import char_verdict as cv

    # 中段 conf + llm 反对 → 升级为 suspect
    v = cv.classify_char(confidence=0.88, text_char="字",
                         ocr_char="字", llm_char="子")
    assert v.severity == cv.SEVERITY_SUSPECT
    # 低段 conf + llm 反对 → 升级为 error
    v2 = cv.classify_char(confidence=0.70, text_char="字",
                          ocr_char="字", llm_char="子")
    assert v2.severity == cv.SEVERITY_ERROR


def test_classify_char_confidence_missing():
    from app.ui.proof import char_verdict as cv

    v = cv.classify_char(confidence=None, text_char="字",
                         ocr_char="字", llm_char=None)
    assert v.severity == cv.SEVERITY_UNVERIFIED
    assert "缺失" in v.evidence


def test_aligned_ribbon_degraded_when_no_chars():
    from app.ui.proof.aligned_ribbon import AlignmentRibbon

    ribbon = AlignmentRibbon()
    ribbon.set_degraded("无逐字 bbox / 仅行级对应", pixmap_width=200)
    assert ribbon.is_degraded()
    assert ribbon.width() == 200


def test_aligned_ribbon_aligned_set_clears_degraded():
    from app.ui.proof.aligned_ribbon import AlignmentRibbon
    from app.ui.proof import char_verdict as cv

    ribbon = AlignmentRibbon()
    ribbon.set_degraded("foo", pixmap_width=100)
    assert ribbon.is_degraded()
    verdicts = [
        cv.classify_char(confidence=0.96, text_char="A",
                         ocr_char="A", llm_char=None),
        cv.classify_char(confidence=0.30, text_char="B",
                         ocr_char="B", llm_char=None),
    ]
    ribbon.set_aligned(
        text="AB",
        x_centers=[10.0, 30.0],
        widths=[16.0, 16.0],
        verdicts=verdicts,
        pixmap_width=120,
        cursor_idx=0,
        hover_idx=-1,
    )
    assert not ribbon.is_degraded()
    assert ribbon.width() == 120


def _make_simple_pair(line_text: str, chars_with_bbox: bool = True):
    """直接构造一个 _LinePair 用于 _refresh_ribbon 单元测试。"""
    from app.models import Block, BBox, Char, Line, Page, ProofStatus
    from app.models.project import BlockType
    from app.core.page_image_cache import PageImageCache
    from app.ui.proof.h_proof import _LinePair

    chars = []
    for i, ch in enumerate(line_text):
        bbox = BBox(10 + i * 30, 0, 35 + i * 30, 30) if chars_with_bbox else None
        chars.append(Char(char=ch, confidence=0.9, bbox=bbox))
    line = Line(
        text=line_text,
        confidence=0.9,
        bbox=BBox(0, 0, 400, 32),
        chars=chars,
        proof_status=ProofStatus.UNCHECKED,
        ocr_text=line_text,
    )
    block = Block(block_type=BlockType.TEXT,
                  bbox=BBox(0, 0, 400, 32), lines=[line])
    page = Page(image_path="", width=400, height=32, blocks=[block])
    cache = PageImageCache.instance()
    pair = _LinePair(0, block, line, page, 1, cache)
    return pair


def test_line_pair_no_ribbon_widget():
    """proof-layout-collections 第 1 任务：_LinePair 不再创建第三行
    AlignmentRibbon。content_layout 仅含 image + editor 两项。"""
    pair = _make_simple_pair("ab", chars_with_bbox=True)
    assert not hasattr(pair, "_ribbon"), "_LinePair 不应再持有 _ribbon"
    layout = pair._content.layout()
    assert layout.count() == 2
    pair.deleteLater()


def test_chars_aligned_rejects_multi_glyph_token():
    """proof-layout-collections 第 2 任务：char.char 多于 1 个字符
    （word/token 粒度）时，_chars_aligned 必须返回 False，避免
    \u201c\u5750\u6807\u5bf9\u4e86\u4f46\u5185\u5bb9\u504f\u79fb\u201d。"""
    from app.models import BBox, Char, Line, Page, Block, ProofStatus
    from app.models.project import BlockType
    from app.core.page_image_cache import PageImageCache
    from app.ui.proof.h_proof import _LinePair

    chars = [
        Char(char="he", confidence=0.9, bbox=BBox(0, 0, 20, 20)),
        Char(char="llo", confidence=0.9, bbox=BBox(20, 0, 60, 20)),
    ]
    line = Line(text="he", confidence=0.9, bbox=BBox(0, 0, 80, 20),
                chars=chars, proof_status=ProofStatus.UNCHECKED)
    block = Block(block_type=BlockType.TEXT,
                  bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="", width=80, height=20, blocks=[block])
    pair = _LinePair(0, block, line, page, 1, PageImageCache.instance())
    # editor 文本长度 == len(chars) (=2)，但 chars[0].char='he' 多 glyph
    pair._editor.blockSignals(True)
    pair._editor.setPlainText("he")
    pair._editor.blockSignals(False)
    assert pair._chars_aligned() is False
    pair.deleteLater()
