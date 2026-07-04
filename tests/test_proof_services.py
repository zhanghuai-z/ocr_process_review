"""Tests for proof_probe_text_service and proof_image_service.

这两个 service 是从 h_proof / v_proof 抽出的共享层。本文件覆盖：

1. probe text service：
   - 未启用评测时退化（displayed_text == line.text、save_displayed_edit_result 仅在差异时落盘）
   - 启用评测时显示叠加 fake_char，但 line.text 永不被污染
   - OCR observation 解析 block/line 索引：成功路径 + line/block 不在 page 中时返回 None

2. image service：
   - clamp_bbox_to_image：完全 in-bounds / 越右下边界 / 完全负坐标
   - adaptive_pad_for_bbox：典型小字 / 极小字 / 极大字
   - clamp_line_box_pixels：典型 / pad_y 拉伸 / 完全异常返回 None
   - verified_char_crop：在 mock cache 上验证 clamp + pad 路径都被走到
"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status

from typing import Optional

import pytest

from app.models import BBox, Block, Line, Page
from app.models.enums import BlockType
from app.models.ocr_observation import find_block_ocr_line_index
from app.core import quality_probe as qp
from app.core.quality_probe import (
    Probe, ProbeKey, ProbeStore,
    set_active_store, reset_active_store,
)
from app.services.proof_probe_text_service import (
    displayed_text, save_displayed_edit_result,
)
from app.services.proof_image_service import (
    clamp_bbox_to_image, adaptive_pad_for_bbox,
    clamp_line_box_pixels, verified_char_crop,
)


# ── 辅助构造 ────────────────────────────────────────────────────

def _line(text: str) -> Line:
    return Line(text=text, confidence=0.95, bbox=BBox(0, 0, 100, 20))


def _block(lines):
    return Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 200),
                 lines=list(lines))


def _page(page_no: int, blocks):
    return Page(image_path=f"/tmp/p{page_no}.png", width=100, height=200,
                blocks=list(blocks), page_number=page_no)


@pytest.fixture(autouse=True)
def _isolate_store():
    """每个测试前后都清掉 active store，避免互相污染。"""
    reset_active_store()
    yield
    reset_active_store()


# ════════════════════════════════════════════════════════════════
# probe text service
# ════════════════════════════════════════════════════════════════

def test_find_block_ocr_line_index_success():
    line = _line("abc")
    block = _block([line])
    page = _page(1, [block])
    assert find_block_ocr_line_index(page, block, line) == (0, 0)


def test_find_block_ocr_line_index_block_not_in_page():
    line = _line("abc")
    block = _block([line])
    other_block = _block([_line("xyz")])
    page = _page(1, [block])
    assert find_block_ocr_line_index(page, other_block, line) is None


def test_find_block_ocr_line_index_line_not_in_block():
    line = _line("abc")
    other_line = _line("xyz")
    block = _block([line])
    page = _page(1, [block])
    assert find_block_ocr_line_index(page, block, other_line) is None


def test_displayed_text_no_active_store_returns_line_text():
    line = _line("hello")
    block = _block([line])
    page = _page(1, [block])
    assert displayed_text(line, page, block) == "hello"


def test_displayed_text_active_store_no_probes_returns_line_text():
    line = _line("hello")
    block = _block([line])
    page = _page(1, [block])
    set_active_store(ProbeStore())
    assert displayed_text(line, page, block) == "hello"


def test_displayed_text_with_probe_does_NOT_overlay_line_text():
    """Round 15 后：probe 不再修改显示文本， line.text 原样返回。"""
    line = _line("今天我们来学习己经发生过的历史")
    block = _block([line])
    page = _page(1, [block])
    store = ProbeStore()
    # 新语义：key.char_index 上本来就当 fake_char "己"， true_char "已" 是 gallery 归属
    store.add(Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己"))
    set_active_store(store)
    shown = displayed_text(line, page, block)
    assert shown == line.text
    assert line.text[7] == "己"


def test_save_displayed_edit_result_no_store_propagates_change():
    line = _line("abc")
    block = _block([line])
    page = _page(1, [block])
    change = save_displayed_edit_result(line, page, block, "abd")
    assert change.text_changed is True
    assert change.changed is True
    assert proof_final_text(line) == "abd"
    assert proof_display_text(line) == "abd"


def test_save_displayed_edit_result_no_store_no_change_returns_empty_change():
    line = _line("abc")
    block = _block([line])
    page = _page(1, [block])
    change = save_displayed_edit_result(line, page, block, "abc")
    assert change.changed is False
    assert line.text == "abc"


def test_save_displayed_edit_result_with_probe_does_not_touch_unrelated_text():
    """Stale probe anchor should not rewrite text, but must exit pending state."""
    line = _line("今天我们来学习己经发生过的历史")
    block = _block([line])
    page = _page(1, [block])
    store = ProbeStore()
    probe = Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")
    store.add(probe)
    set_active_store(store)
    shown = displayed_text(line, page, block)
    # 锚点已不再是 true_char：正文不动，但 probe 不能继续 pending。
    change = save_displayed_edit_result(line, page, block, shown)
    assert change.probe_changed is True
    assert change.changed is True
    assert probe.observation == "corrected"
    assert line.text[7] == "己"


def test_save_displayed_edit_result_block_not_in_page_is_cancelled():
    """resolve 失败时拒绝写入，避免 stale UI 把文本写进错误 line。"""
    line = _line("abc")
    block = _block([line])
    other_block = _block([_line("xyz")])
    page = _page(1, [block])
    set_active_store(ProbeStore())
    change = save_displayed_edit_result(line, page, other_block, "abd")
    assert change.cancelled is True
    assert change.blocked is True
    assert proof_display_text(line) == "abc"


# ════════════════════════════════════════════════════════════════
# image service — pure functions
# ════════════════════════════════════════════════════════════════

def test_clamp_bbox_in_bounds_unchanged():
    bb = BBox(10, 20, 30, 40)
    out = clamp_bbox_to_image(bb, 100, 100)
    assert (out.x, out.y, out.w, out.h) == (10, 20, 30, 40)


def test_clamp_bbox_overflows_right_bottom():
    bb = BBox(90, 95, 30, 40)
    out = clamp_bbox_to_image(bb, 100, 100)
    # 左上不动，宽高被裁到边界
    assert (out.x, out.y) == (90, 95)
    assert out.x + out.w <= 100
    assert out.y + out.h <= 100
    assert out.w >= 1 and out.h >= 1


def test_clamp_bbox_negative_topleft():
    bb = BBox(-5, -10, 30, 40)
    out = clamp_bbox_to_image(bb, 100, 100)
    assert out.x == 0 and out.y == 0
    # 关键：宽高用 clamped 后的左上角算（修了之前的越界 1px bug）
    assert out.x + out.w <= 100
    assert out.y + out.h <= 100


def test_clamp_bbox_far_beyond_image():
    bb = BBox(500, 500, 100, 100)
    out = clamp_bbox_to_image(bb, 100, 100)
    # x clamp 到 W-1=99，宽至少 1
    assert out.x == 99 and out.y == 99
    assert out.w == 1 and out.h == 1


def test_adaptive_pad_typical_small_char():
    # 40x40 字 → 10% = 4
    assert adaptive_pad_for_bbox(BBox(0, 0, 40, 40)) == 4


def test_adaptive_pad_very_small_char_floors_to_two():
    # Narrow punctuation needs a small safety margin; 3x3 -> 0.3 -> floor to 2.
    assert adaptive_pad_for_bbox(BBox(0, 0, 3, 3)) == 2


def test_adaptive_pad_large_char_caps_at_max():
    # 200x200 字 → 20，但被 max_pad=6 截断
    assert adaptive_pad_for_bbox(BBox(0, 0, 200, 200)) == 6


def test_adaptive_pad_custom_params():
    assert adaptive_pad_for_bbox(BBox(0, 0, 100, 100), max_pad=10, ratio=0.05) == 5


def test_clamp_line_box_typical():
    bb = BBox(10, 20, 100, 30)   # x2=110, y2=50
    out = clamp_line_box_pixels(bb, 200, 200, pad_y=4)
    assert out == (10, 16, 110, 54)


def test_clamp_line_box_pad_y_clamped_to_image_top():
    bb = BBox(10, 2, 100, 30)
    out = clamp_line_box_pixels(bb, 200, 200, pad_y=10)
    # y - 10 = -8 → 0；y2 + 10 = 42
    assert out == (10, 0, 110, 42)


def test_clamp_line_box_completely_invalid_returns_none():
    bb = BBox(500, 500, 10, 10)   # 完全在 100x100 之外
    out = clamp_line_box_pixels(bb, 100, 100, pad_y=4)
    assert out is None


def test_clamp_line_box_zero_size_returns_none():
    bb = BBox(10, 10, 0, 0)
    out = clamp_line_box_pixels(bb, 100, 100, pad_y=0)
    assert out is None


# ════════════════════════════════════════════════════════════════
# image service — verified_char_crop with mock cache
# ════════════════════════════════════════════════════════════════

class _StubCache:
    """模拟 PageImageCache：只关心 get_page_image / get_char_crop 的调用契约。"""

    def __init__(self, image_shape: Optional[tuple]):
        self._shape = image_shape
        self.crop_calls: list = []

    def get_page_image(self, path):
        if self._shape is None:
            return None
        import numpy as np
        return np.zeros(self._shape, dtype="uint8")

    def get_char_crop(self, path, bbox, size, *, pad):
        self.crop_calls.append((path, bbox, size, pad))
        return "PIXMAP_SENTINEL"


def test_verified_char_crop_returns_none_when_image_missing():
    cache = _StubCache(image_shape=None)
    out = verified_char_crop(cache, "/no/such.png", BBox(0, 0, 10, 10), size=20)
    assert out is None
    assert cache.crop_calls == []


def test_verified_char_crop_inbounds_passes_bbox_unchanged():
    cache = _StubCache(image_shape=(100, 100, 3))
    bb = BBox(10, 20, 30, 40)
    out = verified_char_crop(cache, "p.png", bb, size=27)
    assert out == "PIXMAP_SENTINEL"
    assert len(cache.crop_calls) == 1
    _, called_bbox, called_size, called_pad = cache.crop_calls[0]
    assert called_size == 27
    assert called_pad == adaptive_pad_for_bbox(bb)
    assert (called_bbox.x, called_bbox.y, called_bbox.w, called_bbox.h) == (10, 20, 30, 40)


def test_verified_char_crop_out_of_bounds_clamps_before_calling_cache():
    cache = _StubCache(image_shape=(100, 100, 3))
    bb = BBox(90, 95, 30, 40)   # x2=120, y2=135 → 越界
    verified_char_crop(cache, "p.png", bb, size=27)
    _, called_bbox, _, _ = cache.crop_calls[0]
    assert called_bbox.x + called_bbox.w <= 100
    assert called_bbox.y + called_bbox.h <= 100


def test_verified_char_crop_explicit_pad_overrides_adaptive():
    cache = _StubCache(image_shape=(100, 100, 3))
    bb = BBox(10, 10, 40, 40)
    verified_char_crop(cache, "p.png", bb, size=27, pad=12)
    _, _, _, called_pad = cache.crop_calls[0]
    assert called_pad == 12


# ════════════════════════════════════════════════════════════════
# 集成回归：proof UI 不应重新内联 probe/edit 逻辑。
# ════════════════════════════════════════════════════════════════

def test_h_proof_uses_shared_display_and_edit_services():
    from app.ui.proof import h_proof
    from app.services import proof_probe_text_service as svc
    from app.services.proof_edit_service import ProofEditService

    assert h_proof._displayed_text is svc.displayed_text
    assert h_proof.ProofEditService is ProofEditService
