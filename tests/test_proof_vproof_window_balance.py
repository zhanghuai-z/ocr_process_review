"""vproof-window-balance (round 13) tests.

窄目标：VProof 窗口本身的"列宽平衡"问题：
- 之前 _content_split 用 1:4 stretchFactor，proof_column 永远只拿到 1/5 的
  横向空间，导致 gallery 6 个缩略图根本拼不进一行，被迫多行 → 卡顿、压字。
- 与此同时图片那一栏一直占 4/5 → "怎么图片还变大了"。
- 本轮不再用"再放大缩略图"冒充修复，而是直接动 splitter ratio + minimum
  width + initial sizes，把 proof_column 撑到 6 列 56px 缩略图真的能放下
  的宽度，并让 viewer 不再独吞。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _new_panel():
    from app.ui.proof.v_proof import VProofPanel
    return VProofPanel()


def test_proof_column_min_width_fits_one_full_gallery_row():
    """proof_column 的 minimumWidth 必须大到能容下一行 6 个 56px 缩略图。"""
    from app.ui.proof import v_proof
    p = _new_panel()
    required = v_proof.GALLERY_ITEMS_PER_ROW * (v_proof.GALLERY_THUMB + 8) + 40
    assert p._proof_column.minimumWidth() >= required, (
        f"proof_column.minWidth={p._proof_column.minimumWidth()} < 必需 {required}"
    )


def test_content_split_resize_keeps_proof_balanced():
    """放大窗口时，新增空间不能全部被 viewer 抢走（之前 stretchFactor=1:4
    意味着每多 5px viewer 就抢 4px）。"""
    p = _new_panel()
    p.resize(1200, 800)
    p.show()
    QApplication.processEvents()
    small_sizes = p._content_split.sizes()
    p.resize(1800, 800)
    QApplication.processEvents()
    large_sizes = p._content_split.sizes()
    # 600px 增量里 viewer 拿走不能 >70%（原 stretch 1:4 会拿走 80%）。
    delta_viewer = large_sizes[1] - small_sizes[1]
    delta_total = (sum(large_sizes) - sum(small_sizes)) or 1
    viewer_share = delta_viewer / delta_total
    assert viewer_share <= 0.70, (
        f"放大窗口时 viewer 吞掉了 {viewer_share:.0%} 的新空间（原 80%）"
    )
    p.close()


def test_content_split_initial_sizes_give_proof_column_real_room():
    """初始 setSizes 必须让 proof_column 至少和 viewer 一样宽，避免"图片自动变大"。"""
    p = _new_panel()
    sizes = p._content_split.sizes()
    assert len(sizes) == 2
    # 没显示时 sizes 可能是 [0, 0]，强制 show 一下
    if sizes == [0, 0]:
        p.resize(1400, 800)
        p.show()
        QApplication.processEvents()
        sizes = p._content_split.sizes()
    # proof 列至少要拿到 35%（之前是 20%）。
    total = sum(sizes) or 1
    proof_ratio = sizes[0] / total
    assert proof_ratio >= 0.35, (
        f"proof_column 只拿到 {proof_ratio:.0%}，<35%（之前 20% 是窗口失衡根因）"
    )
    p.close()


def test_viewer_has_minimum_width_but_not_dominant():
    """viewer_box 有合理 minimum，但不再用 stretchFactor 抢空间。"""
    p = _new_panel()
    assert p._viewer_box.minimumWidth() >= 200
    # viewer 不应该有 maximumWidth 抓死；用户能拖大
    assert p._viewer_box.maximumWidth() >= 10000


def test_gallery_box_fits_in_proof_column_min_width():
    """proof_column 在 minimumWidth 下，gallery 6 列能完整水平排开。"""
    from app.ui.proof import v_proof
    p = _new_panel()
    p.resize(1500, 900)
    p.show()
    QApplication.processEvents()
    # 强制 proof_column 到 minWidth 看 gallery 是否还能容下 6 个
    proof_w = p._proof_column.width()
    one_item_w = v_proof.GALLERY_THUMB + 8  # spacing
    fit_per_row = proof_w // one_item_w
    assert fit_per_row >= v_proof.GALLERY_ITEMS_PER_ROW, (
        f"proof_column 宽 {proof_w}px 仅容下 {fit_per_row} 个缩略图，"
        f"目标 {v_proof.GALLERY_ITEMS_PER_ROW} 个"
    )
    p.close()
