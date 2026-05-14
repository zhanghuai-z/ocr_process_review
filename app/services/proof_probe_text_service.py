"""Quality-probe display↔true text bridge — shared between h_proof and v_proof.

把"显示文本（可能含 fake_char）"和"真实文本 line.text"之间的转换集中收口，
供横校 / 纵校两个面板统一调用。

调用方约束（未变）：
- 显示文本 → ``displayed_text(line, page, block)``
- 用户编辑落盘 → ``save_displayed_edit(line, page, block, new_text)``

quality-probe 的硬边界由 ``app.core.quality_probe`` 内部保证：
- ``apply_probes_to_display`` 是位置等长替换 → 显示空间字符索引和真实空间一一对应
- ``observe_user_action`` 通过 ``reverse_display_to_true`` 反推真实文本，并在
  small-change / large-change 两条路径末尾各做一次全文 scrub → 返回的
  真实文本绝不含任何 ``probe.fake_char``，从而保证写回 ``line.text`` 的内容
  仍是真实文本，导出器/外部读 ``line.text`` 永远拿不到 fake_char。

这层抽离纯粹是把 h_proof / v_proof 里逐行重复的同一段逻辑下沉到一个地方，
不修改任何边界条件或采样口径。
"""
from __future__ import annotations

from typing import Optional, Tuple

from app.models import Block, Line, Page
from app.core import quality_probe as qp


def resolve_block_line_index(
    page: Page, block: Block, line: Line
) -> Optional[Tuple[int, int]]:
    """返回 (block_index, line_index) 或 None（line/block 已不在 page 中）。"""
    try:
        bi = page.blocks.index(block)
        li = block.lines.index(line)
    except ValueError:
        return None
    return bi, li


def displayed_text(line: Line, page: Page, block: Block) -> str:
    """返回应展示给用户的文本：未启用评测时即 ``line.text``。"""
    text = line.text or ""
    store = qp.get_active_store()
    if store is None:
        return text
    idx = resolve_block_line_index(page, block, line)
    if idx is None:
        return text
    bi, li = idx
    probes = store.for_line(page.page_number, bi, li)
    if not probes:
        return text
    return qp.apply_probes_to_display(text, probes)


def save_displayed_edit(
    line: Line, page: Page, block: Block, displayed_new_text: str
) -> bool:
    """把"显示空间"的编辑结果落盘到 ``line.text``（真实空间）。

    返回 ``True`` 表示真的发生了变化（已 ``update_text``）。
    未启用评测时退化为：``displayed_new_text != line.text`` → ``line.update_text``。
    """
    store = qp.get_active_store()
    if store is None:
        if displayed_new_text != (line.text or ""):
            line.update_text(displayed_new_text)
            return True
        return False
    idx = resolve_block_line_index(page, block, line)
    if idx is None:
        if displayed_new_text != (line.text or ""):
            line.update_text(displayed_new_text)
            return True
        return False
    bi, li = idx
    true_text = qp.observe_user_action(
        store, page.page_number, bi, li, line.text or "", displayed_new_text,
    )
    if true_text != (line.text or ""):
        line.update_text(true_text)
        return True
    return False
