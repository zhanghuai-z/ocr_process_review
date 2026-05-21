"""Quality-probe service layer — Round 15 之后只剩"槽位编辑观测"桥。

历史：本模块原是 h_proof / v_proof 用来把"显示空间文本（含 fake_char）" 
和"真实 line.text" 之间互转的桥。Round 15 改造后 line.text **从不被污染**，
所以：

- ``displayed_text`` 永远返回 ``line.text`` 原值；保留这个函数仅是为了不动
  现有调用方签名（h_proof / v_proof 都直接调它）。
- ``save_displayed_edit`` 不再走 ``reverse_display_to_true``，只在差异时直接
  ``update_text`` 并同步 chars/glyphs。
- ``observe_slot_edit_at`` 是新增的薄包装：v_proof 槽位编辑流程在落盘前调用，
  若该 (page, block, line, char_index) 命中 active 评测的 probe，则把 probe
  observation 标 ``corrected``。
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
    """显示给用户的文本 —— Round 15 后始终等于 line.text。"""
    return line.text or ""


def save_displayed_edit(
    line: Line, page: Page, block: Block, displayed_new_text: str
) -> bool:
    """把编辑结果落盘到 ``line.text``。返回 ``True`` 表示发生变化。"""
    if displayed_new_text == (line.text or ""):
        return False
    line.update_text(displayed_new_text)
    _sync_chars_glyphs(line, displayed_new_text)
    return True


def observe_slot_edit_at(
    page: Page, block: Block, line: Line, char_index: int,
) -> bool:
    """槽位编辑触发观测：若 (page, block, line, char_index) 命中 active probe，
    把 probe.observation 标 corrected。"""
    store = qp.get_active_store()
    if store is None:
        return False
    idx = resolve_block_line_index(page, block, line)
    if idx is None:
        return False
    bi, li = idx
    return qp.observe_slot_edit(store, page.page_number, bi, li, char_index)


def _sync_chars_glyphs(line: Line, new_text: str) -> None:
    """改字后同步 line.chars[i].char，让 CharIndexService 重建后落到新集合。"""
    chars = getattr(line, "chars", None) or []
    if not chars:
        return
    if len(new_text) != len(chars):
        return
    for ch, glyph in zip(chars, new_text):
        if not getattr(ch, "char", None):
            continue
        if len(ch.char) != 1:
            return
    for ch, glyph in zip(chars, new_text):
        ch.char = glyph
