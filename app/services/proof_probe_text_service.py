"""Quality-probe service layer —— Round 18 重新启用 display ↔ true 桥。

设计（Round 18）：

- ``line.final_text`` 永远保存"真实"文本（未被掺沙），CharIndexService 等正常集合
  视图始终基于真实文本。
- ``displayed_text(line, page, block)`` 返回"显示空间"文本：若该 line 有处于
  ``observation == "pending"`` 的 probe，则把 ``line.final_text[probe.char_index]``
  替换成 ``probe.fake_char``，得到与"真实 OCR 错字"在文本窗口中表现一致的
  视图。**长度不变、位置一一对应**（apply_probes_to_display 是位置等长替换）。
- ``save_displayed_edit(line, page, block, displayed_new)`` 把"用户在显示空
  间里编辑后的整行"反向映射回 ``line.final_text``：
    - 若 displayed_new[i] == 旧 displayed_old[i]：该位置用户没动 → 保持
      line.final_text[i] 不变（probe 仍 pending）；
    - 若 displayed_new[i] != displayed_old[i]：该位置发生了真实编辑 →
      line.final_text[i] = displayed_new[i]；若 i 命中某个 pending probe，则把该
      probe 标 ``corrected``（即"以文本为锚点回正确集合"——line.final_text 上的
      字符变了，CharIndexService 重建后该位置自动归到新字符的 gallery）。
    - 若 displayed_new 与 displayed_old 长度不同：用旧显示文本和真实文本
      做差异映射；未触碰的旧显示片段回填真实文本，避免 fake_char 回灌到
      final_text。结构性编辑会让本行 pending probe 全部转 corrected。

- ``observe_slot_edit_at`` 仍保留：原 v_proof 槽位编辑流程在某些路径里会
  直接报点，因此保留这个薄包装作为冗余报点入口。
"""
from __future__ import annotations

from difflib import SequenceMatcher
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
    """显示给用户的文本 —— Round 18 起恢复"显示空间"语义：

    若该 line 上有 ``observation == "pending"`` 的 probe，则把对应位置替换成
    ``fake_char``；其余位置原样输出 ``line.final_text``。无 active store / 无 probe
    时返回 ``line.final_text``。
    """
    base = line.display_text
    store = qp.get_active_store()
    if store is None or not base:
        return base
    idx = resolve_block_line_index(page, block, line)
    if idx is None:
        return base
    bi, li = idx
    probes = store.for_line(page.page_number, bi, li)
    if not probes:
        return base
    chars = list(base)
    for probe in probes:
        if probe.observation != "pending":
            continue
        ci = probe.key.char_index
        if 0 <= ci < len(chars):
            chars[ci] = probe.fake_char
    return "".join(chars)


def save_displayed_edit(
    line: Line, page: Page, block: Block, displayed_new_text: str
) -> bool:
    """把编辑结果落盘到 ``line.final_text``。返回 ``True`` 表示发生变化。

    若 (page, block, line) 上挂有 probes，则按"显示空间 → 真实空间"逐位反向
    映射：原显示位置若是 fake_char 且用户没动它，line.final_text 保持原 true_char；
    用户动过的位置直接写回 line.final_text，并把命中的 pending probe 标 corrected。
    """
    displayed_old = displayed_text(line, page, block)
    if displayed_new_text == displayed_old:
        return False

    store = qp.get_active_store()
    idx = resolve_block_line_index(page, block, line)

    # 走 probe-aware 反向路径的前提：有 store、能定位 (bi, li)
    probes_pending: list[qp.Probe] = []
    if store is not None and idx is not None:
        bi, li = idx
        probes_pending = [p for p in store.for_line(page.page_number, bi, li)
                          if p.observation == "pending"]

    base_true = line.display_text
    if probes_pending and len(displayed_new_text) == len(displayed_old) == len(base_true):
        # 等长路径：精确按位反向映射
        new_true_chars = list(base_true)
        probes_by_idx = {p.key.char_index: p for p in probes_pending}
        for ci in range(len(displayed_old)):
            if displayed_new_text[ci] == displayed_old[ci]:
                continue  # 用户没动这一位
            # 用户动过这一位：写真实文本
            if 0 <= ci < len(new_true_chars):
                new_true_chars[ci] = displayed_new_text[ci]
            probe = probes_by_idx.get(ci)
            if probe is not None:
                _mark_probe_corrected(probe, page.page_number, idx[0], idx[1], ci)
        new_true = "".join(new_true_chars)
    else:
        if probes_pending:
            new_true = _map_non_equal_display_edit_to_true_text(
                base_true=base_true,
                displayed_old=displayed_old,
                displayed_new=displayed_new_text,
            )
        else:
            new_true = displayed_new_text
        # 结构性编辑后 probe 的位置语义可能漂移；保守转 corrected。
        if probes_pending and idx is not None:
            bi, li = idx
            for p in probes_pending:
                _mark_probe_corrected(p, page.page_number, bi, li, p.key.char_index)

    if new_true == line.display_text:
        return False
    line.update_text(new_true)
    _sync_chars_glyphs(line, new_true)
    return True


def _map_non_equal_display_edit_to_true_text(
    *,
    base_true: str,
    displayed_old: str,
    displayed_new: str,
) -> str:
    """Map a structural display-space edit back into true text space.

    Equal spans are copied from ``base_true`` so unchanged fake chars injected
    into ``displayed_old`` cannot be persisted. Insert/replace spans are user
    input and are copied from ``displayed_new``.
    """
    if len(base_true) != len(displayed_old):
        return displayed_new
    parts: list[str] = []
    matcher = SequenceMatcher(a=displayed_old, b=displayed_new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(base_true[i1:i2])
        elif tag in {"insert", "replace"}:
            parts.append(displayed_new[j1:j2])
        elif tag == "delete":
            continue
    return "".join(parts)


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


def _mark_probe_corrected(
    probe: qp.Probe,
    page_number: int,
    block_index: int,
    line_index: int,
    char_index: int,
) -> None:
    if probe.observation == "corrected":
        return
    probe.observation = "corrected"
    try:
        from app.core.proof_state import ProbeObservation
        from app.core.proof_state_bus import ProofStateBus
        ProofStateBus.instance().publish_probe_observed(ProbeObservation(
            page_number=page_number,
            block_index=block_index,
            line_index=line_index,
            char_index=char_index,
            true_char=probe.true_char,
            fake_char=probe.fake_char,
        ))
    except Exception:
        pass


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
