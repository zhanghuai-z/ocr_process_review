"""Quality-probe display/true-text bridge.

- proof runtime state 永远保存"真实"文本（未被掺沙），CharIndexService 等正常集合
  视图始终基于 ``proof_display_text(line)``。
- ``displayed_text(line, page, block)`` 返回"显示空间"文本：若该 line 有处于
  ``observation == "pending"`` 的 probe，则把真实文本对应位置
  替换成 ``probe.fake_char``，得到与"真实 OCR 错字"在文本窗口中表现一致的
  视图。**长度不变、位置一一对应**（apply_probes_to_display 是位置等长替换）。
- ``save_displayed_edit_result(line, page, block, displayed_new)`` 把"用户在显示空
  间里编辑后的整行"反向映射回 proof runtime state：
    - 若 displayed_new[i] == 旧 displayed_old[i]：该位置用户没动 → 保持
      真实文本不变（probe 仍 pending）；
    - 若 displayed_new[i] != displayed_old[i]：该位置发生了真实编辑 →
      真实文本对应位置写为 displayed_new[i]；若 i 命中某个 pending probe，则把该
      probe 标 ``corrected``（即"以文本为锚点回正确集合"——proof text 上的
      字符变了，CharIndexService 重建后该位置自动归到新字符的 gallery）。
    - 若 displayed_new 与 displayed_old 长度不同：用旧显示文本和真实文本
      做差异映射；未触碰的旧显示片段回填真实文本，避免 fake_char 回灌到
      final_text。结构性编辑会让本行 pending probe 全部转 corrected。
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional, Tuple

from app.models import Block, Line, Page
from app.core import quality_probe as qp
from app.core.proof_char_text import chars_display_spans, is_display_carrier
from app.core.proof_change import ProofChangeSet
from app.core.proof_line_facts import proof_display_text
from app.core.proof_line_mutation import set_line_proof_text


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
    """显示给用户的文本。

    若该 line 上有 ``observation == "pending"`` 的 probe，则把对应位置替换成
    ``fake_char``；其余位置原样输出 proof text。无 active store / 无 probe
    时返回 ``proof_display_text(line)``。
    """
    base = proof_display_text(line)
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
        if 0 <= ci < len(chars) and chars[ci] == probe.true_char:
            chars[ci] = probe.fake_char
    return "".join(chars)


def save_displayed_edit_result(
    line: Line, page: Page, block: Block, displayed_new_text: str
) -> ProofChangeSet:
    """把编辑结果写入 proof runtime state 并返回结构化变更集。

    若 (page, block, line) 上挂有 probes，则按"显示空间 → 真实空间"逐位反向
    映射：原显示位置若是 fake_char 且用户没动它，真实文本保持原 true_char；
    用户动过的位置直接写回真实文本，并把命中的 pending probe 标 corrected。
    """
    store = qp.get_active_store()
    idx = resolve_block_line_index(page, block, line)

    if idx is None:
        return ProofChangeSet(cancelled=True)

    base_true = proof_display_text(line)
    probes_pending = _pending_probes_for_line(store, page, idx)
    probes_pending, probe_changed = _correct_stale_probe_anchors(
        page_number=page.page_number,
        line_index=idx,
        base_true=base_true,
        probes=probes_pending,
    )

    displayed_old = displayed_text(line, page, block)
    if displayed_new_text == displayed_old:
        return ProofChangeSet(probe_changed=probe_changed)

    new_true, mapped_probe_changed = _reconcile_display_edit_to_true_text(
        page_number=page.page_number,
        line_index=idx,
        base_true=base_true,
        displayed_old=displayed_old,
        displayed_new=displayed_new_text,
        probes_pending=probes_pending,
    )
    probe_changed = probe_changed or mapped_probe_changed

    if new_true == proof_display_text(line):
        return ProofChangeSet(text_changed=False, probe_changed=probe_changed)
    set_line_proof_text(line, new_true)
    _sync_chars_glyphs(line, new_true)
    return ProofChangeSet(text_changed=True, probe_changed=probe_changed, index_changed=True)


def _pending_probes_for_line(
    store: qp.ProbeStore | None,
    page: Page,
    line_index: Tuple[int, int],
) -> list[qp.Probe]:
    if store is None:
        return []
    bi, li = line_index
    return [
        probe
        for probe in store.for_line(page.page_number, bi, li)
        if probe.observation == "pending"
    ]


def _correct_stale_probe_anchors(
    *,
    page_number: int,
    line_index: Tuple[int, int],
    base_true: str,
    probes: list[qp.Probe],
) -> tuple[list[qp.Probe], bool]:
    bi, li = line_index
    changed = False
    for probe in list(probes):
        ci = probe.key.char_index
        if ci < 0 or ci >= len(base_true) or base_true[ci] != probe.true_char:
            changed = _mark_probe_corrected(probe, page_number, bi, li, ci) or changed
    return [probe for probe in probes if probe.observation == "pending"], changed


def _reconcile_display_edit_to_true_text(
    *,
    page_number: int,
    line_index: Tuple[int, int],
    base_true: str,
    displayed_old: str,
    displayed_new: str,
    probes_pending: list[qp.Probe],
) -> tuple[str, bool]:
    if probes_pending and len(displayed_new) == len(displayed_old) == len(base_true):
        return _map_equal_display_edit_to_true_text(
            page_number=page_number,
            line_index=line_index,
            base_true=base_true,
            displayed_old=displayed_old,
            displayed_new=displayed_new,
            probes_pending=probes_pending,
        )
    if probes_pending:
        new_true = _map_non_equal_display_edit_to_true_text(
            base_true=base_true,
            displayed_old=displayed_old,
            displayed_new=displayed_new,
        )
        return new_true, _mark_structural_edit_probes_corrected(
            page_number=page_number,
            line_index=line_index,
            probes_pending=probes_pending,
        )
    return displayed_new, False


def _map_equal_display_edit_to_true_text(
    *,
    page_number: int,
    line_index: Tuple[int, int],
    base_true: str,
    displayed_old: str,
    displayed_new: str,
    probes_pending: list[qp.Probe],
) -> tuple[str, bool]:
    bi, li = line_index
    changed = False
    new_true_chars = list(base_true)
    probes_by_idx = {probe.key.char_index: probe for probe in probes_pending}
    for ci in range(len(displayed_old)):
        if displayed_new[ci] == displayed_old[ci]:
            continue
        if 0 <= ci < len(new_true_chars):
            new_true_chars[ci] = displayed_new[ci]
        probe = probes_by_idx.get(ci)
        if probe is not None:
            changed = _mark_probe_corrected(probe, page_number, bi, li, ci) or changed
    return "".join(new_true_chars), changed


def _mark_structural_edit_probes_corrected(
    *,
    page_number: int,
    line_index: Tuple[int, int],
    probes_pending: list[qp.Probe],
) -> bool:
    bi, li = line_index
    changed = False
    for probe in probes_pending:
        changed = (
            _mark_probe_corrected(probe, page_number, bi, li, probe.key.char_index)
            or changed
        )
    return changed


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


def _mark_probe_corrected(
    probe: qp.Probe,
    page_number: int,
    block_index: int,
    line_index: int,
    char_index: int,
) -> bool:
    if probe.observation == "corrected":
        return False
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
    return True


def _sync_chars_glyphs(line: Line, new_text: str) -> None:
    """改字后同步 ``line.chars`` 表达的文本事实。

    ``token_text`` 在 word/formula 场景可能是多字符载体。同步时不能只看
    ``len(new_text) == len(chars)``，而要先用当前 chars 重建旧显示串，再按
    显示 span 回写。若无法无损对齐，就不改 chars；下游 ProofAtom/CharIndex
    会根据 mismatch 降级，避免旧 carrier 被当成可靠事实继续传播。
    """
    chars = getattr(line, "chars", None) or []
    if not chars:
        return
    actions = _char_sync_actions(chars, new_text)
    if actions is None:
        return
    for indices, replacement in actions:
        span_chars = [chars[idx] for idx in indices]
        if len(span_chars) == 1:
            ch = span_chars[0]
            old_char = ch.char
            ch.char = replacement
            token_text = getattr(ch, "token_text", None)
            if is_display_carrier(ch) or len(replacement) > 1:
                ch.token_text = replacement
            elif _token_text_should_follow_glyph(ch, token_text, old_char):
                ch.token_text = replacement
            continue
        for ch, glyph in zip(span_chars, replacement):
            old_char = ch.char
            ch.char = glyph
            token_text = getattr(ch, "token_text", None)
            if _token_text_should_follow_glyph(ch, token_text, old_char):
                ch.token_text = glyph


def _char_sync_actions(chars, new_text: str):
    spans = chars_display_spans(chars)
    old_display = "".join(span.text for span in spans)
    if len(old_display) != len(new_text):
        return None
    actions: list[tuple[tuple[int, ...], str]] = []
    for span in spans:
        replacement = new_text[span.start:span.end]
        span_chars = [chars[idx] for idx in span.char_indices]
        if len(span_chars) == 1:
            if len(replacement) == 1 or is_display_carrier(span_chars[0]) or len(span.text) > 1:
                actions.append((span.char_indices, replacement))
                continue
            return None
        if len(replacement) != len(span_chars):
            return None
        if any(len(str(getattr(ch, "char", "") or "")) > 1 for ch in span_chars):
            return None
        actions.append((span.char_indices, replacement))
    return actions


def _token_text_should_follow_glyph(ch, token_text, old_char: str) -> bool:
    if token_text is None:
        return False
    token = str(token_text)
    if not token:
        return False
    if len(token) <= 1 or token == old_char:
        return True
    if not is_display_carrier(ch):
        return False
    source = f"{getattr(ch, 'bbox_source', '')} {getattr(ch, 'bbox_granularity', '')}".lower()
    if "formula" in source or "equation" in source:
        return False
    return True
