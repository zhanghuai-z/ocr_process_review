"""校对质量评测系统单元测试。"""
from __future__ import annotations

import pytest

from app.core.quality_probe import (
    CONFUSION_MAP,
    EXCLUDED_BLOCK_TYPES,
    LINE_EDGE_GUARD,
    MIN_CJK_RUN,
    MIN_LINE_LEN,
    Probe,
    ProbeKey,
    ProbeSampler,
    ProbeStore,
    QualityReport,
    SamplerConfig,
    apply_probes_to_display,
    candidate_indices_in_line,
    is_block_eligible,
    is_line_eligible,
    observe_user_action,
    reverse_display_to_true,
    score,
)
from app.models import BBox, Block, Line, OcrProject, Page
from app.models.enums import BlockType, ProofStatus


# ──────────────────────────────────────────────────────────────────
# 辅助：构建测试 project
# ──────────────────────────────────────────────────────────────────

def _line(text: str, conf: float = 0.95) -> Line:
    return Line(text=text, confidence=conf, bbox=BBox(0, 0, 100, 20))


def _block(btype: BlockType, lines: list[Line]) -> Block:
    return Block(block_type=btype, bbox=BBox(0, 0, 100, 200), lines=lines)


def _page(page_no: int, blocks: list[Block]) -> Page:
    return Page(image_path=f"/tmp/p{page_no}.png", width=100, height=200,
                blocks=blocks, page_number=page_no)


def _project(pages: list[Page]) -> OcrProject:
    return OcrProject(name="t", pages=pages)


# ──────────────────────────────────────────────────────────────────
# 字符级判别
# ──────────────────────────────────────────────────────────────────

def test_confusion_map_is_symmetric_and_nonempty():
    assert CONFUSION_MAP["己"]
    assert "已" in CONFUSION_MAP["己"]
    assert "己" in CONFUSION_MAP["已"]
    # 自己不能出现在自己的混淆候选里
    for k, vs in CONFUSION_MAP.items():
        assert k not in vs


def test_block_eligibility_excludes_titles_equations_tables():
    line = _line("这是一段足够长的正文用来测试可投放性")
    for bt in EXCLUDED_BLOCK_TYPES:
        assert not is_block_eligible(_block(bt, [line])), bt
    assert is_block_eligible(_block(BlockType.TEXT, [line]))


def test_line_eligibility_rejects_short_digit_math_lines():
    assert not is_line_eligible(_line("太短"))
    assert not is_line_eligible(_line("123456789012"))   # 数字串
    assert not is_line_eligible(_line("公式 $x^2 = y$"))   # 含数学
    assert is_line_eligible(_line("这是一段足够长的中文正文"))


def test_candidate_indices_skip_edges_and_short_runs():
    # "这是A段太短的混合B文本测试" -- short CJK runs split by ASCII
    text = "ab这是中文cd其他段落ef"
    line = _line(text)
    cands = candidate_indices_in_line(line)
    # 没有任一段 >= MIN_CJK_RUN(=4) 且包含混淆字 -- 期待空
    # "这是中文" len=4，但 inner = [run+1, run+3) -> 1 个内部位置
    # 但仅当该字在 CONFUSION_MAP 中才入选
    for idx in cands:
        assert text[idx] in CONFUSION_MAP


def test_candidate_indices_basic_long_run():
    text = "今天我们来学习已经发生过的历史事件本身"
    line = _line(text)
    cands = candidate_indices_in_line(line)
    # 必须至少有一个 (因为含"已"、"本"等混淆字)
    assert cands
    # 不能是行首或行尾
    assert all(LINE_EDGE_GUARD <= i < len(text) - LINE_EDGE_GUARD for i in cands)
    # 命中字符必须在混淆表
    assert all(text[i] in CONFUSION_MAP for i in cands)


# ──────────────────────────────────────────────────────────────────
# 采样器
# ──────────────────────────────────────────────────────────────────

def _build_dense_project(n_pages: int = 5, lines_per_page: int = 8) -> OcrProject:
    """构造一个 CJK 字符量充足的 project。"""
    pages = []
    template_lines = [
        "今天我们来学习已经发生过的历史事件本身",
        "千万记得不要把入这件事和人那件事混淆起来",
        "他设法解决了一个非常困难的实际工程问题",
        "她日记里记录了候鸟迁徙的几个重要时间点",
        "我们应当辨别这件事情的本质和它的表面现象",
        "卷宗里写明了这次拨款的具体来源和去向情况",
        "同学们站起来说明了自己对这道题目的理解",
        "他在田间观察了很多关于植物生长的细节变化",
    ]
    for p in range(1, n_pages + 1):
        lines = [_line(template_lines[i % len(template_lines)])
                 for i in range(lines_per_page)]
        block = _block(BlockType.TEXT, lines)
        pages.append(_page(p, [block]))
    return _project(pages)


def test_sampler_respects_per_page_and_per_line_caps():
    proj = _build_dense_project(n_pages=10, lines_per_page=6)
    cfg = SamplerConfig(target_ratio=0.10, min_total=8, max_total=35,
                        max_per_page=2, max_per_line=1, seed=42)
    store = ProbeSampler(cfg).sample(proj)
    # 每页 ≤ 2
    per_page: dict[int, int] = {}
    per_line: dict[tuple[int, int, int], int] = {}
    for p in store.all():
        per_page[p.key.page_number] = per_page.get(p.key.page_number, 0) + 1
        lk = (p.key.page_number, p.key.block_index, p.key.line_index)
        per_line[lk] = per_line.get(lk, 0) + 1
    assert all(c <= 2 for c in per_page.values()), per_page
    assert all(c <= 1 for c in per_line.values()), per_line
    # 总数在 [8, 35]
    assert 8 <= len(store) <= 35


def test_sampler_min_total_floor():
    """即使 target_ratio 极小，也要保证至少 min_total 个 probe 被投放。"""
    proj = _build_dense_project(n_pages=10, lines_per_page=6)
    cfg = SamplerConfig(target_ratio=0.0001, min_total=8, max_total=35,
                        max_per_page=2, max_per_line=1, seed=7)
    store = ProbeSampler(cfg).sample(proj)
    assert len(store) >= 8


def test_sampler_excluded_blocks_never_get_probes():
    """标题/公式/表格等块绝不投放。"""
    title = _block(BlockType.TITLE, [_line("第一章 历史的开端与结束")])
    eq = _block(BlockType.EQUATION, [_line("已知方程的解是已经存在的")])
    text = _block(BlockType.TEXT, [_line("今天我们来学习已经发生过的历史事件本身")])
    page = _page(1, [title, eq, text])
    proj = _project([page] * 5)
    cfg = SamplerConfig(target_ratio=0.5, min_total=8, max_total=35,
                        max_per_page=2, max_per_line=1, seed=0)
    store = ProbeSampler(cfg).sample(proj)
    # 所有 probe 必须落在 block_index=2 (text 块)
    assert all(p.key.block_index == 2 for p in store.all())


def test_sampler_deterministic_with_seed():
    proj = _build_dense_project(n_pages=5, lines_per_page=6)
    cfg = SamplerConfig(seed=123)
    s1 = [p.to_dict() for p in ProbeSampler(cfg).sample(proj).all()]
    s2 = [p.to_dict() for p in ProbeSampler(cfg).sample(proj).all()]
    assert s1 == s2


# ──────────────────────────────────────────────────────────────────
# 显示空间 ↔ 真实空间映射
# ──────────────────────────────────────────────────────────────────

def test_apply_probes_substitutes_only_at_indices():
    text = "今天我们来学习已经发生过的历史"
    # 0:今 1:天 2:我 3:们 4:来 5:学 6:习 7:已 ...
    probes = [
        Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己"),
    ]
    out = apply_probes_to_display(text, probes)
    assert out[7] == "己"
    assert out.replace("己", "已") == text


def test_apply_probes_safe_when_text_changed():
    """如果真实文本已被改过 (true_char 不再匹配)，跳过该 probe。"""
    text = "今天我们来学习是经发生过的历史"  # 第7字改成"是"
    probes = [Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")]
    assert apply_probes_to_display(text, probes) == text


def test_reverse_corrected_missed_edited_other():
    before = "今天我们来学习已经发生过的历史"
    probes = [Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")]
    displayed = apply_probes_to_display(before, probes)
    assert displayed[7] == "己"

    # case A: 用户察觉并改回了 "已"
    true_a, results_a = reverse_display_to_true(before, displayed[:7] + "已" + displayed[8:], probes)
    assert true_a == before
    assert results_a == [(probes[0], "corrected")]

    # case B: 用户没察觉，保留 "己"
    true_b, results_b = reverse_display_to_true(before, displayed, probes)
    assert true_b == before          # 还原成 true_char
    assert results_b == [(probes[0], "missed")]

    # case C: 用户改成第三个字符 "巳"
    other = displayed[:7] + "巳" + displayed[8:]
    true_c, results_c = reverse_display_to_true(before, other, probes)
    assert true_c == other           # 保留用户修改
    assert results_c == [(probes[0], "edited_other")]


def test_reverse_large_change_preserves_user_intent():
    """用户大幅改写，放弃逐位对齐，但仍保证返回不含 fake_char。"""
    before = "今天我们来学习已经发生过的历史"
    probes = [Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")]
    displayed = apply_probes_to_display(before, probes)
    rewritten = "用户重新写了一段完全不同的文字内容已经"
    true_text, results = reverse_display_to_true(before, rewritten, probes)
    # 全部标 edited_other
    assert all(obs == "edited_other" for _, obs in results)
    # 即使是大改也不应留下 fake_char "己"（如果"已"已存在则不替换）
    if "已" not in true_text:
        assert "己" not in true_text


def test_reverse_deleted_position():
    before = "今天我们来学习已经发生过的历史"
    probes = [Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")]
    truncated = before[:7]   # 用户删掉了第7字之后的所有内容
    true_text, results = reverse_display_to_true(before, truncated, probes)
    assert results[0][1] in ("deleted", "edited_other")  # 大改或删除


def test_observe_priority_does_not_downgrade():
    """已被判 corrected 的 probe 不会被后续 pending 误覆盖。"""
    store = ProbeStore()
    p = Probe(ProbeKey(1, 0, 0, 7), true_char="已", fake_char="己")
    store.add(p)
    before = "今天我们来学习已经发生过的历史"
    displayed = apply_probes_to_display(before, [p])
    # 第一次：用户改回正字 -> corrected
    observe_user_action(store, 1, 0, 0, before, displayed[:7] + "已" + displayed[8:])
    assert p.observation == "corrected"
    # 模拟一次"未触及该位置"的保存（重新拿真实文本去渲染再保存）
    again_displayed = apply_probes_to_display(before, [p])  # 又会被替换成 fake
    # 但用户这次没看到 -- 模拟保存时用户保留了 fake：按规则会判 missed
    # 优先级机制应保留 corrected 不被 missed 覆盖
    observe_user_action(store, 1, 0, 0, before, again_displayed)
    assert p.observation == "corrected"


# ──────────────────────────────────────────────────────────────────
# 评分
# ──────────────────────────────────────────────────────────────────

def test_score_insufficient_when_few_observed():
    store = ProbeStore()
    for i in range(3):
        store.add(Probe(ProbeKey(1, 0, 0, i), true_char="已", fake_char="己"))
    rep = score(store)
    assert rep.grade == "INSUFFICIENT"
    assert rep.grade_label == "样本不足"


def test_score_grades_by_band():
    def make(n_corrected: int, n_missed: int) -> QualityReport:
        s = ProbeStore()
        for i in range(n_corrected):
            p = Probe(ProbeKey(1, 0, 0, i), true_char="已", fake_char="己")
            p.observation = "corrected"
            s.add(p)
        for i in range(n_missed):
            p = Probe(ProbeKey(1, 0, 0, 100 + i), true_char="己", fake_char="已")
            p.observation = "missed"
            s.add(p)
        return score(s)

    assert make(9, 1).grade == "A"     # 90%
    assert make(7, 3).grade == "B"     # 70%
    assert make(5, 5).grade == "C"     # 50%
    assert make(2, 8).grade == "D"     # 20%


# ──────────────────────────────────────────────────────────────────
# 不污染最终文本 —— 关键不变量
# ──────────────────────────────────────────────────────────────────

def test_export_text_never_contains_fake_chars_via_reverse_path():
    """模拟主程序保存路径：用户保存的 line.text 必须不含 fake_char。"""
    proj = _build_dense_project(n_pages=3, lines_per_page=4)
    cfg = SamplerConfig(target_ratio=0.10, min_total=8, max_total=20,
                        max_per_page=2, max_per_line=1, seed=11)
    store = ProbeSampler(cfg).sample(proj)
    assert len(store) > 0

    # 模拟：每行的"显示文本"被用户原样保存（即用户没改）
    for page in proj.pages:
        for bi, block in enumerate(page.blocks):
            for li, line in enumerate(block.lines):
                probes = store.for_line(page.page_number, bi, li)
                if not probes:
                    continue
                displayed = apply_probes_to_display(line.text, probes)
                true_text = observe_user_action(
                    store, page.page_number, bi, li, line.text, displayed,
                )
                # 写回 line.text -- 这是导出器唯一会读的地方
                line.text = true_text

    # 关键不变量：所有 line.text 必须不含任何 probe 的 fake_char
    fakes_in_play = {p.fake_char for p in store.all()}
    for page in proj.pages:
        for block in page.blocks:
            for line in block.lines:
                for fake in fakes_in_play:
                    # 注意：fake_char 也可能正好出现在文本其他位置作为正字。
                    # 严格的不变量是"probe 位置的字符"应是 true_char。
                    # 这里换一种检查：每个 probe 位置必须 == true_char
                    pass
    for p in store.all():
        page = next(pg for pg in proj.pages if pg.page_number == p.key.page_number)
        block = page.blocks[p.key.block_index]
        line = block.lines[p.key.line_index]
        assert line.text[p.key.char_index] == p.true_char, (
            f"probe 位置 {p.key.to_tuple()} 写回后竟然是 {line.text[p.key.char_index]!r}, "
            f"期望 {p.true_char!r} (fake={p.fake_char!r})"
        )

    # 评分应给出 missed=全部 (用户全没改)，但等级有效
    rep = score(store)
    assert rep.total_probes == len(store)
    assert rep.missed >= rep.corrected   # 全没改


def test_txt_exporter_emits_no_fake_chars_after_round_trip(tmp_path):
    """端到端：经过 probe round-trip 后，TxtExporter 输出绝不含 fake_char。"""
    from app.export.txt import TxtExporter

    proj = _build_dense_project(n_pages=3, lines_per_page=4)
    cfg = SamplerConfig(target_ratio=0.10, min_total=8, max_total=20,
                        max_per_page=2, max_per_line=1, seed=99)
    store = ProbeSampler(cfg).sample(proj)
    assert len(store) > 0

    # 模拟用户校对：一半的 probe 用户改回正字，另一半保留 fake (没察觉)
    probes = store.all()
    for i, p in enumerate(probes):
        page = next(pg for pg in proj.pages if pg.page_number == p.key.page_number)
        block = page.blocks[p.key.block_index]
        line = block.lines[p.key.line_index]
        displayed = apply_probes_to_display(line.text, [p])
        if i % 2 == 0:
            # 用户察觉，改回正字
            user_text = displayed[:p.key.char_index] + p.true_char + displayed[p.key.char_index + 1:]
        else:
            # 用户没察觉，原样保存
            user_text = displayed
        true_text = observe_user_action(
            store, page.page_number, p.key.block_index, p.key.line_index,
            line.text, user_text,
        )
        line.text = true_text

    # 调用真实的 TxtExporter
    out_path = tmp_path / "out.txt"
    TxtExporter().export(proj, str(out_path))
    content = out_path.read_text(encoding="utf-8")

    # 关键：每个 probe 位置最终落到导出文件里时必须是 true_char
    # 这里间接检查 -- 每行的真实文本必须出现在导出里
    for page in proj.pages:
        for block in page.blocks:
            for line in block.lines:
                if line.text:
                    assert line.text in content, f"line.text 应原样出现在导出: {line.text!r}"

    # 强不变量：每个 probe 位置上的字符 == true_char
    for p in probes:
        page = next(pg for pg in proj.pages if pg.page_number == p.key.page_number)
        block = page.blocks[p.key.block_index]
        line = block.lines[p.key.line_index]
        assert line.text[p.key.char_index] == p.true_char
