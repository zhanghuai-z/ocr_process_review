"""校对质量评测系统单元测试 — Round 15 语义。

关键不变量：
- line.text 永不被修改。
- 假象字（fake_char）始终来自文档真实位置；true_char 也必须在文档中出现过。
- 找不到合适的"文中近形字"时跳过候选，不硬造错配。
- score 用 corrected/total 计算 A/B/C/D 段；总数 < min_observed 时报 INSUFFICIENT。
- v1 sidecar 不再兼容（语义已反），读到 v1 返回空 store。
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.core import quality_probe as qp
from app.core.quality_probe import (
    CONFUSION_MAP,
    EXCLUDED_BLOCK_TYPES,
    MIN_CJK_RUN,
    MIN_LINE_LEN,
    Probe,
    ProbeKey,
    ProbeSampler,
    ProbeStore,
    QualityReport,
    SamplerConfig,
    candidate_indices_in_line,
    extras_for_gallery_char,
    is_block_eligible,
    is_line_eligible,
    load_store_from_path,
    observe_slot_edit,
    register_confusion_group,
    sampler_config_from_app_config,
    save_store_to_path,
    score,
)
from app.models import BBox, Block, Char, Line, OcrProject, Page
from app.models.enums import BlockType


# ──────────────────────────────────────────────────────────────────
# 辅助构造
# ──────────────────────────────────────────────────────────────────

def _line(text: str, conf: float = 0.95) -> Line:
    return Line(
        text=text,
        confidence=conf,
        bbox=BBox(0, 0, max(1, len(text)) * 12, 20),
        chars=[
            Char(
                char=ch,
                confidence=conf,
                bbox=BBox(i * 12, 0, 12, 20),
                bbox_source="ocr",
                bbox_granularity="char",
                token_text=ch,
            )
            for i, ch in enumerate(text)
        ],
    )


def _block(btype: BlockType, lines: list[Line]) -> Block:
    return Block(block_type=btype, bbox=BBox(0, 0, 100, 200), lines=lines)


def _page(page_no: int, blocks: list[Block]) -> Page:
    return Page(image_path=f"/tmp/p{page_no}.png", width=100, height=200,
                blocks=blocks, page_number=page_no)


def _project(pages: list[Page]) -> OcrProject:
    return OcrProject(name="t", pages=pages)


# ──────────────────────────────────────────────────────────────────
# CONFUSION_MAP / 候选 / 资格判定
# ──────────────────────────────────────────────────────────────────

def test_confusion_map_symmetric_and_self_excluded():
    assert "已" in CONFUSION_MAP["己"] and "己" in CONFUSION_MAP["已"]
    assert "休" in CONFUSION_MAP["体"] and "体" in CONFUSION_MAP["休"]
    for k, vs in CONFUSION_MAP.items():
        assert k not in vs


def test_block_eligibility_excludes_specific_types():
    line = _line("这是一段足够长的正文用来测试可投放性")
    for bt in EXCLUDED_BLOCK_TYPES:
        assert not is_block_eligible(_block(bt, [line])), bt
    assert is_block_eligible(_block(BlockType.TEXT, [line]))


def test_block_eligibility_uses_dispatch_policy_for_source_labels():
    from app.core.ocr_dispatch_policy import default_ocr_policy_for_block

    line = _line("这是一段足够长的正文用来测试可投放性")
    formula_like = _block(BlockType.TEXT, [line])
    formula_like.source_label = "inline_formula"
    formula_like.ocr_policy = default_ocr_policy_for_block(formula_like)

    table_like = _block(BlockType.TEXT, [line])
    table_like.source_label = "table"
    table_like.ocr_policy = default_ocr_policy_for_block(table_like)

    text_without_structured_label = _block(BlockType.TEXT, [line])
    text_without_structured_label.ocr_policy = default_ocr_policy_for_block(text_without_structured_label)

    assert not is_block_eligible(formula_like)
    assert not is_block_eligible(table_like)
    assert is_block_eligible(text_without_structured_label)
    assert is_block_eligible(_block(BlockType.TEXT, [line]))


def test_line_eligibility_rejects_short_digit_math_lines():
    assert not is_line_eligible(_line("太短"))
    assert not is_line_eligible(_line("12345678901234"))
    assert not is_line_eligible(_line("公式 $x^2 = y$"))
    assert is_line_eligible(_line("这是一段足够长的中文正文"))


def test_candidate_indices_require_in_confusion_map():
    # 一段够长的 CJK，但里面没有 CONFUSION_MAP 里的字 → 空候选
    line = _line("窗外秋叶纷纷落下")
    assert candidate_indices_in_line(line) == []


def test_candidate_indices_skip_edges_and_short_runs():
    # CONFUSION_MAP 里的字 "体" 出现在内部位置应能被选
    line = _line("这本书要靠身体力行去理解")
    cands = candidate_indices_in_line(line)
    text = line.text
    assert all(0 < i < len(text) - 1 for i in cands)
    assert any(text[i] in CONFUSION_MAP for i in cands)


def test_candidate_indices_avoid_name_quotes():
    # 「...」紧邻时，effective_min_run 提高到 6
    line = _line("人物「体力很好」之类的描述要被书名号守卫")
    # 不强断言条数，只保证不会把书名号内的 "体" 选成候选
    cands = candidate_indices_in_line(line)
    text = line.text
    # 确保没有任何候选落在 「」 内紧邻
    for i in cands:
        if i - 1 >= 0:
            assert text[i - 1] not in "「『《〈"
        if i + 1 < len(text):
            assert text[i + 1] not in "」』》〉"


# ──────────────────────────────────────────────────────────────────
# Sampler
# ──────────────────────────────────────────────────────────────────

def _doc_with_both(true_ch: str, fake_ch: str, n_fake: int = 5) -> OcrProject:
    """构造一份文档：至少 1 处 true_ch，n_fake 处 fake_ch，足以做沙源。"""
    # 一行包含 true_ch（让 true_ch 在 chars_in_doc 中），多行包含 fake_ch（候选源）
    lines = [_line(f"开头有一个{true_ch}字在这里读着没问题")]
    for _ in range(n_fake):
        lines.append(_line(f"这一句里有个{fake_ch}字插进来给采样用"))
    block = _block(BlockType.TEXT, lines)
    return _project([_page(1, [block])])


def test_sampler_only_plants_pairs_both_in_doc():
    project = _doc_with_both("已", "己", n_fake=5)
    cfg = SamplerConfig(target_ratio=0.5, seed=1, max_per_line=1,
                        max_per_page=10, max_per_true_char=10)
    store = ProbeSampler(cfg).sample(project)
    assert len(store) > 0
    for probe in store.all():
        # Round 18：true_char 是 line.text 中 key 位置的原字符；
        # fake_char 是要注入"显示空间"的近形字，也必须在文档别处出现过。
        page = project.pages[0]
        block = page.blocks[probe.key.block_index]
        line = block.lines[probe.key.line_index]
        assert line.text[probe.key.char_index] == probe.true_char
        all_text = "".join(
            l.text for b in page.blocks for l in b.lines
        )
        assert probe.fake_char in all_text
        assert probe.true_char != probe.fake_char
        assert probe.fake_char in CONFUSION_MAP.get(probe.true_char, ())


def test_sampler_skips_when_no_in_doc_confusable_exists():
    # fake_char 候选 "拼"（CONFUSION 是 "并"），但文档里没有 "并"
    # → sampler 应跳过，store 为空（且 line.text 当然没被改）
    line = _line("拼起来读读看再拼一次拼写练习好不好")
    project = _project([_page(1, [_block(BlockType.TEXT, [line])])])
    store = ProbeSampler(SamplerConfig(target_ratio=1.0, seed=1)).sample(project)
    assert len(store) == 0


def test_sampler_max_per_true_char_respects_explicit_cap():
    """Round 17：默认 max_per_true_char 已放宽到 99；显式压回 1 仍生效。"""
    project = _doc_with_both("体", "休", n_fake=20)
    cfg = SamplerConfig(target_ratio=1.0, seed=42, max_per_page=20,
                        max_per_line=1, max_per_true_char=1)
    store = ProbeSampler(cfg).sample(project)
    by_true = {}
    for p in store.all():
        by_true[p.true_char] = by_true.get(p.true_char, 0) + 1
    for tc, n in by_true.items():
        assert n <= 1, f"true_char={tc} got {n} probes"


def test_sampler_respects_max_per_page_and_per_line():
    project = _doc_with_both("已", "己", n_fake=10)
    cfg = SamplerConfig(target_ratio=1.0, seed=7, max_per_page=2,
                        max_per_line=1, max_per_true_char=10)
    store = ProbeSampler(cfg).sample(project)
    page_counts = {}
    line_counts = {}
    for p in store.all():
        page_counts[p.key.page_number] = page_counts.get(p.key.page_number, 0) + 1
        lk = (p.key.page_number, p.key.block_index, p.key.line_index)
        line_counts[lk] = line_counts.get(lk, 0) + 1
    for n in page_counts.values():
        assert n <= 2
    for n in line_counts.values():
        assert n <= 1


def test_sampler_seed_deterministic():
    project = _doc_with_both("已", "己", n_fake=10)
    cfg = lambda: SamplerConfig(target_ratio=0.5, seed=123, max_per_page=20,
                                 max_per_true_char=10)
    s1 = ProbeSampler(cfg()).sample(project)
    s2 = ProbeSampler(cfg()).sample(project)
    a1 = [p.key.to_tuple() for p in s1.all()]
    a2 = [p.key.to_tuple() for p in s2.all()]
    assert a1 == a2


def test_sampler_zero_sand_count_disables():
    project = _doc_with_both("已", "己", n_fake=10)
    cfg = SamplerConfig(sand_count=0, sand_unit_chars=1000, seed=1)
    store = ProbeSampler(cfg).sample(project)
    assert len(store) == 0


def test_sampler_density_via_sand_count():
    project = _doc_with_both("已", "己", n_fake=30)
    # density = 50/1000 = 5%，min_total=8 不生效（min_total 只对 ratio 路径生效）
    cfg = SamplerConfig(sand_count=50, sand_unit_chars=1000, seed=1,
                        max_per_page=50, max_per_true_char=50, max_per_line=50,
                        max_total=100)
    store = ProbeSampler(cfg).sample(project)
    assert len(store) >= 1


def test_sampler_excluded_block_never_sampled():
    line = _line("已经过去的事情还要继续讨论吗")
    title_block = _block(BlockType.TITLE, [line])
    text_block = _block(BlockType.TEXT, [_line("已经过去的事情还要继续讨论吗")])
    project = _project([_page(1, [title_block, text_block])])
    store = ProbeSampler(SamplerConfig(target_ratio=1.0, seed=1,
                                       max_per_true_char=10)).sample(project)
    for p in store.all():
        # block_index 0 = TITLE，不该出现
        assert p.key.block_index != 0


def test_line_text_never_modified_by_sampler():
    project = _doc_with_both("已", "己", n_fake=5)
    snapshot = [
        (li, l.text)
        for p in project.pages for b in p.blocks for li, l in enumerate(b.lines)
    ]
    ProbeSampler(SamplerConfig(target_ratio=1.0, seed=1,
                               max_per_true_char=10)).sample(project)
    after = [
        (li, l.text)
        for p in project.pages for b in p.blocks for li, l in enumerate(b.lines)
    ]
    assert snapshot == after


# ──────────────────────────────────────────────────────────────────
# Gallery 注入 + 观测
# ──────────────────────────────────────────────────────────────────

def test_extras_for_gallery_char_returns_by_true_char():
    store = ProbeStore()
    store.add(Probe(ProbeKey(1, 0, 0, 5), true_char="体", fake_char="休"))
    store.add(Probe(ProbeKey(1, 0, 1, 3), true_char="已", fake_char="己"))
    assert len(extras_for_gallery_char(store, "体")) == 1
    assert len(extras_for_gallery_char(store, "已")) == 1
    assert extras_for_gallery_char(store, "毫无关系") == []
    assert extras_for_gallery_char(None, "体") == []


def test_observe_slot_edit_marks_corrected_only_on_key_hit():
    store = ProbeStore()
    p = Probe(ProbeKey(1, 0, 0, 5), true_char="体", fake_char="休")
    store.add(p)
    assert observe_slot_edit(store, 1, 0, 0, 99) is False
    assert p.observation == "pending"
    assert observe_slot_edit(store, 1, 0, 0, 5) is True
    assert p.observation == "corrected"
    assert observe_slot_edit(None, 1, 0, 0, 5) is False


# ──────────────────────────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────────────────────────

def test_score_insufficient_when_too_few_probes():
    store = ProbeStore()
    store.add(Probe(ProbeKey(1, 0, 0, 1), "已", "己", "corrected"))
    r = score(store, min_observed_for_grade=4)
    assert r.grade == "INSUFFICIENT"


def test_score_bands_A_B_C_D():
    def _store_with(ratio: float, total: int = 20) -> ProbeStore:
        s = ProbeStore()
        n_corr = round(ratio * total)
        for i in range(total):
            obs = "corrected" if i < n_corr else "pending"
            s.add(Probe(ProbeKey(1, 0, 0, i), "已", "己", obs))
        return s
    assert score(_store_with(0.90)).grade == "A"
    assert score(_store_with(0.70)).grade == "B"
    assert score(_store_with(0.50)).grade == "C"
    assert score(_store_with(0.20)).grade == "D"


# ──────────────────────────────────────────────────────────────────
# Sidecar
# ──────────────────────────────────────────────────────────────────

def test_sidecar_round_trip_v2(tmp_path: Path):
    store = ProbeStore()
    store.sampled_from_chars = 1234
    store.target_probes = 10
    store.sand_count = 25
    store.add(Probe(ProbeKey(1, 0, 0, 5), "体", "休", "corrected"))
    store.add(Probe(ProbeKey(2, 1, 0, 3), "已", "己"))
    path = tmp_path / "x.qprobe.json"
    assert save_store_to_path(store, str(path))
    loaded = load_store_from_path(str(path))
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded.sampled_from_chars == 1234
    assert loaded.sand_count == 25
    by_key = {p.key.to_tuple(): p for p in loaded.all()}
    p = by_key[(1, 0, 0, 5)]
    assert (p.true_char, p.fake_char, p.observation) == ("体", "休", "corrected")


def test_sidecar_v1_returns_empty(tmp_path: Path):
    # 旧 v1 格式：版本号 1，应被识别为不兼容 → 空 store
    path = tmp_path / "old.qprobe.json"
    path.write_text(json.dumps({
        "version": 1,
        "probes": [{
            "key": {"page_number": 1, "block_index": 0,
                    "line_index": 0, "char_index": 5},
            "true_char": "已", "fake_char": "己",
            "observation": "pending",
        }],
    }), encoding="utf-8")
    loaded = load_store_from_path(str(path))
    assert loaded is not None
    assert len(loaded) == 0


# ──────────────────────────────────────────────────────────────────
# AppConfig 桥
# ──────────────────────────────────────────────────────────────────

def test_sampler_config_from_app_config_reads_max_per_true_char(monkeypatch):
    class _FakeAppConfig:
        _data = {
            "quality_probe_sand_count": 30,
            "quality_probe_max_per_true_char": 2,
            "quality_probe_max_per_page": 3,
        }
        @classmethod
        def instance(cls):
            return cls()
        def get(self, key, default=None):
            return self._data.get(key, default)

    import app.core.app_config as mod
    monkeypatch.setattr(mod, "AppConfig", _FakeAppConfig, raising=False)
    cfg = sampler_config_from_app_config()
    assert cfg.sand_count == 30
    assert cfg.max_per_true_char == 2
    assert cfg.max_per_page == 3


# ──────────────────────────────────────────────────────────────────
# register_confusion_group
# ──────────────────────────────────────────────────────────────────

def test_register_confusion_group_extends_map():
    # 选两个原本不在表里的字（先取一个独家组合不存在的字符对）
    a, b = "甘", "苷"
    assert b not in qp.CONFUSION_MAP.get(a, ())
    register_confusion_group([a, b])
    try:
        assert b in qp.CONFUSION_MAP[a]
        assert a in qp.CONFUSION_MAP[b]
    finally:
        # 清理：从 _EXTRA_CONFUSABLES 删掉并重建
        qp._EXTRA_CONFUSABLES.clear()
        qp._rebuild_confusion_map()
