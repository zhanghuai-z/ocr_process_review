"""校对质量评测 — 评测位/假象字（Quality Probe）系统。

## 产品口径（Round 15 后）

"掺沙子"用来验证人工校对的质量。机制必须满足：

1. **少量、隐蔽** — 单次评测全文最多十几个 probe，单个同字 gallery 最多 1 个
   假象 crop，绝不能"整集合都变成假象"。
2. **假象字来源 = 文中真实存在的字** — 不许凭空生成假字，不许从字典里乱挑。
3. **混淆发生在切图 / 字形层** — line.text **绝不被修改**。混淆的是 VProof
   同字 gallery 里的 crop：在 "体" 的 gallery 中混入一个真实来自文档另一处
   "休" 的 crop（视觉接近）。
4. **若无合适的"文中已存在近形字"来源，则跳过** — 宁可不投放，也不硬造
   荒谬错配。
5. **不许图文错配** — 旧机制把 line.text[i] 替换成 fake_char、crop 仍然是
   true_char 的图，肉眼一比就穿帮。新机制只动 gallery 来源、不动文本。

## 数据模型

``Probe``：
- ``key`` → 文档中一个真实位置，该位置 line.text[char_index] 上的字就是
  ``true_char``（OCR 原字 / 用户视为"对"的字）。
- ``fake_char`` → 该 true_char 的视觉近形字（且在文档别处出现过）；它只在
  VProof "显示空间" 被注入到该位置，line.text 本身**永不改动**——这样
  CharIndexService 重建时该位置仍归属于 true_char 的正确集合。
- ``CONFUSION_MAP``
  中互为混淆字；同时文档里也至少有 1 处真实出现，否则用户根本不会进 gallery。
- ``observation`` → ``pending`` / ``corrected``。当用户在 VProof 槽位编辑
  框对 ``key`` 位置做出任何修改时，置 ``corrected``；其余保持 ``pending``。

## VProof 集成

- 用户点 "体" → ``CharIndexService.query("体")`` 给出真实 "体" 的 entries。
- 同时调 ``extras_for_gallery_char(store, "体", project)``，返回若干
  ``ProbeKey`` —— 它们指向文档里真实 "休" 的位置；VProof 把这些位置上的
  ``CharEntry`` 作为掺沙 crop 附加到 gallery 末尾。
- 用户若发现某个 crop 看起来不是 "体"，可点选并在槽位编辑框做修改；
  ``observe_slot_edit`` 接收该信号，将 probe 标 ``corrected``。

本模块**完全无 UI / Qt 依赖**，可被 CLI / pytest 单测。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

from app.core.proof_state import TOPIC_PROBE_OBSERVED
from app.models import OcrProject, Page, Block, Line
from app.models.enums import BlockType


# ──────────────────────────────────────────────────────────────────
# 混淆字表 —— 形似 CJK 字符对，用作"假象字"的来源。
# ──────────────────────────────────────────────────────────────────

_CONFUSABLES_RAW: tuple[tuple[str, ...], ...] = (
    ("己", "已", "巳"),
    ("末", "未"),
    ("戊", "戌", "戍"),
    ("千", "干"),
    ("入", "人", "八"),
    ("日", "曰"),
    ("土", "士"),
    ("大", "太", "犬"),
    ("王", "玉", "主"),
    ("田", "由", "甲", "申"),
    ("兔", "免"),
    ("鸟", "乌"),
    ("体", "休"),
    ("设", "没"),
    ("候", "侯"),
    ("即", "既"),
    ("拨", "拔"),
    ("折", "拆"),
    ("茶", "荼"),
    ("卷", "券"),
    ("睛", "晴", "情"),
    ("辨", "辩", "瓣"),
    ("载", "戴", "栽", "裁"),
    ("梁", "粱"),
    ("微", "徽"),
    ("赢", "羸", "嬴"),
    ("哀", "衷", "衰"),
    ("拼", "并"),
    ("渴", "喝", "竭"),
    ("睡", "腄"),
    ("淡", "谈"),
    ("辰", "晨"),
    ("木", "本"),
    ("自", "白"),
    ("洪", "供"),
    ("准", "淮"),
    ("住", "往", "佳"),
    ("待", "侍"),
    ("挺", "诞"),
    ("抑", "仰"),
    ("查", "杳"),
    ("喻", "渝"),
    ("氏", "民"),
    ("免", "兔"),
    ("乎", "平"),
    ("竞", "竟"),
    ("书", "韦"),
    ("辑", "缉"),
    ("贪", "贫"),
    ("饥", "饿"),
    ("园", "圆"),
    ("刺", "剌"),
)


def _build_confusion_map(raw: Iterable[Iterable[str]]) -> dict[str, tuple[str, ...]]:
    table: dict[str, list[str]] = {}
    for group in raw:
        chars = [c for c in group if c]
        for c in chars:
            for other in chars:
                if other != c:
                    table.setdefault(c, []).append(other)
    return {k: tuple(dict.fromkeys(v)) for k, v in table.items()}


CONFUSION_MAP: dict[str, tuple[str, ...]] = _build_confusion_map(_CONFUSABLES_RAW)


# ──────────────────────────────────────────────────────────────────
# 字符级判别工具
# ──────────────────────────────────────────────────────────────────

def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
    )


def _is_digit_like(ch: str) -> bool:
    return ch.isdigit() or ch in "０１２３４５６７８９.,，。·-"


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if _is_digit_like(c))
    return n / len(text)


_MATH_MARKERS = set("$^_{}=≈≠≤≥±∑∏∫√")
_MATH_MARKERS.add("\\")

_NAME_QUOTE_CHARS: frozenset[str] = frozenset(
    "「」『』《》〈〉【】〖〗［］〔〕“”‘’"
)
_NAME_GUARDED_MIN_RUN = 6


def _looks_like_math(text: str) -> bool:
    return any(c in _MATH_MARKERS for c in text)


# ──────────────────────────────────────────────────────────────────
# 行/块/字符可探测性判定（沿用 Round 14 前的口径）
# ──────────────────────────────────────────────────────────────────

EXCLUDED_BLOCK_TYPES: frozenset[BlockType] = frozenset({
    BlockType.TITLE,
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.TABLE_CAPTION,
    BlockType.FIGURE,
    BlockType.FIGURE_CAPTION,
    BlockType.REFERENCE,
    BlockType.UNKNOWN,
})

MIN_LINE_LEN = 6
MIN_CJK_RUN = 4
MAX_DIGIT_RATIO_IN_LINE = 0.5
LINE_EDGE_GUARD = 1


def is_block_eligible(block: Block) -> bool:
    if block.block_type in EXCLUDED_BLOCK_TYPES:
        return False
    if not block.recognizable:
        return False
    if not block.lines:
        return False
    return True


def is_line_eligible(line: Line) -> bool:
    text = line.text or ""
    if len(text) < MIN_LINE_LEN:
        return False
    if _digit_ratio(text) >= MAX_DIGIT_RATIO_IN_LINE:
        return False
    if _looks_like_math(text):
        return False
    return True


def candidate_indices_in_line(line: Line) -> list[int]:
    """返回行内"足以充当假象字来源"的字符 index 列表（按 line.text 空间）。

    新口径下，候选的语义是：**这个字本身就在 CONFUSION_MAP 中**（即它至少
    有一个视觉近形字），未来若需要 plant 它的近形字到对方 gallery 时，就
    从这些位置里选一个真实坐标。
    """
    text = line.text or ""
    if not is_line_eligible(line):
        return []
    n = len(text)
    cjk_flags = [_is_cjk(c) for c in text]

    cands: list[int] = []
    i = 0
    while i < n:
        if not cjk_flags[i]:
            i += 1
            continue
        j = i
        while j < n and cjk_flags[j]:
            j += 1
        run_start, run_end = i, j
        run_len = run_end - run_start
        left_neighbor = text[run_start - 1] if run_start - 1 >= 0 else ""
        right_neighbor = text[run_end] if run_end < n else ""
        adj_quote = (left_neighbor in _NAME_QUOTE_CHARS) or (right_neighbor in _NAME_QUOTE_CHARS)
        effective_min_run = _NAME_GUARDED_MIN_RUN if adj_quote else MIN_CJK_RUN
        if run_len >= effective_min_run:
            inner_start = max(run_start + LINE_EDGE_GUARD, LINE_EDGE_GUARD)
            inner_end = min(run_end - LINE_EDGE_GUARD, n - LINE_EDGE_GUARD)
            for k in range(inner_start, inner_end):
                ch = text[k]
                if ch in CONFUSION_MAP:
                    cands.append(k)
        i = j
    return cands


# ──────────────────────────────────────────────────────────────────
# Probe 数据结构
# ──────────────────────────────────────────────────────────────────

@dataclass
class ProbeKey:
    """文档中一个真实位置的结构化定位（重载安全）。"""
    page_number: int
    block_index: int
    line_index: int
    char_index: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.page_number, self.block_index, self.line_index, self.char_index)


@dataclass
class Probe:
    """掺沙记录。

    - ``key`` 指向**文档真实位置**，该位置 line.text 上的字符就是 ``fake_char``。
    - ``true_char`` 是该 fake_char 的视觉近形字，且它本身在文档里也至少出现过
      一次（否则用户不会进它的 gallery）。
    - ``observation`` = ``pending`` / ``corrected``；当用户在 VProof 槽位编辑
      对 ``key`` 这个位置做出任何修改时，置 ``corrected``。

    **重要不变量**：本模块**不会**修改 line.text 中任何字符；line.text[key.char_index]
    在投放前后始终等于 ``true_char``。fake_char 只通过
    ``app.services.proof_probe_text_service.displayed_text`` 注入到显示空间。
    """
    key: ProbeKey
    true_char: str
    fake_char: str
    observation: str = "pending"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = asdict(self.key)
        return d


# ──────────────────────────────────────────────────────────────────
# Probe 存储 —— 索引到 (page, block, line) 与 true_char
# ──────────────────────────────────────────────────────────────────

class ProbeStore:
    """单项目内的 probe 集合。"""

    def __init__(self) -> None:
        self._by_line: dict[tuple[int, int, int], list[Probe]] = {}
        self._by_true_char: dict[str, list[Probe]] = {}
        self._by_key: dict[tuple[int, int, int, int], Probe] = {}
        self._all: list[Probe] = []
        self.sampled_from_chars: int = 0
        self.target_probes: int = 0
        self.sand_count: Optional[int] = None
        self.sand_unit_chars: int = 1000

    def __len__(self) -> int:
        return len(self._all)

    def add(self, probe: Probe) -> None:
        self._all.append(probe)
        line_key = (probe.key.page_number, probe.key.block_index, probe.key.line_index)
        self._by_line.setdefault(line_key, []).append(probe)
        self._by_line[line_key].sort(key=lambda p: p.key.char_index)
        self._by_true_char.setdefault(probe.true_char, []).append(probe)
        self._by_key[probe.key.to_tuple()] = probe

    def for_line(self, page_number: int, block_index: int, line_index: int) -> list[Probe]:
        return list(self._by_line.get((page_number, block_index, line_index), ()))

    def for_true_char(self, char: str) -> list[Probe]:
        return list(self._by_true_char.get(char, ()))

    def by_key(self, key: ProbeKey) -> Optional[Probe]:
        return self._by_key.get(key.to_tuple())

    def all(self) -> list[Probe]:
        return list(self._all)

    def true_chars(self) -> list[str]:
        return list(self._by_true_char.keys())

    def clear(self) -> None:
        self._by_line.clear()
        self._by_true_char.clear()
        self._by_key.clear()
        self._all.clear()


# ──────────────────────────────────────────────────────────────────
# 候选位置辅助
# ──────────────────────────────────────────────────────────────────

def _has_existing_cut_char(line: Line, idx: int) -> bool:
    text = line.text or ""
    if not (0 <= idx < len(text) and 0 <= idx < len(line.chars)):
        return False
    ch = line.chars[idx]
    if ch.char != text[idx]:
        return False
    bbox = ch.bbox
    return bbox is not None and bbox.w > 0 and bbox.h > 0


def candidate_indices_with_existing_crops(line: Line) -> list[int]:
    """既可作沙源、又有现成 crop 的候选位置。"""
    return [idx for idx in candidate_indices_in_line(line) if _has_existing_cut_char(line, idx)]


def count_existing_cjk_crop_chars(line: Line) -> int:
    return sum(
        1
        for idx, ch in enumerate(line.text or "")
        if _is_cjk(ch) and _has_existing_cut_char(line, idx)
    )


# ──────────────────────────────────────────────────────────────────
# 采样器
# ──────────────────────────────────────────────────────────────────

@dataclass
class SamplerConfig:
    """掺沙采样配置。

    Round 17：上一轮 (max_per_page=2, max_per_line=1, max_per_true_char=1,
    max_total=35) 把"密度"挤成"硬上限"，导致 1000 字文档配置 "每千字 20" 也只
    出 2~4 颗沙。本轮把硬上限全部放宽，让 ``target_density`` 真正决定投放数
    （仅保留极少量防御性上限以避免病态文档下整行/整页爆炸）。
    """
    target_ratio: float = 0.02
    sand_count: Optional[int] = None
    sand_unit_chars: int = 1000
    min_total: int = 4
    max_total: int = 999          # 仅作硬性保险（数千字工程也不会触发）
    max_per_page: int = 999       # 不再按页限流，密度自己决定
    max_per_line: int = 5         # 单行不超过 5 颗，避免病态短行被打满
    max_per_true_char: int = 99   # 单字 gallery 不再被强行卡 1
    seed: Optional[int] = None

    def target_density(self) -> float:
        if self.sand_count is None:
            return max(0.0, float(self.target_ratio))
        return max(0.0, float(self.sand_count)) / max(1, int(self.sand_unit_chars))


@dataclass
class _Candidate:
    key: ProbeKey
    char: str                     # 该位置的真实字符（= 未来 probe.true_char）


class ProbeSampler:
    """新口径采样器（Round 15）：

    1. 扫全文，收集所有 "在 CONFUSION_MAP 中、有 crop 的可探测位置"。这些
       位置就是未来沙子的**来源**（fake_char 真实出现的位置）。
    2. 把候选按字符聚合得到 ``pool_by_char``。
    3. 对每个候选 (key, true_char=cand.char)，确定 ``fake_char`` 候选 =
       ``CONFUSION_MAP[true_char]`` 与 chars_in_doc 的交集（保证
       fake_char 也确实在文档里出现过；并且 ``fake_char != true_char``）。
       若交集为空，跳过该候选 —— 这就是"宁可不投，不硬造错配"。
    4. 随机选 ``target`` 个 (key, true_char) → 给每个分配 fake_char，构造 Probe。
       同时遵守 ``max_per_page`` / ``max_per_line`` / ``max_per_true_char``
       （max_per_true_char 现在按 cand.char = true_char 计上限）。
    """

    def __init__(self, config: Optional[SamplerConfig] = None) -> None:
        self.cfg = config or SamplerConfig()

    def sample(self, project: OcrProject) -> ProbeStore:
        rng = random.Random(self.cfg.seed)
        store = ProbeStore()

        # 1. 扫整文档收集候选（每个位置即是未来 probe 的 true_char 位置）
        pool: list[_Candidate] = []
        total_cut_cjk = 0
        for page in project.pages:
            for bi, block in enumerate(page.blocks):
                if not is_block_eligible(block):
                    continue
                for li, line in enumerate(block.lines):
                    total_cut_cjk += count_existing_cjk_crop_chars(line)
                    text = line.text or ""
                    for idx in candidate_indices_with_existing_crops(line):
                        pool.append(_Candidate(
                            key=ProbeKey(page.page_number, bi, li, idx),
                            char=text[idx],
                        ))

        store.sampled_from_chars = total_cut_cjk
        store.sand_count = self.cfg.sand_count
        store.sand_unit_chars = self.cfg.sand_unit_chars
        if not pool:
            return store

        # 2. 按字符聚合 + 计算目标投放数
        chars_in_doc = {c.char for c in pool}
        if self.cfg.sand_count is None:
            target = int(round(total_cut_cjk * self.cfg.target_density()))
            target = max(self.cfg.min_total, target)
        else:
            target = math.ceil(total_cut_cjk * self.cfg.target_density()) if self.cfg.sand_count > 0 else 0
        target = min(self.cfg.max_total, target)
        store.target_probes = target
        if target <= 0:
            return store

        # 3. 随机洗牌，按上限投放
        rng.shuffle(pool)
        per_page_count: dict[int, int] = {}
        per_line_count: dict[tuple[int, int, int], int] = {}
        per_true_char_count: dict[str, int] = {}
        placed = 0

        for cand in pool:
            if placed >= target:
                break
            line_key = (cand.key.page_number, cand.key.block_index, cand.key.line_index)
            if per_page_count.get(cand.key.page_number, 0) >= self.cfg.max_per_page:
                continue
            if per_line_count.get(line_key, 0) >= self.cfg.max_per_line:
                continue
            # Round 18 起重定义：
            #   probe.true_char = cand.char = 该位置 line.text 上的原字符（"正确字"）
            #   probe.fake_char = CONFUSION_MAP[cand.char] ∩ chars_in_doc \ {cand.char}
            #                      （即"显示空间里要注入的假象字"，且文档中实际存在）
            # line.text 永远保持 true_char；displayed_text 才把该位置渲染成 fake_char。
            if per_true_char_count.get(cand.char, 0) >= self.cfg.max_per_true_char:
                continue
            confusables = CONFUSION_MAP.get(cand.char, ())
            fake_options = [c for c in confusables if c in chars_in_doc and c != cand.char]
            if not fake_options:
                # 关键产品口径：找不到"文中已存在的近形字"做 fake_char，则跳过此候选
                continue
            fake_ch = rng.choice(fake_options)
            probe = Probe(key=cand.key, true_char=cand.char, fake_char=fake_ch)
            store.add(probe)
            per_page_count[cand.key.page_number] = per_page_count.get(cand.key.page_number, 0) + 1
            per_line_count[line_key] = per_line_count.get(line_key, 0) + 1
            per_true_char_count[cand.char] = per_true_char_count.get(cand.char, 0) + 1
            placed += 1

        return store


# ──────────────────────────────────────────────────────────────────
# Gallery 注入接口（供 VProof 调用）
# ──────────────────────────────────────────────────────────────────

def extras_for_gallery_char(store: Optional[ProbeStore], char: str) -> list[Probe]:
    """返回所有"应当被掺入 ``char`` 的同字 gallery"的 probes。

    VProof 拿到这些 probe 后，用 ``probe.key`` 去定位文档真实位置，构造对应
    ``CharEntry`` 追加到 gallery 末尾。

    若 ``store`` 为 None（评测未开启），返回空列表。
    """
    if store is None:
        return []
    return store.for_true_char(char)


# ──────────────────────────────────────────────────────────────────
# 观测接口（供 VProof 槽位编辑流程调用）
# ──────────────────────────────────────────────────────────────────

# TOPIC_PROBE_OBSERVED is imported from app.core.proof_state and re-exported
# here as a compatibility alias; do not redefine the literal in this module.


def observe_slot_edit(
    store: Optional[ProbeStore],
    page_number: int,
    block_index: int,
    line_index: int,
    char_index: int,
) -> bool:
    """用户在 VProof 槽位编辑框对 ``(page, block, line, char_index)`` 做出修改时调用。

    如果该位置正好命中某个 probe，则将 observation 标 ``corrected``、
    通过 ProofStateBus 广播 ``probe.observed`` 事件、并返回 ``True``；
    否则返回 ``False`` 不做事。
    """
    if store is None:
        return False
    key = ProbeKey(page_number, block_index, line_index, char_index)
    probe = store.by_key(key)
    if probe is None:
        return False
    was_pending = probe.observation != "corrected"
    probe.observation = "corrected"
    if was_pending:
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


# ──────────────────────────────────────────────────────────────────
# 评分
# ──────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    total_probes: int
    corrected: int
    pending: int
    grade: str            # "A" / "B" / "C" / "D" / "INSUFFICIENT"
    grade_label: str
    band: str             # 简短描述区间
    detect_ratio: float   # corrected / total （仅用于内部展示，不暴露假精度）


_GRADE_LABELS = {
    "A": "优秀",
    "B": "良好",
    "C": "一般",
    "D": "较差",
    "INSUFFICIENT": "样本不足",
}


def score(store: ProbeStore, *, min_observed_for_grade: int = 4) -> QualityReport:
    total = len(store)
    corrected = sum(1 for p in store.all() if p.observation == "corrected")
    pending = total - corrected
    if total < min_observed_for_grade:
        return QualityReport(
            total_probes=total, corrected=corrected, pending=pending,
            grade="INSUFFICIENT", grade_label=_GRADE_LABELS["INSUFFICIENT"],
            band=f"抽样字符过少（{total}/{min_observed_for_grade}）",
            detect_ratio=0.0,
        )
    ratio = corrected / total if total else 0.0
    if ratio >= 0.85:
        grade = "A"
    elif ratio >= 0.65:
        grade = "B"
    elif ratio >= 0.40:
        grade = "C"
    else:
        grade = "D"
    band = f"抽样字符 {total}，识破 {corrected}（{int(round(ratio * 100))}%）"
    return QualityReport(
        total_probes=total, corrected=corrected, pending=pending,
        grade=grade, grade_label=_GRADE_LABELS[grade], band=band,
        detect_ratio=ratio,
    )


# ──────────────────────────────────────────────────────────────────
# AppConfig → SamplerConfig
# ──────────────────────────────────────────────────────────────────

_SAMPLER_CONFIG_KEYS: dict[str, tuple[str, type]] = {
    "quality_probe_sand_count": ("sand_count", int),
    "quality_probe_sand_unit_chars": ("sand_unit_chars", int),
    "quality_probe_target_ratio": ("target_ratio", float),
    "quality_probe_min_total": ("min_total", int),
    "quality_probe_max_total": ("max_total", int),
    "quality_probe_max_per_page": ("max_per_page", int),
    "quality_probe_max_per_line": ("max_per_line", int),
    "quality_probe_max_per_true_char": ("max_per_true_char", int),
}


def sampler_config_from_app_config() -> SamplerConfig:
    cfg = SamplerConfig()
    try:
        from app.core.app_config import AppConfig
    except Exception:
        return cfg
    try:
        app_cfg = AppConfig.instance()
    except Exception:
        return cfg
    for key, (field_name, caster) in _SAMPLER_CONFIG_KEYS.items():
        raw = app_cfg.get(key, None)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, field_name, caster(raw))
        except (TypeError, ValueError):
            continue
    return cfg


# ──────────────────────────────────────────────────────────────────
# 混淆字表运行时扩展
# ──────────────────────────────────────────────────────────────────

_EXTRA_CONFUSABLES: list[tuple[str, ...]] = []


def _rebuild_confusion_map() -> None:
    global CONFUSION_MAP
    merged = list(_CONFUSABLES_RAW) + [tuple(g) for g in _EXTRA_CONFUSABLES]
    CONFUSION_MAP = _build_confusion_map(merged)


def register_confusion_group(group: Iterable[str]) -> None:
    chars = tuple(dict.fromkeys(c for c in group if c))
    if len(chars) < 2:
        return
    _EXTRA_CONFUSABLES.append(chars)
    _rebuild_confusion_map()


def _load_confusables_extra_file() -> None:
    import json
    from pathlib import Path
    try:
        path = Path(__file__).with_name("confusables_extra.json")
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return
        for group in data:
            if isinstance(group, (list, tuple)):
                register_confusion_group(group)
    except Exception:
        return


_load_confusables_extra_file()


# ──────────────────────────────────────────────────────────────────
# 持久化 sidecar
# ──────────────────────────────────────────────────────────────────

_SIDECAR_SUFFIX = ".qprobe.json"
# Round 15 改了 Probe 语义：旧 v1 sidecar 的 (true_char, fake_char) 现在含义反了，
# 一律不读，让 sampler 重采。
_SIDECAR_VERSION = 2


def sidecar_path_for_project(db_path: Optional[str]) -> Optional[str]:
    if not db_path:
        return None
    return str(db_path) + _SIDECAR_SUFFIX


def store_to_dict(store: ProbeStore) -> dict:
    return {
        "version": _SIDECAR_VERSION,
        "sampled_from_chars": store.sampled_from_chars,
        "target_probes": store.target_probes,
        "sand_count": store.sand_count,
        "sand_unit_chars": store.sand_unit_chars,
        "probes": [p.to_dict() for p in store.all()],
    }


def store_from_dict(data: dict) -> ProbeStore:
    store = ProbeStore()
    if not isinstance(data, dict):
        return store
    if data.get("version") != _SIDECAR_VERSION:
        return store
    try:
        store.sampled_from_chars = int(data.get("sampled_from_chars", 0) or 0)
        store.target_probes = int(data.get("target_probes", 0) or 0)
        sand_count = data.get("sand_count", None)
        store.sand_count = None if sand_count is None else int(sand_count)
        store.sand_unit_chars = int(data.get("sand_unit_chars", 1000) or 1000)
    except (TypeError, ValueError):
        pass
    for raw in data.get("probes", []):
        try:
            key_d = raw.get("key", {})
            probe = Probe(
                key=ProbeKey(
                    page_number=int(key_d["page_number"]),
                    block_index=int(key_d["block_index"]),
                    line_index=int(key_d["line_index"]),
                    char_index=int(key_d["char_index"]),
                ),
                true_char=str(raw["true_char"]),
                fake_char=str(raw["fake_char"]),
                observation=str(raw.get("observation", "pending")),
            )
            store.add(probe)
        except (KeyError, TypeError, ValueError):
            continue
    return store


def save_store_to_path(store: ProbeStore, path: str) -> bool:
    import json
    from pathlib import Path
    try:
        Path(path).write_text(
            json.dumps(store_to_dict(store), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def load_store_from_path(path: str) -> Optional[ProbeStore]:
    import json
    from pathlib import Path
    try:
        p = Path(path)
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return store_from_dict(data)


# ──────────────────────────────────────────────────────────────────
# 全局单例（单项目）
# ──────────────────────────────────────────────────────────────────

_active_store: Optional[ProbeStore] = None


def get_active_store() -> Optional[ProbeStore]:
    return _active_store


def set_active_store(store: Optional[ProbeStore]) -> None:
    global _active_store
    _active_store = store


def reset_active_store() -> None:
    set_active_store(None)


# ──────────────────────────────────────────────────────────────────
# Round 18：基于文本锚点的批量校验
# ──────────────────────────────────────────────────────────────────

def detect_corrections(
    store: Optional[ProbeStore],
    project_or_pages,
) -> int:
    """以 line.text 为锚扫一遍 store 内所有 pending probe：

    若 ``line.text[probe.key.char_index]`` 已不再是 ``probe.fake_char``（
    或更严格地说，已不再是 probe 当初投放时的位置内容），即认为该位置发生过
    "用户真实编辑"。把该 probe 标 corrected 并广播 ``probe.observed``。

    返回新被标 corrected 的 probe 数量。无 store / 无 project 时返回 0。

    用途：
    - VProof 保存当前页之后做一次主动校验，避免 displayed↔true 桥外的路径
      （比如 _apply_replacement_to_selected）漏报；
    - QualityStatsDialog 的"立刻刷新"按钮调一次，做最可靠的兜底刷新触发。
    """
    if store is None or project_or_pages is None:
        return 0
    pages = getattr(project_or_pages, "pages", project_or_pages)
    # 先把 page_number → page 缓存好
    pages_by_no: dict[int, "Page"] = {pg.page_number: pg for pg in pages}
    newly = 0
    for probe in store.all():
        if probe.observation == "corrected":
            continue
        page = pages_by_no.get(probe.key.page_number)
        if page is None:
            continue
        if not (0 <= probe.key.block_index < len(page.blocks)):
            continue
        block = page.blocks[probe.key.block_index]
        if not (0 <= probe.key.line_index < len(block.lines)):
            continue
        line = block.lines[probe.key.line_index]
        text = line.text or ""
        ci = probe.key.char_index
        if ci < 0 or ci >= len(text):
            # 行被截短 → 位置已被破坏，视为"用户改过"
            _mark_and_broadcast(probe)
            newly += 1
            continue
        # 正确性判定：line.text 上该位置不再是 fake_char ⇒ 用户改过
        # （probe 投放后 displayed_text 给出 fake_char，line.text 始终为 true_char；
        #  用户在显示空间里改成任何非 fake_char 的字符，反向写回都会让 line.text
        #  脱离原 true_char——这正是"以文本为锚点回正确集合"的信号。）
        if text[ci] != probe.true_char:
            _mark_and_broadcast(probe)
            newly += 1
    return newly


def _mark_and_broadcast(probe: Probe) -> None:
    probe.observation = "corrected"
    try:
        from app.core.proof_state import ProbeObservation
        from app.core.proof_state_bus import ProofStateBus
        ProofStateBus.instance().publish_probe_observed(ProbeObservation(
            page_number=probe.key.page_number,
            block_index=probe.key.block_index,
            line_index=probe.key.line_index,
            char_index=probe.key.char_index,
            true_char=probe.true_char,
            fake_char=probe.fake_char,
        ))
    except Exception:
        pass
