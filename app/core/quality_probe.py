"""校对质量评测 — 评测位/假象字（Quality Probe）系统。

设计要点（必须严格遵守，否则会污染最终交付文本）：

1. **数据分层**
   - 评测位只是一个旁路记录，**绝不改写 line.text**。
   - 真实文本始终保存在 ``line.text``。
   - 用户校对界面在渲染时通过 ``apply_probes_to_display`` 把若干字符
     替换成 ``fake_char`` 显示出来。
   - 用户保存时通过 ``reverse_display_to_true`` 把"显示空间文本"还原成
     "真实空间文本"，再写回 ``line.text``。
   - 因此任何导出器（txt / docx / html / pdf / rtf / xml）只读 ``line.text``，
     就一定不会拿到 ``fake_char``。

2. **采样约束**（在 ``ProbeSampler`` 中实现）
   - 新配置按 ``sand_count / sand_unit_chars`` 计算全局密度。
   - 旧比例配置仍兼容 ``target_ratio``、单任务下限和单页上限。
   - 新密度模式不再套用旧的单页 2 个硬上限，避免用户调高密度后无感。
   - 单行最多 ``1`` 个
   - 用 seedable RNG 保证可复现

3. **避让规则**
   - ``BlockType`` 排除：TITLE / EQUATION / TABLE / TABLE_CAPTION /
     FIGURE / FIGURE_CAPTION / REFERENCE / UNKNOWN
   - 整行排除：行长 < ``MIN_LINE_LEN``、含 ≥50% 数字（数字串）、
     行内含 LaTeX/数学符号（``$``、反斜杠、``^``、``_``、``{`` 等）。
   - 字符排除：非 CJK（数字/字母/标点/空白）、不在混淆表里的字符、
     行首/行尾各 1 字、行内"短 CJK 段"（连续 < 4 个 CJK 字符）的所有位置。
     这条"短 CJK 段排除"是工程上对"人名/地名"的近似避让 —— 短独立 CJK
     段往往就是人名地名签名等敏感串。

4. **观测/统计**
   - ``observe_user_action(line_id, displayed_new_text)`` 在用户保存时被调用，
     既返回应该写回 ``line.text`` 的真实文本，也更新 probe 的 observation：
     - ``"corrected"``：用户在该位置写回了 true_char（成功识破假象）
     - ``"missed"``：用户保留了 fake_char（被假象骗过）
     - ``"edited_other"``：用户把该位置写成了既不是 true 也不是 fake 的内容
       （视为"修了，但可能是改错了"，单独归类，不计入正错）
     - ``"deleted"``：该字符被删掉（改字数，按 missed 处理 + 标记）
   - ``QualityScorer`` 给出**等级 + 区间**（不给伪精确百分比）。

5. **本模块完全无 UI 依赖**，可被 CLI / pytest 单测。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

from app.models import OcrProject, Page, Block, Line
from app.models.enums import BlockType


# ──────────────────────────────────────────────────────────────────
# 混淆字表 —— 形似 CJK 字符对，用作"假象字"的来源。
# 仅包含视觉上人眼可能扫读时看错、但内容上足够明确的"明显形近字"。
# 不包含同义字、繁简对、罕见字，避免给用户造成"系统居然在乱来"的不信任感。
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
        0x4E00 <= cp <= 0x9FFF       # CJK Unified
        or 0x3400 <= cp <= 0x4DBF    # Ext A
        or 0xF900 <= cp <= 0xFAFF    # Compat
    )


def _is_digit_like(ch: str) -> bool:
    return ch.isdigit() or ch in "０１２３４５６７８９.,，。·-"


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if _is_digit_like(c))
    return n / len(text)


_MATH_MARKERS = set("$^_{}=≈≠≤≥±∑∏∫√")  # noqa: RUF001
# 也屏蔽反斜杠 (LaTeX 命令引导符)
_MATH_MARKERS.add("\\")

# CJK 引号/书名号/方头括号 —— 用于"人名/地名/书名"近似避让。
# 出现这些符号附近的短 CJK 段（≤5 字）极可能是人名地名书名，强制提高 MIN_CJK_RUN
# 阈值，避免在敏感串里投放假象字。
_NAME_QUOTE_CHARS: frozenset[str] = frozenset(
    "「」『』《》〈〉【】〖〗［］〔〕“”‘’"
)
_NAME_GUARDED_MIN_RUN = 6  # 紧邻名号符号时，CJK 段必须 ≥ 这么长才允许投放


def _looks_like_math(text: str) -> bool:
    return any(c in _MATH_MARKERS for c in text)


# ──────────────────────────────────────────────────────────────────
# 行/块/字符可探测性判定
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

MIN_LINE_LEN = 6                # 行少于这么多字不投放
MIN_CJK_RUN = 4                 # 连续 CJK 段至少这么长，且只能投放在段中间（非首末两字）
MAX_DIGIT_RATIO_IN_LINE = 0.5   # 数字串行排除
LINE_EDGE_GUARD = 1             # 行首行尾各保护这么多字符


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
    """返回行内可投放假象字的字符 index 列表（按真实文本空间）。"""
    text = line.text or ""
    if not is_line_eligible(line):
        return []
    n = len(text)
    cjk_flags = [_is_cjk(c) for c in text]

    # 找出每个 index 所属的"连续 CJK 段"长度，以及在段内的位置
    # 只接受位于"段长 >= MIN_CJK_RUN"且"非段首末 LINE_EDGE_GUARD 个字符"的 index
    cands: list[int] = []
    i = 0
    while i < n:
        if not cjk_flags[i]:
            i += 1
            continue
        j = i
        while j < n and cjk_flags[j]:
            j += 1
        run_start, run_end = i, j  # [run_start, run_end)
        run_len = run_end - run_start
        # 名号近邻判定：若 CJK 段紧邻引号/书名号（左 1 字或右 1 字），
        # 则该段视为"可能是人名/地名/书名"，将最低段长门槛抬到 _NAME_GUARDED_MIN_RUN。
        left_neighbor = text[run_start - 1] if run_start - 1 >= 0 else ""
        right_neighbor = text[run_end] if run_end < n else ""
        adj_quote = (left_neighbor in _NAME_QUOTE_CHARS) or (right_neighbor in _NAME_QUOTE_CHARS)
        effective_min_run = _NAME_GUARDED_MIN_RUN if adj_quote else MIN_CJK_RUN
        if run_len >= effective_min_run:
            inner_start = run_start + LINE_EDGE_GUARD
            inner_end = run_end - LINE_EDGE_GUARD
            # 同时也要避开整行行首/行尾保护
            inner_start = max(inner_start, LINE_EDGE_GUARD)
            inner_end = min(inner_end, n - LINE_EDGE_GUARD)
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
    """结构化定位 key（不依赖 DB id，重载安全）。"""
    page_number: int
    block_index: int            # Page.blocks 中的下标
    line_index: int             # Block.lines 中的下标
    char_index: int             # line.text 中字符下标（真实文本空间）

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.page_number, self.block_index, self.line_index, self.char_index)


@dataclass
class Probe:
    key: ProbeKey
    true_char: str
    fake_char: str
    observation: str = "pending"   # pending / corrected / missed / edited_other / deleted

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = asdict(self.key)
        return d


# ──────────────────────────────────────────────────────────────────
# Probe 存储 —— 按 (page_number, block_index, line_index) 索引
# ──────────────────────────────────────────────────────────────────

class ProbeStore:
    """单项目内的 probe 集合。线程内部默认非并发。"""

    def __init__(self) -> None:
        self._by_line: dict[tuple[int, int, int], list[Probe]] = {}
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
        # 行内按 char_index 升序，方便逆向映射
        self._by_line[line_key].sort(key=lambda p: p.key.char_index)

    def for_line(self, page_number: int, block_index: int, line_index: int) -> list[Probe]:
        return list(self._by_line.get((page_number, block_index, line_index), ()))

    def all(self) -> list[Probe]:
        return list(self._all)

    def clear(self) -> None:
        self._by_line.clear()
        self._all.clear()


# ──────────────────────────────────────────────────────────────────
# 采样器
# ──────────────────────────────────────────────────────────────────

@dataclass
class SamplerConfig:
    target_ratio: float = 0.025      # 旧配置兼容：未设置 sand_count 时使用
    sand_count: Optional[int] = None # 每 sand_unit_chars 个可切图字符投放几个沙子
    sand_unit_chars: int = 1000
    min_total: int = 8
    max_total: int = 35
    max_per_page: int = 2
    max_per_line: int = 1
    seed: Optional[int] = None       # None = 随机；测试可固定

    def target_density(self) -> float:
        if self.sand_count is None:
            return max(0.0, float(self.target_ratio))
        return max(0.0, float(self.sand_count)) / max(1, int(self.sand_unit_chars))


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
    """返回既可投放假象字、又有现成字符切图支撑的位置。"""
    return [idx for idx in candidate_indices_in_line(line) if _has_existing_cut_char(line, idx)]


def count_existing_cjk_crop_chars(line: Line) -> int:
    """统计 line 中已有切图且与文本对齐的 CJK 字符数。"""
    return sum(
        1
        for idx, ch in enumerate(line.text or "")
        if _is_cjk(ch) and _has_existing_cut_char(line, idx)
    )


class ProbeSampler:
    def __init__(self, config: Optional[SamplerConfig] = None) -> None:
        self.cfg = config or SamplerConfig()

    def sample(self, project: OcrProject) -> ProbeStore:
        rng = random.Random(self.cfg.seed)
        store = ProbeStore()

        # 1. 收集所有 (page_number, block_index, line_index, candidate_char_indices)
        per_line_pool: list[tuple[int, int, int, Line, list[int]]] = []
        total_cut_cjk = 0
        for page in project.pages:
            for bi, block in enumerate(page.blocks):
                if not is_block_eligible(block):
                    continue
                for li, line in enumerate(block.lines):
                    total_cut_cjk += count_existing_cjk_crop_chars(line)
                    cands = candidate_indices_with_existing_crops(line)
                    if cands:
                        per_line_pool.append((page.page_number, bi, li, line, cands))

        store.sampled_from_chars = total_cut_cjk
        store.sand_count = self.cfg.sand_count
        store.sand_unit_chars = self.cfg.sand_unit_chars
        if not per_line_pool:
            return store

        # 2. 计算目标投放数。新密度模式按全局字符池直接生效，不套旧 min_total。
        if self.cfg.sand_count is None:
            target = int(round(total_cut_cjk * self.cfg.target_density()))
            target = max(self.cfg.min_total, target)
        else:
            target = math.ceil(total_cut_cjk * self.cfg.target_density()) if self.cfg.sand_count > 0 else 0
        target = min(self.cfg.max_total, target)
        store.target_probes = target
        if target <= 0:
            return store

        # 3. 按 (page) 分桶，每页不超过 max_per_page；每行不超过 max_per_line
        rng.shuffle(per_line_pool)
        per_page_count: dict[int, int] = {}
        placed = 0
        for page_no, bi, li, line, cands in per_line_pool:
            if placed >= target:
                break
            if self._page_limit_reached(per_page_count, page_no):
                continue
            # 每行投 1 个；从行内候选随机选一个 char_index
            char_idx = rng.choice(cands)
            true_ch = (line.text or "")[char_idx]
            fake_options = CONFUSION_MAP.get(true_ch, ())
            if not fake_options:
                continue
            fake_ch = rng.choice(fake_options)
            probe = Probe(
                key=ProbeKey(page_no, bi, li, char_idx),
                true_char=true_ch,
                fake_char=fake_ch,
            )
            store.add(probe)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            placed += 1

        # 4. 旧比例模式若不足 min_total 但池子还有空间，做一轮补足。
        #    重要：此轮**仍然遵守 max_per_page / max_per_line**，绝不绕过页/行上限。
        #    （早期实现允许在这里突破 max_per_page，会导致小项目所有 probe 砸到同一页，
        #    严重影响校对体验且违反产品口径。）
        if self.cfg.sand_count is None and placed < self.cfg.min_total:
            for page_no, bi, li, line, cands in per_line_pool:
                if placed >= self.cfg.min_total:
                    break
                if self._page_limit_reached(per_page_count, page_no):
                    continue
                if store.for_line(page_no, bi, li):
                    continue  # 该行已投，遵守 max_per_line
                char_idx = rng.choice(cands)
                true_ch = (line.text or "")[char_idx]
                fake_options = CONFUSION_MAP.get(true_ch, ())
                if not fake_options:
                    continue
                fake_ch = rng.choice(fake_options)
                store.add(Probe(
                    key=ProbeKey(page_no, bi, li, char_idx),
                    true_char=true_ch,
                    fake_char=fake_ch,
                ))
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                placed += 1

        return store

    def _page_limit_reached(self, per_page_count: dict[int, int], page_no: int) -> bool:
        if self.cfg.sand_count is not None:
            return False
        return per_page_count.get(page_no, 0) >= self.cfg.max_per_page


# ──────────────────────────────────────────────────────────────────
# 显示空间 ↔ 真实空间映射
# ──────────────────────────────────────────────────────────────────

def apply_probes_to_display(text: str, probes: list[Probe]) -> str:
    """把真实文本 ``text`` 按 ``probes`` 替换得到展示给用户的文本。

    ``probes`` 列表必须按 ``char_index`` 升序（``ProbeStore`` 已保证）。
    若某 probe 的 ``true_char`` 与 ``text[char_index]`` 不一致（说明真实文本
    已被用户改过），跳过该 probe，返回原字符。
    """
    if not probes:
        return text
    chars = list(text)
    for p in probes:
        idx = p.key.char_index
        if 0 <= idx < len(chars) and chars[idx] == p.true_char:
            chars[idx] = p.fake_char
    return "".join(chars)


def reverse_display_to_true(
    line_text_before: str,
    displayed_new_text: str,
    probes: list[Probe],
) -> tuple[str, list[tuple[Probe, str]]]:
    """把"用户编辑后的显示空间文本"还原回"真实空间文本"，并给出每个 probe 的判定。

    判定原则：
    - 显示文本和真实文本的对齐做不到完美（用户可能在 probe 之前/后插入或删字）。
      因此采用**长度相同时按位置直接比对**的最简单策略；如果长度变化，
      只要长度差异不大（≤ 2 字符），仍按头部对齐做比对；否则放弃判定，
      所有 probe 标 ``edited_other``，并把整段文本作为真实文本写回（
      用户已大幅改动）。

    返回：(true_text_to_save, [(probe, observation)...])

    重要不变量：
    - 返回的 ``true_text_to_save`` **不含任何 fake_char**：
      若用户保留了某 probe 的 fake_char，会被还原成 true_char 写回。
    - 若用户在 probe 位置写了既非 true 也非 fake 的字符，原样保留
      （视为用户的真实修改），probe 标 ``edited_other``。
    """
    if not probes:
        return displayed_new_text, []

    n_before = len(line_text_before)
    n_now = len(displayed_new_text)

    # 大幅改动 → 放弃逐位对齐，整段视为用户修改
    if abs(n_now - n_before) > 2:
        observations = [(p, "edited_other") for p in probes]
        # 严格清洗：必须确保返回的真实文本里**没有任何已投放过的 fake_char**。
        # 旧实现 "if true_char not in cleaned" 的守卫在 true_char 偶然出现在用户改写
        # 文本里时会跳过替换，导致 fake_char 残留 → 写回 line.text → 污染最终导出。
        # 现在对每个 probe 都至少替换一次（fake → true），即使 true_char 已存在；
        # 然后再做一遍兜底扫描，确保没有任何 probe 的 fake_char 还留在结果里。
        cleaned = displayed_new_text
        for p in probes:
            if p.fake_char and p.fake_char in cleaned:
                cleaned = cleaned.replace(p.fake_char, p.true_char, 1)
        # 兜底：可能某 probe 的 fake_char 仍残留（例如它在文本里出现了多次），
        # 把它们全部替换成 true_char。这是写回真实文本前的最后一道防线。
        for p in probes:
            if p.fake_char and p.fake_char in cleaned:
                cleaned = cleaned.replace(p.fake_char, p.true_char)
        return cleaned, observations

    chars = list(displayed_new_text)
    observations: list[tuple[Probe, str]] = []
    for p in probes:
        idx = p.key.char_index
        if idx >= len(chars):
            observations.append((p, "deleted"))
            continue
        ch = chars[idx]
        if ch == p.fake_char:
            # 用户没察觉假象 —— 写回真实文本时还原
            chars[idx] = p.true_char
            observations.append((p, "missed"))
        elif ch == p.true_char:
            # 用户把假象改回了正字
            observations.append((p, "corrected"))
        else:
            # 用户改成了别的字 —— 保留用户修改，不强制还原
            observations.append((p, "edited_other"))

    # ──────────────────────────────────────────────────────────
    # 关键安全网 (Blocker A 修复)：small-change 路径下，用户在 probe 之前/后
    # 插入或删除 1-2 个字符，会让真正含 fake_char 的位置漂移到 char_index ± δ。
    # 上面的 ``chars[idx]`` 比对只看精确位置，错过了漂移后的 fake_char。
    # 在这里做一次"全文 scrub"：对每个 probe，检查最终 chars 里是否仍残留它的
    # ``fake_char``；若是，就把所有出现替换成 ``true_char``。
    #
    # 这是写回 ``line.text`` 前的最后一道防线 —— 即使前面所有判定都失误，
    # 这一步也能保证返回的真实文本绝不含 probe 的 fake_char。
    #
    # 副作用 trade-off：若用户**有意**输入了与某 probe 的 fake_char 相同的
    # 字符（例如输入"己"，而某 probe 的 fake_char 也是"己"），这里会被改成
    # ``true_char``。这是已知且可接受的代价 —— 与污染最终导出文本相比，
    # 偶尔被替换一个字符是远更轻的影响，并且用户可以重新输入。
    # ──────────────────────────────────────────────────────────
    cleaned_str = "".join(chars)
    for p in probes:
        if p.fake_char and p.fake_char != p.true_char and p.fake_char in cleaned_str:
            cleaned_str = cleaned_str.replace(p.fake_char, p.true_char)
    return cleaned_str, observations


def observe_user_action(
    store: ProbeStore,
    page_number: int,
    block_index: int,
    line_index: int,
    line_text_before: str,
    displayed_new_text: str,
) -> str:
    """便捷入口：在 UI 的"保存"路径上调用一次。

    会就地更新 store 中相关 probe 的 observation，并返回应该写回 line.text
    的真实文本。
    """
    probes = store.for_line(page_number, block_index, line_index)
    if not probes:
        return displayed_new_text
    true_text, results = reverse_display_to_true(line_text_before, displayed_new_text, probes)
    for probe, obs in results:
        # 一次"已观察"判定不应被后续重复保存覆写为更弱的状态。
        # 优先级（高 → 低）：corrected > edited_other > missed > deleted > pending
        priority = {"corrected": 4, "edited_other": 3, "missed": 2, "deleted": 1, "pending": 0}
        if priority.get(obs, 0) >= priority.get(probe.observation, 0):
            probe.observation = obs
    return true_text


# ──────────────────────────────────────────────────────────────────
# 评分 —— 输出抽样字符等级 / 区间，不输出伪精确的全量错误率
# ──────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    total_probes: int
    corrected: int
    missed: int
    edited_other: int
    deleted: int
    pending: int
    grade: str           # "A" / "B" / "C" / "D" / "INSUFFICIENT"
    grade_label: str     # "优秀 / 良好 / 一般 / 待加强 / 样本不足"
    band: str            # 如 "抽样字符假象识破 ≥ 85%"，不是全量正确率
    raw_score: float     # 内部用，不作为 UI 全量百分比暴露

    def to_dict(self) -> dict:
        return asdict(self)


def score(store: ProbeStore, *, min_observed_for_grade: int = 4) -> QualityReport:
    total = len(store)
    corrected = missed = edited_other = deleted = pending = 0
    for p in store.all():
        if p.observation == "corrected":
            corrected += 1
        elif p.observation == "missed":
            missed += 1
        elif p.observation == "edited_other":
            edited_other += 1
        elif p.observation == "deleted":
            deleted += 1
        else:
            pending += 1

    judged = corrected + missed  # 只有这两类是干净的"是否识破"判定
    if judged < min_observed_for_grade:
        return QualityReport(
            total_probes=total,
            corrected=corrected,
            missed=missed,
            edited_other=edited_other,
            deleted=deleted,
            pending=pending,
            grade="INSUFFICIENT",
            grade_label="样本不足",
            band="抽样字符观察不足，暂不给出等级",
            raw_score=0.0,
        )

    raw = corrected / max(1, judged)
    if raw >= 0.85:
        grade, label, band = "A", "优秀", "抽样字符假象识破 ≥ 85%"
    elif raw >= 0.65:
        grade, label, band = "B", "良好", "抽样字符假象识破 65%–85%"
    elif raw >= 0.40:
        grade, label, band = "C", "一般", "抽样字符假象识破 40%–65%"
    else:
        grade, label, band = "D", "待加强", "抽样字符假象识破 < 40%"

    return QualityReport(
        total_probes=total,
        corrected=corrected,
        missed=missed,
        edited_other=edited_other,
        deleted=deleted,
        pending=pending,
        grade=grade,
        grade_label=label,
        band=band,
        raw_score=raw,
    )


# ──────────────────────────────────────────────────────────────────
# 配置回灌 —— SamplerConfig 可从 AppConfig 读取，避免硬编码魔法数
# ──────────────────────────────────────────────────────────────────

# AppConfig key -> (SamplerConfig field, type-cast)
_SAMPLER_CONFIG_KEYS: dict[str, tuple[str, type]] = {
    "quality_probe_sand_count": ("sand_count", int),
    "quality_probe_sand_unit_chars": ("sand_unit_chars", int),
    "quality_probe_target_ratio": ("target_ratio", float),
    "quality_probe_min_total": ("min_total", int),
    "quality_probe_max_total": ("max_total", int),
    "quality_probe_max_per_page": ("max_per_page", int),
    "quality_probe_max_per_line": ("max_per_line", int),
}


def sampler_config_from_app_config() -> SamplerConfig:
    """从 AppConfig 读取阈值，缺失键回落到 SamplerConfig 默认值。

    Qt 不可用（CLI/测试环境）时静默回落。
    """
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
# 混淆字表扩展 —— 支持 JSON sidecar 与运行时注册
# ──────────────────────────────────────────────────────────────────

# 内部记录："扩展组"。集中保存允许后续重建 CONFUSION_MAP。
_EXTRA_CONFUSABLES: list[tuple[str, ...]] = []


def _rebuild_confusion_map() -> None:
    """根据 ``_CONFUSABLES_RAW`` + ``_EXTRA_CONFUSABLES`` 重建 CONFUSION_MAP。"""
    global CONFUSION_MAP
    merged = list(_CONFUSABLES_RAW) + [tuple(g) for g in _EXTRA_CONFUSABLES]
    CONFUSION_MAP = _build_confusion_map(merged)


def register_confusion_group(group: Iterable[str]) -> None:
    """运行时追加一组形近字（去重后并入 CONFUSION_MAP）。

    供 UI 设置面板 / 测试 / 项目热扩展使用。重复字符自动去重。
    """
    chars = tuple(dict.fromkeys(c for c in group if c))
    if len(chars) < 2:
        return
    _EXTRA_CONFUSABLES.append(chars)
    _rebuild_confusion_map()


def _load_confusables_extra_file() -> None:
    """启动时从 ``app/core/confusables_extra.json`` 加载用户/项目维护的扩展组。

    JSON schema：``[["己","已","巳"], ["末","未"], ...]``。
    缺文件、解析失败、格式不对都静默跳过 —— 这是软扩展点，绝不能让模块导入崩溃。
    """
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
# 持久化 —— ProbeStore 序列化为 JSON sidecar
# ──────────────────────────────────────────────────────────────────

_SIDECAR_SUFFIX = ".qprobe.json"
_SIDECAR_VERSION = 1


def sidecar_path_for_project(db_path: Optional[str]) -> Optional[str]:
    """根据项目 ``.ocrproj`` 路径推导 probe sidecar 路径。

    None / 空字符串 / 临时项目（无 db_path） → 返回 None，调用方应跳过持久化。
    """
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
    """写入 sidecar JSON。失败返回 False，不抛异常（持久化是辅助路径）。"""
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
    """读取 sidecar JSON；不存在/损坏均返回 None。"""
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
