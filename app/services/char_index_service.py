"""字符索引服务：为纵校提供频次统计、稳定排序与全书同字检索。

职责：
    1. 遍历全书，建立 char -> [CharEntry] 索引
    2. 自动检测纵排/横排，给出每字 bbox 估算
    3. 去重：同 (line, char_idx) 不会被重复加入
    4. 提供稳定排序键（拼音 → 数字 → 标点 → 符号 → 其他）
    5. query 返回时按 (page_number, line.bbox.y, line.bbox.x, char_idx) 稳定排列

用法：
    svc = CharIndexService()
    svc.build(pages)
    chars_sorted = svc.sorted_chars()           # [(char, count), ...]
    entries = svc.query("永")                    # [CharEntry, ...]
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models import BBox, Line, Page

# 可选 pypinyin —— 没有则降级到 unicode codepoint 排序
try:
    from pypinyin import lazy_pinyin, Style  # type: ignore
    _HAS_PYPINYIN = True
except ImportError:  # pragma: no cover
    _HAS_PYPINYIN = False


# ─────────────────────────────────────────────────────────────
# 字符分类（用于稳定排序）
# ─────────────────────────────────────────────────────────────

# 字符类型权重：越小越靠前
_KIND_LETTER = 0   # 拉丁字母 / 中文（按拼音）
_KIND_DIGIT  = 1   # 数字
_KIND_PUNCT  = 2   # 标点
_KIND_SYMBOL = 3   # 符号（公式符号等）
_KIND_OTHER  = 4   # 其它


def _char_kind(c: str) -> int:
    """按 unicode 类别返回字符种类权重。"""
    if not c:
        return _KIND_OTHER
    cat = unicodedata.category(c)
    # L* 字母（含中文 Lo）
    if cat.startswith("L"):
        return _KIND_LETTER
    # N* 数字
    if cat.startswith("N"):
        return _KIND_DIGIT
    # P* 标点
    if cat.startswith("P"):
        return _KIND_PUNCT
    # S* 符号（数学 / 货币 / 修饰）
    if cat.startswith("S"):
        return _KIND_SYMBOL
    return _KIND_OTHER


def _sort_key(c: str) -> Tuple[int, str, str]:
    """稳定排序键：(种类权重, 拼音/字母, 原字符)。"""
    kind = _char_kind(c)
    if kind == _KIND_LETTER:
        if _HAS_PYPINYIN and ord(c) > 0x2E80:  # CJK 起点附近
            py = lazy_pinyin(c, style=Style.NORMAL)
            label = py[0] if py else c.lower()
        else:
            label = c.lower()
    else:
        label = c
    return (kind, label, c)


# ─────────────────────────────────────────────────────────────
# bbox 估算（纵排 / 横排自动判别）
# ─────────────────────────────────────────────────────────────

def _is_vertical_line(bbox: BBox) -> bool:
    """高度显著大于宽度 → 纵排。"""
    if bbox.w <= 0:
        return True
    return bbox.h >= bbox.w * 1.5


def _estimate_char_bbox(line: Line, idx: int, total: int) -> Optional[BBox]:
    """根据行 bbox 等分估算字符 bbox。

    自动判别纵排/横排。仅在 line.chars[*].bbox 为 None 时使用。
    """
    if total <= 0:
        return None
    bb = line.bbox
    if _is_vertical_line(bb):
        ch = max(bb.h / total, 1)
        return BBox(bb.x, int(bb.y + idx * ch), bb.w, int(ch))
    cw = max(bb.w / total, 1)
    return BBox(int(bb.x + idx * cw), bb.y, int(cw), bb.h)


# ─────────────────────────────────────────────────────────────
# CharEntry
# ─────────────────────────────────────────────────────────────

@dataclass
class CharEntry:
    """一个字符在全书中的一次出现记录。"""
    char: str
    page_path: str
    page_number: int
    line: Line
    char_idx: int
    bbox: BBox

    @property
    def _sort_key(self) -> Tuple[int, int, int, int]:
        """稳定排序：(页码, 行 y, 行 x, 字号)。"""
        return (
            self.page_number,
            self.line.bbox.y,
            self.line.bbox.x,
            self.char_idx,
        )


# ─────────────────────────────────────────────────────────────
# CharIndexService
# ─────────────────────────────────────────────────────────────

class CharIndexService:
    """全书字符索引。每次 build() 全量重建。"""

    def __init__(self) -> None:
        self._index: Dict[str, List[CharEntry]] = {}
        self._freq: Dict[str, int] = {}

    # ── 构建 ───────────────────────────────────────────────────

    def build(self, pages: List[Page]) -> None:
        self._index = {}
        self._freq = {}

        # 收集 → 去重 → 排序，三段式
        seen: set[Tuple[int, int]] = set()  # (id(line), char_idx)

        for page in pages:
            path = page.display_image_path
            num = page.page_number
            for block in page.text_blocks:
                for line in block.lines:
                    self._index_line(line, path, num, seen)

        # 每个字的 entries 按 (页, 行 y, 行 x, 字号) 排序
        for entries in self._index.values():
            entries.sort(key=lambda e: e._sort_key)

    def _index_line(
        self,
        line: Line,
        page_path: str,
        page_number: int,
        seen: set,
    ) -> None:
        if line.chars:
            for ci, ch_obj in enumerate(line.chars):
                self._maybe_add(
                    ch_obj.char, line, ci, page_path, page_number,
                    explicit_bbox=ch_obj.bbox, seen=seen,
                )
        else:
            text = line.text or ""
            total = len(text)
            for ci, c in enumerate(text):
                self._maybe_add(
                    c, line, ci, page_path, page_number,
                    explicit_bbox=None, seen=seen,
                    fallback_total=total,
                )

    def _maybe_add(
        self,
        c: str,
        line: Line,
        ci: int,
        page_path: str,
        page_number: int,
        *,
        explicit_bbox: Optional[BBox],
        seen: set,
        fallback_total: Optional[int] = None,
    ) -> None:
        if not c or c.isspace():
            return
        key = (id(line), ci)
        if key in seen:
            return
        bbox = explicit_bbox
        if bbox is None:
            total = fallback_total if fallback_total is not None else (
                len(line.chars) if line.chars else len(line.text or "")
            )
            bbox = _estimate_char_bbox(line, ci, total)
        if bbox is None:
            return
        seen.add(key)
        entry = CharEntry(
            char=c, page_path=page_path, page_number=page_number,
            line=line, char_idx=ci, bbox=bbox,
        )
        self._index.setdefault(c, []).append(entry)
        self._freq[c] = self._freq.get(c, 0) + 1

    # ── 查询 ───────────────────────────────────────────────────

    def char_frequency(self) -> List[Tuple[str, int]]:
        """按频次降序，频次相同时按 _sort_key 升序。"""
        return sorted(
            self._freq.items(),
            key=lambda x: (-x[1], _sort_key(x[0])),
        )

    def sorted_chars(self) -> List[Tuple[str, int]]:
        """按字符种类 → 拼音 → 原字符 升序排列。

        顺序：拉丁字母/中文（按拼音 a-z）→ 数字 → 标点 → 符号 → 其他
        """
        return sorted(
            self._freq.items(),
            key=lambda x: _sort_key(x[0]),
        )

    def query(self, char: str) -> List[CharEntry]:
        return self._index.get(char, [])

    def unique_chars(self) -> int:
        return len(self._index)

    def total_chars(self) -> int:
        return sum(self._freq.values())

    def first_entry(self, char: str) -> Optional[CharEntry]:
        entries = self._index.get(char)
        return entries[0] if entries else None
