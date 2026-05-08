"""字符索引服务：为纵校提供频次统计和全书同字检索。

用法：
    from app.services.char_index_service import CharIndexService, CharEntry

    svc = CharIndexService()
    svc.build(pages)

    freqs = svc.char_frequency()          # [(char, count), ...]  降序
    entries = svc.query("也")             # [CharEntry, ...]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models import BBox, Block, Line, Page


@dataclass
class CharEntry:
    """一个字符的出现记录。"""
    char: str
    page_path: str          # 对应 Page.display_image_path
    page_number: int
    line: Line
    char_idx: int           # 在 line.chars 中的下标（若无 chars 则为 text 下标）
    bbox: BBox              # 字符在整页坐标系中的 bbox


def _estimate_char_bbox(line: Line, idx: int, total: int) -> Optional[BBox]:
    """按行宽等分估算字符 bbox（无逐字坐标时回退）。"""
    if total <= 0:
        return None
    bb = line.bbox
    cw = max(bb.w / total, 1)
    return BBox(int(bb.x + idx * cw), bb.y, int(cw), bb.h)


class CharIndexService:
    """全书字符索引。

    调用 build() 重建，支持多次调用（每次全量重建）。
    """

    def __init__(self) -> None:
        # char -> 所有出现记录
        self._index: Dict[str, List[CharEntry]] = {}
        # char -> 出现次数
        self._freq: Dict[str, int] = {}

    # ------------------------------------------------------------------ 构建

    def build(self, pages: List[Page]) -> None:
        """遍历全部页面、块、行，建立字符索引。"""
        self._index = {}
        self._freq = {}

        for page in pages:
            path = page.display_image_path
            num = page.page_number
            for block in page.text_blocks:
                for line in block.lines:
                    self._index_line(line, path, num)

    def _index_line(self, line: Line, page_path: str, page_number: int) -> None:
        if line.chars:
            # 有逐字坐标
            for ci, ch_obj in enumerate(line.chars):
                c = ch_obj.char
                if not c or c.isspace():
                    continue
                bbox = ch_obj.bbox
                if bbox is None:
                    bbox = _estimate_char_bbox(line, ci, len(line.chars))
                if bbox is None:
                    continue
                self._add(c, page_path, page_number, line, ci, bbox)
        else:
            # 无逐字坐标，按行宽等分
            text = line.text
            total = len(text)
            for ci, c in enumerate(text):
                if c.isspace():
                    continue
                bbox = _estimate_char_bbox(line, ci, total)
                if bbox is None:
                    continue
                self._add(c, page_path, page_number, line, ci, bbox)

    def _add(
        self,
        char: str,
        page_path: str,
        page_number: int,
        line: Line,
        char_idx: int,
        bbox: BBox,
    ) -> None:
        entry = CharEntry(
            char=char,
            page_path=page_path,
            page_number=page_number,
            line=line,
            char_idx=char_idx,
            bbox=bbox,
        )
        self._index.setdefault(char, []).append(entry)
        self._freq[char] = self._freq.get(char, 0) + 1

    # ------------------------------------------------------------------ 查询

    def char_frequency(self) -> List[Tuple[str, int]]:
        """返回 [(char, count), ...] 按频次降序排列。"""
        return sorted(self._freq.items(), key=lambda x: -x[1])

    def query(self, char: str) -> List[CharEntry]:
        """返回该字在全书的所有出现记录，按页码、行序排列。"""
        return self._index.get(char, [])

    def unique_chars(self) -> int:
        """独立字符总数。"""
        return len(self._index)

    def total_chars(self) -> int:
        """总字符数（含重复）。"""
        return sum(self._freq.values())

    def first_entry(self, char: str) -> Optional[CharEntry]:
        """返回该字的首次出现记录（用于缩略图预览）。"""
        entries = self._index.get(char)
        return entries[0] if entries else None
