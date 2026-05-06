from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time

from .enums import BlockType, ProofStatus


@dataclass
class BBox:
    """像素坐标包围盒（左上角原点）。"""
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BBox":
        return cls(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class Char:
    """单个字符识别结果。"""
    char: str
    confidence: float
    bbox: Optional[BBox] = None
    id: Optional[int] = None  # 数据库 rowid


@dataclass
class Line:
    """一行文字识别结果。"""
    text: str
    confidence: float          # 行平均置信度
    bbox: BBox
    chars: List[Char] = field(default_factory=list)
    proof_status: ProofStatus = ProofStatus.UNCHECKED
    original_text: str = ""    # 修改前的原始文字（保留用于对比）
    id: Optional[int] = None

    def update_text(self, new_text: str) -> None:
        if self.original_text == "":
            self.original_text = self.text
        self.text = new_text
        self.proof_status = ProofStatus.MODIFIED


@dataclass
class Block:
    """版面分析得到的一个内容块。"""
    block_type: BlockType
    bbox: BBox
    lines: List[Line] = field(default_factory=list)
    order: int = 0             # 阅读顺序（从版面分析得到）
    id: Optional[int] = None

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def avg_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.confidence for l in self.lines) / len(self.lines)


@dataclass
class Page:
    """一页（对应一张图片）。"""
    image_path: str            # 原始图片绝对路径
    width: int
    height: int
    blocks: List[Block] = field(default_factory=list)
    page_number: int = 1
    id: Optional[int] = None

    @property
    def is_analyzed(self) -> bool:
        return len(self.blocks) > 0

    @property
    def text_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.block_type in (
            BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE
        )]


@dataclass
class OcrProject:
    """一个 OCR 处理项目。"""
    name: str
    pages: List[Page] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    id: Optional[int] = None
    db_path: Optional[str] = None   # 项目文件 .ocrproj 路径

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_lines(self) -> int:
        return sum(
            len(b.lines) for p in self.pages for b in p.blocks
        )

    @property
    def flagged_lines(self) -> int:
        return sum(
            1 for p in self.pages for b in p.blocks
            for l in b.lines if l.proof_status == ProofStatus.AUTO_FLAGGED
        )
