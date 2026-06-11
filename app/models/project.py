from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

from .enums import (
    BlockSource, BlockType, LlmReviewStatus, PageStatus, ProofStatus,
)


@dataclass
class BBox:
    """像素坐标包围盒（左上角原点）。"""
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BBox":
        return cls(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def clamp(self, max_w: int, max_h: int) -> "BBox":
        """裁剪到图像边界内。"""
        x1 = max(0, min(self.x, max_w))
        y1 = max(0, min(self.y, max_h))
        x2 = max(x1, min(self.x + self.w, max_w))
        y2 = max(y1, min(self.y + self.h, max_h))
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def normalize(self) -> "BBox":
        """确保宽高为正数（处理右下到左上拖拽）。"""
        x = min(self.x, self.x + self.w)
        y = min(self.y, self.y + self.h)
        return BBox(x, y, abs(self.w), abs(self.h))

    def expand(self, pad: int) -> "BBox":
        """等量扩边。"""
        return BBox(self.x - pad, self.y - pad, self.w + 2 * pad, self.h + 2 * pad)

    def translated(self, dx: int, dy: int) -> "BBox":
        """平移坐标。"""
        return BBox(self.x + dx, self.y + dy, self.w, self.h)

    def iou(self, other: "BBox") -> float:
        """计算与另一个 BBox 的交并比。"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(x=int(d["x"]), y=int(d["y"]), w=int(d["w"]), h=int(d["h"]))


@dataclass
class Char:
    """单个字符识别结果。"""
    char: str
    confidence: float
    bbox: Optional[BBox] = None
    id: Optional[int] = None  # 数据库 rowid
    bbox_source: str = ""
    bbox_granularity: str = ""
    token_text: str = ""


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

    # --- Phase 1 新增字段 ---
    final_text: str = ""                  # 人工最终文本；text 仅作兼容镜像
    ocr_text: str = ""                    # OCR 原始文本（与 original_text 互补）
    llm_suggestion: str = ""              # LLM 预审建议文本
    llm_reason: str = ""                  # LLM 修改原因
    llm_review_status: LlmReviewStatus = LlmReviewStatus.DISABLED
    review_flags: List[str] = field(default_factory=list)  # 疑点标签

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == "text":
            object.__setattr__(self, "final_text", value)
        elif name == "final_text" and (value or getattr(self, "_line_initialized", False)):
            object.__setattr__(self, "text", value)

    def __post_init__(self) -> None:
        if not self.final_text:
            self.final_text = self.text
        elif not self.text:
            self.text = self.final_text
        object.__setattr__(self, "_line_initialized", True)

    def update_text(self, new_text: str) -> None:
        if self.original_text == "":
            self.original_text = self.final_text
        self.final_text = new_text
        self.text = new_text
        self.proof_status = ProofStatus.MODIFIED

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "final_text": self.final_text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "proof_status": self.proof_status.value,
            "original_text": self.original_text,
            "ocr_text": self.ocr_text,
            "llm_suggestion": self.llm_suggestion,
            "llm_reason": self.llm_reason,
            "llm_review_status": self.llm_review_status.value,
        }


@dataclass
class Block:
    """版面分析得到的一个内容块。"""
    block_type: BlockType
    bbox: BBox
    lines: List[Line] = field(default_factory=list)
    order: int = 0             # 阅读顺序（从版面分析得到）
    id: Optional[int] = None

    # --- Phase 1 新增字段 ---
    source: BlockSource = BlockSource.AUTO_LAYOUT  # 块来源
    is_locked: bool = False                         # 锁定后自动分析不覆盖
    recognizable: bool = True                       # 是否送 OCR
    note: str = ""                                  # 用户备注或系统说明
    source_label: str = ""                          # 原始 PP-VL/Paddle label
    raw_payload: dict[str, Any] = field(default_factory=dict)  # 原始块属性

    @property
    def full_text(self) -> str:
        return "\n".join(line.final_text for line in self.lines)

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

    # --- Phase 1 新增字段 ---
    source_path: str = ""                   # 原始导入文件路径
    source_type: str = "image"              # "image" | "pdf"
    source_page_index: int = 1              # PDF 页码（1-based）
    cache_image_path: str = ""              # 处理后的实际工作图片路径
    thumbnail_path: str = ""                # 缩略图路径
    status: PageStatus = PageStatus.IMPORTED
    error_message: str = ""                 # 当前页失败原因
    ocr_invalidated_reason: str = ""        # 版面变更导致 OCR 结果失效的原因
    ppvl_parsing_res_list: List[dict[str, Any]] = field(default_factory=list)

    @property
    def is_analyzed(self) -> bool:
        return len(self.blocks) > 0

    @property
    def display_image_path(self) -> str:
        """当前用于渲染和裁剪的工作图路径。

        OCR、版面分析和画布渲染都必须基于同一张图，否则坐标会漂移。
        """
        return self.cache_image_path or self.image_path

    @property
    def text_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.block_type in (
            BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE
        )]

    @property
    def recognizable_blocks(self) -> List[Block]:
        """返回可送 OCR 的块。"""
        return [b for b in self.blocks if b.recognizable]

    @property
    def total_lines(self) -> int:
        return sum(len(b.lines) for b in self.blocks)

    @property
    def has_ocr_result(self) -> bool:
        """当前页是否已有可展示/校对的 OCR 行。"""
        return self.total_lines > 0

    @property
    def is_ocr_done(self) -> bool:
        """当前页是否处于 OCR 完成状态。"""
        return self.status == PageStatus.OCR_DONE

    @property
    def needs_ocr_rerun(self) -> bool:
        """版面变更后，已有 OCR 结果是否被显式标记为失效。"""
        return bool(self.ocr_invalidated_reason)

    def invalidate_ocr(self, reason: str) -> None:
        self.ocr_invalidated_reason = str(reason or "layout_changed")

    def clear_ocr_invalidation(self) -> None:
        self.ocr_invalidated_reason = ""

    @property
    def proofed_lines(self) -> int:
        """已校对行数。"""
        return sum(
            1 for b in self.blocks for l in b.lines
            if l.proof_status in (ProofStatus.OK, ProofStatus.MODIFIED)
        )

    @property
    def flagged_lines(self) -> int:
        """低置信或疑点标记行数。"""
        return sum(
            1 for b in self.blocks for l in b.lines
            if l.proof_status == ProofStatus.AUTO_FLAGGED
            or (l.review_flags and l.proof_status != ProofStatus.OK)
        )


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
        return sum(p.total_lines for p in self.pages)

    @property
    def total_flagged_lines(self) -> int:
        return sum(p.flagged_lines for p in self.pages)

    @property
    def total_unproofed_lines(self) -> int:
        return self.total_lines - sum(p.proofed_lines for p in self.pages)

    @property
    def ocr_completed(self) -> bool:
        """兼容旧调用：项目是否已有任意 OCR 结果。

        不再把它作为“项目所有页面 OCR 完成”的权威语义；新代码应使用
        has_any_ocr_result / all_pages_ocr_done。
        """
        return self.has_any_ocr_result

    @property
    def has_any_ocr_result(self) -> bool:
        return any(p.has_ocr_result for p in self.pages)

    @property
    def all_pages_ocr_done(self) -> bool:
        return bool(self.pages) and all(p.is_ocr_done for p in self.pages)

    @property
    def has_pending_ocr_pages(self) -> bool:
        return any(
            p.is_analyzed and (not p.is_ocr_done or p.needs_ocr_rerun)
            for p in self.pages
        )

    @property
    def has_unrecognized_blocks(self) -> bool:
        return any(
            b.recognizable and not b.lines
            for p in self.pages for b in p.blocks
        )

    def get_export_summary(self) -> dict:
        """导出前状态摘要。"""
        return {
            "total_pages": self.page_count,
            "total_lines": self.total_lines,
            "unproofed_lines": self.total_unproofed_lines,
            "flagged_lines": self.total_flagged_lines,
            "unrecognized_blocks": sum(
                1 for p in self.pages for b in p.blocks
                if b.recognizable and not b.lines
            ),
        }
