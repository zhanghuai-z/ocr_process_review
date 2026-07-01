from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

from .enums import (
    BlockSource, BlockType, PageStatus, ProofStatus,
)
from .entity_id import ensure_entity_uid
from .proof_line_state import ProofLineState


OCR_AVAILABLE_PAGE_STATUSES = {
    PageStatus.OCR_DONE,
    PageStatus.PROOFING,
    PageStatus.PROOF_DONE,
}


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
    uid: str = ""             # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "char")


@dataclass
class Line:
    """一行文字识别结果。"""
    text: str
    confidence: float          # 行平均置信度
    bbox: BBox
    chars: List[Char] = field(default_factory=list)
    original_text: str = ""    # 修改前的原始文字（保留用于对比）
    id: Optional[int] = None
    ocr_text: str = ""                    # OCR 原始文本（与 original_text 互补）
    review_flags: List[str] = field(default_factory=list)  # 疑点标签
    uid: str = ""                         # 稳定业务 ID
    proof_state: ProofLineState | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "line")
        if not self.ocr_text:
            self.ocr_text = self.text
        if not self.original_text:
            self.original_text = self.ocr_text or self.text
        state = self.proof_state or ProofLineState(line_uid=self.uid)
        self.apply_proof_state(state)

    def set_proof_text(
        self,
        new_text: str,
        *,
        status: ProofStatus = ProofStatus.MODIFIED,
    ) -> None:
        if self.original_text == "":
            self.original_text = self.ocr_text or self.text
        state = getattr(self, "proof_state", None) or ProofLineState(line_uid=self.uid)
        state.final_text = new_text
        state.final_text_set = True
        state.proof_status = status
        self.apply_proof_state(state)

    def set_proof_status(self, status: ProofStatus) -> None:
        state = getattr(self, "proof_state", None) or ProofLineState(line_uid=self.uid)
        state.proof_status = status
        self.apply_proof_state(state)

    def apply_proof_state(self, state: ProofLineState) -> None:
        state.line_uid = self.uid
        object.__setattr__(self, "proof_state", state)

    def ensure_text_contract(self, *, fill_original: bool = False) -> None:
        """Normalize OCR text fields and ensure proof state exists."""
        state = getattr(self, "proof_state", None)
        if state is None:
            self.apply_proof_state(ProofLineState(line_uid=self.uid))
        else:
            state.line_uid = self.uid
        ocr_text = self.ocr_text or self.text
        if not self.text:
            self.text = ocr_text
        if fill_original and not self.original_text:
            self.original_text = ocr_text or self.text
        self.ocr_text = ocr_text

    def to_dict(self) -> dict:
        state = self.proof_state or ProofLineState(line_uid=self.uid)
        return {
            "text": self.text,
            "uid": self.uid,
            "proof_text": state.final_text,
            "proof_text_set": state.final_text_set,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "proof_status": state.proof_status.value,
            "original_text": self.original_text,
            "ocr_text": self.ocr_text,
        }


@dataclass
class Block:
    """版面分析得到的一个内容块。"""
    block_type: BlockType
    bbox: BBox
    lines: List[Line] = field(default_factory=list)
    order: int = 0             # 阅读顺序（从版面分析得到）
    id: Optional[int] = None

    source: BlockSource = BlockSource.AUTO_LAYOUT  # 块来源
    recognizable: bool = True                       # 是否送 OCR
    note: str = ""                                  # 用户备注或系统说明
    source_label: str = ""                          # 原始 PP-VL/Paddle label
    raw_payload: dict[str, Any] = field(default_factory=dict)  # 外部引擎原始块属性
    app_payload: dict[str, Any] = field(default_factory=dict)  # 应用派生状态/人工绑定
    uid: str = ""                                   # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "block")

    @property
    def avg_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.confidence for l in self.lines) / len(self.lines)


@dataclass
class Page:
    """一页（对应一张导入后生成的页面工作图）。"""
    image_path: str            # 页面工作图路径；导入时写入标准化图片或 PDF 渲染页图
    width: int
    height: int
    blocks: List[Block] = field(default_factory=list)
    page_number: int = 1
    id: Optional[int] = None

    source_path: str = ""                   # 原始导入文件路径（图片或 PDF）
    source_type: str = "image"              # "image" | "pdf"
    source_page_index: int = 1              # PDF 页码（1-based）
    cache_image_path: str = ""              # 当前优先工作图路径；display_image_path 会优先使用它
    thumbnail_path: str = ""                # 缩略图路径
    status: PageStatus = PageStatus.IMPORTED
    error_message: str = ""                 # 当前页失败原因
    ocr_invalidated_reason: str = ""        # 版面变更导致 OCR 结果失效的原因
    ppvl_parsing_res_list: List[dict[str, Any]] = field(default_factory=list)
    uid: str = ""                           # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "page")

    @property
    def is_analyzed(self) -> bool:
        return len(self.blocks) > 0

    @property
    def display_image_path(self) -> str:
        """当前全流程使用的页面图路径。

        版面分析、OCR、校对裁图、画布渲染和导出图片层都以这个
        property 为准，确保 bbox 坐标始终落在同一张工作图上。
        """
        return self.cache_image_path or self.image_path

    @property
    def text_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.block_type in (
            BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE
        )]

    @property
    def text_ocr_blocks(self) -> List[Block]:
        """返回按统一 dispatch 策略应送文字 OCR 的块。"""
        from app.core.ocr_dispatch_policy import should_dispatch_to_text_ocr

        return [b for b in self.blocks if should_dispatch_to_text_ocr(b)]

    @property
    def total_lines(self) -> int:
        return sum(len(b.lines) for b in self.blocks)

    @property
    def has_ocr_result(self) -> bool:
        """当前页是否已有可展示/校对的 OCR 行。"""
        return self.total_lines > 0

    @property
    def is_ocr_done(self) -> bool:
        """当前页是否已通过 OCR 阶段，可进入或继续校对。"""
        return self.status in OCR_AVAILABLE_PAGE_STATUSES

    @property
    def needs_ocr_rerun(self) -> bool:
        """版面变更后，已有 OCR 结果是否被显式标记为失效。"""
        return bool(self.ocr_invalidated_reason)

    def invalidate_ocr(self, reason: str) -> None:
        self.ocr_invalidated_reason = str(reason or "layout_changed")

    def clear_ocr_invalidation(self) -> None:
        self.ocr_invalidated_reason = ""

    def reconcile_ocr_done_from_result(self) -> None:
        """Promote loaded OCR content into explicit page state."""
        if (
            self.has_ocr_result
            and not self.is_ocr_done
            and not self.needs_ocr_rerun
            and not self.error_message
        ):
            self.status = PageStatus.OCR_DONE


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
    def has_any_ocr_result(self) -> bool:
        return any(p.has_ocr_result for p in self.pages)

    @property
    def all_pages_ocr_done(self) -> bool:
        return bool(self.pages) and all(p.is_ocr_done for p in self.pages)

    @property
    def has_any_ocr_done_page(self) -> bool:
        return any(p.is_ocr_done for p in self.pages)

    @property
    def has_pending_ocr_pages(self) -> bool:
        return any(
            p.is_analyzed and (not p.is_ocr_done or p.needs_ocr_rerun)
            for p in self.pages
        )

    @property
    def has_unrecognized_blocks(self) -> bool:
        return any(
            not b.lines
            for p in self.pages for b in p.text_ocr_blocks
        )
