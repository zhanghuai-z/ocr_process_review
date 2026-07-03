from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

from .enums import (
    BlockSource, BlockType, OcrPolicy, PageStatus,
)
from .entity_id import ensure_entity_uid


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
    id: Optional[int] = None
    ocr_text: str = ""                    # OCR 原始文本
    review_flags: List[str] = field(default_factory=list)  # 疑点标签
    uid: str = ""                         # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "line")
        if not self.ocr_text:
            self.ocr_text = self.text

@dataclass
class BlockOrigin:
    """版面块来源事实。

    只记录块最初从哪里来、外部引擎当时给了什么标签和框。
    当前可编辑 bbox/type 仍在 Block 自身，人工修改不应覆盖 origin。
    """
    created_by: str = BlockSource.AUTO_LAYOUT.value
    source_engine: str = ""
    source_run_id: str = ""
    source_label: str = ""
    source_confidence: Optional[float] = None
    original_bbox: Optional[BBox] = None
    original_kind: Optional[BlockType] = None
    raw_artifact_uid: str = ""
    raw_json_path: str = ""
    raw_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_by": self.created_by,
            "source_engine": self.source_engine,
            "source_run_id": self.source_run_id,
            "source_label": self.source_label,
            "source_confidence": self.source_confidence,
            "original_bbox": self.original_bbox.to_dict() if self.original_bbox else None,
            "original_kind": self.original_kind.value if self.original_kind else "",
            "raw_artifact_uid": self.raw_artifact_uid,
            "raw_json_path": self.raw_json_path,
            "raw_index": self.raw_index,
        }


@dataclass
class PaddleBinding:
    """人工框与 Paddle 原始事实的结构化绑定。

    这是应用层 OCR/版面绑定状态，不属于 vendor raw payload，也不应再写入
    block payload as the primary fact.
    """
    status: str = ""
    source: str = ""
    block_type: str = ""
    source_label: str = ""
    text: str = ""
    parent_index: int = -1
    candidate_index: int = -1
    score: float = 0.0
    candidate_bbox: List[int] = field(default_factory=list)
    manual_bbox: List[int] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PaddleBinding | None":
        if not isinstance(payload, dict) or not payload:
            return None
        return cls(
            status=str(payload.get("status") or ""),
            source=str(payload.get("source") or ""),
            block_type=str(payload.get("block_type") or ""),
            source_label=str(payload.get("source_label") or ""),
            text=str(payload.get("text") or ""),
            parent_index=_int_or_default(payload.get("parent_index"), -1),
            candidate_index=_int_or_default(payload.get("candidate_index"), -1),
            score=_float_or_default(payload.get("score"), 0.0),
            candidate_bbox=_int_list(payload.get("candidate_bbox")),
            manual_bbox=_int_list(payload.get("manual_bbox")),
            review_flags=[str(value) for value in payload.get("review_flags", [])]
            if isinstance(payload.get("review_flags"), list) else [],
            candidates=[str(value) for value in payload.get("candidates", [])]
            if isinstance(payload.get("candidates"), list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "source": self.source,
            "block_type": self.block_type,
            "source_label": self.source_label,
            "text": self.text,
            "parent_index": self.parent_index,
            "candidate_index": self.candidate_index,
            "score": round(float(self.score), 6),
            "review_flags": list(self.review_flags),
        }
        if self.candidate_bbox:
            payload["candidate_bbox"] = [int(value) for value in self.candidate_bbox]
        if self.manual_bbox:
            payload["manual_bbox"] = [int(value) for value in self.manual_bbox]
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        return payload


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_list(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


@dataclass
class Block:
    """版面分析得到的一个内容块。"""
    block_type: BlockType
    bbox: BBox
    lines: List[Line] = field(default_factory=list)
    order: int = 0             # 阅读顺序（从版面分析得到）
    id: Optional[int] = None

    source: BlockSource = BlockSource.AUTO_LAYOUT  # 块来源
    ocr_policy: OcrPolicy = OcrPolicy.TEXT_OCR      # OCR 调度策略
    note: str = ""                                  # 用户备注或系统说明
    source_label: str = ""                          # 原始 PP-VL/Paddle label
    origin: BlockOrigin | None = None               # 块来源事实；新代码优先读这里
    paddle_binding: PaddleBinding | None = None     # 人工框与 Paddle 原始事实的结构化绑定
    ocr_invalidated_reason: str = ""                # 块级 OCR 结果失效原因
    ocr_audit: dict[str, Any] = field(default_factory=dict)  # OCR 调试/审计事实
    table_text_layer_cells: list[dict[str, Any]] = field(default_factory=list)  # 表格双层 PDF 文本层
    uid: str = ""                                   # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "block")

    @property
    def avg_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.confidence for l in self.lines) / len(self.lines)


@dataclass
class RawOcrArtifact:
    """外部 OCR/版面引擎的原始证据引用。"""
    engine: str
    engine_version: str
    run_id: str = ""
    artifact_path: str = ""
    artifact_hash: str = ""
    records: List[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    page_uid: str = ""
    uid: str = ""
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "rawocr")

    @classmethod
    def from_paddle_layout_records(
        cls,
        records: List[dict[str, Any]],
        *,
        page_uid: str = "",
        run_id: str = "",
        artifact_path: str = "",
        artifact_hash: str = "",
    ) -> "RawOcrArtifact":
        return cls(
            engine="paddleocr-vl",
            engine_version="1.6",
            run_id=run_id,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            records=list(records),
            page_uid=page_uid,
        )


@dataclass
class LayoutEditEvent:
    """人工/系统版面编辑事件。"""
    page_uid: str
    op: str
    target_uid: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    actor: str = "user"
    created_at: float = field(default_factory=time.time)
    uid: str = ""
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "layoutedit")


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
    raw_layout_artifact: RawOcrArtifact | None = None
    layout_edit_events: List[LayoutEditEvent] = field(default_factory=list)
    uid: str = ""                           # 稳定业务 ID

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "page")
        if self.raw_layout_artifact is not None:
            self.raw_layout_artifact.page_uid = self.uid
        for event in self.layout_edit_events:
            event.page_uid = self.uid

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
        """返回当前策略明确应送文字 OCR 的块。"""
        return [b for b in self.blocks if b.ocr_policy == OcrPolicy.TEXT_OCR]

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
