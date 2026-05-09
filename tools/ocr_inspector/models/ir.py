"""OCR Inspector intermediate representation (IR).

All UI code consumes *only* these types.  Adapters convert raw engine JSON into IR.
Coordinate space: all bbox values in image pixel space (origin = top-left of raw image).
"""
from __future__ import annotations
import json
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence

OcrIrKind = Literal["text", "digit", "formula", "punct", "symbol", "other"]


def classify_ir_text(text: str) -> OcrIrKind:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if not compact:
        return "other"
    if compact.isdigit():
        return "digit"
    categories = [unicodedata.category(ch) for ch in compact]
    if all(c.startswith("P") for c in categories):
        return "punct"
    if all(c.startswith("S") for c in categories):
        return "symbol"
    if any(c.startswith("L") for c in categories):
        return "text"
    return "other"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in image pixel space."""
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        return cls(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_list(cls, vals: Sequence[float]) -> "BBox":
        if len(vals) == 4:
            x, y, w, h = vals
            if w > x and h > y:
                return cls.from_xyxy(x, y, w, h)
            return cls(x, y, w, h)
        raise ValueError(f"Expected 4 values, got {len(vals)}")

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True)
class Polygon:
    """Arbitrary polygon in image pixel space."""
    points: tuple  # tuple of (x, y) pairs

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["Polygon"]:
        if raw is None:
            return None
        try:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                first = raw[0]
                if isinstance(first, (list, tuple)):
                    return cls(tuple(tuple(p) for p in raw))
                if isinstance(first, (int, float)):
                    pts = [(raw[i], raw[i + 1]) for i in range(0, len(raw) - 1, 2)]
                    return cls(tuple(pts))
        except Exception:
            pass
        return None

    def to_bbox(self) -> Optional[BBox]:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))


@dataclass
class CharNode:
    id: str
    char: str
    bbox: Optional[BBox]
    polygon: Optional[Polygon] = None
    confidence: float = 1.0
    kind: OcrIrKind = "text"
    bbox_source: str = "ocr"
    bbox_granularity: str = "char"
    token_text: str = ""
    collection_kind: str = "char"
    raw: Any = None

    @classmethod
    def make(cls, char: str, **kw) -> "CharNode":
        return cls(id=str(uuid.uuid4()), char=char, **kw)


@dataclass
class LineNode:
    id: str
    text: str
    confidence: float
    bbox: Optional[BBox]
    polygon: Optional[Polygon] = None
    chars: List[CharNode] = field(default_factory=list)
    source_field: str = ""
    raw: Any = None

    @classmethod
    def make(cls, text: str, confidence: float, bbox: Optional[BBox], **kw) -> "LineNode":
        return cls(id=str(uuid.uuid4()), text=text, confidence=confidence, bbox=bbox, **kw)


@dataclass
class BlockNode:
    id: str
    label: str
    bbox: Optional[BBox]
    polygon: Optional[Polygon] = None
    content: str = ""
    lines: List[LineNode] = field(default_factory=list)
    order: int = 0
    raw: Any = None

    @classmethod
    def make(cls, label: str, bbox: Optional[BBox], **kw) -> "BlockNode":
        return cls(id=str(uuid.uuid4()), label=label, bbox=bbox, **kw)


@dataclass
class PageNode:
    id: str
    page_number: int
    image_path: str
    width: int = 0
    height: int = 0
    blocks: List[BlockNode] = field(default_factory=list)
    orphan_lines: List[LineNode] = field(default_factory=list)
    raw: Any = None

    @classmethod
    def make(cls, page_number: int, image_path: str, **kw) -> "PageNode":
        return cls(id=str(uuid.uuid4()), page_number=page_number, image_path=image_path, **kw)

    @property
    def all_lines(self) -> List[LineNode]:
        lines = self.orphan_lines[:]
        for block in self.blocks:
            lines.extend(block.lines)
        return lines

    @property
    def all_chars(self) -> List[CharNode]:
        return [c for ln in self.all_lines for c in ln.chars]


@dataclass
class DocumentNode:
    id: str
    source_path: str
    engine: str
    pages: List[PageNode] = field(default_factory=list)
    parse_log: List[str] = field(default_factory=list)
    raw: Any = None

    @classmethod
    def make(cls, source_path: str, engine: str = "unknown", **kw) -> "DocumentNode":
        return cls(id=str(uuid.uuid4()), source_path=source_path, engine=engine, **kw)
