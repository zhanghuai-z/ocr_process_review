"""Shared immutable geometry value objects."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """Pixel bounding box using a top-left origin."""

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

    @classmethod
    def from_dict(cls, value: dict) -> "BBox":
        return cls(
            x=int(value["x"]),
            y=int(value["y"]),
            w=int(value["w"]),
            h=int(value["h"]),
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def clamp(self, max_w: int, max_h: int) -> "BBox":
        x1 = max(0, min(self.x, max_w))
        y1 = max(0, min(self.y, max_h))
        x2 = max(x1, min(self.x2, max_w))
        y2 = max(y1, min(self.y2, max_h))
        return BBox.from_xyxy(x1, y1, x2, y2)

    def normalize(self) -> "BBox":
        x = min(self.x, self.x2)
        y = min(self.y, self.y2)
        return BBox(x, y, abs(self.w), abs(self.h))

    def expand(self, pad: int) -> "BBox":
        return BBox(self.x - pad, self.y - pad, self.w + 2 * pad, self.h + 2 * pad)

    def translated(self, dx: int, dy: int) -> "BBox":
        return BBox(self.x + dx, self.y + dy, self.w, self.h)

    def iou(self, other: "BBox") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


__all__ = ["BBox"]
