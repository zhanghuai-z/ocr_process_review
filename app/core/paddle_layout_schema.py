"""Typed view over Paddle layout records used by the app.

The raw Paddle record is still preserved, but app code should read the
normalized label/bbox/text/score through this module instead of guessing keys
at every call site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.bbox_extraction import (
    bbox_from_variant,
    raw_bbox_max_from_variant,
)
from app.core.bbox_utils import scale_bbox
from app.core.paddle_labels import authoritative_paddle_label, normalize_paddle_label
from app.models import BBox

PADDLE_TEXT_KEYS = ("block_content",)
PADDLE_SCORE_KEYS = ("score",)
PADDLE_LAYOUT_BBOX_KEYS = (
    "block_bbox",
    "block_polygon_points",
    "coordinate",
    "polygon_points",
)


@dataclass(frozen=True)
class PaddleLayoutRecord:
    raw: dict[str, Any]
    label: str
    normalized_label: str
    bbox: BBox
    text: str = ""
    score: float | None = None

    @property
    def signature(self) -> tuple[str, int, int, int, int]:
        return (self.label, self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h)


def paddle_record_label(record: dict[str, Any], default: str = "unknown") -> str:
    return authoritative_paddle_label(record, default)


def paddle_record_text(record: dict[str, Any], *, include_markdown: bool = True) -> str:
    for key in PADDLE_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def paddle_record_score(record: dict[str, Any]) -> float | None:
    for key in PADDLE_SCORE_KEYS:
        value = record.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def paddle_record_bbox(record: dict[str, Any], page_width: int, page_height: int) -> BBox | None:
    for key in PADDLE_LAYOUT_BBOX_KEYS:
        if key not in record:
            continue
        bbox = bbox_from_variant(record.get(key), max_w=page_width, max_h=page_height)
        if bbox is not None:
            return bbox
    return None


def raw_bbox_max_from_record(record: dict[str, Any]) -> tuple[float, float] | None:
    for key in PADDLE_LAYOUT_BBOX_KEYS:
        if key not in record:
            continue
        max_xy = raw_bbox_max_from_variant(record.get(key))
        if max_xy is not None:
            return max_xy
    return None


def normalize_paddle_layout_record(
    record: dict[str, Any],
    *,
    page_width: int,
    page_height: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    default_label: str = "unknown",
) -> PaddleLayoutRecord | None:
    bbox = paddle_record_bbox(record, page_width, page_height)
    if bbox is None or bbox.area <= 0:
        return None
    if scale_x != 1.0 or scale_y != 1.0:
        bbox = scale_bbox(bbox, scale_x, scale_y).clamp(page_width, page_height)
        if bbox.area <= 0:
            return None

    label = paddle_record_label(record, default=default_label)
    return PaddleLayoutRecord(
        raw=dict(record),
        label=label,
        normalized_label=normalize_paddle_label(label),
        bbox=bbox,
        text=paddle_record_text(record),
        score=paddle_record_score(record),
    )


def route_subblock_payload(record: PaddleLayoutRecord) -> dict[str, Any]:
    return {
        "block_label": record.label,
        "block_bbox": list(record.bbox.to_xyxy()),
        "raw_payload": dict(record.raw),
    }
