"""Shared OCR/layout bbox extraction helpers."""
from __future__ import annotations

from collections.abc import Sequence

from app.core.bbox_utils import bbox_from_quad, bbox_from_xyxy
from app.models import BBox

BBOX_FIELD_KEYS = (
    "coordinate", "bbox", "box", "block_bbox", "block_box",
    "polygon", "poly", "points", "block_polygon_points",
    "rec_box", "rec_bbox", "rec_poly", "rec_polys",
)


def _image_bounds(image_shape=None, max_w: int | None = None, max_h: int | None = None) -> tuple[int | None, int | None]:
    if image_shape is not None:
        try:
            height, width = image_shape[:2]
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    return max_w, max_h


def _clamp(bbox: BBox, image_shape=None, max_w: int | None = None, max_h: int | None = None) -> BBox:
    bbox = bbox.normalize()
    width, height = _image_bounds(image_shape, max_w, max_h)
    if width is None or height is None:
        return bbox
    return bbox.clamp(int(width), int(height))


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(values: Sequence) -> list[float] | None:
    numbers: list[float] = []
    for value in values:
        number = _as_float(value)
        if number is None:
            return None
        numbers.append(number)
    return numbers


def bbox_from_variant(
    value,
    *,
    image_shape=None,
    max_w: int | None = None,
    max_h: int | None = None,
) -> BBox | None:
    """Parse common OCR/layout bbox variants and optionally clamp to bounds."""
    if isinstance(value, dict):
        keys = set(value.keys())
        if {"x", "y", "w", "h"} <= keys:
            numbers = [_as_float(value.get(key)) for key in ("x", "y", "w", "h")]
            if any(number is None for number in numbers):
                return None
            x, y, w, h = numbers
            bbox = BBox.from_xyxy(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))
            return _clamp(bbox, image_shape, max_w, max_h)
        if {"x1", "y1", "x2", "y2"} <= keys:
            bbox = bbox_from_xyxy([value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")])
            return _clamp(bbox, image_shape, max_w, max_h)
        point_values: list[tuple[float, float]] = []
        for x_key, y_key in (("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")):
            if x_key not in value or y_key not in value:
                continue
            x = _as_float(value.get(x_key))
            y = _as_float(value.get(y_key))
            if x is None or y is None:
                return None
            point_values.append((x, y))
        if point_values:
            return _clamp(bbox_from_quad(point_values), image_shape, max_w, max_h)
        for key in BBOX_FIELD_KEYS:
            nested = value.get(key)
            if nested is None:
                continue
            bbox = bbox_from_variant(nested, image_shape=image_shape, max_w=max_w, max_h=max_h)
            if bbox is not None:
                return bbox
        return None

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) == 1 and isinstance(value[0], (list, tuple, dict)):
        return bbox_from_variant(value[0], image_shape=image_shape, max_w=max_w, max_h=max_h)

    if value and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value):
        points = [(point[0], point[1]) for point in value]
        return _clamp(bbox_from_quad(points), image_shape, max_w, max_h)

    numbers = _numeric_values(value)
    if numbers is None:
        return None
    if len(numbers) >= 8 and len(numbers) % 2 == 0:
        points = list(zip(numbers[0::2], numbers[1::2]))
        return _clamp(bbox_from_quad(points), image_shape, max_w, max_h)
    if len(numbers) >= 4:
        return _clamp(bbox_from_xyxy(numbers[:4]), image_shape, max_w, max_h)
    return None


def raw_bbox_max_from_variant(value) -> tuple[float, float] | None:
    """Return raw max x/y before scaling or clamping."""
    if value is None:
        return None
    if isinstance(value, dict):
        keys = set(value.keys())
        if {"x", "y", "w", "h"} <= keys:
            x = _as_float(value.get("x"))
            y = _as_float(value.get("y"))
            w = _as_float(value.get("w"))
            h = _as_float(value.get("h"))
            if None in (x, y, w, h):
                return None
            return x + w, y + h
        if {"x1", "y1", "x2", "y2"} <= keys:
            values = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
            numbers = _numeric_values(values)
            if numbers is None:
                return None
            return max(numbers[0], numbers[2]), max(numbers[1], numbers[3])
        xs: list[float] = []
        ys: list[float] = []
        for x_key, y_key in (("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")):
            if x_key not in value or y_key not in value:
                continue
            x = _as_float(value.get(x_key))
            y = _as_float(value.get(y_key))
            if x is None or y is None:
                return None
            xs.append(x)
            ys.append(y)
        if xs and ys:
            return max(xs), max(ys)
        for key in BBOX_FIELD_KEYS:
            nested = value.get(key)
            if nested is None:
                continue
            max_xy = raw_bbox_max_from_variant(nested)
            if max_xy is not None:
                return max_xy
        return None

    if isinstance(value, (list, tuple)):
        if value and all(isinstance(point, (list, tuple)) for point in value):
            xs: list[float] = []
            ys: list[float] = []
            for point in value:
                if len(point) < 2:
                    continue
                x = _as_float(point[0])
                y = _as_float(point[1])
                if x is None or y is None:
                    return None
                xs.append(x)
                ys.append(y)
            return (max(xs), max(ys)) if xs and ys else None
        numbers = _numeric_values(value)
        if numbers is None:
            return None
        if len(numbers) >= 8 and len(numbers) % 2 == 0:
            return max(numbers[0::2]), max(numbers[1::2])
        if len(numbers) >= 4:
            return max(numbers[0], numbers[2]), max(numbers[1], numbers[3])
    return None
