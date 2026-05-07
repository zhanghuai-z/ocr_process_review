"""BBox 规整工具：统一处理模型返回的脏框与坐标系回写。"""
from __future__ import annotations

from typing import Iterable, Sequence

from app.models import BBox


def _to_int(value) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def bbox_from_xyxy(coords: Iterable[float | int]) -> BBox:
    values = list(coords)[:4]
    while len(values) < 4:
        values.append(0)
    x1, y1, x2, y2 = (_to_int(v) for v in values)
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()


def bbox_from_quad(points: Sequence[Sequence[float | int]]) -> BBox:
    xs = [_to_int(point[0]) for point in points[:4]]
    ys = [_to_int(point[1]) for point in points[:4]]
    return BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys)).normalize()


def sanitize_xyxy_bbox(coords: Iterable[float | int], image_w: int, image_h: int) -> BBox:
    """将 xyxy/浮点/越界框规整为图像内的有效 bbox。"""
    return bbox_from_xyxy(coords).clamp(image_w, image_h)


def is_crop_relative_bbox(bbox: BBox, crop_w: int, crop_h: int, tolerance: int = 2) -> bool:
    """判断行框更像是 crop 局部坐标，而不是整页坐标。"""
    normalized = bbox.normalize()
    return (
        normalized.x >= -tolerance
        and normalized.y >= -tolerance
        and normalized.x2 <= crop_w + tolerance
        and normalized.y2 <= crop_h + tolerance
    )


def project_line_bbox(
    line_bbox: BBox,
    *,
    crop_origin_x: int,
    crop_origin_y: int,
    crop_w: int,
    crop_h: int,
    page_w: int,
    page_h: int,
) -> BBox:
    """把 OCR 行框统一回写到整页坐标。

    - crop-relative: 加上 crop 起点
    - page-relative: 保持原样
    最终都 clamp 到当前工作图范围内
    """
    normalized = line_bbox.normalize()
    if is_crop_relative_bbox(normalized, crop_w, crop_h):
        normalized = normalized.translated(crop_origin_x, crop_origin_y)
    return normalized.clamp(page_w, page_h)


def scale_bbox(bbox: BBox, scale_x: float, scale_y: float) -> BBox:
    """按比例缩放 bbox 并转为整数像素。"""
    x1 = _to_int(bbox.x * scale_x)
    y1 = _to_int(bbox.y * scale_y)
    x2 = _to_int((bbox.x + bbox.w) * scale_x)
    y2 = _to_int((bbox.y + bbox.h) * scale_y)
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()


def scale_bbox_to_page(
    bbox: BBox,
    *,
    source_w: int,
    source_h: int,
    page_w: int,
    page_h: int,
) -> BBox:
    """将 source 坐标系中的 bbox 缩放到 page 坐标系。"""
    if source_w <= 0 or source_h <= 0:
        return bbox.clamp(page_w, page_h)
    if source_w == page_w and source_h == page_h:
        return bbox.clamp(page_w, page_h)
    return scale_bbox(
        bbox,
        page_w / float(source_w),
        page_h / float(source_h),
    ).clamp(page_w, page_h)
