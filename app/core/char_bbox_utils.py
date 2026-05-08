from __future__ import annotations

from typing import List

from app.models import BBox, Char, Line


LINE_DIRECTION_HORIZONTAL = "horizontal"
LINE_DIRECTION_VERTICAL = "vertical"


def infer_line_direction(line_bbox: BBox, text_length: int) -> str:
    """根据行框长宽比推断字符分布方向。"""
    if text_length <= 1:
        return LINE_DIRECTION_HORIZONTAL
    bbox = line_bbox.normalize()
    if bbox.h > bbox.w * 1.4:
        return LINE_DIRECTION_VERTICAL
    return LINE_DIRECTION_HORIZONTAL


def split_line_bbox_into_char_bboxes(line_bbox: BBox, text: str) -> List[BBox]:
    """在缺少字符级 bbox 时，按行框方向切分出字符框。"""
    text_length = len(text)
    if text_length <= 0:
        return []

    bbox = line_bbox.normalize()
    if bbox.w <= 0 or bbox.h <= 0:
        return []
    if text_length == 1:
        return [bbox]

    direction = infer_line_direction(bbox, text_length)
    char_bboxes: List[BBox] = []
    if direction == LINE_DIRECTION_VERTICAL:
        edges = [
            bbox.y + round(idx * bbox.h / float(text_length))
            for idx in range(text_length + 1)
        ]
        for idx in range(text_length):
            y1 = edges[idx]
            y2 = max(y1 + 1, edges[idx + 1])
            char_bboxes.append(BBox(bbox.x, y1, bbox.w, y2 - y1))
    else:
        edges = [
            bbox.x + round(idx * bbox.w / float(text_length))
            for idx in range(text_length + 1)
        ]
        for idx in range(text_length):
            x1 = edges[idx]
            x2 = max(x1 + 1, edges[idx + 1])
            char_bboxes.append(BBox(x1, bbox.y, x2 - x1, bbox.h))
    return char_bboxes


def ensure_line_char_bboxes(line: Line) -> List[Char]:
    """确保 line.chars 至少拥有与文本长度一致的 page-space bbox。"""
    if not line.text:
        line.chars = []
        return []

    split_bboxes = split_line_bbox_into_char_bboxes(line.bbox, line.text)
    chars: List[Char] = []
    for idx, glyph in enumerate(line.text):
        existing = line.chars[idx] if idx < len(line.chars) else None
        bbox = (
            existing.bbox.normalize()
            if existing and existing.bbox is not None and existing.bbox.area > 0
            else split_bboxes[idx]
        )
        confidence = (
            float(existing.confidence)
            if existing is not None
            else float(line.confidence)
        )
        char_id = existing.id if existing is not None else None
        chars.append(Char(char=glyph, confidence=confidence, bbox=bbox, id=char_id))
    line.chars = chars
    return chars
