"""Display-only quarter-turn transforms for images and source bboxes."""
from __future__ import annotations

from PySide6.QtGui import QImage, QPixmap, QTransform


XYXY = tuple[int, int, int, int]


def quarter_turns(value: int) -> int:
    return int(value) % 4


def rotated_size(width: int, height: int, turns: int) -> tuple[int, int]:
    return (height, width) if quarter_turns(turns) % 2 else (width, height)


def rotate_bbox(bbox: XYXY, width: int, height: int, turns: int) -> XYXY:
    """Map one source-coordinate half-open bbox into displayed coordinates."""
    x1, y1, x2, y2 = bbox
    normalized = quarter_turns(turns)
    if normalized == 1:
        return height - y2, x1, height - y1, x2
    if normalized == 2:
        return width - x2, height - y2, width - x1, height - y1
    if normalized == 3:
        return y1, width - x2, y2, width - x1
    return bbox


def source_point_from_display(
    x: float,
    y: float,
    width: int,
    height: int,
    turns: int,
) -> tuple[float, float]:
    normalized = quarter_turns(turns)
    if normalized == 1:
        return y, height - x
    if normalized == 2:
        return width - x, height - y
    if normalized == 3:
        return width - y, x
    return x, y


def rotate_image(image: QImage, turns: int) -> QImage:
    normalized = quarter_turns(turns)
    if image.isNull() or normalized == 0:
        return image
    return image.transformed(QTransform().rotate(90 * normalized))


def rotate_pixmap(pixmap: QPixmap, turns: int) -> QPixmap:
    normalized = quarter_turns(turns)
    if pixmap.isNull() or normalized == 0:
        return pixmap
    return pixmap.transformed(QTransform().rotate(90 * normalized))


__all__ = [
    "quarter_turns",
    "rotate_bbox",
    "rotate_image",
    "rotate_pixmap",
    "rotated_size",
    "source_point_from_display",
]
