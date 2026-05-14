"""Proof 侧裁图 / bbox / padding 共享 service。

把原本散落在 ``h_proof.py`` / ``v_proof.py`` 里的几段相同/相近的逻辑下沉到
一个地方：

- ``clamp_bbox_to_image(bbox, W, H)`` —— 把 BBox 修正到图像边界内（之前在
  v_proof._verified_char_crop 里手写一份）。
- ``adaptive_pad_for_bbox(bbox, max_pad=6, ratio=0.10)`` —— bbox 自适应 padding，
  避免纵排小字被固定 12px padding 拽进邻字（之前在 v_proof._verified_char_crop
  里手写一份）。
- ``verified_char_crop(cache, page_path, bbox, size, pad=None)`` —— 调
  ``PageImageCache.get_char_crop`` 前的坐标校验封装。
- ``clamp_line_box_pixels(bbox, W, H, pad_y=0)`` —— h_proof line crop 的
  边界 clamp，返回 (x1, y1, x2, y2) 像素元组。

这些纯函数没有状态、不依赖 Qt，只依赖 ``BBox`` 和 ``PageImageCache``。
不改动任何裁图行为：边界 clamp 规则、adaptive padding 公式、字符裁图调用都和
原 inline 实现逐字节等价。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtGui import QPixmap

from app.models import BBox
from app.core.page_image_cache import PageImageCache

logger = logging.getLogger(__name__)


def clamp_bbox_to_image(bbox: BBox, image_w: int, image_h: int) -> BBox:
    """把 bbox 修正到 [0, W) × [0, H) 内，返回新的 BBox。

    与原 v_proof inline 实现等价：先 clamp 左上角，再用 clamped 后的 x/y 算宽高。
    （之前一个已修复的 bug：用原始 ``bbox.x`` 计算宽度会越界 1px。）
    """
    clamped_x = max(0, min(bbox.x, image_w - 1))
    clamped_y = max(0, min(bbox.y, image_h - 1))
    return BBox(
        clamped_x,
        clamped_y,
        max(1, min(bbox.w, image_w - clamped_x)),
        max(1, min(bbox.h, image_h - clamped_y)),
    )


def adaptive_pad_for_bbox(bbox: BBox, *, max_pad: int = 6, ratio: float = 0.10) -> int:
    """根据 bbox 较短边自适应 padding：~10% 且不超 ``max_pad`` 像素。

    避免纵排字（高度 ~40px）被固定 12px padding 拽进邻字（之前在 v_proof
    里写死的 ``max(1, min(int(min(bbox.w, bbox.h) * 0.10), 6))``）。
    """
    return max(1, min(int(min(bbox.w, bbox.h) * ratio), max_pad))


def verified_char_crop(
    cache: PageImageCache,
    page_path: str,
    bbox: BBox,
    size: int,
    pad: Optional[int] = None,
) -> Optional[QPixmap]:
    """带坐标校验的字符裁图：超出图像范围时记 WARNING 并 clamp。

    pad 为 None 时根据 bbox 尺寸自适应（``adaptive_pad_for_bbox``）。
    """
    img = cache.get_image(page_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    if pad is None:
        pad = adaptive_pad_for_bbox(bbox)
    if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.w > W or bbox.y + bbox.h > H:
        logger.warning(
            "char bbox out of bounds: bbox=(%d,%d,%d,%d) img=(%d×%d) path=%s",
            bbox.x, bbox.y, bbox.w, bbox.h, W, H, page_path,
        )
        bbox = clamp_bbox_to_image(bbox, W, H)
    return cache.get_char_crop(page_path, bbox, size, pad=pad)


def clamp_line_box_pixels(
    bbox: BBox, image_w: int, image_h: int, *, pad_y: int = 0
) -> Optional[Tuple[int, int, int, int]]:
    """把行 bbox clamp 到图像内并加上下 pad_y，返回 (x1, y1, x2, y2)。

    若 clamp 后宽或高 ≤ 0，返回 None（行框完全异常）。
    与 h_proof 里 inline 实现等价（``ROW_PAD_Y`` 由调用方作为 ``pad_y`` 传入）。
    """
    line_box = bbox.normalize()
    x1 = max(0, line_box.x)
    y1 = max(0, line_box.y - pad_y)
    x2 = min(image_w, line_box.x2)
    y2 = min(image_h, line_box.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
