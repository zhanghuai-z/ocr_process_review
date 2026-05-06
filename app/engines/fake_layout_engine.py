"""Fake 版面分析引擎：用于测试。

根据图片尺寸生成固定块，行为稳定。
"""
from __future__ import annotations
from typing import List

from app.models import Block, BlockSource, BlockType, BBox


class FakeLayoutEngine:
    """Fake 版面分析引擎。

    根据图片尺寸生成固定块，用于测试。
    """

    def analyze(self, image_path: str) -> List[Block]:
        """分析图像版面，生成模拟块。

        Args:
            image_path: 图像路径（用于获取尺寸，实际使用 OpenCV 读取）。

        Returns:
            模拟的版面块列表。
        """
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return [self._make_fallback_block(800, 600)]

        h, w = img.shape[:2]
        blocks = [
            Block(
                block_type=BlockType.TITLE,
                bbox=BBox(20, 20, w - 40, min(80, h // 4)),
                source=BlockSource.AUTO_LAYOUT,
                order=0,
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(20, max(100, h // 4 + 10), w - 40, max(100, h // 2)),
                source=BlockSource.AUTO_LAYOUT,
                order=1,
            ),
        ]
        return blocks

    def _make_fallback_block(self, w: int, h: int) -> Block:
        return Block(
            block_type=BlockType.TEXT,
            bbox=BBox(20, 20, w - 40, h - 40),
            source=BlockSource.AUTO_LAYOUT,
            order=0,
        )
