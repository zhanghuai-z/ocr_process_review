"""OCR 和版面分析引擎接口定义。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, TypeGuard

import numpy as np

from app.models import Block, BlockType, BBox, Line, Page

OCR_BBOX_SPACE_CROP = "crop"
OCR_BBOX_SPACE_PAGE = "page"


@dataclass
class OcrContext:
    """OCR 调用的上下文信息。"""
    page_image_path: str = ""
    page_number: int = 0
    block_id: Optional[int] = None
    block_type: BlockType = BlockType.TEXT
    language: str = "ch"
    page_bbox: Optional[BBox] = None
    crop_bbox: Optional[BBox] = None
    expected_bbox_space: str = OCR_BBOX_SPACE_CROP


class OcrEngine(Protocol):
    """OCR 引擎接口。

    所有适配器（本地 PaddleOCR、AiStudio API、Fake）都实现此接口。
    """

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        """识别图像中的文字。

        Args:
            image_bgr: BGR 格式的 numpy 图像数组。
            context: OCR 上下文信息。

        Returns:
            识别到的行列表，每行包含文本、置信度和 BBox。
            坐标是相对 image_bgr 的（调用方负责转换到整页坐标）。
        """
        ...


class PageBlockOcrEngine(Protocol):
    """Page-level engine that writes OCR results back to layout blocks.

    Hanwang micro-recblock uses this capability: Paddle/PP-VL supplies layout
    blocks, PP-OCR can supply page-line hints, and the engine returns block
    results in page coordinates.
    """

    engine_id: str
    prefer_page_hybrid_blocks: bool
    bbox_space: str

    def recognize_page_blocks(
        self,
        image_bgr: np.ndarray,
        page: Page,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> object:
        """Recognize a full page using existing layout blocks."""
        ...


def supports_page_block_ocr(engine: object) -> TypeGuard[PageBlockOcrEngine]:
    """Return whether an engine exposes the page-block OCR capability."""
    recognize_page_blocks = getattr(engine, "recognize_page_blocks", None)
    return bool(
        getattr(engine, "prefer_page_hybrid_blocks", False)
        and callable(recognize_page_blocks)
    )


def get_engine_bbox_space(engine: object) -> str:
    space = getattr(engine, "bbox_space", OCR_BBOX_SPACE_CROP)
    if space in (OCR_BBOX_SPACE_CROP, OCR_BBOX_SPACE_PAGE):
        return space
    return OCR_BBOX_SPACE_CROP


class LayoutEngine(Protocol):
    """版面分析引擎接口。"""

    def analyze(self, image_path: str) -> List[Block]:
        """分析图像版面。

        Args:
            image_path: 图像文件路径。

        Returns:
            版面块列表。
        """
        ...
