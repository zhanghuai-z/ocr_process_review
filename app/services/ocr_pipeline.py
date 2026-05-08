"""OCR 管线：遍历 page/block，调用 engine，转换坐标。

职责：
- 遍历 page/block
- 跳过不可识别块
- 裁剪 block ROI
- 调用 OCR engine
- 将行 bbox 从 crop 坐标转换回 page 坐标
- 设置 proof status
- 记录失败
- 发出进度
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

import cv2
import numpy as np

from app.core.coordinate_seam import CropCoordinateSeam
from app.engines import OcrContext, get_engine_bbox_space
from app.engines.fake_ocr_engine import FakeOcrEngine
from app.models import (
    Block, BlockType, BBox, Line, OcrProject, Page, ProofStatus,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# 非文字块类型默认不送 OCR
NON_OCR_BLOCK_TYPES = {BlockType.FIGURE, BlockType.TABLE, BlockType.UNKNOWN}

# 自动标记低置信行阈值
AUTO_FLAG_THRESHOLD = 0.80


@dataclass
class OcrProgress:
    """OCR 进度信息。"""
    current_page: int = 0
    total_pages: int = 0
    current_block: int = 0
    total_blocks: int = 0
    completed_pages: int = 0
    message: str = ""


@dataclass
class OcrResult:
    """OCR 处理结果。"""
    pages: List[Page] = field(default_factory=list)
    failed_blocks: List[tuple[int, int, str]] = field(default_factory=list)  # (page_idx, block_idx, error)


class OcrPipeline:
    """OCR 管线。"""

    def __init__(self, engine: Optional[object] = None):
        """初始化 OCR 管线。

        Args:
            engine: OcrEngine 实现。如果为 None，使用 FakeOcrEngine。
        """
        self._engine = engine or FakeOcrEngine()

    def close(self) -> None:
        engine = self._engine
        if engine is not None and hasattr(engine, "close"):
            engine.close()

    def process_project(
        self,
        project: OcrProject,
        progress_callback: Optional[Callable[[OcrProgress], None]] = None,
    ) -> OcrResult:
        """处理项目所有页的所有可识别块。"""
        result = OcrResult()
        total_pages = len(project.pages)

        try:
            for page_idx, page in enumerate(project.pages):
                img = cv2.imread(page.display_image_path)
                if img is None:
                    logger.warning("Cannot read image: %s", page.display_image_path)
                    for block in page.blocks:
                        if block.recognizable:
                            result.failed_blocks.append(
                                (page_idx, block.order, f"Cannot read image: {page.display_image_path}")
                            )
                    if progress_callback:
                        progress_callback(OcrProgress(
                            current_page=page_idx + 1,
                            total_pages=total_pages,
                            current_block=0,
                            total_blocks=0,
                            completed_pages=page_idx + 1,
                            message=f"OCR 跳过：第 {page_idx + 1}/{total_pages} 页图像读取失败",
                        ))
                    continue

                total_blocks = len([b for b in page.blocks if b.recognizable])
                block_idx = 0

                for block in page.blocks:
                    if not block.recognizable:
                        continue

                    try:
                        lines = self._process_block(img, block, page, page_idx)
                        block.lines = lines
                    except Exception as e:
                        logger.error(
                            "OCR failed: page=%d block=%d: %s",
                            page_idx, block.order, e,
                        )
                        result.failed_blocks.append(
                            (page_idx, block.order, str(e))
                        )

                    block_idx += 1
                    if progress_callback:
                        progress_callback(OcrProgress(
                            current_page=page_idx + 1,
                            total_pages=total_pages,
                            current_block=block_idx,
                            total_blocks=total_blocks,
                            completed_pages=(page_idx + 1) if block_idx == total_blocks else page_idx,
                            message=f"OCR 识别中… 第 {page_idx + 1}/{total_pages} 页，块 {block_idx}/{total_blocks}",
                        ))

                if total_blocks == 0 and progress_callback:
                    progress_callback(OcrProgress(
                        current_page=page_idx + 1,
                        total_pages=total_pages,
                        current_block=0,
                        total_blocks=0,
                        completed_pages=page_idx + 1,
                        message=f"OCR 跳过：第 {page_idx + 1}/{total_pages} 页没有可识别块",
                    ))

                result.pages.append(page)
        finally:
            self.close()

        return result

    def process_block(
        self,
        block: Block,
        page_image_path: str,
    ) -> Block:
        """处理单个块（用于块级重跑）。"""
        img = cv2.imread(page_image_path)
        if img is None:
            logger.warning("Cannot read image: %s", page_image_path)
            return block

        page = Page(image_path=page_image_path, width=0, height=0)
        lines = self._process_block(img, block, page, 0)
        block.lines = lines
        return block

    def _process_block(
        self,
        img: np.ndarray,
        block: Block,
        page: Page,
        page_idx: int,
    ) -> List[Line]:
        """处理单个块的 OCR。"""
        if block.block_type in NON_OCR_BLOCK_TYPES:
            return []

        bb = block.bbox.normalize().clamp(img.shape[1], img.shape[0])
        block.bbox = bb
        # clamp to image boundaries
        x1 = max(0, bb.x)
        y1 = max(0, bb.y)
        x2 = min(img.shape[1], bb.x + bb.w)
        y2 = min(img.shape[0], bb.y + bb.h)

        if x2 <= x1 or y2 <= y1:
            return []

        seam = CropCoordinateSeam.from_page_bbox(
            bb,
            page_w=img.shape[1],
            page_h=img.shape[0],
        )
        crop = seam.crop_image(img)
        if crop.size == 0:
            return []

        bbox_space = get_engine_bbox_space(self._engine)
        context = OcrContext(
            page_image_path=page.display_image_path,
            page_number=page.page_number,
            block_id=block.id,
            block_type=block.block_type,
            page_bbox=bb,
            crop_bbox=seam.crop_bbox,
            expected_bbox_space=bbox_space,
        )

        lines = self._engine.recognize(crop, context)

        # 转换 bbox 从 crop 坐标到 page 坐标
        for line in lines:
            line.bbox = seam.to_page_bbox(
                line.bbox,
                source_space=bbox_space,
            )
            # 设置 proof status
            if line.confidence < AUTO_FLAG_THRESHOLD:
                if line.proof_status == ProofStatus.UNCHECKED:
                    line.proof_status = ProofStatus.AUTO_FLAGGED
            # 保存 OCR 原文
            if not line.ocr_text:
                line.ocr_text = line.text
            if not line.original_text:
                line.original_text = line.text

        return lines
