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
from typing import Callable, List, Optional

import cv2
import numpy as np

from app.core.coordinate_seam import CropCoordinateSeam
from app.core.ocr_dispatch_policy import (
    should_block_page_ocr_line,
    should_dispatch_to_text_ocr,
)
from app.core.page_errors import is_ocr_error_message
from app.core.proof_status import apply_auto_flag
from app.core.spatial_matching import merge_bboxes, select_container_block_for_line
from app.engines import OcrContext, get_engine_bbox_space, supports_page_block_ocr
from app.engines.fake_ocr_engine import FakeOcrEngine
from app.models import (
    Block, BlockType, BBox, Line, OcrProject, Page,
)
from app.core.logging import get_logger
from app.services.proof_crop_service import ProofCropService

logger = get_logger(__name__)

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

    def __init__(self, engine: Optional[object] = None, hybrid_prepass_engine: Optional[object] = None):
        """初始化 OCR 管线。

        Args:
            engine: OcrEngine 实现。如果为 None，使用 FakeOcrEngine。
        """
        self._engine = engine or FakeOcrEngine()
        self._hybrid_prepass_engine = hybrid_prepass_engine
        self._proof_crop_service = ProofCropService()

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
                self._clear_ocr_error(page)
                if progress_callback:
                    progress_callback(OcrProgress(
                        current_page=page_idx + 1,
                        total_pages=total_pages,
                        current_block=0,
                        total_blocks=0,
                        completed_pages=page_idx,
                        message=f"OCR 识别准备中… 第 {page_idx + 1}/{total_pages} 页",
                    ))
                img = cv2.imread(page.display_image_path)
                if img is None:
                    logger.warning("Cannot read image: %s", page.display_image_path)
                    page.error_message = f"OCR 图像读取失败：{page.display_image_path}"
                    for block in page.blocks:
                        if should_dispatch_to_text_ocr(block):
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

                total_blocks = len([b for b in page.blocks if should_dispatch_to_text_ocr(b)])

                if self._prefers_page_hybrid_blocks():
                    def emit_hybrid_progress(current: int, total: int, message: str) -> None:
                        if progress_callback:
                            progress_callback(OcrProgress(
                                current_page=page_idx + 1,
                                total_pages=total_pages,
                                current_block=current,
                                total_blocks=total,
                                completed_pages=page_idx,
                                message=message,
                            ))
                    try:
                        self._process_page_with_hybrid_blocks(
                            img,
                            page,
                            page_idx=page_idx,
                            progress_callback=emit_hybrid_progress,
                        )
                    except Exception as e:
                        logger.error(
                            "Page hybrid OCR failed: page=%d: %s",
                            page_idx, e,
                        )
                        page.error_message = f"OCR 失败：{e}"
                        result.failed_blocks.append((page_idx, -1, str(e)))
                    if progress_callback:
                        progress_callback(OcrProgress(
                            current_page=page_idx + 1,
                            total_pages=total_pages,
                            current_block=max(1, total_blocks),
                            total_blocks=max(1, total_blocks),
                            completed_pages=page_idx + 1,
                            message=f"OCR 识别中… 第 {page_idx + 1}/{total_pages} 页，Hanwang micro-recblock 已写回版面块",
                        ))
                    self._normalize_proof_crops(
                        page,
                        page_idx=page_idx,
                        total_pages=total_pages,
                        progress_callback=progress_callback,
                    )
                    result.pages.append(page)
                    continue

                if self._prefers_page_ocr():
                    try:
                        self._process_page_with_page_ocr(img, page, page_idx)
                    except Exception as e:
                        logger.error(
                            "Page OCR failed: page=%d: %s",
                            page_idx, e,
                        )
                        page.error_message = f"OCR 失败：{e}"
                        result.failed_blocks.append((page_idx, -1, str(e)))
                    if progress_callback:
                        progress_callback(OcrProgress(
                            current_page=page_idx + 1,
                            total_pages=total_pages,
                            current_block=max(1, total_blocks),
                            total_blocks=max(1, total_blocks),
                            completed_pages=page_idx + 1,
                            message=f"OCR 识别中… 第 {page_idx + 1}/{total_pages} 页，PP-OCRv5 页级 proof line 已归属到版面块",
                        ))
                    self._normalize_proof_crops(
                        page,
                        page_idx=page_idx,
                        total_pages=total_pages,
                        progress_callback=progress_callback,
                    )
                    result.pages.append(page)
                    continue

                block_idx = 0
                page_failures: list[str] = []

                for block in page.blocks:
                    if not should_dispatch_to_text_ocr(block):
                        continue

                    try:
                        lines = self._process_block(img, block, page, page_idx)
                        block.lines = lines
                    except Exception as e:
                        logger.error(
                            "OCR failed: page=%d block=%d: %s",
                            page_idx, block.order, e,
                        )
                        self._append_block_failure_note(block, str(e))
                        page_failures.append(f"块 {block.order}: {e}")
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
                elif page.total_lines == 0 and page_failures and not page.error_message:
                    summary = "；".join(page_failures[:3])
                    if len(page_failures) > 3:
                        summary += "；…"
                    page.error_message = f"OCR 失败：{summary}"

                self._normalize_proof_crops(
                    page,
                    page_idx=page_idx,
                    total_pages=total_pages,
                    progress_callback=progress_callback,
                )
                result.pages.append(page)
        finally:
            self.close()

        return result

    @staticmethod
    def _clear_ocr_error(page: Page) -> None:
        """Clear stale OCR-owned errors before retrying OCR on a page."""
        if is_ocr_error_message(page.error_message):
            page.error_message = ""

    def _normalize_proof_crops(
        self,
        page: Page,
        *,
        page_idx: int,
        total_pages: int,
        progress_callback: Optional[Callable[[OcrProgress], None]] = None,
    ):
        stats = self._proof_crop_service.normalize_pages([page])
        fallback_total = stats.fallback_chars + stats.unavailable_chars
        if fallback_total <= 0:
            return stats

        message = (
            f"警告：第 {page_idx + 1}/{total_pages} 页触发 proof fallback，"
            f"{stats.fallback_lines} 行/{fallback_total} 字使用估算或不可用字框"
        )
        logger.warning(message)
        if progress_callback:
            progress_callback(OcrProgress(
                current_page=page_idx + 1,
                total_pages=total_pages,
                current_block=0,
                total_blocks=0,
                completed_pages=page_idx + 1,
                message=message,
            ))
        return stats

    def _prefers_page_ocr(self) -> bool:
        return bool(getattr(self._engine, "prefer_page_ocr", False))

    def _prefers_page_hybrid_blocks(self) -> bool:
        return supports_page_block_ocr(self._engine)

    def _append_block_failure_note(self, block: Block, message: str) -> None:
        note = f"OCR failed: {message}"
        if not block.note:
            block.note = note
            return
        if note not in block.note:
            block.note = f"{block.note}\n{note}"

    def _process_page_with_page_ocr(
        self,
        img: np.ndarray,
        page: Page,
        page_idx: int,
        engine: object | None = None,
    ) -> None:
        """Run PP-OCRv5 once on the page, then assign each line to one layout block.

        Structure blocks stay as containers; proof-facing line text/geometry and
        char/token boxes come only from the OCR engine result.  A line is consumed
        at most once by choosing the best spatial container.
        """
        full_page = BBox(0, 0, img.shape[1], img.shape[0])
        seam = CropCoordinateSeam.from_page_bbox(
            full_page,
            page_w=img.shape[1],
            page_h=img.shape[0],
        )
        ocr_engine = engine or self._engine
        bbox_space = get_engine_bbox_space(ocr_engine)
        context = OcrContext(
            page_image_path=page.display_image_path,
            page_number=page.page_number,
            block_id=None,
            block_type=BlockType.TEXT,
            page_bbox=full_page,
            crop_bbox=seam.crop_bbox,
            expected_bbox_space=bbox_space,
        )
        lines = ocr_engine.recognize(img, context)
        self._normalize_engine_lines(lines, seam, bbox_space)
        self._assign_page_ocr_lines_to_blocks(page, lines)

    def _hybrid_page_ocr_prepass_engine(self) -> object:
        if self._hybrid_prepass_engine is not None:
            return self._hybrid_prepass_engine
        from app.engines.real_ocr_adapter import ApiOcrEngine
        return ApiOcrEngine()

    def _process_page_with_hybrid_blocks(
        self,
        img: np.ndarray,
        page: Page,
        page_idx: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        prepass_engine = self._hybrid_page_ocr_prepass_engine()
        if not bool(getattr(prepass_engine, "prefer_page_ocr", False)):
            raise RuntimeError("Hanwang hybrid OCR requires a PP-OCRv5 page-line prepass engine")
        self._process_page_with_page_ocr(
            img,
            page,
            page_idx,
            engine=prepass_engine,
        )
        if progress_callback:
            line_count = sum(len(block.lines) for block in page.blocks)
            progress_callback(0, max(1, len(page.blocks)), f"PP-OCRv5 page-line prepass complete: {line_count} lines")
        if not supports_page_block_ocr(self._engine):
            raise RuntimeError("Configured OCR engine does not support page-block OCR")
        self._engine.recognize_page_blocks(
            img,
            page,
            progress_callback=progress_callback,
        )

    def _normalize_engine_lines(
        self,
        lines: List[Line],
        seam: CropCoordinateSeam,
        bbox_space: str,
    ) -> None:
        for line in lines:
            line.bbox = seam.to_page_bbox(
                line.bbox,
                source_space=bbox_space,
            )
            for char in line.chars:
                if char.bbox is None or char.bbox.area <= 0:
                    continue
                char.bbox = seam.to_page_bbox(
                    char.bbox,
                    source_space=bbox_space,
                )
            apply_auto_flag(line)
            line.ensure_text_contract(fill_original=True)

    def _assign_page_ocr_lines_to_blocks(self, page: Page, lines: list[Line]) -> None:
        for block in page.blocks:
            if should_dispatch_to_text_ocr(block):
                block.lines = []

        containers = [
            block for block in page.blocks
            if should_dispatch_to_text_ocr(block)
        ]
        blockers = [
            block for block in page.blocks
            if should_block_page_ocr_line(block)
        ]
        unmatched: list[Line] = []
        for line in sorted(lines, key=lambda item: (item.bbox.y, item.bbox.x)):
            blocker = select_container_block_for_line(line, blockers)
            if blocker is not None:
                continue
            block = select_container_block_for_line(line, containers)
            if block is None:
                unmatched.append(line)
            else:
                block.lines.append(line)

        if not unmatched:
            return
        merged = self._merge_line_bboxes(unmatched)
        if merged is None:
            return
        synthetic = Block(
            block_type=BlockType.TEXT,
            bbox=merged,
            lines=unmatched,
            order=(max((block.order for block in page.blocks), default=-1) + 1),
            note="PP-OCRv5 unmatched proof lines",
        )
        page.blocks.append(synthetic)

    def assign_page_ocr_lines_to_blocks(self, page: Page, lines: list[Line]) -> None:
        """Public wrapper used when PP-OCRv5 proof lines finish before layout."""
        self._assign_page_ocr_lines_to_blocks(page, lines)

    def _merge_line_bboxes(self, lines: list[Line]) -> BBox | None:
        return merge_bboxes([line.bbox for line in lines])

    def process_block(
        self,
        block: Block,
        page_image_path: str,
    ) -> Block:
        """处理单个块（用于块级重跑）。"""
        if not should_dispatch_to_text_ocr(block):
            return block

        img = cv2.imread(page_image_path)
        if img is None:
            logger.warning("Cannot read image: %s", page_image_path)
            return block

        page = Page(
            image_path=page_image_path,
            width=img.shape[1],
            height=img.shape[0],
            blocks=[block],
        )
        lines = self._process_block(img, block, page, 0)
        block.lines = lines
        self._normalize_proof_crops(
            page,
            page_idx=0,
            total_pages=1,
            progress_callback=None,
        )
        return block

    def _process_block(
        self,
        img: np.ndarray,
        block: Block,
        page: Page,
        page_idx: int,
    ) -> List[Line]:
        """处理单个块的 OCR。"""
        if not should_dispatch_to_text_ocr(block):
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
        self._normalize_engine_lines(lines, seam, bbox_space)

        return lines
