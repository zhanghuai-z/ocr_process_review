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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable, List, Optional

import cv2
import numpy as np

from app.core.coordinate_seam import CropCoordinateSeam
from app.models.block_state import is_ocr_text_invalidated
from app.core.line_text_contract import ensure_line_text_contract
from app.core.ocr_dispatch_policy import (
    should_dispatch_to_text_ocr,
)
from app.core.ocr_line_hints import is_ppocr_page_line_hint, mark_ppocr_page_line_hint
from app.core.page_errors import is_ocr_error_message
from app.core.spatial_matching import merge_bboxes, select_container_block_for_line
from app.engines import OcrContext, get_engine_bbox_space, supports_page_block_ocr
from app.engines.fake_ocr_engine import FakeOcrEngine
from app.models import (
    Block, BlockType, BBox, Line, OcrProject, Page,
)
from app.models.layout_block_state import append_layout_block_note_once, set_layout_block_bbox
from app.models.layout_projection import (
    append_page_layout_block,
    replace_page_layout_blocks,
)
from app.models.layout_block_view import iter_page_layout_block_views
from app.models.ocr_character_observation import line_ocr_chars, set_ocr_char_bbox
from app.models.ocr_observation import (
    append_block_ocr_line,
    block_ocr_line_count,
    block_ocr_lines,
    clear_block_ocr_lines,
    line_ocr_bbox,
    page_ocr_line_count,
    replace_block_ocr_lines,
    set_ocr_line_bbox,
)
from app.models.page_state import clear_page_error_message, mark_page_ocr_failed, page_error_message
from app.core.logging import get_logger
from app.services.proof_crop_service import ProofCropService
from app.services.table_text_layer_service import TableTextLayerService
from app.services.ocr_dispatch_plan import (
    build_text_ocr_dispatch_plan,
)
from app.services.ocr_run_result import (
    OcrProgress,
    OcrRunResult,
    PageOcrRunResult,
)

logger = get_logger(__name__)
OCR_PAGE_CONCURRENCY_CAP = 20


class OcrPipeline:
    """OCR 管线。"""

    def __init__(
        self,
        engine: Optional[object] = None,
        hybrid_prepass_engine: Optional[object] = None,
        *,
        page_concurrency: int = 1,
    ):
        """初始化 OCR 管线。

        Args:
            engine: OcrEngine 实现。如果为 None，使用 FakeOcrEngine。
        """
        self._engine = engine or FakeOcrEngine()
        self._hybrid_prepass_engine = hybrid_prepass_engine
        self._proof_crop_service = ProofCropService()
        self._table_text_layer_service = TableTextLayerService()
        try:
            configured_concurrency = int(page_concurrency)
        except (TypeError, ValueError):
            configured_concurrency = 1
        self._page_concurrency = max(1, min(OCR_PAGE_CONCURRENCY_CAP, configured_concurrency))

    def close(self) -> None:
        engine = self._engine
        if engine is not None and hasattr(engine, "close"):
            engine.close()

    def process_project(
        self,
        project: OcrProject,
        progress_callback: Optional[Callable[[OcrProgress], None]] = None,
    ) -> OcrRunResult:
        """处理项目所有页的所有可识别块。"""
        result = OcrRunResult()
        total_pages = len(project.pages)

        try:
            if self._should_parallelize_page_hybrid(total_pages):
                return self._process_project_page_hybrid_parallel(project, progress_callback)

            for page_idx, page in enumerate(project.pages):
                self._clear_ocr_error(page)
                dispatch_plan = build_text_ocr_dispatch_plan(page)
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
                    mark_page_ocr_failed(page, f"OCR 图像读取失败：{page.display_image_path}")
                    for target in dispatch_plan.text_blocks:
                        result.failed_blocks.append(
                            (page_idx, target.block.order, f"Cannot read image: {page.display_image_path}")
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

                total_blocks = dispatch_plan.total_text_blocks

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
                        mark_page_ocr_failed(page, f"OCR 失败：{e}")
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
                    self._enrich_table_text_layer(page)
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
                        mark_page_ocr_failed(page, f"OCR 失败：{e}")
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
                    self._enrich_table_text_layer(page)
                    result.pages.append(page)
                    continue

                block_idx = 0
                page_failures: list[str] = []

                for target in dispatch_plan.text_blocks:
                    block = target.block

                    try:
                        lines = self._process_block(img, block, page, page_idx)
                        replace_block_ocr_lines(block, lines)
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
                elif page_ocr_line_count(page) == 0 and page_failures and not page_error_message(page):
                    summary = "；".join(page_failures[:3])
                    if len(page_failures) > 3:
                        summary += "；…"
                    mark_page_ocr_failed(page, f"OCR 失败：{summary}")

                self._normalize_proof_crops(
                    page,
                    page_idx=page_idx,
                    total_pages=total_pages,
                    progress_callback=progress_callback,
                )
                self._enrich_table_text_layer(page)
                result.pages.append(page)
        finally:
            self.close()

        return result

    def _should_parallelize_page_hybrid(self, total_pages: int) -> bool:
        if total_pages <= 1 or self._page_concurrency <= 1:
            return False
        if not self._prefers_page_hybrid_blocks():
            return False
        # Keep shared prepass engines serial until they have an explicit thread-safety contract.
        return self._hybrid_prepass_engine is None

    def _process_project_page_hybrid_parallel(
        self,
        project: OcrProject,
        progress_callback: Optional[Callable[[OcrProgress], None]],
    ) -> OcrRunResult:
        total_pages = len(project.pages)
        result = OcrRunResult(pages=[None] * total_pages)  # type: ignore[list-item]
        completed_pages = 0
        completed_lock = Lock()
        emit_lock = Lock()

        def current_completed_pages() -> int:
            with completed_lock:
                return completed_pages

        def emit(progress: OcrProgress) -> None:
            if progress_callback is None:
                return
            with emit_lock:
                progress_callback(progress)

        workers = min(self._page_concurrency, total_pages)
        logger.info("OCR page-hybrid parallel start: pages=%d workers=%d", total_pages, workers)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="charocr-page") as executor:
            futures = {
                executor.submit(
                    self._process_page_hybrid_work,
                    page_idx,
                    page,
                    total_pages,
                    current_completed_pages,
                    emit,
                ): page_idx
                for page_idx, page in enumerate(project.pages)
            }
            for future in as_completed(futures):
                page_idx = futures[future]
                try:
                    work = future.result()
                except Exception as exc:
                    page = project.pages[page_idx]
                    logger.error("Page hybrid OCR worker crashed: page=%d: %s", page_idx, exc)
                    mark_page_ocr_failed(page, f"OCR 失败：{exc}")
                    work = PageOcrRunResult(
                        page_idx=page_idx,
                        page=page,
                        failed_blocks=[(page_idx, -1, str(exc))],
                        completion_message=f"OCR 失败：第 {page_idx + 1}/{total_pages} 页：{exc}",
                    )

                result.pages[work.page_idx] = work.page
                result.failed_blocks.extend(work.failed_blocks)
                with completed_lock:
                    completed_pages += 1
                    done = completed_pages
                emit(OcrProgress(
                    current_page=work.page_idx + 1,
                    total_pages=total_pages,
                    current_block=max(1, work.total_blocks),
                    total_blocks=max(1, work.total_blocks),
                    completed_pages=done,
                    message=work.completion_message
                    or f"OCR 识别中… 第 {work.page_idx + 1}/{total_pages} 页，CharOCR 已写回版面块",
                ))

        logger.info("OCR page-hybrid parallel done: pages=%d workers=%d", total_pages, workers)
        return result

    def _process_page_hybrid_work(
        self,
        page_idx: int,
        page: Page,
        total_pages: int,
        completed_pages_getter: Callable[[], int],
        emit: Callable[[OcrProgress], None],
    ) -> PageOcrRunResult:
        self._clear_ocr_error(page)
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        emit(OcrProgress(
            current_page=page_idx + 1,
            total_pages=total_pages,
            current_block=0,
            total_blocks=0,
            completed_pages=completed_pages_getter(),
            message=f"CharOCR 准备中… 第 {page_idx + 1}/{total_pages} 页",
        ))

        img = cv2.imread(page.display_image_path)
        if img is None:
            logger.warning("Cannot read image: %s", page.display_image_path)
            mark_page_ocr_failed(page, f"OCR 图像读取失败：{page.display_image_path}")
            failed = [
                (page_idx, target.block.order, f"Cannot read image: {page.display_image_path}")
                for target in dispatch_plan.text_blocks
            ]
            return PageOcrRunResult(
                page_idx=page_idx,
                page=page,
                failed_blocks=failed,
                completion_message=f"OCR 跳过：第 {page_idx + 1}/{total_pages} 页图像读取失败",
            )

        total_blocks = dispatch_plan.total_text_blocks

        def emit_hybrid_progress(current: int, total: int, message: str) -> None:
            stage_message = message.replace("Hanwang micro-recblock", "CharOCR")
            emit(OcrProgress(
                current_page=page_idx + 1,
                total_pages=total_pages,
                current_block=current,
                total_blocks=total,
                completed_pages=completed_pages_getter(),
                message=f"第 {page_idx + 1}/{total_pages} 页：{stage_message}",
            ))

        failed: list[tuple[int, int, str]] = []
        try:
            self._process_page_with_hybrid_blocks(
                img,
                page,
                page_idx=page_idx,
                progress_callback=emit_hybrid_progress,
            )
        except Exception as exc:
            logger.error("Page hybrid OCR failed: page=%d: %s", page_idx, exc)
            mark_page_ocr_failed(page, f"OCR 失败：{exc}")
            failed.append((page_idx, -1, str(exc)))

        stats = self._proof_crop_service.normalize_pages([page])
        self._enrich_table_text_layer(page)
        fallback_total = stats.fallback_chars + stats.unavailable_chars
        if fallback_total > 0:
            message = (
                f"警告：第 {page_idx + 1}/{total_pages} 页触发 proof fallback，"
                f"{stats.fallback_lines} 行/{fallback_total} 字使用估算或不可用字框"
            )
            logger.warning(message)
            emit(OcrProgress(
                current_page=page_idx + 1,
                total_pages=total_pages,
                current_block=0,
                total_blocks=0,
                completed_pages=completed_pages_getter(),
                message=message,
            ))

        if failed:
            completion = f"OCR 失败：第 {page_idx + 1}/{total_pages} 页，CharOCR 处理失败"
        else:
            completion = f"OCR 识别中… 第 {page_idx + 1}/{total_pages} 页，CharOCR 已写回版面块"
        return PageOcrRunResult(
            page_idx=page_idx,
            page=page,
            total_blocks=total_blocks,
            failed_blocks=failed,
            completion_message=completion,
        )

    def _enrich_table_text_layer(self, page: Page) -> None:
        try:
            updated = self._table_text_layer_service.enrich_page(page)
        except Exception as exc:
            logger.warning("Table text-layer enrichment failed: page=%s: %s", page.page_number, exc)
            return
        if updated:
            logger.info("Table text-layer enrichment complete: page=%s tables=%d", page.page_number, updated)

    @staticmethod
    def _clear_ocr_error(page: Page) -> None:
        """Clear stale OCR-owned errors before retrying OCR on a page."""
        if is_ocr_error_message(page_error_message(page)):
            clear_page_error_message(page)

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
        append_layout_block_note_once(block, note)

    def _process_page_with_page_ocr(
        self,
        img: np.ndarray,
        page: Page,
        page_idx: int,
        engine: object | None = None,
        *,
        mark_page_line_hints: bool = False,
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
        if mark_page_line_hints:
            for line in lines:
                mark_ppocr_page_line_hint(line)
            self._assign_page_ocr_line_hints_to_blocks(page, lines)
        else:
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
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        total_text_blocks = max(1, dispatch_plan.total_text_blocks)
        if self._has_reusable_page_line_hints(page):
            if progress_callback:
                progress_callback(
                    0,
                    total_text_blocks,
                    "PP-OCRv5 page-line prepass skipped: "
                    f"{self._reusable_page_line_hint_summary(page)}",
                )
        else:
            prepass_engine = self._hybrid_page_ocr_prepass_engine()
            if not bool(getattr(prepass_engine, "prefer_page_ocr", False)):
                raise RuntimeError("Hanwang hybrid OCR requires a PP-OCRv5 page-line prepass engine")
            if progress_callback:
                progress_callback(0, total_text_blocks, "PP-OCRv5 page-line prepass 中…")
            self._process_page_with_page_ocr(
                img,
                page,
                page_idx,
                engine=prepass_engine,
                mark_page_line_hints=True,
            )
            if progress_callback:
                line_count = sum(block_ocr_line_count(block) for block in dispatch_plan.text_block_models)
                progress_callback(0, total_text_blocks, f"PP-OCRv5 page-line prepass complete: {line_count} lines")
        if not supports_page_block_ocr(self._engine):
            raise RuntimeError("Configured OCR engine does not support page-block OCR")
        self._engine.recognize_page_blocks(
            img,
            page,
            progress_callback=progress_callback,
        )

    def _has_reusable_page_line_hints(self, page: Page) -> bool:
        """Return whether routing already has trusted PP-OCRv5 line geometry.

        CharOCR/Hanwang result lines are not reusable as page-line hints: they
        are produced after formula/table slicing and may be narrower than the
        original PP-OCRv5 row. Reusing them would feed derived geometry back
        into the next OCR run and can make formula regions affect text crops.
        """
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        if not dispatch_plan.text_blocks:
            return True
        for target in dispatch_plan.text_blocks:
            block = target.block
            if is_ocr_text_invalidated(block):
                return False
            if self._has_marked_page_line_hints(block):
                continue
            return False
        return True

    @staticmethod
    def _has_marked_page_line_hints(block: Block) -> bool:
        return any(
            line_ocr_bbox(line) is not None
            and line_ocr_bbox(line).area > 0
            and is_ppocr_page_line_hint(line)
            for line in block_ocr_lines(block)
        )

    def _reusable_page_line_hint_summary(self, page: Page) -> str:
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        marked_lines = sum(
            1
            for target in dispatch_plan.text_blocks
            for block in (target.block,)
            for line in block_ocr_lines(block)
            if line_ocr_bbox(line) is not None
            and line_ocr_bbox(line).area > 0
            and is_ppocr_page_line_hint(line)
        )
        parts: list[str] = []
        if marked_lines:
            parts.append(f"reused {marked_lines} PP-OCRv5 line hints")
        return "; ".join(parts) if parts else "reused trusted line geometry"

    def _normalize_engine_lines(
        self,
        lines: List[Line],
        seam: CropCoordinateSeam,
        bbox_space: str,
    ) -> None:
        for line in lines:
            set_ocr_line_bbox(line, seam.to_page_bbox(line_ocr_bbox(line), source_space=bbox_space))
            for char in line_ocr_chars(line):
                if char.bbox is None or char.bbox.area <= 0:
                    continue
                set_ocr_char_bbox(char, seam.to_page_bbox(char.bbox, source_space=bbox_space))
            ensure_line_text_contract(line)

    def _assign_page_ocr_lines_to_blocks(self, page: Page, lines: list[Line]) -> None:
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        containers = list(dispatch_plan.text_blocks)
        blockers = list(dispatch_plan.blocked_blocks)
        for target in containers:
            clear_block_ocr_lines(target.block)
        unmatched: list[Line] = []
        for line in sorted(lines, key=lambda item: (line_ocr_bbox(item).y, line_ocr_bbox(item).x)):
            blocker = select_container_block_for_line(line, blockers)
            if blocker is not None:
                continue
            target = select_container_block_for_line(line, containers)
            if target is None:
                unmatched.append(line)
            else:
                append_block_ocr_line(target.block, line)

        if not unmatched:
            return
        merged = self._merge_line_bboxes(unmatched)
        if merged is None:
            return
        synthetic = Block(
            block_type=BlockType.TEXT,
            bbox=merged,
            order=(max((view.order for view in iter_page_layout_block_views(page)), default=-1) + 1),
            note="PP-OCRv5 unmatched proof lines",
        )
        replace_block_ocr_lines(synthetic, unmatched)
        append_page_layout_block(page, synthetic)

    def _assign_page_ocr_line_hints_to_blocks(self, page: Page, lines: list[Line]) -> None:
        """Assign PP-OCRv5 geometry hints to text containers for Hanwang routing.

        These lines are not proof text.  They are only physical row geometry used
        to split text slices before Hanwang OCR, so inline formulas/tables must
        not drop the whole row here.  Structural regions are carved out later by
        the route builder.
        """
        dispatch_plan = build_text_ocr_dispatch_plan(page)
        containers = list(dispatch_plan.text_blocks)
        for target in containers:
            clear_block_ocr_lines(target.block)

        for line in sorted(lines, key=lambda item: (line_ocr_bbox(item).y, line_ocr_bbox(item).x)):
            target = select_container_block_for_line(line, containers)
            if target is not None:
                append_block_ocr_line(target.block, line)

    def assign_page_ocr_lines_to_blocks(self, page: Page, lines: list[Line]) -> None:
        """Public wrapper used when PP-OCRv5 proof lines finish before layout."""
        self._assign_page_ocr_lines_to_blocks(page, lines)

    def _merge_line_bboxes(self, lines: list[Line]) -> BBox | None:
        return merge_bboxes([line_ocr_bbox(line) for line in lines])

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
        )
        replace_page_layout_blocks(page, [block])
        lines = self._process_block(img, block, page, 0)
        replace_block_ocr_lines(block, lines)
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
        set_layout_block_bbox(block, bb)
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
