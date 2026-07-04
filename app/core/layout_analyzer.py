"""Layout analysis wrapper. Supports local / api modes.

主链的 layout 角色固定使用 PaddleOCR-VL-1.6：
  resolve_api_endpoint_for_role(api_url, role="layout")
    -> https://paddleocr.aistudio-app.com/api/v2/ocr/jobs  (官方预置)

VL1.6 响应在这里采用固定抽取路径：
  result.layoutParsingResults[0].prunedResult:
    layout_det_res.boxes[]:                    # 版面 bbox（供主程序块提取）
      label:      str  (e.g. "text", "paragraph_title", "display_formula",
                        "inline_formula", "formula_number", "footnote",
                        "header", "number", ...)
      coordinate: [x1, y1, x2, y2]
    parsing_res_list[]:                        # 顶层阅读顺序 / 块内容
      block_label / block_bbox / block_content
    markdown.text:                             # 供下游导出 Markdown / LaTeX

PP-OCRv5 仍然负责 line/word/char bbox，VL 只接管版面块。
"""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable, List

from PySide6.QtCore import QThread, Signal

from app.core.api_profiles import (
    FIXED_LAYOUT_PROFILE,
    resolve_api_endpoint_for_role,
)
from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.bbox_extraction import bbox_from_variant
from app.core.bbox_utils import sanitize_xyxy_bbox, scale_bbox
from app.core.logging import get_logger
from app.core.ocr_dispatch_policy import default_ocr_policy_for_block
from app.core.paddle_layout_schema import (
    normalize_paddle_layout_record,
    paddle_record_label,
    raw_bbox_max_from_record,
    route_subblock_payload,
)
from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
from app.core.paddle_labels import (
    is_hanwang_skip_label,
    normalize_paddle_label,
)
from app.core.paddle_response import (
    layout_geometry_records_from_item,
    parsing_records_from_item,
    pruned_result,
    result_dict,
    result_items,
)
from app.core.raw_ocr_artifact import set_paddle_raw_layout_records
from app.core.paddle_v16_client import (
    PaddleV16LayoutClient,
    PaddleV16RequestCancelled,
    build_paddle_v16_optional_payload,
    is_paddle_v16_endpoint,
)
from app.models import Block, BlockOrigin, BlockSource, BlockType, Page
from app.models.layout_block_state import set_layout_block_ocr_policy
from app.models.layout_projection import (
    append_page_layout_block,
    page_layout_blocks,
    replace_page_layout_blocks,
)
from app.models.page_state import clear_page_error_message, mark_page_layout_failed
from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.services.layout_snapshot import (
    layout_snapshot_from_normalized_artifact,
    project_layout_snapshot_to_blocks,
)

logger = get_logger(__name__)
LOCAL_LAYOUT_CANVAS_W = 800
LOCAL_LAYOUT_CANVAS_H = 608
LAYOUT_API_TIMEOUT_FLOOR = 180
LAYOUT_API_REQUEST_TIMEOUT_CAP = 30
LAYOUT_API_CONCURRENCY_CAP = 10


def _truthy_config(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _display_image_path(page: Page) -> Path:
    image_path = Path(page.display_image_path)
    if image_path.is_file():
        return image_path
    normalized = Path(str(page.display_image_path).replace("\\", "/"))
    return normalized


def _layout_block_origin(
    *,
    source_engine: str,
    source_run_id: str,
    source_label: str,
    bbox,
    block_type: BlockType,
    confidence: float | None = None,
    raw_index: int | None = None,
) -> BlockOrigin:
    return BlockOrigin(
        created_by=BlockSource.AUTO_LAYOUT.value,
        source_engine=source_engine,
        source_run_id=source_run_id,
        source_label=source_label,
        source_confidence=confidence,
        original_bbox=bbox,
        original_kind=block_type,
        raw_index=raw_index,
    )


def _layout_worker_max_workers(total_pages: int) -> int:
    if total_pages <= 1:
        return 1
    try:
        from app.core.app_config import get_config

        cfg = get_config()
    except Exception:
        return 1
    mode = str(cfg.get("mode", "local") or "local").lower()
    if mode not in {"api", "hanwang"}:
        return 1
    try:
        configured = int(cfg.get("layout_concurrency", 2))
    except (TypeError, ValueError):
        configured = 2
    return max(1, min(total_pages, LAYOUT_API_CONCURRENCY_CAP, configured))


class LayoutWorker(QThread):
    """版面分析 Worker 线程，避免阻塞 UI。"""
    page_done = Signal(int, int)   # (completed_index, total)
    all_done  = Signal(list)       # List[Page]
    error     = Signal(str)
    cancelled = Signal()
    stage_update = Signal(int, int, str)  # page_index, total, message

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages
        self._batch_id = f"ocr-process-layout-{int(time.time() * 1000)}"
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True
        self.requestInterruption()

    def _is_cancelled(self) -> bool:
        return self._cancel_requested or self.isInterruptionRequested()

    def _emit_stage(self, index: int, message: str) -> None:
        if not self._is_cancelled():
            self.stage_update.emit(index, len(self._pages), message)

    def _analyze_page(self, index: int, page: Page) -> tuple[int, str | None]:
        if self._is_cancelled():
            return index, None
        analyzer = LayoutAnalyzer(
            layout_batch_id=self._batch_id,
            cancel_callback=self._is_cancelled,
            status_callback=lambda message, page_index=index: self._emit_stage(page_index, message),
        )
        try:
            analyzer.analyze(page)
            if self._is_cancelled():
                return index, None
            clear_page_error_message(page)
            return index, None
        except PaddleV16RequestCancelled:
            return index, None
        except Exception as e:
            if self._is_cancelled():
                return index, None
            logger.error("Layout analysis failed for page %s: %s", page.display_image_path, e)
            replace_page_layout_blocks(page, [])
            mark_page_layout_failed(page, f"版面分析失败：{e}")
            return index, f"第 {page.page_number} 页：{e}"

    def run(self) -> None:
        total = len(self._pages)
        fatal_errors: list[tuple[int, str]] = []
        completed = 0
        max_workers = _layout_worker_max_workers(total)

        if max_workers <= 1:
            for i, page in enumerate(self._pages):
                if self._is_cancelled():
                    self.cancelled.emit()
                    return
                page_idx, error_message = self._analyze_page(i, page)
                if self._is_cancelled():
                    self.cancelled.emit()
                    return
                if error_message:
                    fatal_errors.append((page_idx, error_message))
                completed += 1
                self.page_done.emit(completed - 1, total)
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="layout-api")
            executor_shutdown = False
            futures = {
                executor.submit(self._analyze_page, i, page): i
                for i, page in enumerate(self._pages)
            }
            pending = set(futures)
            try:
                while pending:
                    if self._is_cancelled():
                        for future in pending:
                            future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor_shutdown = True
                        self.cancelled.emit()
                        return
                    done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in done:
                        page_idx = futures[future]
                        try:
                            page_idx, error_message = future.result()
                        except PaddleV16RequestCancelled:
                            error_message = None
                        except Exception as exc:
                            if self._is_cancelled():
                                error_message = None
                            else:
                                page = self._pages[page_idx]
                                logger.error("Layout analysis failed for page %s: %s", page.display_image_path, exc)
                                replace_page_layout_blocks(page, [])
                                mark_page_layout_failed(page, f"版面分析失败：{exc}")
                                error_message = f"第 {page.page_number} 页：{exc}"
                        if self._is_cancelled():
                            for pending_future in pending:
                                pending_future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            executor_shutdown = True
                            self.cancelled.emit()
                            return
                        if error_message:
                            fatal_errors.append((page_idx, error_message))
                        completed += 1
                        self.page_done.emit(completed - 1, total)
            finally:
                if not executor_shutdown:
                    if self._is_cancelled():
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True, cancel_futures=True)

        if len(fatal_errors) == total and total > 0:
            messages = [message for _idx, message in sorted(fatal_errors, key=lambda item: item[0])]
            self.error.emit("所有页面版面分析失败：\n" + "\n".join(messages[:5]))
            return
        self.all_done.emit(self._pages)


class LayoutAnalyzer:

    def __init__(
        self,
        *,
        layout_batch_id: str = "",
        cancel_callback: Callable[[], bool] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = None
        self._layout_batch_id = layout_batch_id
        self._cancel_callback = cancel_callback
        self._status_callback = status_callback

    def _raise_if_cancelled(self) -> None:
        if self._cancel_callback is not None and self._cancel_callback():
            raise PaddleV16RequestCancelled("Paddle layout analysis cancelled")

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    # ── local mode ─────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            logger.info(
                "Initializing PaddleOCR layout engine: lang=ch, "
                "use_angle_cls=False, layout=True, table=False, ocr=False, "
                "models=package defaults"
            )
            self._engine = PaddleOCR(
                use_angle_cls=False,
                lang="ch",
                show_log=False,
                layout=True,
                table=False,
                ocr=False,
            )
        return self._engine

    def _unwrap_layout_items(self, result) -> List[dict]:
        """兼容不同 Paddle 结果包装结构。"""
        if result is None:
            return []
        if isinstance(result, dict):
            if isinstance(result.get("layout"), list):
                return result["layout"]
            if isinstance(result.get("result"), list):
                return result["result"]
            return [result]
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                wrapped = result[0]
                if isinstance(wrapped.get("layout"), list):
                    return wrapped["layout"]
                if isinstance(wrapped.get("result"), list):
                    return wrapped["result"]
            return [item for item in result if isinstance(item, dict)]
        return []

    def _extract_bbox_from_coordinate(self, coord, page: Page):
        """兼容 API 返回的 xyxy / 四点坐标 / 扁平 polygon / xywh dict。"""
        return bbox_from_variant(coord, max_w=page.width, max_h=page.height)

    def _raw_bbox_max_from_item(self, item: dict) -> tuple[float, float] | None:
        max_x = 0.0
        max_y = 0.0
        found = False
        for record in parsing_records_from_item(item) + layout_geometry_records_from_item(item):
            max_xy = raw_bbox_max_from_record(record)
            if max_xy is None:
                continue
            max_x = max(max_x, max_xy[0])
            max_y = max(max_y, max_xy[1])
            found = True
        return (max_x, max_y) if found else None

    def _append_overlay_record(
        self,
        *,
        page: Page,
        record: dict,
        scale_x: float,
        scale_y: float,
        raw_overlay_items: List[tuple[str, object]],
    ) -> None:
        normalized = normalize_paddle_layout_record(
            record,
            page_width=page.width,
            page_height=page.height,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if normalized is None:
            return
        raw_overlay_items.append((normalized.label, normalized.bbox))

    def _record_bbox_in_page_space(
        self,
        record: dict,
        page: Page,
        scale_x: float,
        scale_y: float,
    ):
        normalized = normalize_paddle_layout_record(
            record,
            page_width=page.width,
            page_height=page.height,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return normalized.bbox if normalized is not None else None

    def _page_space_artifact_record(
        self,
        *,
        page: Page,
        record: dict,
        scale_x: float,
        scale_y: float,
    ) -> dict | None:
        """Return a layout artifact record whose bbox is in page image space."""
        normalized = normalize_paddle_layout_record(
            record,
            page_width=page.width,
            page_height=page.height,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if normalized is None:
            return None
        payload = dict(record)
        payload["block_label"] = normalized.label
        payload["block_bbox"] = list(normalized.bbox.to_xyxy())
        if normalized.text:
            payload["block_content"] = normalized.text
        if normalized.score is not None:
            payload["score"] = normalized.score
        return payload

    def _attach_route_subblocks(
        self,
        *,
        page: Page,
        parsing_records: list[dict],
        geometry_records: list[dict],
        scale_x: float,
        scale_y: float,
    ) -> None:
        route_records = []
        for record in geometry_records:
            normalized = normalize_paddle_layout_record(
                record,
                page_width=page.width,
                page_height=page.height,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if normalized is None or not is_hanwang_skip_label(normalized.label):
                continue
            route_records.append(normalized)
        if not route_records:
            return

        parent_entries: list[tuple[dict, object]] = []
        subblocks_by_parent: dict[int, list[dict]] = {}
        for parent in parsing_records:
            parent_label = paddle_record_label(parent)
            if is_hanwang_skip_label(parent_label):
                continue
            parent_bbox = self._record_bbox_in_page_space(parent, page, scale_x, scale_y)
            if parent_bbox is None:
                continue
            parent_entries.append((parent, parent_bbox))
            subblocks_by_parent[id(parent)] = []

        for child in route_records:
            bbox = child.bbox
            best_parent: dict | None = None
            best_score = 0.0
            for parent, parent_bbox in parent_entries:
                x1 = max(parent_bbox.x1, bbox.x1)
                y1 = max(parent_bbox.y1, bbox.y1)
                x2 = min(parent_bbox.x2, bbox.x2)
                y2 = min(parent_bbox.y2, bbox.y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                inter_area = (x2 - x1) * (y2 - y1)
                center_inside = (
                    parent_bbox.x1 <= (bbox.x1 + bbox.x2) / 2 <= parent_bbox.x2
                    and parent_bbox.y1 <= (bbox.y1 + bbox.y2) / 2 <= parent_bbox.y2
                )
                if not center_inside and inter_area < bbox.area * 0.5:
                    continue
                score = (inter_area / max(1, bbox.area)) + (0.25 if center_inside else 0.0)
                if score > best_score:
                    best_score = score
                    best_parent = parent
            if best_parent is not None:
                subblocks_by_parent[id(best_parent)].append(route_subblock_payload(child))

        for parent, _parent_bbox in parent_entries:
            subblocks = subblocks_by_parent.get(id(parent), [])
            if subblocks:
                parent[ROUTE_SUBBLOCKS_FIELD] = subblocks

    def _extract_api_blocks(self, page: Page, data: dict) -> tuple[List[Block], List[tuple[str, object]]]:
        result = result_dict(data)
        layout_results = result_items(data, "layoutParsingResults")
        data_info = result.get("dataInfo") if isinstance(result, dict) else None
        raw_overlay_items: List[tuple[str, object]] = []
        artifact_records: list[dict] = []

        for item in layout_results:
            parsing_records = parsing_records_from_item(item)
            scale_x, scale_y = self._detect_api_canvas_scale(page, item, data_info)
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                logger.info(
                    "API 返回坐标空间不同于原图，采用 scale_x=%.3f scale_y=%.3f 修正 (%s)",
                    scale_x, scale_y, page.display_image_path,
                )

            geometry_records = layout_geometry_records_from_item(item)
            if parsing_records:
                self._attach_route_subblocks(
                    page=page,
                    parsing_records=parsing_records,
                    geometry_records=geometry_records,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            artifact_records.extend(
                record
                for record in (
                    self._page_space_artifact_record(
                        page=page,
                        record=parsing_record,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                    for parsing_record in parsing_records
                )
                if record is not None
            )

            for record in parsing_records:
                self._append_overlay_record(
                    page=page,
                    record=record,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    raw_overlay_items=raw_overlay_items,
                )
            for record in geometry_records:
                self._append_overlay_record(
                    page=page,
                    record=record,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    raw_overlay_items=raw_overlay_items,
                )

        set_paddle_raw_layout_records(page, artifact_records, run_id=self._layout_batch_id)
        snapshot = layout_snapshot_from_normalized_artifact(
            normalized_layout_artifact_from_page(page),
            source_run_id=self._layout_batch_id,
        )
        page_blocks = project_layout_snapshot_to_blocks(snapshot)
        return page_blocks, raw_overlay_items

    def _shape_from_data_info(self, data_info: dict | None) -> tuple[float, float] | None:
        if not isinstance(data_info, dict):
            return None
        try:
            width = float(data_info.get("width"))
            height = float(data_info.get("height"))
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _scale_from_api_shape(
        self,
        page: Page,
        api_w: float,
        api_h: float,
        raw_max: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if raw_max is not None:
            max_x, max_y = raw_max
            metadata_contradicted = max_x > api_w * 1.05 or max_y > api_h * 1.05
            looks_like_page_space = max_x <= page.width * 1.05 and max_y <= page.height * 1.05
            if metadata_contradicted and looks_like_page_space:
                return 1.0, 1.0
        return page.width / api_w, page.height / api_h

    def _detect_api_canvas_scale(
        self, page: Page, item: dict, data_info: dict | None = None
    ) -> tuple[float, float]:
        """根据 API 返回内容反推坐标空间，返回 (scale_x, scale_y)。

        优先级：
          1) result.dataInfo.width/height（AIStudio 上传图像尺寸）
          2) prunedResult.input_img_shape / doc_preprocessor_res 实际尺寸（若有）
          3) parsing_res_list / layout_det_res.boxes 的最大坐标外推
          5) (1.0, 1.0)
        """
        pruned = pruned_result(item)
        raw_max = self._raw_bbox_max_from_item(item)

        data_shape = self._shape_from_data_info(data_info)
        if data_shape is not None and page.width > 0 and page.height > 0:
            api_w, api_h = data_shape
            return self._scale_from_api_shape(page, api_w, api_h, raw_max)

        # (2) 显式的预处理输出尺寸
        for key in ("input_img_shape", "img_shape", "image_shape"):
            shape = pruned.get(key)
            if (
                isinstance(shape, (list, tuple))
                and len(shape) >= 2
                and shape[0]
                and shape[1]
                and page.width
                and page.height
            ):
                api_h, api_w = float(shape[0]), float(shape[1])
                if api_w > 0 and api_h > 0:
                    return self._scale_from_api_shape(page, api_w, api_h, raw_max)

        doc_pre = pruned.get("doc_preprocessor_res") or {}
        if isinstance(doc_pre, dict):
            for key in ("output_img_shape", "img_shape"):
                shape = doc_pre.get(key)
                if (
                    isinstance(shape, (list, tuple))
                    and len(shape) >= 2
                    and shape[0]
                    and shape[1]
                ):
                    api_h, api_w = float(shape[0]), float(shape[1])
                    if api_w > 0 and api_h > 0:
                        return self._scale_from_api_shape(page, api_w, api_h, raw_max)

        # (2)/(3) 用最大坐标外推
        if raw_max is None or page.width <= 0 or page.height <= 0:
            return 1.0, 1.0

        api_max_x, api_max_y = raw_max
        if api_max_x <= 0 or api_max_y <= 0:
            return 1.0, 1.0

        # 若坐标已经接近原图尺寸（误差 < 5%），认为坐标空间一致
        if (
            api_max_x >= page.width * 0.6
            and api_max_y >= page.height * 0.6
            and api_max_x <= page.width * 1.05
            and api_max_y <= page.height * 1.05
        ):
            return 1.0, 1.0

        # 否则用「实际能到的最大坐标 ≈ 处理画布」推回
        scale_x = page.width / api_max_x
        scale_y = page.height / api_max_y

        # 仅当 X / Y 比例接近时才认为是均匀缩放（防止把方向问题误判成缩放）
        ratio_gap = max(scale_x, scale_y) / max(min(scale_x, scale_y), 1e-6)
        if ratio_gap > 1.4:
            logger.warning(
                "API canvas scale 非均匀 (sx=%.3f sy=%.3f)，可能存在方向/裁剪问题；"
                "暂按 1.0 处理，请检查 .layout-api.json: %s",
                scale_x, scale_y, page.display_image_path,
            )
            return 1.0, 1.0
        return scale_x, scale_y

    def _write_api_debug_response(self, page: Page, data: dict) -> None:
        """把 API 原始响应落到工作图旁边，方便复盘漂移页。"""
        image_path = page.display_image_path
        if not image_path:
            return
        debug_path = Path(image_path).with_suffix(".layout-api.json")
        payload = {
            "page": {
                "display_image_path": image_path,
                "source_path": page.source_path,
                "width": page.width,
                "height": page.height,
                "page_number": page.page_number,
            },
            "response": data,
        }
        try:
            debug_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write API layout debug response: %s", exc)

    def _artifact_root_for_page(self, page: Page) -> Path:
        image_path = Path(page.display_image_path)
        if image_path.parent.name == "images" and image_path.parent.parent.name == ".cache":
            return image_path.parent.parent / "paddle_artifacts"
        return image_path.parent / ".cache" / "paddle_artifacts"

    def _write_paddle_raw_artifact(self, page: Page, data: dict) -> None:
        artifact = page.raw_layout_artifact
        if artifact is None:
            return
        job = data.get("paddle_v16", {}) if isinstance(data, dict) else {}
        run_id = str(job.get("jobId") or self._layout_batch_id or f"layout-{int(time.time() * 1000)}")
        safe_run_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
        payload = {
            "page": {
                "uid": page.uid,
                "display_image_path": page.display_image_path,
                "source_path": page.source_path,
                "width": page.width,
                "height": page.height,
                "page_number": page.page_number,
            },
            "response": data,
        }
        blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        artifact_dir = self._artifact_root_for_page(page) / page.uid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{safe_run_id}.json"
        artifact_path.write_bytes(blob)

        artifact.page_uid = page.uid
        artifact.run_id = run_id
        artifact.artifact_path = str(artifact_path)
        artifact.artifact_hash = hashlib.sha256(blob).hexdigest()

    def _write_bbox_overlay(
        self,
        page: Page,
        *,
        items: List[tuple[str, object]],
        suffix: str,
        color: tuple[int, int, int],
    ) -> None:
        import cv2

        image_path = page.display_image_path
        if not image_path:
            return
        img = cv2.imread(image_path)
        if img is None:
            return

        canvas = img.copy()
        for label, bbox in items:
            if bbox is None or getattr(bbox, "area", 0) <= 0:
                continue
            x1, y1, x2, y2 = bbox.to_xyxy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(
                    canvas,
                    str(label),
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out_path = Path(image_path).with_suffix(suffix)
        try:
            cv2.imwrite(str(out_path), canvas)
        except Exception as exc:
            logger.warning("Failed to write bbox overlay %s: %s", out_path, exc)

    def _rescale_blocks_if_suspicious(self, page: Page) -> None:
        """当所有框都像落在较小坐标系上时，做一次统一比例修正。"""
        valid_blocks = [block for block in page_layout_blocks(page) if block.bbox.area > 0]
        if len(valid_blocks) < 2 or page.width <= 0 or page.height <= 0:
            return

        max_x2 = max(block.bbox.x2 for block in valid_blocks)
        max_y2 = max(block.bbox.y2 for block in valid_blocks)
        if max_x2 <= 0 or max_y2 <= 0:
            return

        looks_like_local_model_canvas = (
            page.width > LOCAL_LAYOUT_CANVAS_W * 1.2
            and page.height > LOCAL_LAYOUT_CANVAS_H * 1.2
            and max_x2 <= LOCAL_LAYOUT_CANVAS_W * 1.05
            and max_y2 <= LOCAL_LAYOUT_CANVAS_H * 1.05
        )
        if looks_like_local_model_canvas:
            scale_x = page.width / LOCAL_LAYOUT_CANVAS_W
            scale_y = page.height / LOCAL_LAYOUT_CANVAS_H
            logger.warning(
                "Layout bbox coordinates look like local model canvas (%dx%d); "
                "rescaling for %s with scale_x=%.3f scale_y=%.3f",
                LOCAL_LAYOUT_CANVAS_W,
                LOCAL_LAYOUT_CANVAS_H,
                page.display_image_path,
                scale_x,
                scale_y,
            )
            for block in page_layout_blocks(page):
                block.bbox = scale_bbox(block.bbox, scale_x, scale_y).clamp(page.width, page.height)
            return

        scale_x = page.width / max_x2
        scale_y = page.height / max_y2
        ratio_gap = max(scale_x, scale_y) / max(min(scale_x, scale_y), 1e-6)

        suspicious_uniform_scale = (
            scale_x >= 1.2
            and scale_y >= 1.2
            and scale_x <= 6.0
            and scale_y <= 6.0
            and ratio_gap <= 1.2
        )
        if not suspicious_uniform_scale:
            return

        logger.warning(
            "Layout bbox coordinates look scaled-down; applying uniform rescale "
            "for %s: scale_x=%.3f scale_y=%.3f page=%dx%d max_bbox=(%d,%d)",
            page.display_image_path, scale_x, scale_y, page.width, page.height, max_x2, max_y2,
        )
        for block in page_layout_blocks(page):
            block.bbox = scale_bbox(block.bbox, scale_x, scale_y).clamp(page.width, page.height)

    def _local_analyze(self, page: Page) -> Page:
        import cv2

        engine = self._get_engine()
        img = cv2.imread(page.display_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        page.height, page.width = img.shape[:2]
        result = engine.predict(page.display_image_path)
        items = self._unwrap_layout_items(result)
        replace_page_layout_blocks(page, [])
        for i, item in enumerate(items):
            raw_type = item.get("type", "unknown")
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            bbox = bbox_from_variant(bbox_raw, max_w=page.width, max_h=page.height)
            if bbox is None:
                continue
            if bbox.area <= 0:
                continue
            block_type = map_paddle_label_to_block_type(raw_type)
            normalized_type = normalize_paddle_label(raw_type)
            block = Block(
                block_type=block_type,
                bbox=bbox,
                order=i,
                source_label=normalized_type,
                origin=_layout_block_origin(
                    source_engine="paddleocr-local",
                    source_run_id=self._layout_batch_id,
                    source_label=normalized_type,
                    bbox=bbox,
                    block_type=block_type,
                    raw_index=i,
                ),
            )
            set_layout_block_ocr_policy(block, default_ocr_policy_for_block(block))
            append_page_layout_block(page, block)
        self._rescale_blocks_if_suspicious(page)
        return page

    # ── api mode ───────────────────────────────────────────────

    def _api_analyze(self, page: Page) -> Page:
        """Call the configured AiStudio model endpoint; raises on network/auth errors."""
        from app.core.app_config import get_config

        self._raise_if_cancelled()
        cfg = get_config()
        url = resolve_api_endpoint_for_role(
            cfg["api_url"],
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        configured_timeout = max(1, int(cfg["api_timeout"]))
        request_timeout = min(
            max(10, configured_timeout),
            LAYOUT_API_REQUEST_TIMEOUT_CAP,
        )
        poll_timeout = max(configured_timeout, LAYOUT_API_TIMEOUT_FLOOR)
        token = cfg.get("api_token", "")

        image_path = _display_image_path(page)
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}") from exc
        if not image_bytes:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        if page.width <= 0 or page.height <= 0:
            import cv2

            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError(f"Cannot decode image dimensions: {page.display_image_path}")
            page.height, page.width = img.shape[:2]
        if not url:
            raise RuntimeError("PaddleOCR-VL-1.6 版面分析 API 地址未配置")
        if not is_paddle_v16_endpoint(url):
            raise RuntimeError(f"版面分析只支持 PaddleOCR-VL-1.6 jobs API: {url}")
        client = PaddleV16LayoutClient(
            jobs_url=url,
            token=token,
            request_timeout=request_timeout,
            poll_timeout=poll_timeout,
            network_mode=str(cfg.get("paddle_api_network_mode", "auto") or "auto"),
            cancel_callback=self._cancel_callback,
            status_callback=self._emit_status,
        )
        data = client.analyze_image_bytes(
            image_bytes,
            optional_payload=build_paddle_v16_optional_payload(),
            batch_id=self._layout_batch_id,
            filename=image_path.name or "page.png",
        )
        telemetry = (
            data.get("paddle_v16", {})
            .get("job", {})
            .get("clientTelemetry", {})
        )
        if telemetry:
            logger.info(
                "Paddle VL1.6 page %s timing: total=%.2fs submit=%.2fs wait=%.2fs download=%.2fs network=%s batch=%s",
                page.display_image_path,
                float(telemetry.get("total_seconds") or 0.0),
                float(telemetry.get("submit_seconds") or 0.0),
                float(telemetry.get("wait_seconds") or 0.0),
                float(telemetry.get("download_seconds") or 0.0),
                telemetry.get("submit_network_mode") or telemetry.get("network_mode") or "",
                telemetry.get("batch_id") or "",
            )
        blocks, raw_overlay_items = self._extract_api_blocks(page, data)
        replace_page_layout_blocks(page, blocks)
        self._write_paddle_raw_artifact(page, data)
        # 注意：不再调用 _rescale_blocks_if_suspicious()——
        # API 模式下坐标空间已在提取阶段通过 _detect_api_canvas_scale 修正，
        # 再走通用启发式只会引入二次缩放。
        if _truthy_config(cfg.get("layout_debug_artifacts", False)):
            self._write_api_debug_response(page, data)
            self._write_bbox_overlay(
                page,
                items=raw_overlay_items,
                suffix=".layout-api-raw.png",
                color=(0, 165, 255),
            )
            self._write_bbox_overlay(
                page,
                items=[(block.block_type.value, block.bbox) for block in page_layout_blocks(page)],
                suffix=".layout-app-overlay.png",
                color=(80, 220, 80),
            )
        return page

    # ── common interface ─────────────────────────────

    def analyze(self, page: Page) -> Page:
        from app.core.app_config import get_config
        mode = get_config()["mode"]
        if mode == "hanwang":
            return self._api_analyze(page)
        if mode == "api":
            return self._api_analyze(page)
        return self._local_analyze(page)

    def analyze_pages(self, pages: List[Page]) -> List[Page]:
        return [self.analyze(page) for page in pages]
