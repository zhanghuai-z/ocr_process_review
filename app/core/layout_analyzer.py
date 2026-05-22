"""Layout analysis wrapper. Supports local / api modes.

主链的 layout 角色已全面切换到 PaddleOCR-VL-1.5（替代 PP-StructureV3）：
  resolve_api_endpoint_for_role(api_url, role="layout")
    -> https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing  (官方预置)

VL 响应与 Structure 在这里采用的抽取路径兑现上兼容：
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
import base64
import json
from pathlib import Path
from typing import Iterable, List

from PySide6.QtCore import QThread, Signal

from app.core.api_profiles import (
    get_api_request_options,
    infer_api_model_profile_from_endpoint,
    resolve_api_endpoint_for_role,
)
from app.core.api_profiles import FIXED_LAYOUT_PROFILE, get_api_request_options, infer_api_model_profile_from_endpoint, resolve_api_endpoint_for_role
from app.core.bbox_extraction import BBOX_FIELD_KEYS, bbox_from_variant, raw_bbox_max_from_variant
from app.core.bbox_utils import sanitize_xyxy_bbox, scale_bbox
from app.core.logging import get_logger
from app.core.paddle_response import (
    iter_layout_records_from_item,
    iter_ocr_records_from_item,
    pruned_result,
    result_dict,
    result_items,
)
from app.models import Block, BlockType, Page

logger = get_logger(__name__)
LOCAL_LAYOUT_CANVAS_W = 800
LOCAL_LAYOUT_CANVAS_H = 608
LAYOUT_API_TIMEOUT_FLOOR = 180
LAYOUT_API_JPEG_QUALITY = 85


class LayoutWorker(QThread):
    """版面分析 Worker 线程，避免阻塞 UI。"""
    page_done = Signal(int, int)   # (current_index, total)
    all_done  = Signal(list)       # List[Page]
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages

    def run(self) -> None:
        analyzer = LayoutAnalyzer()
        total = len(self._pages)
        fatal_errors: list[str] = []
        for i, page in enumerate(self._pages):
            try:
                analyzer.analyze(page)
                page.error_message = ""
            except Exception as e:
                logger.error("Layout analysis failed for page %s: %s", page.display_image_path, e)
                page.blocks = []
                page.error_message = f"版面分析失败：{e}"
                fatal_errors.append(f"第 {page.page_number} 页：{e}")
            finally:
                self.page_done.emit(i, total)
        if len(fatal_errors) == total and total > 0:
            self.error.emit("所有页面版面分析失败：\n" + "\n".join(fatal_errors[:5]))
            return
        self.all_done.emit(self._pages)


class LayoutAnalyzer:

    def __init__(self) -> None:
        self._engine = None
        self._hanwang_layout_engine = None

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

    def _extract_label_from_record(self, record: dict, default: str = "unknown") -> str:
        for key in (
            "label", "type", "category", "category_name", "cls_name",
            "class_name", "block_label", "block_type", "layout_label",
        ):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    def _extract_score_from_record(self, record: dict) -> float | None:
        for key in (
            "score", "confidence", "layout_score", "cls_score",
            "block_score", "prob", "probability",
        ):
            value = record.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_bbox_from_record(self, record: dict, page: Page):
        for key in BBOX_FIELD_KEYS:
            if key in record:
                bbox = self._extract_bbox_from_coordinate(record.get(key), page)
                if bbox and bbox.area > 0:
                    return bbox
        return None

    def _raw_bbox_max_from_record(self, record: dict) -> tuple[float, float] | None:
        for key in BBOX_FIELD_KEYS:
            if key not in record:
                continue
            max_xy = self._raw_bbox_max_from_coordinate(record.get(key))
            if max_xy is not None:
                return max_xy
        return None

    def _raw_bbox_max_from_coordinate(self, coord: object) -> tuple[float, float] | None:
        """Extract raw max x/y before scaling or clamping."""
        return raw_bbox_max_from_variant(coord)

    def _raw_bbox_max_from_item(self, item: dict) -> tuple[float, float] | None:
        max_x = 0.0
        max_y = 0.0
        found = False
        for record in self._iter_layout_records_from_item(item) + self._iter_ocr_records_from_item(item):
            max_xy = self._raw_bbox_max_from_record(record)
            if max_xy is None:
                continue
            max_x = max(max_x, max_xy[0])
            max_y = max(max_y, max_xy[1])
            found = True
        return (max_x, max_y) if found else None

    def _iter_layout_records_from_item(self, item: dict) -> List[dict]:
        return iter_layout_records_from_item(item)

    def _iter_ocr_records_from_item(self, item: dict) -> List[dict]:
        return iter_ocr_records_from_item(item)

    def _append_api_block(
        self,
        *,
        page: Page,
        record: dict,
        scale_x: float,
        scale_y: float,
        order: int,
        seen: set[tuple],
        page_blocks: List[Block],
        raw_overlay_items: List[tuple[str, object]],
        default_label: str = "unknown",
    ) -> int:
        bbox = self._extract_bbox_from_record(record, page)
        if not bbox or bbox.area <= 0:
            return order

        if scale_x != 1.0 or scale_y != 1.0:
            bbox = scale_bbox(bbox, scale_x, scale_y).clamp(page.width, page.height)
            if bbox.area <= 0:
                return order

        raw_type = self._extract_label_from_record(record, default=default_label)
        signature = (raw_type, bbox.x, bbox.y, bbox.w, bbox.h)
        if signature in seen:
            return order
        seen.add(signature)

        preview = ""
        for key in ("block_content", "text", "content", "markdown"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                preview = value.strip()
                break
        score = self._extract_score_from_record(record)
        note_parts = []
        if preview:
            note_parts.append(preview[:120])
        if score is not None:
            note_parts.append(f"score={score:.3f}")
        normalized_type = raw_type.strip().lower().replace("-", "_").replace(" ", "_") if raw_type else ""
        if normalized_type in {
            "page_number",
            "number",
            "formula_number",
            "header",
            "footer",
            "footnote",
            "sidebar_text",
        }:
            note_parts.append(f"source_label={raw_type}")

        raw_overlay_items.append((raw_type, bbox))
        page_blocks.append(Block(
            block_type=BlockType.from_paddle(raw_type),
            bbox=bbox,
            order=order,
            note=" | ".join(note_parts),
        ))
        return order + 1

    def _extract_api_blocks(self, page: Page, data: dict) -> tuple[List[Block], List[tuple[str, object]]]:
        result = result_dict(data)
        layout_results = result_items(data, "layoutParsingResults")
        data_info = result.get("dataInfo") if isinstance(result, dict) else None
        page_blocks: List[Block] = []
        raw_overlay_items: List[tuple[str, object]] = []
        seen: set[tuple] = set()
        order = 0

        for item in layout_results:
            scale_x, scale_y = self._detect_api_canvas_scale(page, item, data_info)
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                logger.info(
                    "API 返回坐标空间不同于原图，采用 scale_x=%.3f scale_y=%.3f 修正 (%s)",
                    scale_x, scale_y, page.display_image_path,
                )

            for record in self._iter_layout_records_from_item(item):
                order = self._append_api_block(
                    page=page,
                    record=record,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    order=order,
                    seen=seen,
                    page_blocks=page_blocks,
                    raw_overlay_items=raw_overlay_items,
                )

        if page_blocks:
            return page_blocks, raw_overlay_items

        ocr_results = result_items(data, "ocrResults")
        for item in ocr_results:
            scale_x, scale_y = self._detect_api_canvas_scale(page, item, data_info)
            for record in self._iter_ocr_records_from_item(item):
                order = self._append_api_block(
                    page=page,
                    record=record,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    order=order,
                    seen=seen,
                    page_blocks=page_blocks,
                    raw_overlay_items=raw_overlay_items,
                    default_label="text",
                )

        return page_blocks, raw_overlay_items

    def _build_api_payload(self, file_b64: str, file_type: int, model_name: str = "") -> dict:
        payload = {"file": file_b64, "fileType": file_type}
        if model_name:
            payload["model_name"] = model_name
        return payload

    def _build_api_request_body(
        self,
        file_b64: str,
        file_type: int,
        model_name: str = "",
        *,
        profile: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict:
        body = self._build_api_payload(file_b64, file_type, model_name)
        body.update(get_api_request_options(profile, endpoint_url))
        return body

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
          3) overall_ocr_res.rec_boxes 的最大坐标外推
          4) layout_det_res.boxes 的最大坐标外推
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
        valid_blocks = [block for block in page.blocks if block.bbox.area > 0]
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
            for block in page.blocks:
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
        for block in page.blocks:
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
        page.blocks = []
        for i, item in enumerate(items):
            raw_type = item.get("type", "unknown")
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            bbox = bbox_from_variant(bbox_raw, max_w=page.width, max_h=page.height)
            if bbox is None:
                continue
            if bbox.area <= 0:
                continue
            page.blocks.append(Block(
                block_type=BlockType.from_paddle(raw_type),
                bbox=bbox,
                order=i,
            ))
        self._rescale_blocks_if_suspicious(page)
        return page

    # ── api mode ───────────────────────────────────────────────

    def _api_analyze(self, page: Page) -> Page:
        """Call the configured AiStudio model endpoint; raises on network/auth errors."""
        import cv2
        from app.core.api_http import post_json_without_env_proxy
        from app.core.ocr_config import get_config

        cfg = get_config()
        url = resolve_api_endpoint_for_role(
            cfg["api_url"],
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        # 主链 layout 已被 strong-redirect 到 VL-1.5；但配置里仍可能是旧
        # profile (pp-structurev3 / pp-ocrv5)。这里按最终 endpoint 重推
        # profile，保证 _build_api_request_body 选出正确的 request_family
        # (vl-layout vs ocr-word-box)，不会把 OCR detector/recognizer 参数误发
        # 到 VL 端点。与 Inspector 处理保持一致。
        effective_profile = (
            infer_api_model_profile_from_endpoint(url)
            or cfg.get("api_model_profile", "")
        )
        timeout = max(int(cfg["api_timeout"]), LAYOUT_API_TIMEOUT_FLOOR)
        token = cfg.get("api_token", "")
        layout_model_name = cfg.get("api_layout_model_name", "").strip()

        img = cv2.imread(page.display_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        page.height, page.width = img.shape[:2]
        ok, buf = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), LAYOUT_API_JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError(f"Cannot encode image: {page.display_image_path}")
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = post_json_without_env_proxy(
            url,
            json=self._build_api_request_body(
                file_b64,
                1,
                layout_model_name,
                profile=effective_profile,
                endpoint_url=url,
            ),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._write_api_debug_response(page, data)

        page.blocks, raw_overlay_items = self._extract_api_blocks(page, data)
        self._write_bbox_overlay(
            page,
            items=raw_overlay_items,
            suffix=".layout-api-raw.png",
            color=(0, 165, 255),
        )
        # 注意：不再调用 _rescale_blocks_if_suspicious()——
        # API 模式下坐标空间已在提取阶段通过 _detect_api_canvas_scale 修正，
        # 再走通用启发式只会引入二次缩放。
        self._write_bbox_overlay(
            page,
            items=[(block.block_type.value, block.bbox) for block in page.blocks],
            suffix=".layout-app-overlay.png",
            color=(80, 220, 80),
        )
        return page

    def _hanwang_analyze(self, page: Page) -> Page:
        """调汉王 doc_seg.dll 跳过 PaddleOCR。

        bbox 以 page 坐标返回，与 _local_analyze 一致。
        """
        import cv2

        from app.engines.hanwang_layout_engine import HanwangLayoutEngine

        img = cv2.imread(page.display_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        page.height, page.width = img.shape[:2]
        if self._hanwang_layout_engine is None:
            self._hanwang_layout_engine = HanwangLayoutEngine()
        blocks = self._hanwang_layout_engine.analyze(page.display_image_path)
        for i, b in enumerate(blocks):
            b.order = i
        page.blocks = blocks
        self._rescale_blocks_if_suspicious(page)
        return page

    # ── common interface ─────────────────────────────

    def analyze(self, page: Page) -> Page:
        from app.core.ocr_config import get_config
        mode = get_config()["mode"]
        if mode == "hanwang":
            return self._hanwang_analyze(page)
        if mode == "api":
            return self._api_analyze(page)
        return self._local_analyze(page)

    def analyze_pages(self, pages: List[Page]) -> List[Page]:
        return [self.analyze(page) for page in pages]
