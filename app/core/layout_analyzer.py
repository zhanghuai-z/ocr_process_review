"""PP-StructureV3 layout analysis wrapper. Supports local / api modes.

AiStudio Serving API (confirmed):
  POST {api_url}/layout-parsing
  Authorization: token <api_token>
  Body: {"file": "<base64_JPEG>", "fileType": 1}
  Response:
    result.layoutParsingResults[0].prunedResult:
      layout_det_res.boxes[]:
        label:      str  (e.g. "doc_title", "paragraph", ...)
        coordinate: [x1, y1, x2, y2]  (float, page coords)
"""
from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Iterable, List

from PySide6.QtCore import QThread, Signal

from app.core.bbox_utils import sanitize_xyxy_bbox, scale_bbox
from app.core.logging import get_logger
from app.models import Block, BlockType, Page

logger = get_logger(__name__)
LOCAL_LAYOUT_CANVAS_W = 800
LOCAL_LAYOUT_CANVAS_H = 608


class LayoutWorker(QThread):
    """版面分析 Worker 线程，避免阻塞 UI。"""
    page_done = Signal(int, int)   # (current_index, total)
    all_done  = Signal(list)       # List[Page]
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages

    def run(self) -> None:
        try:
            analyzer = LayoutAnalyzer()
            total = len(self._pages)
            for i, page in enumerate(self._pages):
                analyzer.analyze(page)
                self.page_done.emit(i, total)
            self.all_done.emit(self._pages)
        except Exception as e:
            self.error.emit(str(e))


class LayoutAnalyzer:

    def __init__(self) -> None:
        self._engine = None

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
        """兼容 API 返回的 xyxy 或四点坐标。"""
        if not isinstance(coord, (list, tuple)):
            return None

        if len(coord) >= 4 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in coord[:4]):
            xs = [float(pt[0]) for pt in coord[:4]]
            ys = [float(pt[1]) for pt in coord[:4]]
            return sanitize_xyxy_bbox(
                [min(xs), min(ys), max(xs), max(ys)],
                page.width,
                page.height,
            )

        if len(coord) >= 4:
            return sanitize_xyxy_bbox(coord[:4], page.width, page.height)
        return None

    def _build_api_payload(self, file_b64: str, file_type: int, model_name: str = "") -> dict:
        payload = {"file": file_b64, "fileType": file_type}
        if model_name:
            payload["model_name"] = model_name
        return payload

    # PaddleX 服务侧默认会做方向分类 / 去畸变 / 文本行方向判断，
    # 这些都会让返回坐标落在「预处理后的图」而不是我们传入的原图，
    # 也是导致 bbox 整体偏移的根本原因。统一关闭。
    _PIPELINE_DISABLE_FLAGS = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }

    def _build_api_request_body(
        self, file_b64: str, file_type: int, model_name: str = ""
    ) -> dict:
        body = self._build_api_payload(file_b64, file_type, model_name)
        body.update(self._PIPELINE_DISABLE_FLAGS)
        return body

    def _detect_api_canvas_scale(
        self, page: Page, item: dict
    ) -> tuple[float, float]:
        """根据 API 返回内容反推坐标空间，返回 (scale_x, scale_y)。

        优先级：
          1) prunedResult.input_img_shape / doc_preprocessor_res 实际尺寸（若有）
          2) overall_ocr_res.rec_boxes 的最大坐标外推
          3) layout_det_res.boxes 的最大坐标外推
          4) (1.0, 1.0)
        """
        pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}

        # (1) 显式的预处理输出尺寸
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
                    return page.width / api_w, page.height / api_h

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
                        return page.width / api_w, page.height / api_h

        # (2)/(3) 用最大坐标外推
        ocr_res = pruned.get("overall_ocr_res") or {}
        layout_res = pruned.get("layout_det_res") or {}

        max_x_candidates: List[float] = []
        max_y_candidates: List[float] = []

        for box in ocr_res.get("rec_boxes", []) or []:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                max_x_candidates.append(float(box[2]))
                max_y_candidates.append(float(box[3]))

        for box in layout_res.get("boxes", []) or []:
            if not isinstance(box, dict):
                continue
            coord = box.get("coordinate")
            if not isinstance(coord, (list, tuple)):
                continue
            if (
                len(coord) >= 4
                and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in coord[:4])
            ):
                max_x_candidates.append(max(float(p[0]) for p in coord[:4]))
                max_y_candidates.append(max(float(p[1]) for p in coord[:4]))
            elif len(coord) >= 4:
                max_x_candidates.append(float(coord[2]))
                max_y_candidates.append(float(coord[3]))

        if not max_x_candidates or not max_y_candidates or page.width <= 0 or page.height <= 0:
            return 1.0, 1.0

        api_max_x = max(max_x_candidates)
        api_max_y = max(max_y_candidates)
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
            bbox = sanitize_xyxy_bbox(bbox_raw, page.width, page.height)
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
        """Call AiStudio /layout-parsing; raises on network/auth errors."""
        import cv2
        import requests
        from app.core.ocr_config import get_config

        cfg = get_config()
        url = cfg["api_url"].rstrip("/") + "/layout-parsing"
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")
        layout_model_name = cfg.get("api_layout_model_name", "").strip()

        img = cv2.imread(page.display_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        page.height, page.width = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError(f"Cannot encode image: {page.display_image_path}")
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = requests.post(
            url,
            json=self._build_api_request_body(file_b64, 1, layout_model_name),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._write_api_debug_response(page, data)

        # result.layoutParsingResults[0].prunedResult.layout_det_res.boxes
        layout_results = data.get("result", {}).get("layoutParsingResults", [])
        page.blocks = []
        raw_overlay_items: List[tuple[str, object]] = []
        order = 0
        for item in layout_results:
            scale_x, scale_y = self._detect_api_canvas_scale(page, item)
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                logger.info(
                    "API 返回坐标空间不同于原图，采用 scale_x=%.3f scale_y=%.3f 修正 (%s)",
                    scale_x, scale_y, page.display_image_path,
                )
            boxes = (
                item.get("prunedResult", {})
                    .get("layout_det_res", {})
                    .get("boxes", [])
            )
            for box in boxes:
                coord = box.get("coordinate", [0, 0, 0, 0])
                bbox = self._extract_bbox_from_coordinate(coord, page)
                if not bbox or bbox.area <= 0:
                    continue
                if scale_x != 1.0 or scale_y != 1.0:
                    bbox = scale_bbox(bbox, scale_x, scale_y).clamp(
                        page.width, page.height,
                    )
                    if bbox.area <= 0:
                        continue
                raw_type = box.get("label", "unknown")
                raw_overlay_items.append((raw_type, bbox))
                page.blocks.append(Block(
                    block_type=BlockType.from_paddle(raw_type),
                    bbox=bbox,
                    order=order,
                ))
                order += 1
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

    # ── common interface ───────────────────────────────────────

    def analyze(self, page: Page) -> Page:
        from app.core.ocr_config import get_config
        if get_config()["mode"] == "api":
            return self._api_analyze(page)
        return self._local_analyze(page)

    def analyze_pages(self, pages: List[Page]) -> List[Page]:
        return [self.analyze(page) for page in pages]
