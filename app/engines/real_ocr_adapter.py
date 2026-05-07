"""真实 OCR 引擎适配器：包装现有的本地 PaddleOCR 和 API 模式。

确保旧代码路径仍然可用，同时兼容新的 OcrEngine 接口。
"""
from __future__ import annotations
from typing import List

import numpy as np

from app.core.bbox_utils import bbox_from_quad, bbox_from_xyxy, sanitize_xyxy_bbox
from app.engines import OcrContext
from app.core.logging import get_logger
from app.core.paddle_result_utils import (
    DEFAULT_LOCAL_LANG,
    DEFAULT_LOCAL_OCR_VERSION,
    prepare_paddle_runtime_env,
    results_to_dicts,
)
from app.models import BBox, Line, ProofStatus
from app.core.app_config import get_config

AUTO_FLAG_THRESHOLD = 0.80
logger = get_logger(__name__)


def normalize_confidence(score) -> float:
    """统一把引擎分数归一化到 0~1。

    有些服务可能返回 0~1，也可能返回 0~100；这里统一处理，避免
    置信度链路断掉后影响自动标记、UI 徽章和后续校对流程。
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0

    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(value, 1.0))


def get_engine_description(mode: str = "") -> str:
    """返回当前 OCR 引擎描述。"""
    if not mode:
        mode = get_config().get("mode", "local")
    if mode == "api":
        cfg = get_config()
        api_url = cfg.get("api_url", "")
        return f"API OCR（/layout-parsing, {api_url or '未配置 URL'}）"
    if mode == "mock":
        return "Mock OCR（FakeOcrEngine）"
    cfg = get_config()
    ocr_version = cfg.get("local_ocr_version", DEFAULT_LOCAL_OCR_VERSION)
    return (
        "PaddleOCR 本地显式模型"
        f"（lang={DEFAULT_LOCAL_LANG}, ocr_version={ocr_version}；"
        "新 3.x predict API）"
    )


def _extract_line_bbox(box_data, crop_w: int, crop_h: int) -> BBox:
    if box_data is None:
        return BBox(0, 0, 0, 0)
    if hasattr(box_data, "tolist"):
        box_data = box_data.tolist()
    if isinstance(box_data, (list, tuple)) and len(box_data) >= 4:
        if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in box_data[:4]):
            bbox = bbox_from_quad(box_data[:4])
            return bbox.clamp(crop_w, crop_h)
        return sanitize_xyxy_bbox(box_data[:4], crop_w, crop_h)
    return BBox(0, 0, 0, 0)


class LocalOcrEngine:
    """本地 PaddleOCR 引擎适配器。"""

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            prepare_paddle_runtime_env()
            from paddleocr import PaddleOCR
            cfg = get_config()
            ocr_version = cfg.get("local_ocr_version", DEFAULT_LOCAL_OCR_VERSION)
            logger.info(
                "Initializing PaddleOCR text engine: lang=%s, "
                "ocr_version=%s, use_doc_orientation_classify=False, "
                "use_doc_unwarping=False, use_textline_orientation=True",
                DEFAULT_LOCAL_LANG,
                ocr_version,
            )
            self._engine = PaddleOCR(
                lang=DEFAULT_LOCAL_LANG,
                ocr_version=ocr_version,
                engine="paddle_dynamic",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
        return self._engine

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        engine = self._get_engine()
        result = engine.predict(image_bgr)
        lines: List[Line] = []
        crop_h, crop_w = image_bgr.shape[:2]
        for payload in results_to_dicts(result):
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_boxes", [])
            polys = payload.get("rec_polys", [])
            for idx, text in enumerate(texts):
                if not text:
                    continue
                score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
                box_data = boxes[idx] if idx < len(boxes) else (polys[idx] if idx < len(polys) else None)
                bbox = _extract_line_bbox(box_data, crop_w, crop_h)
                if bbox.area <= 0:
                    continue
                lines.append(Line(
                    text=text,
                    confidence=float(score),
                    bbox=bbox,
                    proof_status=(
                        ProofStatus.AUTO_FLAGGED
                        if score < AUTO_FLAG_THRESHOLD
                        else ProofStatus.UNCHECKED
                    ),
                ))
        return lines


class ApiOcrEngine:
    """AiStudio API OCR 引擎适配器。"""

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        import base64
        import cv2
        import requests
        from app.core.app_config import get_config

        cfg = get_config()
        url = cfg["api_url"].rstrip("/") + "/layout-parsing"
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")

        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            return []
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = requests.post(
            url,
            json={"file": file_b64, "fileType": 1},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        layout_results = data.get("result", {}).get("layoutParsingResults", [])
        lines: List[Line] = []
        for item in layout_results:
            ocr_res = (
                item.get("prunedResult", {})
                    .get("overall_ocr_res", {})
            )
            texts = ocr_res.get("rec_texts", [])
            scores = ocr_res.get("rec_scores", [])
            boxes = ocr_res.get("rec_boxes", [])
            for idx, text in enumerate(texts):
                score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
                if idx < len(boxes):
                    bbox = _extract_line_bbox(boxes[idx], image_bgr.shape[1], image_bgr.shape[0])
                else:
                    bbox = BBox(0, 0, 0, 0)
                lines.append(Line(
                    text=text,
                    confidence=score,
                    bbox=bbox,
                    proof_status=(
                        ProofStatus.AUTO_FLAGGED
                        if score < AUTO_FLAG_THRESHOLD
                        else ProofStatus.UNCHECKED
                    ),
                ))
        return lines


def create_engine(mode: str = ""):
    """根据配置创建合适的 OCR 引擎。

    Args:
        mode: "local", "api", "mock"(fake), 或 ""(从配置读取)

    Returns:
        OcrEngine 实现实例。
    """
    if not mode:
        from app.core.app_config import get_config
        cfg = get_config()
        mode = cfg.get("mode", "local")

    if mode == "api":
        return ApiOcrEngine()
    elif mode == "mock":
        from app.engines.fake_ocr_engine import FakeOcrEngine
        return FakeOcrEngine()
    else:
        return LocalOcrEngine()
