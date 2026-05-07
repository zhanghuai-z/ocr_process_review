"""真实 OCR 引擎适配器：包装现有的本地 PaddleOCR 和 API 模式。

确保旧代码路径仍然可用，同时兼容新的 OcrEngine 接口。
"""
from __future__ import annotations
from typing import List

import numpy as np

from app.core.api_response_utils import (
    build_api_payload,
    detect_api_result_kind,
    extract_api_markdown_text,
    extract_bbox_from_box_data,
    get_api_result_items,
    iter_api_ocr_payloads,
    resolve_api_endpoint,
    split_markdown_lines,
)
from app.engines import OcrContext
from app.core.logging import get_logger
from app.core.paddle_result_utils import (
    get_local_ocr_init_kwargs,
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
        api_url = resolve_api_endpoint(cfg.get("api_url", ""), default_suffix="/layout-parsing")
        return f"API OCR（{api_url or '未配置 URL'}）"
    if mode == "mock":
        return "Mock OCR（FakeOcrEngine）"
    return "PaddleOCR 本地默认模型（3.x predict API）"


def _extract_api_lines(item: dict, crop_w: int, crop_h: int) -> List[Line]:
    lines: List[Line] = []

    for payload in iter_api_ocr_payloads(item):
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_boxes", [])
        polys = payload.get("rec_polys", [])
        for idx, text in enumerate(texts):
            if not text:
                continue
            score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
            box_data = boxes[idx] if idx < len(boxes) else (polys[idx] if idx < len(polys) else None)
            bbox = extract_bbox_from_box_data(box_data, crop_w, crop_h)
            if bbox.area <= 0:
                bbox = BBox(0, 0, crop_w, crop_h)
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

    if lines:
        return lines

    markdown_text = extract_api_markdown_text(item)
    fallback_bbox = BBox(0, 0, crop_w, crop_h)
    return [
        Line(text=text, confidence=1.0, bbox=fallback_bbox, proof_status=ProofStatus.UNCHECKED)
        for text in split_markdown_lines(markdown_text)
    ]


class LocalOcrEngine:
    """本地 PaddleOCR 引擎适配器。"""

    def __init__(self) -> None:
        self._engine = None

    def close(self) -> None:
        import gc

        engine = self._engine
        self._engine = None
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        gc.collect()

    def _get_engine(self):
        if self._engine is None:
            prepare_paddle_runtime_env()
            from paddleocr import PaddleOCR
            init_kwargs = get_local_ocr_init_kwargs()
            logger.info(
                "Initializing PaddleOCR text engine: "
                "text_det_limit_side_len=%s, text_det_limit_type=%s, "
                "use_doc_orientation_classify=False, use_doc_unwarping=False, "
                "use_textline_orientation=%s",
                init_kwargs["text_det_limit_side_len"],
                init_kwargs["text_det_limit_type"],
                init_kwargs["use_textline_orientation"],
            )
            self._engine = PaddleOCR(**init_kwargs)
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
                bbox = extract_bbox_from_box_data(box_data, crop_w, crop_h)
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

    def close(self) -> None:
        return None

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        import base64
        import cv2
        import requests
        from app.core.app_config import get_config

        cfg = get_config()
        url = resolve_api_endpoint(cfg["api_url"], default_suffix="/layout-parsing")
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
            json=build_api_payload(file_b64, 1),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        lines: List[Line] = []
        for item in get_api_result_items(data):
            lines.extend(_extract_api_lines(item, image_bgr.shape[1], image_bgr.shape[0]))
        logger.info("Parsed AIStudio API response: endpoint=%s kind=%s lines=%d", url, detect_api_result_kind(data), len(lines))
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
