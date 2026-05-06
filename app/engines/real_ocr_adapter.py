"""真实 OCR 引擎适配器：包装现有的本地 PaddleOCR 和 API 模式。

确保旧代码路径仍然可用，同时兼容新的 OcrEngine 接口。
"""
from __future__ import annotations
from typing import List

import numpy as np

from app.engines import OcrContext
from app.models import BBox, Line, ProofStatus
from app.core.app_config import get_config

AUTO_FLAG_THRESHOLD = 0.80


class LocalOcrEngine:
    """本地 PaddleOCR 引擎适配器。"""

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                layout=False,
                table=False,
            )
        return self._engine

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        engine = self._get_engine()
        result = engine.ocr(image_bgr, cls=True)
        lines: List[Line] = []
        if not result or not result[0]:
            return lines
        for line_data in result[0]:
            pts, (text, score) = line_data
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lx, ly = int(min(xs)), int(min(ys))
            lw, lh = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            lines.append(Line(
                text=text,
                confidence=float(score),
                bbox=BBox(lx, ly, lw, lh),
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
                score = float(scores[idx]) if idx < len(scores) else 0.0
                if idx < len(boxes):
                    b = boxes[idx]
                    lx, ly = int(b[0]), int(b[1])
                    lw, lh = int(b[2]) - lx, int(b[3]) - ly
                else:
                    lx = ly = lw = lh = 0
                lines.append(Line(
                    text=text,
                    confidence=score,
                    bbox=BBox(lx, ly, lw, lh),
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
