"""真实 OCR 引擎适配器：包装现有的本地 PaddleOCR 和 API 模式。

确保旧代码路径仍然可用，同时兼容新的 OcrEngine 接口。
"""
from __future__ import annotations
from dataclasses import replace
from typing import List

import numpy as np

from app.core.api_profiles import FIXED_OCR_PROFILE, get_api_request_options, resolve_api_endpoint_for_role
from app.core.bbox_extraction import bbox_from_variant
from app.core.bbox_utils import sanitize_xyxy_bbox
from app.core.ocr_ir import (
    OcrIrLine,
    OcrIrToken,
    is_cjk_char,
)
from app.core.ocr_ir_builder import (
    CHAR_BBOX_GRANULARITY_CHAR,
    build_ir_lines_from_item,
)
from app.core.paddle_response import iter_ocr_preferred_items
from app.core.proof_status import normalize_confidence, proof_status_for
from app.core.token_char_mapper import build_line_chars
from app.core.wordbox_anchor import refine_wordbox_anchors
from app.engines import OcrContext
from app.core.logging import get_logger
from app.models import BBox, Line
from app.core.app_config import get_config

logger = get_logger(__name__)

def get_engine_description(mode: str = "") -> str:
    """返回当前 OCR 引擎描述。"""
    if not mode:
        mode = get_config().get("mode", "local")
    if mode == "api":
        cfg = get_config()
        api_url = cfg.get("api_url", "")
        endpoint = resolve_api_endpoint_for_role(
            api_url,
            profile=FIXED_OCR_PROFILE,
            role="ocr",
        )
        return f"API OCR / Proof（PP-OCRv5：{endpoint or '未配置 URL'}）"
    if mode == "mock":
        return "Mock OCR（FakeOcrEngine）"
    if mode == "hanwang":
        return "PP-VL 版面 + 汉王 micro-recblock OCR（文字块走 Hanwang，公式/表格/图片保留 PP-VL）"
    return (
        "PaddleOCR 本地默认中文模型组合"
        "（lang=ch, use_angle_cls=True, layout=False, table=False；"
        "未显式固定 det/rec/cls 模型名）"
    )


class LocalOcrEngine:
    """本地 PaddleOCR 引擎适配器。"""

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            logger.info(
                "Initializing PaddleOCR text engine: lang=ch, "
                "use_angle_cls=True, layout=False, table=False, "
                "models=package defaults"
            )
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
        crop_h, crop_w = image_bgr.shape[:2]
        if not result or not result[0]:
            return lines
        for line_data in result[0]:
            pts, (text, score) = line_data
            score = normalize_confidence(score)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = sanitize_xyxy_bbox(
                [min(xs), min(ys), max(xs), max(ys)],
                crop_w,
                crop_h,
            )
            if bbox.area <= 0:
                continue
            lines.append(Line(
                text=text,
                confidence=float(score),
                bbox=bbox,
                proof_status=proof_status_for(score),
            ))
        return lines


class ApiOcrEngine:
    """AiStudio API OCR 引擎适配器。"""

    bbox_space = "crop"
    prefer_page_ocr = True

    def _build_request_body(
        self,
        file_b64: str,
        file_type: int = 1,
        *,
        profile: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict:
        body = {"file": file_b64, "fileType": file_type}
        body.update(get_api_request_options(profile, endpoint_url))
        return body

    def _bbox_from_region(self, region, image_shape=None) -> BBox | None:
        return bbox_from_variant(region, image_shape=image_shape)

    def _unverified_full_crop_bbox(self, image_bgr: np.ndarray) -> BBox:
        height, width = image_bgr.shape[:2]
        return BBox(0, 0, max(1, int(width)), max(1, int(height)))

    def _refine_tokens_with_wordbox_anchor(
        self,
        *,
        page_image: np.ndarray,
        line_bbox: BBox,
        tokens: list[OcrIrToken],
    ) -> list[OcrIrToken]:
        pairs = [
            (token.text, token.bbox)
            for token in tokens
            if token.bbox is not None and token.bbox.area > 0
        ]
        if not pairs:
            return tokens
        anchors = refine_wordbox_anchors(
            page_image,
            line_bbox,
            pairs,
            include_non_cjk=True,
        )
        if len(anchors) != len(pairs):
            return tokens

        refined: list[OcrIrToken] = []
        pair_idx = 0
        for token in tokens:
            if token.bbox is None or token.bbox.area <= 0:
                refined.append(token)
                continue
            anchor = anchors[pair_idx]
            pair_idx += 1
            if (
                anchor.kind == "cjk"
                and is_cjk_char(token.text.strip())
                and anchor.crop_bbox.area > 0
            ):
                refined.append(replace(
                    token,
                    bbox=anchor.crop_bbox,
                    bbox_granularity=CHAR_BBOX_GRANULARITY_CHAR,
                ))
            else:
                refined.append(token)
        return refined

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        import base64
        import cv2
        from app.core.api_http import post_json_without_env_proxy
        from app.core.app_config import get_config

        cfg = get_config()
        url = resolve_api_endpoint_for_role(
            cfg["api_url"],
            profile=FIXED_OCR_PROFILE,
            role="ocr",
        )
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")

        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            return []
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = post_json_without_env_proxy(
            url,
            json=self._build_request_body(
                file_b64,
                1,
                profile=FIXED_OCR_PROFILE,
                endpoint_url=url,
            ),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        ir_lines: list[OcrIrLine] = []
        def refine_tokens(line_bbox: BBox, tokens: list[OcrIrToken]) -> list[OcrIrToken]:
            return self._refine_tokens_with_wordbox_anchor(
                page_image=image_bgr,
                line_bbox=line_bbox,
                tokens=tokens,
            )

        for item in iter_ocr_preferred_items(data):
            ir_lines.extend(build_ir_lines_from_item(
                item,
                image_shape=image_bgr.shape[:2],
                fallback_bbox=self._unverified_full_crop_bbox(image_bgr),
                existing_lines=ir_lines,
                refine_tokens=refine_tokens,
            ))

        lines: List[Line] = []
        for ir_line in ir_lines:
            chars = build_line_chars(
                page_image=image_bgr,
                line_text=ir_line.text,
                line_confidence=ir_line.confidence,
                tokens=ir_line.tokens,
            )
            lines.append(Line(
                text=ir_line.text,
                confidence=ir_line.confidence,
                bbox=ir_line.bbox,
                chars=chars,
                ocr_text=ir_line.text,
                review_flags=ir_line.review_flags,
                proof_status=proof_status_for(ir_line.confidence, ir_line.review_flags),
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
    elif mode == "hanwang":
        from app.engines.hanwang.micro_recblock import HanwangMicroRecBlockEngine
        return HanwangMicroRecBlockEngine()
    else:
        return LocalOcrEngine()
