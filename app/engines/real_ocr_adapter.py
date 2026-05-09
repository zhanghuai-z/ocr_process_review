"""真实 OCR 引擎适配器：包装现有的本地 PaddleOCR 和 API 模式。

确保旧代码路径仍然可用，同时兼容新的 OcrEngine 接口。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.core.bbox_utils import bbox_from_quad, bbox_from_xyxy, sanitize_xyxy_bbox
from app.core.char_bbox_utils import is_meaningful_text_bbox
from app.engines import OcrContext
from app.core.logging import get_logger
from app.models import BBox, Char, Line, ProofStatus
from app.core.app_config import get_config

AUTO_FLAG_THRESHOLD = 0.80
logger = get_logger(__name__)
CHAR_BBOX_SOURCE_OCR = "ocr"
CHAR_BBOX_SOURCE_FALLBACK = "fallback"
CHAR_BBOX_GRANULARITY_CHAR = "char"
CHAR_BBOX_GRANULARITY_WORD = "word"
CHAR_BBOX_GRANULARITY_FALLBACK = "fallback"

PADDLE_OCR_TUNING = {
    "textDetLimitSideLen": 1536,
    "textDetBoxThresh": 0.6,
}


@dataclass
class TokenRow:
    tokens: list[str]
    regions: list
    bbox: Optional[BBox]


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
                proof_status=(
                    ProofStatus.AUTO_FLAGGED
                    if score < AUTO_FLAG_THRESHOLD
                    else ProofStatus.UNCHECKED
                ),
            ))
        return lines


class ApiOcrEngine:
    """AiStudio API OCR 引擎适配器。"""

    bbox_space = "crop"
    _PIPELINE_DISABLE_FLAGS = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }

    def _build_request_body(self, file_b64: str, file_type: int = 1) -> dict:
        body = {"file": file_b64, "fileType": file_type, "returnWordBox": True}
        body.update(self._PIPELINE_DISABLE_FLAGS)
        body.update(PADDLE_OCR_TUNING)
        return body

    def _iter_result_items(self, data: dict) -> list[dict]:
        result = data.get("result", {}) if isinstance(data, dict) else {}
        items: list[dict] = []
        for key in ("layoutParsingResults", "ocrResults"):
            value = result.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
        return items

    def _extract_overall_ocr_res(self, item: dict) -> dict:
        pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}
        if not isinstance(pruned, dict):
            pruned = {}
        ocr_res = pruned.get("overall_ocr_res")
        if isinstance(ocr_res, dict):
            return ocr_res
        direct = item.get("overall_ocr_res")
        return direct if isinstance(direct, dict) else {}

    def _extract_word_box_rows(self, item: dict) -> tuple[list, list]:
        pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}
        if not isinstance(pruned, dict):
            pruned = {}
        ocr_res = self._extract_overall_ocr_res(item)
        token_rows = (
            pruned.get("text_word")
            or pruned.get("textWord")
            or ocr_res.get("text_word")
            or ocr_res.get("textWord")
            or []
        )
        region_rows = (
            pruned.get("text_word_region")
            or pruned.get("textWordRegion")
            or ocr_res.get("text_word_region")
            or ocr_res.get("textWordRegion")
            or []
        )
        return token_rows, region_rows

    def _bbox_from_region(self, region) -> BBox | None:
        if isinstance(region, dict):
            if {"x", "y", "w", "h"} <= set(region.keys()):
                return BBox.from_dict(region).normalize()
            for key in ("coordinate", "bbox", "box"):
                value = region.get(key)
                if value is not None:
                    return self._bbox_from_region(value)
            return None
        if not isinstance(region, (list, tuple)):
            return None
        if len(region) >= 4 and all(not isinstance(v, (list, tuple)) for v in region[:4]):
            return bbox_from_xyxy(region[:4]).normalize()
        if len(region) >= 4 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in region[:4]):
            return bbox_from_quad(region[:4]).normalize()
        return None

    def _normalize_token_texts(self, token_row) -> list[str]:
        if isinstance(token_row, str):
            return [token_row]
        if not isinstance(token_row, (list, tuple)):
            return []
        tokens: list[str] = []
        for token in token_row:
            if token is None:
                continue
            token_text = str(token)
            if token_text == "":
                continue
            tokens.append(token_text)
        return tokens

    def _merge_bboxes(self, boxes: list[BBox]) -> Optional[BBox]:
        if not boxes:
            return None
        x1 = min(box.x for box in boxes)
        y1 = min(box.y for box in boxes)
        x2 = max(box.x2 for box in boxes)
        y2 = max(box.y2 for box in boxes)
        return BBox.from_xyxy(x1, y1, x2, y2).normalize()

    def _bbox_overlap_ratio(self, first: Optional[BBox], second: Optional[BBox]) -> float:
        if first is None or second is None or first.area <= 0 or second.area <= 0:
            return 0.0
        x1 = max(first.x, second.x)
        y1 = max(first.y, second.y)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        return inter / float(min(first.area, second.area))

    def _build_token_rows(self, item: dict) -> list[TokenRow]:
        token_rows, region_rows = self._extract_word_box_rows(item)
        rows: list[TokenRow] = []
        for token_row, region_row in zip(token_rows, region_rows):
            tokens = [token.strip() for token in self._normalize_token_texts(token_row) if token.strip()]
            if not tokens or not isinstance(region_row, (list, tuple)):
                continue
            paired_tokens: list[str] = []
            paired_regions: list = []
            boxes: list[BBox] = []
            for token_text, raw_region in zip(tokens, region_row):
                bbox = self._bbox_from_region(raw_region)
                if bbox is None or bbox.area <= 0:
                    continue
                paired_tokens.append(token_text)
                paired_regions.append(raw_region)
                boxes.append(bbox)
            if not paired_tokens:
                continue
            rows.append(TokenRow(
                tokens=paired_tokens,
                regions=paired_regions,
                bbox=self._merge_bboxes(boxes),
            ))
        return rows

    def _select_token_row_for_line(self, token_rows: list[TokenRow], line_bbox: BBox) -> Optional[TokenRow]:
        scored: list[tuple[float, float, int, int, TokenRow]] = []
        line_center_y = line_bbox.y + line_bbox.h / 2.0
        for idx, row in enumerate(token_rows):
            overlap = self._bbox_overlap_ratio(row.bbox, line_bbox)
            if overlap <= 0:
                continue
            row_center_y = row.bbox.y + row.bbox.h / 2.0 if row.bbox else line_center_y
            distance = abs(row_center_y - line_center_y)
            scored.append((-overlap, distance, row.bbox.x if row.bbox else 0, idx, row))
        if not scored:
            return None
        scored.sort()
        return scored[0][4]

    def _find_token_span(
        self,
        line_text: str,
        token_text: str,
        cursor: int,
        occupied: list[bool],
    ) -> tuple[int, int] | None:
        token_len = len(token_text)
        if token_len <= 0 or token_len > len(line_text):
            return None
        search_start = max(0, min(cursor, len(line_text) - token_len))
        for start in range(search_start, len(line_text) - token_len + 1):
            end = start + token_len
            if line_text[start:end] != token_text:
                continue
            if any(occupied[start:end]):
                continue
            return start, end
        for start in range(0, search_start):
            end = start + token_len
            if line_text[start:end] != token_text:
                continue
            if any(occupied[start:end]):
                continue
            return start, end
        return None

    def _build_line_chars(
        self,
        *,
        page_image: np.ndarray,
        line_text: str,
        line_confidence: float,
        token_row,
        region_row,
    ) -> list[Char]:
        chars = [
            Char(
                char=glyph,
                confidence=float(line_confidence),
                bbox=None,
                bbox_source=CHAR_BBOX_SOURCE_FALLBACK,
                bbox_granularity=CHAR_BBOX_GRANULARITY_FALLBACK,
                token_text=glyph,
            )
            for glyph in line_text
        ]
        if not line_text:
            return chars

        tokens = self._normalize_token_texts(token_row)
        if not isinstance(region_row, (list, tuple)):
            return chars

        occupied = [False] * len(line_text)
        cursor = 0
        for token, raw_region in zip(tokens, region_row):
            token_text = token.strip()
            if not token_text:
                continue
            bbox = self._bbox_from_region(raw_region)
            if bbox is None or bbox.area <= 0:
                continue
            if not is_meaningful_text_bbox(page_image, bbox, token_text):
                continue
            span = self._find_token_span(line_text, token_text, cursor, occupied)
            if span is None:
                continue
            start, end = span
            granularity = (
                CHAR_BBOX_GRANULARITY_CHAR
                if len(token_text) == 1
                else CHAR_BBOX_GRANULARITY_WORD
            )
            for idx in range(start, end):
                chars[idx] = Char(
                    char=line_text[idx],
                    confidence=float(line_confidence),
                    bbox=bbox,
                    bbox_source=CHAR_BBOX_SOURCE_OCR,
                    bbox_granularity=granularity,
                    token_text=token_text,
                )
                occupied[idx] = True
            cursor = end
        return chars

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
            json=self._build_request_body(file_b64, 1),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        lines: List[Line] = []
        for item in self._iter_result_items(data):
            ocr_res = self._extract_overall_ocr_res(item)
            texts = ocr_res.get("rec_texts", [])
            scores = ocr_res.get("rec_scores", [])
            boxes = (
                ocr_res.get("rec_boxes")
                or ocr_res.get("rec_polys")
                or ocr_res.get("rec_polygons")
                or ocr_res.get("dt_polys")
                or []
            )
            token_rows = self._build_token_rows(item)
            for idx, text in enumerate(texts):
                line_text = str(text)
                if not line_text:
                    continue
                score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
                bbox = self._bbox_from_region(boxes[idx]) if idx < len(boxes) else None
                if bbox is None or bbox.area <= 0:
                    continue
                token_row = self._select_token_row_for_line(token_rows, bbox)
                chars = self._build_line_chars(
                    page_image=image_bgr,
                    line_text=line_text,
                    line_confidence=score,
                    token_row=token_row.tokens if token_row else [],
                    region_row=token_row.regions if token_row else [],
                )
                lines.append(Line(
                    text=line_text,
                    confidence=score,
                    bbox=bbox,
                    chars=chars,
                    ocr_text=line_text,
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
