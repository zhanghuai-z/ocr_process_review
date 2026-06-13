"""Build OCR intermediate representation from Paddle response records."""
from __future__ import annotations

from collections.abc import Sequence

from app.core.bbox_extraction import bbox_from_variant
from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
from app.core.ocr_ir import (
    OCR_IR_SOURCE_REC_TEXT,
    OcrIrLine,
)
from app.core.paddle_response import overall_ocr_res
from app.core.proof_status import normalize_confidence
from app.models import BBox

CHAR_BBOX_SOURCE_FALLBACK = "fallback"
CHAR_BBOX_GRANULARITY_FALLBACK = "fallback"


def _as_sequence(value) -> Sequence:
    return value if isinstance(value, (list, tuple)) else []


def build_ir_lines_from_item(
    item: dict,
    *,
    image_shape=None,
    fallback_bbox: BBox,
) -> list[OcrIrLine]:
    """Build line IR from Paddle OCR line-level response fields."""
    ocr_res = overall_ocr_res(item)
    texts = _as_sequence(ocr_res.get("rec_texts", []))
    scores = _as_sequence(ocr_res.get("rec_scores", []))
    boxes = _as_sequence(
        ocr_res.get("rec_boxes")
        or ocr_res.get("rec_polys")
        or ocr_res.get("rec_polygons")
        or ocr_res.get("dt_polys")
        or []
    )
    ir_lines: list[OcrIrLine] = []

    for idx, text in enumerate(texts):
        line_text = str(text)
        if not line_text:
            continue
        score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
        bbox = bbox_from_variant(boxes[idx], image_shape=image_shape) if idx < len(boxes) else None
        review_flags: list[str] = []
        if bbox is None or bbox.area <= 0:
            bbox = fallback_bbox
            review_flags.append(MISSING_LINE_BBOX_FLAG)
        ir_lines.append(OcrIrLine(
            text=line_text,
            confidence=score,
            bbox=bbox,
            source_text=OCR_IR_SOURCE_REC_TEXT,
            tokens=[],
            review_flags=review_flags,
        ))

    return ir_lines
