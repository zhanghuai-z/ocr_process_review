"""Build OCR intermediate representation from Paddle response records."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from app.core.bbox_extraction import bbox_from_variant
from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
from app.core.ocr_ir import (
    OCR_IR_SOURCE_REC_TEXT,
    OCR_IR_SOURCE_TOKEN_TEXT,
    OCR_IR_TOKEN_TEXT_FALLBACK_FLAG,
    OcrIrLine,
    OcrIrToken,
    classify_ir_text,
)
from app.core.paddle_response import overall_ocr_res, word_box_rows
from app.core.proof_status import normalize_confidence
from app.models import BBox

CHAR_BBOX_SOURCE_OCR = "ocr"
CHAR_BBOX_SOURCE_FALLBACK = "fallback"
CHAR_BBOX_GRANULARITY_CHAR = "char"
CHAR_BBOX_GRANULARITY_WORD = "word"
CHAR_BBOX_GRANULARITY_FALLBACK = "fallback"

TokenRefiner = Callable[[BBox, list[OcrIrToken]], list[OcrIrToken]]


@dataclass
class TokenRow:
    tokens: list[str]
    regions: list
    bbox: Optional[BBox]
    ir_tokens: list[OcrIrToken]


def _as_sequence(value) -> Sequence:
    return value if isinstance(value, (list, tuple)) else []


def normalize_token_texts(token_row) -> list[str]:
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


def merge_bboxes(boxes: list[BBox]) -> Optional[BBox]:
    if not boxes:
        return None
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return BBox.from_xyxy(x1, y1, x2, y2).normalize()


def bbox_overlap_ratio(first: Optional[BBox], second: Optional[BBox]) -> float:
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


def build_token_rows(item: dict, image_shape=None) -> list[TokenRow]:
    token_rows, region_rows = word_box_rows(item)
    rows: list[TokenRow] = []
    for token_row, region_row in zip(token_rows, region_rows):
        tokens = [token.strip() for token in normalize_token_texts(token_row) if token.strip()]
        if not tokens or not isinstance(region_row, (list, tuple)):
            continue
        paired_tokens: list[str] = []
        paired_regions: list = []
        boxes: list[BBox] = []
        ir_tokens: list[OcrIrToken] = []
        for token_idx, (token_text, raw_region) in enumerate(zip(tokens, region_row)):
            bbox = bbox_from_variant(raw_region, image_shape=image_shape)
            if bbox is None or bbox.area <= 0:
                continue
            paired_tokens.append(token_text)
            paired_regions.append(raw_region)
            boxes.append(bbox)
            ir_tokens.append(OcrIrToken(
                text=token_text,
                bbox=bbox,
                row_index=len(rows),
                token_index=token_idx,
                raw_region=raw_region,
                kind=classify_ir_text(token_text),
                bbox_source=CHAR_BBOX_SOURCE_OCR,
                bbox_granularity=(
                    CHAR_BBOX_GRANULARITY_CHAR
                    if len(token_text) == 1
                    else CHAR_BBOX_GRANULARITY_WORD
                ),
            ))
        if not paired_tokens:
            continue
        rows.append(TokenRow(
            tokens=paired_tokens,
            regions=paired_regions,
            bbox=merge_bboxes(boxes),
            ir_tokens=ir_tokens,
        ))
    return rows


def compact_text(text: str) -> str:
    return "".join(ch for ch in str(text) if not ch.isspace())


def token_row_matches_line(row: TokenRow, line_text: str) -> bool:
    return compact_text("".join(row.tokens)) == compact_text(line_text)


def fallback_token_row_for_missing_line_bbox(
    token_rows: list[TokenRow],
    row_idx: int,
    line_text: str,
) -> Optional[TokenRow]:
    if row_idx >= len(token_rows):
        return None
    row = token_rows[row_idx]
    if row.bbox is None or row.bbox.area <= 0:
        return None
    if not token_row_matches_line(row, line_text):
        return None
    return row


def select_token_row_for_line(token_rows: list[TokenRow], line_bbox: BBox) -> Optional[TokenRow]:
    scored: list[tuple[float, float, int, int, TokenRow]] = []
    line_center_y = line_bbox.y + line_bbox.h / 2.0
    for idx, row in enumerate(token_rows):
        overlap = bbox_overlap_ratio(row.bbox, line_bbox)
        if overlap <= 0:
            continue
        row_center_y = row.bbox.y + row.bbox.h / 2.0 if row.bbox else line_center_y
        distance = abs(row_center_y - line_center_y)
        scored.append((-overlap, distance, row.bbox.x if row.bbox else 0, idx, row))
    if not scored:
        return None
    scored.sort()
    return scored[0][4]


def looks_like_existing_ir_line(ir_lines: list[OcrIrLine], row: TokenRow) -> bool:
    row_text = compact_text("".join(row.tokens))
    if not row_text or row.bbox is None:
        return True
    for ir_line in ir_lines:
        if compact_text(ir_line.text) != row_text:
            continue
        if bbox_overlap_ratio(ir_line.bbox, row.bbox) >= 0.80:
            return True
    return False


def _refine_or_keep(
    refiner: TokenRefiner | None,
    line_bbox: BBox,
    tokens: list[OcrIrToken],
) -> list[OcrIrToken]:
    return refiner(line_bbox, tokens) if refiner else tokens


def build_ir_lines_from_item(
    item: dict,
    *,
    image_shape=None,
    fallback_bbox: BBox,
    refine_tokens: TokenRefiner | None = None,
    existing_lines: list[OcrIrLine] | None = None,
) -> list[OcrIrLine]:
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
    token_rows = build_token_rows(item, image_shape)
    used_token_rows: set[int] = set()
    ir_lines: list[OcrIrLine] = []

    for idx, text in enumerate(texts):
        line_text = str(text)
        if not line_text:
            continue
        score = normalize_confidence(scores[idx]) if idx < len(scores) else 0.0
        bbox = bbox_from_variant(boxes[idx], image_shape=image_shape) if idx < len(boxes) else None
        review_flags: list[str] = []
        if bbox is None or bbox.area <= 0:
            token_row = fallback_token_row_for_missing_line_bbox(token_rows, idx, line_text)
            if token_row is None:
                bbox = fallback_bbox
                review_flags.append(MISSING_LINE_BBOX_FLAG)
            else:
                bbox = token_row.bbox
                used_token_rows.add(id(token_row))
        else:
            token_row = select_token_row_for_line(token_rows, bbox)
            if token_row is not None:
                used_token_rows.add(id(token_row))
        line_tokens = (
            _refine_or_keep(refine_tokens, bbox, token_row.ir_tokens)
            if token_row else []
        )
        ir_lines.append(OcrIrLine(
            text=line_text,
            confidence=score,
            bbox=bbox,
            source_text=OCR_IR_SOURCE_REC_TEXT,
            tokens=line_tokens,
            review_flags=review_flags,
        ))

    if not texts:
        known_lines = (existing_lines or []) + ir_lines
        for token_row in token_rows:
            if id(token_row) in used_token_rows or looks_like_existing_ir_line(known_lines, token_row):
                continue
            if token_row.bbox is None or token_row.bbox.area <= 0:
                continue
            line_tokens = _refine_or_keep(refine_tokens, token_row.bbox, token_row.ir_tokens)
            ir_lines.append(OcrIrLine(
                text="".join(token_row.tokens),
                confidence=0.0,
                bbox=token_row.bbox,
                source_text=OCR_IR_SOURCE_TOKEN_TEXT,
                tokens=line_tokens,
                review_flags=[OCR_IR_TOKEN_TEXT_FALLBACK_FLAG],
            ))

    return ir_lines
