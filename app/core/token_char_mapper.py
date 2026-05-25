"""Map OCR IR tokens onto proof-facing Char objects."""
from __future__ import annotations

import numpy as np

from app.core.char_bbox_utils import is_meaningful_text_bbox
from app.core.ocr_ir import OcrIrLine, OcrIrToken
from app.core.ocr_ir_builder import (
    CHAR_BBOX_GRANULARITY_FALLBACK,
    CHAR_BBOX_SOURCE_FALLBACK,
)
from app.models import Char, Line


def find_token_span(
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


def build_line_chars(
    *,
    page_image: np.ndarray | None,
    line_text: str,
    line_confidence: float,
    tokens: list[OcrIrToken],
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
    occupied = [False] * len(line_text)
    cursor = 0
    for token in tokens:
        token_text = token.text.strip()
        if not token_text:
            continue
        bbox = token.bbox
        if bbox is None or bbox.area <= 0:
            continue
        if page_image is not None and not is_meaningful_text_bbox(page_image, bbox, token_text):
            continue
        span = find_token_span(line_text, token_text, cursor, occupied)
        if span is None:
            continue
        start, end = span
        confidence = (
            float(token.confidence)
            if token.confidence is not None
            else float(line_confidence)
        )
        for idx in range(start, end):
            chars[idx] = Char(
                char=line_text[idx],
                confidence=confidence,
                bbox=bbox,
                bbox_source=token.bbox_source,
                bbox_granularity=token.bbox_granularity,
                token_text=token_text,
            )
            occupied[idx] = True
        cursor = end
    return chars


def build_line_from_ir(
    ir_line: OcrIrLine,
    *,
    page_image: np.ndarray | None = None,
    proof_status=None,
) -> Line:
    """Convert one OCR_IR line into the proof-facing Line/Char model."""
    line = Line(
        text=ir_line.text,
        final_text=ir_line.text,
        confidence=float(ir_line.confidence),
        bbox=ir_line.bbox,
        chars=build_line_chars(
            page_image=page_image,
            line_text=ir_line.text,
            line_confidence=float(ir_line.confidence),
            tokens=ir_line.tokens,
        ),
        ocr_text=ir_line.source_text or ir_line.text,
        review_flags=list(ir_line.review_flags),
    )
    if proof_status is not None:
        line.proof_status = proof_status
    return line
