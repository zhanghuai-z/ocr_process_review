"""Project OCR observations into proof-facing runtime models.

OCR engines should produce ``OcrIrLine``/``OcrIrToken`` observations first.
This module is the single adapter that turns those observations into the
current proof-facing ``Line``/``Char`` objects while the storage model is being
migrated away from embedding OCR and proof facts in one object tree.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.core.char_bbox_utils import is_meaningful_text_bbox
from app.core.ocr_ir import OcrIrLine, OcrIrToken
from app.core.ocr_ir_builder import (
    CHAR_BBOX_GRANULARITY_FALLBACK,
    CHAR_BBOX_SOURCE_FALLBACK,
)
from app.core.proof_line_mutation import set_line_proof_status
from app.models import Char, Line
from app.models.ocr_character_observation import replace_line_ocr_chars
from app.models.ocr_text_observation import create_ocr_text_line


ProofStatusFactory = Callable[[OcrIrLine], object | None]


def find_unoccupied_token_span(
    line_text: str,
    token_text: str,
    cursor: int,
    occupied: list[bool],
) -> tuple[int, int] | None:
    """Find the next unoccupied literal token span in OCR line text."""
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


def project_ocr_tokens_to_proof_chars(
    *,
    page_image: np.ndarray | None,
    line_text: str,
    line_confidence: float,
    tokens: list[OcrIrToken],
) -> list[Char]:
    """Project OCR token geometry onto proof ``Char`` compatibility objects."""
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
        span = find_unoccupied_token_span(line_text, token_text, cursor, occupied)
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


def project_ocr_line_to_proof_line(
    ir_line: OcrIrLine,
    *,
    page_image: np.ndarray | None = None,
    proof_status: object | None = None,
) -> Line:
    """Project one OCR observation line into the current proof line model."""
    line = create_ocr_text_line(
        text=ir_line.text,
        confidence=float(ir_line.confidence),
        bbox=ir_line.bbox,
        source_text=ir_line.source_text or ir_line.text,
        review_flags=list(ir_line.review_flags),
    )
    replace_line_ocr_chars(
        line,
        project_ocr_tokens_to_proof_chars(
            page_image=page_image,
            line_text=ir_line.text,
            line_confidence=float(ir_line.confidence),
            tokens=ir_line.tokens,
        ),
    )
    if proof_status is not None:
        set_line_proof_status(line, proof_status)
    return line


def project_ocr_lines_to_proof_lines(
    ir_lines: list[OcrIrLine],
    *,
    page_image: np.ndarray | None = None,
    proof_status_for: ProofStatusFactory | None = None,
) -> list[Line]:
    """Project a batch of OCR observation lines into proof lines."""
    lines: list[Line] = []
    for ir_line in ir_lines:
        status = proof_status_for(ir_line) if proof_status_for is not None else None
        lines.append(
            project_ocr_line_to_proof_line(
                ir_line,
                page_image=page_image,
                proof_status=status,
            )
        )
    return lines
