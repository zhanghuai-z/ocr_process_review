from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2

from app.core.char_bbox_utils import (
    BBOX_GRANULARITY_UNAVAILABLE,
    BBOX_SOURCE_UNAVAILABLE,
    MISSING_LINE_BBOX_FLAG,
    split_line_bbox_into_char_bboxes,
)
from app.core.proof_line_facts import proof_display_text
from app.models import BBox, Char, Line, OcrProject, Page
from app.models.ocr_observation import block_ocr_lines

INLINE_FORMULA_REVIEW_FLAG = "hanwang_route_inline_formula"


def _is_tokenized_char(char: Char) -> bool:
    return char.bbox_granularity == "word" or len(char.char or "") > 1


def _fallback_char(glyph: str, line: Line, bbox: BBox | None, source: str, granularity: str) -> Char:
    return Char(
        char=glyph,
        confidence=float(line.confidence),
        bbox=bbox,
        bbox_source=source,
        bbox_granularity=granularity,
        token_text=glyph,
    )


def _complete_positional_chars(line: Line, text: str, boxes: list[BBox] | None) -> list[Char]:
    completed: list[Char] = []
    existing = list(line.chars)
    for idx, glyph in enumerate(text):
        current = existing[idx] if idx < len(existing) else None
        if current is not None and current.char == glyph and current.bbox is not None:
            completed.append(current)
            continue
        completed.append(
            _fallback_char(
                glyph,
                line,
                boxes[idx] if boxes is not None and idx < len(boxes) else None,
                "fallback",
                "fallback",
            )
        )
    return completed


@dataclass
class ProofCropStats:
    pages: int = 0
    lines: int = 0
    line_bbox_updates: int = 0
    char_bbox_updates: int = 0
    fallback_lines: int = 0
    fallback_chars: int = 0
    unavailable_chars: int = 0


class ProofCropService:
    """统一收口 proof 使用的行框/字框几何。"""

    def normalize_project(self, project: OcrProject) -> ProofCropStats:
        return self.normalize_pages(project.pages)

    def normalize_pages(self, pages: Iterable[Page]) -> ProofCropStats:
        stats = ProofCropStats()
        for page in pages:
            self._normalize_page(page, stats)
        return stats

    def _normalize_page(self, page: Page, stats: ProofCropStats) -> None:
        image = cv2.imread(page.display_image_path, cv2.IMREAD_COLOR)
        if image is None:
            return

        page_changed = False
        page_line_updates = 0
        page_char_updates = 0

        for block in page.text_blocks:
            for line in block_ocr_lines(block):
                stats.lines += 1
                old_line_bbox = line.bbox
                old_char_boxes = [
                    char.bbox.to_dict() if char.bbox is not None else None
                    for char in line.chars
                ]

                has_tokenized_chars = any(_is_tokenized_char(char) for char in line.chars)
                needs_fallback_chars = (
                    not line.chars
                    or (
                        not has_tokenized_chars
                        and len(line.chars) != len(proof_display_text(line))
                    )
                )
                if needs_fallback_chars and INLINE_FORMULA_REVIEW_FLAG not in line.review_flags:
                    text = proof_display_text(line)
                    if text and MISSING_LINE_BBOX_FLAG in line.review_flags:
                        stats.fallback_lines += 1
                        stats.unavailable_chars += len(text)
                        line.chars = [
                            _fallback_char(
                                glyph,
                                line,
                                None,
                                BBOX_SOURCE_UNAVAILABLE,
                                BBOX_GRANULARITY_UNAVAILABLE,
                            )
                            for glyph in text
                        ]
                    elif text:
                        boxes = split_line_bbox_into_char_bboxes(line.bbox, text)
                        stats.fallback_lines += 1
                        stats.fallback_chars += len(text)
                        line.chars = _complete_positional_chars(line, text, boxes)

                if line.bbox != old_line_bbox:
                    page_line_updates += 1
                    page_changed = True

                new_char_boxes = [
                    char.bbox.to_dict() if char.bbox is not None else None
                    for char in line.chars
                ]
                if new_char_boxes != old_char_boxes:
                    page_char_updates += 1
                    page_changed = True

        if page_changed:
            stats.pages += 1
        stats.line_bbox_updates += page_line_updates
        stats.char_bbox_updates += page_char_updates
