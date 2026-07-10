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
from app.core.proof_char_text import is_display_carrier
from app.core.proof_geometry_quality import is_estimated_or_unavailable_geometry
from app.core.proof_line_facts import proof_display_text
from app.core.proof_line_utils import iter_unique_page_text_lines
from app.models import BBox, Block, Char, Line, OcrProject, Page
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.models.ocr_character_observation import line_ocr_chars_by_uid, replace_line_ocr_char_observations
from app.models.ocr_observation import block_ocr_line_observations_by_uid, line_ocr_bbox
from app.models.ocr_text_observation import line_has_ocr_review_flag, line_ocr_confidence
from app.services.ocr_dispatch_plan import build_text_ocr_dispatch_plan
from app.utils.image_io import read_cv_image

INLINE_FORMULA_REVIEW_FLAG = "hanwang_route_inline_formula"


def _fallback_char(glyph: str, line: Line, bbox: BBox | None, source: str, granularity: str) -> Char:
    return Char(
        char=glyph,
        confidence=line_ocr_confidence(line),
        bbox=bbox,
        bbox_source=source,
        bbox_granularity=granularity,
        token_text=glyph,
    )


def _complete_positional_chars(line: Line, text: str, boxes: list[BBox] | None) -> list[Char]:
    completed: list[Char] = []
    existing = list(line_ocr_chars_by_uid(line.uid))
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


def proof_fallback_warning(stats: ProofCropStats, pages: Iterable[Page] | None = None) -> str:
    """Return a user-visible warning for proof geometry fallback state."""

    fallback_total = int(stats.fallback_chars) + int(stats.unavailable_chars)
    fallback_lines = int(stats.fallback_lines)
    if fallback_total <= 0 and pages:
        seen_lines: set[int] = set()
        for page in pages:
            if layout_snapshot_for_page(page) is None:
                continue
            for _block, line, _line_idx in iter_unique_page_text_lines(page):
                line_fallback_chars = 0
                for char in line_ocr_chars_by_uid(line.uid):
                    if is_estimated_or_unavailable_geometry(char.bbox_source, char.bbox_granularity):
                        line_fallback_chars += 1
                if line_fallback_chars:
                    fallback_total += line_fallback_chars
                    if id(line) not in seen_lines:
                        fallback_lines += 1
                        seen_lines.add(id(line))
    if fallback_total <= 0:
        return ""
    return (
        f"警告：proof fallback {fallback_lines} 行/"
        f"{fallback_total} 字，字框为估算或不可用"
    )


class ProofCropService:
    """统一收口 proof 使用的行框/字框几何。"""

    def normalize_project(self, project: OcrProject) -> ProofCropStats:
        return self.normalize_pages(project.pages)

    def normalize_pages(self, pages: Iterable[Page]) -> ProofCropStats:
        stats = ProofCropStats()
        for page in pages:
            self._normalize_page(page, stats)
        return stats

    def normalize_block(self, block: Block) -> ProofCropStats:
        stats = ProofCropStats()
        self._normalize_block_uid(block.uid, stats)
        return stats

    def _normalize_page(self, page: Page, stats: ProofCropStats) -> None:
        # Imported pages have no adopted layout or OCR observations yet.  They
        # are valid project state, but not proof input.
        if layout_snapshot_for_page(page) is None:
            return
        image = read_cv_image(page.display_image_path, cv2.IMREAD_COLOR)
        if image is None:
            return

        page_changed = False

        dispatch_plan = build_text_ocr_dispatch_plan(page)
        for target in dispatch_plan.text_blocks:
            page_changed = self._normalize_block_uid(target.view.uid, stats) or page_changed

        if page_changed:
            stats.pages += 1

    def _normalize_block_uid(self, block_uid: str, stats: ProofCropStats) -> bool:
        changed = False
        for line in block_ocr_line_observations_by_uid(block_uid):
            stats.lines += 1
            old_line_bbox = line_ocr_bbox(line)
            old_char_boxes = [
                char.bbox.to_dict() if char.bbox is not None else None
                for char in line_ocr_chars_by_uid(line.uid)
            ]

            chars = line_ocr_chars_by_uid(line.uid)
            has_tokenized_chars = any(is_display_carrier(char) for char in chars)
            needs_fallback_chars = (
                not chars
                or (
                    not has_tokenized_chars
                    and len(chars) != len(proof_display_text(line))
                )
            )
            if needs_fallback_chars and not line_has_ocr_review_flag(line, INLINE_FORMULA_REVIEW_FLAG):
                text = proof_display_text(line)
                if text and line_has_ocr_review_flag(line, MISSING_LINE_BBOX_FLAG):
                    stats.fallback_lines += 1
                    stats.unavailable_chars += len(text)
                    replace_line_ocr_char_observations(line.uid, [
                        _fallback_char(
                            glyph,
                            line,
                            None,
                            BBOX_SOURCE_UNAVAILABLE,
                            BBOX_GRANULARITY_UNAVAILABLE,
                        )
                        for glyph in text
                    ])
                elif text:
                    boxes = split_line_bbox_into_char_bboxes(line_ocr_bbox(line), text)
                    stats.fallback_lines += 1
                    stats.fallback_chars += len(text)
                    replace_line_ocr_char_observations(
                        line.uid,
                        _complete_positional_chars(line, text, boxes),
                    )

            if line_ocr_bbox(line) != old_line_bbox:
                stats.line_bbox_updates += 1
                changed = True

            new_char_boxes = [
                char.bbox.to_dict() if char.bbox is not None else None
                for char in line_ocr_chars_by_uid(line.uid)
            ]
            if new_char_boxes != old_char_boxes:
                stats.char_bbox_updates += 1
                changed = True
        return changed
