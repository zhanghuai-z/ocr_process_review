from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2

from app.core.char_bbox_utils import ensure_line_char_bboxes, refine_line_bbox
from app.models import OcrProject, Page

INLINE_FORMULA_REVIEW_FLAG = "hanwang_route_inline_formula"


@dataclass
class ProofCropStats:
    pages: int = 0
    lines: int = 0
    line_bbox_updates: int = 0
    char_bbox_updates: int = 0


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
            for line in block.lines:
                stats.lines += 1
                old_line_bbox = line.bbox
                old_char_boxes = [
                    char.bbox.to_dict() if char.bbox is not None else None
                    for char in line.chars
                ]

                line.bbox = refine_line_bbox(line.bbox, image)
                if INLINE_FORMULA_REVIEW_FLAG not in line.review_flags:
                    ensure_line_char_bboxes(line, page_image=image)

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
