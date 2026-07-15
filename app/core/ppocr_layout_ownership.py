"""Assign PP-OCR line observations to the adopted layout snapshot.

This stage owns only layout facts: which snapshot block contains an observed
line, which structural block excludes it, and which text blocks form the
normalization domains. It does not classify text or create CharOCR routes.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint
from app.core.ppocr_line_geometry import TextBlockLineGroup
from app.models.enums import BlockType, OcrPolicy
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot


XYXY = tuple[int, int, int, int]
MIN_TEXT_OWNERSHIP_COVERAGE = 0.1
STRUCTURAL_BLOCK_TYPES = frozenset({
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.UNKNOWN,
})


@dataclass(frozen=True)
class LayoutBlockCandidate:
    """A snapshot block with page-clamped geometry for this routing run."""

    block: LayoutBlockSnapshot
    bbox: XYXY


@dataclass(frozen=True)
class LayoutLineOwnership:
    """Deterministic ownership decision for one physical line observation."""

    line: PpOcrV6LineHint
    line_bbox: XYXY
    text_block: LayoutBlockCandidate | None
    structural_block: LayoutBlockCandidate | None

    @property
    def is_vertical_text(self) -> bool:
        return self.text_block is not None and _is_vertical_text_block(self.text_block.block)

    @property
    def is_structural_exclusion(self) -> bool:
        if self.structural_block is None:
            return False
        if self.text_block is None:
            return True
        return _overlap_score(self.line_bbox, self.structural_block.bbox) >= _overlap_score(
            self.line_bbox,
            self.text_block.bbox,
        )


@dataclass(frozen=True)
class LayoutOwnership:
    """Layout ownership context and raw-line decisions for one page."""

    text_blocks: tuple[LayoutBlockCandidate, ...]
    structural_blocks: tuple[LayoutBlockCandidate, ...]
    decisions: tuple[LayoutLineOwnership, ...]
    page_width: int
    page_height: int

    def ownership_for(self, line: PpOcrV6LineHint) -> LayoutLineOwnership:
        """Re-evaluate ownership after physical-row geometry is normalized."""
        bounded_line = _replace_bbox(
            line,
            _clamp(line.bbox, self.page_width, self.page_height),
        )
        return _decide_line(
            bounded_line,
            self.text_blocks,
            self.structural_blocks,
        )

    def text_line_groups(self) -> tuple[TextBlockLineGroup, ...]:
        """Return raw PP rows grouped only within their text block owner."""
        grouped: dict[str, tuple[LayoutBlockCandidate, list[PpOcrV6LineHint]]] = {}
        for decision in self.decisions:
            if (
                decision.text_block is None
                or decision.is_vertical_text
                or decision.is_structural_exclusion
            ):
                continue
            entry = grouped.setdefault(decision.text_block.block.uid, (decision.text_block, []))
            entry[1].append(
                _replace_bbox(decision.line, decision.line_bbox)
            )
        return tuple(
            TextBlockLineGroup(
                block_uid=block_uid,
                block_bbox=target.bbox,
                lines=tuple(lines),
                excluded_bboxes=tuple(
                    candidate.bbox
                    for candidate in self.structural_blocks
                    if _intersect(candidate.bbox, target.bbox) is not None
                ),
            )
            for block_uid, (target, lines) in grouped.items()
        )


def build_layout_ownership(
    snapshot: LayoutSnapshot,
    lines: tuple[PpOcrV6LineHint, ...],
    *,
    page_width: int,
    page_height: int,
) -> LayoutOwnership:
    """Build layout-only ownership context from the authoritative snapshot."""
    text_blocks: list[LayoutBlockCandidate] = []
    structural_blocks: list[LayoutBlockCandidate] = []
    for block in snapshot.blocks:
        candidate = LayoutBlockCandidate(
            block=block,
            bbox=_clamp(block.bbox.to_xyxy(), page_width, page_height),
        )
        if _is_text_dispatch_block(block):
            text_blocks.append(candidate)
        else:
            structural_blocks.append(candidate)
    typed_text_blocks = tuple(text_blocks)
    typed_structural_blocks = tuple(structural_blocks)
    decisions = tuple(
        _decide_line(
            _replace_bbox(line, _clamp(line.bbox, page_width, page_height)),
            typed_text_blocks,
            typed_structural_blocks,
        )
        for line in lines
    )
    return LayoutOwnership(
        text_blocks=typed_text_blocks,
        structural_blocks=typed_structural_blocks,
        decisions=decisions,
        page_width=page_width,
        page_height=page_height,
    )


def structural_masks_for_line(
    line_bbox: XYXY,
    structural_blocks: tuple[LayoutBlockCandidate, ...],
) -> tuple[tuple[LayoutBlockSnapshot, XYXY, XYXY], ...]:
    """Return exact structural intersections and their canonical bboxes."""
    masks = [
        (candidate.block, overlap, candidate.bbox)
        for candidate in structural_blocks
        if (overlap := _intersect(line_bbox, candidate.bbox)) is not None
    ]
    return tuple(sorted(masks, key=lambda item: (item[1][0], item[1][1], item[0].order)))


def _decide_line(
    line: PpOcrV6LineHint,
    text_blocks: tuple[LayoutBlockCandidate, ...],
    structural_blocks: tuple[LayoutBlockCandidate, ...],
) -> LayoutLineOwnership:
    line_bbox = line.bbox
    text_block = _select_text_container(line_bbox, text_blocks)
    structural_block = _select_structural_owner(line_bbox, structural_blocks)
    return LayoutLineOwnership(
        line=line,
        line_bbox=line_bbox,
        text_block=text_block,
        structural_block=structural_block,
    )


def _select_text_container(
    line_bbox: XYXY,
    candidates: tuple[LayoutBlockCandidate, ...],
) -> LayoutBlockCandidate | None:
    best: tuple[float, int, LayoutBlockCandidate] | None = None
    for candidate in candidates:
        score = _overlap_score(line_bbox, candidate.bbox)
        if score <= 0:
            continue
        ranked = (score, -candidate.block.order, candidate)
        if best is None or ranked[:2] > best[:2]:
            best = ranked
    return best[2] if best is not None and best[0] >= MIN_TEXT_OWNERSHIP_COVERAGE else None


def _select_structural_owner(
    line_bbox: XYXY,
    candidates: tuple[LayoutBlockCandidate, ...],
) -> LayoutBlockCandidate | None:
    """Select a structure containing the line center, with layout tie order."""
    center_x = (line_bbox[0] + line_bbox[2]) / 2.0
    center_y = (line_bbox[1] + line_bbox[3]) / 2.0
    best: tuple[float, int, LayoutBlockCandidate] | None = None
    for candidate in candidates:
        bbox = candidate.bbox
        if not (bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]):
            continue
        ranked = (_overlap_score(line_bbox, bbox), -candidate.block.order, candidate)
        if best is None or ranked[:2] > best[:2]:
            best = ranked
    return best[2] if best is not None else None


def _is_text_dispatch_block(block: LayoutBlockSnapshot) -> bool:
    return block.ocr_policy == OcrPolicy.TEXT_OCR and block.block_type not in STRUCTURAL_BLOCK_TYPES


def _is_vertical_text_block(block: LayoutBlockSnapshot) -> bool:
    return str(block.source_label or "").strip().lower() == "vertical_text"


def _overlap_score(line_bbox: XYXY, candidate_bbox: XYXY) -> float:
    overlap = _intersect(line_bbox, candidate_bbox)
    return _area(overlap) / max(1, _area(line_bbox)) if overlap is not None else 0.0


def _replace_bbox(line: PpOcrV6LineHint, bbox: XYXY) -> PpOcrV6LineHint:
    if line.bbox == bbox:
        return line
    return PpOcrV6LineHint(
        index=line.index,
        text=line.text,
        bbox=bbox,
        words=line.words,
    )


def _clamp(bbox: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _area(bbox: XYXY | None) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


__all__ = [
    "LayoutBlockCandidate",
    "LayoutLineOwnership",
    "LayoutOwnership",
    "MIN_TEXT_OWNERSHIP_COVERAGE",
    "STRUCTURAL_BLOCK_TYPES",
    "build_layout_ownership",
    "structural_masks_for_line",
]
