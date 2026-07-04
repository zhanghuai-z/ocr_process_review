"""Paddle layout artifact lookup for manual annotation boxes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.bbox_extraction import bbox_from_variant
from app.models.block_state import set_paddle_binding
from app.models.ocr_observation import clear_block_ocr_lines, replace_block_ocr_lines
from app.models.ocr_text_observation import create_ocr_text_line
from app.core.paddle_labels import is_hanwang_skip_label, normalize_paddle_label
from app.core.paddle_line_routing import (
    block_bbox_xyxy,
    block_text,
    is_formula_label,
    is_formula_style_position_block,
    is_table_label,
    route_authority_label,
    route_subblocks_for_block,
    vertical_overlap_ratio,
)
from app.core.normalized_layout_artifact import LayoutRegion, normalized_layout_regions
from app.models import BBox, Block, BlockOrigin, BlockType, OcrPolicy
from app.models.layout_block_state import set_layout_block_ocr_policy


XYXY = tuple[int, int, int, int]

BINDING_GEOMETRY_HIT = "paddle_geometry_hit"
BINDING_PARENT_FORMULA_INFERRED = "paddle_parent_formula_inferred"
BINDING_PARENT_TABLE_HIT = "paddle_parent_table_hit"
BINDING_PARENT_FIGURE_HIT = "paddle_parent_figure_hit"
BINDING_FORMULA_CROP_OCR = "paddle_formula_crop_ocr"
BINDING_AMBIGUOUS = "paddle_binding_ambiguous"
BINDING_EMPTY_REVIEW = "paddle_empty_review"


def _formula_spans(text: str) -> list[str]:
    # Keep this local to avoid widening paddle_line_routing's public API while
    # using the same delimiter rule for user-facing binding.
    import re

    pattern = re.compile(
        r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
        re.DOTALL,
    )
    return [match.group(0) for match in pattern.finditer(text or "")]


def _area(box: XYXY) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersect(a: XYXY, b: XYXY) -> XYXY | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(a: XYXY, b: XYXY) -> float:
    overlap = _intersect(a, b)
    if overlap is None:
        return 0.0
    inter_area = _area(overlap)
    union_area = _area(a) + _area(b) - inter_area
    return inter_area / max(1, union_area)


def _coverage(target: XYXY, candidate: XYXY) -> float:
    overlap = _intersect(target, candidate)
    if overlap is None:
        return 0.0
    return _area(overlap) / max(1, _area(candidate))


def _manual_overlap_score(manual: XYXY, candidate: XYXY) -> float:
    center_inside = _center_inside(candidate, manual)
    return (
        _coverage(manual, candidate) * 0.55
        + _iou(manual, candidate) * 0.25
        + vertical_overlap_ratio(manual, candidate) * 0.15
        + (0.05 if center_inside else 0.0)
    )


def _parent_overlap_score(manual: XYXY, parent: XYXY) -> float:
    overlap = _intersect(manual, parent)
    if overlap is None:
        return 0.0
    manual_coverage = _area(overlap) / max(1, _area(manual))
    return manual_coverage + (0.25 if _center_inside(manual, parent) else 0.0)


def _center_inside(inner: XYXY, outer: XYXY) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _reading_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: (value["bbox"][1], value["bbox"][0])):
        item_bbox = tuple(item["bbox"])
        for bucket in buckets:
            bucket_bbox = _union([tuple(value["bbox"]) for value in bucket])
            if vertical_overlap_ratio(bucket_bbox, item_bbox) >= 0.25:
                bucket.append(item)
                break
        else:
            buckets.append([item])

    ordered: list[dict[str, Any]] = []
    for bucket in sorted(
        buckets,
        key=lambda values: (
            min(value["bbox"][1] for value in values),
            min(value["bbox"][0] for value in values),
        ),
    ):
        ordered.extend(sorted(bucket, key=lambda value: (value["bbox"][0], value["bbox"][1])))
    return ordered


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _xyxy_from_bbox(bbox: BBox) -> XYXY:
    return tuple(int(value) for value in bbox.to_xyxy())


def _figure_like_label(label: object) -> bool:
    normalized = normalize_paddle_label(label)
    return normalized in {
        "figure",
        "graphic",
        "image",
        "picture",
        "photo",
        "chart",
        "seal",
        "stamp",
    }


@dataclass(frozen=True)
class PaddleParentArtifact:
    index: int
    label: str
    bbox: XYXY
    text: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def formula_spans(self) -> list[str]:
        return _formula_spans(self.text)

    @property
    def is_formula(self) -> bool:
        return is_formula_label(self.label) or is_formula_style_position_block(self.raw)

    @property
    def is_table(self) -> bool:
        return is_table_label(self.label)

    @property
    def is_figure(self) -> bool:
        return _figure_like_label(self.label)


@dataclass(frozen=True)
class PaddleGeometryArtifact:
    kind: str
    label: str
    bbox: XYXY
    parent_index: int
    raw: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    span_index: int = -1


@dataclass(frozen=True)
class PaddleManualBinding:
    status: str
    block_type: BlockType
    source: str
    source_label: str
    text: str = ""
    parent_index: int = -1
    candidate_index: int = -1
    score: float = 0.0
    candidate_bbox: XYXY | None = None
    manual_bbox: XYXY | None = None
    review_flags: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    @property
    def is_bound(self) -> bool:
        return self.status not in {BINDING_EMPTY_REVIEW, BINDING_AMBIGUOUS}

    @property
    def ocr_policy(self) -> OcrPolicy:
        if self.block_type in {BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE, BlockType.UNKNOWN}:
            if self.block_type == BlockType.EQUATION:
                return OcrPolicy.PRESERVE_AS_FORMULA
            if self.block_type == BlockType.TABLE:
                return OcrPolicy.PRESERVE_AS_TABLE
            return OcrPolicy.SKIP
        return OcrPolicy.MANUAL_ONLY if self.is_bound else OcrPolicy.TEXT_OCR

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "source": self.source,
            "block_type": self.block_type.value,
            "source_label": self.source_label,
            "text": self.text,
            "parent_index": self.parent_index,
            "candidate_index": self.candidate_index,
            "score": round(float(self.score), 6),
            "review_flags": list(self.review_flags),
        }
        if self.candidate_bbox is not None:
            payload["candidate_bbox"] = list(self.candidate_bbox)
        if self.manual_bbox is not None:
            payload["manual_bbox"] = list(self.manual_bbox)
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        return payload


class PaddleArtifactIndex:
    """Index Paddle parent truth and geometry candidates for a page."""

    def __init__(self, page_width: int, page_height: int, records: list[dict[str, Any]]) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self.parents: list[PaddleParentArtifact] = []
        self.formula_geometry: list[PaddleGeometryArtifact] = []
        self._build(records)

    @classmethod
    def from_page(cls, page) -> "PaddleArtifactIndex":
        index = cls(page.width, page.height, [])
        index._build_regions(list(normalized_layout_regions(page)))
        return index

    def _build(self, records: list[dict[str, Any]]) -> None:
        for index, record in enumerate(records):
            label = route_authority_label(record)
            bbox = block_bbox_xyxy(record, self.page_width, self.page_height)
            parent = PaddleParentArtifact(
                index=index,
                label=label,
                bbox=bbox,
                text=block_text(record),
                raw=dict(record),
            )
            self.parents.append(parent)
            self._collect_formula_geometry(parent, record)

    def _build_regions(self, regions: list[LayoutRegion]) -> None:
        for region in regions:
            raw = dict(region.raw or {})
            parent = PaddleParentArtifact(
                index=region.index,
                label=region.label,
                bbox=region.bbox,
                text=region.text,
                raw=raw,
            )
            self.parents.append(parent)
            self._collect_formula_geometry_from_region(parent, region)

    def _collect_formula_geometry(self, parent: PaddleParentArtifact, record: dict[str, Any]) -> None:
        subblocks = [
            subblock
            for subblock in route_subblocks_for_block(record, self.page_width, self.page_height)
            if is_formula_label(subblock["label"])
        ]
        spans = parent.formula_spans
        ordered = _reading_order([
            {"bbox": subblock["bbox"], "subblock": subblock}
            for subblock in subblocks
        ])
        count_matches = len(spans) == len(ordered)
        for local_index, item in enumerate(ordered):
            subblock = item["subblock"]
            text = str(subblock.get("text") or "")
            span_index = local_index if local_index < len(spans) else -1
            if not text and count_matches and span_index >= 0:
                text = spans[span_index]
            self.formula_geometry.append(
                PaddleGeometryArtifact(
                    kind="formula",
                    label=str(subblock["label"]),
                    bbox=tuple(subblock["bbox"]),
                    parent_index=parent.index,
                    raw=dict(subblock.get("raw") or {}),
                    text=text,
                    span_index=span_index,
                )
            )

    def _collect_formula_geometry_from_region(
        self,
        parent: PaddleParentArtifact,
        region: LayoutRegion,
    ) -> None:
        subblocks = [
            subregion
            for subregion in region.subregions
            if is_formula_label(subregion.label)
        ]
        spans = parent.formula_spans
        ordered = _reading_order([
            {"bbox": subregion.bbox, "subblock": subregion}
            for subregion in subblocks
        ])
        count_matches = len(spans) == len(ordered)
        for local_index, item in enumerate(ordered):
            subregion = item["subblock"]
            text = str(subregion.text or "")
            span_index = local_index if local_index < len(spans) else -1
            if not text and count_matches and span_index >= 0:
                text = spans[span_index]
            self.formula_geometry.append(
                PaddleGeometryArtifact(
                    kind="formula",
                    label=str(subregion.label),
                    bbox=tuple(subregion.bbox),
                    parent_index=parent.index,
                    raw=dict(subregion.raw or {}),
                    text=text,
                    span_index=span_index,
                )
            )

    def bind_manual_bbox(self, bbox: BBox, block_type: BlockType) -> PaddleManualBinding:
        manual = _xyxy_from_bbox(bbox)
        if block_type == BlockType.EQUATION:
            return self._bind_formula(manual)
        if block_type == BlockType.TABLE:
            return self._bind_parent_artifact(manual, BlockType.TABLE)
        if block_type == BlockType.FIGURE:
            return self._bind_parent_artifact(manual, BlockType.FIGURE)
        return PaddleManualBinding(
            status=BINDING_EMPTY_REVIEW,
            block_type=block_type,
            source="unsupported_manual_type",
            source_label=block_type.value,
            manual_bbox=manual,
        )

    def _bind_formula(self, manual: XYXY) -> PaddleManualBinding:
        geometry_hits = self._geometry_hits(manual, self.formula_geometry)
        if len(geometry_hits) == 1:
            candidate_index, candidate, score = geometry_hits[0]
            text = candidate.text
            flags: tuple[str, ...] = ("manual_paddle_binding",)
            if not text:
                flags = (*flags, "manual_formula_needs_text")
            return PaddleManualBinding(
                status=BINDING_GEOMETRY_HIT,
                block_type=BlockType.EQUATION,
                source="paddle_geometry",
                source_label=candidate.label or "inline_formula",
                text=text,
                parent_index=candidate.parent_index,
                candidate_index=candidate_index,
                score=score,
                candidate_bbox=candidate.bbox,
                manual_bbox=manual,
                review_flags=flags,
            )
        if len(geometry_hits) > 1:
            return self._ambiguous_formula(manual, geometry_hits)

        parent = self._best_parent_for_manual_formula(manual)
        if parent is None:
            return self._empty_formula(manual)
        return self._empty_formula(manual, parent_index=parent.index)

    def _geometry_hits(
        self,
        manual: XYXY,
        candidates: list[PaddleGeometryArtifact],
    ) -> list[tuple[int, PaddleGeometryArtifact, float]]:
        hits: list[tuple[int, PaddleGeometryArtifact, float]] = []
        for index, candidate in enumerate(candidates):
            score = _manual_overlap_score(manual, candidate.bbox)
            if score >= 0.35 and _coverage(manual, candidate.bbox) >= 0.35:
                hits.append((index, candidate, score))
        hits.sort(key=lambda item: item[2], reverse=True)
        return hits

    def _span_for_geometry_candidate(self, candidate: PaddleGeometryArtifact) -> str:
        parent = self.parents[candidate.parent_index] if 0 <= candidate.parent_index < len(self.parents) else None
        if parent is None:
            return ""
        spans = parent.formula_spans
        parent_geometries = [
            item for item in self.formula_geometry
            if item.parent_index == candidate.parent_index
        ]
        ordered = _reading_order([
            {"bbox": item.bbox, "candidate": item}
            for item in parent_geometries
        ])
        for index, item in enumerate(ordered):
            if item["candidate"] is candidate and index < len(spans):
                return spans[index]
        return ""

    def _span_for_manual_insert(
        self,
        parent: PaddleParentArtifact,
        parent_geometries: list[PaddleGeometryArtifact],
        manual: XYXY,
    ) -> str:
        spans = parent.formula_spans
        ordered = _reading_order(
            [
                {"bbox": candidate.bbox, "manual": False}
                for candidate in parent_geometries
            ]
            + [{"bbox": manual, "manual": True}]
        )
        manual_index = next(
            (index for index, item in enumerate(ordered) if item.get("manual")),
            -1,
        )
        if 0 <= manual_index < len(spans):
            return spans[manual_index]
        return ""

    def _best_parent_for_manual_formula(self, manual: XYXY) -> PaddleParentArtifact | None:
        candidates: list[tuple[PaddleParentArtifact, float]] = []
        for parent in self.parents:
            if parent.is_formula or parent.is_table or parent.is_figure:
                continue
            label = normalize_paddle_label(parent.label)
            if is_hanwang_skip_label(label):
                continue
            score = _parent_overlap_score(manual, parent.bbox)
            if score >= 0.1:
                candidates.append((parent, score))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    def _bind_parent_artifact(self, manual: XYXY, block_type: BlockType) -> PaddleManualBinding:
        if block_type == BlockType.TABLE:
            parents = [parent for parent in self.parents if parent.is_table]
            status = BINDING_PARENT_TABLE_HIT
            source = "manual_geometry+paddle_parent_table"
            label_fallback = "table"
            flags = ("manual_table_from_paddle_parent",)
        else:
            parents = [parent for parent in self.parents if parent.is_figure]
            status = BINDING_PARENT_FIGURE_HIT
            source = "manual_geometry+paddle_parent_figure"
            label_fallback = "figure"
            flags = ("manual_figure_from_paddle_parent",)

        hits = [
            (parent, _parent_overlap_score(manual, parent.bbox))
            for parent in parents
            if _parent_overlap_score(manual, parent.bbox) >= 0.25
        ]
        hits.sort(key=lambda item: item[1], reverse=True)
        if len(hits) == 1:
            parent, score = hits[0]
            review_flags = flags
            if not parent.text:
                review_flags = (*review_flags, f"manual_{block_type.value}_needs_text")
            return PaddleManualBinding(
                status=status,
                block_type=block_type,
                source=source,
                source_label=parent.label or label_fallback,
                text=parent.text,
                parent_index=parent.index,
                score=score,
                candidate_bbox=parent.bbox,
                manual_bbox=manual,
                review_flags=review_flags,
            )
        if len(hits) > 1:
            return PaddleManualBinding(
                status=BINDING_AMBIGUOUS,
                block_type=block_type,
                source=f"manual_geometry+ambiguous_paddle_parent_{block_type.value}",
                source_label=label_fallback,
                manual_bbox=manual,
                review_flags=(f"manual_{block_type.value}_ambiguous",),
                candidates=tuple(parent.text for parent, _score in hits if parent.text),
            )
        return PaddleManualBinding(
            status=BINDING_EMPTY_REVIEW,
            block_type=block_type,
            source=f"manual_geometry_empty_{block_type.value}_review",
            source_label=label_fallback,
            manual_bbox=manual,
            review_flags=(f"manual_{block_type.value}_needs_text",),
        )

    def _ambiguous_formula(
        self,
        manual: XYXY,
        geometry_hits: list[tuple[int, PaddleGeometryArtifact, float]],
    ) -> PaddleManualBinding:
        return PaddleManualBinding(
            status=BINDING_AMBIGUOUS,
            block_type=BlockType.EQUATION,
            source="manual_geometry+ambiguous_paddle_geometry",
            source_label="inline_formula",
            manual_bbox=manual,
            review_flags=("manual_formula_ambiguous",),
            candidates=tuple(candidate.text or candidate.label for _index, candidate, _score in geometry_hits),
        )

    def _ambiguous_parent_formula(
        self,
        manual: XYXY,
        parent: PaddleParentArtifact,
    ) -> PaddleManualBinding:
        return PaddleManualBinding(
            status=BINDING_AMBIGUOUS,
            block_type=BlockType.EQUATION,
            source="manual_geometry+ambiguous_paddle_parent_text",
            source_label="inline_formula",
            parent_index=parent.index,
            score=_parent_overlap_score(manual, parent.bbox),
            manual_bbox=manual,
            review_flags=("manual_formula_ambiguous",),
            candidates=tuple(parent.formula_spans),
        )

    def _empty_formula(self, manual: XYXY, parent_index: int = -1) -> PaddleManualBinding:
        return PaddleManualBinding(
            status=BINDING_EMPTY_REVIEW,
            block_type=BlockType.EQUATION,
            source="manual_geometry_empty_formula_review",
            source_label="inline_formula",
            parent_index=parent_index,
            manual_bbox=manual,
            review_flags=("manual_formula_needs_text",),
        )


def apply_paddle_binding_to_block(block: Block, binding: PaddleManualBinding) -> None:
    """Persist a binding on a layout block without changing its geometry."""
    existing_origin = block.origin
    block.origin = BlockOrigin(
        created_by=existing_origin.created_by if existing_origin else block.source.value,
        source_engine=existing_origin.source_engine if existing_origin else "paddleocr-vl",
        source_run_id=existing_origin.source_run_id if existing_origin else "",
        source_label=binding.source_label or (existing_origin.source_label if existing_origin else ""),
        source_confidence=binding.score if binding.score else (existing_origin.source_confidence if existing_origin else None),
        original_bbox=existing_origin.original_bbox if existing_origin and existing_origin.original_bbox else block.bbox,
        original_kind=existing_origin.original_kind if existing_origin and existing_origin.original_kind else block.block_type,
        raw_artifact_uid=existing_origin.raw_artifact_uid if existing_origin else "",
        raw_json_path=existing_origin.raw_json_path if existing_origin else "",
        raw_index=binding.parent_index if binding.parent_index >= 0 else (existing_origin.raw_index if existing_origin else None),
    )
    set_paddle_binding(block, binding.to_payload())
    block.source_label = binding.source_label or block.source_label or block.block_type.value
    set_layout_block_ocr_policy(block, binding.ocr_policy)
    if binding.text:
        replace_block_ocr_lines(block, [
            create_ocr_text_line(
                text=binding.text,
                confidence=0.0,
                bbox=block.bbox,
                source_text=binding.text,
                review_flags=list(binding.review_flags),
            )
        ])
    else:
        clear_block_ocr_lines(block)
    if binding.status == BINDING_EMPTY_REVIEW:
        block.note = binding.source
    elif binding.status == BINDING_AMBIGUOUS:
        block.note = "manual_binding_ambiguous"


__all__ = [
    "BINDING_AMBIGUOUS",
    "BINDING_EMPTY_REVIEW",
    "BINDING_FORMULA_CROP_OCR",
    "BINDING_GEOMETRY_HIT",
    "BINDING_PARENT_FIGURE_HIT",
    "BINDING_PARENT_FORMULA_INFERRED",
    "BINDING_PARENT_TABLE_HIT",
    "PaddleArtifactIndex",
    "PaddleGeometryArtifact",
    "PaddleManualBinding",
    "PaddleParentArtifact",
    "apply_paddle_binding_to_block",
]
