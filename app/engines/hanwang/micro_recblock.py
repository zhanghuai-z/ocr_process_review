"""Hanwang native linecut micro-recblock adapter."""
from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.geometry.char_reconciler import reconcile_char_geometry
from app.models.charocr_routing import (
    PageRoutingPlan,
    PpOcrLatinTokenObservation,
    PpOcrSymbolObservation,
    VlSemanticMarkerObservation,
    ROUTE_SEGMENT_TEXT_LATIN,
    TEXT_AXIS_HORIZONTAL,
    TEXT_AXIS_VERTICAL,
    RoutingLine,
    is_text_route_segment_kind,
)
from app.models.char_geometry import NativeGeometryProposal
from app.models.charocr_execution import (
    CharOcrAtomObservation,
    CharOcrCandidateObservation,
    CharOcrInputRow,
    CharOcrLineObservation,
    CharOcrPageRequest,
    CharOcrPageResult,
    CharOcrRegionObservation,
)
from app.models.enums import OcrPolicy
from app.core.logging import get_logger
from .engcut_payload import (
    EngcutChar,
    engcut_chars_from_payload,
    offset_engcut_chars,
)
from .geometry_postprocess import (
    LineAtomGeometry,
    conservative_cjk_bbox_cleanup,
    is_latin_right_slant_fallback,
    measure_latin_right_slant,
)


_ENGCUT_NATIVE_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="charocr-engcut",
)
_MAX_ENGCUT_LINES_PER_PAGE = 4
from app.core.paddle_line_routing import (
    ROUTE_INLINE_FORMULA_FLAG,
    ROUTE_TABLE_FLAG,
    block_bbox_xyxy,
    is_formula_label,
    is_table_label,
    route_authority_label,
    union_xyxy,
    vertical_overlap_ratio,
)
from app.core.paddle_labels import (
    PADDLE_HANWANG_SKIP_LABELS,
    PADDLE_HANWANG_TEXT_LABELS,
    is_hanwang_skip_label,
    normalize_paddle_label,
)

from . import native_bridge

logger = get_logger(__name__)


ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY = "_hanwang_bbox_audit"

TEXT_LABELS: set[str] = set(PADDLE_HANWANG_TEXT_LABELS)
SKIP_LABELS: set[str] = set(PADDLE_HANWANG_SKIP_LABELS)
DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG = "hanwang_digitlike_numeric_context"
LATIN_ENGCUT_ROUTE_SOURCE = "hanwang:EngCut:latin_route"
PPOCR_LATIN_TOKEN_ALIGNMENT_SOURCE = "ppocrv6:latin_token_text_alignment"
PPOCR_LATIN_TOKEN_DISAGREEMENT_FLAG = "latin_token_text_disagreement"
PPOCR_LATIN_TOKEN_GEOMETRY_FALLBACK_SOURCE = (
    "ppocrv6:latin_token_text_route_foreground_geometry"
)
PPOCR_LATIN_TOKEN_GEOMETRY_FALLBACK_FLAG = "latin_token_geometry_fallback"
PPOCR_LATIN_ITALIC_WORD_FALLBACK_SOURCE = (
    "ppocrv6:latin_token_text_route_foreground_geometry:italic_postcheck"
)
PPOCR_LATIN_ITALIC_WORD_FALLBACK_FLAG = "latin_italic_word_fallback"
ENGCUT_DEGRADED_NATIVE_SOURCE = "hanwang:EngCut:latin_route:geometry_degraded_observation"
LATIN_EMPTY_NATIVE_FALLBACK_SOURCE = "ppocrv6:latin_route_empty_native"
LATIN_EMPTY_NATIVE_FALLBACK_FLAG = "latin_route_empty_native_ppocr_fallback"
PPOCR_SYMBOL_FOREGROUND_SOURCE = "ppocrv6:symbol_foreground_observation"
VL_SEMANTIC_MARKER_SOURCE = "paddlevl:semantic_marker"
LINECUT_NATIVE_ATOM_SOURCE = "hanwang:micro_recblock"
LINECUT_CJK_CLEANUP_SOURCE = "hanwang:micro_recblock:cjk_empty_seam_cleanup"
LINECUT_CJK_CLEANUP_FLAG = "linecut_cjk_empty_seam_bbox_cleanup"
_DIGITLIKE_ZERO_CHARS = {"o", "O"}
_DIGITLIKE_ONE_CHARS = {"l", "I"}
_DIGITLIKE_NUMERIC_CONTEXT_FOLLOWERS = {"", "，", ",", "。", ".", "；", ";", "、", ")", "）"}

MAX_RECOG_BATCH_GROUPS = int(os.environ.get("HANWANG_MICRO_RECBLOCK_BATCH_GROUPS", "64"))
_BATCH_ENABLED_BY_ENV = (
    os.environ.get("HANWANG_MICRO_RECBLOCK_BATCH", "").strip().lower()
    in {"1", "true", "yes", "on"}
)
_BATCH_DISABLED_FOR_SESSION = not _BATCH_ENABLED_BY_ENV
_BATCH_DISABLE_REASON = (
    ""
    if _BATCH_ENABLED_BY_ENV
    else "native batch-list disabled by default; set HANWANG_MICRO_RECBLOCK_BATCH=1 to enable"
)


@dataclass
class _NativeAtomResult:
    text: str
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    candidates: list[str] = field(default_factory=list)
    candidate_confidences: list[float] = field(default_factory=list)
    external_candidates: list[CharOcrCandidateObservation] = field(default_factory=list)
    source: str = "hanwang:micro_recblock"
    bbox_granularity: str = ""
    token_text: str = ""


@dataclass
class _NativeLineResult:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    chars: list[_NativeAtomResult] = field(default_factory=list)
    source: str = "hanwang"
    bbox_source: str = ""
    review_flags: list[str] = field(default_factory=list)


@dataclass
class _NativeRegionResult:
    block_idx: int
    block_uid: str
    block_label: str
    block_bbox: tuple[int, int, int, int]
    source: str
    text: str
    ppvl_text: str
    group_count: int = 0
    lines: list[_NativeLineResult] = field(default_factory=list)
    fallback_reason: str = ""
    raw_block: dict[str, Any] = field(default_factory=dict)
    layout_bbox: tuple[int, int, int, int] | None = None
    block_bbox_source: str = ""
    route_text_slice_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    recog_group_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    segimg_group_audits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_idx": self.block_idx,
            "block_uid": self.block_uid,
            "block_label": self.block_label,
            "block_bbox": list(self.block_bbox),
            "layout_bbox": list(self.layout_bbox) if self.layout_bbox else None,
            "block_bbox_source": self.block_bbox_source,
            "route_text_slice_bboxes": [list(bbox) for bbox in self.route_text_slice_bboxes],
            "recog_group_bboxes": [list(bbox) for bbox in self.recog_group_bboxes],
            "segimg_group_audits": [dict(item) for item in self.segimg_group_audits],
            "source": self.source,
            "text": self.text,
            "ppvl_text": self.ppvl_text,
            "group_count": self.group_count,
            "fallback_reason": self.fallback_reason,
            "raw_block": dict(self.raw_block),
            "lines": [
                {
                    "text": line.text,
                    "bbox": list(line.bbox),
                    "bbox_source": line.bbox_source,
                    "confidence": line.confidence,
                    "source": line.source,
                    "review_flags": list(line.review_flags),
                    "chars": [
                        {
                            "text": char.text,
                            "confidence": char.confidence,
                            "bbox": list(char.bbox) if char.bbox else None,
                            "candidates": list(char.candidates),
                            "external_candidates": [
                                {
                                    "text": candidate.text,
                                    "confidence": candidate.confidence,
                                    "source": candidate.source,
                                    "bbox": list(candidate.bbox) if candidate.bbox else None,
                                }
                                for candidate in char.external_candidates
                            ],
                            "source": char.source,
                            "bbox_granularity": char.bbox_granularity,
                            "token_text": char.token_text,
                        }
                        for char in line.chars
                    ],
                }
                for line in self.lines
            ],
        }


@dataclass
class RunStats:
    n_blocks_total: int = 0
    n_blocks_hanwang: int = 0
    n_blocks_ppvl: int = 0
    n_blocks_fallback: int = 0
    n_unknown_paddle_labels: int = 0
    n_groups: int = 0
    seg_seconds: float = 0.0
    recog_seconds: float = 0.0
    recog_full_page_pixels: int = 0
    recog_crop_pixels: int = 0
    recog_probe_calls: int = 0
    recog_group_failures: int = 0
    recog_group_retry_attempts: int = 0
    recog_group_retry_successes: int = 0
    recog_group_retry_failures: int = 0
    recog_batch_chunks: int = 0
    recog_batch_failures: int = 0
    recog_batch_disabled: bool = False
    recog_batch_guarded_chunks: int = 0
    recog_max_batch_crop_width: int = 0
    recog_max_batch_crop_height: int = 0
    recog_max_batch_crop_pixels: int = 0
    engcut_route_calls: int = 0
    geometry_conflict_groups: int = 0
    geometry_token_atoms: int = 0
    latin_empty_native_fallbacks: int = 0
    latin_token_geometry_fallbacks: int = 0
    latin_italic_word_fallbacks: int = 0
    latin_token_text_disagreements: int = 0
    linecut_cjk_bbox_cleanups: int = 0
    ppocr_symbol_candidates_bound: int = 0
    ppocr_symbol_atoms_inserted: int = 0
    ppocr_symbol_observations_unbound: int = 0
    vl_marker_observations_bound: int = 0
    vl_marker_observations_unbound: int = 0


@dataclass
class _GroupPlacement:
    area_idx: int
    page_bbox: tuple[int, int, int, int]
    group_bbox: tuple[int, int, int, int]
    native_core_height_target: int | None = None
    owner_key: tuple[int, int, int] | None = None
    owner_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class _PreparedRecogCrop:
    image: np.ndarray
    scale_x: float = 1.0
    scale_y: float = 1.0

    @property
    def normalized(self) -> bool:
        return self.scale_x != 1.0 or self.scale_y != 1.0


@dataclass(frozen=True)
class _OrientedNativeCrop:
    image: np.ndarray
    rotation_quarters_clockwise: int = 0
    source_width: int = 0
    source_height: int = 0

    def bbox_to_source(
        self,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        turns = self.rotation_quarters_clockwise % 4
        if turns == 0:
            return bbox
        if turns == 1:
            return (
                y1,
                self.source_height - x2,
                y2,
                self.source_height - x1,
            )
        if turns == 2:
            return (
                self.source_width - x2,
                self.source_height - y2,
                self.source_width - x1,
                self.source_height - y1,
            )
        return (
            self.source_width - y2,
            x1,
            self.source_width - y1,
            x2,
        )


@dataclass
class _TextRoute:
    block_idx: int
    line_idx: int
    segment_idx: int
    bbox: tuple[int, int, int, int]
    carved: bool = False
    kind: str = "text"
    ppocr_latin_fallback_text: str = ""
    ppocr_latin_tokens: tuple[PpOcrLatinTokenObservation, ...] = ()
    content_bbox: tuple[int, int, int, int] | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        return self.block_idx, self.line_idx, self.segment_idx


@dataclass(frozen=True)
class _EngCutMaskedLineRoute:
    """One physical routing line materialized as a Latin-only EngCut canvas.

    The routing plan remains segment-oriented.  This is only the native-input
    representation needed by EngCut: it keeps the PP-OCR line geometry while
    exposing pixels from Latin/digit segments while whitening every other
    region.
    """

    block_idx: int
    line_idx: int
    bbox: tuple[int, int, int, int]
    segments: tuple[_TextRoute, ...]
    text_axis: str = TEXT_AXIS_HORIZONTAL
    orientation_angle: int = -1
    ppocr_symbol_observations: tuple[PpOcrSymbolObservation, ...] = ()


@dataclass(frozen=True)
class _LineCutMaskedLineRoute:
    """One physical PP row with only LineCut-owned pixels exposed."""

    block_idx: int
    line_idx: int
    bbox: tuple[int, int, int, int]
    linecut_segments: tuple[_TextRoute, ...]
    excluded_segments: tuple[_TextRoute, ...]
    text_axis: str = TEXT_AXIS_HORIZONTAL
    orientation_angle: int = -1


def _is_engcut_text_route(route: _TextRoute) -> bool:
    return route.kind == ROUTE_SEGMENT_TEXT_LATIN


def _label_from_block(block: dict[str, Any], default: str = "unknown") -> str:
    return route_authority_label(block, default)


def _is_skip_label(label: str) -> bool:
    return is_hanwang_skip_label(label)


def _is_text_label(label: str) -> bool:
    return normalize_paddle_label(label) in TEXT_LABELS or not _is_skip_label(label)


def _is_unknown_hanwang_label(label: str) -> bool:
    normalized = normalize_paddle_label(label)
    if normalized in TEXT_LABELS or normalized in SKIP_LABELS:
        return False
    return not _is_skip_label(label)


def _block_text(block: dict[str, Any]) -> str:
    return str(block.get("block_content") or "")


def _effective_label_for_block(block: dict[str, Any]) -> str:
    return _label_from_block(block)


def _intersect_xyxy(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _text_route_bboxes_from_lines(
    block_idx: int,
    lines: tuple[RoutingLine, ...],
) -> list[_TextRoute]:
    """Expand typed page routes without reading raw Paddle route fields."""
    return [
        _TextRoute(
            block_idx=block_idx,
            line_idx=line_idx,
            segment_idx=segment_idx,
            bbox=segment.bbox,
            carved=len(line.segments) > 1,
            kind=segment.kind,
            ppocr_latin_fallback_text=(segment.text if segment.kind == ROUTE_SEGMENT_TEXT_LATIN else ""),
            ppocr_latin_tokens=segment.ppocr_latin_tokens,
            content_bbox=segment.content_bbox,
        )
        for line_idx, line in enumerate(lines)
        for segment_idx, segment in enumerate(line.segments)
        if is_text_route_segment_kind(segment.kind)
    ]


def _engcut_masked_line_routes_from_lines(
    block_idx: int,
    lines: tuple[RoutingLine, ...],
) -> list[_EngCutMaskedLineRoute]:
    """Group typed EngCut routes by their physical PP-OCR line.

    This is deliberately derived from ``RoutingLine`` rather than from raw
    Paddle payloads.  The selected segments remain the only places whose
    pixels may appear in the EngCut input canvas.
    """
    grouped: list[_EngCutMaskedLineRoute] = []
    for line_idx, line in enumerate(lines):
        segments = tuple(
            _TextRoute(
                block_idx=block_idx,
                line_idx=line_idx,
                segment_idx=segment_idx,
                bbox=segment.bbox,
                carved=len(line.segments) > 1,
                kind=segment.kind,
                ppocr_latin_fallback_text=(segment.text if segment.kind == ROUTE_SEGMENT_TEXT_LATIN else ""),
                ppocr_latin_tokens=segment.ppocr_latin_tokens,
                content_bbox=segment.content_bbox,
            )
            for segment_idx, segment in enumerate(line.segments)
            if segment.kind == ROUTE_SEGMENT_TEXT_LATIN
        )
        if segments:
            grouped.append(
                _EngCutMaskedLineRoute(
                    block_idx=block_idx,
                    line_idx=line_idx,
                    bbox=line.bbox,
                    segments=segments,
                    text_axis=line.text_axis,
                    orientation_angle=line.orientation_angle,
                    ppocr_symbol_observations=line.ppocr_symbol_observations,
                )
            )
    return grouped


def _linecut_masked_line_routes_from_lines(
    block_idx: int,
    lines: tuple[RoutingLine, ...],
) -> list[_LineCutMaskedLineRoute]:
    routes: list[_LineCutMaskedLineRoute] = []
    for line_idx, line in enumerate(lines):
        typed = tuple(
            _TextRoute(
                block_idx=block_idx,
                line_idx=line_idx,
                segment_idx=segment_idx,
                bbox=segment.bbox,
                carved=len(line.segments) > 1,
                kind=segment.kind,
                ppocr_latin_fallback_text=(segment.text if segment.kind == ROUTE_SEGMENT_TEXT_LATIN else ""),
                ppocr_latin_tokens=segment.ppocr_latin_tokens,
                content_bbox=segment.content_bbox,
            )
            for segment_idx, segment in enumerate(line.segments)
        )
        linecut = tuple(
            segment for segment in typed
            if is_text_route_segment_kind(segment.kind)
            and segment.kind != ROUTE_SEGMENT_TEXT_LATIN
        )
        if linecut:
            routes.append(_LineCutMaskedLineRoute(
                block_idx=block_idx,
                line_idx=line_idx,
                bbox=line.bbox,
                linecut_segments=linecut,
                excluded_segments=tuple(segment for segment in typed if segment not in linecut),
                text_axis=line.text_axis,
                orientation_angle=line.orientation_angle,
            ))
    return routes


def _materialize_linecut_masked_page(
    image_bgr: np.ndarray,
    routes: list[_LineCutMaskedLineRoute],
) -> np.ndarray:
    """Expose exact LineCut-owned rectangles on a page-sized white canvas."""
    canvas = np.full_like(image_bgr, 255)
    height, width = image_bgr.shape[:2]
    for route in routes:
        for segment in route.linecut_segments:
            x1, y1, x2, y2 = _clamp_xyxy(segment.bbox, width, height)
            canvas[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]
    # Exclusions have explicit precedence.  Most importantly, an inline
    # formula whites only its true two-dimensional intersection; its x-range
    # is never projected through the full height of a neighboring row.
    for route in routes:
        for segment in route.excluded_segments:
            x1, y1, x2, y2 = _clamp_xyxy(segment.bbox, width, height)
            canvas[y1:y2, x1:x2] = 255
    return canvas


def _materialize_linecut_masked_line_crop(
    image_bgr: np.ndarray,
    route: _LineCutMaskedLineRoute,
) -> tuple[np.ndarray, int, int]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(route.bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid masked LineCut line bbox: {route.bbox}")
    canvas = np.full_like(image_bgr[y1:y2, x1:x2], 255)
    for segment in route.linecut_segments:
        sx1, sy1, sx2, sy2 = _intersect_xyxy(segment.bbox, route.bbox) or (0, 0, 0, 0)
        if sx2 <= sx1 or sy2 <= sy1:
            raise RuntimeError(
                "masked LineCut segment does not intersect its physical line: "
                f"line={route.bbox} segment={segment.bbox}"
            )
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = image_bgr[sy1:sy2, sx1:sx2]
    for segment in route.excluded_segments:
        overlap = _intersect_xyxy(segment.bbox, route.bbox)
        if overlap is None:
            continue
        sx1, sy1, sx2, sy2 = overlap
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = 255
    return canvas, x1, y1


def _merge_physical_routing_line(
    lines: list[_NativeLineResult],
    *,
    route_bbox: tuple[int, int, int, int],
) -> list[_NativeLineResult]:
    """Materialize one PP physical row without promoting native group geometry."""
    if not lines:
        return []
    if len(lines) == 1:
        return [replace(
            lines[0],
            bbox=route_bbox,
            bbox_source="ppocrv6_physical_routing_line",
        )]
    ordered = sorted(lines, key=lambda item: (item.bbox[0], item.bbox[1]))
    chars = [char for line in ordered for char in line.chars]
    confidence_values = [line.confidence for line in ordered if line.confidence > 0]
    flags = sorted({flag for line in ordered for flag in line.review_flags})
    return [_NativeLineResult(
        text="".join(line.text for line in ordered if line.text),
        bbox=route_bbox,
        confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        chars=chars,
        source="hanwang+ppocrv6_routing_line",
        bbox_source="ppocrv6_physical_routing_line",
        review_flags=flags,
    )]


@dataclass(frozen=True)
class _PpSymbolApplicationStats:
    candidates_bound: int = 0
    atoms_inserted: int = 0
    observations_unbound: int = 0


def _apply_ppocr_symbol_observations(
    lines: list[_NativeLineResult],
    observations: tuple[PpOcrSymbolObservation, ...],
) -> tuple[list[_NativeLineResult], _PpSymbolApplicationStats]:
    """Bind PP symbols to native atoms or insert an observed missing glyph."""

    if not lines or not observations:
        return lines, _PpSymbolApplicationStats(
            observations_unbound=len(observations),
        )
    candidates_bound = 0
    atoms_inserted = 0
    observations_unbound = 0
    for observation in sorted(observations, key=lambda item: (item.bbox[0], item.bbox[1])):
        center_x, center_y = _bbox_center(observation.bbox)
        owners = [
            index
            for index, line in enumerate(lines)
            if line.bbox[0] <= center_x < line.bbox[2]
            and line.bbox[1] <= center_y < line.bbox[3]
        ]
        if len(owners) != 1:
            observations_unbound += 1
            continue
        line = lines[owners[0]]
        claimed = [
            index
            for index, char in enumerate(line.chars)
            if char.bbox is not None
            and _point_in_xyxy(_bbox_center(char.bbox), observation.proposal_bbox)
            and _intersect_xyxy(char.bbox, observation.bbox) is not None
        ]
        intersecting = [
            index
            for index, char in enumerate(line.chars)
            if char.bbox is not None
            and _intersect_xyxy(char.bbox, observation.bbox) is not None
        ]
        if len(claimed) == 1:
            candidates_bound += 1
            char = line.chars[claimed[0]]
            if char.text != observation.text:
                candidate = CharOcrCandidateObservation(
                    text=observation.text,
                    confidence=0.0,
                    source=PPOCR_SYMBOL_FOREGROUND_SOURCE,
                    bbox=observation.bbox,
                )
                if candidate not in char.external_candidates:
                    char.external_candidates.append(candidate)
            continue
        if claimed or intersecting:
            observations_unbound += 1
            continue
        insertion = next((
            index
            for index, char in enumerate(line.chars)
            if char.bbox is not None
            and _bbox_center(char.bbox)[0] > center_x
        ), len(line.chars))
        line.chars.insert(insertion, _NativeAtomResult(
            text=observation.text,
            confidence=0.0,
            bbox=observation.bbox,
            candidates=[observation.text],
            source=PPOCR_SYMBOL_FOREGROUND_SOURCE,
            bbox_granularity="char",
            token_text=observation.text,
        ))
        line.text = _line_chars_text(line.chars).strip()
        if "ppocr_symbol_missing_native_atom" not in line.review_flags:
            line.review_flags.append("ppocr_symbol_missing_native_atom")
        atoms_inserted += 1
    return lines, _PpSymbolApplicationStats(
        candidates_bound=candidates_bound,
        atoms_inserted=atoms_inserted,
        observations_unbound=observations_unbound,
    )


def _attach_vl_marker_candidates(
    lines: list[_NativeLineResult],
    observations: tuple[VlSemanticMarkerObservation, ...],
) -> tuple[list[_NativeLineResult], int, int]:
    """Retain a uniquely bound VL marker as a non-authoritative candidate."""
    return _attach_owned_text_candidates(
        lines,
        observations,
        source=VL_SEMANTIC_MARKER_SOURCE,
    )


def _attach_owned_text_candidates(
    lines: list[_NativeLineResult],
    observations: tuple[PpOcrSymbolObservation | VlSemanticMarkerObservation, ...],
    *,
    source: str,
) -> tuple[list[_NativeLineResult], int, int]:
    if not lines or not observations:
        return lines, 0, len(observations)
    bound = 0
    unbound = 0
    for observation in sorted(observations, key=lambda item: (item.bbox[0], item.bbox[1])):
        center_x, center_y = _bbox_center(observation.bbox)
        owners = [
            index
            for index, line in enumerate(lines)
            if line.bbox[0] <= center_x < line.bbox[2]
            and line.bbox[1] <= center_y < line.bbox[3]
        ]
        if len(owners) != 1:
            unbound += 1
            continue
        line = lines[owners[0]]
        claimed = [
            index
            for index, char in enumerate(line.chars)
            if char.bbox is not None
            and _point_in_xyxy(_bbox_center(char.bbox), observation.proposal_bbox)
            and _intersect_xyxy(char.bbox, observation.bbox) is not None
        ]
        if len(claimed) != 1:
            unbound += 1
            continue
        bound += 1
        char = line.chars[claimed[0]]
        if char.text == observation.text:
            continue
        candidate = CharOcrCandidateObservation(
            text=observation.text,
            confidence=0.0,
            source=source,
            bbox=observation.bbox,
        )
        if candidate not in char.external_candidates:
            char.external_candidates.append(candidate)
    return lines, bound, unbound


def _apply_route_text_observations(
    lines: list[_NativeLineResult],
    route: RoutingLine,
    stats: RunStats,
) -> list[_NativeLineResult]:
    lines, pp_stats = _apply_ppocr_symbol_observations(
        lines,
        route.ppocr_symbol_observations,
    )
    lines, vl_bound, vl_unbound = _attach_vl_marker_candidates(
        lines,
        route.vl_marker_observations,
    )
    stats.ppocr_symbol_candidates_bound += pp_stats.candidates_bound
    stats.ppocr_symbol_atoms_inserted += pp_stats.atoms_inserted
    stats.ppocr_symbol_observations_unbound += pp_stats.observations_unbound
    stats.vl_marker_observations_bound += vl_bound
    stats.vl_marker_observations_unbound += vl_unbound
    return lines


def _distribute_linecut_results(
    lines: list[_NativeLineResult],
    route: _LineCutMaskedLineRoute,
) -> dict[tuple[int, int, int], list[_NativeLineResult]]:
    """Return native observations to their explicit route regions.

    Character centers must have exactly one LineCut owner.  Ambiguous or
    unowned native geometry is invalid page state rather than a reason to
    guess from text or proximity.
    """
    distributed = {segment.key: [] for segment in route.linecut_segments}
    for line in lines:
        if not line.text and not line.chars:
            continue
        if line.chars:
            buckets: dict[tuple[int, int, int], list[_NativeAtomResult]] = {
                segment.key: [] for segment in route.linecut_segments
            }
            for char in line.chars:
                if char.bbox is None:
                    raise RuntimeError(
                        "LineCut returned a character without geometry for a typed route: "
                        f"line={route.bbox} text={char.text!r}"
                    )
                center_x, center_y = _bbox_center(char.bbox)
                owners = [
                    segment for segment in route.linecut_segments
                    if segment.bbox[0] <= center_x < segment.bbox[2]
                    and segment.bbox[1] <= center_y < segment.bbox[3]
                ]
                if len(owners) != 1:
                    raise RuntimeError(
                        "LineCut character has no unique typed route owner: "
                        f"char={char.text!r} bbox={char.bbox} owners={len(owners)}"
                    )
                buckets[owners[0].key].append(char)
            for segment in route.linecut_segments:
                chars = buckets[segment.key]
                if not chars:
                    continue
                boxes = [char.bbox for char in chars if char.bbox is not None]
                observed_bbox = _intersect_xyxy(line.bbox, segment.bbox)
                distributed[segment.key].append(_NativeLineResult(
                    text="".join(char.text for char in chars),
                    bbox=observed_bbox or union_xyxy(boxes),
                    confidence=line.confidence,
                    chars=chars,
                    source=line.source,
                    bbox_source=line.bbox_source,
                    review_flags=list(line.review_flags),
                ))
            continue

        owners = [
            segment for segment in route.linecut_segments
            if _intersect_xyxy(line.bbox, segment.bbox) is not None
        ]
        if len(owners) != 1:
            raise RuntimeError(
                "LineCut line without character geometry has no unique typed route owner: "
                f"bbox={line.bbox} owners={len(owners)}"
            )
        distributed[owners[0].key].append(line)
    return distributed


def _recognize_oriented_linecut_route(
    image_bgr: np.ndarray,
    route: _LineCutMaskedLineRoute,
    stats: RunStats,
    *,
    timeout: float,
    include_chars: bool,
) -> dict[tuple[int, int, int], list[_NativeLineResult]]:
    """Recognize one non-horizontal physical line through a reversible crop."""
    crop, offset_x, offset_y = _materialize_linecut_masked_line_crop(image_bgr, route)
    oriented = _orient_crop_for_native(
        crop,
        text_axis=route.text_axis,
        orientation_angle=route.orientation_angle,
    )
    prepared = _prepare_full_line_recog_crop(oriented.image)
    stats.recog_probe_calls += 1
    raw = native_bridge.run_linecut_recog(
        prepared.image,
        recblock_xyxy=None,
        with_charrcg=True,
        timeout=timeout,
    )
    native_height, native_width = prepared.image.shape[:2]
    local_lines = _line_results_from_recog(
        raw,
        fallback_bbox=(0, 0, native_width, native_height),
        include_chars=include_chars,
    )
    local_lines = _rescale_line_results(
        local_lines,
        scale_x=prepared.scale_x,
        scale_y=prepared.scale_y,
        target_width=oriented.image.shape[1],
        target_height=oriented.image.shape[0],
    )
    if include_chars:
        _reconcile_native_char_geometry(oriented.image, local_lines, stats)
    source_lines = _line_results_from_oriented_crop(local_lines, oriented)
    page_lines = _offset_line_results(source_lines, dx=offset_x, dy=offset_y)
    return _distribute_linecut_results(
        _filter_line_results_to_route_bbox(page_lines, route.bbox),
        route,
    )


def _cluster_lines_by_shape(lines: list[_NativeLineResult]) -> list[list[_NativeLineResult]]:
    buckets: list[list[_NativeLineResult]] = []
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        for bucket in buckets:
            if vertical_overlap_ratio(bucket[0].bbox, line.bbox) >= 0.5:
                bucket.append(line)
                break
        else:
            buckets.append([line])
    return buckets


def _box_width(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0])


def _box_height(box: tuple[int, int, int, int]) -> int:
    return max(0, box[3] - box[1])


def _assemble_layout_route_line(
    *,
    block_idx: int,
    line_idx: int,
    route: RoutingLine,
    grouped_lines: dict[tuple[int, int, int], list[_NativeLineResult]],
    stats: RunStats,
    image_bgr: np.ndarray | None = None,
) -> list[_NativeLineResult]:
    segments = route.segments
    if not segments:
        return []

    if all(segment.kind == "skip" for segment in segments):
        segment = segments[0]
        text = segment.text
        if not text:
            return []
        flags = [ROUTE_TABLE_FLAG] if is_table_label(segment.label) else []
        return [
            _NativeLineResult(
                text=text,
                bbox=segment.bbox,
                confidence=0.0,
                chars=[],
                source=f"ppvl_route:{segment.label or 'skip'}",
                bbox_source="ppvl_route_skip_segment",
                review_flags=flags,
            )
        ]

    slice_lines_by_segment: dict[int, list[_NativeLineResult]] = {}
    all_text_lines: list[_NativeLineResult] = []
    has_formula = any(segment.kind == "formula" for segment in segments)
    for segment_idx, segment in enumerate(segments):
        if not is_text_route_segment_kind(segment.kind):
            continue
        key = (block_idx, line_idx, segment_idx)
        current_lines = [
            line
            for line in grouped_lines.get(key, [])
            if line.text or line.chars
        ]
        current_lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        slice_lines_by_segment[segment_idx] = current_lines
        all_text_lines.extend(current_lines)

    if not has_formula:
        merged_text_lines: list[_NativeLineResult] = []
        for segment_idx in range(len(segments)):
            merged_text_lines.extend(slice_lines_by_segment.get(segment_idx, []))
        lines = _apply_route_text_observations(
            _merge_physical_routing_line(
                merged_text_lines,
                route_bbox=route.bbox,
            ),
            route,
            stats,
        )
        return _postprocess_cjk_line_geometry(image_bgr, route, lines, stats)

    clusters = _cluster_lines_by_shape(all_text_lines)
    if not clusters:
        return []

    assembled: list[_NativeLineResult] = []
    for cluster in clusters:
        cluster_bbox = union_xyxy([line.bbox for line in cluster])
        components: list[
            tuple[
                tuple[int, int, int, int],
                str,
                list[_NativeAtomResult],
                float,
                set[str],
            ]
        ] = []
        for segment_idx, segment in enumerate(segments):
            segment_bbox = segment.bbox
            kind = segment.kind
            if is_text_route_segment_kind(kind):
                segment_lines = [
                    line
                    for line in slice_lines_by_segment.get(segment_idx, [])
                    if vertical_overlap_ratio(line.bbox, cluster_bbox) >= 0.5
                ]
                if not segment_lines and len(clusters) == 1:
                    segment_lines = slice_lines_by_segment.get(segment_idx, [])
                for line in segment_lines:
                    components.append((
                        line.bbox,
                        line.text,
                        list(line.chars),
                        line.confidence,
                        set(line.review_flags),
                    ))
            elif kind == "formula":
                formula_text = segment.text
                if not formula_text:
                    continue
                content_bbox = segment.content_bbox or segment_bbox
                formula_char = _NativeAtomResult(
                    text=formula_text,
                    confidence=0.0,
                    bbox=content_bbox,
                    candidates=[formula_text],
                    source="paddle_inline_formula",
                    bbox_granularity="word",
                    token_text=formula_text,
                )
                components.append((
                    content_bbox,
                    formula_text,
                    [formula_char],
                    0.0,
                    {ROUTE_INLINE_FORMULA_FLAG},
                ))

        components.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
        text_parts: list[str] = []
        chars: list[_NativeAtomResult] = []
        confidence_values: list[float] = []
        flags: set[str] = set()
        for _bbox, text, component_chars, confidence, component_flags in components:
            text_parts.append(text)
            chars.extend(component_chars)
            flags.update(component_flags)
            if confidence > 0:
                confidence_values.append(confidence)
        merged_text = "".join(text_parts)
        if not merged_text:
            continue
        assembled.append(
            _NativeLineResult(
                text=merged_text,
                bbox=route.bbox,
                confidence=(
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else 0.0
                ),
                chars=chars,
                source="layout_route+hanwang",
                bbox_source="layout_route_assembled",
                review_flags=sorted(flags),
            )
        )
    lines = _apply_route_text_observations(assembled, route, stats)
    return _postprocess_cjk_line_geometry(image_bgr, route, lines, stats)


def _assemble_routing_lines(
    *,
    block_idx: int,
    line_routes: tuple[RoutingLine, ...],
    grouped_lines: dict[tuple[int, int, int], list[_NativeLineResult]],
    stats: RunStats,
    image_bgr: np.ndarray | None = None,
) -> list[_NativeLineResult]:
    if not line_routes:
        return []
    assembled: list[_NativeLineResult] = []
    for line_idx, route in enumerate(line_routes):
        assembled.extend(
            _assemble_layout_route_line(
                block_idx=block_idx,
                line_idx=line_idx,
                route=route,
                grouped_lines=grouped_lines,
                stats=stats,
                image_bgr=image_bgr,
            )
        )
    assembled.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return assembled


def _postprocess_cjk_line_geometry(
    image_bgr: np.ndarray | None,
    route: RoutingLine,
    lines: list[_NativeLineResult],
    stats: RunStats,
) -> list[_NativeLineResult]:
    if (
        image_bgr is None
        or route.text_axis != TEXT_AXIS_HORIZONTAL
        or route.orientation_angle == 180
    ):
        return lines
    processed: list[_NativeLineResult] = []
    for line in lines:
        atoms = [
            LineAtomGeometry(
                index=index,
                text=atom.text,
                bbox=atom.bbox,
                source=atom.source,
                granularity=atom.bbox_granularity,
            )
            for index, atom in enumerate(line.chars)
        ]
        proposals = conservative_cjk_bbox_cleanup(
            image_bgr,
            line.bbox,
            atoms,
            linecut_source=LINECUT_NATIVE_ATOM_SOURCE,
        )
        if not proposals:
            processed.append(line)
            continue
        chars: list[_NativeAtomResult] = []
        for index, atom in enumerate(line.chars):
            proposal = proposals.get(index)
            if proposal is None or atom.bbox is None:
                chars.append(atom)
                continue
            chars.append(replace(
                atom,
                bbox=proposal,
                source=LINECUT_CJK_CLEANUP_SOURCE,
                external_candidates=[
                    *atom.external_candidates,
                    CharOcrCandidateObservation(
                        text=atom.text,
                        confidence=atom.confidence,
                        source=LINECUT_NATIVE_ATOM_SOURCE,
                        bbox=atom.bbox,
                    ),
                ],
            ))
        stats.linecut_cjk_bbox_cleanups += len(proposals)
        processed.append(replace(
            line,
            chars=chars,
            review_flags=sorted({*line.review_flags, LINECUT_CJK_CLEANUP_FLAG}),
        ))
    return processed


def decode_gbk(code: int) -> str:
    """Decode Hanwang little-endian GBK code and strip NUL/control noise."""
    if not code:
        return ""
    try:
        text = code.to_bytes(2, "little").decode("gbk", errors="ignore")
    except Exception:
        return ""
    return "".join(ch for ch in text if ch >= " " or ch in "\n\t")


def _score_to_confidence(score: object) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    conf = 1.0 - value / 100.0
    return max(0.0, min(1.0, conf))


def _code_to_int(code: object) -> int:
    if isinstance(code, str):
        return int(code, 0) if code.startswith(("0x", "0X")) else int(code, 16)
    return int(code)


def _bbox_tuple(raw: object, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not isinstance(raw, dict):
        return fallback
    try:
        left = int(raw.get("left", raw.get("x", fallback[0])))
        top = int(raw.get("top", raw.get("y", fallback[1])))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw.get("width", max(0, fallback[2] - fallback[0])))
            bottom = top + int(raw.get("height", max(0, fallback[3] - fallback[1])))
    except (TypeError, ValueError):
        return fallback
    if right <= left or bottom <= top:
        return fallback
    return left, top, right, bottom


def _optional_bbox_tuple(raw: object) -> tuple[int, int, int, int] | None:
    """Parse native character geometry without inventing a line-sized box."""
    if not isinstance(raw, dict):
        return None
    try:
        left = int(raw.get("left", raw.get("x")))
        top = int(raw.get("top", raw.get("y")))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw["width"])
            bottom = top + int(raw["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _clamp_xyxy(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left = max(0, min(int(left), width))
    top = max(0, min(int(top), height))
    right = max(left, min(int(right), width))
    bottom = max(top, min(int(bottom), height))
    return left, top, right, bottom


def _expand_xyxy(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    pad_x: int = 0,
    pad_y: int = 0,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return _clamp_xyxy(
        (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
        width,
        height,
    )


def _layout_block_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return block_bbox_xyxy(raw, width, height)


def _block_geometry_from_page_routes(
    raw: dict,
    width: int,
    height: int,
    routes: tuple[RoutingLine, ...],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]:
    """Get native audit geometry from the explicit typed plan, not raw fields."""
    layout_bbox = _layout_block_bbox(raw, width, height)
    if routes:
        return (
            union_xyxy([route.bbox for route in routes]),
            layout_bbox,
            "page_routing_plan_union",
        )
    return layout_bbox, layout_bbox, "layout_block_bbox"


def _block_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return _layout_block_bbox(raw, width, height)


def _bbox_lists(values: list[tuple[int, int, int, int]]) -> list[list[int]]:
    return [list(bbox) for bbox in values]


def _hanwang_bbox_audit(
    raw: dict[str, Any],
    width: int,
    height: int,
    *,
    block_bbox: tuple[int, int, int, int] | None = None,
    block_bbox_source: str | None = None,
    routing_line_bboxes: list[tuple[int, int, int, int]],
    route_text_slice_bboxes: list[tuple[int, int, int, int]] | None = None,
    recog_group_bboxes: list[tuple[int, int, int, int]] | None = None,
    segimg_group_audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    layout_bbox = _layout_block_bbox(raw, width, height)
    line_route_bboxes = list(routing_line_bboxes)
    effective_bbox = block_bbox or (union_xyxy(line_route_bboxes) if line_route_bboxes else layout_bbox)
    text_slice_bboxes = route_text_slice_bboxes or []
    group_bboxes = recog_group_bboxes or []
    group_audits = segimg_group_audits or []
    label = _effective_label_for_block(raw)
    unknown_label = _is_unknown_hanwang_label(label)
    return {
        "schema": "hanwang_bbox_audit.v1",
        "paddle_label": label,
        "paddle_label_unknown": unknown_label,
        "paddle_label_unknown_action": "default_text_ocr" if unknown_label else "",
        "layout_block_bbox": list(layout_bbox),
        "effective_block_bbox": list(effective_bbox),
        "effective_block_bbox_source": block_bbox_source or "layout_block_bbox",
        "routing_line_bboxes": _bbox_lists(line_route_bboxes),
        "route_text_slice_bboxes": _bbox_lists(text_slice_bboxes),
        "hanwang_recog_group_bboxes": _bbox_lists(group_bboxes),
        "hanwang_segimg_groups": [dict(item) for item in group_audits],
        "hanwang_segimg_group_clipped_count": sum(1 for item in group_audits if item.get("clipped")),
        "hanwang_segimg_group_dropped_count": sum(1 for item in group_audits if item.get("dropped")),
        "hanwang_recog_group_failed_count": sum(1 for item in group_audits if item.get("recog_failed")),
        "route_text_slice_count": len(text_slice_bboxes),
        "hanwang_recog_group_count": len(group_bboxes),
    }


def _raw_block_with_bbox_audit(
    raw: dict[str, Any],
    width: int,
    height: int,
    *,
    block_bbox: tuple[int, int, int, int] | None = None,
    block_bbox_source: str | None = None,
    routing_line_bboxes: list[tuple[int, int, int, int]],
    route_text_slice_bboxes: list[tuple[int, int, int, int]] | None = None,
    recog_group_bboxes: list[tuple[int, int, int, int]] | None = None,
    segimg_group_audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = dict(raw)
    value[ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY] = _hanwang_bbox_audit(
        raw,
        width,
        height,
        block_bbox=block_bbox,
        block_bbox_source=block_bbox_source,
        routing_line_bboxes=routing_line_bboxes,
        route_text_slice_bboxes=route_text_slice_bboxes,
        recog_group_bboxes=recog_group_bboxes,
        segimg_group_audits=segimg_group_audits,
    )
    return value


def _char_result(raw: dict) -> _NativeAtomResult:
    codes = raw.get("codes") or []
    scores = raw.get("scores") or []
    candidates: list[str] = []
    for code in codes:
        try:
            text = decode_gbk(_code_to_int(code))
        except (TypeError, ValueError):
            text = ""
        if text:
            candidates.append(text)
    candidate_confidences = [
        _score_to_confidence(score)
        for score in scores[:len(candidates)]
    ]
    if not candidates:
        raw_candidates = raw.get("candidates")
        if isinstance(raw_candidates, (list, tuple)):
            for candidate in raw_candidates:
                if isinstance(candidate, dict):
                    candidate_text = str(candidate.get("text") or "")
                    candidate_confidence = candidate.get("confidence")
                    try:
                        candidate_confidence = float(candidate_confidence)
                    except (TypeError, ValueError):
                        candidate_confidence = 0.0
                else:
                    candidate_text = str(candidate or "")
                    candidate_confidence = 0.0
                if candidate_text:
                    candidates.append(candidate_text)
                    candidate_confidences.append(candidate_confidence)
        if not candidates:
            text_value = str(raw.get("text") or "")
            if text_value:
                candidates.append(text_value)
                candidate_confidences.append(0.0)
    text = candidates[0] if candidates else ""
    confidence = candidate_confidences[0] if candidate_confidences else 0.0
    return _NativeAtomResult(
        text=text,
        confidence=confidence,
        bbox=_optional_bbox_tuple(raw.get("bbox")),
        candidates=candidates,
        candidate_confidences=candidate_confidences,
    )


def _fallback_line(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    source: str,
    synthesize_chars: bool = True,
) -> _NativeLineResult:
    return _NativeLineResult(
        text=text,
        bbox=bbox,
        confidence=0.0,
        source=source,
        bbox_source=f"{source}:bbox",
        chars=(
            [
                _NativeAtomResult(
                    text=ch,
                    confidence=0.0,
                    bbox=None,
                    candidates=[ch],
                    candidate_confidences=[0.0],
                    source=source,
                )
                for ch in text
            ]
            if synthesize_chars
            else []
        ),
    )


def _line_results_from_recog(
    raw: dict,
    *,
    fallback_bbox: tuple[int, int, int, int],
    include_chars: bool,
    fallback_empty: bool = True,
) -> list[_NativeLineResult]:
    lines: list[_NativeLineResult] = []
    for area in raw.get("lines", []) or []:
        for group in area.get("groups", []) or []:
            line_bbox = _bbox_tuple(group.get("bbox"), fallback_bbox)
            raw_chars = group.get("chars") or []
            chars = [
                char
                for char in (_char_result(char) for char in raw_chars)
                if char.text
            ]
            text = "".join(char.text for char in chars).strip()
            if not text and not chars:
                continue
            confidence = (
                sum(char.confidence for char in chars) / len(chars)
                if chars else 0.0
            )
            review_flags = (
                ["hanwang_missing_char_geometry"]
                if any(char.bbox is None for char in chars)
                else []
            )
            lines.append(
                _NativeLineResult(
                    text=text,
                    bbox=line_bbox,
                    confidence=confidence,
                    chars=chars if include_chars else [],
                    bbox_source="hanwang_recog_group",
                    review_flags=review_flags,
                )
            )
    if not lines and fallback_empty:
        lines.append(_fallback_line("", fallback_bbox, source="hanwang_empty"))
    return lines


def _offset_line_results(
    lines: list[_NativeLineResult],
    *,
    dx: int,
    dy: int,
) -> list[_NativeLineResult]:
    shifted: list[_NativeLineResult] = []
    for line in lines:
        lx1, ly1, lx2, ly2 = line.bbox
        chars: list[_NativeAtomResult] = []
        for char in line.chars:
            bbox = None
            if char.bbox is not None:
                cx1, cy1, cx2, cy2 = char.bbox
                bbox = (cx1 + dx, cy1 + dy, cx2 + dx, cy2 + dy)
            chars.append(
                _NativeAtomResult(
                    text=char.text,
                    confidence=char.confidence,
                    bbox=bbox,
                    candidates=list(char.candidates),
                    candidate_confidences=list(char.candidate_confidences),
                    external_candidates=list(char.external_candidates),
                    source=char.source,
                    bbox_granularity=char.bbox_granularity,
                    token_text=char.token_text,
                )
            )
        shifted.append(
            _NativeLineResult(
                text=line.text,
                bbox=(lx1 + dx, ly1 + dy, lx2 + dx, ly2 + dy),
                confidence=line.confidence,
                chars=chars,
                source=line.source,
                bbox_source=line.bbox_source,
                review_flags=list(line.review_flags),
            )
        )
    return shifted


def _rescale_line_results(
    lines: list[_NativeLineResult],
    *,
    scale_x: float,
    scale_y: float,
    target_width: int,
    target_height: int,
) -> list[_NativeLineResult]:
    if scale_x == 1.0 and scale_y == 1.0:
        return lines

    def rescale_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        mapped = (
            int(round(x1 / scale_x)),
            int(round(y1 / scale_y)),
            int(round(x2 / scale_x)),
            int(round(y2 / scale_y)),
        )
        return _clamp_xyxy(mapped, target_width, target_height)

    scaled: list[_NativeLineResult] = []
    for line in lines:
        chars = [
            _NativeAtomResult(
                text=char.text,
                confidence=char.confidence,
                bbox=rescale_bbox(char.bbox) if char.bbox is not None else None,
                candidates=list(char.candidates),
                candidate_confidences=list(char.candidate_confidences),
                external_candidates=list(char.external_candidates),
                source=char.source,
                bbox_granularity=char.bbox_granularity,
                token_text=char.token_text,
            )
            for char in line.chars
        ]
        scaled.append(_NativeLineResult(
            text=line.text,
            bbox=rescale_bbox(line.bbox),
            confidence=line.confidence,
            chars=chars,
            source=line.source,
            bbox_source=line.bbox_source,
            review_flags=list(line.review_flags),
        ))
    return scaled


def _char_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _point_in_xyxy(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _filter_line_results_to_route_bbox(
    lines: list[_NativeLineResult],
    route_bbox: tuple[int, int, int, int],
) -> list[_NativeLineResult]:
    """Drop OCR context characters that fall outside the current text route.

    Recognition crops may intentionally include a few pixels of neighbouring
    formula/line context so native OCR sees complete glyphs.  The text fact
    still belongs to the route slice, so characters centered outside that slice
    must not enter the assembled line.
    """
    filtered: list[_NativeLineResult] = []
    for line in lines:
        if not line.chars:
            if _intersection_area(line.bbox, route_bbox) > 0:
                filtered.append(line)
            continue
        kept_chars: list[_NativeAtomResult] = []
        dropped = False
        for char in line.chars:
            if char.bbox is None or _point_in_xyxy(_char_center(char.bbox), route_bbox):
                kept_chars.append(char)
            else:
                dropped = True
        if not kept_chars:
            continue
        if not dropped:
            filtered.append(line)
            continue
        boxes = [char.bbox for char in kept_chars if char.bbox is not None]
        text = "".join(char.text for char in kept_chars)
        confidence = (
            sum(char.confidence for char in kept_chars) / len(kept_chars)
            if kept_chars
            else line.confidence
        )
        filtered.append(
            _NativeLineResult(
                text=text,
                bbox=union_xyxy(boxes) if boxes else line.bbox,
                confidence=confidence,
                chars=kept_chars,
                source=f"{line.source}:route_context_filtered",
                bbox_source=line.bbox_source,
                review_flags=list(line.review_flags),
            )
        )
    return filtered


def _line_chars_text(chars: list[_NativeAtomResult]) -> str:
    return "".join(char.text for char in chars)


def _normalize_digitlike_numeric_context_lines(lines: list[_NativeLineResult]) -> None:
    for line in lines:
        if not line.chars:
            continue
        changed = False
        for index, char in enumerate(line.chars):
            replacement = _digitlike_numeric_context_replacement(line.chars, index)
            if replacement is None:
                continue
            original = char.text
            char.text = replacement
            char.token_text = replacement
            char.candidates = [replacement, *[candidate for candidate in char.candidates if candidate != replacement]]
            if f"digitlike:{original}" not in char.source:
                char.source = f"{char.source}:digitlike:{original}"
            changed = True
        if changed:
            line.text = _line_chars_text(line.chars).strip()
            if DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG not in line.review_flags:
                line.review_flags.append(DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG)


def _digitlike_numeric_context_replacement(chars: list[_NativeAtomResult], index: int) -> str | None:
    char = chars[index]
    text = str(char.text or "")
    if text in _DIGITLIKE_ZERO_CHARS:
        replacement = "0"
    elif text in _DIGITLIKE_ONE_CHARS:
        replacement = "1"
    else:
        return None
    if float(char.confidence or 0.0) > 0.35:
        return None
    if _adjacent_visible_text(chars, index, step=-1) != "取":
        return None
    if _adjacent_visible_text(chars, index, step=1) not in _DIGITLIKE_NUMERIC_CONTEXT_FOLLOWERS:
        return None
    return replacement


def _adjacent_visible_text(chars: list[_NativeAtomResult], index: int, *, step: int) -> str:
    pos = index + step
    while 0 <= pos < len(chars):
        text = str(chars[pos].text or "")
        if text.strip():
            return text
        pos += step
    return ""


def _reconcile_native_char_geometry(
    crop_bgr: np.ndarray,
    lines: list[_NativeLineResult],
    stats: RunStats,
) -> None:
    if crop_bgr.size == 0:
        return
    for line in lines:
        if not line.chars:
            continue
        proposals = tuple(
            NativeGeometryProposal(index, char.bbox, char.confidence)
            for index, char in enumerate(line.chars)
            if char.bbox is not None
        )
        if len(proposals) < 2:
            continue
        result = reconcile_char_geometry(
            crop_bgr,
            proposals,
            region_bbox=line.bbox,
        )
        atoms_by_start = {min(atom.proposal_indices): atom for atom in result.atoms}
        rebuilt: list[_NativeAtomResult] = []
        consumed: set[int] = set()
        for index, char in enumerate(line.chars):
            if index in consumed:
                continue
            atom = atoms_by_start.get(index)
            if atom is None:
                rebuilt.append(char)
                continue
            members = [line.chars[item] for item in atom.proposal_indices]
            consumed.update(atom.proposal_indices)
            if len(members) == 1:
                rebuilt.append(char)
                continue
            text = _line_chars_text(members)
            rebuilt.append(_NativeAtomResult(
                text=text,
                confidence=sum(member.confidence for member in members) / len(members),
                bbox=atom.bbox,
                candidates=[text] if text else [],
                source="hanwang:geometry_reconciled",
                bbox_granularity="word",
                token_text=text,
            ))
            stats.geometry_conflict_groups += 1
            stats.geometry_token_atoms += 1
        if len(rebuilt) == len(line.chars):
            continue
        line.chars[:] = rebuilt
        line.text = _line_chars_text(rebuilt).strip()
        line.confidence = sum(char.confidence for char in rebuilt) / len(rebuilt)
        if "hanwang_geometry_reconciled" not in line.review_flags:
            line.review_flags.append("hanwang_geometry_reconciled")


RECOG_GROUP_CROP_PAD_X = 8
RECOG_GROUP_CROP_PAD_Y = 10
# The native Recog model becomes unstable when a horizontal glyph row is
# materially taller than its 60px model family.  PP-OCR remains authoritative
# for page geometry; only the transient native input is normalized.
NATIVE_RECOG_MAX_CORE_HEIGHT = 60
NATIVE_RECOG_TARGET_CORE_HEIGHT = 56
ENGCUT_LINE_CONTEXT_PAD_X = 2
ENGCUT_LINE_CONTEXT_PAD_Y = 2


def _requires_native_line_height_normalization(
    route: _LineCutMaskedLineRoute,
    group_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if group_bbox is None:
        return False
    return (
        _box_width(route.bbox) > _box_height(route.bbox)
        and _box_height(group_bbox) > NATIVE_RECOG_MAX_CORE_HEIGHT
    )


def _requires_typed_segment_recognition(route: _LineCutMaskedLineRoute) -> bool:
    """Keep native grouping from crossing explicit route ownership borders."""
    return (
        len(route.linecut_segments) > 1
        or any(segment.kind == ROUTE_SEGMENT_TEXT_LATIN for segment in route.excluded_segments)
    )


def _prepare_recog_crop(
    crop: np.ndarray,
    placement: _GroupPlacement,
) -> _PreparedRecogCrop:
    target = placement.native_core_height_target
    if target is None:
        return _PreparedRecogCrop(crop)
    core_height = _box_height(placement.group_bbox)
    if core_height <= 0:
        raise RuntimeError(f"invalid native recognition core bbox: {placement.group_bbox}")
    scale = float(target) / float(core_height)
    if scale >= 1.0:
        return _PreparedRecogCrop(crop)

    import cv2

    crop_height, crop_width = crop.shape[:2]
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return _PreparedRecogCrop(
        resized,
        scale_x=float(resized_width) / float(crop_width),
        scale_y=float(resized_height) / float(crop_height),
    )


def _prepare_full_line_recog_crop(crop: np.ndarray) -> _PreparedRecogCrop:
    crop_height, crop_width = crop.shape[:2]
    if (
        crop_height <= NATIVE_RECOG_MAX_CORE_HEIGHT
        or crop_width <= crop_height
    ):
        return _PreparedRecogCrop(crop)
    scale = float(NATIVE_RECOG_TARGET_CORE_HEIGHT) / float(crop_height)
    import cv2

    resized_width = max(1, int(round(crop_width * scale)))
    resized = cv2.resize(
        crop,
        (resized_width, NATIVE_RECOG_TARGET_CORE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return _PreparedRecogCrop(
        resized,
        scale_x=float(resized_width) / float(crop_width),
        scale_y=float(NATIVE_RECOG_TARGET_CORE_HEIGHT) / float(crop_height),
    )


def _orient_crop_for_native(
    crop: np.ndarray,
    *,
    text_axis: str,
    orientation_angle: int,
) -> _OrientedNativeCrop:
    source_height, source_width = crop.shape[:2]
    if text_axis == TEXT_AXIS_VERTICAL:
        if orientation_angle == -1:
            raise RuntimeError(
                "rotated PP-OCRv6 line is missing textline orientation evidence"
            )
        # Paddle canonicalizes a tall detector crop with a counter-clockwise
        # quarter turn, then reports whether that crop still needs 180 degrees.
        turns = 3 if orientation_angle == 0 else 1
    else:
        turns = 2 if orientation_angle == 180 else 0
    return _OrientedNativeCrop(
        image=np.rot90(crop, k=-turns).copy() if turns else crop,
        rotation_quarters_clockwise=turns,
        source_width=source_width,
        source_height=source_height,
    )


def _line_results_from_oriented_crop(
    lines: list[_NativeLineResult],
    oriented: _OrientedNativeCrop,
) -> list[_NativeLineResult]:
    if oriented.rotation_quarters_clockwise % 4 == 0:
        return lines
    transformed: list[_NativeLineResult] = []
    for line in lines:
        chars = [
            replace(
                char,
                bbox=(
                    oriented.bbox_to_source(char.bbox)
                    if char.bbox is not None
                    else None
                ),
            )
            for char in line.chars
        ]
        transformed.append(replace(
            line,
            bbox=oriented.bbox_to_source(line.bbox),
            chars=chars,
        ))
    return transformed


def _engcut_chars_from_oriented_crop(
    chars: list[EngcutChar],
    oriented: _OrientedNativeCrop,
) -> list[EngcutChar]:
    if oriented.rotation_quarters_clockwise % 4 == 0:
        return chars
    return [
        replace(
            char,
            bbox=(
                oriented.bbox_to_source(char.bbox)
                if char.bbox is not None
                else None
            ),
        )
        for char in chars
    ]


def _engcut_route_line_text_and_chars(
    chars: list[EngcutChar],
    *,
    source: str = LATIN_ENGCUT_ROUTE_SOURCE,
    ppocr_tokens: tuple[PpOcrLatinTokenObservation, ...] = (),
    foreground_word_bbox: tuple[int, int, int, int] | None = None,
    italic_fallback_tokens: frozenset[PpOcrLatinTokenObservation] = frozenset(),
) -> tuple[str, list[_NativeAtomResult], bool]:
    text_parts: list[str] = []
    results: list[_NativeAtomResult] = []
    has_output_group = False
    visible_groups = [
        visible
        for group in _engcut_groups(chars)
        if (visible := [char for char in group if str(char.text or "")])
    ]
    has_ppocr_text_disagreement = False
    token_bindings = [
        _ppocr_token_for_engcut_group(group, ppocr_tokens)
        for group in visible_groups
    ]
    token_binding_counts = {
        token: token_bindings.count(token)
        for token in token_bindings
        if token is not None
    }
    token_group_indices: dict[PpOcrLatinTokenObservation, list[int]] = {}
    for group_index, token in enumerate(token_bindings):
        if token is not None:
            token_group_indices.setdefault(token, []).append(group_index)
    degraded_tokens: set[PpOcrLatinTokenObservation] = set()
    for group_index, (visible, token) in enumerate(zip(visible_groups, token_bindings)):
        if not _engcut_group_has_overlapping_char_bboxes(visible):
            continue
        if token is None:
            group_text = "".join(str(char.text or "") for char in visible)
            raise RuntimeError(
                "EngCut group has degraded character geometry without one "
                f"uniquely bound PP word token: text={group_text!r}"
            )
        degraded_tokens.add(token)
    degraded_tokens.update(
        token for token in italic_fallback_tokens if token in token_binding_counts
    )
    for token in degraded_tokens:
        indices = token_group_indices[token]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise RuntimeError(
                "EngCut groups bound to one degraded PP word token are not contiguous: "
                f"token={token.text!r} groups={indices}"
            )
    emitted_degraded_tokens: set[PpOcrLatinTokenObservation] = set()
    for visible, token in zip(visible_groups, token_bindings):
        if token in degraded_tokens:
            assert token is not None
            if len(ppocr_tokens) != 1 or foreground_word_bbox is None:
                raise RuntimeError(
                    "EngCut degraded word requires one PP token with uniquely owned "
                    "route foreground geometry"
                )
            if token in emitted_degraded_tokens:
                continue
            if has_output_group:
                text_parts.append(" ")
                results.append(
                    _NativeAtomResult(
                        text=" ",
                        confidence=0.0,
                        bbox=None,
                        candidates=[" "],
                        source=source,
                        bbox_granularity="space",
                        token_text=" ",
                    )
                )
            token_groups = [visible_groups[index] for index in token_group_indices[token]]
            native_chars = [char for group in token_groups for char in group]
            native_text = "".join(str(char.text or "") for char in native_chars)
            native_boxes = [char.bbox for char in native_chars if char.bbox is not None]
            fallback_source = (
                PPOCR_LATIN_ITALIC_WORD_FALLBACK_SOURCE
                if token in italic_fallback_tokens
                else PPOCR_LATIN_TOKEN_GEOMETRY_FALLBACK_SOURCE
            )
            text_parts.append(token.text)
            results.append(_NativeAtomResult(
                text=token.text,
                confidence=0.0,
                bbox=foreground_word_bbox,
                candidates=[token.text],
                external_candidates=[CharOcrCandidateObservation(
                    text=native_text,
                    confidence=0.0,
                    source=ENGCUT_DEGRADED_NATIVE_SOURCE,
                    bbox=union_xyxy(native_boxes),
                )],
                source=fallback_source,
                bbox_granularity="word",
                token_text=token.text,
            ))
            if native_text != token.text:
                has_ppocr_text_disagreement = True
            has_output_group = True
            emitted_degraded_tokens.add(token)
            continue
        if has_output_group:
            text_parts.append(" ")
            results.append(
                _NativeAtomResult(
                    text=" ",
                    confidence=0.0,
                    bbox=None,
                    candidates=[" "],
                    source=source,
                    bbox_granularity="space",
                    token_text=" ",
                )
            )
        group_text = "".join(str(char.text or "") for char in visible)
        has_output_group = True
        uniquely_bound_token = (
            token
            if token is not None and token_binding_counts.get(token) == 1
            else None
        )
        can_bind_ppocr_candidates = (
            uniquely_bound_token is not None
            and group_text != uniquely_bound_token.text
            and len(visible) == len(uniquely_bound_token.text)
            and all(len(str(char.text or "")) == 1 for char in visible)
        )
        if uniquely_bound_token is not None and group_text != uniquely_bound_token.text:
            has_ppocr_text_disagreement = True
        if can_bind_ppocr_candidates:
            text_parts.append(group_text)
            results.extend(
                _NativeAtomResult(
                    text=str(native_char.text or ""),
                    confidence=0.0,
                    bbox=native_char.bbox,
                    candidates=[str(native_char.text or "")],
                    external_candidates=[CharOcrCandidateObservation(
                        text=aligned_text,
                        confidence=0.0,
                        source=PPOCR_LATIN_TOKEN_ALIGNMENT_SOURCE,
                        bbox=uniquely_bound_token.bbox,
                    )],
                    source=source,
                    bbox_granularity="char",
                    token_text=str(native_char.text or ""),
                )
                for native_char, aligned_text in zip(visible, uniquely_bound_token.text)
            )
            continue
        text_parts.append(group_text)
        results.extend(
            _NativeAtomResult(
                text=str(char.text or ""),
                confidence=0.0,
                bbox=char.bbox,
                candidates=[str(char.text or "")],
                source=source,
                bbox_granularity="char",
                token_text=str(char.text or ""),
            )
            for char in visible
        )
    return "".join(text_parts), results, has_ppocr_text_disagreement


def _latin_letter_count(text: str) -> int:
    return sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _italic_word_fallback_tokens(
    image_bgr: np.ndarray,
    route: _EngCutMaskedLineRoute,
    segment: _TextRoute,
    chars: list[EngcutChar],
) -> frozenset[PpOcrLatinTokenObservation]:
    if (
        route.text_axis != TEXT_AXIS_HORIZONTAL
        or route.orientation_angle == 180
        or len(segment.ppocr_latin_tokens) != 1
    ):
        return frozenset()
    token = segment.ppocr_latin_tokens[0]
    if _latin_letter_count(token.text) < 2:
        return frozenset()
    native_text = "".join(
        str(char.text or "")
        for group in _engcut_groups(chars)
        for char in group
        if str(char.text or "")
    )
    extra_symbols = {
        char for char in native_text if not char.isalnum() and char not in token.text
    }
    has_owned_symbol_conflict = any(
        observation.text in extra_symbols
        and (
            _intersect_xyxy(observation.bbox, segment.bbox) is not None
            or _intersect_xyxy(observation.proposal_bbox, segment.bbox) is not None
        )
        for observation in route.ppocr_symbol_observations
    )
    if has_owned_symbol_conflict:
        return frozenset()
    measurement = measure_latin_right_slant(image_bgr, segment.bbox)
    return (
        frozenset((token,))
        if is_latin_right_slant_fallback(measurement)
        else frozenset()
    )


def _engcut_group_has_overlapping_char_bboxes(chars: list[EngcutChar]) -> bool:
    boxes = [char.bbox for char in chars]
    if any(box is None for box in boxes):
        return False
    ordered = sorted(
        (box for box in boxes if box is not None),
        key=lambda box: (box[0], box[1], box[2], box[3]),
    )
    return any(right[0] < left[2] for left, right in zip(ordered, ordered[1:]))


def _ppocr_token_for_engcut_group(
    chars: list[EngcutChar],
    tokens: tuple[PpOcrLatinTokenObservation, ...],
) -> PpOcrLatinTokenObservation | None:
    boxes = [char.bbox for char in chars if char.bbox is not None]
    if len(boxes) != len(chars) or not boxes:
        return None
    center_x, center_y = _bbox_center(union_xyxy(boxes))
    matches = [
        token
        for token in tokens
        if token.bbox[0] <= center_x < token.bbox[2]
        and token.bbox[1] <= center_y < token.bbox[3]
    ]
    return matches[0] if len(matches) == 1 else None


def _materialize_engcut_masked_line_crop(
    image_bgr: np.ndarray,
    route: _EngCutMaskedLineRoute,
) -> tuple[np.ndarray, int, int]:
    """Return a full-line canvas containing only approved Latin/digit pixels.

    EngCut sees one physical line with a small outer context margin.  Approved
    segment pixels are copied without expansion; CJK, formulas, and structural
    regions stay white and cannot leak back across a route boundary.
    """
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _expand_xyxy(
        route.bbox,
        width,
        height,
        pad_x=ENGCUT_LINE_CONTEXT_PAD_X,
        pad_y=ENGCUT_LINE_CONTEXT_PAD_Y,
    )
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid masked EngCut line bbox: {route.bbox}")
    canvas = np.full_like(image_bgr[y1:y2, x1:x2], 255)
    for segment in route.segments:
        sx1, sy1, sx2, sy2 = segment.bbox
        sx1 = max(x1, sx1)
        sy1 = max(y1, sy1)
        sx2 = min(x2, sx2)
        sy2 = min(y2, sy2)
        if sx2 <= sx1 or sy2 <= sy1:
            raise RuntimeError(
                "masked EngCut segment does not intersect its physical line: "
                f"line={route.bbox} segment={segment.bbox}"
            )
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = image_bgr[sy1:sy2, sx1:sx2]
    return canvas, x1, y1


def _write_masked_engcut_line_hook(
    crop: np.ndarray,
    route: _EngCutMaskedLineRoute,
    *,
    offset_x: int,
    offset_y: int,
    source_shape: tuple[int, int] | None = None,
    rotation_quarters_clockwise: int = 0,
) -> None:
    """Write the actual EngCut input only when the native debug hook is enabled."""
    hook_dir = os.environ.get("HANWANG_MICRO_RECBLOCK_HOOK_DIR", "").strip()
    if not hook_dir:
        return
    try:
        import cv2

        out_dir = Path(hook_dir) / "engcut_masked_inputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        stem = (
            f"engcut_masked_{stamp}_block_{route.block_idx:03d}_"
            f"line_{route.line_idx:03d}"
        )
        image_path = out_dir / f"{stem}.png"
        if not cv2.imwrite(str(image_path), crop):
            raise RuntimeError(f"cannot write {image_path.name}")
        source_height, source_width = source_shape or crop.shape[:2]
        crop_bbox = [
            offset_x,
            offset_y,
            offset_x + int(source_width),
            offset_y + int(source_height),
        ]
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "schema": "hanwang_engcut_masked_input.v3",
                    "line_bbox": list(route.bbox),
                    "crop_bbox": crop_bbox,
                    "native_rotation_quarters_clockwise": rotation_quarters_clockwise,
                    "engcut_segments": [
                        {
                            "route_key": list(segment.key),
                            "bbox": list(segment.bbox),
                            **(
                                {"ppocr_latin_fallback_text": segment.ppocr_latin_fallback_text}
                                if segment.ppocr_latin_fallback_text
                                else {}
                            ),
                            **(
                                {"content_bbox": list(segment.content_bbox)}
                                if segment.content_bbox is not None
                                else {}
                            ),
                            **(
                                {
                                    "ppocr_latin_tokens": [
                                        {"text": token.text, "bbox": list(token.bbox)}
                                        for token in segment.ppocr_latin_tokens
                                    ]
                                }
                                if segment.ppocr_latin_tokens
                                else {}
                            ),
                        }
                        for segment in route.segments
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write masked EngCut input hook: %s", exc)


def _engcut_groups(chars: list[EngcutChar]) -> list[list[EngcutChar]]:
    """Preserve native group order while exposing the geometry of each group."""
    groups: dict[tuple[int, int], list[EngcutChar]] = {}
    order: list[tuple[int, int]] = []
    for char in chars:
        key = char.line_index, char.group_index
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(char)
    return [groups[key] for key in order if groups[key]]


def _engcut_segment_for_char(
    char: EngcutChar,
    segments: tuple[_TextRoute, ...],
) -> _TextRoute:
    if char.bbox is None:
        raise RuntimeError("EngCut masked-line character has no page geometry")
    center_x, center_y = _bbox_center(char.bbox)
    matches = [
        segment
        for segment in segments
        if segment.bbox[0] <= center_x < segment.bbox[2]
        and segment.bbox[1] <= center_y < segment.bbox[3]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "EngCut masked-line character cannot be uniquely rebound to one route: "
            f"char_bbox={char.bbox} owners={[item.bbox for item in matches]}"
        )
    return matches[0]


def _recognize_engcut_masked_line(
    image_bgr: np.ndarray,
    route: _EngCutMaskedLineRoute,
    stats: RunStats,
    *,
    timeout: float,
) -> dict[tuple[int, int, int], _NativeLineResult]:
    """Recognize a Latin-only physical line and rebind each native character.

    Every character must belong to exactly one Latin segment. An empty segment
    may use its explicit PP token fallback as a reviewable word carrier.
    """
    crop, offset_x, offset_y = _materialize_engcut_masked_line_crop(image_bgr, route)
    if crop.size == 0:
        raise RuntimeError(f"empty masked EngCut line crop: {route.bbox}")
    oriented = _orient_crop_for_native(
        crop,
        text_axis=route.text_axis,
        orientation_angle=route.orientation_angle,
    )
    _write_masked_engcut_line_hook(
        oriented.image,
        route,
        offset_x=offset_x,
        offset_y=offset_y,
        source_shape=crop.shape[:2],
        rotation_quarters_clockwise=oriented.rotation_quarters_clockwise,
    )
    stats.engcut_route_calls += 1
    raw_eng20 = native_bridge.run_eng20_recogline(oriented.image, timeout=timeout)
    local_chars = _engcut_chars_from_oriented_crop(
        engcut_chars_from_payload(raw_eng20),
        oriented,
    )
    page_chars = offset_engcut_chars(local_chars, dx=offset_x, dy=offset_y)
    groups = _engcut_groups(page_chars)
    grouped_by_route: dict[tuple[int, int, int], list[EngcutChar]] = {
        segment.key: [] for segment in route.segments
    }
    for group in groups:
        for char in group:
            owner = _engcut_segment_for_char(char, route.segments)
            grouped_by_route[owner.key].append(char)

    results: dict[tuple[int, int, int], _NativeLineResult] = {}
    for segment in route.segments:
        if segment.kind != ROUTE_SEGMENT_TEXT_LATIN:
            raise RuntimeError(f"EngCut received a non-Latin route: {segment.kind!r}")
        chars = grouped_by_route[segment.key]
        source = LATIN_ENGCUT_ROUTE_SOURCE
        italic_fallback_tokens = _italic_word_fallback_tokens(
            image_bgr,
            route,
            segment,
            chars,
        )
        text, char_results, has_token_disagreement = _engcut_route_line_text_and_chars(
            chars,
            source=source,
            ppocr_tokens=segment.ppocr_latin_tokens,
            foreground_word_bbox=(
                segment.bbox if len(segment.ppocr_latin_tokens) == 1 else None
            ),
            italic_fallback_tokens=italic_fallback_tokens,
        )
        if not text or not any(char.text.strip() for char in char_results):
            fallback_text = str(segment.ppocr_latin_fallback_text or "").strip()
            if not fallback_text:
                raise RuntimeError(
                    "EngCut masked-line result is missing a Latin routing segment "
                    "without PP fallback text: "
                    f"line={route.bbox} segment={segment.bbox}"
                )
            stats.latin_empty_native_fallbacks += 1
            results[segment.key] = _NativeLineResult(
                text=fallback_text,
                bbox=segment.bbox,
                confidence=0.0,
                chars=[_NativeAtomResult(
                    text=fallback_text,
                    confidence=0.0,
                    bbox=segment.bbox,
                    candidates=[fallback_text],
                    source=LATIN_EMPTY_NATIVE_FALLBACK_SOURCE,
                    bbox_granularity="word",
                    token_text=fallback_text,
                )],
                source=LATIN_EMPTY_NATIVE_FALLBACK_SOURCE,
                bbox_source="ppocrv6_latin_route_mask",
                review_flags=[LATIN_EMPTY_NATIVE_FALLBACK_FLAG],
            )
            continue
        boxes = [char.bbox for char in char_results if char.bbox is not None]
        token_geometry_fallback_count = sum(
            char.source in {
                PPOCR_LATIN_TOKEN_GEOMETRY_FALLBACK_SOURCE,
                PPOCR_LATIN_ITALIC_WORD_FALLBACK_SOURCE,
            }
            for char in char_results
        )
        italic_word_fallback_count = sum(
            char.source == PPOCR_LATIN_ITALIC_WORD_FALLBACK_SOURCE
            for char in char_results
        )
        stats.latin_token_geometry_fallbacks += token_geometry_fallback_count
        stats.latin_italic_word_fallbacks += italic_word_fallback_count
        if has_token_disagreement:
            stats.latin_token_text_disagreements += 1
        review_flags: list[str] = []
        if token_geometry_fallback_count:
            review_flags.append(PPOCR_LATIN_TOKEN_GEOMETRY_FALLBACK_FLAG)
        if italic_word_fallback_count:
            review_flags.append(PPOCR_LATIN_ITALIC_WORD_FALLBACK_FLAG)
        if has_token_disagreement:
            review_flags.append(PPOCR_LATIN_TOKEN_DISAGREEMENT_FLAG)
        results[segment.key] = _NativeLineResult(
            text=text,
            bbox=union_xyxy(boxes) if boxes else segment.bbox,
            confidence=0.0,
            chars=char_results,
            source=(
                f"{source}+ppocrv6_token_text_route_foreground_geometry"
                if token_geometry_fallback_count
                else source
            ),
            bbox_source="text_latin_masked_line_engcut",
            review_flags=review_flags,
        )
    return results


def _recognize_engcut_masked_lines(
    image_bgr: np.ndarray,
    routes: list[_EngCutMaskedLineRoute],
    stats: RunStats,
    *,
    timeout: float,
) -> list[tuple[_EngCutMaskedLineRoute, dict[tuple[int, int, int], _NativeLineResult]]]:
    """Recognize independent physical lines with bounded native concurrency.

    The process-wide executor caps total EngCut subprocess pressure while the
    per-page chunks keep one page from occupying every worker when page OCR is
    already concurrent.
    """
    results: list[tuple[_EngCutMaskedLineRoute, dict[tuple[int, int, int], _NativeLineResult]]] = []

    def recognize(route: _EngCutMaskedLineRoute):
        local_stats = RunStats()
        route_results = _recognize_engcut_masked_line(
            image_bgr,
            route,
            local_stats,
            timeout=timeout,
        )
        return route_results, local_stats

    for start in range(0, len(routes), _MAX_ENGCUT_LINES_PER_PAGE):
        chunk = routes[start:start + _MAX_ENGCUT_LINES_PER_PAGE]
        if len(chunk) == 1:
            route_results, local_stats = recognize(chunk[0])
            stats.engcut_route_calls += local_stats.engcut_route_calls
            stats.latin_empty_native_fallbacks += local_stats.latin_empty_native_fallbacks
            stats.latin_token_geometry_fallbacks += local_stats.latin_token_geometry_fallbacks
            stats.latin_italic_word_fallbacks += local_stats.latin_italic_word_fallbacks
            stats.latin_token_text_disagreements += local_stats.latin_token_text_disagreements
            results.append((chunk[0], route_results))
            continue
        futures = [_ENGCUT_NATIVE_EXECUTOR.submit(recognize, route) for route in chunk]
        for route, future in zip(chunk, futures):
            route_results, local_stats = future.result()
            stats.engcut_route_calls += local_stats.engcut_route_calls
            stats.latin_empty_native_fallbacks += local_stats.latin_empty_native_fallbacks
            stats.latin_token_geometry_fallbacks += local_stats.latin_token_geometry_fallbacks
            stats.latin_italic_word_fallbacks += local_stats.latin_italic_word_fallbacks
            stats.latin_token_text_disagreements += local_stats.latin_token_text_disagreements
            results.append((route, route_results))
    return results


def _intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0, right - left) * max(0, bottom - top)


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _chunk_group_bboxes(
    group_bboxes: list[tuple[int, int, int, int]],
    area_indices: list[int],
    *,
    max_groups: int | None = None,
) -> list[tuple[list[tuple[int, int, int, int]], list[int]]]:
    if max_groups is None:
        max_groups = MAX_RECOG_BATCH_GROUPS
    chunks: list[tuple[list[tuple[int, int, int, int]], list[int]]] = []
    current_bboxes: list[tuple[int, int, int, int]] = []
    current_areas: list[int] = []
    for bbox, area_idx in zip(group_bboxes, area_indices):
        candidate_bboxes = [*current_bboxes, bbox]
        over_limit = len(candidate_bboxes) > max_groups
        if current_bboxes and over_limit:
            chunks.append((current_bboxes, current_areas))
            current_bboxes = [bbox]
            current_areas = [area_idx]
        else:
            current_bboxes = candidate_bboxes
            current_areas = [*current_areas, area_idx]
    if current_bboxes:
        chunks.append((current_bboxes, current_areas))
    return chunks


def _write_micro_recblock_hook(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict],
    rows: list[_NativeRegionResult],
    stats: RunStats,
) -> None:
    hook_dir = os.environ.get("HANWANG_MICRO_RECBLOCK_HOOK_DIR", "").strip()
    if not hook_dir:
        return
    try:
        out_dir = Path(hook_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        payload = {
            "schema": "hanwang_micro_recblock_hook.v1",
            "timestamp_ms": stamp,
            "image_shape": list(image_bgr.shape),
            "crop_padding": {
                "pad_x": RECOG_GROUP_CROP_PAD_X,
                "pad_y": RECOG_GROUP_CROP_PAD_Y,
                "retry_strategy": "routing_segment_context",
            },
            "stats": dict(stats.__dict__),
            "input_blocks": ppvl_blocks,
            "rows": [row.to_dict() for row in rows],
        }
        (out_dir / f"micro_recblock_{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        try:
            import cv2

            canvas = image_bgr.copy()
            for row in rows:
                x1, y1, x2, y2 = row.block_bbox
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
                for route in row.route_text_slice_bboxes:
                    rx1, ry1, rx2, ry2 = route
                    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                for group in row.recog_group_bboxes:
                    gx1, gy1, gx2, gy2 = group
                    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
                for line in row.lines:
                    lx1, ly1, lx2, ly2 = line.bbox
                    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (0, 165, 255), 2)
                    for char in line.chars:
                        if char.bbox is None:
                            continue
                        cx1, cy1, cx2, cy2 = char.bbox
                        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"micro_recblock_{stamp}_overlay.png"), canvas)
        except Exception as exc:
            logger.warning("Failed to write Hanwang micro_recblock hook overlay: %s", exc)
    except Exception as exc:
        logger.warning("Failed to write Hanwang micro_recblock hook: %s", exc)


def _compile_native_route_map(
    input_rows: tuple[CharOcrInputRow, ...],
    routing_plan: PageRoutingPlan,
) -> dict[int, tuple[RoutingLine, ...]]:
    """Map immutable page routes to transient native row indexes.

    ``PageRoutingPlan`` is the only production CharOCR dispatch input. Stable
    identity comes from typed input rows; the transient vendor dictionaries
    contain only image-region fields consumed by Hanwang.
    """
    if not routing_plan.is_dispatchable:
        raise RuntimeError("Hanwang native runner received a non-dispatchable page routing plan")
    rows_by_uid = {row.block_uid: index for index, row in enumerate(input_rows)}
    routes_by_block_index: dict[int, tuple[RoutingLine, ...]] = {}
    for block_route in routing_plan.blocks:
        block_index = rows_by_uid.get(block_route.block_uid)
        if block_index is None:
            raise RuntimeError(
                "CharOCR routing plan references a layout block missing from native input: "
                f"{block_route.block_uid}"
            )
        routes_by_block_index[block_index] = tuple(block_route.plan.lines)
    missing_text_rows = [
        index
        for index, row in enumerate(input_rows)
        if row.ocr_policy is OcrPolicy.TEXT_OCR
        and index not in routes_by_block_index
    ]
    if missing_text_rows:
        raise RuntimeError(
            "CharOCR native input contains text rows outside the explicit page routing plan: "
            + ", ".join(str(index) for index in missing_text_rows)
        )
    return routes_by_block_index


def run_micro_recblock(
    image_bgr: np.ndarray,
    input_rows: tuple[CharOcrInputRow, ...],
    *,
    seg_timeout: float = 120.0,
    recog_timeout: float = 60.0,
    include_chars: bool = True,
    routing_plan: PageRoutingPlan,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[tuple[CharOcrRegionObservation, ...], RunStats]:
    """Run native CharOCR from one explicit page routing plan."""
    global _BATCH_DISABLED_FOR_SESSION, _BATCH_DISABLE_REASON
    if not isinstance(input_rows, tuple) or any(
        not isinstance(row, CharOcrInputRow) for row in input_rows
    ):
        raise TypeError("Hanwang native runner requires typed CharOcrInputRow values")
    ppvl_blocks = [
        {
            "block_label": row.label,
            "source_label": row.label,
            "block_bbox": list(row.bbox),
            "block_content": row.content,
        }
        for row in input_rows
    ]
    height, width = image_bgr.shape[:2]
    native_routes_by_block_index = _compile_native_route_map(input_rows, routing_plan)
    stats = RunStats(n_blocks_total=len(ppvl_blocks))
    text_indices: list[int] = []
    skip_indices: list[int] = []
    for idx, block in enumerate(ppvl_blocks):
        label = _effective_label_for_block(block)
        if _is_unknown_hanwang_label(label):
            stats.n_unknown_paddle_labels += 1
        if input_rows[idx].ocr_policy is OcrPolicy.TEXT_OCR:
            text_indices.append(idx)
        else:
            skip_indices.append(idx)

    stats.n_blocks_hanwang = len(text_indices)
    stats.n_blocks_ppvl = len(skip_indices)
    rows: list[_NativeRegionResult | None] = [None] * len(ppvl_blocks)
    text_routes: list[_TextRoute] = []
    engcut_masked_line_routes: list[_EngCutMaskedLineRoute] = []
    linecut_masked_line_routes: list[_LineCutMaskedLineRoute] = []
    block_text_routes: dict[int, list[_TextRoute]] = {}
    recog_group_bboxes_by_route: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    segimg_group_audits_by_route: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for block_idx in text_indices:
        routes = _text_route_bboxes_from_lines(
            block_idx,
            native_routes_by_block_index.get(block_idx, ()),
        )
        block_text_routes[block_idx] = routes
        text_routes.extend(routes)
        engcut_masked_line_routes.extend(
            _engcut_masked_line_routes_from_lines(
                block_idx,
                native_routes_by_block_index.get(block_idx, ()),
            )
        )
        linecut_masked_line_routes.extend(
            _linecut_masked_line_routes_from_lines(
                block_idx,
                native_routes_by_block_index.get(block_idx, ()),
            )
        )
        for route in routes:
            recog_group_bboxes_by_route[route.key] = []
            segimg_group_audits_by_route[route.key] = []
    oriented_linecut_routes = [
        route
        for route in linecut_masked_line_routes
        if route.text_axis == TEXT_AXIS_VERTICAL or route.orientation_angle == 180
    ]
    linecut_masked_line_routes = [
        route for route in linecut_masked_line_routes
        if route not in oriented_linecut_routes
    ]
    typed_segment_area_indices = {
        area_idx
        for area_idx, route in enumerate(linecut_masked_line_routes)
        if _requires_typed_segment_recognition(route)
    }
    text_route_recblocks = [route.bbox for route in linecut_masked_line_routes]
    linecut_input_image = _materialize_linecut_masked_page(
        image_bgr,
        linecut_masked_line_routes,
    )

    for idx in skip_indices:
        block = ppvl_blocks[idx]
        label = _effective_label_for_block(block)
        bbox = _block_bbox(block, width, height)
        layout_bbox = _layout_block_bbox(block, width, height)
        block_bbox_source = "layout_block_bbox"
        ppvl_text = _block_text(block)
        synthesize_chars = not is_formula_label(label)
        rows[idx] = _NativeRegionResult(
            block_idx=idx,
            block_uid=input_rows[idx].block_uid,
            block_label=label,
            block_bbox=bbox,
            layout_bbox=layout_bbox,
            block_bbox_source=block_bbox_source,
            source="ppvl",
            text=ppvl_text,
            ppvl_text=ppvl_text,
            lines=[_fallback_line(ppvl_text, bbox, source="ppvl", synthesize_chars=synthesize_chars)],
            raw_block=_raw_block_with_bbox_audit(
                block,
                width,
                height,
                block_bbox=bbox,
                block_bbox_source="layout_block_bbox",
                routing_line_bboxes=[],
            ),
        )

    if text_routes:
        recblocks = text_route_recblocks
        if progress_callback and text_route_recblocks:
            progress_callback(
                0,
                max(1, len(text_route_recblocks)),
                "Hanwang micro-recblock SegImg 分块中…",
            )
        if text_route_recblocks:
            started = time.time()
            seg = native_bridge.run_linecut_segimg(
                linecut_input_image,
                recblocks_xyxy=recblocks,
                timeout=seg_timeout,
            )
            stats.seg_seconds = time.time() - started
        else:
            seg = {"lines": []}

        groups: list[dict] = []
        for area_idx, area in enumerate(seg.get("lines", []) or []):
            for group in area.get("groups", []) or []:
                group["_area_idx"] = area_idx
                groups.append(group)
        started = time.time()
        grouped_lines: dict[tuple[int, int, int], list[_NativeLineResult]] = {
            route.key: []
            for route in text_routes
        }
        for _masked_line_route, engcut_results in _recognize_engcut_masked_lines(
            image_bgr,
            engcut_masked_line_routes,
            stats,
            timeout=min(30.0, max(1.0, float(recog_timeout))),
        ):
            for route_key, result in engcut_results.items():
                grouped_lines[route_key].append(result)
        for route in oriented_linecut_routes:
            oriented_results = _recognize_oriented_linecut_route(
                image_bgr,
                route,
                stats,
                timeout=recog_timeout,
                include_chars=include_chars,
            )
            for route_key, results in oriented_results.items():
                grouped_lines[route_key].extend(results)
            for segment in route.linecut_segments:
                segimg_group_audits_by_route.setdefault(segment.key, []).append({
                    "route_text_slice_bbox": list(route.bbox),
                    "segimg_group_bbox": None,
                    "recog_group_bbox": list(route.bbox),
                    "clipped": False,
                    "dropped": False,
                    "native_input_mode": "oriented_physical_line",
                    "text_axis": route.text_axis,
                    "orientation_angle": route.orientation_angle,
                })
        group_bboxes: list[tuple[int, int, int, int]] = []
        group_area_indices: list[int] = []
        group_core_bboxes: dict[
            tuple[int, tuple[int, int, int, int]],
            tuple[int, int, int, int],
        ] = {}
        group_geometries: list[tuple[
            dict[str, Any],
            tuple[int, int, int, int],
            _LineCutMaskedLineRoute,
            tuple[int, int, int, int],
            tuple[int, int, int, int] | None,
            tuple[int, int, int, int] | None,
        ]] = []
        for group in groups:
            recblock = recblocks[group["_area_idx"]]
            route = linecut_masked_line_routes[group["_area_idx"]]
            raw_group_bbox = _clamp_xyxy(
                _bbox_tuple(group.get("bbox"), recblock),
                width,
                height,
            )
            bbox = _intersect_xyxy(raw_group_bbox, recblock)
            recog_bbox = (
                _expand_xyxy(
                    bbox,
                    width,
                    height,
                    pad_x=RECOG_GROUP_CROP_PAD_X,
                    pad_y=RECOG_GROUP_CROP_PAD_Y,
                )
                if bbox is not None
                else None
            )
            group_geometries.append((group, recblock, route, raw_group_bbox, bbox, recog_bbox))

        normalized_area_indices = {
            group["_area_idx"]
            for group, _recblock, route, _raw_group_bbox, bbox, _recog_bbox in group_geometries
            if _requires_native_line_height_normalization(route, bbox)
        }
        normalized_area_indices -= typed_segment_area_indices
        for group, recblock, route, raw_group_bbox, bbox, recog_bbox in group_geometries:
            for segment in route.linecut_segments:
                segimg_group_audits_by_route.setdefault(segment.key, []).append({
                    "route_text_slice_bbox": list(recblock),
                    "segimg_group_bbox": list(raw_group_bbox),
                    "recog_group_bbox": list(recog_bbox) if recog_bbox is not None else None,
                    "recog_group_bbox_before_padding": list(bbox) if bbox is not None else None,
                    "recog_group_bbox_padded": recog_bbox is not None and recog_bbox != bbox,
                    "clipped": bbox is not None and bbox != raw_group_bbox,
                    "dropped": bbox is None,
                    **(
                        {"native_recog_superseded_by": "normalized_physical_line"}
                        if group["_area_idx"] in normalized_area_indices
                        else (
                            {"native_recog_superseded_by": "typed_linecut_segments"}
                            if group["_area_idx"] in typed_segment_area_indices
                            else {}
                        )
                    ),
                })
            if (
                group["_area_idx"] in normalized_area_indices
                or group["_area_idx"] in typed_segment_area_indices
            ):
                continue
            if recog_bbox is None:
                continue
            if recog_bbox[2] <= recog_bbox[0] or recog_bbox[3] <= recog_bbox[1]:
                continue
            group_bboxes.append(recog_bbox)
            group_area_indices.append(group["_area_idx"])
            group_core_bboxes[(group["_area_idx"], recog_bbox)] = bbox
            recog_group_bboxes_by_route.setdefault(route.linecut_segments[0].key, []).append(recog_bbox)

        for area_idx in sorted(normalized_area_indices):
            route = linecut_masked_line_routes[area_idx]
            recog_bbox = _expand_xyxy(
                route.bbox,
                width,
                height,
                pad_x=RECOG_GROUP_CROP_PAD_X,
                pad_y=RECOG_GROUP_CROP_PAD_Y,
            )
            group_bboxes.append(recog_bbox)
            group_area_indices.append(area_idx)
            group_core_bboxes[(area_idx, recog_bbox)] = route.bbox
            recog_group_bboxes_by_route.setdefault(route.linecut_segments[0].key, []).append(recog_bbox)
            target = route.linecut_segments[0]
            segimg_group_audits_by_route.setdefault(target.key, []).append({
                "route_text_slice_bbox": list(route.bbox),
                "segimg_group_bbox": None,
                "recog_group_bbox": list(recog_bbox),
                "recog_group_bbox_before_padding": list(route.bbox),
                "recog_group_bbox_padded": recog_bbox != route.bbox,
                "clipped": False,
                "dropped": False,
                "native_input_mode": "normalized_physical_line",
                "native_input_core_height": _box_height(route.bbox),
                "native_input_core_height_target": NATIVE_RECOG_TARGET_CORE_HEIGHT,
            })

        direct_segment_owners: dict[
            tuple[int, tuple[int, int, int, int]],
            tuple[tuple[int, int, int], tuple[int, int, int, int]],
        ] = {}
        for area_idx in sorted(typed_segment_area_indices):
            route = linecut_masked_line_routes[area_idx]
            for segment in route.linecut_segments:
                recog_bbox = _expand_xyxy(
                    segment.bbox,
                    width,
                    height,
                    pad_x=RECOG_GROUP_CROP_PAD_X,
                    pad_y=RECOG_GROUP_CROP_PAD_Y,
                )
                group_bboxes.append(recog_bbox)
                group_area_indices.append(area_idx)
                group_core_bboxes[(area_idx, recog_bbox)] = segment.bbox
                direct_segment_owners[(area_idx, recog_bbox)] = (segment.key, segment.bbox)
                recog_group_bboxes_by_route.setdefault(segment.key, []).append(recog_bbox)
                target_height = (
                    NATIVE_RECOG_TARGET_CORE_HEIGHT
                    if _requires_native_line_height_normalization(route, segment.bbox)
                    else None
                )
                segimg_group_audits_by_route.setdefault(segment.key, []).append({
                    "route_text_slice_bbox": list(route.bbox),
                    "segimg_group_bbox": None,
                    "recog_group_bbox": list(recog_bbox),
                    "recog_group_bbox_before_padding": list(segment.bbox),
                    "recog_group_bbox_padded": recog_bbox != segment.bbox,
                    "clipped": False,
                    "dropped": False,
                    "native_input_mode": "typed_linecut_segment",
                    "native_input_core_height": _box_height(segment.bbox),
                    **(
                        {"native_input_core_height_target": target_height}
                        if target_height is not None
                        else {}
                    ),
                })

        stats.n_groups = len(group_bboxes)

        total_groups = max(1, len(group_bboxes))
        if progress_callback:
            progress_callback(
                0,
                total_groups,
                f"Hanwang micro-recblock Recog 准备中：{len(group_bboxes)} 个 group",
            )

        def update_recog_group_audit(placement: _GroupPlacement, values: dict[str, Any]) -> None:
            route = linecut_masked_line_routes[placement.area_idx]
            target_segments = (
                [segment for segment in route.linecut_segments if segment.key == placement.owner_key]
                if placement.owner_key is not None
                else list(route.linecut_segments)
            )
            audits = [
                item
                for segment in target_segments
                for item in segimg_group_audits_by_route.setdefault(segment.key, [])
            ]
            if placement.native_core_height_target is not None:
                for item in audits:
                    if (
                        item.get("native_input_mode") == "normalized_physical_line"
                        and item.get("recog_group_bbox") == list(placement.page_bbox)
                    ):
                        item.update(values)
                        return
            for item in audits:
                if item.get("recog_group_bbox") == list(placement.page_bbox):
                    item.update(values)
                    return
            target = route.linecut_segments[0]
            segimg_group_audits_by_route.setdefault(target.key, []).append({
                "route_text_slice_bbox": list(route.bbox),
                "segimg_group_bbox": list(placement.page_bbox),
                "recog_group_bbox": list(placement.page_bbox),
                "clipped": False,
                "dropped": False,
                **values,
            })

        def mark_recog_group_failure(
            placement: _GroupPlacement,
            error: Exception,
            *,
            retry_bbox: tuple[int, int, int, int] | None = None,
            retry_error: Exception | None = None,
        ) -> None:
            stats.recog_group_failures += 1
            values: dict[str, Any] = {
                "recog_failed": True,
                "recog_error": str(error),
            }
            if retry_bbox is not None:
                values.update({
                    "recog_retry_attempted": True,
                    "recog_retry_strategy": "routing_segment_context",
                    "recog_retry_bbox": list(retry_bbox),
                    "recog_retry_succeeded": False,
                })
            if retry_error is not None:
                values["recog_retry_error"] = str(retry_error)
            update_recog_group_audit(placement, values)

        def mark_recog_group_retry_success(
            placement: _GroupPlacement,
            *,
            original_error: Exception,
            retry_bbox: tuple[int, int, int, int],
        ) -> None:
            update_recog_group_audit(
                placement,
                {
                    "recog_retry_attempted": True,
                    "recog_retry_strategy": "routing_segment_context",
                    "recog_retry_original_error": str(original_error),
                    "recog_retry_bbox": list(retry_bbox),
                    "recog_retry_succeeded": True,
                },
            )

        def retry_bbox_from_routing_segment(
            placement: _GroupPlacement,
        ) -> tuple[int, int, int, int] | None:
            route = linecut_masked_line_routes[placement.area_idx]
            core_bbox = placement.owner_bbox or route.bbox
            bbox = _expand_xyxy(
                core_bbox,
                width,
                height,
                pad_x=RECOG_GROUP_CROP_PAD_X,
                pad_y=RECOG_GROUP_CROP_PAD_Y,
            )
            return bbox if bbox != placement.page_bbox else None

        def recognize_individually(placements: list[_GroupPlacement]) -> None:
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crop = linecut_input_image[top:bottom, left:right].copy()
                prepared = _prepare_recog_crop(crop, placement)
                crop_h, crop_w = prepared.image.shape[:2]
                offset_left = left
                offset_top = top
                active_crop = prepared.image
                active_prepared = prepared
                used_route_context_retry = False
                if prepared.normalized:
                    update_recog_group_audit(placement, {
                        "native_input_original_shape": list(crop.shape[:2]),
                        "native_input_shape": list(prepared.image.shape[:2]),
                        "native_input_scale_x": prepared.scale_x,
                        "native_input_scale_y": prepared.scale_y,
                    })
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        prepared.image,
                        recblock_xyxy=None,
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    retry_bbox = retry_bbox_from_routing_segment(placement)
                    if retry_bbox is not None:
                        retry_left, retry_top, retry_right, retry_bottom = retry_bbox
                        retry_crop = linecut_input_image[retry_top:retry_bottom, retry_left:retry_right].copy()
                        try:
                            stats.recog_group_retry_attempts += 1
                            stats.recog_probe_calls += 1
                            raw = native_bridge.run_linecut_recog(
                                retry_crop,
                                recblock_xyxy=None,
                                with_charrcg=True,
                                timeout=recog_timeout,
                            )
                        except Exception as retry_exc:
                            stats.recog_group_retry_failures += 1
                            logger.warning(
                                "Hanwang micro_recblock group failed bbox=%s retry_bbox=%s: %s; retry: %s",
                                placement.page_bbox,
                                retry_bbox,
                                exc,
                                retry_exc,
                            )
                            mark_recog_group_failure(
                                placement,
                                exc,
                                retry_bbox=retry_bbox,
                                retry_error=retry_exc,
                            )
                            raw = {}
                        else:
                            stats.recog_group_retry_successes += 1
                            logger.info(
                                "Hanwang micro_recblock group retry succeeded bbox=%s retry_bbox=%s",
                                placement.page_bbox,
                                retry_bbox,
                            )
                            mark_recog_group_retry_success(
                                placement,
                                original_error=exc,
                                retry_bbox=retry_bbox,
                            )
                            offset_left, offset_top = retry_left, retry_top
                            active_crop = retry_crop
                            active_prepared = _PreparedRecogCrop(retry_crop)
                            used_route_context_retry = True
                            crop_h, crop_w = retry_crop.shape[:2]
                    else:
                        logger.warning("Hanwang micro_recblock group failed bbox=%s: %s", placement.page_bbox, exc)
                        mark_recog_group_failure(placement, exc)
                        raw = {}
                local_lines = _line_results_from_recog(
                    raw,
                    fallback_bbox=(0, 0, crop_w, crop_h),
                    include_chars=include_chars,
                )
                if include_chars:
                    _reconcile_native_char_geometry(
                        active_crop,
                        local_lines,
                        stats,
                    )
                local_lines = _rescale_line_results(
                    local_lines,
                    scale_x=active_prepared.scale_x,
                    scale_y=active_prepared.scale_y,
                    target_width=(right - left if not used_route_context_retry else active_crop.shape[1]),
                    target_height=(bottom - top if not used_route_context_retry else active_crop.shape[0]),
                )
                route = linecut_masked_line_routes[placement.area_idx]
                offset_lines = _offset_line_results(local_lines, dx=offset_left, dy=offset_top)
                ownership_bbox = placement.owner_bbox or route.bbox
                if used_route_context_retry:
                    offset_lines = _filter_line_results_to_route_bbox(
                        offset_lines,
                        ownership_bbox,
                    )
                routed = _distribute_linecut_results(
                    _filter_line_results_to_route_bbox(offset_lines, ownership_bbox),
                    route,
                )
                for route_key, results in routed.items():
                    grouped_lines[route_key].extend(results)

        def recognize_batch_list(placements: list[_GroupPlacement]) -> None:
            crops: list[np.ndarray] = []
            prepared_crops: list[_PreparedRecogCrop] = []
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crop = linecut_input_image[top:bottom, left:right].copy()
                prepared = _prepare_recog_crop(crop, placement)
                crops.append(crop)
                prepared_crops.append(prepared)
                if prepared.normalized:
                    update_recog_group_audit(placement, {
                        "native_input_original_shape": list(crop.shape[:2]),
                        "native_input_shape": list(prepared.image.shape[:2]),
                        "native_input_scale_x": prepared.scale_x,
                        "native_input_scale_y": prepared.scale_y,
                    })
            stats.recog_probe_calls += 1
            raws = native_bridge.run_linecut_recog_batch_list(
                [prepared.image for prepared in prepared_crops],
                with_charrcg=True,
                timeout=recog_timeout,
            )
            if len(raws) != len(placements):
                raise RuntimeError(f"batch-list result count mismatch: {len(raws)}/{len(placements)}")
            for placement, crop, prepared, raw in zip(placements, crops, prepared_crops, raws):
                crop_h, crop_w = prepared.image.shape[:2]
                local_lines = _line_results_from_recog(
                    raw,
                    fallback_bbox=(0, 0, crop_w, crop_h),
                    include_chars=include_chars,
                )
                if include_chars:
                    _reconcile_native_char_geometry(
                        prepared.image,
                        local_lines,
                        stats,
                    )
                local_lines = _rescale_line_results(
                    local_lines,
                    scale_x=prepared.scale_x,
                    scale_y=prepared.scale_y,
                    target_width=crop.shape[1],
                    target_height=crop.shape[0],
                )
                route = linecut_masked_line_routes[placement.area_idx]
                offset_lines = _offset_line_results(
                    local_lines,
                    dx=placement.page_bbox[0],
                    dy=placement.page_bbox[1],
                )
                ownership_bbox = placement.owner_bbox or route.bbox
                routed = _distribute_linecut_results(
                    _filter_line_results_to_route_bbox(offset_lines, ownership_bbox),
                    route,
                )
                for route_key, results in routed.items():
                    grouped_lines[route_key].extend(results)

        batch_enabled = not _BATCH_DISABLED_FOR_SESSION
        stats.recog_batch_disabled = _BATCH_DISABLED_FOR_SESSION
        completed_groups = 0
        chunks = _chunk_group_bboxes(
            group_bboxes,
            group_area_indices,
            max_groups=MAX_RECOG_BATCH_GROUPS,
        )
        for chunk_index, (chunk_bboxes, chunk_area_indices) in enumerate(chunks):
            placements: list[_GroupPlacement] = []
            for page_bbox, area_idx in zip(chunk_bboxes, chunk_area_indices):
                left, top, right, bottom = page_bbox
                owner = direct_segment_owners.get((area_idx, page_bbox))
                placements.append(
                    _GroupPlacement(
                        area_idx=area_idx,
                        page_bbox=page_bbox,
                        group_bbox=group_core_bboxes[(area_idx, page_bbox)],
                        native_core_height_target=(
                            NATIVE_RECOG_TARGET_CORE_HEIGHT
                            if (
                                area_idx in normalized_area_indices
                                or (
                                    owner is not None
                                    and _requires_native_line_height_normalization(
                                        linecut_masked_line_routes[area_idx],
                                        owner[1],
                                    )
                                )
                            )
                            else None
                        ),
                        owner_key=owner[0] if owner is not None else None,
                        owner_bbox=owner[1] if owner is not None else None,
                    )
                )
            if not placements:
                continue
            stats.recog_batch_chunks += 1
            max_crop_w = max(max(0, right - left) for left, _top, right, _bottom in chunk_bboxes)
            max_crop_h = max(max(0, bottom - top) for _left, top, _right, bottom in chunk_bboxes)
            crop_pixels = sum(
                max(0, right - left) * max(0, bottom - top)
                for left, top, right, bottom in chunk_bboxes
            )
            stats.recog_max_batch_crop_width = max(stats.recog_max_batch_crop_width, max_crop_w)
            stats.recog_max_batch_crop_height = max(stats.recog_max_batch_crop_height, max_crop_h)
            stats.recog_max_batch_crop_pixels = max(stats.recog_max_batch_crop_pixels, crop_pixels)
            stats.recog_full_page_pixels += width * height * len(placements)
            stats.recog_crop_pixels += crop_pixels
            use_batch = batch_enabled and len(placements) > 1
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    f"Hanwang OCR 识别中 {completed_groups}/{total_groups}",
                )
            if use_batch:
                try:
                    recognize_batch_list(placements)
                except Exception as exc:
                    stats.recog_batch_failures += 1
                    stats.recog_batch_disabled = True
                    batch_enabled = False
                    _BATCH_DISABLED_FOR_SESSION = True
                    _BATCH_DISABLE_REASON = (
                        f"chunk={chunk_index + 1} groups={len(placements)} "
                        f"batch-list: {exc}"
                    )
                    logger.warning(
                        "Hanwang micro_recblock batch-list failed; disabling batch for this process "
                        "chunk=%d groups=%d: %s",
                        chunk_index + 1,
                        len(placements),
                        exc,
                    )
                    recognize_individually(placements)
            else:
                recognize_individually(placements)

            completed_groups += len(placements)
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    f"Hanwang OCR 已完成 {completed_groups}/{total_groups}",
                )
        stats.recog_seconds = time.time() - started

        for block_idx in text_indices:
            block = ppvl_blocks[block_idx]
            label = _effective_label_for_block(block)
            native_line_routes = native_routes_by_block_index.get(block_idx, ())
            bbox, layout_bbox, block_bbox_source = _block_geometry_from_page_routes(
                block,
                width,
                height,
                native_line_routes,
            )
            ppvl_text = _block_text(block)
            route_text_slice_bboxes = [route.bbox for route in block_text_routes.get(block_idx, [])]
            recog_group_bboxes = [
                bbox
                for route in block_text_routes.get(block_idx, [])
                for bbox in recog_group_bboxes_by_route.get(route.key, [])
            ]
            segimg_group_audits = [
                dict(item)
                for route in block_text_routes.get(block_idx, [])
                for item in segimg_group_audits_by_route.get(route.key, [])
            ]
            lines = _assemble_routing_lines(
                block_idx=block_idx,
                line_routes=native_line_routes,
                grouped_lines=grouped_lines,
                stats=stats,
                image_bgr=image_bgr,
            )
            _normalize_digitlike_numeric_context_lines(lines)
            hw_text = "".join(line.text for line in lines).strip()
            source = "hanwang"
            text = hw_text
            rows[block_idx] = _NativeRegionResult(
                block_idx=block_idx,
                block_uid=input_rows[block_idx].block_uid,
                block_label=label,
                block_bbox=bbox,
                layout_bbox=layout_bbox,
                block_bbox_source=block_bbox_source,
                route_text_slice_bboxes=route_text_slice_bboxes,
                recog_group_bboxes=recog_group_bboxes,
                segimg_group_audits=segimg_group_audits,
                source=source,
                text=text,
                ppvl_text=ppvl_text,
                group_count=len(lines),
                lines=lines,
                fallback_reason="",
                raw_block=_raw_block_with_bbox_audit(
                    block,
                    width,
                    height,
                    block_bbox=bbox,
                    block_bbox_source=block_bbox_source,
                    routing_line_bboxes=[line.bbox for line in native_line_routes],
                    route_text_slice_bboxes=route_text_slice_bboxes,
                    recog_group_bboxes=recog_group_bboxes,
                    segimg_group_audits=segimg_group_audits,
                ),
            )

    for block_idx in text_indices:
        if rows[block_idx] is not None:
            continue
        block = ppvl_blocks[block_idx]
        label = _effective_label_for_block(block)
        native_line_routes = native_routes_by_block_index.get(block_idx, ())
        bbox, layout_bbox, block_bbox_source = _block_geometry_from_page_routes(
            block,
            width,
            height,
            native_line_routes,
        )
        route_text_slice_bboxes = [route.bbox for route in block_text_routes.get(block_idx, [])]
        recog_group_bboxes = [
            bbox
            for route in block_text_routes.get(block_idx, [])
            for bbox in recog_group_bboxes_by_route.get(route.key, [])
        ]
        segimg_group_audits = [
            dict(item)
            for route in block_text_routes.get(block_idx, [])
            for item in segimg_group_audits_by_route.get(route.key, [])
        ]
        ppvl_text = _block_text(block)
        rows[block_idx] = _NativeRegionResult(
            block_idx=block_idx,
            block_uid=input_rows[block_idx].block_uid,
            block_label=label,
            block_bbox=bbox,
            layout_bbox=layout_bbox,
            block_bbox_source=block_bbox_source,
            route_text_slice_bboxes=route_text_slice_bboxes,
            recog_group_bboxes=recog_group_bboxes,
            segimg_group_audits=segimg_group_audits,
            source="hanwang",
            text="",
            ppvl_text=ppvl_text,
            group_count=0,
            lines=[],
            fallback_reason="",
            raw_block=_raw_block_with_bbox_audit(
                block,
                width,
                height,
                block_bbox=bbox,
                block_bbox_source=block_bbox_source,
                routing_line_bboxes=[line.bbox for line in native_line_routes],
                route_text_slice_bboxes=route_text_slice_bboxes,
                recog_group_bboxes=recog_group_bboxes,
                segimg_group_audits=segimg_group_audits,
            ),
        )

    final_rows = [row for row in rows if row is not None]
    _write_micro_recblock_hook(image_bgr, ppvl_blocks, final_rows, stats)
    return tuple(_native_region_observation(row) for row in final_rows), stats


def _has_geometry(bbox: tuple[int, int, int, int] | None) -> bool:
    return bool(
        bbox is not None
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _observation_candidates(
    atom: _NativeAtomResult,
) -> tuple[CharOcrCandidateObservation, ...]:
    texts = list(atom.candidates)
    if not texts and atom.text:
        texts = [atom.text]
    native_candidates = tuple(
        CharOcrCandidateObservation(
            text=text,
            confidence=(
                atom.candidate_confidences[index]
                if index < len(atom.candidate_confidences)
                else (atom.confidence if index == 0 else 0.0)
            ),
        )
        for index, text in enumerate(texts)
    )
    return (*native_candidates, *atom.external_candidates)


def _native_line_observation(line: _NativeLineResult) -> CharOcrLineObservation:
    atoms = tuple(
        CharOcrAtomObservation(
            text=atom.text,
            bbox=(
                atom.bbox
                if _has_geometry(atom.bbox)
                else line.bbox
            ),
            confidence=atom.confidence,
            source=atom.source,
            granularity=(
                atom.bbox_granularity
                or ("char" if _has_geometry(atom.bbox) else "fallback")
            ),
            token_text=atom.token_text or atom.text,
            candidates=_observation_candidates(atom),
        )
        for atom in line.chars
    )
    return CharOcrLineObservation(
        text=line.text,
        bbox=line.bbox,
        confidence=line.confidence,
        source=line.source,
        atoms=atoms,
        review_flags=tuple(line.review_flags),
    )


def _native_region_observation(
    row: _NativeRegionResult,
) -> CharOcrRegionObservation:
    audit = row.raw_block.get(ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY, {})
    if not isinstance(audit, dict):
        audit = {}
    return CharOcrRegionObservation(
        block_uid=row.block_uid,
        label=row.block_label,
        bbox=row.block_bbox,
        source=row.source,
        lines=tuple(_native_line_observation(line) for line in row.lines),
        audit_json=json.dumps(
            audit,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def _stats_metrics(stats: RunStats) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, str(value))
        for name, value in stats.__dict__.items()
    )


def _validate_runner_regions(
    regions: object,
    request: CharOcrPageRequest,
) -> tuple[CharOcrRegionObservation, ...]:
    if not isinstance(regions, (list, tuple)):
        raise TypeError("Hanwang native runner must return a sequence of regions")
    observed: dict[str, CharOcrRegionObservation] = {}
    for region in regions:
        if not isinstance(region, CharOcrRegionObservation):
            raise TypeError(
                "Hanwang native runner must return CharOcrRegionObservation values"
            )
        if not region.block_uid:
            raise RuntimeError("Hanwang native output region has an empty block UID")
        if region.block_uid in observed:
            raise RuntimeError(
                "Hanwang native runner returned duplicate region UID: "
                + region.block_uid
            )
        observed[region.block_uid] = region

    expected = tuple(row.block_uid for row in request.rows)
    unexpected = sorted(set(observed) - set(expected))
    if unexpected:
        raise RuntimeError(
            "Hanwang native runner returned regions outside the request rows: "
            + ", ".join(unexpected)
        )
    missing = [uid for uid in expected if uid not in observed]
    if missing:
        raise RuntimeError(
            "Hanwang native runner did not return request rows: "
            + ", ".join(missing[:5])
        )
    return tuple(observed[uid] for uid in expected)


class HanwangMicroRecBlockEngine:
    """Hanwang native adapter with one immutable CharOCR page contract."""

    engine_id = "hanwang.micro_recblock"
    bbox_space = "page"

    def __init__(
        self,
        *,
        seg_timeout: float = 120.0,
        recog_timeout: float = 60.0,
        runner=run_micro_recblock,
    ) -> None:
        self._seg_timeout = seg_timeout
        self._recog_timeout = recog_timeout
        self._runner = runner

    def recognize_page(
        self,
        image_bgr: np.ndarray,
        request: CharOcrPageRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> CharOcrPageResult:
        if not isinstance(request, CharOcrPageRequest):
            raise TypeError(
                "Hanwang native adapter requires CharOcrPageRequest"
            )
        runner_result = self._runner(
            image_bgr,
            request.rows,
            seg_timeout=self._seg_timeout,
            recog_timeout=self._recog_timeout,
            include_chars=True,
            routing_plan=request.routing_plan,
            progress_callback=progress_callback,
        )
        if (
            not isinstance(runner_result, tuple)
            or len(runner_result) != 2
        ):
            raise TypeError(
                "Hanwang native runner must return (regions, RunStats)"
            )
        regions, stats = runner_result
        if not isinstance(stats, RunStats):
            raise TypeError("Hanwang native runner returned invalid RunStats")
        ordered_regions = _validate_runner_regions(regions, request)
        logger.info(
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d "
            "groups=%d group_failures=%d",
            request.page.uid,
            stats.n_blocks_total,
            stats.n_blocks_hanwang,
            stats.n_blocks_ppvl,
            stats.n_groups,
            stats.recog_group_failures,
        )
        return CharOcrPageResult(
            page_uid=request.page.uid,
            input_fingerprint=request.input_fingerprint,
            regions=ordered_regions,
            metrics=_stats_metrics(stats),
        )


__all__ = [
    "TEXT_LABELS",
    "SKIP_LABELS",
    "RunStats",
    "HanwangMicroRecBlockEngine",
    "decode_gbk",
    "run_micro_recblock",
]
