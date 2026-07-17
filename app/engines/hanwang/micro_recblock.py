"""Page-level PP-VL block -> Hanwang linecut micro-recblock integration."""
from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.bbox_extraction import bbox_from_variant
from app.geometry.char_reconciler import reconcile_char_geometry
from app.models.block_state import (
    clear_ocr_text_invalidation,
    is_ocr_text_invalidated,
    paddle_binding_dict,
    set_paddle_binding,
)
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_observation import block_ocr_line_observations_by_uid, replace_block_ocr_line_observations
from app.core.block_attributes import route_source_label
from app.core.inline_formula_edit_state import filter_handled_inline_formula_subblocks
from app.models.charocr_routing import (
    PageRoutingPlan,
    PpOcrLatinTokenObservation,
    PpOcrSymbolObservation,
    ROUTE_SEGMENT_TEXT_LATIN,
    TEXT_AXIS_HORIZONTAL,
    TEXT_AXIS_VERTICAL,
    RoutingLine,
    is_text_route_segment_kind,
)
from app.models.char_geometry import NativeGeometryProposal
from app.core.logging import get_logger
from .engcut_payload import (
    EngcutChar,
    engcut_chars_from_payload,
    offset_engcut_chars,
)


_ENGCUT_NATIVE_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="charocr-engcut",
)
_MAX_ENGCUT_LINES_PER_PAGE = 4
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    ROUTE_INLINE_FORMULA_FLAG,
    ROUTE_SUBBLOCKS_FIELD,
    ROUTE_TABLE_FLAG,
    block_bbox_xyxy,
    block_text as paddle_block_text,
    is_formula_label,
    is_formula_style_position_block,
    is_table_label,
    route_authority_label,
    union_xyxy,
    vertical_overlap_ratio,
)
from app.services.formula_crop_ocr_service import recognize_formula_bboxes_with_retry
from app.core.paddle_artifact_index import (
    BINDING_AMBIGUOUS,
    BINDING_EMPTY_REVIEW,
    BINDING_FORMULA_CROP_OCR,
    BINDING_GEOMETRY_HIT,
    BINDING_PARENT_FIGURE_HIT,
    BINDING_PARENT_FORMULA_INFERRED,
    BINDING_PARENT_TABLE_HIT,
)
from app.core.paddle_labels import (
    PADDLE_HANWANG_SKIP_LABELS,
    PADDLE_HANWANG_TEXT_LABELS,
    is_hanwang_skip_label,
    normalize_paddle_label,
)
from app.core.proof_line_facts import proof_block_text, proof_display_text
from app.core.proof_status import proof_status_for
from app.core.raw_ocr_artifact import (
    layout_records_with_route_attachments,
    raw_block_payload,
    raw_layout_records,
)
from app.engines import OCR_BBOX_SPACE_PAGE
from app.models import (
    BBox,
    Block,
    BlockType,
    Char,
    Line,
    OcrPolicy,
    Page,
    ProofLineState,
)
from app.models.layout_block_state import (
    block_source_value,
    is_user_authored_layout_block,
    is_user_authored_layout_source,
)
from app.models.ocr_character_observation import replace_line_ocr_char_observations
from app.models.ocr_text_observation import create_ocr_text_line
from app.models.proof_line_state_store import set_proof_state_for_line

from . import native_bridge

logger = get_logger(__name__)


ROUTE_ROW_PADDLE_BINDING_KEY = "paddle_binding"
ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY = "_hanwang_bbox_audit"
ROUTE_ROW_LAYOUT_BLOCK_UID_KEY = "_layout_block_uid"
ROUTE_ROW_OCR_POLICY_KEY = "_layout_block_ocr_policy"

TEXT_LABELS: set[str] = set(PADDLE_HANWANG_TEXT_LABELS)
SKIP_LABELS: set[str] = set(PADDLE_HANWANG_SKIP_LABELS)
DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG = "hanwang_digitlike_numeric_context"
FORMULA_CROP_OCR_REVIEW_FLAG = "paddle_formula_crop_ocr"
FORMULA_CROP_OCR_FAILED_FLAG = "paddle_formula_crop_ocr_failed"
LATIN_ENGCUT_ROUTE_SOURCE = "hanwang:EngCut:latin_route"
PPOCR_LATIN_TOKEN_ALIGNMENT_SOURCE = "ppocrv6:latin_token_text_alignment"
PPOCR_LATIN_TOKEN_DISAGREEMENT_SUFFIX = ":ppocr_text_disagreement"
PPOCR_LATIN_TOKEN_DISAGREEMENT_FLAG = "latin_token_text_disagreement"
LATIN_EMPTY_NATIVE_FALLBACK_SOURCE = "ppocrv6:latin_route_empty_native"
LATIN_EMPTY_NATIVE_FALLBACK_FLAG = "latin_route_empty_native_ppocr_fallback"
PPOCR_SYMBOL_FOREGROUND_SOURCE = "ppocrv6:symbol_foreground_observation"
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
class CharResult:
    text: str
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    candidates: list[str] = field(default_factory=list)
    source: str = "hanwang:micro_recblock"
    bbox_granularity: str = ""
    token_text: str = ""


@dataclass
class LineResult:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    chars: list[CharResult] = field(default_factory=list)
    source: str = "hanwang"
    bbox_source: str = ""
    review_flags: list[str] = field(default_factory=list)


@dataclass
class BlockResult:
    block_idx: int
    block_label: str
    block_bbox: tuple[int, int, int, int]
    source: str
    text: str
    ppvl_text: str
    group_count: int = 0
    lines: list[LineResult] = field(default_factory=list)
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
    latin_token_text_disagreements: int = 0


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


@dataclass(frozen=True)
class _LayoutOcrInputPlan:
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _LayoutOcrEntry:
    view: LayoutBlockView
    block: Block
    row: dict[str, Any]


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


def _row_ocr_policy(block: dict[str, Any]) -> OcrPolicy:
    raw = str(block.get(ROUTE_ROW_OCR_POLICY_KEY) or "").strip()
    if not raw:
        raise RuntimeError("CharOCR native input row is missing its explicit OCR policy")
    try:
        return OcrPolicy(raw)
    except ValueError as exc:
        raise RuntimeError(f"CharOCR native input row has invalid OCR policy: {raw!r}") from exc


def _row_dispatches_to_text_ocr(block: dict[str, Any]) -> bool:
    return _row_ocr_policy(block) == OcrPolicy.TEXT_OCR


def _block_text(block: dict[str, Any]) -> str:
    return paddle_block_text(block)


def _effective_label_for_block(block: dict[str, Any]) -> str:
    label = _label_from_block(block)
    if map_paddle_label_to_block_type(label) == BlockType.EQUATION:
        return label
    if is_formula_style_position_block(block):
        return "formula"
    return label


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
    lines: list[LineResult],
    *,
    route_bbox: tuple[int, int, int, int],
) -> list[LineResult]:
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
    return [LineResult(
        text="".join(line.text for line in ordered if line.text),
        bbox=route_bbox,
        confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        chars=chars,
        source="hanwang+ppocrv6_routing_line",
        bbox_source="ppocrv6_physical_routing_line",
        review_flags=flags,
    )]


def _apply_missing_symbol_observations(
    lines: list[LineResult],
    observations: tuple[PpOcrSymbolObservation, ...],
) -> list[LineResult]:
    """Add only symbol geometry that both native branches omitted.

    Existing native characters are never merged or resized here. A missing
    symbol is materialized only from a single-glyph PP token whose bbox was
    independently measured from foreground components during route compile.
    """
    if not lines or not observations:
        return lines
    resolved = list(lines)
    for observation in observations:
        center_x, center_y = _bbox_center(observation.bbox)
        owners = [
            index
            for index, line in enumerate(resolved)
            if line.bbox[0] <= center_x < line.bbox[2]
            and line.bbox[1] <= center_y < line.bbox[3]
        ]
        if len(owners) != 1:
            continue
        line_index = owners[0]
        line = resolved[line_index]
        if any(
            char.bbox is not None
            and _intersect_xyxy(char.bbox, observation.bbox) is not None
            for char in line.chars
        ):
            continue
        symbol = CharResult(
            text=observation.text,
            confidence=0.0,
            bbox=observation.bbox,
            candidates=[observation.text],
            source=PPOCR_SYMBOL_FOREGROUND_SOURCE,
            bbox_granularity="char",
            token_text=observation.text,
        )
        insertion = len(line.chars)
        for index, char in enumerate(line.chars):
            if char.bbox is None:
                continue
            if _bbox_center(char.bbox)[0] > center_x:
                insertion = index
                break
        chars = [*line.chars[:insertion], symbol, *line.chars[insertion:]]
        resolved[line_index] = replace(
            line,
            text="".join(char.text for char in chars),
            chars=chars,
        )
    return resolved


def _distribute_linecut_results(
    lines: list[LineResult],
    route: _LineCutMaskedLineRoute,
) -> dict[tuple[int, int, int], list[LineResult]]:
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
            buckets: dict[tuple[int, int, int], list[CharResult]] = {
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
                distributed[segment.key].append(LineResult(
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
) -> dict[tuple[int, int, int], list[LineResult]]:
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


def _cluster_lines_by_shape(lines: list[LineResult]) -> list[list[LineResult]]:
    buckets: list[list[LineResult]] = []
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
    grouped_lines: dict[tuple[int, int, int], list[LineResult]],
) -> list[LineResult]:
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
            LineResult(
                text=text,
                bbox=segment.bbox,
                confidence=0.0,
                chars=[],
                source=f"ppvl_route:{segment.label or 'skip'}",
                bbox_source="ppvl_route_skip_segment",
                review_flags=flags,
            )
        ]

    slice_lines_by_segment: dict[int, list[LineResult]] = {}
    all_text_lines: list[LineResult] = []
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
        merged_text_lines: list[LineResult] = []
        for segment_idx in range(len(segments)):
            merged_text_lines.extend(slice_lines_by_segment.get(segment_idx, []))
        return _apply_missing_symbol_observations(
            _merge_physical_routing_line(
                merged_text_lines,
                route_bbox=route.bbox,
            ),
            route.ppocr_symbol_observations,
        )

    clusters = _cluster_lines_by_shape(all_text_lines)
    if not clusters:
        return []

    assembled: list[LineResult] = []
    for cluster in clusters:
        cluster_bbox = union_xyxy([line.bbox for line in cluster])
        components: list[
            tuple[
                tuple[int, int, int, int],
                str,
                list[CharResult],
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
                formula_char = CharResult(
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
        chars: list[CharResult] = []
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
            LineResult(
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
    return _apply_missing_symbol_observations(
        assembled,
        route.ppocr_symbol_observations,
    )


def _assemble_routing_lines(
    *,
    block_idx: int,
    line_routes: tuple[RoutingLine, ...],
    grouped_lines: dict[tuple[int, int, int], list[LineResult]],
) -> list[LineResult]:
    if not line_routes:
        return []
    assembled: list[LineResult] = []
    for line_idx, route in enumerate(line_routes):
        assembled.extend(
            _assemble_layout_route_line(
                block_idx=block_idx,
                line_idx=line_idx,
                route=route,
                grouped_lines=grouped_lines,
            )
        )
    assembled.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return assembled


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


def _char_result(raw: dict) -> CharResult:
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
    text = candidates[0] if candidates else ""
    confidence = _score_to_confidence(scores[0] if scores else 100)
    return CharResult(
        text=text,
        confidence=confidence,
        bbox=_optional_bbox_tuple(raw.get("bbox")),
        candidates=candidates,
    )


def _fallback_line(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    source: str,
    synthesize_chars: bool = True,
) -> LineResult:
    return LineResult(
        text=text,
        bbox=bbox,
        confidence=0.0,
        source=source,
        bbox_source=f"{source}:bbox",
        chars=(
            [
                CharResult(text=ch, confidence=0.0, bbox=None, candidates=[ch], source=source)
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
) -> list[LineResult]:
    lines: list[LineResult] = []
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
                LineResult(
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
    lines: list[LineResult],
    *,
    dx: int,
    dy: int,
) -> list[LineResult]:
    shifted: list[LineResult] = []
    for line in lines:
        lx1, ly1, lx2, ly2 = line.bbox
        chars: list[CharResult] = []
        for char in line.chars:
            bbox = None
            if char.bbox is not None:
                cx1, cy1, cx2, cy2 = char.bbox
                bbox = (cx1 + dx, cy1 + dy, cx2 + dx, cy2 + dy)
            chars.append(
                CharResult(
                    text=char.text,
                    confidence=char.confidence,
                    bbox=bbox,
                    candidates=list(char.candidates),
                    source=char.source,
                    bbox_granularity=char.bbox_granularity,
                    token_text=char.token_text,
                )
            )
        shifted.append(
            LineResult(
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
    lines: list[LineResult],
    *,
    scale_x: float,
    scale_y: float,
    target_width: int,
    target_height: int,
) -> list[LineResult]:
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

    scaled: list[LineResult] = []
    for line in lines:
        chars = [
            CharResult(
                text=char.text,
                confidence=char.confidence,
                bbox=rescale_bbox(char.bbox) if char.bbox is not None else None,
                candidates=list(char.candidates),
                source=char.source,
                bbox_granularity=char.bbox_granularity,
                token_text=char.token_text,
            )
            for char in line.chars
        ]
        scaled.append(LineResult(
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
    lines: list[LineResult],
    route_bbox: tuple[int, int, int, int],
) -> list[LineResult]:
    """Drop OCR context characters that fall outside the current text route.

    Recognition crops may intentionally include a few pixels of neighbouring
    formula/line context so native OCR sees complete glyphs.  The text fact
    still belongs to the route slice, so characters centered outside that slice
    must not enter the assembled line.
    """
    filtered: list[LineResult] = []
    for line in lines:
        if not line.chars:
            if _intersection_area(line.bbox, route_bbox) > 0:
                filtered.append(line)
            continue
        kept_chars: list[CharResult] = []
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
            LineResult(
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


def _line_chars_text(chars: list[CharResult]) -> str:
    return "".join(char.text for char in chars)


def _normalize_digitlike_numeric_context_lines(lines: list[LineResult]) -> None:
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


def _digitlike_numeric_context_replacement(chars: list[CharResult], index: int) -> str | None:
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


def _adjacent_visible_text(chars: list[CharResult], index: int, *, step: int) -> str:
    pos = index + step
    while 0 <= pos < len(chars):
        text = str(chars[pos].text or "")
        if text.strip():
            return text
        pos += step
    return ""


def _reconcile_native_char_geometry(
    crop_bgr: np.ndarray,
    lines: list[LineResult],
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
        rebuilt: list[CharResult] = []
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
            rebuilt.append(CharResult(
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
    lines: list[LineResult],
    oriented: _OrientedNativeCrop,
) -> list[LineResult]:
    if oriented.rotation_quarters_clockwise % 4 == 0:
        return lines
    transformed: list[LineResult] = []
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
) -> tuple[str, list[CharResult]]:
    text_parts: list[str] = []
    results: list[CharResult] = []
    has_output_group = False
    visible_groups = [
        visible
        for group in _engcut_groups(chars)
        if (visible := [char for char in group if str(char.text or "")])
    ]
    token_bindings = [
        _ppocr_token_for_engcut_group(group, ppocr_tokens)
        for group in visible_groups
    ]
    token_binding_counts = {
        token: token_bindings.count(token)
        for token in token_bindings
        if token is not None
    }
    for visible, token in zip(visible_groups, token_bindings):
        if has_output_group:
            text_parts.append(" ")
            results.append(
                CharResult(
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
        can_align_ppocr_text = (
            token is not None
            and token_binding_counts.get(token) == 1
            and group_text != token.text
            and len(visible) == len(token.text)
            and all(len(str(char.text or "")) == 1 for char in visible)
        )
        if can_align_ppocr_text:
            assert token is not None
            text_parts.append(token.text)
            results.extend(
                CharResult(
                    text=aligned_text,
                    confidence=0.0,
                    bbox=native_char.bbox,
                    candidates=list(dict.fromkeys([
                        str(native_char.text or ""),
                        aligned_text,
                    ])),
                    source=PPOCR_LATIN_TOKEN_ALIGNMENT_SOURCE,
                    bbox_granularity="char",
                    token_text=aligned_text,
                )
                for native_char, aligned_text in zip(visible, token.text)
            )
            continue
        text_parts.append(group_text)
        result_source = source
        if (
            token is not None
            and token_binding_counts.get(token) == 1
            and group_text != token.text
        ):
            result_source = f"{source}{PPOCR_LATIN_TOKEN_DISAGREEMENT_SUFFIX}"
        results.extend(
            CharResult(
                text=str(char.text or ""),
                confidence=0.0,
                bbox=char.bbox,
                candidates=[str(char.text or "")],
                source=result_source,
                bbox_granularity="char",
                token_text=str(char.text or ""),
            )
            for char in visible
        )
    return "".join(text_parts), results


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
) -> dict[tuple[int, int, int], LineResult]:
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

    results: dict[tuple[int, int, int], LineResult] = {}
    for segment in route.segments:
        if segment.kind != ROUTE_SEGMENT_TEXT_LATIN:
            raise RuntimeError(f"EngCut received a non-Latin route: {segment.kind!r}")
        chars = grouped_by_route[segment.key]
        source = LATIN_ENGCUT_ROUTE_SOURCE
        text, char_results = _engcut_route_line_text_and_chars(
            chars,
            source=source,
            ppocr_tokens=segment.ppocr_latin_tokens,
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
            results[segment.key] = LineResult(
                text=fallback_text,
                bbox=segment.bbox,
                confidence=0.0,
                chars=[CharResult(
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
        token_disagreement_count = sum(
            char.source.endswith(PPOCR_LATIN_TOKEN_DISAGREEMENT_SUFFIX)
            for char in char_results
        )
        if token_disagreement_count:
            stats.latin_token_text_disagreements += 1
        results[segment.key] = LineResult(
            text=text,
            bbox=union_xyxy(boxes) if boxes else segment.bbox,
            confidence=0.0,
            chars=char_results,
            source=source,
            bbox_source="text_latin_masked_line_engcut",
            review_flags=(
                [PPOCR_LATIN_TOKEN_DISAGREEMENT_FLAG]
                if token_disagreement_count
                else []
            ),
        )
    return results


def _recognize_engcut_masked_lines(
    image_bgr: np.ndarray,
    routes: list[_EngCutMaskedLineRoute],
    stats: RunStats,
    *,
    timeout: float,
) -> list[tuple[_EngCutMaskedLineRoute, dict[tuple[int, int, int], LineResult]]]:
    """Recognize independent physical lines with bounded native concurrency.

    The process-wide executor caps total EngCut subprocess pressure while the
    per-page chunks keep one page from occupying every worker when page OCR is
    already concurrent.
    """
    results: list[tuple[_EngCutMaskedLineRoute, dict[tuple[int, int, int], LineResult]]] = []

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
            stats.latin_token_text_disagreements += local_stats.latin_token_text_disagreements
            results.append((chunk[0], route_results))
            continue
        futures = [_ENGCUT_NATIVE_EXECUTOR.submit(recognize, route) for route in chunk]
        for route, future in zip(chunk, futures):
            route_results, local_stats = future.result()
            stats.engcut_route_calls += local_stats.engcut_route_calls
            stats.latin_empty_native_fallbacks += local_stats.latin_empty_native_fallbacks
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
    rows: list[BlockResult],
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


def _formula_texts_by_layout_bbox(page: Page) -> dict[tuple[int, int, int, int], str]:
    """Read current formula observations only while projecting a routing plan.

    Formula text remains owned by the formula branch.  This bridge supplies it
    to the transient typed route map consumed by the native runner.
    """
    values: dict[tuple[int, int, int, int], str] = {}
    for view in iter_page_layout_block_views(page):
        if view.block_type != BlockType.EQUATION:
            continue
        lines = block_ocr_line_observations_by_uid(view.uid)
        text = "".join(proof_display_text(line) for line in lines if proof_display_text(line)).strip()
        if text:
            values[tuple(int(value) for value in view.bbox.to_xyxy())] = text
    return values


def _routing_line_for_native_runner(
    line: RoutingLine,
    formula_texts_by_bbox: dict[tuple[int, int, int, int], str],
) -> RoutingLine:
    segments = []
    for segment in line.segments:
        text = segment.text
        if segment.kind == "formula" and not text:
            content_bbox = segment.content_bbox or segment.bbox
            _center_x, center_y = _bbox_center(content_bbox)
            if line.bbox[1] <= center_y < line.bbox[3]:
                text = formula_texts_by_bbox.get(content_bbox, "")
                if not text:
                    containing = [
                        candidate_text
                        for bbox, candidate_text in formula_texts_by_bbox.items()
                        if bbox[0] <= content_bbox[0]
                        and bbox[1] <= content_bbox[1]
                        and bbox[2] >= content_bbox[2]
                        and bbox[3] >= content_bbox[3]
                    ]
                    if len(containing) == 1:
                        text = containing[0]
        segments.append(replace(segment, text=text))
    return replace(line, segments=tuple(segments))


def _compile_native_route_map(
    ppvl_blocks: list[dict],
    routing_plan: PageRoutingPlan,
    page: Page,
) -> dict[int, tuple[RoutingLine, ...]]:
    """Map immutable page routes to transient native row indexes.

    ``PageRoutingPlan`` is the only production CharOCR dispatch input.  Native
    rows still carry PP-VL metadata required by Hanwang, but are never mutated
    with route records or treated as a second routing authority.
    """
    if not routing_plan.is_dispatchable:
        raise RuntimeError("Hanwang native runner received a non-dispatchable page routing plan")
    rows_by_uid: dict[str, tuple[int, dict]] = {}
    for index, row in enumerate(ppvl_blocks):
        uid = str(row.get(ROUTE_ROW_LAYOUT_BLOCK_UID_KEY) or "")
        if uid:
            rows_by_uid[uid] = (index, row)
    formula_texts_by_bbox = _formula_texts_by_layout_bbox(page)
    routes_by_block_index: dict[int, tuple[RoutingLine, ...]] = {}
    for block_route in routing_plan.blocks:
        entry = rows_by_uid.get(block_route.block_uid)
        if entry is None:
            raise RuntimeError(
                "CharOCR routing plan references a layout block missing from native input: "
                f"{block_route.block_uid}"
            )
        block_index, _row = entry
        routes_by_block_index[block_index] = tuple(
            _routing_line_for_native_runner(line, formula_texts_by_bbox)
            for line in block_route.plan.lines
        )
    missing_text_rows = [
        index
        for index, row in enumerate(ppvl_blocks)
        if _row_dispatches_to_text_ocr(row)
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
    ppvl_blocks: list[dict],
    *,
    seg_timeout: float = 120.0,
    recog_timeout: float = 60.0,
    include_chars: bool = True,
    routing_plan: PageRoutingPlan,
    page: Page,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[BlockResult], RunStats]:
    """Run native CharOCR from one explicit page routing plan."""
    global _BATCH_DISABLED_FOR_SESSION, _BATCH_DISABLE_REASON
    height, width = image_bgr.shape[:2]
    if routing_plan.page_uid != page.uid:
        raise RuntimeError("Hanwang native runner requires the current page's explicit routing plan")
    native_routes_by_block_index = _compile_native_route_map(ppvl_blocks, routing_plan, page)
    stats = RunStats(n_blocks_total=len(ppvl_blocks))
    text_indices: list[int] = []
    skip_indices: list[int] = []
    for idx, block in enumerate(ppvl_blocks):
        label = _effective_label_for_block(block)
        if _is_unknown_hanwang_label(label):
            stats.n_unknown_paddle_labels += 1
        if _row_dispatches_to_text_ocr(block):
            text_indices.append(idx)
        else:
            skip_indices.append(idx)

    stats.n_blocks_hanwang = len(text_indices)
    stats.n_blocks_ppvl = len(skip_indices)
    rows: list[BlockResult | None] = [None] * len(ppvl_blocks)
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
        synthesize_chars = map_paddle_label_to_block_type(label) != BlockType.EQUATION
        rows[idx] = BlockResult(
            block_idx=idx,
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
        grouped_lines: dict[tuple[int, int, int], list[LineResult]] = {
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
            )
            _normalize_digitlike_numeric_context_lines(lines)
            hw_text = "".join(line.text for line in lines).strip()
            source = "hanwang"
            text = hw_text
            rows[block_idx] = BlockResult(
                block_idx=block_idx,
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
        rows[block_idx] = BlockResult(
            block_idx=block_idx,
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
    return final_rows, stats


def _bbox_from_xyxy_tuple(raw: tuple[int, int, int, int], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = _clamp_xyxy(raw, width, height)
    return BBox.from_xyxy(x1, y1, x2, y2)


def _char_to_model(char: CharResult) -> Char:
    bbox = BBox.from_xyxy(*char.bbox) if char.bbox else None
    return Char(
        char=char.text,
        confidence=char.confidence,
        bbox=bbox,
        bbox_source=char.source,
        bbox_granularity=char.bbox_granularity or ("char" if bbox is not None else "fallback"),
        token_text=char.token_text or char.text,
    )


def _line_to_model(line: LineResult, width: int, height: int, review_flags: list[str]) -> Line:
    chars = [_char_to_model(char) for char in line.chars]
    merged_review_flags = [*review_flags, *line.review_flags]
    model = create_ocr_text_line(
        text=line.text,
        confidence=line.confidence,
        bbox=_bbox_from_xyxy_tuple(line.bbox, width, height),
        source_text=line.text,
        review_flags=merged_review_flags,
    )
    replace_line_ocr_char_observations(model.uid, chars)
    set_proof_state_for_line(
        model,
        ProofLineState(
            line_uid=model.uid,
            proof_status=proof_status_for(line.confidence, merged_review_flags),
        ),
    )
    return model


_PARENT_BINDING_STATUSES = {
    BINDING_GEOMETRY_HIT,
    BINDING_FORMULA_CROP_OCR,
    BINDING_PARENT_FORMULA_INFERRED,
    BINDING_PARENT_TABLE_HIT,
    BINDING_PARENT_FIGURE_HIT,
}

_INTERNAL_LAYOUT_NOTES = {
    "manual_draw_merge_requires_ocr_rerun",
    "manual_merge_requires_ocr_rerun",
    "manual_binding_ambiguous",
    "manual_geometry_empty_formula_review",
}


def _payload_bbox_xyxy(raw_payload: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = bbox_from_variant(raw_payload, max_w=width, max_h=height)
    if bbox is None:
        return None
    return _clamp_xyxy(bbox.to_xyxy(), width, height)


def _int_value(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parent_index_for_raw_payload(page: Page, raw_payload: dict[str, Any]) -> int:
    records = raw_layout_records(page)
    if not raw_payload or not records:
        return -1

    label = route_authority_label(raw_payload)
    bbox = _payload_bbox_xyxy(raw_payload, page.width, page.height)
    text = paddle_block_text(raw_payload)
    for index, record in enumerate(records):
        if label and label != route_authority_label(record):
            continue
        if bbox is not None and bbox != block_bbox_xyxy(record, page.width, page.height):
            continue
        record_text = paddle_block_text(record)
        if text and record_text and text != record_text:
            continue
        return index
    return -1


def _origin_raw_index(block: Block) -> int:
    origin = getattr(block, "origin", None)
    return _int_value(getattr(origin, "raw_index", None))


def _route_source_label_from_view(block: Block, view: LayoutBlockView | None) -> str:
    if view is None:
        return route_source_label(block)
    return str(view.source_label or "") or route_source_label(block) or view.block_type.value


def _layout_row_from_block(
    page: Page,
    block: Block,
    *,
    view: LayoutBlockView | None = None,
) -> dict[str, Any]:
    raw_payload = raw_block_payload(block, page)
    raw_payload.pop(LAYOUT_LINE_ROUTES_FIELD, None)
    parent_index = _origin_raw_index(block)
    if parent_index < 0:
        parent_index = _parent_index_for_raw_payload(page, raw_payload)
    binding = paddle_binding_dict(block)
    if parent_index < 0 and isinstance(binding, dict):
        parent_index = _int_value(binding.get("parent_index"))
    layout_bbox = view.bbox if view is not None else block.bbox
    source_label = _route_source_label_from_view(block, view)
    ocr_policy = view.ocr_policy if view is not None else block.ocr_policy
    note = view.note if view is not None else block.note
    row = {
        **raw_payload,
        "block_label": source_label,
        "block_bbox": list(layout_bbox.to_xyxy()),
        "block_content": _layout_block_content(block, raw_payload, note=note),
        "source_label": source_label,
        ROUTE_ROW_LAYOUT_BLOCK_UID_KEY: view.uid if view is not None else block.uid,
        "_layout_block_source": block_source_value(block),
        ROUTE_ROW_OCR_POLICY_KEY: ocr_policy.value,
    }
    if binding:
        row[ROUTE_ROW_PADDLE_BINDING_KEY] = dict(binding)
    records = layout_records_with_route_attachments(page)
    if ROUTE_SUBBLOCKS_FIELD not in row and 0 <= parent_index < len(records):
        parent_record = records[parent_index]
        if isinstance(parent_record, dict) and ROUTE_SUBBLOCKS_FIELD in parent_record:
            row[ROUTE_SUBBLOCKS_FIELD] = parent_record[ROUTE_SUBBLOCKS_FIELD]
    current_subblocks = row.get(ROUTE_SUBBLOCKS_FIELD)
    if isinstance(current_subblocks, list):
        row[ROUTE_SUBBLOCKS_FIELD] = filter_handled_inline_formula_subblocks(page, current_subblocks)
    if parent_index >= 0:
        row["_layout_paddle_parent_index"] = parent_index
    return row


def _layout_block_uid_from_route_row(row: BlockResult) -> str:
    uid = str(dict(row.raw_block or {}).get(ROUTE_ROW_LAYOUT_BLOCK_UID_KEY) or "")
    if not uid:
        raise RuntimeError(
            "Hanwang route row missing layout block uid; OCR cannot write layout observations safely"
        )
    return uid


def _ocr_audit_from_route_row(row: BlockResult) -> dict[str, Any]:
    raw = dict(row.raw_block or {})
    value = raw.get(ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY)
    return dict(value) if isinstance(value, dict) else {}


def _apply_row_ocr_audit(block: Block, *, ocr_audit: dict[str, Any]) -> None:
    block.ocr_audit = dict(ocr_audit)


def _layout_block_content(
    block: Block,
    raw_payload: dict[str, Any] | None = None,
    *,
    note: str | None = None,
) -> str:
    raw_text = paddle_block_text(raw_payload or {})
    if raw_text:
        return raw_text
    text = proof_block_text(block)
    if text:
        return text
    note_text = str(block.note if note is None else note or "")
    if note_text in _INTERNAL_LAYOUT_NOTES:
        return ""
    return note_text


def _binding_payload_from_block(block: Block) -> dict[str, Any] | None:
    binding = paddle_binding_dict(block)
    if not binding:
        return None
    status = str(binding.get("status") or "")
    if status in {BINDING_EMPTY_REVIEW, BINDING_AMBIGUOUS}:
        return None
    if status not in _PARENT_BINDING_STATUSES:
        return None
    if _int_value(binding.get("parent_index")) < 0:
        return None
    return binding


def _row_matches_paddle_parent_record(page: Page, row: dict[str, Any], parent_index: int) -> bool:
    records = raw_layout_records(page)
    if not (0 <= parent_index < len(records)):
        return False
    parent_record = records[parent_index]
    if not isinstance(parent_record, dict):
        return False
    row_bbox = block_bbox_xyxy(row, page.width, page.height)
    parent_bbox = block_bbox_xyxy(parent_record, page.width, page.height)
    if row_bbox != parent_bbox:
        return False
    row_label = route_authority_label(row)
    parent_label = route_authority_label(parent_record)
    return not row_label or not parent_label or row_label == parent_label


def _manual_bbox_from_entry(entry: _LayoutOcrEntry) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in entry.view.bbox.to_xyxy())


def _route_subblock_overlaps_bbox(
    value: dict[str, Any],
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    current = block_bbox_xyxy(value, width, height)
    if current == bbox:
        return True
    overlap = _intersect_xyxy(current, bbox)
    if overlap is None:
        return False
    area = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])
    current_area = max(1, (current[2] - current[0]) * (current[3] - current[1]))
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return area / min(current_area, bbox_area) >= 0.7


def _manual_binding_route_subblock(entry: _LayoutOcrEntry, binding: dict[str, Any]) -> dict[str, Any]:
    block = entry.block
    manual_bbox = _manual_bbox_from_entry(entry)
    label = str(binding.get("source_label") or _route_source_label_from_view(block, entry.view))
    text_is_stale = _manual_binding_text_is_stale(manual_bbox, binding)
    text = "" if text_is_stale else str(binding.get("text") or proof_block_text(block) or "")
    payload = {
        "block_label": label,
        "block_bbox": list(manual_bbox),
        "block_content": text,
        ROUTE_ROW_PADDLE_BINDING_KEY: dict(binding),
        "_layout_block_source": block_source_value(block),
        "_layout_manual_route_subblock": True,
    }
    if text_is_stale:
        payload["_layout_manual_binding_text_stale"] = True
    return payload


def _manual_binding_text_is_stale(
    manual_bbox: tuple[int, int, int, int],
    binding: dict[str, Any],
) -> bool:
    candidate = bbox_from_variant(binding.get("candidate_bbox"))
    if candidate is None:
        return False
    candidate_bbox = tuple(int(value) for value in candidate.to_xyxy())
    return candidate_bbox != manual_bbox


def _manual_unbound_route_subblock(entry: _LayoutOcrEntry) -> dict[str, Any]:
    block = entry.block
    manual_bbox = _manual_bbox_from_entry(entry)
    label = _route_source_label_from_view(block, entry.view)
    if entry.view.block_type == BlockType.EQUATION and normalize_paddle_label(label) in {"", "equation", "formula"}:
        label = "inline_formula"
    return {
        "block_label": label,
        "block_bbox": list(manual_bbox),
        "block_content": proof_block_text(block),
        "_layout_block_source": block_source_value(block),
        "_layout_manual_route_subblock": True,
        "_layout_manual_unbound_route_subblock": True,
    }


def _replace_or_append_route_subblock(
    row: dict[str, Any],
    route_subblock: dict[str, Any],
    binding: dict[str, Any],
    width: int,
    height: int,
) -> None:
    current_values = row.get(ROUTE_SUBBLOCKS_FIELD)
    values = [dict(value) for value in current_values if isinstance(value, dict)] if isinstance(current_values, list) else []
    candidate_bbox = bbox_from_variant(binding.get("candidate_bbox"))
    target_bbox = tuple(int(value) for value in candidate_bbox.to_xyxy()) if candidate_bbox is not None else None
    manual_bbox = tuple(route_subblock["block_bbox"])

    replaced = False
    next_values: list[dict[str, Any]] = []
    for value in values:
        current_bbox = block_bbox_xyxy(value, width, height)
        value_is_manual = bool(value.get("_layout_manual_route_subblock"))
        if current_bbox == manual_bbox:
            if not replaced:
                next_values.append(route_subblock)
                replaced = True
            continue
        if value_is_manual:
            next_values.append(value)
            continue
        should_replace = (
            target_bbox is not None
            and _route_subblock_overlaps_bbox(value, target_bbox, width, height)
        ) or _route_subblock_overlaps_bbox(value, manual_bbox, width, height)
        if should_replace:
            if not replaced:
                next_values.append(route_subblock)
                replaced = True
            continue
        next_values.append(value)
    if not replaced:
        next_values.append(route_subblock)
    row[ROUTE_SUBBLOCKS_FIELD] = next_values
    row.pop(LAYOUT_LINE_ROUTES_FIELD, None)


def _append_manual_route_subblock(
    row: dict[str, Any],
    route_subblock: dict[str, Any],
    width: int,
    height: int,
) -> None:
    current_values = row.get(ROUTE_SUBBLOCKS_FIELD)
    values = [dict(value) for value in current_values if isinstance(value, dict)] if isinstance(current_values, list) else []
    manual_bbox = tuple(route_subblock["block_bbox"])
    next_values: list[dict[str, Any]] = []
    replaced = False
    for value in values:
        current_bbox = block_bbox_xyxy(value, width, height)
        if current_bbox == manual_bbox:
            next_values.append(route_subblock)
            replaced = True
        else:
            next_values.append(value)
    if not replaced:
        next_values.append(route_subblock)
    row[ROUTE_SUBBLOCKS_FIELD] = next_values
    row.pop(LAYOUT_LINE_ROUTES_FIELD, None)


def _manual_structure_parent_row(
    entries: list[_LayoutOcrEntry],
    entry: _LayoutOcrEntry,
    page: Page,
) -> dict[str, Any] | None:
    block = entry.block
    if entry.view.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
        return None
    if not is_user_authored_layout_block(block):
        return None
    block_bbox = tuple(int(value) for value in entry.view.bbox.to_xyxy())
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in entries:
        candidate_row = candidate.row
        if candidate is entry:
            continue
        label = route_authority_label(candidate_row)
        if map_paddle_label_to_block_type(label) != BlockType.TEXT:
            continue
        if is_hanwang_skip_label(label) or is_formula_label(label) or is_table_label(label):
            continue
        candidate_bbox = block_bbox_xyxy(candidate_row, page.width, page.height)
        overlap_area = _intersection_area(block_bbox, candidate_bbox)
        if overlap_area <= 0:
            continue
        block_area = max(1, (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1]))
        score = overlap_area / block_area
        center_inside = (
            candidate_bbox[0] <= (block_bbox[0] + block_bbox[2]) / 2 <= candidate_bbox[2]
            and candidate_bbox[1] <= (block_bbox[1] + block_bbox[3]) / 2 <= candidate_bbox[3]
        )
        score += 0.25 if center_inside else 0.0
        if best is None or score > best[0]:
            best = (score, candidate_row)
    if best is None or best[0] < 0.25:
        return None
    return best[1]


def _apply_manual_parent_binding(
    *,
    page: Page,
    parent_row: dict[str, Any],
    entry: _LayoutOcrEntry,
    binding: dict[str, Any],
) -> None:
    block = entry.block
    block_type = map_paddle_label_to_block_type(str(binding.get("block_type") or entry.view.block_type.value))
    parent_index = _int_value(binding.get("parent_index"))
    records = raw_layout_records(page)
    parent_record = records[parent_index] if 0 <= parent_index < len(records) else {}
    if block_type == BlockType.EQUATION:
        parent_text = paddle_block_text(parent_record)
        if parent_text:
            parent_row["block_content"] = parent_text
            parent_row["_layout_block_content_authority"] = "paddle_parent_binding"
        route_subblock = _manual_binding_route_subblock(entry, binding)
        _replace_or_append_route_subblock(
            parent_row,
            route_subblock,
            binding,
            page.width,
            page.height,
        )
        return

    manual_bbox = list(_manual_bbox_from_entry(entry))
    parent_row["block_bbox"] = manual_bbox
    if binding.get("text"):
        parent_row["block_content"] = str(binding.get("text") or "")
    parent_row["_layout_parent_replaced_by_manual_binding"] = True


def _apply_manual_unbound_parent_route(
    *,
    page: Page,
    parent_row: dict[str, Any],
    entry: _LayoutOcrEntry,
) -> None:
    route_subblock = _manual_unbound_route_subblock(entry)
    _append_manual_route_subblock(parent_row, route_subblock, page.width, page.height)


def _compile_layout_ocr_input_plan(page: Page) -> _LayoutOcrInputPlan:
    entries: list[_LayoutOcrEntry] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        entries.append(
            _LayoutOcrEntry(
                view=view,
                block=block,
                row=_layout_row_from_block(page, block, view=view),
            )
        )
    parent_rows: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row = entry.row
        parent_index = _int_value(row.get("_layout_paddle_parent_index"))
        if (
            parent_index >= 0
            and parent_index not in parent_rows
            and _row_matches_paddle_parent_record(page, row, parent_index)
        ):
            parent_rows[parent_index] = row

    skip_block_ids: set[int] = set()
    for entry in entries:
        block = entry.block
        row = entry.row
        if entry.view.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
            continue
        binding = _binding_payload_from_block(block)
        parent_row = parent_rows.get(_int_value(binding.get("parent_index"))) if binding is not None else None
        if parent_row is None:
            parent_row = _manual_structure_parent_row(entries, entry, page)
        if parent_row is None or parent_row is row:
            continue
        if binding is not None:
            _apply_manual_parent_binding(
                page=page,
                parent_row=parent_row,
                entry=entry,
                binding=binding,
            )
        else:
            _apply_manual_unbound_parent_route(
                page=page,
                parent_row=parent_row,
                entry=entry,
            )
        skip_block_ids.add(id(block))

    blocks: list[dict] = []
    for entry in entries:
        block = entry.block
        if id(block) in skip_block_ids:
            continue
        blocks.append(entry.row)
    return _LayoutOcrInputPlan(
        rows=tuple(blocks),
    )


def _page_blocks_from_layout(page: Page) -> list[dict]:
    return [dict(row) for row in _compile_layout_ocr_input_plan(page).rows]


def _current_layout_blocks_for_ocr(page: Page) -> list[dict]:
    """Build OCR input from the current layout truth only.

    Paddle parsing records remain available as origin/reference data for binding,
    but OCR dispatch must not switch data sources based on whether a user edited
    the page.
    """
    return _page_blocks_from_layout(page)


def _inline_formula_crop_ocr_targets(page: Page) -> list[Block]:
    targets: list[tuple[LayoutBlockView, Block]] = []
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        if view.block_type != BlockType.EQUATION:
            continue
        label = _route_source_label_from_view(block, view)
        if label not in {"inline_formula", "formula"}:
            continue
        if view.bbox is None or view.bbox.area <= 0:
            continue
        targets.append((view, block))
    targets.sort(key=lambda item: (item[0].bbox.y, item[0].bbox.x, item[0].order))
    return [block for _view, block in targets]


def _existing_parent_index(block: Block) -> int:
    origin_index = _origin_raw_index(block)
    if origin_index >= 0:
        return origin_index
    binding = paddle_binding_dict(block)
    if binding:
        parent_index = _int_value(binding.get("parent_index"))
        if parent_index >= 0:
            return parent_index
    return -1


def _set_inline_formula_crop_ocr_text(block: Block, text: str) -> None:
    bbox_xyxy = [int(value) for value in block.bbox.to_xyxy()]
    parent_index = _existing_parent_index(block)
    binding: dict[str, Any] = {
        "status": BINDING_FORMULA_CROP_OCR,
        "source": "paddle_formula_crop_ocr",
        "block_type": BlockType.EQUATION.value,
        "source_label": "inline_formula",
        "text": text,
        "parent_index": parent_index,
        "candidate_index": -1,
        "score": 1.0,
        "manual_bbox": bbox_xyxy,
        "review_flags": [FORMULA_CROP_OCR_REVIEW_FLAG],
    }
    set_paddle_binding(block, binding)
    clear_ocr_text_invalidation(block)
    replace_block_ocr_line_observations(block.uid, [
        create_ocr_text_line(
            text=text,
            confidence=1.0,
            bbox=block.bbox,
            source_text=text,
            review_flags=[FORMULA_CROP_OCR_REVIEW_FLAG],
        )
    ])


def _mark_inline_formula_needs_text(block: Block, reason: str = "") -> None:
    bbox_xyxy = [int(value) for value in block.bbox.to_xyxy()]
    parent_index = _existing_parent_index(block)
    flags = ["manual_formula_needs_text"]
    if reason:
        flags.append(FORMULA_CROP_OCR_FAILED_FLAG)
    set_paddle_binding(block, {
            "status": BINDING_EMPTY_REVIEW,
            "source": "paddle_formula_crop_ocr_empty",
            "block_type": BlockType.EQUATION.value,
            "source_label": "inline_formula",
            "text": "",
            "parent_index": parent_index,
            "candidate_index": -1,
            "score": 0.0,
            "manual_bbox": bbox_xyxy,
            "review_flags": flags,
    })
    replace_block_ocr_line_observations(block.uid, [
        create_ocr_text_line(
            text="",
            confidence=0.0,
            bbox=block.bbox,
            source_text="",
            review_flags=flags,
        )
    ])


def _mark_formula_crop_ocr_unavailable(blocks: list[Block]) -> None:
    for block in blocks:
        binding = paddle_binding_dict(block)
        binding_source = str(binding.get("source") or "") if isinstance(binding, dict) else ""
        invalidated = is_ocr_text_invalidated(block)
        stale_parent_binding = "parent_text" in binding_source or binding_source == "paddle_geometry"
        if invalidated or stale_parent_binding or not proof_block_text(block):
            _mark_inline_formula_needs_text(block)


def _mark_formula_crop_ocr_failed(blocks: list[Block], reason: str) -> None:
    for block in blocks:
        _mark_inline_formula_needs_text(block, reason)


def _build_formula_rebind_client(progress_callback: Callable[[int, int, str], None] | None) -> Any | None:
    try:
        from app.core.app_config import get_config
        from app.core.api_profiles import FIXED_LAYOUT_PROFILE, resolve_api_endpoint_for_role
        from app.core.paddle_v16_client import (
            PaddleV16LayoutClient,
            is_paddle_v16_endpoint,
        )

        cfg = get_config()
        url = resolve_api_endpoint_for_role(
            cfg.get("api_url", ""),
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        if not url or not is_paddle_v16_endpoint(url):
            return None
        configured_timeout = max(1, int(cfg.get("api_timeout", 180)))
        return PaddleV16LayoutClient(
            jobs_url=url,
            token=str(cfg.get("api_token", "") or ""),
            request_timeout=min(max(10, configured_timeout), 30),
            poll_timeout=max(configured_timeout, 180),
            network_mode=str(cfg.get("paddle_api_network_mode", "auto") or "auto"),
            status_callback=(
                (lambda message: progress_callback(0, 1, f"Paddle 公式重识别：{message}"))
                if progress_callback
                else None
            ),
        )
    except Exception as exc:
        logger.warning("Cannot create Paddle formula crop OCR client: %s", exc)
        return None


class HanwangMicroRecBlockEngine:
    """OcrPipeline page-level engine for PP-VL layout + Hanwang text OCR."""

    engine_id = "hanwang.micro_recblock"
    prefer_page_hybrid_blocks = True
    bbox_space = OCR_BBOX_SPACE_PAGE

    def __init__(
        self,
        *,
        seg_timeout: float = 120.0,
        recog_timeout: float = 60.0,
        formula_rebind_client: Any | None = None,
        runner=run_micro_recblock,
    ) -> None:
        self._seg_timeout = seg_timeout
        self._recog_timeout = recog_timeout
        self._formula_rebind_client = formula_rebind_client
        self._runner = runner

    def recognize_page_blocks(
        self,
        image_bgr: np.ndarray,
        page: Page,
        progress_callback: Callable[[int, int, str], None] | None = None,
        routing_plan: object | None = None,
    ) -> RunStats:
        if not isinstance(routing_plan, PageRoutingPlan):
            raise RuntimeError(
                "Hanwang micro_recblock requires an explicit PageRoutingPlan; "
                "legacy page-line routing is not a production OCR path"
            )
        self._refresh_inline_formula_texts_from_current_crops(
            image_bgr,
            page,
            progress_callback=progress_callback,
        )
        input_plan = _compile_layout_ocr_input_plan(page)
        ppvl_blocks = deepcopy(list(input_plan.rows))
        if not ppvl_blocks:
            raise RuntimeError("Hanwang micro_recblock requires PP-VL parsing_res_list blocks")
        expected_uids = {
            str(row.get(ROUTE_ROW_LAYOUT_BLOCK_UID_KEY) or "")
            for row in ppvl_blocks
        }
        expected_uids.discard("")

        runner_kwargs: dict[str, Any] = {
            "seg_timeout": self._seg_timeout,
            "recog_timeout": self._recog_timeout,
            "include_chars": True,
            "routing_plan": routing_plan,
            "page": page,
            "progress_callback": progress_callback,
        }
        rows, stats = self._runner(image_bgr, ppvl_blocks, **runner_kwargs)

        height, width = image_bgr.shape[:2]
        runtime_blocks_by_uid = {
            view.uid: view.runtime_block
            for view in iter_page_layout_block_views(page)
            if view.runtime_block is not None
        }
        written_uids: set[str] = set()
        line_updates: list[tuple[str, list[Line]]] = []
        for row in rows:
            block_uid = _layout_block_uid_from_route_row(row)
            written_uids.add(block_uid)
            block_type = map_paddle_label_to_block_type(row.block_label)
            flags: list[str] = []
            if (
                block_type == BlockType.EQUATION
                and not row.text
                and is_user_authored_layout_source(row.raw_block.get("_layout_block_source"))
            ):
                flags.append("manual_formula_needs_text")
            lines = [_line_to_model(line, width, height, flags) for line in row.lines if line.text]
            if not lines and block_type == BlockType.EQUATION and "manual_formula_needs_text" in flags:
                lines = [
                    _line_to_model(
                        LineResult(
                            text="",
                            bbox=row.block_bbox,
                            confidence=0.0,
                            source="manual_formula_placeholder",
                            review_flags=["manual_formula_needs_text"],
                        ),
                        width,
                        height,
                        flags,
                    )
                ]
            line_updates.append((block_uid, lines))
            ocr_audit = _ocr_audit_from_route_row(row)
            block = runtime_blocks_by_uid.get(block_uid)
            if block is not None and ocr_audit:
                _apply_row_ocr_audit(block, ocr_audit=ocr_audit)
        missing_uids = sorted(expected_uids - written_uids)
        if missing_uids:
            raise RuntimeError(
                "Hanwang OCR did not return rows for layout blocks: "
                + ", ".join(missing_uids[:5])
            )
        for block_uid, lines in line_updates:
            replace_block_ocr_line_observations(block_uid, lines)
        logger.info(
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d fallback=%d "
            "unknown_labels=%d groups=%d group_failures=%d chunks=%d guarded_chunks=%d "
            "batch_failures=%d batch_disabled=%s "
            "seg=%.2fs recog=%.2fs "
            "max_batch_crop=%dx%d probe_calls=%d recog_pixels=%d/%d "
            "engcut_route_calls=%d "
            "geometry_conflicts=%d geometry_token_atoms=%d",
            page.page_number,
            stats.n_blocks_total,
            stats.n_blocks_hanwang,
            stats.n_blocks_ppvl,
            stats.n_blocks_fallback,
            stats.n_unknown_paddle_labels,
            stats.n_groups,
            stats.recog_group_failures,
            stats.recog_batch_chunks,
            stats.recog_batch_guarded_chunks,
            stats.recog_batch_failures,
            stats.recog_batch_disabled,
            stats.seg_seconds,
            stats.recog_seconds,
            stats.recog_max_batch_crop_width,
            stats.recog_max_batch_crop_height,
            stats.recog_probe_calls,
            stats.recog_crop_pixels,
            stats.recog_full_page_pixels,
            stats.engcut_route_calls,
            stats.geometry_conflict_groups,
            stats.geometry_token_atoms,
        )
        return stats

    def _refresh_inline_formula_texts_from_current_crops(
        self,
        image_bgr: np.ndarray,
        page: Page,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        targets = _inline_formula_crop_ocr_targets(page)
        if not targets:
            return
        client = self._formula_rebind_client or _build_formula_rebind_client(progress_callback)
        if client is None:
            _mark_formula_crop_ocr_unavailable(targets)
            if progress_callback:
                progress_callback(0, max(1, len(targets)), "Paddle 公式重识别未配置，公式框保留待确认")
            return

        if progress_callback:
            progress_callback(0, max(1, len(targets)), f"Paddle 公式重识别中… {len(targets)} 个框")
        try:
            bboxes = [tuple(int(value) for value in block.bbox.to_xyxy()) for block in targets]
            outcome = recognize_formula_bboxes_with_retry(
                image_bgr,
                bboxes,
                client=client,
                batch_id_prefix=f"ocr-process-formula-{page.page_number}-{int(time.time() * 1000)}",
                filename_prefix=f"page-{page.page_number}-inline-formula-pseudo",
            )
        except Exception as exc:
            logger.warning("Paddle formula crop OCR failed for page=%s: %s", page.page_number, exc)
            _mark_formula_crop_ocr_failed(targets, str(exc))
            if progress_callback:
                progress_callback(0, max(1, len(targets)), "公式文本为空，需人工补全")
            return

        if outcome.error:
            logger.warning(
                "Paddle formula crop OCR left empty formulas page=%s failed=%s error=%s",
                page.page_number,
                list(outcome.failed_indices),
                outcome.error,
            )
        text_by_index = outcome.texts_by_index
        for index, block in enumerate(targets):
            text = str(text_by_index.get(index) or "").strip()
            if text:
                _set_inline_formula_crop_ocr_text(block, text)
            else:
                reason = outcome.error if index in outcome.failed_indices else ""
                _mark_inline_formula_needs_text(block, reason)
        if progress_callback:
            progress_callback(
                len(text_by_index),
                max(1, len(targets)),
                f"公式文本识别：{len(text_by_index)}/{len(targets)}",
            )


__all__ = [
    "TEXT_LABELS",
    "SKIP_LABELS",
    "CharResult",
    "LineResult",
    "BlockResult",
    "RunStats",
    "HanwangMicroRecBlockEngine",
    "decode_gbk",
    "run_micro_recblock",
]
