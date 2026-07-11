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
    RoutingLine,
    is_text_route_segment_kind,
)
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


def _replace_line_result_char_span(line: "LineResult", start: int, end: int, chars: list["CharResult"]) -> None:
    line.chars[start:end] = chars

ROUTE_ROW_PADDLE_BINDING_KEY = "paddle_binding"
ROUTE_ROW_HANWANG_BBOX_AUDIT_KEY = "_hanwang_bbox_audit"
ROUTE_ROW_LAYOUT_BLOCK_UID_KEY = "_layout_block_uid"

TEXT_LABELS: set[str] = set(PADDLE_HANWANG_TEXT_LABELS)
SKIP_LABELS: set[str] = set(PADDLE_HANWANG_SKIP_LABELS)
DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG = "hanwang_digitlike_numeric_context"
FORMULA_CROP_OCR_REVIEW_FLAG = "paddle_formula_crop_ocr"
FORMULA_CROP_OCR_FAILED_FLAG = "paddle_formula_crop_ocr_failed"
LATIN_ENGCUT_ROUTE_SOURCE = "hanwang:EngCut:latin_route"
CHINESE_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】［］〔〕—…·．")
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
    latin_engcut_route_calls: int = 0
    overlap_merge_probe_calls: int = 0
    overlap_merge_probe_failures: int = 0
    overlap_merge_clusters: int = 0
    overlap_merge_replacements: int = 0


@dataclass
class _GroupPlacement:
    area_idx: int
    page_bbox: tuple[int, int, int, int]


@dataclass
class _TextRoute:
    block_idx: int
    line_idx: int
    segment_idx: int
    bbox: tuple[int, int, int, int]
    carved: bool = False
    kind: str = "text"

    @property
    def key(self) -> tuple[int, int, int]:
        return self.block_idx, self.line_idx, self.segment_idx


@dataclass(frozen=True)
class _LatinMaskedLineRoute:
    """One physical routing line materialized as a white EngCut canvas.

    The routing plan remains segment-oriented.  This is only the native-input
    representation needed by EngCut: it keeps the PP-OCR line geometry while
    exposing pixels from Latin/digit segments and whitening every other region.
    """

    block_idx: int
    line_idx: int
    bbox: tuple[int, int, int, int]
    segments: tuple[_TextRoute, ...]


def _is_latin_text_route(route: _TextRoute) -> bool:
    return route.kind == "text_latin"


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
        )
        for line_idx, line in enumerate(lines)
        for segment_idx, segment in enumerate(line.segments)
        if is_text_route_segment_kind(segment.kind)
    ]


def _latin_masked_line_routes_from_lines(
    block_idx: int,
    lines: tuple[RoutingLine, ...],
) -> list[_LatinMaskedLineRoute]:
    """Group typed Latin routes by their physical PP-OCR line.

    This is deliberately derived from ``RoutingLine`` rather than from raw
    Paddle payloads.  The selected segments remain the only places whose
    pixels may appear in the EngCut input canvas.
    """
    grouped: list[_LatinMaskedLineRoute] = []
    for line_idx, line in enumerate(lines):
        segments = tuple(
            _TextRoute(
                block_idx=block_idx,
                line_idx=line_idx,
                segment_idx=segment_idx,
                bbox=segment.bbox,
                carved=len(line.segments) > 1,
                kind=segment.kind,
            )
            for segment_idx, segment in enumerate(line.segments)
            if segment.kind == "text_latin"
        )
        if segments:
            grouped.append(
                _LatinMaskedLineRoute(
                    block_idx=block_idx,
                    line_idx=line_idx,
                    bbox=line.bbox,
                    segments=segments,
                )
            )
    return grouped


def _merge_peer_text_lines(lines: list[LineResult]) -> list[LineResult]:
    buckets: list[list[LineResult]] = []
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        if ROUTE_TABLE_FLAG in line.review_flags:
            buckets.append([line])
            continue
        for bucket in buckets:
            head = bucket[0]
            if ROUTE_TABLE_FLAG in head.review_flags:
                continue
            if vertical_overlap_ratio(head.bbox, line.bbox) >= 0.5:
                bucket.append(line)
                break
        else:
            buckets.append([line])

    merged: list[LineResult] = []
    for bucket in buckets:
        if len(bucket) == 1:
            merged.append(bucket[0])
            continue
        bucket.sort(key=lambda item: (item.bbox[0], item.bbox[1]))
        chars: list[CharResult] = []
        for line in bucket:
            chars.extend(line.chars)
        confidence_values = [line.confidence for line in bucket if line.confidence > 0]
        flags = sorted({flag for line in bucket for flag in line.review_flags})
        merged.append(LineResult(
            text="".join(line.text for line in bucket if line.text),
            bbox=union_xyxy([line.bbox for line in bucket]),
            confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            chars=chars,
            source="hanwang+ppvl_route_merged",
            bbox_source="merged_peer_text_lines",
            review_flags=flags,
        ))
    merged.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    return merged


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


def _is_usable_text_char_box(char: CharResult) -> bool:
    if char.bbox is None or char.source == "paddle_inline_formula":
        return False
    if char.text in CHINESE_PUNCT:
        return False
    return _box_width(char.bbox) >= 8 and _box_height(char.bbox) >= 12


def _nearest_char_box(
    chars: list[CharResult],
    index: int,
    *,
    step: int,
    text_only: bool = False,
) -> tuple[int, int, int, int] | None:
    pos = index + step
    while 0 <= pos < len(chars):
        char = chars[pos]
        if char.bbox is not None and (not text_only or _is_usable_text_char_box(char)):
            return char.bbox
        pos += step
    return None


def _recover_degenerate_punctuation_bboxes(
    line: LineResult,
    char_bounds: list[tuple[int, int, int, int] | None] | None = None,
) -> LineResult:
    if not line.chars:
        return line
    recovered: list[CharResult] = []
    changed = False
    for index, char in enumerate(line.chars):
        if char.bbox is None or char.text not in CHINESE_PUNCT:
            recovered.append(char)
            continue
        width = _box_width(char.bbox)
        height = _box_height(char.bbox)
        if width >= 4 and height >= 8:
            recovered.append(char)
            continue

        prev_box = _nearest_char_box(line.chars, index, step=-1)
        next_box = _nearest_char_box(line.chars, index, step=1)
        ref_box = (
            _nearest_char_box(line.chars, index, step=1, text_only=True)
            or _nearest_char_box(line.chars, index, step=-1, text_only=True)
        )
        if ref_box is None:
            recovered.append(char)
            continue

        target_width = max(8, min(18, round(_box_width(ref_box) * 0.35)))
        left: int
        right: int
        if prev_box is not None and next_box is not None and prev_box[2] <= next_box[0]:
            gap_width = next_box[0] - prev_box[2]
            if 4 <= gap_width <= target_width * 2:
                left, right = prev_box[2], next_box[0]
            else:
                center = (char.bbox[0] + char.bbox[2]) // 2
                left = center - target_width // 2
                right = left + target_width
        elif next_box is not None:
            right = next_box[0]
            left = right - target_width
        elif prev_box is not None:
            left = prev_box[2]
            right = left + target_width
        else:
            recovered.append(char)
            continue

        bound = (
            char_bounds[index]
            if char_bounds is not None and index < len(char_bounds) and char_bounds[index] is not None
            else line.bbox
        )
        line_x1, line_y1, line_x2, line_y2 = bound
        left = max(line_x1, left)
        right = min(line_x2, right)
        if right - left < 8:
            recovered.append(
                CharResult(
                    text=char.text,
                    confidence=char.confidence,
                    bbox=None,
                    candidates=list(char.candidates),
                    source=f"{char.source}:punct_bbox_dropped_at_route_boundary",
                    bbox_granularity=char.bbox_granularity or "char",
                    token_text=char.token_text,
                )
            )
            changed = True
            continue
        if right - left < 4:
            recovered.append(char)
            continue
        top = max(line_y1, ref_box[1])
        bottom = min(line_y2, ref_box[3])
        if bottom - top < 8:
            top, bottom = line_y1, line_y2
        recovered.append(
            CharResult(
                text=char.text,
                confidence=char.confidence,
                bbox=(left, top, right, bottom),
                candidates=list(char.candidates),
                source=f"{char.source}:punct_bbox_recovered",
                bbox_granularity=char.bbox_granularity or "char",
                token_text=char.token_text,
            )
        )
        changed = True

    if not changed:
        return line
    return LineResult(
        text=line.text,
        bbox=line.bbox,
        confidence=line.confidence,
        chars=recovered,
        source=line.source,
        bbox_source=line.bbox_source,
        review_flags=list(line.review_flags),
    )


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
        return _merge_peer_text_lines(merged_text_lines)

    clusters = _cluster_lines_by_shape(all_text_lines)
    if not clusters:
        return []

    assembled: list[LineResult] = []
    for cluster in clusters:
        cluster_bbox = union_xyxy([line.bbox for line in cluster])
        text_parts: list[str] = []
        chars: list[CharResult] = []
        char_bounds: list[tuple[int, int, int, int] | None] = []
        component_boxes: list[tuple[int, int, int, int]] = []
        confidence_values: list[float] = []
        flags: set[str] = set()
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
                if not segment_lines:
                    continue
                text_parts.append("".join(line.text for line in segment_lines if line.text))
                for line in segment_lines:
                    chars.extend(line.chars)
                    char_bounds.extend([segment_bbox] * len(line.chars))
                    component_boxes.append(line.bbox)
                    flags.update(line.review_flags)
                    if line.confidence > 0:
                        confidence_values.append(line.confidence)
            elif kind == "formula":
                if len(clusters) > 1 and vertical_overlap_ratio(segment_bbox, cluster_bbox) < 0.5:
                    continue
                formula_text = segment.text
                if not formula_text:
                    continue
                content_bbox = segment.content_bbox or segment_bbox
                text_parts.append(formula_text)
                component_boxes.append(content_bbox)
                chars.append(
                    CharResult(
                        text=formula_text,
                        confidence=0.0,
                        bbox=content_bbox,
                        candidates=[formula_text],
                        source="paddle_inline_formula",
                        bbox_granularity="word",
                        token_text=formula_text,
                    )
                )
                char_bounds.append(content_bbox)
                flags.add(ROUTE_INLINE_FORMULA_FLAG)
        merged_text = "".join(text_parts)
        if not merged_text:
            continue
        assembled.append(
            _recover_degenerate_punctuation_bboxes(
                LineResult(
                    text=merged_text,
                    bbox=union_xyxy(component_boxes) if component_boxes else route.bbox,
                    confidence=(
                        sum(confidence_values) / len(confidence_values)
                        if confidence_values
                        else 0.0
                    ),
                    chars=chars,
                    source="layout_route+hanwang",
                    bbox_source="layout_route_assembled",
                    review_flags=sorted(flags),
                ),
                char_bounds=char_bounds,
            )
        )
    return assembled


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


def _char_result(raw: dict, fallback_bbox: tuple[int, int, int, int]) -> CharResult:
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
        bbox=_bbox_tuple(raw.get("bbox"), fallback_bbox),
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
                for char in (_char_result(char, line_bbox) for char in raw_chars)
                if char.text
            ]
            text = "".join(char.text for char in chars).strip()
            if not text and not chars:
                continue
            confidence = (
                sum(char.confidence for char in chars) / len(chars)
                if chars else 0.0
            )
            lines.append(
                LineResult(
                    text=text,
                    bbox=line_bbox,
                    confidence=confidence,
                    chars=chars if include_chars else [],
                    bbox_source="hanwang_recog_group",
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


def _bbox_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _axis_overlap_fraction(a1: int, a2: int, b1: int, b2: int) -> float:
    overlap = max(0, min(a2, b2) - max(a1, b1))
    smaller = min(max(0, a2 - a1), max(0, b2 - b1))
    if smaller <= 0:
        return 0.0
    return overlap / smaller


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _overlap_merge_pair(left: CharResult, right: CharResult) -> bool:
    if left.bbox is None or right.bbox is None:
        return False
    if max(left.confidence, right.confidence) > OVERLAP_MERGE_LOW_CONFIDENCE:
        return False
    left_area = _bbox_area(left.bbox)
    right_area = _bbox_area(right.bbox)
    if left_area <= 0 or right_area <= 0:
        return False
    ioa = _intersection_area(left.bbox, right.bbox) / max(1, min(left_area, right_area))
    vertical = _axis_overlap_fraction(left.bbox[1], left.bbox[3], right.bbox[1], right.bbox[3])
    if ioa < OVERLAP_MERGE_IOA_THRESHOLD or vertical < OVERLAP_MERGE_VERTICAL_THRESHOLD:
        return False
    left_cx, _ = _bbox_center(left.bbox)
    right_cx, _ = _bbox_center(right.bbox)
    max_height = max(left.bbox[3] - left.bbox[1], right.bbox[3] - right.bbox[1])
    return abs(right_cx - left_cx) <= max(8.0, max_height * 0.85)


def _looks_percent_fragment(text: str) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < 2 or len(compact) > 5:
        return False
    if "%" in compact or "％" in compact:
        return False
    zero_like = set("0Oo°")
    slash_like = set("/\\")
    percent_curve_like = set("Pp")
    has_zero = any(ch in zero_like for ch in compact)
    has_slash = any(ch in slash_like for ch in compact)
    has_curve = any(ch in percent_curve_like for ch in compact)
    if has_zero and (has_slash or has_curve):
        return True
    if has_slash and (has_zero or has_curve):
        return True
    return False


def _overlap_merge_cluster_actionable(chars: list[CharResult]) -> bool:
    if len(chars) < 2 or len(chars) > OVERLAP_MERGE_MAX_CLUSTER_CHARS:
        return False
    boxes = [char.bbox for char in chars if char.bbox is not None]
    if len(boxes) != len(chars):
        return False
    text = "".join(char.text for char in chars)
    if _looks_percent_fragment(text):
        return True
    union = _union_bbox(boxes)
    height = max(1, union[3] - union[1])
    width = max(1, union[2] - union[0])
    low_conf = sum(1 for char in chars if char.confidence <= OVERLAP_MERGE_LOW_CONFIDENCE)
    return low_conf == len(chars) and width <= height * 1.35


def _find_overlap_merge_clusters(line: LineResult) -> list[tuple[int, int]]:
    clusters: list[tuple[int, int]] = []
    chars = line.chars
    idx = 0
    while idx < len(chars) - 1:
        if not _overlap_merge_pair(chars[idx], chars[idx + 1]):
            idx += 1
            continue
        start = idx
        end = idx + 2
        while (
            end < len(chars)
            and end - start < OVERLAP_MERGE_MAX_CLUSTER_CHARS
            and _overlap_merge_pair(chars[end - 1], chars[end])
        ):
            end += 1
        if _overlap_merge_cluster_actionable(chars[start:end]):
            clusters.append((start, end))
        idx = end
    return clusters


def _offset_char_result(char: CharResult, *, dx: int, dy: int, source: str) -> CharResult:
    bbox = None
    if char.bbox is not None:
        x1, y1, x2, y2 = char.bbox
        bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    return CharResult(
        text=char.text,
        confidence=char.confidence,
        bbox=bbox,
        candidates=list(char.candidates),
        source=source,
        bbox_granularity=char.bbox_granularity or ("char" if bbox is not None else "fallback"),
        token_text=char.token_text or char.text,
    )


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


def _replacement_chars_from_recrop(
    crop_bgr: np.ndarray,
    cluster_bbox: tuple[int, int, int, int],
    *,
    stats: RunStats,
    timeout: float,
) -> list[CharResult]:
    x1, y1, x2, y2 = cluster_bbox
    retry_crop = crop_bgr[y1:y2, x1:x2].copy()
    if retry_crop.size == 0:
        return []
    try:
        stats.overlap_merge_probe_calls += 1
        raw = native_bridge.run_linecut_recog(
            retry_crop,
            recblock_xyxy=None,
            with_charrcg=True,
            timeout=timeout,
        )
    except Exception as exc:
        stats.overlap_merge_probe_failures += 1
        logger.debug("Hanwang overlap-merge recrop failed bbox=%s: %s", cluster_bbox, exc)
        return []
    local_lines = _line_results_from_recog(
        raw,
        fallback_bbox=(0, 0, max(0, x2 - x1), max(0, y2 - y1)),
        include_chars=True,
        fallback_empty=False,
    )
    replacement: list[CharResult] = []
    for local_line in local_lines:
        for char in local_line.chars:
            if char.text:
                replacement.append(_offset_char_result(char, dx=x1, dy=y1, source="hanwang:overlap_merge_recrop"))
    return replacement


def _cluster_replacement_accepted(
    old_chars: list[CharResult],
    replacement: list[CharResult],
) -> bool:
    if not replacement:
        return False
    old_text = _line_chars_text(old_chars)
    new_text = _line_chars_text(replacement)
    if not new_text or new_text == old_text:
        return False
    if new_text in {"%", "％"}:
        return _looks_percent_fragment(old_text)
    old_conf = sum(char.confidence for char in old_chars) / max(1, len(old_chars))
    new_conf = sum(char.confidence for char in replacement) / max(1, len(replacement))
    return len(replacement) < len(old_chars) and new_conf >= old_conf


def _refine_overlap_fragments_with_recrop(
    crop_bgr: np.ndarray,
    lines: list[LineResult],
    stats: RunStats,
    *,
    timeout: float,
) -> None:
    if crop_bgr.size == 0:
        return
    crop_h, crop_w = crop_bgr.shape[:2]
    for line in lines:
        if not line.chars:
            continue
        clusters = _find_overlap_merge_clusters(line)
        if not clusters:
            continue
        for start, end in reversed(clusters):
            old_chars = line.chars[start:end]
            boxes = [char.bbox for char in old_chars if char.bbox is not None]
            if len(boxes) != len(old_chars):
                continue
            union = _union_bbox(boxes)
            cluster_bbox = _expand_xyxy(
                union,
                crop_w,
                crop_h,
                pad_x=OVERLAP_MERGE_PAD_X,
                pad_y=OVERLAP_MERGE_PAD_Y,
            )
            stats.overlap_merge_clusters += 1
            replacement = _replacement_chars_from_recrop(
                crop_bgr,
                cluster_bbox,
                stats=stats,
                timeout=timeout,
            )
            if _cluster_replacement_accepted(old_chars, replacement):
                _replace_line_result_char_span(line, start, end, replacement)
                stats.overlap_merge_replacements += 1
            else:
                continue
            line.text = "".join(char.text for char in line.chars).strip()
            if line.chars:
                line.confidence = sum(char.confidence for char in line.chars) / len(line.chars)


RECOG_GROUP_CROP_PAD_X = 8
RECOG_GROUP_CROP_PAD_Y = 10
RECOG_GROUP_RETRY_TOP_TRIM = 3
ENGCUT_LINE_CONTEXT_PAD_X = 2
ENGCUT_LINE_CONTEXT_PAD_Y = 2
OVERLAP_MERGE_LOW_CONFIDENCE = 0.35
OVERLAP_MERGE_IOA_THRESHOLD = 0.45
OVERLAP_MERGE_VERTICAL_THRESHOLD = 0.55
OVERLAP_MERGE_MAX_CLUSTER_CHARS = 5
OVERLAP_MERGE_PAD_X = 3
OVERLAP_MERGE_PAD_Y = 3


def _engcut_route_line_text_and_chars(
    chars: list[EngcutChar],
) -> tuple[str, list[CharResult]]:
    text_parts: list[str] = []
    results: list[CharResult] = []
    previous_group: tuple[int, int] | None = None
    for char in chars:
        text = str(char.text or "")
        if not text:
            continue
        group = (char.line_index, char.group_index)
        if previous_group is not None and group != previous_group:
            text_parts.append(" ")
            results.append(
                CharResult(
                    text=" ",
                    confidence=0.0,
                    bbox=None,
                    candidates=[" "],
                    source=LATIN_ENGCUT_ROUTE_SOURCE,
                    bbox_granularity="space",
                    token_text=" ",
                )
            )
        text_parts.append(text)
        results.append(
            CharResult(
                text=text,
                confidence=0.0,
                bbox=char.bbox,
                candidates=[text],
                source=LATIN_ENGCUT_ROUTE_SOURCE,
                bbox_granularity="char",
                token_text=text,
            )
        )
        previous_group = group
    return "".join(text_parts), results


def _materialize_latin_masked_line_crop(
    image_bgr: np.ndarray,
    route: _LatinMaskedLineRoute,
) -> tuple[np.ndarray, int, int]:
    """Return a full-line EngCut canvas containing only approved Latin pixels.

    EngCut sees one physical line with a small outer context margin.  Approved
    segment pixels are copied without expansion; CJK, punctuation, formulas,
    and structural regions stay white and cannot leak back across a route
    boundary.
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


def _write_masked_latin_line_hook(
    crop: np.ndarray,
    route: _LatinMaskedLineRoute,
    *,
    offset_x: int,
    offset_y: int,
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
        crop_bbox = [
            offset_x,
            offset_y,
            offset_x + int(crop.shape[1]),
            offset_y + int(crop.shape[0]),
        ]
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "schema": "hanwang_engcut_masked_input.v1",
                    "line_bbox": list(route.bbox),
                    "crop_bbox": crop_bbox,
                    "latin_segments": [
                        {
                            "route_key": list(segment.key),
                            "bbox": list(segment.bbox),
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


def _latin_segment_for_engcut_group(
    group: list[EngcutChar],
    segments: tuple[_TextRoute, ...],
) -> _TextRoute:
    boxes = [char.bbox for char in group if char.bbox is not None]
    if len(boxes) != len(group):
        raise RuntimeError("EngCut masked-line group has a character without page geometry")
    group_bbox = union_xyxy(boxes)
    owners: dict[tuple[int, int, int], _TextRoute] = {}
    for char in group:
        assert char.bbox is not None
        center_x, center_y = _bbox_center(char.bbox)
        matches = [
            segment
            for segment in segments
            if segment.bbox[0] <= center_x < segment.bbox[2]
            and segment.bbox[1] <= center_y < segment.bbox[3]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "EngCut masked-line character cannot be uniquely rebound to a Latin route: "
                f"char_bbox={char.bbox} group_bbox={group_bbox} "
                f"owners={[item.bbox for item in matches]}"
            )
        owners[matches[0].key] = matches[0]
    if len(owners) != 1:
        raise RuntimeError(
            "EngCut masked-line group spans multiple Latin routes: "
            f"group_bbox={group_bbox} owners={[item.bbox for item in owners.values()]}"
        )
    return next(iter(owners.values()))


def _recognize_latin_masked_line_with_engcut(
    image_bgr: np.ndarray,
    route: _LatinMaskedLineRoute,
    stats: RunStats,
    *,
    timeout: float,
) -> dict[tuple[int, int, int], LineResult]:
    """Recognize a white-masked physical line and strictly rebind native groups.

    There is intentionally no per-segment fallback.  A group that cannot be
    attributed to exactly one routing segment means the CharOCR dispatch plan
    and native result disagree, so the caller must fail the page rather than
    persist ambiguous character geometry.
    """
    crop, offset_x, offset_y = _materialize_latin_masked_line_crop(image_bgr, route)
    if crop.size == 0:
        raise RuntimeError(f"empty masked EngCut line crop: {route.bbox}")
    _write_masked_latin_line_hook(
        crop,
        route,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    stats.latin_engcut_route_calls += 1
    raw_eng20 = native_bridge.run_eng20_recogline(crop, timeout=timeout)
    page_chars = offset_engcut_chars(engcut_chars_from_payload(raw_eng20), dx=offset_x, dy=offset_y)
    groups = _engcut_groups(page_chars)
    if not groups:
        raise RuntimeError(f"EngCut returned no groups for masked line bbox={route.bbox}")

    grouped_by_route: dict[tuple[int, int, int], list[EngcutChar]] = {
        segment.key: [] for segment in route.segments
    }
    for group in groups:
        owner = _latin_segment_for_engcut_group(group, route.segments)
        grouped_by_route[owner.key].extend(group)

    results: dict[tuple[int, int, int], LineResult] = {}
    for segment in route.segments:
        chars = grouped_by_route[segment.key]
        text, char_results = _engcut_route_line_text_and_chars(chars)
        if not text or not any(char.text.strip() for char in char_results):
            raise RuntimeError(
                "EngCut masked-line result is missing a Latin routing segment: "
                f"line={route.bbox} segment={segment.bbox}"
            )
        boxes = [char.bbox for char in char_results if char.bbox is not None]
        results[segment.key] = LineResult(
            text=text,
            bbox=union_xyxy(boxes) if boxes else segment.bbox,
            confidence=0.0,
            chars=char_results,
            source=LATIN_ENGCUT_ROUTE_SOURCE,
            bbox_source="text_latin_masked_line_engcut",
            review_flags=[],
        )
    return results


def _recognize_latin_masked_lines_with_engcut(
    image_bgr: np.ndarray,
    routes: list[_LatinMaskedLineRoute],
    stats: RunStats,
    *,
    timeout: float,
) -> list[tuple[_LatinMaskedLineRoute, dict[tuple[int, int, int], LineResult]]]:
    """Recognize independent physical lines with bounded native concurrency.

    The process-wide executor caps total EngCut subprocess pressure while the
    per-page chunks keep one page from occupying every worker when page OCR is
    already concurrent.
    """
    results: list[tuple[_LatinMaskedLineRoute, dict[tuple[int, int, int], LineResult]]] = []

    def recognize(route: _LatinMaskedLineRoute):
        local_stats = RunStats()
        route_results = _recognize_latin_masked_line_with_engcut(
            image_bgr,
            route,
            local_stats,
            timeout=timeout,
        )
        return route_results, local_stats.latin_engcut_route_calls

    for start in range(0, len(routes), _MAX_ENGCUT_LINES_PER_PAGE):
        chunk = routes[start:start + _MAX_ENGCUT_LINES_PER_PAGE]
        if len(chunk) == 1:
            route_results, call_count = recognize(chunk[0])
            stats.latin_engcut_route_calls += call_count
            results.append((chunk[0], route_results))
            continue
        futures = [_ENGCUT_NATIVE_EXECUTOR.submit(recognize, route) for route in chunk]
        for route, future in zip(chunk, futures):
            route_results, call_count = future.result()
            stats.latin_engcut_route_calls += call_count
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
                "retry_top_trim": RECOG_GROUP_RETRY_TOP_TRIM,
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
        if _is_text_label(_effective_label_for_block(row))
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
        if _is_skip_label(label):
            skip_indices.append(idx)
        elif _is_text_label(label):
            text_indices.append(idx)

    stats.n_blocks_hanwang = len(text_indices)
    stats.n_blocks_ppvl = len(skip_indices)
    rows: list[BlockResult | None] = [None] * len(ppvl_blocks)
    text_routes: list[_TextRoute] = []
    latin_masked_line_routes: list[_LatinMaskedLineRoute] = []
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
        latin_masked_line_routes.extend(
            _latin_masked_line_routes_from_lines(
                block_idx,
                native_routes_by_block_index.get(block_idx, ()),
            )
        )
        for route in routes:
            recog_group_bboxes_by_route[route.key] = []
            segimg_group_audits_by_route[route.key] = []
    linecut_text_routes = [route for route in text_routes if not _is_latin_text_route(route)]
    text_route_recblocks = [route.bbox for route in linecut_text_routes]

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
        if progress_callback and linecut_text_routes:
            progress_callback(
                0,
                max(1, len(linecut_text_routes)),
                "Hanwang micro-recblock SegImg 分块中…",
            )
        if linecut_text_routes:
            started = time.time()
            seg = native_bridge.run_linecut_segimg(
                image_bgr,
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
        stats.n_groups = len(groups)
        if progress_callback:
            progress_callback(
                0,
                max(1, len(groups)),
                f"Hanwang micro-recblock Recog 准备中：{len(groups)} 个 group",
            )

        started = time.time()
        grouped_lines: dict[tuple[int, int, int], list[LineResult]] = {
            route.key: []
            for route in text_routes
        }
        for _masked_line_route, latin_results in _recognize_latin_masked_lines_with_engcut(
            image_bgr,
            latin_masked_line_routes,
            stats,
            timeout=min(30.0, max(1.0, float(recog_timeout))),
        ):
            for route_key, result in latin_results.items():
                grouped_lines[route_key].append(result)
        total_groups = max(1, len(groups))
        group_bboxes: list[tuple[int, int, int, int]] = []
        group_area_indices: list[int] = []
        for group in groups:
            recblock = recblocks[group["_area_idx"]]
            route = linecut_text_routes[group["_area_idx"]]
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
            segimg_group_audits_by_route.setdefault(route.key, []).append({
                "route_text_slice_bbox": list(recblock),
                "segimg_group_bbox": list(raw_group_bbox),
                "recog_group_bbox": list(recog_bbox) if recog_bbox is not None else None,
                "recog_group_bbox_before_padding": list(bbox) if bbox is not None else None,
                "recog_group_bbox_padded": recog_bbox is not None and recog_bbox != bbox,
                "clipped": bbox is not None and bbox != raw_group_bbox,
                "dropped": bbox is None,
            })
            if recog_bbox is None:
                continue
            if recog_bbox[2] <= recog_bbox[0] or recog_bbox[3] <= recog_bbox[1]:
                continue
            group_bboxes.append(recog_bbox)
            group_area_indices.append(group["_area_idx"])
            recog_group_bboxes_by_route.setdefault(route.key, []).append(recog_bbox)

        def update_recog_group_audit(placement: _GroupPlacement, values: dict[str, Any]) -> None:
            route = linecut_text_routes[placement.area_idx]
            audits = segimg_group_audits_by_route.setdefault(route.key, [])
            for item in audits:
                if item.get("recog_group_bbox") == list(placement.page_bbox):
                    item.update(values)
                    return
            audits.append({
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
                    "recog_retry_strategy": f"trim_top_{RECOG_GROUP_RETRY_TOP_TRIM}px",
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
                    "recog_retry_strategy": f"trim_top_{RECOG_GROUP_RETRY_TOP_TRIM}px",
                    "recog_retry_original_error": str(original_error),
                    "recog_retry_bbox": list(retry_bbox),
                    "recog_retry_succeeded": True,
                },
            )

        def retry_bbox_after_top_trim(
            bbox: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int] | None:
            left, top, right, bottom = bbox
            if bottom - top <= RECOG_GROUP_RETRY_TOP_TRIM + 8:
                return None
            retry_top = min(bottom - 1, top + RECOG_GROUP_RETRY_TOP_TRIM)
            if bottom - retry_top < 8:
                return None
            return left, retry_top, right, bottom

        def recognize_individually(placements: list[_GroupPlacement]) -> None:
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crop = image_bgr[top:bottom, left:right].copy()
                crop_h, crop_w = crop.shape[:2]
                offset_left = left
                offset_top = top
                active_crop = crop
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        crop,
                        recblock_xyxy=None,
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    retry_bbox = retry_bbox_after_top_trim(placement.page_bbox)
                    if retry_bbox is not None:
                        retry_left, retry_top, retry_right, retry_bottom = retry_bbox
                        retry_crop = image_bgr[retry_top:retry_bottom, retry_left:retry_right].copy()
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
                    _refine_overlap_fragments_with_recrop(
                        active_crop,
                        local_lines,
                        stats,
                        timeout=recog_timeout,
                    )
                route = linecut_text_routes[placement.area_idx]
                offset_lines = _offset_line_results(local_lines, dx=offset_left, dy=offset_top)
                grouped_lines[route.key].extend(
                    _filter_line_results_to_route_bbox(offset_lines, route.bbox)
                )

        def recognize_batch_list(placements: list[_GroupPlacement]) -> None:
            crops: list[np.ndarray] = []
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crops.append(image_bgr[top:bottom, left:right].copy())
            stats.recog_probe_calls += 1
            raws = native_bridge.run_linecut_recog_batch_list(
                crops,
                with_charrcg=True,
                timeout=recog_timeout,
            )
            if len(raws) != len(placements):
                raise RuntimeError(f"batch-list result count mismatch: {len(raws)}/{len(placements)}")
            for placement, crop, raw in zip(placements, crops, raws):
                crop_h, crop_w = crop.shape[:2]
                local_lines = _line_results_from_recog(
                    raw,
                    fallback_bbox=(0, 0, crop_w, crop_h),
                    include_chars=include_chars,
                )
                if include_chars:
                    _refine_overlap_fragments_with_recrop(
                        crop,
                        local_lines,
                        stats,
                        timeout=recog_timeout,
                    )
                route = linecut_text_routes[placement.area_idx]
                offset_lines = _offset_line_results(
                    local_lines,
                    dx=placement.page_bbox[0],
                    dy=placement.page_bbox[1],
                )
                grouped_lines[route.key].extend(
                    _filter_line_results_to_route_bbox(offset_lines, route.bbox)
                )

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
                placements.append(
                    _GroupPlacement(
                        area_idx=area_idx,
                        page_bbox=page_bbox,
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
        "_layout_block_ocr_policy": ocr_policy.value,
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
            "latin_engcut_route_calls=%d "
            "overlap_merge_clusters=%d overlap_merge_calls=%d overlap_merge_failures=%d "
            "overlap_merge_replacements=%d",
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
            stats.latin_engcut_route_calls,
            stats.overlap_merge_clusters,
            stats.overlap_merge_probe_calls,
            stats.overlap_merge_probe_failures,
            stats.overlap_merge_replacements,
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
