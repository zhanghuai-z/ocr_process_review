"""Page-level PP-VL block -> Hanwang linecut micro-recblock integration."""
from __future__ import annotations

import os
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.bbox_extraction import bbox_from_variant
from app.core.block_payload import (
    HANWANG_BBOX_AUDIT_KEY,
    OCR_TEXT_INVALIDATED_KEY,
    OCR_INVALIDATION_KIND_KEY,
    PADDLE_BINDING_KEY,
    PADDLE_BLOCK_BBOX_KEY,
    PADDLE_BLOCK_LABEL_KEY,
    set_payload_entries,
    split_legacy_raw_payload,
    strip_runtime_layout_payload,
)
from app.core.ocr_dispatch_policy import default_ocr_policy_for_block
from app.core.ocr_line_hints import is_ppocr_page_line_hint
from app.core.logging import get_logger
from app.core.latin_span_recovery import (
    LATIN_ENGCUT_BBOX_GRANULARITY,
    LATIN_ENGCUT_BBOX_SOURCE,
    LATIN_ENGCUT_EXACT_STATUS,
    LATIN_ENGCUT_MULTILINE_STATUS,
    LATIN_ENGCUT_REVIEW_FLAG,
    LATIN_ENGCUT_REVERSE_STATUS,
    LATIN_ENGCUT_VARIANT_STATUS,
    LATIN_ENGCUT_WORD_BBOX_GRANULARITY,
    LATIN_ENGCUT_WORD_BBOX_SOURCE,
    LATIN_ENGCUT_WORD_FALLBACK_STATUS,
    EngcutChar,
    LatinToken,
    bind_latin_tokens_to_engcut_chars,
    compact_latin_key,
    engcut_chars_from_payload,
    engcut_geometry_reasons,
    fuzzy_latin_word_allowed,
    offset_engcut_chars,
    text_token_spans,
    token_variants,
)
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    ROUTE_INLINE_FORMULA_FLAG,
    ROUTE_SUBBLOCKS_FIELD,
    ROUTE_TABLE_FLAG,
    attach_page_ocr_line_routes,
    block_bbox_xyxy,
    block_text as paddle_block_text,
    has_layout_line_routes,
    is_formula_label,
    is_formula_style_position_block,
    is_table_label,
    line_routes_for_block,
    route_authority_label,
    route_subblocks_for_block,
    text_slice_routes_for_block,
    union_xyxy,
    vertical_overlap_ratio,
)
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
    authoritative_paddle_label,
    is_hanwang_skip_label,
    normalize_paddle_label,
)
from app.core.proof_line_facts import proof_block_text
from app.core.proof_status import proof_status_for
from app.core.raw_ocr_artifact import raw_layout_records
from app.engines import OCR_BBOX_SPACE_PAGE
from app.models import BBox, Block, BlockSource, BlockType, Char, Line, OcrPolicy, Page

from . import native_bridge

logger = get_logger(__name__)

TEXT_LABELS: set[str] = set(PADDLE_HANWANG_TEXT_LABELS)
SKIP_LABELS: set[str] = set(PADDLE_HANWANG_SKIP_LABELS)
TEXT_ROUTE_INK_THRESHOLD = 220
TEXT_ROUTE_INK_MIN_ROW_PIXELS = 2
TEXT_ROUTE_INK_MIN_ROW_RATIO = 0.015
TEXT_ROUTE_INK_PAD_Y = 6
DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG = "hanwang_digitlike_numeric_context"
LATIN_ENGCUT_TEXT_REWRITE_MAX_CONFIDENCE = 0.25
FORMULA_CROP_OCR_REVIEW_FLAG = "paddle_formula_crop_ocr"
FORMULA_CROP_OCR_FAILED_FLAG = "paddle_formula_crop_ocr_failed"
_DIGITLIKE_ZERO_CHARS = {"o", "O"}
_DIGITLIKE_ONE_CHARS = {"l", "I"}
_DIGITLIKE_NUMERIC_CONTEXT_FOLLOWERS = {"", "，", ",", "。", ".", "；", ";", "、", ")", "）"}
_INLINE_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)
_SIMPLE_FORMULA_LETTER_LIST_RE = re.compile(r"[A-Za-z\s,，、;；]+")

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
    latin_engcut_probe_calls: int = 0
    latin_engcut_probe_failures: int = 0
    latin_engcut_exact_tokens: int = 0
    latin_engcut_word_tokens: int = 0
    latin_engcut_review_tokens: int = 0
    latin_engcut_disabled: bool = False
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

    @property
    def key(self) -> tuple[int, int, int]:
        return self.block_idx, self.line_idx, self.segment_idx


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


def _route_subblocks(block: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    return route_subblocks_for_block(block, width, height)


def _text_route_bboxes_for_block(
    block: dict[str, Any],
    block_idx: int,
    width: int,
    height: int,
) -> list[_TextRoute]:
    routes = text_slice_routes_for_block(block, width, height)
    return [
        _TextRoute(
            block_idx=block_idx,
            line_idx=int(route.get("line_idx", -1)),
            segment_idx=int(route.get("segment_idx", 0)),
            bbox=tuple(route["bbox"]),
            carved=bool(route.get("carved", False)),
        )
        for route in routes
    ]


def _refined_text_segment_bbox_from_ink(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None
    page_h, page_w = image_bgr.shape[:2]
    left, top, right, bottom = _clamp_xyxy((left, top, right, bottom), page_w, page_h)
    if right <= left or bottom <= top:
        return None
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    if crop.ndim == 3 and crop.shape[2] >= 3:
        gray = (
            crop[:, :, 0].astype(np.float32) * 0.114
            + crop[:, :, 1].astype(np.float32) * 0.587
            + crop[:, :, 2].astype(np.float32) * 0.299
        )
    else:
        gray = crop.astype(np.float32)
    dark = gray < TEXT_ROUTE_INK_THRESHOLD
    row_counts = dark.sum(axis=1)
    min_pixels = max(TEXT_ROUTE_INK_MIN_ROW_PIXELS, int(round((right - left) * TEXT_ROUTE_INK_MIN_ROW_RATIO)))
    rows = np.flatnonzero(row_counts >= min_pixels)
    if rows.size == 0:
        return None
    refined_top = max(top, top + int(rows[0]) - TEXT_ROUTE_INK_PAD_Y)
    refined_bottom = min(bottom, top + int(rows[-1]) + 1 + TEXT_ROUTE_INK_PAD_Y)
    if refined_bottom <= refined_top:
        return None
    return left, refined_top, right, refined_bottom


def _refine_layout_text_route_bands_from_image(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict],
    width: int,
    height: int,
) -> None:
    """Tighten text route vertical bands using page pixels, not formula-heavy PP-OCR rows."""
    if image_bgr.size == 0:
        return
    for block in ppvl_blocks:
        routes = line_routes_for_block(block, width, height)
        if not routes:
            continue
        changed = False
        refined_routes: list[dict[str, Any]] = []
        for route in routes:
            segments = route.get("segments")
            if not isinstance(segments, list):
                refined_routes.append(route)
                continue
            has_formula = any(
                isinstance(segment, dict) and segment.get("kind") == "formula"
                for segment in segments
            )
            if not has_formula:
                refined_routes.append(route)
                continue
            refined_segments: list[dict[str, Any]] = []
            route_changed = False
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                next_segment = dict(segment)
                if segment.get("kind") == "text":
                    raw_bbox = segment.get("bbox")
                    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                        refined_segments.append(next_segment)
                        continue
                    try:
                        bbox = _clamp_xyxy(
                            (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])),
                            width,
                            height,
                        )
                    except (TypeError, ValueError):
                        refined_segments.append(next_segment)
                        continue
                    refined_bbox = _refined_text_segment_bbox_from_ink(image_bgr, bbox)
                    if refined_bbox is not None and refined_bbox != bbox:
                        next_segment["bbox"] = list(refined_bbox)
                        route_changed = True
                refined_segments.append(next_segment)
            if route_changed and refined_segments:
                next_route = dict(route)
                next_route["segments"] = refined_segments
                next_route["bbox"] = list(union_xyxy([tuple(segment["bbox"]) for segment in refined_segments]))
                refined_routes.append(next_route)
                changed = True
            else:
                refined_routes.append(route)
        if changed:
            block[LAYOUT_LINE_ROUTES_FIELD] = refined_routes


def _has_route_subblocks(block: dict[str, Any], width: int, height: int) -> bool:
    return has_layout_line_routes(block, width, height)


def _semantic_text_routes(
    ppvl_blocks: list[dict],
    width: int,
    height: int,
) -> list[_TextRoute]:
    routes: list[_TextRoute] = []
    for block_idx, block in enumerate(ppvl_blocks):
        label = _effective_label_for_block(block)
        if not _is_text_label(label):
            continue
        routes.extend(_text_route_bboxes_for_block(block, block_idx, width, height))
    return routes


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
    route: dict[str, Any],
    grouped_lines: dict[tuple[int, int, int], list[LineResult]],
) -> list[LineResult]:
    segments = route.get("segments") or []
    if not segments:
        return []

    if all(segment.get("kind") == "skip" for segment in segments):
        segment = segments[0]
        text = str(segment.get("text") or "")
        if not text:
            return []
        flags = [ROUTE_TABLE_FLAG] if is_table_label(str(segment.get("label") or "")) else []
        return [
            LineResult(
                text=text,
                bbox=tuple(segment["bbox"]),
                confidence=0.0,
                chars=[],
                source=f"ppvl_route:{segment.get('label') or 'skip'}",
                bbox_source="ppvl_route_skip_segment",
                review_flags=flags,
            )
        ]

    slice_lines_by_segment: dict[int, list[LineResult]] = {}
    all_text_lines: list[LineResult] = []
    has_formula = any(segment.get("kind") == "formula" for segment in segments)
    for segment_idx, segment in enumerate(segments):
        if segment.get("kind") != "text":
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
            segment_bbox = tuple(segment["bbox"])
            kind = segment.get("kind")
            if kind == "text":
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
                formula_text = str(segment.get("text") or "")
                if not formula_text:
                    continue
                text_parts.append(formula_text)
                component_boxes.append(segment_bbox)
                chars.append(
                    CharResult(
                        text=formula_text,
                        confidence=0.0,
                        bbox=segment_bbox,
                        candidates=[formula_text],
                        source="paddle_inline_formula",
                        bbox_granularity="word",
                        token_text=formula_text,
                    )
                )
                char_bounds.append(segment_bbox)
                flags.add(ROUTE_INLINE_FORMULA_FLAG)
        merged_text = "".join(text_parts)
        if not merged_text:
            continue
        assembled.append(
            _recover_degenerate_punctuation_bboxes(
                LineResult(
                    text=merged_text,
                    bbox=union_xyxy(component_boxes) if component_boxes else tuple(route["bbox"]),
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


def _assemble_layout_route_lines(
    *,
    block_idx: int,
    block: dict[str, Any],
    grouped_lines: dict[tuple[int, int, int], list[LineResult]],
    width: int,
    height: int,
) -> list[LineResult]:
    line_routes = line_routes_for_block(block, width, height)
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


def _layout_line_route_bboxes(raw: dict, width: int, height: int) -> list[tuple[int, int, int, int]]:
    line_routes = line_routes_for_block(raw, width, height)
    return [tuple(route["bbox"]) for route in line_routes if route.get("bbox")]


def _effective_block_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int]:
    line_routes = _layout_line_route_bboxes(raw, width, height)
    if line_routes:
        return union_xyxy(line_routes)
    return _layout_block_bbox(raw, width, height)


def _effective_block_bbox_source(raw: dict, width: int, height: int) -> str:
    return "layout_line_routes_union" if _layout_line_route_bboxes(raw, width, height) else "layout_block_bbox"


def _block_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return _effective_block_bbox(raw, width, height)


def _bbox_lists(values: list[tuple[int, int, int, int]]) -> list[list[int]]:
    return [list(bbox) for bbox in values]


def _hanwang_bbox_audit(
    raw: dict[str, Any],
    width: int,
    height: int,
    *,
    block_bbox: tuple[int, int, int, int] | None = None,
    route_text_slice_bboxes: list[tuple[int, int, int, int]] | None = None,
    recog_group_bboxes: list[tuple[int, int, int, int]] | None = None,
    segimg_group_audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    layout_bbox = _layout_block_bbox(raw, width, height)
    line_route_bboxes = _layout_line_route_bboxes(raw, width, height)
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
        "effective_block_bbox_source": (
            "layout_line_routes_union" if line_route_bboxes else "layout_block_bbox"
        ),
        "layout_line_route_bboxes": _bbox_lists(line_route_bboxes),
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
    route_text_slice_bboxes: list[tuple[int, int, int, int]] | None = None,
    recog_group_bboxes: list[tuple[int, int, int, int]] | None = None,
    segimg_group_audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = dict(raw)
    value[HANWANG_BBOX_AUDIT_KEY] = _hanwang_bbox_audit(
        raw,
        width,
        height,
        block_bbox=block_bbox,
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
    left_cx, _ = _box_center(left.bbox)
    right_cx, _ = _box_center(right.bbox)
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
                line.chars[start:end] = replacement
                stats.overlap_merge_replacements += 1
            else:
                continue
            line.text = "".join(char.text for char in line.chars).strip()
            if line.chars:
                line.confidence = sum(char.confidence for char in line.chars) / len(line.chars)


def _line_char_offset_map(line: LineResult) -> list[int] | None:
    char_text = "".join(char.text for char in line.chars)
    if char_text != line.text:
        return None
    offsets: list[int] = []
    for char_index, char in enumerate(line.chars):
        offsets.extend([char_index] * len(char.text))
    return offsets


def _mark_latin_engcut_review(line: LineResult) -> None:
    if LATIN_ENGCUT_REVIEW_FLAG not in line.review_flags:
        line.review_flags.append(LATIN_ENGCUT_REVIEW_FLAG)


CHINESE_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】［］〔〕—…·．")
LATIN_REVERSE_OCCUPY_CONFIDENCE = 0.50
LATIN_CANDIDATE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./&+-")
RECOG_GROUP_CROP_PAD_X = 8
RECOG_GROUP_CROP_PAD_Y = 10
RECOG_GROUP_RETRY_TOP_TRIM = 3
ENGCUT_LINE_CROP_PAD_X = 2
ENGCUT_LINE_CROP_PAD_Y = 2
OVERLAP_MERGE_LOW_CONFIDENCE = 0.35
OVERLAP_MERGE_IOA_THRESHOLD = 0.45
OVERLAP_MERGE_VERTICAL_THRESHOLD = 0.55
OVERLAP_MERGE_MAX_CLUSTER_CHARS = 5
OVERLAP_MERGE_PAD_X = 3
OVERLAP_MERGE_PAD_Y = 3


@dataclass
class _EngcutLine:
    line: LineResult
    bbox: tuple[int, int, int, int]
    chars: list[EngcutChar]
    text: str
    order: int


@dataclass
class _EngcutEntry:
    char: EngcutChar
    line_record: _EngcutLine


@dataclass
class _TokenBinding:
    token: LatinToken
    status: str
    matched_text: str
    chars: list[EngcutChar]
    line_records: list[_EngcutLine]
    line_span: tuple[int, int] | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        boxes = [char.bbox for char in self.chars if char.bbox is not None]
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )


def _looks_cjk_text(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text or ""
    )


def _is_occupied_cjk_or_punct(char: CharResult) -> bool:
    if not char.text or char.bbox is None:
        return False
    if char.confidence < LATIN_REVERSE_OCCUPY_CONFIDENCE:
        return False
    return _looks_cjk_text(char.text) or char.text in CHINESE_PUNCT


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _contains_point(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _overlap_ratio(
    candidate: tuple[int, int, int, int],
    blocker: tuple[int, int, int, int],
) -> float:
    area = _intersection_area(candidate, blocker)
    if area <= 0:
        return 0.0
    return area / max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))


def _engcut_candidate_char(text: str) -> bool:
    return len(text) == 1 and text in LATIN_CANDIDATE_CHARS


def _engcut_char_blocked(char: EngcutChar, occupied: list[tuple[int, int, int, int]]) -> bool:
    if char.bbox is None:
        return True
    point = _box_center(char.bbox)
    for blocker in occupied:
        if _contains_point(blocker, point):
            return True
    return any(_overlap_ratio(char.bbox, blocker) >= 0.55 for blocker in occupied)


def _engcut_stream(records: list[_EngcutLine]) -> tuple[str, list[_EngcutEntry | None]]:
    parts: list[str] = []
    entries: list[_EngcutEntry | None] = []
    for record in records:
        for char in record.chars:
            parts.append(char.text)
            entries.append(_EngcutEntry(char=char, line_record=record))
        parts.append("\n")
        entries.append(None)
    return "".join(parts), entries


def _future_token_before(
    stream_text: str,
    cursor: int,
    candidate_start: int,
    future_tokens: list[LatinToken],
) -> bool:
    for future in future_tokens:
        for variant in token_variants(future.text):
            pos = stream_text.find(variant, cursor)
            if 0 <= pos < candidate_start:
                return True
    return False


def _line_span_from_stream_entries(
    entries: list[_EngcutEntry | None],
    start: int,
    end: int,
    record: _EngcutLine,
) -> tuple[int, int] | None:
    if start < 0 or end <= start or end > len(entries):
        return None
    if any(item is None or item.line_record is not record for item in entries[start:end]):
        return None
    line_start = start
    while line_start > 0:
        previous = entries[line_start - 1]
        if previous is None or previous.line_record is not record:
            break
        line_start -= 1
    return start - line_start, end - line_start


def _bind_token_in_stream(
    token: LatinToken,
    future_tokens: list[LatinToken],
    stream_text: str,
    entries: list[_EngcutEntry | None],
    cursor: int,
) -> tuple[_TokenBinding | None, int]:
    search_cursor = cursor
    while search_cursor < len(stream_text):
        best: tuple[int, str] | None = None
        for variant in token_variants(token.text):
            pos = stream_text.find(variant, search_cursor)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, variant)
        if best is None:
            return _bind_token_from_word_fallback_group(token, future_tokens, stream_text, entries, cursor)
        found, variant = best
        if token.kind != "formula_letter" and _future_token_before(stream_text, cursor, found, future_tokens):
            return None, cursor
        end = found + len(variant)
        entry_slice = entries[found:end]
        if len(entry_slice) != len(variant) or any(item is None or item.char.bbox is None for item in entry_slice):
            search_cursor = found + 1
            continue
        records: list[_EngcutLine] = []
        chars: list[EngcutChar] = []
        for item in entry_slice:
            assert item is not None
            chars.append(item.char)
            if item.line_record not in records:
                records.append(item.line_record)
        status = LATIN_ENGCUT_EXACT_STATUS if variant == token.text else LATIN_ENGCUT_VARIANT_STATUS
        if len(records) > 1:
            status = LATIN_ENGCUT_MULTILINE_STATUS
        line_span = _line_span_from_stream_entries(entries, found, end, records[0]) if len(records) == 1 else None
        binding = _TokenBinding(
            token=token,
            status=status,
            matched_text=variant,
            chars=chars,
            line_records=records,
            line_span=line_span,
        )
        if _formula_letter_binding_hits_protected_span(binding):
            search_cursor = found + 1
            continue
        return binding, end
    return _bind_token_from_word_fallback_group(token, future_tokens, stream_text, entries, cursor)


def _bind_token_from_word_fallback_group(
    token: LatinToken,
    future_tokens: list[LatinToken],
    stream_text: str,
    entries: list[_EngcutEntry | None],
    cursor: int,
) -> tuple[_TokenBinding | None, int]:
    if not token.text.isalpha():
        return None, cursor
    index = cursor
    while index < len(entries):
        item = entries[index]
        if item is None:
            index += 1
            continue
        group_key = (id(item.line_record), item.char.line_index, item.char.group_index)
        end = index + 1
        while end < len(entries):
            next_item = entries[end]
            if next_item is None:
                break
            next_key = (id(next_item.line_record), next_item.char.line_index, next_item.char.group_index)
            if next_key != group_key:
                break
            end += 1
        group_entries = [entry for entry in entries[index:end] if entry is not None and entry.char.bbox is not None]
        candidate_text = "".join(entry.char.text for entry in group_entries)
        if (
            len(group_entries) == end - index
            and fuzzy_latin_word_allowed(token.text, candidate_text)
            and not _future_token_before(stream_text, cursor, index, future_tokens)
        ):
            records: list[_EngcutLine] = []
            chars: list[EngcutChar] = []
            for entry in group_entries:
                chars.append(entry.char)
                if entry.line_record not in records:
                    records.append(entry.line_record)
            return (
                _TokenBinding(
                    token=token,
                    status=LATIN_ENGCUT_WORD_FALLBACK_STATUS,
                    matched_text=candidate_text,
                    chars=chars,
                    line_records=records,
                    line_span=_line_span_from_stream_entries(entries, index, end, records[0]) if len(records) == 1 else None,
                ),
                end,
            )
        index = end
    return None, cursor


def _reverse_groups(records: list[_EngcutLine]) -> list[_TokenBinding]:
    groups: list[_TokenBinding] = []
    for record in records:
        occupied = [
            char.bbox
            for char in record.line.chars
            if _is_occupied_cjk_or_punct(char) and char.bbox is not None
        ]
        current: list[EngcutChar] = []
        for char in record.chars:
            keep = (
                _engcut_candidate_char(char.text)
                and char.bbox is not None
                and not _engcut_char_blocked(char, occupied)
            )
            if keep:
                current.append(char)
                continue
            if current:
                text = "".join(item.text for item in current)
                groups.append(_TokenBinding(
                    token=LatinToken(text=text, start=-1, end=-1),
                    status=LATIN_ENGCUT_REVERSE_STATUS,
                    matched_text=text,
                    chars=current,
                    line_records=[record],
                ))
                current = []
        if current:
            text = "".join(item.text for item in current)
            groups.append(_TokenBinding(
                token=LatinToken(text=text, start=-1, end=-1),
                status=LATIN_ENGCUT_REVERSE_STATUS,
                matched_text=text,
                chars=current,
                line_records=[record],
            ))
    return [group for group in groups if len(group.matched_text) >= 2]


def _reverse_group_matches(group_text: str, token_text: str) -> tuple[bool, str]:
    for variant in token_variants(token_text):
        if group_text == variant:
            return True, variant
        if group_text.strip("/f!lI1|") == variant:
            return True, group_text
    return False, ""


def _bind_token_from_reverse_groups(
    token: LatinToken,
    groups: list[_TokenBinding],
    cursor: int,
) -> tuple[_TokenBinding | None, int]:
    for idx in range(cursor, len(groups)):
        group = groups[idx]
        ok, variant = _reverse_group_matches(group.matched_text, token.text)
        if ok:
            return _TokenBinding(
                token=token,
                status=LATIN_ENGCUT_REVERSE_STATUS,
                matched_text=variant,
                chars=group.chars,
                line_records=group.line_records,
            ), idx + 1
    return None, cursor


def _formula_letter_binding_hits_protected_span(binding: _TokenBinding) -> bool:
    if binding.token.kind != "formula_letter":
        return False
    if len(binding.line_records) != 1 or not binding.chars:
        return False
    binding_bbox = binding.bbox
    if binding_bbox is None:
        return False
    line = binding.line_records[0].line
    span = _span_for_binding(line, binding_bbox, binding.token.text, binding.matched_text, binding.line_span)
    if span is None:
        return False
    start, end = span
    return any(_is_occupied_cjk_or_punct(char) for char in line.chars[start:end])


def _char_center_inside_binding(
    char: CharResult,
    binding_bbox: tuple[int, int, int, int],
) -> bool:
    if char.bbox is None:
        return False
    x1, y1, x2, y2 = binding_bbox
    cx, cy = _box_center(char.bbox)
    return (x1 - 2) <= cx <= (x2 + 2) and (y1 - 4) <= cy <= (y2 + 4)


def _span_text_for_chars(chars: list[CharResult], start: int, end: int) -> str:
    return "".join(char.text for char in chars[start:end])


def _span_matches_binding_text(span_text: str, token: str, matched_text: str) -> bool:
    if not span_text:
        return False
    for candidate in (token, matched_text):
        if not candidate:
            continue
        if span_text == candidate:
            return True
        if span_text in token_variants(candidate) or candidate in token_variants(span_text):
            return True
    return False


def _low_confidence_latin_rewrite_candidate(char: CharResult) -> bool:
    if float(char.confidence or 0.0) > LATIN_ENGCUT_TEXT_REWRITE_MAX_CONFIDENCE:
        return False
    text = str(char.text or "")
    if len(text) != 1:
        return False
    return text in LATIN_CANDIDATE_CHARS or _looks_cjk_text(text)


def _span_allows_engcut_text_rewrite(chars: list[CharResult], token_text: str) -> bool:
    current = [str(char.text or "") for char in chars]
    if len(chars) == len(token_text):
        changed = False
        for char, expected in zip(chars, token_text):
            if str(char.text or "") == expected:
                continue
            changed = True
            if not _low_confidence_latin_rewrite_candidate(char):
                return False
        return changed

    prefix = 0
    while prefix < len(current) and prefix < len(token_text) and current[prefix] == token_text[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(current) - prefix
        and suffix < len(token_text) - prefix
        and current[len(current) - 1 - suffix] == token_text[len(token_text) - 1 - suffix]
    ):
        suffix += 1

    changed_chars = chars[prefix:len(chars) - suffix if suffix else len(chars)]
    replacement_text = token_text[prefix:len(token_text) - suffix if suffix else len(token_text)]
    if not changed_chars or not replacement_text:
        return False
    if not all(ch in LATIN_CANDIDATE_CHARS for ch in replacement_text):
        return False
    return all(_low_confidence_latin_rewrite_candidate(char) for char in changed_chars)


def _binding_span_score(
    line: LineResult,
    span: tuple[int, int],
    binding_bbox: tuple[int, int, int, int],
) -> int:
    start, end = span
    return sum(
        1
        for char in line.chars[start:end]
        if _char_center_inside_binding(char, binding_bbox)
    )


def _find_text_span_for_binding(
    line: LineResult,
    binding_bbox: tuple[int, int, int, int],
    token: str,
    matched_text: str,
) -> tuple[int, int] | None:
    offset_map = _line_char_offset_map(line)
    best: tuple[int, int, int] | None = None
    for candidate in (token, matched_text):
        if not candidate:
            continue
        cursor = 0
        while True:
            pos = line.text.find(candidate, cursor)
            if pos < 0:
                break
            cursor = pos + 1
            if offset_map is not None and pos + len(candidate) <= len(offset_map):
                mapped = offset_map[pos:pos + len(candidate)]
                if not mapped:
                    continue
                span = (min(mapped), max(mapped) + 1)
            else:
                span = (pos, pos + len(candidate))
            start, end = span
            if start < 0 or end > len(line.chars) or start >= end:
                continue
            score = _binding_span_score(line, span, binding_bbox)
            if best is None or score > best[0] or (score == best[0] and start < best[1]):
                best = (score, start, end)
    if best is None:
        return None
    return best[1], best[2]


def _span_for_binding(
    line: LineResult,
    binding_bbox: tuple[int, int, int, int],
    token: str,
    matched_text: str,
    line_span: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    offset_map = _line_char_offset_map(line)
    if line_span is not None:
        start, end = line_span
        if 0 <= start < end:
            if offset_map is not None and end <= len(offset_map):
                mapped = offset_map[start:end]
                if mapped and _span_matches_binding_text(
                    _span_text_for_chars(line.chars, min(mapped), max(mapped) + 1),
                    token,
                    matched_text,
                ):
                    return min(mapped), max(mapped) + 1
            if end <= len(line.chars) and _span_matches_binding_text(
                _span_text_for_chars(line.chars, start, end),
                token,
                matched_text,
            ):
                return start, end
    text_span = _find_text_span_for_binding(line, binding_bbox, token, matched_text)
    if text_span is not None:
        return text_span
    indices = [
        idx
        for idx, char in enumerate(line.chars)
        if _char_center_inside_binding(char, binding_bbox)
    ]
    if indices:
        return min(indices), max(indices) + 1
    return None


def _replace_line_span_with_binding(line: LineResult, binding: _TokenBinding) -> bool:
    if len(binding.line_records) != 1 or not binding.chars:
        return False
    if binding.line_records[0].line is not line:
        return False
    binding_bbox = binding.bbox
    if binding_bbox is None:
        return False
    span = _span_for_binding(line, binding_bbox, binding.token.text, binding.matched_text, binding.line_span)
    if span is None:
        return False
    start, end = span
    token_text = binding.token.text
    if len(binding.chars) != len(token_text):
        return False
    current_chars = line.chars[start:end]
    if _span_text_for_chars(current_chars, 0, len(current_chars)) != token_text:
        if not _span_allows_engcut_text_rewrite(current_chars, token_text):
            return False
    replacement = [
        CharResult(
            text=token_text[idx],
            confidence=line.confidence,
            bbox=char.bbox,
            candidates=[token_text[idx]],
            source=LATIN_ENGCUT_BBOX_SOURCE,
            bbox_granularity=LATIN_ENGCUT_BBOX_GRANULARITY,
            token_text=token_text,
        )
        for idx, char in enumerate(binding.chars)
    ]
    line.chars[start:end] = replacement
    line.text = "".join(char.text for char in line.chars)
    return True


def _binding_word_fallback_reasons(binding: _TokenBinding) -> tuple[str, ...]:
    if not binding.token.text.isalpha():
        return ()
    if binding.status == LATIN_ENGCUT_WORD_FALLBACK_STATUS:
        return ("word_text_fallback",)
    return engcut_geometry_reasons(binding.chars)


def _replace_line_span_with_word_binding(line: LineResult, binding: _TokenBinding) -> bool:
    if len(binding.line_records) != 1 or not binding.chars:
        return False
    if binding.line_records[0].line is not line:
        return False
    binding_bbox = binding.bbox
    if binding_bbox is None:
        return False
    span = _span_for_binding(line, binding_bbox, binding.token.text, binding.matched_text, binding.line_span)
    if span is None:
        return False
    start, end = span
    token_text = binding.token.text
    current_chars = line.chars[start:end]
    if _span_text_for_chars(current_chars, 0, len(current_chars)) != token_text:
        if not all(_low_confidence_latin_rewrite_candidate(char) for char in current_chars):
            return False
    line.chars[start:end] = [
        CharResult(
            text=token_text,
            confidence=line.confidence,
            bbox=binding_bbox,
            candidates=[token_text],
            source=LATIN_ENGCUT_WORD_BBOX_SOURCE,
            bbox_granularity=LATIN_ENGCUT_WORD_BBOX_GRANULARITY,
            token_text=token_text,
        )
    ]
    line.text = "".join(char.text for char in line.chars)
    return True


def _binding_can_update_text(binding: _TokenBinding) -> bool:
    if binding.status == LATIN_ENGCUT_EXACT_STATUS:
        return binding.matched_text == binding.token.text
    if binding.status == LATIN_ENGCUT_REVERSE_STATUS:
        return binding.matched_text == binding.token.text
    return False


def _line_text_stream(lines: list[LineResult]) -> tuple[str, list[int | None]]:
    parts: list[str] = []
    entries: list[int | None] = []
    for order, line in enumerate(lines):
        for char in line.text:
            parts.append(char)
            entries.append(order)
        parts.append("\n")
        entries.append(None)
    return "".join(parts), entries


def _line_has_text_token(line: LineResult) -> bool:
    return bool(text_token_spans(line.text, skip_formula_spans=False))


def _subsequence_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    cursor = 0
    matched = 0
    for char in left:
        pos = right.find(char, cursor)
        if pos < 0:
            continue
        matched += 1
        cursor = pos + 1
    return matched


def _line_may_contain_block_token(line_text: str, token_text: str) -> bool:
    line_key = compact_latin_key(line_text)
    token_key = compact_latin_key(token_text)
    if len(line_key) < 2 or len(token_key) < 2:
        return False
    if token_key in line_key or line_key in token_key:
        return True
    common = _subsequence_length(line_key, token_key)
    return common >= 3 and common / max(1, len(token_key)) >= 0.55


def _block_text_engcut_tokens(block_text: str) -> list[LatinToken]:
    tokens = list(text_token_spans(block_text)) if block_text else []
    tokens.extend(_simple_formula_letter_tokens(block_text))
    tokens.sort(key=lambda token: (token.start, token.end, token.text))
    deduped: list[LatinToken] = []
    seen: set[tuple[int, int, str]] = set()
    for token in tokens:
        key = (token.start, token.end, token.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def _simple_formula_letter_tokens(block_text: str) -> list[LatinToken]:
    tokens: list[LatinToken] = []
    for match in _INLINE_FORMULA_SPAN_RE.finditer(block_text or ""):
        raw = match.group(0)
        if raw.startswith("$$") and raw.endswith("$$"):
            inner = raw[2:-2]
            offset = 2
        elif raw.startswith("$") and raw.endswith("$"):
            inner = raw[1:-1]
            offset = 1
        else:
            continue
        if not _SIMPLE_FORMULA_LETTER_LIST_RE.fullmatch(inner.strip()):
            continue
        letters = list(re.finditer(r"[A-Za-z]", inner))
        if len(letters) < 2:
            continue
        for letter in letters:
            start = match.start() + offset + letter.start()
            tokens.append(LatinToken(text=letter.group(0), start=start, end=start + 1, kind="formula_letter"))
    return tokens


def _has_formula_letter_tokens(tokens: list[LatinToken]) -> bool:
    return any(token.kind == "formula_letter" for token in tokens)


def _line_has_low_confidence_formula_letter_candidate(line: LineResult) -> bool:
    if not line.chars:
        return False
    has_latin_hint = any(
        len(char.text or "") == 1 and (char.text in LATIN_CANDIDATE_CHARS)
        for char in line.chars
    )
    has_low_confidence_candidate = any(
        float(char.confidence or 0.0) <= 0.25
        and len(char.text or "") == 1
        and (
            char.text in LATIN_CANDIDATE_CHARS
            or _looks_cjk_text(char.text)
        )
        for char in line.chars
    )
    return has_latin_hint and has_low_confidence_candidate


def _engcut_target_line_orders(
    lines: list[LineResult],
    block_tokens: list[LatinToken],
) -> set[int]:
    if not block_tokens:
        return {
            order
            for order, line in enumerate(lines)
            if _line_has_text_token(line)
        }

    stream_text, entries = _line_text_stream(lines)
    cursor = 0
    target_orders: set[int] = set()
    unresolved_tokens: list[LatinToken] = []
    for token_index, token in enumerate(block_tokens):
        best: tuple[int, str] | None = None
        for variant in token_variants(token.text):
            pos = stream_text.find(variant, cursor)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, variant)
        if best is None:
            unresolved_tokens.append(token)
            continue
        found, variant = best
        if _future_token_before(stream_text, cursor, found, block_tokens[token_index + 1:]):
            unresolved_tokens.append(token)
            continue
        end = found + len(variant)
        for order in entries[found:end]:
            if order is not None:
                target_orders.add(order)
        cursor = end

    # Lightweight fallback: if Hanwang has already exposed Latin/digit fragments
    # on a line, probe that line. This preserves cases such as "Gua吨lia" where
    # Paddle has "Guariglia" but Hanwang damaged the middle span before EngCut.
    has_formula_letter_tokens = _has_formula_letter_tokens(block_tokens)
    for order, line in enumerate(lines):
        if _line_has_text_token(line):
            target_orders.add(order)
            continue
        if has_formula_letter_tokens and _line_has_low_confidence_formula_letter_candidate(line):
            target_orders.add(order)
            continue
        if any(_line_may_contain_block_token(line.text, token.text) for token in unresolved_tokens):
            target_orders.add(order)
    return target_orders


def _enhance_lines_with_latin_engcut(
    image_bgr: np.ndarray,
    lines: list[LineResult],
    stats: RunStats,
    *,
    timeout: float,
    block_text: str = "",
) -> None:
    height, width = image_bgr.shape[:2]
    block_tokens = _block_text_engcut_tokens(block_text)
    target_orders = _engcut_target_line_orders(lines, block_tokens)
    records: list[_EngcutLine] = []
    for order, line in enumerate(lines):
        if not line.text or not line.chars:
            continue
        if order not in target_orders:
            continue
        x1, y1, x2, y2 = _expand_xyxy(
            line.bbox,
            width,
            height,
            pad_x=ENGCUT_LINE_CROP_PAD_X,
            pad_y=ENGCUT_LINE_CROP_PAD_Y,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        try:
            stats.latin_engcut_probe_calls += 1
            raw_eng20 = native_bridge.run_eng20_recogline(crop, timeout=timeout)
        except Exception as exc:
            stats.latin_engcut_probe_failures += 1
            logger.debug("EngCut Latin geometry probe failed bbox=%s text=%r: %s", line.bbox, line.text, exc)
            continue
        local_chars = engcut_chars_from_payload(raw_eng20)
        page_chars = offset_engcut_chars(local_chars, dx=x1, dy=y1)
        records.append(_EngcutLine(
            line=line,
            bbox=(x1, y1, x2, y2),
            chars=page_chars,
            text="".join(char.text for char in page_chars),
            order=order,
        ))

    if not records:
        return

    if not block_tokens:
        for record in records:
            exact_count = 0
            word_count = 0
            review_count = 0
            bindings = bind_latin_tokens_to_engcut_chars(record.line.text, record.chars)
            for binding in bindings:
                if binding.status == LATIN_ENGCUT_WORD_FALLBACK_STATUS:
                    token_binding = _TokenBinding(
                        token=binding.token,
                        status=binding.status,
                        matched_text=binding.token.text,
                        chars=[
                            EngcutChar(text=record.chars[binding.engcut_start + idx].text, bbox=box)
                            for idx, box in enumerate(binding.char_bboxes)
                        ],
                        line_records=[record],
                        line_span=(binding.token.start, binding.token.end),
                    )
                    if _replace_line_span_with_word_binding(record.line, token_binding):
                        word_count += 1
                    else:
                        review_count += 1
                    continue
                if binding.status != LATIN_ENGCUT_EXACT_STATUS:
                    review_count += 1
                    continue
                token_binding = _TokenBinding(
                    token=binding.token,
                    status=binding.status,
                    matched_text=binding.token.text,
                    chars=[
                        EngcutChar(text=binding.token.text[idx], bbox=box)
                        for idx, box in enumerate(binding.char_bboxes)
                    ],
                    line_records=[record],
                    line_span=(binding.token.start, binding.token.end),
                )
                fallback_reasons = _binding_word_fallback_reasons(token_binding)
                if fallback_reasons and _replace_line_span_with_word_binding(record.line, token_binding):
                    word_count += 1
                elif not fallback_reasons and _replace_line_span_with_binding(record.line, token_binding):
                    exact_count += 1
                else:
                    review_count += 1
            if review_count:
                _mark_latin_engcut_review(record.line)
            stats.latin_engcut_exact_tokens += exact_count
            stats.latin_engcut_word_tokens += word_count
            stats.latin_engcut_review_tokens += review_count
        return

    stream_text, entries = _engcut_stream(records)
    reverse = _reverse_groups(records)
    stream_cursor = 0
    reverse_cursor = 0
    exact_count = 0
    word_count = 0
    review_count = 0
    for token_idx, token in enumerate(block_tokens):
        binding, next_cursor = _bind_token_in_stream(
            token,
            block_tokens[token_idx + 1:],
            stream_text,
            entries,
            stream_cursor,
        )
        if binding is not None:
            stream_cursor = next_cursor
        else:
            binding, next_reverse = _bind_token_from_reverse_groups(token, reverse, reverse_cursor)
            if binding is not None:
                reverse_cursor = next_reverse

        if binding is None:
            continue
        if binding.status == LATIN_ENGCUT_MULTILINE_STATUS:
            for record in binding.line_records:
                _mark_latin_engcut_review(record.line)
            review_count += 1
            continue
        target_line = binding.line_records[0].line if binding.line_records else None
        if target_line is None:
            review_count += 1
            continue
        fallback_reasons = _binding_word_fallback_reasons(binding)
        if fallback_reasons and _replace_line_span_with_word_binding(target_line, binding):
            word_count += 1
        elif _binding_can_update_text(binding) and _replace_line_span_with_binding(target_line, binding):
            exact_count += 1
        else:
            _mark_latin_engcut_review(target_line)
            review_count += 1
    stats.latin_engcut_exact_tokens += exact_count
    stats.latin_engcut_word_tokens += word_count
    stats.latin_engcut_review_tokens += review_count


def _intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0, right - left) * max(0, bottom - top)


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


def _drop_cached_layout_line_routes(ppvl_blocks: list[dict]) -> None:
    for block in ppvl_blocks:
        if isinstance(block, dict):
            block.pop(LAYOUT_LINE_ROUTES_FIELD, None)


def run_micro_recblock(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict],
    *,
    seg_timeout: float = 120.0,
    recog_timeout: float = 60.0,
    include_chars: bool = True,
    page_ocr_lines: list[Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[BlockResult], RunStats]:
    """Run Hanwang Recog for text-like PP-VL blocks and keep PP-VL for others."""
    global _BATCH_DISABLED_FOR_SESSION, _BATCH_DISABLE_REASON
    height, width = image_bgr.shape[:2]
    if page_ocr_lines:
        attach_page_ocr_line_routes(ppvl_blocks, page_ocr_lines, width, height)
        _refine_layout_text_route_bands_from_image(image_bgr, ppvl_blocks, width, height)
    else:
        _drop_cached_layout_line_routes(ppvl_blocks)
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
    block_text_routes: dict[int, list[_TextRoute]] = {}
    recog_group_bboxes_by_route: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    segimg_group_audits_by_route: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for block_idx in text_indices:
        routes = _text_route_bboxes_for_block(ppvl_blocks[block_idx], block_idx, width, height)
        block_text_routes[block_idx] = routes
        text_routes.extend(routes)
        for route in routes:
            recog_group_bboxes_by_route[route.key] = []
            segimg_group_audits_by_route[route.key] = []
    text_route_recblocks = [route.bbox for route in text_routes]

    for idx in skip_indices:
        block = ppvl_blocks[idx]
        label = _effective_label_for_block(block)
        bbox = _block_bbox(block, width, height)
        layout_bbox = _layout_block_bbox(block, width, height)
        block_bbox_source = _effective_block_bbox_source(block, width, height)
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
            raw_block=_raw_block_with_bbox_audit(block, width, height, block_bbox=bbox),
        )

    if text_routes:
        recblocks = text_route_recblocks
        if progress_callback:
            progress_callback(
                0,
                max(1, len(text_routes)),
                "Hanwang micro-recblock SegImg 分块中…",
            )
        started = time.time()
        seg = native_bridge.run_linecut_segimg(
            image_bgr,
            recblocks_xyxy=recblocks,
            timeout=seg_timeout,
        )
        stats.seg_seconds = time.time() - started

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
        total_groups = max(1, len(groups))
        group_bboxes: list[tuple[int, int, int, int]] = []
        group_area_indices: list[int] = []
        for group in groups:
            recblock = recblocks[group["_area_idx"]]
            route = text_routes[group["_area_idx"]]
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
            route = text_routes[placement.area_idx]
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
                route = text_routes[placement.area_idx]
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
                route = text_routes[placement.area_idx]
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
            bbox = _block_bbox(block, width, height)
            layout_bbox = _layout_block_bbox(block, width, height)
            block_bbox_source = _effective_block_bbox_source(block, width, height)
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
            raw_text_lines: list[LineResult] = []
            for route in block_text_routes.get(block_idx, []):
                raw_text_lines.extend(grouped_lines.get(route.key, []))
            lines = [
                line for line in raw_text_lines
                if line.text or line.chars
            ]
            lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
            if _has_route_subblocks(block, width, height):
                lines = _assemble_layout_route_lines(
                    block_idx=block_idx,
                    block=block,
                    grouped_lines=grouped_lines,
                    width=width,
                    height=height,
                )
            _enhance_lines_with_latin_engcut(
                image_bgr,
                lines,
                stats,
                timeout=min(30.0, max(1.0, float(recog_timeout))),
                block_text=ppvl_text,
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
        bbox = _block_bbox(block, width, height)
        layout_bbox = _layout_block_bbox(block, width, height)
        block_bbox_source = _effective_block_bbox_source(block, width, height)
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
    model = Line(
        text=line.text,
        confidence=line.confidence,
        bbox=_bbox_from_xyxy_tuple(line.bbox, width, height),
        chars=chars,
        ocr_text=line.text,
        original_text=line.text,
        review_flags=merged_review_flags,
    )
    model.set_proof_status(proof_status_for(line.confidence, merged_review_flags))
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
    explicit = _int_value(
        raw_payload.get("_layout_paddle_parent_index", raw_payload.get("paddle_parent_index")),
    )
    if 0 <= explicit < len(records):
        return explicit

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


def _layout_row_from_block(page: Page, block: Block) -> dict[str, Any]:
    raw_payload = dict(block.raw_payload)
    app_payload = dict(block.app_payload)
    raw_payload.pop(LAYOUT_LINE_ROUTES_FIELD, None)
    app_payload.pop(LAYOUT_LINE_ROUTES_FIELD, None)
    parent_index = _parent_index_for_raw_payload(page, raw_payload)
    if parent_index < 0:
        parent_index = _int_value(
            app_payload.get("_layout_paddle_parent_index", app_payload.get("paddle_parent_index")),
        )
    binding = app_payload.get(PADDLE_BINDING_KEY)
    if parent_index < 0 and isinstance(binding, dict):
        parent_index = _int_value(binding.get("parent_index"))
    source_label = (
        authoritative_paddle_label(raw_payload)
        or str(app_payload.get("block_label") or "")
        or (
            str(binding.get("source_label") or binding.get("block_type") or "")
            if isinstance(binding, dict)
            else ""
        )
        or block.source_label
        or block.block_type.value
    )
    row = {
        **raw_payload,
        "block_label": source_label,
        "block_bbox": list(block.bbox.to_xyxy()),
        "block_content": _layout_block_content(block, raw_payload),
        "source_label": block.source_label or source_label,
        "_layout_block_source": getattr(block.source, "value", str(block.source)),
        "_layout_block_ocr_policy": block.ocr_policy.value,
    }
    if isinstance(binding, dict) and binding:
        row[PADDLE_BINDING_KEY] = dict(binding)
    for key in (ROUTE_SUBBLOCKS_FIELD,):
        if key in app_payload:
            row[key] = app_payload[key]
    records = raw_layout_records(page)
    if ROUTE_SUBBLOCKS_FIELD not in row and 0 <= parent_index < len(records):
        parent_record = records[parent_index]
        if isinstance(parent_record, dict) and ROUTE_SUBBLOCKS_FIELD in parent_record:
            row[ROUTE_SUBBLOCKS_FIELD] = parent_record[ROUTE_SUBBLOCKS_FIELD]
    if parent_index >= 0:
        row["_layout_paddle_parent_index"] = parent_index
    return row


def _strip_cached_layout_line_routes_from_page(page: Page) -> None:
    for record in raw_layout_records(page):
        if isinstance(record, dict):
            record.pop(LAYOUT_LINE_ROUTES_FIELD, None)
    for block in page.blocks:
        block.raw_payload.pop(LAYOUT_LINE_ROUTES_FIELD, None)
        block.app_payload.pop(LAYOUT_LINE_ROUTES_FIELD, None)


def _layout_block_content(block: Block, raw_payload: dict[str, Any] | None = None) -> str:
    raw_text = paddle_block_text(raw_payload or {})
    if raw_text:
        return raw_text
    text = proof_block_text(block)
    if text:
        return text
    note = str(block.note or "")
    if note in _INTERNAL_LAYOUT_NOTES:
        return ""
    return note


def _binding_payload_from_block(block: Block) -> dict[str, Any] | None:
    binding = dict(block.app_payload.get(PADDLE_BINDING_KEY) or {})
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


def _manual_bbox_from_binding(block: Block, binding: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = bbox_from_variant(binding.get("manual_bbox")) or block.bbox
    return tuple(int(value) for value in bbox.to_xyxy())


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


def _manual_binding_route_subblock(block: Block, binding: dict[str, Any]) -> dict[str, Any]:
    manual_bbox = _manual_bbox_from_binding(block, binding)
    label = str(binding.get("source_label") or block.source_label or block.block_type.value)
    text_is_stale = _manual_binding_text_is_stale(manual_bbox, binding)
    text = "" if text_is_stale else str(binding.get("text") or proof_block_text(block) or "")
    payload = {
        "block_label": label,
        "block_bbox": list(manual_bbox),
        "block_content": text,
        PADDLE_BINDING_KEY: dict(binding),
        "_layout_block_source": getattr(block.source, "value", str(block.source)),
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


def _manual_unbound_route_subblock(block: Block) -> dict[str, Any]:
    manual_bbox = tuple(int(value) for value in block.bbox.to_xyxy())
    label = block.source_label or block.block_type.value
    if block.block_type == BlockType.EQUATION and normalize_paddle_label(label) in {"", "equation", "formula"}:
        label = "inline_formula"
    return {
        "block_label": label,
        "block_bbox": list(manual_bbox),
        "block_content": proof_block_text(block),
        "_layout_block_source": getattr(block.source, "value", str(block.source)),
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
    entries: list[tuple[Block, dict[str, Any]]],
    block: Block,
    row: dict[str, Any],
    page: Page,
) -> dict[str, Any] | None:
    if block.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
        return None
    if block.source not in (BlockSource.MANUAL_DRAW, BlockSource.USER_EDITED):
        return None
    block_bbox = tuple(int(value) for value in block.bbox.to_xyxy())
    best: tuple[float, dict[str, Any]] | None = None
    for candidate_block, candidate_row in entries:
        if candidate_block is block or candidate_row is row:
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
    block: Block,
    binding: dict[str, Any],
) -> None:
    block_type = map_paddle_label_to_block_type(str(binding.get("block_type") or block.block_type.value))
    parent_index = _int_value(binding.get("parent_index"))
    records = raw_layout_records(page)
    parent_record = records[parent_index] if 0 <= parent_index < len(records) else {}
    if block_type == BlockType.EQUATION:
        parent_text = paddle_block_text(parent_record)
        if parent_text:
            parent_row["block_content"] = parent_text
            parent_row["_layout_block_content_authority"] = "paddle_parent_binding"
        route_subblock = _manual_binding_route_subblock(block, binding)
        _replace_or_append_route_subblock(
            parent_row,
            route_subblock,
            binding,
            page.width,
            page.height,
        )
        return

    manual_bbox = list(_manual_bbox_from_binding(block, binding))
    parent_row["block_bbox"] = manual_bbox
    if binding.get("text"):
        parent_row["block_content"] = str(binding.get("text") or "")
    parent_row["_layout_parent_replaced_by_manual_binding"] = True


def _apply_manual_unbound_parent_route(
    *,
    page: Page,
    parent_row: dict[str, Any],
    block: Block,
) -> None:
    route_subblock = _manual_unbound_route_subblock(block)
    _append_manual_route_subblock(parent_row, route_subblock, page.width, page.height)


def _page_blocks_from_layout(page: Page) -> list[dict]:
    entries: list[tuple[Block, dict[str, Any]]] = [
        (block, _layout_row_from_block(page, block))
        for block in page.blocks
    ]
    parent_rows: dict[int, dict[str, Any]] = {}
    for _block, row in entries:
        parent_index = _int_value(row.get("_layout_paddle_parent_index"))
        if (
            parent_index >= 0
            and parent_index not in parent_rows
            and _row_matches_paddle_parent_record(page, row, parent_index)
        ):
            parent_rows[parent_index] = row

    skip_block_ids: set[int] = set()
    for block, row in entries:
        if block.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
            continue
        binding = _binding_payload_from_block(block)
        parent_row = parent_rows.get(_int_value(binding.get("parent_index"))) if binding is not None else None
        if parent_row is None:
            parent_row = _manual_structure_parent_row(entries, block, row, page)
        if parent_row is None or parent_row is row:
            continue
        if binding is not None:
            _apply_manual_parent_binding(
                page=page,
                parent_row=parent_row,
                block=block,
                binding=binding,
            )
        else:
            _apply_manual_unbound_parent_route(
                page=page,
                parent_row=parent_row,
                block=block,
            )
        skip_block_ids.add(id(block))

    blocks: list[dict] = []
    for block in page.blocks:
        if id(block) in skip_block_ids:
            continue
        row = next(entry_row for entry_block, entry_row in entries if entry_block is block)
        blocks.append(row)
    return blocks


def _current_layout_blocks_for_ocr(page: Page) -> list[dict]:
    """Build OCR input from the current layout truth only.

    Paddle parsing records remain available as origin/reference data for binding,
    but OCR dispatch must not switch data sources based on whether a user edited
    the page.
    """
    return _page_blocks_from_layout(page)


def _page_ocr_lines_from_layout(page: Page) -> list[Line]:
    return [
        line
        for block in page.blocks
        for line in block.lines
        if line.bbox is not None
        and line.bbox.area > 0
        and is_ppocr_page_line_hint(line)
    ]


def _routed_manual_structure_blocks(page: Page) -> list[Block]:
    """Manual structural boxes consumed by parent routing must survive OCR writeback."""
    entries: list[tuple[Block, dict[str, Any]]] = [
        (block, _layout_row_from_block(page, block))
        for block in page.blocks
    ]
    parent_rows: dict[int, dict[str, Any]] = {}
    for _block, row in entries:
        parent_index = _int_value(row.get("_layout_paddle_parent_index"))
        if (
            parent_index >= 0
            and parent_index not in parent_rows
            and _row_matches_paddle_parent_record(page, row, parent_index)
        ):
            parent_rows[parent_index] = row

    preserved: list[Block] = []
    for block, row in entries:
        if block.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
            continue
        if block.source not in (BlockSource.MANUAL_DRAW, BlockSource.USER_EDITED):
            continue
        binding = _binding_payload_from_block(block)
        parent_row = parent_rows.get(_int_value(binding.get("parent_index"))) if binding is not None else None
        if parent_row is None:
            parent_row = _manual_structure_parent_row(entries, block, row, page)
        if parent_row is None or parent_row is row:
            continue
        preserved.append(block)
    return preserved


def _inline_formula_crop_ocr_targets(page: Page) -> list[Block]:
    targets: list[Block] = []
    for block in page.blocks:
        if block.block_type != BlockType.EQUATION:
            continue
        label = normalize_paddle_label(
            block.source_label
            or str(block.app_payload.get("block_label") or "")
            or str(block.raw_payload.get("block_label") or "")
        )
        if label not in {"inline_formula", "formula"}:
            continue
        if block.bbox is None or block.bbox.area <= 0:
            continue
        targets.append(block)
    targets.sort(key=lambda item: (item.bbox.y, item.bbox.x, item.order))
    return targets


def _existing_parent_index(block: Block) -> int:
    binding = block.app_payload.get(PADDLE_BINDING_KEY)
    if isinstance(binding, dict):
        parent_index = _int_value(binding.get("parent_index"))
        if parent_index >= 0:
            return parent_index
    return _int_value(block.app_payload.get("_layout_paddle_parent_index", block.raw_payload.get("_layout_paddle_parent_index")))


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
    block.source_label = "inline_formula"
    block.ocr_policy = OcrPolicy.PRESERVE_AS_FORMULA
    set_payload_entries(block, {
        PADDLE_BINDING_KEY: binding,
        PADDLE_BLOCK_LABEL_KEY: "inline_formula",
        PADDLE_BLOCK_BBOX_KEY: bbox_xyxy,
    })
    block.app_payload.pop(OCR_TEXT_INVALIDATED_KEY, None)
    block.app_payload.pop(OCR_INVALIDATION_KIND_KEY, None)
    block.lines = [
        Line(
            text=text,
            confidence=1.0,
            bbox=block.bbox,
            ocr_text=text,
            original_text=text,
            review_flags=[FORMULA_CROP_OCR_REVIEW_FLAG],
        )
    ]


def _mark_inline_formula_needs_text(block: Block, reason: str = "") -> None:
    bbox_xyxy = [int(value) for value in block.bbox.to_xyxy()]
    parent_index = _existing_parent_index(block)
    flags = ["manual_formula_needs_text"]
    if reason:
        flags.append(FORMULA_CROP_OCR_FAILED_FLAG)
    block.source_label = "inline_formula"
    block.ocr_policy = OcrPolicy.PRESERVE_AS_FORMULA
    set_payload_entries(block, {
        PADDLE_BINDING_KEY: {
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
        },
        PADDLE_BLOCK_LABEL_KEY: "inline_formula",
        PADDLE_BLOCK_BBOX_KEY: bbox_xyxy,
    })
    block.lines = [
        Line(
            text="",
            confidence=0.0,
            bbox=block.bbox,
            ocr_text="",
            original_text="",
            review_flags=flags,
        )
    ]


def _mark_formula_crop_ocr_unavailable(blocks: list[Block]) -> None:
    for block in blocks:
        binding = block.app_payload.get(PADDLE_BINDING_KEY)
        binding_source = str(binding.get("source") or "") if isinstance(binding, dict) else ""
        invalidated = bool(block.app_payload.get(OCR_TEXT_INVALIDATED_KEY))
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
    ) -> RunStats:
        _strip_cached_layout_line_routes_from_page(page)
        self._refresh_inline_formula_texts_from_current_crops(
            image_bgr,
            page,
            progress_callback=progress_callback,
        )
        preserved_manual_blocks = _routed_manual_structure_blocks(page)
        layout_blocks = _current_layout_blocks_for_ocr(page)
        ppvl_blocks = deepcopy(layout_blocks)
        if not ppvl_blocks:
            raise RuntimeError("Hanwang micro_recblock requires PP-VL parsing_res_list blocks")

        rows, stats = self._runner(
            image_bgr,
            ppvl_blocks,
            seg_timeout=self._seg_timeout,
            recog_timeout=self._recog_timeout,
            include_chars=True,
            page_ocr_lines=_page_ocr_lines_from_layout(page),
            progress_callback=progress_callback,
        )

        new_blocks: list[Block] = []
        height, width = image_bgr.shape[:2]
        for order, row in enumerate(rows):
            bbox = _bbox_from_xyxy_tuple(row.block_bbox, width, height)
            block_type = map_paddle_label_to_block_type(row.block_label)
            flags: list[str] = []
            if (
                block_type == BlockType.EQUATION
                and not row.text
                and str(row.raw_block.get("_layout_block_source") or "") in {
                    BlockSource.MANUAL_DRAW.value,
                    BlockSource.USER_EDITED.value,
                }
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
            note_parts = [
                f"source_label={row.block_label}",
                f"micro_recblock_source={row.source}",
            ]
            if row.ppvl_text:
                note_parts.append(f"ppvl_text={row.ppvl_text[:120]}")
            raw_payload, app_payload = split_legacy_raw_payload(row.raw_block)
            raw_payload = strip_runtime_layout_payload(raw_payload)
            app_payload = strip_runtime_layout_payload(app_payload)
            audit = app_payload.get(HANWANG_BBOX_AUDIT_KEY)
            if isinstance(audit, dict):
                failed_groups = int(audit.get("hanwang_recog_group_failed_count") or 0)
                if failed_groups:
                    note_parts.append(f"hanwang_recog_group_failed={failed_groups}")
                if audit.get("paddle_label_unknown"):
                    note_parts.append(f"unknown_paddle_label={row.block_label}")
            new_block = Block(
                block_type=block_type,
                bbox=bbox,
                lines=lines,
                order=order,
                source=BlockSource.AUTO_LAYOUT,
                note=" | ".join(note_parts),
                source_label=row.block_label,
                raw_payload=raw_payload,
                app_payload=app_payload,
            )
            new_block.ocr_policy = default_ocr_policy_for_block(new_block)
            if row.source != "hanwang" and new_block.ocr_policy == OcrPolicy.TEXT_OCR:
                new_block.ocr_policy = OcrPolicy.MANUAL_ONLY
            new_blocks.append(new_block)

        if preserved_manual_blocks:
            new_blocks.extend(preserved_manual_blocks)
        for order, block in enumerate(new_blocks):
            block.order = order
        page.blocks = new_blocks
        logger.info(
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d fallback=%d "
            "unknown_labels=%d groups=%d group_failures=%d chunks=%d guarded_chunks=%d "
            "batch_failures=%d batch_disabled=%s "
            "seg=%.2fs recog=%.2fs "
            "max_batch_crop=%dx%d probe_calls=%d recog_pixels=%d/%d "
            "latin_engcut_calls=%d latin_engcut_failures=%d latin_engcut_exact=%d "
            "latin_engcut_review=%d latin_engcut_disabled=%s "
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
            stats.latin_engcut_probe_calls,
            stats.latin_engcut_probe_failures,
            stats.latin_engcut_exact_tokens,
            stats.latin_engcut_review_tokens,
            stats.latin_engcut_disabled,
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
            from app.experimental.inline_formula_rebind import (
                build_formula_pseudo_page,
                recognize_formula_pseudo_page,
            )

            bboxes = [tuple(int(value) for value in block.bbox.to_xyxy()) for block in targets]
            pseudo_page = build_formula_pseudo_page(image_bgr, bboxes)
            recognitions, _response = recognize_formula_pseudo_page(
                pseudo_page,
                client=client,
                batch_id=f"ocr-process-formula-{page.page_number}-{int(time.time() * 1000)}",
                filename=f"page-{page.page_number}-inline-formula-pseudo.png",
            )
        except Exception as exc:
            logger.warning("Paddle formula crop OCR failed for page=%s: %s", page.page_number, exc)
            _mark_formula_crop_ocr_failed(targets, str(exc))
            if progress_callback:
                progress_callback(0, max(1, len(targets)), f"Paddle 公式重识别失败：{exc}")
            return

        text_by_index = {item.crop_index: item.text for item in recognitions if str(item.text or "").strip()}
        for index, block in enumerate(targets):
            text = str(text_by_index.get(index) or "").strip()
            if text:
                _set_inline_formula_crop_ocr_text(block, text)
            else:
                _mark_inline_formula_needs_text(block)
        if progress_callback:
            progress_callback(
                len(text_by_index),
                max(1, len(targets)),
                f"Paddle 公式重识别完成：{len(text_by_index)}/{len(targets)}",
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
