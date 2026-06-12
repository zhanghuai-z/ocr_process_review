"""Page-level PP-VL block -> Hanwang linecut micro-recblock integration."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.bbox_extraction import bbox_from_variant
from app.core.block_payload import HANWANG_BBOX_AUDIT_KEY, PADDLE_BINDING_KEY, split_legacy_raw_payload
from app.core.logging import get_logger
from app.core.latin_span_recovery import (
    LATIN_ENGCUT_BBOX_GRANULARITY,
    LATIN_ENGCUT_BBOX_SOURCE,
    LATIN_ENGCUT_EXACT_STATUS,
    LATIN_ENGCUT_REVIEW_FLAG,
    bind_latin_tokens_to_engcut_chars,
    engcut_chars_from_payload,
    has_latin_token,
    offset_engcut_chars,
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
from app.core.proof_status import proof_status_for
from app.engines import OCR_BBOX_SPACE_PAGE
from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page

from . import native_bridge

logger = get_logger(__name__)

TEXT_LABELS: set[str] = set(PADDLE_HANWANG_TEXT_LABELS)
SKIP_LABELS: set[str] = set(PADDLE_HANWANG_SKIP_LABELS)

MAX_RECOG_BATCH_GROUPS = 6
MAX_RECOG_COLLAGE_WIDTH = 1600
MAX_RECOG_COLLAGE_HEIGHT = 1600
MAX_RECOG_COLLAGE_PIXELS = 2_000_000
MAX_RECOG_COLLAGE_ASPECT = 4.5
_BATCH_DISABLED_FOR_SESSION = False
_BATCH_DISABLE_REASON = ""
_LATIN_ENGCUT_DISABLED_FOR_SESSION = False
_LATIN_ENGCUT_DISABLE_REASON = ""


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
    recog_batch_chunks: int = 0
    recog_batch_failures: int = 0
    recog_batch_disabled: bool = False
    recog_batch_guarded_chunks: int = 0
    recog_max_collage_width: int = 0
    recog_max_collage_height: int = 0
    recog_max_collage_pixels: int = 0
    latin_engcut_probe_calls: int = 0
    latin_engcut_probe_failures: int = 0
    latin_engcut_exact_tokens: int = 0
    latin_engcut_review_tokens: int = 0
    latin_engcut_disabled: bool = False


@dataclass
class _GroupPlacement:
    area_idx: int
    page_bbox: tuple[int, int, int, int]
    collage_bbox: tuple[int, int, int, int]


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
    if BlockType.from_paddle(label) == BlockType.EQUATION:
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
                flags.add(ROUTE_INLINE_FORMULA_FLAG)
        merged_text = "".join(text_parts)
        if not merged_text:
            continue
        assembled.append(
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


def _apply_latin_engcut_geometry(line: LineResult, raw_eng20: dict[str, Any], *, dx: int, dy: int) -> tuple[int, int]:
    local_chars = engcut_chars_from_payload(raw_eng20)
    page_chars = offset_engcut_chars(local_chars, dx=dx, dy=dy)
    bindings = bind_latin_tokens_to_engcut_chars(line.text, page_chars)
    if not bindings:
        return 0, 0

    offset_map = _line_char_offset_map(line)
    exact_count = 0
    review_count = 0
    for binding in bindings:
        if binding.status != LATIN_ENGCUT_EXACT_STATUS:
            review_count += 1
            continue
        if offset_map is None or binding.token.end > len(offset_map):
            review_count += 1
            continue
        char_indices = offset_map[binding.token.start:binding.token.end]
        if len(char_indices) != len(binding.char_bboxes) or len(set(char_indices)) != len(char_indices):
            review_count += 1
            continue
        if any(line.chars[index].text != binding.token.text[offset] for offset, index in enumerate(char_indices)):
            review_count += 1
            continue

        for offset, char_index in enumerate(char_indices):
            char = line.chars[char_index]
            char.bbox = binding.char_bboxes[offset]
            char.source = LATIN_ENGCUT_BBOX_SOURCE
            char.bbox_granularity = LATIN_ENGCUT_BBOX_GRANULARITY
            char.token_text = binding.token.text
        exact_count += 1

    if review_count:
        _mark_latin_engcut_review(line)
    return exact_count, review_count


def _enhance_lines_with_latin_engcut(
    image_bgr: np.ndarray,
    lines: list[LineResult],
    stats: RunStats,
    *,
    timeout: float,
) -> None:
    height, width = image_bgr.shape[:2]
    for line in lines:
        if not line.text or not line.chars or not has_latin_token(line.text):
            continue
        x1, y1, x2, y2 = _clamp_xyxy(line.bbox, width, height)
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
        exact_count, review_count = _apply_latin_engcut_geometry(line, raw_eng20, dx=x1, dy=y1)
        stats.latin_engcut_exact_tokens += exact_count
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


def _placement_for_line(
    line: LineResult,
    placements: list[_GroupPlacement],
) -> _GroupPlacement | None:
    if not placements:
        return None
    x1, y1, x2, y2 = line.bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    for placement in placements:
        left, top, right, bottom = placement.collage_bbox
        if left <= cx <= right and top <= cy <= bottom:
            return placement
    return max(
        placements,
        key=lambda placement: _intersection_area(line.bbox, placement.collage_bbox),
    )


def _build_group_collage(
    image_bgr: np.ndarray,
    group_bboxes: list[tuple[int, int, int, int]],
    area_indices: list[int],
) -> tuple[np.ndarray, list[_GroupPlacement]]:
    if not group_bboxes:
        return image_bgr[:0, :0].copy(), []
    gap = 2
    crops: list[np.ndarray] = []
    max_width = 1
    total_height = 0
    for bbox in group_bboxes:
        left, top, right, bottom = bbox
        crop = image_bgr[top:bottom, left:right].copy()
        crops.append(crop)
        crop_h, crop_w = crop.shape[:2]
        max_width = max(max_width, crop_w)
        total_height += crop_h
    total_height += gap * max(0, len(crops) - 1)
    if image_bgr.ndim == 2:
        collage = np.full((total_height, max_width), 255, dtype=image_bgr.dtype)
    else:
        collage = np.full((total_height, max_width, image_bgr.shape[2]), 255, dtype=image_bgr.dtype)
    placements: list[_GroupPlacement] = []
    y = 0
    for crop, page_bbox, area_idx in zip(crops, group_bboxes, area_indices):
        crop_h, crop_w = crop.shape[:2]
        collage[y:y + crop_h, 0:crop_w] = crop
        placements.append(
            _GroupPlacement(
                area_idx=area_idx,
                page_bbox=page_bbox,
                collage_bbox=(0, y, crop_w, y + crop_h),
            )
        )
        y += crop_h + gap
    return collage, placements


def _estimate_collage_shape(
    group_bboxes: list[tuple[int, int, int, int]],
    *,
    gap: int = 2,
) -> tuple[int, int, int]:
    if not group_bboxes:
        return 0, 0, 0
    width = max(max(0, right - left) for left, _top, right, _bottom in group_bboxes)
    height = sum(max(0, bottom - top) for _left, top, _right, bottom in group_bboxes)
    height += gap * max(0, len(group_bboxes) - 1)
    return width, height, width * height


def _chunk_group_bboxes(
    group_bboxes: list[tuple[int, int, int, int]],
    area_indices: list[int],
    *,
    max_groups: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_pixels: int | None = None,
    max_aspect: float | None = None,
) -> list[tuple[list[tuple[int, int, int, int]], list[int]]]:
    if max_groups is None:
        max_groups = MAX_RECOG_BATCH_GROUPS
    if max_width is None:
        max_width = MAX_RECOG_COLLAGE_WIDTH
    if max_height is None:
        max_height = MAX_RECOG_COLLAGE_HEIGHT
    if max_pixels is None:
        max_pixels = MAX_RECOG_COLLAGE_PIXELS
    if max_aspect is None:
        max_aspect = MAX_RECOG_COLLAGE_ASPECT
    chunks: list[tuple[list[tuple[int, int, int, int]], list[int]]] = []
    current_bboxes: list[tuple[int, int, int, int]] = []
    current_areas: list[int] = []
    for bbox, area_idx in zip(group_bboxes, area_indices):
        candidate_bboxes = [*current_bboxes, bbox]
        width, height, pixels = _estimate_collage_shape(candidate_bboxes)
        aspect = (width / height) if height > 0 else 0.0
        over_limit = (
            len(candidate_bboxes) > max_groups
            or width > max_width
            or height > max_height
            or pixels > max_pixels
            or aspect > max_aspect
        )
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
    has_layout_routes = any(_has_route_subblocks(ppvl_blocks[idx], width, height) for idx in text_indices)

    for idx in skip_indices:
        block = ppvl_blocks[idx]
        label = _effective_label_for_block(block)
        bbox = _block_bbox(block, width, height)
        layout_bbox = _layout_block_bbox(block, width, height)
        block_bbox_source = _effective_block_bbox_source(block, width, height)
        ppvl_text = _block_text(block)
        synthesize_chars = BlockType.from_paddle(label) != BlockType.EQUATION
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
            segimg_group_audits_by_route.setdefault(route.key, []).append({
                "route_text_slice_bbox": list(recblock),
                "segimg_group_bbox": list(raw_group_bbox),
                "recog_group_bbox": list(bbox) if bbox is not None else None,
                "clipped": bbox is not None and bbox != raw_group_bbox,
                "dropped": bbox is None,
            })
            if bbox is None:
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            group_bboxes.append(bbox)
            group_area_indices.append(group["_area_idx"])
            recog_group_bboxes_by_route.setdefault(route.key, []).append(bbox)

        def mark_recog_group_failure(placement: _GroupPlacement, error: Exception) -> None:
            stats.recog_group_failures += 1
            route = text_routes[placement.area_idx]
            audits = segimg_group_audits_by_route.setdefault(route.key, [])
            for item in audits:
                if item.get("recog_group_bbox") == list(placement.page_bbox):
                    item["recog_failed"] = True
                    item["recog_error"] = str(error)
                    return
            audits.append({
                "route_text_slice_bbox": list(route.bbox),
                "segimg_group_bbox": list(placement.page_bbox),
                "recog_group_bbox": list(placement.page_bbox),
                "clipped": False,
                "dropped": False,
                "recog_failed": True,
                "recog_error": str(error),
            })

        def recognize_individually(placements: list[_GroupPlacement]) -> None:
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crop = image_bgr[top:bottom, left:right].copy()
                crop_h, crop_w = crop.shape[:2]
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        crop,
                        recblock_xyxy=None,
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    logger.warning("Hanwang micro_recblock group failed bbox=%s: %s", placement.page_bbox, exc)
                    mark_recog_group_failure(placement, exc)
                    raw = {}
                local_lines = _line_results_from_recog(
                    raw,
                    fallback_bbox=(0, 0, crop_w, crop_h),
                    include_chars=include_chars,
                )
                grouped_lines[text_routes[placement.area_idx].key].extend(
                    _offset_line_results(local_lines, dx=left, dy=top)
                )

        batch_enabled = not _BATCH_DISABLED_FOR_SESSION and not has_layout_routes
        stats.recog_batch_disabled = _BATCH_DISABLED_FOR_SESSION or has_layout_routes
        completed_groups = 0
        chunks = _chunk_group_bboxes(group_bboxes, group_area_indices)
        for chunk_index, (chunk_bboxes, chunk_area_indices) in enumerate(chunks):
            collage, placements = _build_group_collage(image_bgr, chunk_bboxes, chunk_area_indices)
            if not placements:
                continue
            stats.recog_batch_chunks += 1
            collage_h, collage_w = collage.shape[:2]
            collage_pixels = collage_w * collage_h
            stats.recog_max_collage_width = max(stats.recog_max_collage_width, collage_w)
            stats.recog_max_collage_height = max(stats.recog_max_collage_height, collage_h)
            stats.recog_max_collage_pixels = max(stats.recog_max_collage_pixels, collage_pixels)
            stats.recog_full_page_pixels += width * height * len(placements)
            stats.recog_crop_pixels += sum(
                (placement.page_bbox[2] - placement.page_bbox[0])
                * (placement.page_bbox[3] - placement.page_bbox[1])
                for placement in placements
            )
            collage_aspect = (collage_w / collage_h) if collage_h > 0 else 0.0
            shape_guarded = (
                collage_w > MAX_RECOG_COLLAGE_WIDTH
                or collage_h > MAX_RECOG_COLLAGE_HEIGHT
                or collage_pixels > MAX_RECOG_COLLAGE_PIXELS
                or collage_aspect > MAX_RECOG_COLLAGE_ASPECT
            )
            if shape_guarded:
                stats.recog_batch_guarded_chunks += 1
            use_batch = batch_enabled and len(placements) > 1 and not shape_guarded
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    (
                        "Hanwang micro-recblock 分块批量识别中… "
                        f"chunk {chunk_index + 1}/{len(chunks)}, groups={len(placements)}, "
                        f"collage={collage_w}x{collage_h}"
                    ),
                )
            if use_batch:
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        collage,
                        recblocks_xyxy=[placement.collage_bbox for placement in placements],
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    stats.recog_batch_failures += 1
                    stats.recog_batch_disabled = True
                    batch_enabled = False
                    _BATCH_DISABLED_FOR_SESSION = True
                    _BATCH_DISABLE_REASON = (
                        f"chunk={chunk_index + 1} groups={len(placements)} "
                        f"collage={collage_w}x{collage_h}: {exc}"
                    )
                    logger.warning(
                        "Hanwang micro_recblock batch failed; disabling batch for this process "
                        "chunk=%d groups=%d collage=%dx%d: %s",
                        chunk_index + 1,
                        len(placements),
                        collage_w,
                        collage_h,
                        exc,
                    )
                    recognize_individually(placements)
                else:
                    local_lines = _line_results_from_recog(
                        raw,
                        fallback_bbox=(0, 0, collage_w, collage_h),
                        include_chars=include_chars,
                        fallback_empty=False,
                    )
                    for local_line in local_lines:
                        placement = _placement_for_line(local_line, placements)
                        if placement is None:
                            continue
                        dx = placement.page_bbox[0] - placement.collage_bbox[0]
                        dy = placement.page_bbox[1] - placement.collage_bbox[1]
                        grouped_lines[text_routes[placement.area_idx].key].extend(
                            _offset_line_results([local_line], dx=dx, dy=dy)
                        )
            else:
                recognize_individually(placements)

            completed_groups += len(placements)
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    f"Hanwang micro-recblock 已完成 {completed_groups}/{total_groups} groups",
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
            )
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

    return [row for row in rows if row is not None], stats


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
        final_text=line.text,
        confidence=line.confidence,
        bbox=_bbox_from_xyxy_tuple(line.bbox, width, height),
        chars=chars,
        ocr_text=line.text,
        original_text=line.text,
        review_flags=merged_review_flags,
        proof_status=proof_status_for(line.confidence, merged_review_flags),
    )
    return model


def _page_layout_has_user_edits(page: Page) -> bool:
    return any(
        block.source in (BlockSource.MANUAL_DRAW, BlockSource.USER_EDITED)
        for block in page.blocks
    )


_PARENT_BINDING_STATUSES = {
    BINDING_GEOMETRY_HIT,
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
    if not raw_payload or not page.ppvl_parsing_res_list:
        return -1
    explicit = _int_value(
        raw_payload.get("_layout_paddle_parent_index", raw_payload.get("paddle_parent_index")),
    )
    if 0 <= explicit < len(page.ppvl_parsing_res_list):
        return explicit

    label = route_authority_label(raw_payload)
    bbox = _payload_bbox_xyxy(raw_payload, page.width, page.height)
    text = paddle_block_text(raw_payload)
    for index, record in enumerate(page.ppvl_parsing_res_list):
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
        "block_content": _layout_block_content(block),
        "source_label": block.source_label or source_label,
        "_layout_block_source": getattr(block.source, "value", str(block.source)),
        "_layout_block_recognizable": bool(block.recognizable),
    }
    if isinstance(binding, dict) and binding:
        row[PADDLE_BINDING_KEY] = dict(binding)
    for key in (ROUTE_SUBBLOCKS_FIELD, LAYOUT_LINE_ROUTES_FIELD):
        if key in app_payload:
            row[key] = app_payload[key]
    if parent_index >= 0:
        row["_layout_paddle_parent_index"] = parent_index
    return row


def _layout_block_content(block: Block) -> str:
    text = block.full_text
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
    text = str(binding.get("text") or block.full_text or "")
    return {
        "block_label": label,
        "block_bbox": list(manual_bbox),
        "block_content": text,
        PADDLE_BINDING_KEY: dict(binding),
        "_layout_block_source": getattr(block.source, "value", str(block.source)),
        "_layout_manual_route_subblock": True,
    }


def _cached_route_bands(
    row: dict[str, Any],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    routes = row.get(LAYOUT_LINE_ROUTES_FIELD)
    if not isinstance(routes, list):
        return []
    bands: list[tuple[int, int, int, int]] = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        bbox = bbox_from_variant(route.get("bbox"), max_w=width, max_h=height)
        if bbox is None or bbox.area <= 0:
            continue
        bands.append(_clamp_xyxy(bbox.to_xyxy(), width, height))
    return bands


def _best_cached_band_for_bbox(
    bbox: tuple[int, int, int, int],
    bands: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not bands:
        return None
    _x1, y1, _x2, y2 = bbox
    cy = (y1 + y2) / 2
    best_band: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for band in bands:
        overlap = vertical_overlap_ratio(bbox, band)
        center_inside = 1.0 if band[1] <= cy <= band[3] else 0.0
        score = overlap + center_inside * 0.25
        if score > best_score:
            best_score = score
            best_band = band
    return best_band if best_score >= 0.1 else None


def _expand_formula_subblock_to_cached_band(
    value: dict[str, Any],
    bands: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> dict[str, Any]:
    if not is_formula_label(route_authority_label(value)):
        return value
    bbox = block_bbox_xyxy(value, width, height)
    band = _best_cached_band_for_bbox(bbox, bands)
    if band is None:
        return value
    expanded = dict(value)
    expanded["block_bbox"] = [bbox[0], band[1], bbox[2], band[3]]
    return expanded


def _expand_formula_subblocks_to_cached_bands(
    row: dict[str, Any],
    width: int,
    height: int,
) -> None:
    bands = _cached_route_bands(row, width, height)
    if not bands:
        return
    values = row.get(ROUTE_SUBBLOCKS_FIELD)
    if not isinstance(values, list):
        return
    row[ROUTE_SUBBLOCKS_FIELD] = [
        _expand_formula_subblock_to_cached_band(dict(value), bands, width, height)
        if isinstance(value, dict)
        else value
        for value in values
    ]


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
        if target_bbox is not None and _route_subblock_overlaps_bbox(value, target_bbox, width, height):
            next_values.append(route_subblock)
            replaced = True
            continue
        if _route_subblock_overlaps_bbox(value, manual_bbox, width, height):
            next_values.append(route_subblock)
            replaced = True
            continue
        next_values.append(value)
    if not replaced:
        next_values.append(route_subblock)
    row[ROUTE_SUBBLOCKS_FIELD] = next_values
    row.pop(LAYOUT_LINE_ROUTES_FIELD, None)


def _apply_manual_parent_binding(
    *,
    page: Page,
    parent_row: dict[str, Any],
    block: Block,
    binding: dict[str, Any],
) -> None:
    block_type = BlockType.from_paddle(str(binding.get("block_type") or block.block_type.value))
    parent_index = _int_value(binding.get("parent_index"))
    parent_record = page.ppvl_parsing_res_list[parent_index] if 0 <= parent_index < len(page.ppvl_parsing_res_list) else {}
    if block_type == BlockType.EQUATION:
        parent_text = paddle_block_text(parent_record)
        if parent_text:
            parent_row["block_content"] = parent_text
            parent_row["_layout_block_content_authority"] = "paddle_parent_binding"
        _expand_formula_subblocks_to_cached_bands(parent_row, page.width, page.height)
        route_subblock = _expand_formula_subblock_to_cached_band(
            _manual_binding_route_subblock(block, binding),
            _cached_route_bands(parent_row, page.width, page.height),
            page.width,
            page.height,
        )
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


def _page_blocks_from_layout(page: Page) -> list[dict]:
    entries: list[tuple[Block, dict[str, Any]]] = [
        (block, _layout_row_from_block(page, block))
        for block in page.blocks
    ]
    parent_rows: dict[int, dict[str, Any]] = {}
    for _block, row in entries:
        parent_index = _int_value(row.get("_layout_paddle_parent_index"))
        if parent_index >= 0 and parent_index not in parent_rows:
            parent_rows[parent_index] = row

    skip_block_ids: set[int] = set()
    for block, row in entries:
        if block.block_type not in (BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE):
            continue
        binding = _binding_payload_from_block(block)
        if binding is None:
            continue
        parent_row = parent_rows.get(_int_value(binding.get("parent_index")))
        if parent_row is None or parent_row is row:
            continue
        _apply_manual_parent_binding(
            page=page,
            parent_row=parent_row,
            block=block,
            binding=binding,
        )
        skip_block_ids.add(id(block))

    blocks: list[dict] = []
    for block in page.blocks:
        if id(block) in skip_block_ids:
            continue
        row = next(entry_row for entry_block, entry_row in entries if entry_block is block)
        blocks.append(row)
    return blocks


def _page_ocr_lines_from_layout(page: Page) -> list[Line]:
    return [
        line
        for block in page.blocks
        for line in block.lines
        if line.bbox is not None and line.bbox.area > 0
    ]


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
        runner=run_micro_recblock,
    ) -> None:
        self._seg_timeout = seg_timeout
        self._recog_timeout = recog_timeout
        self._runner = runner

    def recognize_page_blocks(
        self,
        image_bgr: np.ndarray,
        page: Page,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> RunStats:
        ppvl_blocks = (
            _page_blocks_from_layout(page)
            if _page_layout_has_user_edits(page)
            else (page.ppvl_parsing_res_list or _page_blocks_from_layout(page))
        )
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
            block_type = BlockType.from_paddle(row.block_label)
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
            audit = app_payload.get(HANWANG_BBOX_AUDIT_KEY)
            if isinstance(audit, dict):
                failed_groups = int(audit.get("hanwang_recog_group_failed_count") or 0)
                if failed_groups:
                    note_parts.append(f"hanwang_recog_group_failed={failed_groups}")
                if audit.get("paddle_label_unknown"):
                    note_parts.append(f"unknown_paddle_label={row.block_label}")
            new_blocks.append(
                Block(
                    block_type=block_type,
                    bbox=bbox,
                    lines=lines,
                    order=order,
                    source=BlockSource.AUTO_LAYOUT,
                    recognizable=row.source == "hanwang",
                    note=" | ".join(note_parts),
                    source_label=row.block_label,
                    raw_payload=raw_payload,
                    app_payload=app_payload,
                )
            )

        page.blocks = new_blocks
        logger.info(
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d fallback=%d "
            "unknown_labels=%d groups=%d group_failures=%d chunks=%d guarded_chunks=%d "
            "batch_failures=%d batch_disabled=%s "
            "max_collage=%dx%d probe_calls=%d recog_pixels=%d/%d "
            "latin_engcut_calls=%d latin_engcut_failures=%d latin_engcut_exact=%d "
            "latin_engcut_review=%d latin_engcut_disabled=%s",
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
            stats.recog_max_collage_width,
            stats.recog_max_collage_height,
            stats.recog_probe_calls,
            stats.recog_crop_pixels,
            stats.recog_full_page_pixels,
            stats.latin_engcut_probe_calls,
            stats.latin_engcut_probe_failures,
            stats.latin_engcut_exact_tokens,
            stats.latin_engcut_review_tokens,
            stats.latin_engcut_disabled,
        )
        return stats


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
