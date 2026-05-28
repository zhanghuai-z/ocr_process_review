"""Layout-stage Paddle routing helpers for Hanwang text-slice OCR."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.bbox_extraction import bbox_from_variant
from app.core.ocr_ir import is_formula_token
from app.core.paddle_labels import authoritative_paddle_label, is_hanwang_skip_label, normalize_paddle_label
from app.models import BlockType

ROUTE_SUBBLOCKS_FIELD = "_route_subblocks"
LAYOUT_LINE_ROUTES_FIELD = "_layout_line_routes"
ROUTE_INLINE_FORMULA_FLAG = "hanwang_route_inline_formula"
ROUTE_TABLE_FLAG = "hanwang_route_table"
_FORMULA_STYLE_POSITION_LABELS = {"footer", "footnote"}


@dataclass(frozen=True)
class PaddleRouteLineHint:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class RecoveredInlineFormulaSegment:
    line_index: int
    text: str
    bbox: tuple[int, int, int, int]
    label: str
    review_flag: str = ROUTE_INLINE_FORMULA_FLAG


@dataclass(frozen=True)
class PageOcrLineHint:
    text: str
    bbox: tuple[int, int, int, int]


def route_authority_label(block: dict[str, Any], default: str = "unknown") -> str:
    return normalize_paddle_label(authoritative_paddle_label(block, default))


def block_text(block: dict[str, Any]) -> str:
    return str(
        block.get("block_content")
        or block.get("text")
        or block.get("content")
        or ""
    ).strip()


def is_formula_style_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if ("\\" + "begin{") in stripped:
        return True
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return bool(stripped[2:-2].strip())
    if stripped.startswith("$") and stripped.endswith("$"):
        return bool(stripped[1:-1].strip())
    compact = "".join(ch for ch in stripped if not ch.isspace())
    return is_formula_token(compact)


def is_formula_style_position_block(block: dict[str, Any]) -> bool:
    label = normalize_paddle_label(authoritative_paddle_label(block))
    if label not in _FORMULA_STYLE_POSITION_LABELS:
        return False
    return is_formula_style_text(block_text(block))


def is_formula_label(label: str) -> bool:
    return BlockType.from_paddle(label) == BlockType.EQUATION


def is_table_label(label: str) -> bool:
    return BlockType.from_paddle(label) == BlockType.TABLE


def clamp_xyxy(
    raw: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = raw
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(x1, min(int(x2), width))
    y2 = max(y1, min(int(y2), height))
    return x1, y1, x2, y2


def block_bbox_xyxy(
    block: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    bbox = bbox_from_variant(block, max_w=width, max_h=height)
    if bbox is None:
        return 0, 0, width, height
    return clamp_xyxy(bbox.to_xyxy(), width, height)


def intersect_xyxy(
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


def subtract_xyxy(
    rect: tuple[int, int, int, int],
    cut: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    overlap = intersect_xyxy(rect, cut)
    if overlap is None:
        return [rect]
    x1, y1, x2, y2 = rect
    ox1, oy1, ox2, oy2 = overlap
    pieces = [
        (x1, y1, x2, oy1),
        (x1, oy2, x2, y2),
        (x1, oy1, ox1, oy2),
        (ox2, oy1, x2, oy2),
    ]
    return [
        piece
        for piece in pieces
        if piece[2] - piece[0] >= 4 and piece[3] - piece[1] >= 4
    ]


def union_xyxy(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def vertical_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1, min(a[3] - a[1], b[3] - b[1]))
    return overlap / denom


def _horizontal_gap_segments(
    line_bbox: tuple[int, int, int, int],
    subblocks: list[dict[str, Any]],
    formula_texts_by_bbox: dict[tuple[int, int, int, int], str],
) -> list[dict[str, Any]]:
    lx1, ly1, lx2, ly2 = line_bbox
    cuts: list[dict[str, Any]] = []
    for subblock in subblocks:
        sub_bbox = tuple(subblock["bbox"])
        if vertical_overlap_ratio(line_bbox, sub_bbox) < 0.25:
            continue
        overlap = intersect_xyxy(line_bbox, sub_bbox)
        if overlap is None:
            continue
        ox1, _oy1, ox2, _oy2 = overlap
        if ox2 <= ox1:
            continue
        kind = "formula" if is_formula_label(subblock["label"]) else "skip"
        cuts.append(
            {
                "kind": kind,
                "label": subblock["label"],
                "bbox": (ox1, ly1, ox2, ly2),
                "text": formula_texts_by_bbox.get(sub_bbox, subblock.get("text", "")),
            }
        )
    cuts.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    if not cuts:
        return [{"kind": "text", "bbox": line_bbox}]

    segments: list[dict[str, Any]] = []
    cursor = lx1
    for cut in cuts:
        cx1, cy1, cx2, cy2 = cut["bbox"]
        if cx1 > cursor:
            segments.append({"kind": "text", "bbox": (cursor, ly1, cx1, ly2)})
        segments.append({**cut, "bbox": (max(cx1, lx1), cy1, min(cx2, lx2), cy2)})
        cursor = max(cursor, cx2)
    if cursor < lx2:
        segments.append({"kind": "text", "bbox": (cursor, ly1, lx2, ly2)})
    return [
        segment for segment in segments
        if segment["bbox"][2] - segment["bbox"][0] >= 4 and segment["bbox"][3] - segment["bbox"][1] >= 4
    ]


def _line_hint_from_value(value: Any, width: int, height: int) -> PageOcrLineHint | None:
    if isinstance(value, PageOcrLineHint):
        bbox_value = value.bbox
        text = value.text
    elif isinstance(value, dict):
        bbox_value = value.get("bbox") or value.get("line_bbox") or value.get("coordinate")
        text = str(value.get("text") or value.get("ocr_text") or "")
    else:
        bbox_value = getattr(value, "bbox", None)
        text = str(getattr(value, "text", "") or getattr(value, "ocr_text", "") or "")
    if hasattr(bbox_value, "to_xyxy"):
        raw_bbox = bbox_value.to_xyxy()
    else:
        bbox = bbox_from_variant(bbox_value, max_w=width, max_h=height)
        if bbox is None:
            return None
        raw_bbox = bbox.to_xyxy()
    bbox_tuple = clamp_xyxy(raw_bbox, width, height)
    if bbox_tuple[2] <= bbox_tuple[0] or bbox_tuple[3] <= bbox_tuple[1]:
        return None
    return PageOcrLineHint(text=text, bbox=bbox_tuple)


def _line_assignment_score(
    line_bbox: tuple[int, int, int, int],
    block_bbox: tuple[int, int, int, int],
) -> float:
    overlap = intersect_xyxy(line_bbox, block_bbox)
    if overlap is None:
        lx1, ly1, lx2, ly2 = line_bbox
        cx = (lx1 + lx2) / 2
        cy = (ly1 + ly2) / 2
        bx1, by1, bx2, by2 = block_bbox
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return 0.1
        return 0.0
    area = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])
    line_area = max(1, (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1]))
    return area / line_area


def attach_page_ocr_line_routes(
    ppvl_blocks: list[dict[str, Any]],
    page_ocr_lines: list[Any],
    width: int,
    height: int,
) -> None:
    line_hints = [
        hint for value in page_ocr_lines
        if (hint := _line_hint_from_value(value, width, height)) is not None
    ]
    if not line_hints:
        return

    parent_entries: list[tuple[int, dict[str, Any], tuple[int, int, int, int]]] = []
    for block_idx, block in enumerate(ppvl_blocks):
        label = route_authority_label(block)
        if is_formula_label(label) or is_table_label(label) or is_hanwang_skip_label(label):
            continue
        parent_entries.append((block_idx, block, block_bbox_xyxy(block, width, height)))

    assigned: dict[int, list[PageOcrLineHint]] = {block_idx: [] for block_idx, _block, _bbox in parent_entries}
    for line in line_hints:
        best_idx: int | None = None
        best_score = 0.0
        for block_idx, _block, block_bbox in parent_entries:
            score = _line_assignment_score(line.bbox, block_bbox)
            if score > best_score:
                best_score = score
                best_idx = block_idx
        if best_idx is not None and best_score >= 0.1:
            assigned[best_idx].append(line)

    for block_idx, block, _block_bbox in parent_entries:
        lines = sorted(assigned.get(block_idx, []), key=lambda item: (item.bbox[1], item.bbox[0]))
        if not lines:
            continue
        subblocks = route_subblocks_for_block(block, width, height)
        formula_subblocks = [subblock for subblock in subblocks if is_formula_label(subblock["label"])]
        line_hints_for_formula = [PaddleRouteLineHint(text=line.text, bbox=line.bbox) for line in lines]
        recovered = recover_inline_formula_segments(
            parent_text=block_text(block),
            line_hints=line_hints_for_formula,
            subblocks=[
                {"label": subblock["label"], "bbox": subblock["bbox"]}
                for subblock in formula_subblocks
            ],
        )
        formula_texts_by_bbox = {segment.bbox: segment.text for segment in recovered}
        routes = [
            _build_route_line(_horizontal_gap_segments(line.bbox, subblocks, formula_texts_by_bbox))
            for line in lines
        ]
        if routes:
            block[LAYOUT_LINE_ROUTES_FIELD] = routes


def _route_subblock_text(subblock: dict[str, Any]) -> str:
    raw = subblock.get("raw") or {}
    raw_payload = raw.get("raw_payload") if isinstance(raw, dict) else None
    if isinstance(raw_payload, dict):
        text = block_text(raw_payload)
        if text:
            return text
    if isinstance(raw, dict):
        text = block_text(raw)
        if text:
            return text
    return ""


def route_subblocks_for_block(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    values = block.get(ROUTE_SUBBLOCKS_FIELD)
    if not isinstance(values, list):
        return []
    parent_bbox = block_bbox_xyxy(block, width, height)
    subblocks: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        label = route_authority_label(value)
        bbox = intersect_xyxy(parent_bbox, block_bbox_xyxy(value, width, height))
        if bbox is None:
            continue
        subblocks.append(
            {
                "label": label,
                "bbox": bbox,
                "raw": dict(value),
                "text": _route_subblock_text(value),
            }
        )
    subblocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return subblocks


def _formula_spans(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"\$.*?\$", text, re.DOTALL)]


def _nearest_line_index(
    bbox: tuple[int, int, int, int],
    line_hints: list[PaddleRouteLineHint],
) -> int:
    if not line_hints:
        return 0
    fx1, fy1, fx2, fy2 = bbox
    fcx = (fx1 + fx2) / 2
    fcy = (fy1 + fy2) / 2
    best_index = 0
    best_score = float("-inf")
    for index, hint in enumerate(line_hints):
        lx1, ly1, lx2, ly2 = hint.bbox
        lcx = (lx1 + lx2) / 2
        lcy = (ly1 + ly2) / 2
        overlap = vertical_overlap_ratio(bbox, hint.bbox)
        center_inside = 1.0 if ly1 <= fcy <= ly2 else 0.0
        horizontal_distance = 0.0 if lx1 <= fcx <= lx2 else min(abs(fcx - lx1), abs(fcx - lx2))
        score = overlap * 1000.0 + center_inside * 100.0 - abs(fcy - lcy) - horizontal_distance * 0.05
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def recover_inline_formula_segments(
    *,
    parent_text: str,
    line_hints: list[PaddleRouteLineHint],
    subblocks: list[dict[str, Any]],
    is_formula_label: Any = is_formula_label,
) -> list[RecoveredInlineFormulaSegment]:
    formula_spans = _formula_spans(parent_text)
    recovered: list[RecoveredInlineFormulaSegment] = []
    formula_cursor = 0
    formula_items: list[tuple[str, tuple[int, int, int, int]]] = []
    for subblock in subblocks:
        label = str(subblock.get("label") or subblock.get("block_label") or "")
        if not is_formula_label(label):
            continue
        bbox_value = subblock.get("bbox") or subblock.get("block_bbox")
        bbox = bbox_from_variant(bbox_value)
        if bbox is None:
            continue
        formula_items.append((label, bbox.to_xyxy()))

    formula_items.sort(key=lambda item: (item[1][1], item[1][0]))
    for label, bbox in formula_items:
        text = formula_spans[formula_cursor] if formula_cursor < len(formula_spans) else ""
        formula_cursor += 1
        if not text:
            continue
        recovered.append(
            RecoveredInlineFormulaSegment(
                line_index=_nearest_line_index(bbox, line_hints),
                text=text,
                bbox=bbox,
                label=label,
            )
        )
    recovered.sort(key=lambda item: (item.line_index, item.bbox[0], item.bbox[1]))
    return recovered


def _build_route_line(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bbox": list(union_xyxy([tuple(segment["bbox"]) for segment in segments])),
        "segments": [
            {
                "kind": segment["kind"],
                "label": segment.get("label", ""),
                "bbox": list(segment["bbox"]),
                "text": segment.get("text", ""),
            }
            for segment in segments
        ],
    }


def build_layout_line_routes(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    parent_bbox = block_bbox_xyxy(block, width, height)
    subblocks = route_subblocks_for_block(block, width, height)
    if not subblocks:
        return []

    text_rects = [parent_bbox]
    for subblock in subblocks:
        next_rects: list[tuple[int, int, int, int]] = []
        for rect in text_rects:
            next_rects.extend(subtract_xyxy(rect, subblock["bbox"]))
        text_rects = next_rects

    formula_subblocks = [item for item in subblocks if is_formula_label(item["label"])]
    skip_subblocks = [item for item in subblocks if not is_formula_label(item["label"])]

    bucket_candidates = [
        {"kind": "text", "bbox": rect}
        for rect in sorted(text_rects, key=lambda item: (item[1], item[0]))
        if rect[2] > rect[0] and rect[3] > rect[1]
    ]
    bucket_candidates.extend(
        {
            "kind": "formula",
            "label": item["label"],
            "bbox": item["bbox"],
            "text": item["text"],
        }
        for item in formula_subblocks
    )
    bucket_candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    buckets: list[list[dict[str, Any]]] = []
    for segment in bucket_candidates:
        for bucket in buckets:
            head_bbox = tuple(bucket[0]["bbox"])
            if vertical_overlap_ratio(head_bbox, tuple(segment["bbox"])) >= 0.5:
                bucket.append(segment)
                break
        else:
            buckets.append([segment])

    formula_spans = _formula_spans(block_text(block))
    formula_cursor = 0
    routes = []
    for bucket in buckets:
        bucket.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
        for segment in bucket:
            if segment["kind"] != "formula":
                continue
            if formula_cursor < len(formula_spans):
                segment["text"] = formula_spans[formula_cursor]
            formula_cursor += 1
        routes.append(_build_route_line(bucket))

    for subblock in skip_subblocks:
        label = subblock["label"]
        route = _build_route_line(
            [
                {
                    "kind": "skip",
                    "label": label,
                    "bbox": subblock["bbox"],
                    "text": subblock["text"],
                }
            ]
        )
        routes.append(route)

    routes.sort(
        key=lambda item: (
            0 if any(segment.get("kind") == "formula" for segment in item.get("segments", [])) else 1,
            item["bbox"][1],
            item["bbox"][0],
        )
    )
    return routes


def _normalize_cached_line_routes(
    routes: object,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not isinstance(routes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        segments = route.get("segments")
        if not isinstance(segments, list):
            continue
        normalized_segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            bbox = block_bbox_xyxy({"block_bbox": segment.get("bbox")}, width, height)
            normalized_segments.append(
                {
                    "kind": str(segment.get("kind") or "").strip() or "text",
                    "label": str(segment.get("label") or ""),
                    "bbox": list(bbox),
                    "text": str(segment.get("text") or ""),
                }
            )
        if normalized_segments:
            normalized.append(_build_route_line(normalized_segments))
    return normalized


def line_routes_for_block(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    cached = _normalize_cached_line_routes(block.get(LAYOUT_LINE_ROUTES_FIELD), width, height)
    if cached:
        block[LAYOUT_LINE_ROUTES_FIELD] = cached
        return cached
    routes = build_layout_line_routes(block, width, height)
    if routes:
        block[LAYOUT_LINE_ROUTES_FIELD] = routes
    return routes


def text_slice_routes_for_block(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    routes = line_routes_for_block(block, width, height)
    if not routes:
        return [
            {
                "line_idx": -1,
                "segment_idx": 0,
                "bbox": list(block_bbox_xyxy(block, width, height)),
                "carved": False,
            }
        ]
    slices = []
    for line_idx, route in enumerate(routes):
        for segment_idx, segment in enumerate(route.get("segments", [])):
            if segment.get("kind") != "text":
                continue
            slices.append(
                {
                    "line_idx": line_idx,
                    "segment_idx": segment_idx,
                    "bbox": list(block_bbox_xyxy({"block_bbox": segment.get("bbox")}, width, height)),
                    "carved": True,
                }
            )
    slices.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    slices.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return slices


def has_layout_line_routes(
    block: dict[str, Any],
    width: int,
    height: int,
) -> bool:
    return bool(line_routes_for_block(block, width, height))


__all__ = [
    "LAYOUT_LINE_ROUTES_FIELD",
    "PaddleRouteLineHint",
    "PageOcrLineHint",
    "ROUTE_INLINE_FORMULA_FLAG",
    "ROUTE_SUBBLOCKS_FIELD",
    "ROUTE_TABLE_FLAG",
    "RecoveredInlineFormulaSegment",
    "attach_page_ocr_line_routes",
    "block_bbox_xyxy",
    "block_text",
    "build_layout_line_routes",
    "clamp_xyxy",
    "has_layout_line_routes",
    "intersect_xyxy",
    "is_formula_label",
    "is_formula_style_position_block",
    "is_formula_style_text",
    "is_table_label",
    "line_routes_for_block",
    "recover_inline_formula_segments",
    "route_authority_label",
    "route_subblocks_for_block",
    "subtract_xyxy",
    "text_slice_routes_for_block",
    "union_xyxy",
    "vertical_overlap_ratio",
]
