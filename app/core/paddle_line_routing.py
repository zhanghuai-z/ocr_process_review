"""Layout-stage Paddle routing helpers for Hanwang text-slice OCR."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.adapters.paddle import map_paddle_label_to_block_type
from app.core.bbox_extraction import bbox_from_variant
from app.core.ocr_ir import is_formula_marker_token, is_formula_token
from app.core.paddle_labels import is_hanwang_skip_label, normalize_paddle_label
from app.core.paddle_layout_schema import paddle_record_label, paddle_record_text
from app.models import BlockType

ROUTE_SUBBLOCKS_FIELD = "_route_subblocks"
LAYOUT_LINE_ROUTES_FIELD = "_layout_line_routes"
LAYOUT_ROUTE_SOURCE_FIELD = "_layout_route_source"
LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS = "ppocr_page_line_hints"
ROUTE_INLINE_FORMULA_FLAG = "hanwang_route_inline_formula"
ROUTE_TABLE_FLAG = "hanwang_route_table"
_FORMULA_STYLE_POSITION_LABELS = {"footer"}
_ROUTE_FORMULA_LABELS = {
    "display_formula",
    "equation",
    "equation_block",
    "formula",
    "inline_formula",
    "isolated_formula",
}
_FORMULA_PATTERN = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)
_CONTEXT_MATCH_WINDOW = 18
_CONTEXT_MATCH_THRESHOLD = 4
_FORMULA_COMMAND_ALIASES = {
    "alpha": ("alpha", "α"),
    "beta": ("beta", "β"),
    "gamma": ("gamma", "γ"),
    "delta": ("delta", "δ"),
    "epsilon": ("epsilon", "ε"),
    "varepsilon": ("varepsilon", "epsilon", "ε"),
    "phi": ("phi", "φ"),
    "varphi": ("varphi", "phi", "φ"),
}
_PHYSICAL_LINE_MIN_VERTICAL_OVERLAP = 0.25
_PHYSICAL_LINE_MAX_CENTER_GAP_RATIO = 0.65
_PHYSICAL_LINE_MAX_CENTER_GAP_PX = 18
_INTER_FORMULA_TEXT_GAP_MIN_WIDTH = 16


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
    return normalize_paddle_label(paddle_record_label(block, default))


def block_text(block: dict[str, Any]) -> str:
    return paddle_record_text(block, include_markdown=False)


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
    label = normalize_paddle_label(paddle_record_label(block))
    if label not in _FORMULA_STYLE_POSITION_LABELS:
        return False
    return is_formula_style_text(block_text(block))


def is_formula_label(label: str) -> bool:
    return map_paddle_label_to_block_type(label) == BlockType.EQUATION


def _is_route_formula_label(label: str) -> bool:
    return normalize_paddle_label(label) in _ROUTE_FORMULA_LABELS


def is_table_label(label: str) -> bool:
    return map_paddle_label_to_block_type(label) == BlockType.TABLE


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
    parent_text: str = "",
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
        kind = "formula" if _is_route_formula_label(subblock["label"]) else "skip"
        text = formula_texts_by_bbox.get(sub_bbox, subblock.get("text", ""))
        if kind == "formula" and is_formula_marker_token(text):
            continue
        bbox = sub_bbox if kind == "formula" else (ox1, ly1, ox2, ly2)
        cuts.append(
            {
                "kind": kind,
                "label": subblock["label"],
                "bbox": bbox,
                "text": text,
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
    segments = _restore_expected_inter_formula_text_gaps(
        line_bbox,
        segments,
        cuts,
        parent_text,
    )
    return [
        segment for segment in segments
        if segment["bbox"][2] - segment["bbox"][0] >= 4 and segment["bbox"][3] - segment["bbox"][1] >= 4
    ]


def _restore_expected_inter_formula_text_gaps(
    line_bbox: tuple[int, int, int, int],
    segments: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    parent_text: str,
) -> list[dict[str, Any]]:
    formula_cuts = [
        cut for cut in cuts
        if cut.get("kind") == "formula" and str(cut.get("text") or "").strip()
    ]
    if len(formula_cuts) < 2 or not parent_text:
        return segments

    span_matches = [
        match for match in _FORMULA_PATTERN.finditer(parent_text or "")
        if not is_formula_marker_token(match.group(0))
    ]
    if len(span_matches) < 2:
        return segments

    matched_indices = _match_formula_cuts_to_parent_spans(formula_cuts, span_matches)
    if len(matched_indices) != len(formula_cuts):
        return segments

    restored = list(segments)
    for left_idx, right_idx in zip(range(len(formula_cuts) - 1), range(1, len(formula_cuts))):
        left_span_index = matched_indices[left_idx]
        right_span_index = matched_indices[right_idx]
        if right_span_index != left_span_index + 1:
            continue
        between_text = parent_text[span_matches[left_span_index].end():span_matches[right_span_index].start()]
        if not _has_visible_inter_formula_text(between_text):
            continue
        left_bbox = tuple(formula_cuts[left_idx]["bbox"])
        right_bbox = tuple(formula_cuts[right_idx]["bbox"])
        gap_bbox = _inter_formula_text_gap_bbox(line_bbox, left_bbox, right_bbox)
        if gap_bbox is None:
            continue
        if _text_segment_covers_gap(restored, gap_bbox):
            continue
        restored.append(
            {
                "kind": "text",
                "label": "inter_formula_text_gap",
                "bbox": gap_bbox,
                "text": between_text.strip(),
            }
        )
    restored.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], 0 if item.get("kind") == "formula" else 1))
    return restored


def _match_formula_cuts_to_parent_spans(
    formula_cuts: list[dict[str, Any]],
    span_matches: list[Any],
) -> list[int]:
    matched: list[int] = []
    cursor = 0
    for cut in formula_cuts:
        text = str(cut.get("text") or "").strip()
        if not text:
            return []
        found = -1
        for index in range(cursor, len(span_matches)):
            if span_matches[index].group(0) == text:
                found = index
                break
        if found < 0:
            return []
        matched.append(found)
        cursor = found + 1
    return matched


def _has_visible_inter_formula_text(text: str) -> bool:
    return any(not ch.isspace() for ch in str(text or ""))


def _inter_formula_text_gap_bbox(
    line_bbox: tuple[int, int, int, int],
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    lx1, ly1, lx2, ly2 = line_bbox
    natural_left = max(lx1, min(left_bbox[2], right_bbox[0]))
    natural_right = min(lx2, max(left_bbox[2], right_bbox[0]))
    if natural_right - natural_left >= 4:
        center = (natural_left + natural_right) // 2
    else:
        center = (left_bbox[2] + right_bbox[0]) // 2
    half = max(_INTER_FORMULA_TEXT_GAP_MIN_WIDTH // 2, (natural_right - natural_left) // 2)
    x1 = max(lx1, center - half)
    x2 = min(lx2, center + half)
    if x2 - x1 < 4:
        return None
    return (x1, ly1, x2, ly2)


def _text_segment_covers_gap(
    segments: list[dict[str, Any]],
    gap_bbox: tuple[int, int, int, int],
) -> bool:
    gap_center = (gap_bbox[0] + gap_bbox[2]) / 2.0
    for segment in segments:
        if segment.get("kind") != "text":
            continue
        bbox = tuple(segment["bbox"])
        if vertical_overlap_ratio(bbox, gap_bbox) < 0.25:
            continue
        if bbox[0] <= gap_center <= bbox[2] and bbox[2] - bbox[0] >= 4:
            return True
    return False


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


def _line_height(bbox: tuple[int, int, int, int]) -> int:
    return max(1, int(bbox[3]) - int(bbox[1]))


def _line_center_y(bbox: tuple[int, int, int, int]) -> float:
    return (int(bbox[1]) + int(bbox[3])) / 2.0


def _same_physical_line(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    if vertical_overlap_ratio(a, b) >= _PHYSICAL_LINE_MIN_VERTICAL_OVERLAP:
        return True
    max_center_gap = max(
        _PHYSICAL_LINE_MAX_CENTER_GAP_PX,
        int(round(max(_line_height(a), _line_height(b)) * _PHYSICAL_LINE_MAX_CENTER_GAP_RATIO)),
    )
    return abs(_line_center_y(a) - _line_center_y(b)) <= max_center_gap


def _bbox_only_physical_line_hints(
    line_hints: list[PageOcrLineHint],
) -> list[PageOcrLineHint]:
    """Collapse PP-OCRv5 fragments into physical row hints.

    PP-OCRv5 may return superscripts, subscripts, inline formulas, and ordinary
    text chunks as separate OCR lines.  Routing only trusts the merged
    page-space row geometry.  The concatenated text is retained only as weak
    context for mapping Paddle parent formula spans back to row geometry; it is
    never emitted as Hanwang OCR text.
    """
    buckets: list[list[PageOcrLineHint]] = []
    for hint in sorted(line_hints, key=lambda item: (_line_center_y(item.bbox), item.bbox[0])):
        for bucket in buckets:
            if any(_same_physical_line(item.bbox, hint.bbox) for item in bucket):
                bucket.append(hint)
                break
        else:
            buckets.append([hint])

    merged: list[PageOcrLineHint] = []
    for bucket in buckets:
        bbox = union_xyxy([item.bbox for item in bucket])
        text = "".join(
            str(item.text or "")
            for item in sorted(bucket, key=lambda value: (value.bbox[0], value.bbox[1]))
        )
        merged.append(PageOcrLineHint(text=text, bbox=bbox))
    merged.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return merged


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
    line_hints = _bbox_only_physical_line_hints(line_hints)

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

    for block_idx, block, block_bbox in parent_entries:
        lines = []
        for line in assigned.get(block_idx, []):
            clipped_bbox = intersect_xyxy(line.bbox, block_bbox)
            if clipped_bbox is None:
                continue
            lines.append(PageOcrLineHint(text=line.text, bbox=clipped_bbox))
        lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        if not lines:
            block.pop(LAYOUT_LINE_ROUTES_FIELD, None)
            continue
        subblocks = route_subblocks_for_block(block, width, height)
        formula_subblocks = [subblock for subblock in subblocks if _is_route_formula_label(subblock["label"])]
        parent_formula_texts_by_bbox = _formula_texts_by_bbox(formula_subblocks, block_text(block))
        routed_subblocks = [
            subblock
            for subblock in subblocks
            if (
                not _is_route_formula_label(subblock["label"])
                or not is_formula_marker_token(
                    parent_formula_texts_by_bbox.get(tuple(subblock["bbox"]), subblock.get("text", ""))
                )
            )
        ]
        formula_subblocks = [subblock for subblock in routed_subblocks if _is_route_formula_label(subblock["label"])]
        line_hints_for_formula = [
            PaddleRouteLineHint(
                text=line.text,
                bbox=line.bbox,
            )
            for line in lines
        ]
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
            _build_route_line(
                _horizontal_gap_segments(
                    line.bbox,
                    routed_subblocks,
                    formula_texts_by_bbox,
                    block_text(block),
                ),
                source=LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS,
            )
            for line in lines
        ]
        if routes:
            block[LAYOUT_LINE_ROUTES_FIELD] = routes


def _route_subblock_text(subblock: dict[str, Any]) -> str:
    text = block_text(subblock)
    if text:
        return text
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
    return [match.group(0) for match in _FORMULA_PATTERN.finditer(text)]


def _route_formula_spans(text: str) -> list[str]:
    return [span for span in _formula_spans(text) if not is_formula_marker_token(span)]


def _formula_span_records(text: str) -> list[dict[str, Any]]:
    matches = list(_FORMULA_PATTERN.finditer(text or ""))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        if is_formula_marker_token(match.group(0)):
            continue
        prev_start = matches[index - 1].end() if index > 0 else 0
        next_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        records.append(
            {
                "text": match.group(0),
                "prev_context": text[prev_start:match.start()],
                "next_context": text[match.end():next_end],
            }
        )
    return records


def _normalize_context_text(text: str) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _context_match_score(fragment: str, line_text: str) -> float:
    if not fragment or not line_text:
        return 0.0
    if fragment in line_text:
        return float(len(fragment) * 2)
    limit = min(len(fragment), len(line_text), _CONTEXT_MATCH_WINDOW)
    for size in range(limit, _CONTEXT_MATCH_THRESHOLD - 1, -1):
        if fragment[:size] in line_text or fragment[-size:] in line_text:
            return float(size)
    return 0.0


def _formula_text_candidates(span: str) -> list[str]:
    text = str(span or "").strip().strip("$").strip()
    candidates: list[str] = []
    for command, aliases in _FORMULA_COMMAND_ALIASES.items():
        if ("\\" + command) in text:
            candidates.extend(aliases)
    for token in re.findall(r"[A-Za-z]+|[0-9]+", text):
        candidates.append(token)
    compact = _normalize_context_text(re.sub(r"\\[A-Za-z]+", "", text))
    if compact:
        candidates.append(compact)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = _normalize_context_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _formula_match_score(span: str, line_text: str) -> float:
    best = 0.0
    for candidate in _formula_text_candidates(span):
        if candidate in line_text:
            best = max(best, float(max(6, len(candidate) * 2)))
        else:
            best = max(best, _context_match_score(candidate, line_text))
    return best


def _infer_formula_spans_by_line(
    parent_text: str,
    line_hints: list[PaddleRouteLineHint],
    formula_box_counts_by_line: dict[int, int],
) -> dict[int, list[str]]:
    """Infer which physical OCR line each parent LaTeX span belongs to.

    PP-OCR/Hanwang line text usually does not preserve ``$...$`` formula
    delimiters, so direct ``_formula_spans(line.text)`` is often empty.  We use
    non-formula context around each parent span instead and keep inline formula
    boxes as word/token-level geometry evidence.  The box counts are used only
    as a tie-breaker so one missing Paddle box does not shift all later spans.
    """
    if not line_hints:
        return {}
    line_texts = [_normalize_context_text(hint.text) for hint in line_hints]
    if not any(line_texts):
        return {}

    spans_by_line: dict[int, list[str]] = {}
    assigned_counts: dict[int, int] = {}
    for record in _formula_span_records(parent_text):
        prev_tail = _normalize_context_text(record["prev_context"])[-_CONTEXT_MATCH_WINDOW:]
        next_head = _normalize_context_text(record["next_context"])[:_CONTEXT_MATCH_WINDOW]
        chosen: int | None = None
        best_score = 0.0
        for index, line_text in enumerate(line_texts):
            score = (
                _context_match_score(prev_tail, line_text)
                + _context_match_score(next_head, line_text)
                + _formula_match_score(str(record["text"]), line_text) * 2.0
            )
            capacity = formula_box_counts_by_line.get(index, 0)
            if capacity > 0 and assigned_counts.get(index, 0) >= capacity:
                score -= 8.0
            if score > best_score:
                best_score = score
                chosen = index

        if best_score < _CONTEXT_MATCH_THRESHOLD:
            chosen = None

        if chosen is None:
            continue
        spans_by_line.setdefault(chosen, []).append(str(record["text"]))
        assigned_counts[chosen] = assigned_counts.get(chosen, 0) + 1
    return spans_by_line


def _reading_order_by_row(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: (value["bbox"][1], value["bbox"][0])):
        item_bbox = tuple(item["bbox"])
        for bucket in buckets:
            bucket_bbox = union_xyxy([tuple(value["bbox"]) for value in bucket])
            if vertical_overlap_ratio(bucket_bbox, item_bbox) >= 0.25:
                bucket.append(item)
                break
        else:
            buckets.append([item])

    ordered: list[dict[str, Any]] = []
    for bucket in sorted(buckets, key=lambda values: (min(value["bbox"][1] for value in values), min(value["bbox"][0] for value in values))):
        ordered.extend(sorted(bucket, key=lambda value: (value["bbox"][0], value["bbox"][1])))
    return ordered


def _formula_texts_by_bbox(
    formula_subblocks: list[dict[str, Any]],
    parent_text: str,
) -> dict[tuple[int, int, int, int], str]:
    formula_spans = _formula_spans(parent_text)
    texts_by_bbox: dict[tuple[int, int, int, int], str] = {}
    ordered = _reading_order_by_row(formula_subblocks)
    guessable: list[dict[str, Any]] = []
    for item in ordered:
        bbox = tuple(item["bbox"])
        explicit_text = str(item.get("text") or "").strip()
        if explicit_text:
            texts_by_bbox[bbox] = explicit_text
            continue
        if _formula_subblock_allows_parent_text_guess(item):
            guessable.append(item)
    if len(guessable) == len(formula_spans):
        for index, item in enumerate(guessable):
            texts_by_bbox.setdefault(tuple(item["bbox"]), formula_spans[index])
    return texts_by_bbox


def _formula_subblock_allows_parent_text_guess(item: dict[str, Any]) -> bool:
    raw = item.get("raw")
    if not isinstance(raw, dict):
        return True
    if raw.get("_layout_manual_route_subblock"):
        return False
    if raw.get("_layout_manual_binding_text_stale"):
        return False
    if raw.get("_layout_manual_unbound_route_subblock"):
        return False
    binding = raw.get("paddle_binding")
    if isinstance(binding, dict):
        source = str(binding.get("source") or "")
        if "parent_text" in source or source == "paddle_geometry":
            return False
    return True


def formula_texts_by_subblock_bbox(
    block: dict[str, Any],
    width: int,
    height: int,
) -> dict[tuple[int, int, int, int], str]:
    formula_subblocks = [
        subblock
        for subblock in route_subblocks_for_block(block, width, height)
        if _is_route_formula_label(subblock["label"])
    ]
    return _formula_texts_by_bbox(formula_subblocks, block_text(block))


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
    is_formula_label: Any = _is_route_formula_label,
) -> list[RecoveredInlineFormulaSegment]:
    parent_spans = _route_formula_spans(parent_text)
    parent_text_by_bbox: dict[tuple[int, int, int, int], str] = {}
    recovered: list[RecoveredInlineFormulaSegment] = []
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

    ordered_formula_items = [
        (item["label"], tuple(item["bbox"]))
        for item in _reading_order_by_row([
            {"label": label, "bbox": bbox}
            for label, bbox in formula_items
        ])
    ]
    all_parent_spans = _formula_spans(parent_text)
    marker_bboxes = {
        bbox
        for index, (_label, bbox) in enumerate(ordered_formula_items)
        if len(ordered_formula_items) == len(all_parent_spans)
        and index < len(all_parent_spans)
        and is_formula_marker_token(all_parent_spans[index])
    }
    if marker_bboxes:
        ordered_formula_items = [
            (label, bbox)
            for label, bbox in ordered_formula_items
            if bbox not in marker_bboxes
        ]
    for index, (_label, bbox) in enumerate(ordered_formula_items):
        if index < len(parent_spans):
            parent_text_by_bbox[bbox] = parent_spans[index]

    formula_items_by_line: dict[int, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for label, bbox in ordered_formula_items:
        formula_items_by_line.setdefault(_nearest_line_index(bbox, line_hints), []).append((label, bbox))
    formula_box_counts_by_line = {
        line_index: len(items)
        for line_index, items in formula_items_by_line.items()
    }
    global_fallback_is_safe = (
        not line_hints or len(parent_spans) == len(ordered_formula_items)
    )
    inferred_spans_by_line = (
        {}
        if global_fallback_is_safe
        else _infer_formula_spans_by_line(
            parent_text,
            line_hints,
            formula_box_counts_by_line,
        )
    )

    for line_index in sorted(formula_items_by_line):
        line_items = sorted(formula_items_by_line[line_index], key=lambda item: (item[1][0], item[1][1]))
        if global_fallback_is_safe:
            # Complete formula geometry is authoritative: PP-OCRv5 contributes
            # line geometry only; formula text comes from the Paddle parent.
            line_spans = []
        else:
            direct_line_spans = _route_formula_spans(line_hints[line_index].text) if line_index < len(line_hints) else []
            inferred_line_spans = inferred_spans_by_line.get(line_index, [])
            if len(direct_line_spans) >= len(line_items):
                line_spans = direct_line_spans
            elif len(inferred_line_spans) == len(line_items):
                line_spans = inferred_line_spans
            else:
                line_spans = []
        for local_index, (label, bbox) in enumerate(line_items):
            text = line_spans[local_index] if local_index < len(line_spans) else ""
            if not text and global_fallback_is_safe:
                text = parent_text_by_bbox.get(bbox, "")
            if not text:
                continue
            recovered.append(
                RecoveredInlineFormulaSegment(
                    line_index=line_index,
                    text=text,
                    bbox=bbox,
                    label=label,
                )
            )
    recovered.sort(key=lambda item: (item.line_index, item.bbox[0], item.bbox[1]))
    return recovered


def _build_route_line(
    segments: list[dict[str, Any]],
    *,
    source: str = "",
) -> dict[str, Any]:
    route = {
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
    if source:
        route[LAYOUT_ROUTE_SOURCE_FIELD] = source
    return route


def _route_has_formula(route: dict[str, Any]) -> bool:
    return any(segment.get("kind") == "formula" for segment in route.get("segments", []))


def _filter_thin_text_artifact_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formula_heights = [
        int(route["bbox"][3]) - int(route["bbox"][1])
        for route in routes
        if _route_has_formula(route)
    ]
    if not formula_heights:
        return routes
    formula_heights.sort()
    median_formula_height = formula_heights[len(formula_heights) // 2]
    min_text_height = max(8, int(round(median_formula_height * 0.6)))

    filtered: list[dict[str, Any]] = []
    for route in routes:
        if _route_has_formula(route):
            filtered.append(route)
            continue
        segments = route.get("segments", [])
        if not segments or any(segment.get("kind") != "text" for segment in segments):
            filtered.append(route)
            continue
        height = int(route["bbox"][3]) - int(route["bbox"][1])
        if height >= min_text_height:
            filtered.append(route)
    return filtered


def build_layout_line_routes(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    parent_bbox = block_bbox_xyxy(block, width, height)
    subblocks = route_subblocks_for_block(block, width, height)
    if not subblocks:
        return []

    all_formula_subblocks = [item for item in subblocks if _is_route_formula_label(item["label"])]
    formula_texts_by_bbox = _formula_texts_by_bbox(all_formula_subblocks, block_text(block))
    marker_formula_subblocks: list[dict[str, Any]] = []
    routed_subblocks: list[dict[str, Any]] = []
    for subblock in subblocks:
        is_formula = _is_route_formula_label(subblock["label"])
        is_marker = is_formula and is_formula_marker_token(
            formula_texts_by_bbox.get(tuple(subblock["bbox"]), subblock.get("text", ""))
        )
        if is_marker:
            marker_formula_subblocks.append(subblock)
            continue
        routed_subblocks.append(subblock)

    text_rects = [parent_bbox]
    for subblock in routed_subblocks:
        next_rects: list[tuple[int, int, int, int]] = []
        for rect in text_rects:
            next_rects.extend(subtract_xyxy(rect, subblock["bbox"]))
        text_rects = next_rects

    formula_subblocks = [item for item in routed_subblocks if _is_route_formula_label(item["label"])]
    skip_subblocks = [item for item in routed_subblocks if not _is_route_formula_label(item["label"])]

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
    if formula_subblocks:
        bucket_candidates.extend(
            {
                "kind": "anchor",
                "label": item["label"],
                "bbox": item["bbox"],
                "text": formula_texts_by_bbox.get(tuple(item["bbox"]), item.get("text", "")),
            }
            for item in marker_formula_subblocks
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

    formula_texts_by_bbox = {
        bbox: text
        for bbox, text in formula_texts_by_bbox.items()
        if not is_formula_marker_token(text)
    }
    routes = []
    for bucket in buckets:
        bucket.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
        if any(segment["kind"] == "formula" for segment in bucket):
            row_bbox = union_xyxy([tuple(segment["bbox"]) for segment in bucket])
            row_subblocks = [
                subblock
                for subblock in formula_subblocks
                if vertical_overlap_ratio(row_bbox, subblock["bbox"]) >= 0.25
            ]
            routes.append(
                _build_route_line(
                    _horizontal_gap_segments(
                        row_bbox,
                        row_subblocks,
                        formula_texts_by_bbox,
                        block_text(block),
                    )
                )
            )
        else:
            text_segments = [segment for segment in bucket if segment["kind"] == "text"]
            if text_segments:
                routes.append(_build_route_line(text_segments))

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
    return _filter_thin_text_artifact_routes(routes)


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
            normalized.append(
                _build_route_line(
                    normalized_segments,
                    source=str(route.get(LAYOUT_ROUTE_SOURCE_FIELD) or ""),
                )
            )
    return normalized


def _routes_have_marker_formula(routes: list[dict[str, Any]]) -> bool:
    return any(
        segment.get("kind") == "formula"
        and is_formula_marker_token(segment.get("text", ""))
        for route in routes
        for segment in route.get("segments", [])
        if isinstance(segment, dict)
    )


def _routes_are_runtime_ppocr_line_routes(routes: list[dict[str, Any]]) -> bool:
    return bool(routes) and all(
        route.get(LAYOUT_ROUTE_SOURCE_FIELD) == LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS
        for route in routes
    )


def line_routes_for_block(
    block: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    cached = _normalize_cached_line_routes(block.get(LAYOUT_LINE_ROUTES_FIELD), width, height)
    if cached:
        if not _routes_are_runtime_ppocr_line_routes(cached):
            block.pop(LAYOUT_LINE_ROUTES_FIELD, None)
        elif _routes_have_marker_formula(cached) and route_subblocks_for_block(block, width, height):
            block.pop(LAYOUT_LINE_ROUTES_FIELD, None)
        else:
            block[LAYOUT_LINE_ROUTES_FIELD] = cached
            return cached
    routes = build_layout_line_routes(block, width, height)
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
    return slices


def has_layout_line_routes(
    block: dict[str, Any],
    width: int,
    height: int,
) -> bool:
    """Return whether layout routes exist or can be derived without mutating block."""
    cached = _normalize_cached_line_routes(block.get(LAYOUT_LINE_ROUTES_FIELD), width, height)
    if _routes_are_runtime_ppocr_line_routes(cached):
        return True
    return bool(route_subblocks_for_block(block, width, height))


__all__ = [
    "LAYOUT_LINE_ROUTES_FIELD",
    "LAYOUT_ROUTE_SOURCE_FIELD",
    "LAYOUT_ROUTE_SOURCE_PPOCR_LINE_HINTS",
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
    "formula_texts_by_subblock_bbox",
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
