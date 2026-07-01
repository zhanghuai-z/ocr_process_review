"""Inline-formula crop OCR utilities.

The production OCR path uses the pseudo-page crop recognition helpers here to
refresh text for the current formula boxes.  Parent paragraph text is not a
truth source for edited/manual formula boxes.

The span-suggestion helpers remain experimental diagnostics for comparing crop
OCR against parent text; they are not used as authoritative binding logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from app.core.api_image_codec import encode_image_bytes_for_paddle
from app.core.bbox_extraction import bbox_from_variant
from app.core.ocr_ir import is_formula_marker_token
from app.core.paddle_layout_schema import paddle_record_label, paddle_record_text
from app.core.paddle_v16_client import (
    PaddleV16LayoutClient,
    build_paddle_v16_optional_payload,
)


XYXY = tuple[int, int, int, int]
FORMULA_PATTERN = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class FormulaCrop:
    index: int
    bbox: XYXY
    crop_bbox: XYXY
    pseudo_bbox: XYXY


@dataclass(frozen=True)
class FormulaPseudoPage:
    image_bgr: np.ndarray
    crops: tuple[FormulaCrop, ...]


@dataclass(frozen=True)
class FormulaRecognition:
    crop_index: int
    bbox: XYXY
    text: str
    normalized_key: str
    records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FormulaSpan:
    span_index: int
    text: str
    normalized_key: str
    start: int
    end: int


@dataclass(frozen=True)
class FormulaBindingSuggestion:
    crop_index: int
    bbox: XYXY
    recognized_text: str
    span_index: int | None
    span_text: str
    status: str
    score: float = 0.0


def build_formula_pseudo_page(
    image_bgr: np.ndarray,
    formula_bboxes: Sequence[Sequence[int] | XYXY],
    *,
    crop_pad: int = 8,
    row_gap: int = 24,
    margin: int = 24,
    min_row_height: int = 48,
) -> FormulaPseudoPage:
    """Stack formula crops into a white pseudo page.

    Crops are not scaled.  Each original bbox maps to one row in the pseudo
    page, so Paddle records can be assigned back by vertical overlap.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("image_bgr must be a BGR image")
    page_h, page_w = image_bgr.shape[:2]
    crops: list[tuple[int, XYXY, XYXY, np.ndarray]] = []
    for index, raw_bbox in enumerate(formula_bboxes):
        bbox = _coerce_xyxy(raw_bbox)
        clamped = _clamp_xyxy(bbox, page_w, page_h)
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            continue
        crop_bbox = _expand_xyxy(clamped, crop_pad, page_w, page_h)
        x1, y1, x2, y2 = crop_bbox
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crops.append((index, clamped, crop_bbox, crop))
    if not crops:
        return FormulaPseudoPage(
            image_bgr=np.full((max(1, margin * 2), max(1, margin * 2), 3), 255, dtype=np.uint8),
            crops=(),
        )

    max_w = max(crop.shape[1] for _index, _bbox, _crop_bbox, crop in crops)
    row_heights = [max(min_row_height, crop.shape[0]) for _index, _bbox, _crop_bbox, crop in crops]
    canvas_w = margin * 2 + max_w
    canvas_h = margin * 2 + sum(row_heights) + row_gap * (len(crops) - 1)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    mapped: list[FormulaCrop] = []
    y = margin
    for (index, bbox, crop_bbox, crop), row_h in zip(crops, row_heights):
        x = margin
        crop_h, crop_w = crop.shape[:2]
        y_offset = y + max(0, (row_h - crop_h) // 2)
        canvas[y_offset:y_offset + crop_h, x:x + crop_w] = crop
        mapped.append(
            FormulaCrop(
                index=index,
                bbox=bbox,
                crop_bbox=crop_bbox,
                pseudo_bbox=(x, y, x + crop_w, y + row_h),
            )
        )
        y += row_h + row_gap

    return FormulaPseudoPage(image_bgr=canvas, crops=tuple(mapped))


def formula_crop_bboxes_from_route_subblocks(
    subblocks: Sequence[dict[str, Any]],
    *,
    min_width: int = 24,
    min_height: int = 12,
) -> list[XYXY]:
    """Return crop bboxes for formula rebinding from route subblocks.

    Manual boxes are authoritative.  If a generated Paddle formula box overlaps
    a manual formula box on the same row, the generated box is split
    horizontally and only the remaining pieces are kept.  This handles the
    common "Paddle merged two inline formulas, user split one side" case without
    sending the merged old box back as one ambiguous crop.
    """
    formula_items: list[tuple[XYXY, bool]] = []
    for item in subblocks:
        label = str(item.get("block_label") or item.get("label") or "")
        if "formula" not in label and "equation" not in label:
            continue
        bbox_value = item.get("block_bbox") or item.get("bbox")
        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            continue
        bbox = _coerce_xyxy(bbox_value)
        formula_items.append((bbox, bool(item.get("_layout_manual_route_subblock"))))

    manual_bboxes = [bbox for bbox, is_manual in formula_items if is_manual]
    candidates: list[XYXY] = []
    for bbox, is_manual in formula_items:
        if is_manual:
            candidates.append(bbox)
            continue
        pieces = [bbox]
        for manual in manual_bboxes:
            next_pieces: list[XYXY] = []
            for piece in pieces:
                next_pieces.extend(_subtract_manual_formula_piece(piece, manual))
            pieces = next_pieces
        candidates.extend(pieces)

    filtered = [
        bbox for bbox in candidates
        if bbox[2] - bbox[0] >= min_width and bbox[3] - bbox[1] >= min_height
    ]
    return _reading_order_bboxes(_dedupe_bboxes(filtered))


def recognize_formula_pseudo_page(
    pseudo_page: FormulaPseudoPage,
    *,
    client: PaddleV16LayoutClient,
    batch_id: str = "",
    filename: str = "inline-formula-pseudo-page.png",
) -> tuple[list[FormulaRecognition], dict[str, Any]]:
    """Call Paddle for a pseudo page and return crop-level formula text."""
    image_bytes = encode_image_bytes_for_paddle(pseudo_page.image_bgr)
    if not image_bytes:
        raise RuntimeError("Cannot encode inline formula pseudo page")
    response = client.analyze_image_bytes(
        image_bytes,
        optional_payload=build_paddle_v16_optional_payload(),
        batch_id=batch_id,
        filename=filename,
    )
    return recognitions_from_paddle_response(pseudo_page, response), response


def recognitions_from_paddle_response(
    pseudo_page: FormulaPseudoPage,
    response: dict[str, Any],
) -> list[FormulaRecognition]:
    """Assign Paddle layout records on the pseudo page back to crop rows."""
    records_by_crop: dict[int, list[dict[str, Any]]] = {crop.index: [] for crop in pseudo_page.crops}
    for record in iter_paddle_text_records(response):
        bbox = bbox_from_variant(record, max_w=pseudo_page.image_bgr.shape[1], max_h=pseudo_page.image_bgr.shape[0])
        if bbox is None:
            continue
        xyxy = tuple(int(value) for value in bbox.to_xyxy())
        crop = _best_crop_for_record(pseudo_page.crops, xyxy)
        if crop is None:
            continue
        records_by_crop[crop.index].append(record)

    recognitions: list[FormulaRecognition] = []
    crop_by_index = {crop.index: crop for crop in pseudo_page.crops}
    for crop_index, records in records_by_crop.items():
        text = _records_formula_text(records)
        if not text:
            continue
        crop = crop_by_index[crop_index]
        recognitions.append(
            FormulaRecognition(
                crop_index=crop_index,
                bbox=crop.bbox,
                text=text,
                normalized_key=normalize_formula_key(text),
                records=tuple(dict(record) for record in records),
            )
        )
    recognitions.sort(key=lambda item: item.crop_index)
    return recognitions


def suggest_formula_bindings(
    parent_text: str,
    recognitions: Iterable[FormulaRecognition],
) -> list[FormulaBindingSuggestion]:
    """Map crop OCR evidence to parent ``$...$`` spans."""
    spans = formula_spans(parent_text)
    spans_by_key: dict[str, list[FormulaSpan]] = {}
    for span in spans:
        if span.normalized_key:
            spans_by_key.setdefault(span.normalized_key, []).append(span)

    used_span_indices: set[int] = set()
    occurrence_by_key: dict[str, int] = {}
    suggestions: list[FormulaBindingSuggestion] = []
    for recognition in sorted(recognitions, key=lambda item: item.crop_index):
        key = recognition.normalized_key
        candidates = [span for span in spans_by_key.get(key, []) if span.span_index not in used_span_indices]
        if candidates:
            offset = occurrence_by_key.get(key, 0)
            span = candidates[min(offset, len(candidates) - 1)]
            occurrence_by_key[key] = offset + 1
            used_span_indices.add(span.span_index)
            status = "exact" if len(spans_by_key.get(key, [])) == 1 else "duplicate_key_ordered"
            suggestions.append(_bound_suggestion(recognition, span, status, 1.0))
            continue

        fuzzy = _best_fuzzy_span(key, spans, used_span_indices)
        if fuzzy is not None:
            span, score = fuzzy
            used_span_indices.add(span.span_index)
            suggestions.append(_bound_suggestion(recognition, span, "fuzzy", score))
            continue

        suggestions.append(
            FormulaBindingSuggestion(
                crop_index=recognition.crop_index,
                bbox=recognition.bbox,
                recognized_text=recognition.text,
                span_index=None,
                span_text="",
                status="unresolved",
                score=0.0,
            )
        )
    return suggestions


def formula_spans(parent_text: str) -> list[FormulaSpan]:
    spans: list[FormulaSpan] = []
    for match in FORMULA_PATTERN.finditer(parent_text or ""):
        text = match.group(0)
        if is_formula_marker_token(text):
            continue
        spans.append(
            FormulaSpan(
                span_index=len(spans),
                text=text,
                normalized_key=normalize_formula_key(text),
                start=match.start(),
                end=match.end(),
            )
        )
    return spans


def normalize_formula_key(text: str) -> str:
    """Normalize formula text for matching Paddle crop OCR to parent spans."""
    value = str(text or "").strip()
    value = re.sub(r"^\s*\$\$\s*", "", value)
    value = re.sub(r"\s*\$\$\s*$", "", value)
    value = re.sub(r"^\s*\$\s*", "", value)
    value = re.sub(r"\s*\$\s*$", "", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\[;,! ]", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("－", "-").replace("−", "-").replace("–", "-")
    return value.lower()


def iter_paddle_text_records(response: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield Paddle records that have both text and bbox-like geometry."""
    for record in _walk_dicts(response):
        text = paddle_record_text(record)
        if not text:
            continue
        if bbox_from_variant(record) is None:
            continue
        yield record


def _records_formula_text(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    ordered = sorted(records, key=lambda record: _record_sort_key(record))
    formula_like: list[str] = []
    other: list[str] = []
    for record in ordered:
        text = paddle_record_text(record)
        label = paddle_record_label(record, default="")
        if "formula" in label or "$" in text or "\\" in text or "_" in text or "^" in text:
            formula_like.append(text)
        else:
            other.append(text)
    values = formula_like or other
    return " ".join(value.strip() for value in values if value.strip()).strip()


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    bbox = bbox_from_variant(record)
    if bbox is None:
        return (0, 0)
    x1, y1, _x2, _y2 = bbox.to_xyxy()
    return int(y1), int(x1)


def _best_crop_for_record(crops: Sequence[FormulaCrop], record_bbox: XYXY) -> FormulaCrop | None:
    best: tuple[float, FormulaCrop] | None = None
    for crop in crops:
        score = _vertical_overlap_ratio(crop.pseudo_bbox, record_bbox)
        center_y = (record_bbox[1] + record_bbox[3]) / 2.0
        if crop.pseudo_bbox[1] <= center_y <= crop.pseudo_bbox[3]:
            score += 1.0
        if best is None or score > best[0]:
            best = (score, crop)
    if best is None or best[0] <= 0:
        return None
    return best[1]


def _best_fuzzy_span(
    key: str,
    spans: Sequence[FormulaSpan],
    used_span_indices: set[int],
) -> tuple[FormulaSpan, float] | None:
    if not key:
        return None
    scored: list[tuple[float, FormulaSpan]] = []
    for span in spans:
        if span.span_index in used_span_indices or not span.normalized_key:
            continue
        score = SequenceMatcher(a=key, b=span.normalized_key, autojunk=False).ratio()
        scored.append((score, span))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 0.86:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.04:
        return None
    return scored[0][1], scored[0][0]


def _bound_suggestion(
    recognition: FormulaRecognition,
    span: FormulaSpan,
    status: str,
    score: float,
) -> FormulaBindingSuggestion:
    return FormulaBindingSuggestion(
        crop_index=recognition.crop_index,
        bbox=recognition.bbox,
        recognized_text=recognition.text,
        span_index=span.span_index,
        span_text=span.text,
        status=status,
        score=score,
    )


def _subtract_manual_formula_piece(piece: XYXY, manual: XYXY) -> list[XYXY]:
    if _vertical_overlap_ratio(piece, manual) < 0.45:
        return [piece]
    overlap_x1 = max(piece[0], manual[0])
    overlap_x2 = min(piece[2], manual[2])
    if overlap_x2 <= overlap_x1:
        return [piece]
    overlap_w = overlap_x2 - overlap_x1
    manual_w = max(1, manual[2] - manual[0])
    if overlap_w / manual_w < 0.45:
        return [piece]
    pieces: list[XYXY] = []
    if overlap_x1 > piece[0]:
        pieces.append((piece[0], piece[1], overlap_x1, piece[3]))
    if overlap_x2 < piece[2]:
        pieces.append((overlap_x2, piece[1], piece[2], piece[3]))
    return pieces


def _dedupe_bboxes(values: Iterable[XYXY]) -> list[XYXY]:
    seen: set[XYXY] = set()
    result: list[XYXY] = []
    for bbox in values:
        if bbox in seen:
            continue
        seen.add(bbox)
        result.append(bbox)
    return result


def _reading_order_bboxes(values: Sequence[XYXY]) -> list[XYXY]:
    buckets: list[list[XYXY]] = []
    for bbox in sorted(values, key=lambda item: (item[1], item[0])):
        for bucket in buckets:
            union = (
                min(item[0] for item in bucket),
                min(item[1] for item in bucket),
                max(item[2] for item in bucket),
                max(item[3] for item in bucket),
            )
            if _vertical_overlap_ratio(union, bbox) >= 0.25:
                bucket.append(bbox)
                break
        else:
            buckets.append([bbox])
    ordered: list[XYXY] = []
    for bucket in sorted(buckets, key=lambda items: (min(item[1] for item in items), min(item[0] for item in items))):
        ordered.extend(sorted(bucket, key=lambda item: (item[0], item[1])))
    return ordered


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _coerce_xyxy(raw_bbox: Sequence[int] | XYXY) -> XYXY:
    if len(raw_bbox) != 4:
        raise ValueError(f"bbox must have four values: {raw_bbox!r}")
    x1, y1, x2, y2 = (int(value) for value in raw_bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _clamp_xyxy(bbox: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(x1, min(int(x2), width))
    y2 = max(y1, min(int(y2), height))
    return x1, y1, x2, y2


def _expand_xyxy(bbox: XYXY, pad: int, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = bbox
    pad = max(0, int(pad))
    return _clamp_xyxy((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def _vertical_overlap_ratio(a: XYXY, b: XYXY) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1, min(a[3] - a[1], b[3] - b[1]))
    return overlap / denom


__all__ = [
    "FormulaBindingSuggestion",
    "FormulaCrop",
    "FormulaPseudoPage",
    "FormulaRecognition",
    "FormulaSpan",
    "build_formula_pseudo_page",
    "formula_spans",
    "formula_crop_bboxes_from_route_subblocks",
    "iter_paddle_text_records",
    "normalize_formula_key",
    "recognitions_from_paddle_response",
    "recognize_formula_pseudo_page",
    "suggest_formula_bindings",
]
