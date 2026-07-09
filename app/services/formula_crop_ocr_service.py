"""Formula crop OCR service for current layout formula boxes."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.api_image_codec import encode_image_bytes_for_paddle
from app.core.bbox_extraction import bbox_from_variant
from app.core.paddle_layout_schema import paddle_record_label, paddle_record_text
from app.core.paddle_v16_client import PaddleV16LayoutClient, build_paddle_v16_optional_payload

XYXY = tuple[int, int, int, int]


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
    normalized_key: str = ""
    records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FormulaCropOcrOutcome:
    texts_by_index: dict[int, str]
    attempted_indices: tuple[int, ...]
    failed_indices: tuple[int, ...]
    attempts: int
    error: str = ""


def build_formula_pseudo_page(
    image_bgr: np.ndarray,
    formula_bboxes: Sequence[Sequence[int] | XYXY],
    *,
    crop_pad: int = 8,
    row_gap: int = 24,
    margin: int = 24,
    min_row_height: int = 48,
) -> FormulaPseudoPage:
    """Stack current formula crops into a white pseudo page."""
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
    """Return formula crop bboxes from explicit route subblocks."""
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


def recognize_formula_bboxes_with_retry(
    image_bgr: np.ndarray,
    formula_bboxes: Sequence[Sequence[int] | XYXY],
    *,
    client: PaddleV16LayoutClient,
    batch_id_prefix: str = "ocr-process-formula",
    filename_prefix: str = "inline-formula-pseudo",
    retry_empty: bool = True,
) -> FormulaCropOcrOutcome:
    """Recognize formula bboxes without making the OCR page depend on success."""
    first = build_formula_pseudo_page(image_bgr, formula_bboxes)
    attempted = tuple(crop.index for crop in first.crops)
    if not attempted:
        return FormulaCropOcrOutcome({}, (), (), attempts=0)

    texts: dict[int, str] = {}
    attempts = 0
    error = ""

    try:
        attempts += 1
        recognitions, _response = recognize_formula_pseudo_page(
            first,
            client=client,
            batch_id=f"{batch_id_prefix}-1",
            filename=f"{filename_prefix}-1.png",
        )
        texts.update(_texts_by_crop_index(recognitions))
    except Exception as exc:
        error = str(exc)
        if not retry_empty:
            return FormulaCropOcrOutcome(
                texts,
                attempted,
                attempted,
                attempts=attempts,
                error=error,
            )
        try:
            attempts += 1
            recognitions, _response = recognize_formula_pseudo_page(
                first,
                client=client,
                batch_id=f"{batch_id_prefix}-retry",
                filename=f"{filename_prefix}-retry.png",
            )
            texts.update(_texts_by_crop_index(recognitions))
            error = ""
        except Exception as retry_exc:
            return FormulaCropOcrOutcome(
                texts,
                attempted,
                tuple(index for index in attempted if not texts.get(index)),
                attempts=attempts,
                error=str(retry_exc),
            )

    missing = tuple(index for index in attempted if not texts.get(index))
    if retry_empty and missing:
        retry_bboxes = [formula_bboxes[index] for index in missing]
        retry_pseudo = build_formula_pseudo_page(image_bgr, retry_bboxes)
        try:
            attempts += 1
            recognitions, _response = recognize_formula_pseudo_page(
                retry_pseudo,
                client=client,
                batch_id=f"{batch_id_prefix}-empty-retry",
                filename=f"{filename_prefix}-empty-retry.png",
            )
            retry_texts = _texts_by_crop_index(recognitions)
            for retry_index, text in retry_texts.items():
                if 0 <= retry_index < len(missing) and text:
                    texts[missing[retry_index]] = text
            error = ""
        except Exception as exc:
            error = str(exc)

    failed = tuple(index for index in attempted if not texts.get(index))
    return FormulaCropOcrOutcome(
        texts,
        attempted,
        failed,
        attempts=attempts,
        error=error,
    )


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


def iter_paddle_text_records(response: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield Paddle records that have both text and bbox-like geometry."""
    for record in _walk_dicts(response):
        text = paddle_record_text(record)
        if not text:
            continue
        if bbox_from_variant(record) is None:
            continue
        yield record


def normalize_formula_key(text: str) -> str:
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


def _texts_by_crop_index(recognitions: Iterable[FormulaRecognition]) -> dict[int, str]:
    return {
        item.crop_index: str(item.text or "").strip()
        for item in recognitions
        if str(item.text or "").strip()
    }


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
    "FormulaCrop",
    "FormulaCropOcrOutcome",
    "FormulaPseudoPage",
    "FormulaRecognition",
    "build_formula_pseudo_page",
    "formula_crop_bboxes_from_route_subblocks",
    "iter_paddle_text_records",
    "normalize_formula_key",
    "recognitions_from_paddle_response",
    "recognize_formula_bboxes_with_retry",
    "recognize_formula_pseudo_page",
]
