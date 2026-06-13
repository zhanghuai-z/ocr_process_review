"""Shared Paddle/AiStudio response unpacking helpers."""
from __future__ import annotations

from typing import Any


def result_dict(data: dict) -> dict:
    result = data.get("result", {}) if isinstance(data, dict) else {}
    return result if isinstance(result, dict) else {}


def result_items(data: dict, key: str) -> list[dict]:
    values = result_dict(data).get(key)
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def iter_ocr_preferred_items(data: dict) -> list[dict]:
    ocr_items = result_items(data, "ocrResults")
    if ocr_items:
        return ocr_items
    return result_items(data, "layoutParsingResults")


def pruned_result(item: dict) -> dict:
    pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}
    return pruned if isinstance(pruned, dict) else {}


def overall_ocr_res(item: dict) -> dict:
    pruned = pruned_result(item)
    ocr_res = pruned.get("overall_ocr_res")
    if isinstance(ocr_res, dict):
        return ocr_res
    if isinstance(pruned, dict) and any(key in pruned for key in ("rec_texts", "rec_boxes", "rec_polys", "dt_polys")):
        return pruned
    direct = item.get("overall_ocr_res") if isinstance(item, dict) else None
    if isinstance(direct, dict):
        return direct
    if isinstance(item, dict) and any(key in item for key in ("rec_texts", "rec_boxes", "rec_polys", "dt_polys")):
        return item
    return {}


def parsing_records_from_item(item: dict) -> list[dict]:
    pruned = pruned_result(item)
    records: list[dict] = []
    for container in (pruned, item):
        values = container.get("parsing_res_list", {}) if isinstance(container, dict) else {}
        if isinstance(values, list):
            records.extend(value for value in values if isinstance(value, dict))
    return records


def layout_geometry_records_from_item(item: dict) -> list[dict]:
    pruned = pruned_result(item)
    candidates: list[dict] = []
    for container in (
        item,
        pruned,
        pruned.get("layout_det_res", {}),
        item.get("layout_det_res", {}) if isinstance(item, dict) else {},
    ):
        if not isinstance(container, dict):
            continue
        for key in ("boxes", "layout_boxes", "regions", "blocks", "formula"):
            values = container.get(key)
            if isinstance(values, list):
                candidates.extend(value for value in values if isinstance(value, dict))
    return candidates


def iter_layout_records_from_item(item: dict) -> list[dict]:
    parsing_records = parsing_records_from_item(item)
    if parsing_records:
        return parsing_records
    return layout_geometry_records_from_item(item)


def _as_sequence(value: Any) -> list:
    return value if isinstance(value, list) else []


def iter_ocr_records_from_item(item: dict) -> list[dict]:
    pruned = pruned_result(item)
    ocr_res = overall_ocr_res(item) or pruned or item
    if not isinstance(ocr_res, dict):
        return []

    texts = _as_sequence(ocr_res.get("rec_texts") or ocr_res.get("texts") or [])
    scores = _as_sequence(ocr_res.get("rec_scores") or ocr_res.get("scores") or [])
    boxes = _as_sequence(
        ocr_res.get("rec_boxes")
        or ocr_res.get("rec_polys")
        or ocr_res.get("rec_polygons")
        or ocr_res.get("boxes")
        or ocr_res.get("polys")
        or ocr_res.get("dt_polys")
        or []
    )

    records: list[dict] = []
    count = max(len(boxes), len(texts))
    for index in range(count):
        record = {
            "label": "text",
            "text": texts[index] if index < len(texts) else "",
        }
        if index < len(boxes):
            record["bbox"] = boxes[index]
        if index < len(scores):
            record["score"] = scores[index]
        records.append(record)
    return records
