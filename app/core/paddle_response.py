"""Shared Paddle/AiStudio response unpacking helpers."""
from __future__ import annotations


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
    values = pruned.get("parsing_res_list", {}) if isinstance(pruned, dict) else {}
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, dict)]


def layout_geometry_records_from_item(item: dict) -> list[dict]:
    pruned = pruned_result(item)
    layout_det_res = pruned.get("layout_det_res", {}) if isinstance(pruned, dict) else {}
    if not isinstance(layout_det_res, dict):
        return []
    values = layout_det_res.get("boxes")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, dict)]
