"""Helpers for AIStudio OCR/layout API endpoints, presets, and response parsing."""
from __future__ import annotations

from typing import Iterator

from app.core.bbox_utils import bbox_from_quad, sanitize_xyxy_bbox
from app.models import BBox

KNOWN_API_ENDPOINT_SUFFIXES = ("/ocr", "/layout-parsing")
DEFAULT_API_MODEL_PROFILE = "pp-ocrv5"
API_MODEL_PROFILES: dict[str, dict[str, str]] = {
    "pp-ocrv5": {
        "label": "PP-OCRv5",
        "url": "https://n6z9feddjca4l7b5.aistudio-app.com/ocr",
    },
    "pp-structurev3": {
        "label": "PP-StructureV3",
        "url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL",
        "url": "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing",
    },
    "paddleocr-vl-1.5": {
        "label": "PaddleOCR-VL-1.5",
        "url": "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
    },
}


def normalize_api_model_profile(profile: str | None) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return profile
    return DEFAULT_API_MODEL_PROFILE


def get_api_model_profile(profile: str | None) -> dict[str, str]:
    return API_MODEL_PROFILES[normalize_api_model_profile(profile)]


def get_api_model_profile_options() -> list[tuple[str, str]]:
    return [(key, value["label"]) for key, value in API_MODEL_PROFILES.items()]


def get_api_model_profile_url(profile: str | None) -> str:
    return get_api_model_profile(profile)["url"]


def get_api_model_profile_label(profile: str | None) -> str:
    return get_api_model_profile(profile)["label"]


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized_url = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized_url:
            return key
    return None


def infer_api_model_profile_from_url(api_url: str | None) -> str:
    matched = match_api_model_profile_from_url(api_url)
    if matched:
        return matched
    return DEFAULT_API_MODEL_PROFILE


def resolve_api_endpoint(api_url: str | None, default_suffix: str = "/layout-parsing") -> str:
    url = (api_url or "").strip().rstrip("/")
    if not url:
        return ""
    if any(url.endswith(suffix) for suffix in KNOWN_API_ENDPOINT_SUFFIXES):
        return url
    return f"{url}{default_suffix}"


def build_api_payload(file_b64: str, file_type: int) -> dict[str, object]:
    return {
        "file": file_b64,
        "fileType": file_type,
    }


def get_api_result_items(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        return []
    result = data.get("result", {})
    if not isinstance(result, dict):
        return []

    for key in ("layoutParsingResults", "ocrResults"):
        items = result.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def detect_api_result_kind(data: dict) -> str:
    if not isinstance(data, dict):
        return "unknown"
    result = data.get("result", {})
    if not isinstance(result, dict):
        return "unknown"
    if isinstance(result.get("ocrResults"), list):
        return "ocr"
    if isinstance(result.get("layoutParsingResults"), list):
        return "layout"
    return "unknown"


def iter_api_ocr_payloads(item: dict) -> Iterator[dict]:
    if not isinstance(item, dict):
        return

    candidates: list[dict] = []
    pruned = item.get("prunedResult")
    if isinstance(pruned, dict):
        overall = pruned.get("overall_ocr_res")
        if isinstance(overall, dict):
            candidates.append(overall)
        candidates.append(pruned)

    overall = item.get("overall_ocr_res")
    if isinstance(overall, dict):
        candidates.append(overall)
    candidates.append(item)

    seen: set[int] = set()
    for candidate in candidates:
        key = id(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (
            isinstance(candidate.get("rec_texts"), list)
            or isinstance(candidate.get("rec_boxes"), list)
            or isinstance(candidate.get("rec_polys"), list)
        ):
            yield candidate


def extract_api_markdown_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    markdown = item.get("markdown")
    if isinstance(markdown, dict):
        text = markdown.get("text")
        if isinstance(text, str):
            return text
    return ""


def extract_bbox_from_box_data(box_data, limit_w: int, limit_h: int) -> BBox:
    if box_data is None:
        return BBox(0, 0, 0, 0)
    if hasattr(box_data, "tolist"):
        box_data = box_data.tolist()
    if isinstance(box_data, (list, tuple)) and len(box_data) >= 4:
        if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in box_data[:4]):
            return bbox_from_quad(box_data[:4]).clamp(limit_w, limit_h)
        return sanitize_xyxy_bbox(box_data[:4], limit_w, limit_h)
    return BBox(0, 0, 0, 0)


def collect_api_text_bboxes(item: dict, limit_w: int, limit_h: int) -> list[BBox]:
    boxes: list[BBox] = []
    for payload in iter_api_ocr_payloads(item):
        rec_boxes = payload.get("rec_boxes", [])
        rec_polys = payload.get("rec_polys", [])
        total = max(len(rec_boxes), len(rec_polys))
        for idx in range(total):
            box_data = rec_boxes[idx] if idx < len(rec_boxes) else rec_polys[idx]
            bbox = extract_bbox_from_box_data(box_data, limit_w, limit_h)
            if bbox.area > 0:
                boxes.append(bbox)
    return boxes


def union_bboxes(boxes: list[BBox], limit_w: int, limit_h: int) -> BBox | None:
    if not boxes:
        return None
    x1 = min(box.x1 for box in boxes)
    y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return sanitize_xyxy_bbox([x1, y1, x2, y2], limit_w, limit_h)


def split_markdown_lines(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    text = text.strip()
    return [text] if text else []


def split_markdown_paragraphs(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    paragraphs: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip():
            current.append(line)
            continue
        if current:
            paragraphs.append("\n".join(current).strip())
            current = []

    if current:
        paragraphs.append("\n".join(current).strip())

    return [paragraph for paragraph in paragraphs if paragraph]
