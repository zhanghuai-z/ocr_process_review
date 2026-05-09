"""API model profiles and endpoint resolution.

The UI lets users store either a service root URL or a full model endpoint.
Core OCR/layout code must resolve that value consistently instead of blindly
appending `/layout-parsing`.
"""
from __future__ import annotations

KNOWN_API_ENDPOINT_SUFFIXES = ("/ocr", "/layout-parsing")

API_MODEL_PROFILES: dict[str, dict[str, str]] = {
    "pp-ocrv5": {
        "label": "PP-OCRv5",
        "url": "https://n6z9feddjca4l7b5.aistudio-app.com/ocr",
        "desc": "通用文字识别（/ocr）",
    },
    "pp-structurev3": {
        "label": "PP-StructureV3",
        "url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
        "desc": "版面 + OCR（/layout-parsing）",
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL",
        "url": "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing",
        "desc": "VL 大模型版面解析",
    },
    "paddleocr-vl-1.5": {
        "label": "PaddleOCR-VL-1.5",
        "url": "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
        "desc": "VL 1.5 升级版",
    },
}


def get_api_model_profile_options() -> list[tuple[str, str]]:
    return [(key, spec["label"]) for key, spec in API_MODEL_PROFILES.items()]


def get_api_model_profile_url(profile: str | None) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return API_MODEL_PROFILES[profile]["url"]
    return API_MODEL_PROFILES["pp-structurev3"]["url"]


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized:
            return key
    return None


def default_endpoint_suffix_for_profile(profile: str | None, fallback: str = "/layout-parsing") -> str:
    return "/ocr" if profile == "pp-ocrv5" else fallback


def resolve_api_endpoint(
    api_url: str | None,
    *,
    default_suffix: str = "/layout-parsing",
    profile: str | None = None,
) -> str:
    url = (api_url or "").strip().rstrip("/")
    if not url:
        return ""
    if any(url.endswith(suffix) for suffix in KNOWN_API_ENDPOINT_SUFFIXES):
        return url
    suffix = default_endpoint_suffix_for_profile(profile, default_suffix)
    return f"{url}{suffix}"


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

