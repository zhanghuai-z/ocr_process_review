"""API model profiles and endpoint resolution.

The UI lets users store either a service root URL or a full model endpoint.
Core OCR/layout code must resolve that value consistently instead of blindly
appending `/layout-parsing`.
"""
from __future__ import annotations

from typing import Any

KNOWN_API_ENDPOINT_SUFFIXES = ("/ocr", "/layout-parsing")

PADDLE_COORD_STABILITY_FLAGS: dict[str, bool] = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}

PADDLE_OCR_WORD_BOX_PARAMS: dict[str, object] = {
    "returnWordBox": True,
    "textDetLimitSideLen": 1536,
    "textDetLimitType": "max",
    "textDetThresh": 0.3,
    "textDetBoxThresh": 0.6,
    "textDetUnclipRatio": 2.0,
    "textRecScoreThresh": 0.0,
}

API_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "pp-ocrv5": {
        "label": "PP-OCRv5",
        "url": "https://n6z9feddjca4l7b5.aistudio-app.com/ocr",
        "desc": "通用文字识别（/ocr）",
        "endpoint_suffix": "/ocr",
        "request_family": "ocr-word-box",
        "returns": ("line", "word", "token"),
        "max_text_bbox_granularity": "word",
        "layout": False,
    },
    "pp-structurev3": {
        "label": "PP-StructureV3",
        "url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
        "desc": "版面 + OCR（/layout-parsing）",
        "endpoint_suffix": "/layout-parsing",
        "request_family": "ocr-word-box",
        "returns": ("layout", "block", "line", "word", "token", "markdown"),
        "max_text_bbox_granularity": "word",
        "layout": True,
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL",
        "url": "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing",
        "desc": "VL 大模型版面解析",
        "endpoint_suffix": "/layout-parsing",
        "request_family": "vl-layout",
        "returns": ("layout", "block", "line", "markdown"),
        "max_text_bbox_granularity": "line",
        "layout": True,
    },
    "paddleocr-vl-1.5": {
        "label": "PaddleOCR-VL-1.5",
        "url": "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
        "desc": "VL 1.5 升级版",
        "endpoint_suffix": "/layout-parsing",
        "request_family": "vl-layout",
        "returns": ("layout", "block", "line", "markdown"),
        "max_text_bbox_granularity": "line",
        "layout": True,
    },
}


def get_api_model_profile_options() -> list[tuple[str, str]]:
    return [(key, spec["label"]) for key, spec in API_MODEL_PROFILES.items()]


def get_api_model_profile_url(profile: str | None) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return str(API_MODEL_PROFILES[profile]["url"])
    return str(API_MODEL_PROFILES["pp-structurev3"]["url"])


def get_api_model_profile(profile: str | None) -> dict[str, Any]:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return API_MODEL_PROFILES[profile]
    return API_MODEL_PROFILES["pp-structurev3"]


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized:
            return key
    return None


def default_endpoint_suffix_for_profile(profile: str | None, fallback: str = "/layout-parsing") -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return str(API_MODEL_PROFILES[profile].get("endpoint_suffix") or fallback)
    return fallback


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


def resolve_api_endpoint_for_role(
    api_url: str | None,
    *,
    profile: str | None = None,
    role: str,
) -> str:
    """Resolve the concrete endpoint for the model role used by the main app.

    The proof workflow is intentionally dual-model:
    - layout role -> Structure/VL `/layout-parsing`
    - OCR proof role -> PP-OCRv5 `/ocr`

    Official AiStudio presets use different hosts, so exact preset URLs are
    switched to their paired role endpoint.  Custom self-hosted URLs keep the
    same root and only swap `/layout-parsing` <-> `/ocr`.
    """
    normalized = (api_url or "").strip().rstrip("/")
    if not normalized:
        return ""

    matched = match_api_model_profile_from_url(normalized)

    if role == "ocr":
        if matched and matched != "pp-ocrv5" and API_MODEL_PROFILES[matched].get("layout"):
            return get_api_model_profile_url("pp-ocrv5")
        if normalized.endswith("/layout-parsing"):
            return f"{normalized[:-len('/layout-parsing')]}/ocr"
        return resolve_api_endpoint(
            normalized,
            default_suffix="/ocr",
            profile="pp-ocrv5",
        )

    if role == "layout":
        if matched == "pp-ocrv5":
            return get_api_model_profile_url("pp-structurev3")
        if normalized.endswith("/ocr"):
            return f"{normalized[:-len('/ocr')]}/layout-parsing"
        return resolve_api_endpoint(
            normalized,
            default_suffix="/layout-parsing",
            profile="pp-structurev3",
        )

    raise ValueError(f"Unknown API endpoint role: {role}")


def infer_api_model_profile_from_endpoint(endpoint_url: str | None) -> str | None:
    normalized = (endpoint_url or "").strip().rstrip("/")
    matched = match_api_model_profile_from_url(normalized)
    if matched:
        return matched
    if normalized.endswith("/ocr"):
        return "pp-ocrv5"
    if normalized.endswith("/layout-parsing"):
        return "pp-structurev3"
    return None


def get_api_request_options(profile: str | None, endpoint_url: str | None = None) -> dict[str, object]:
    """Return request options supported by the selected model family.

    OCR detector/recognizer tuning is meaningful for PP-OCRv5 and the
    Structure OCR path.  VL layout endpoints may ignore or reject these fields,
    so they intentionally do not receive them.  The orientation/unwarping flags
    are coordinate-space guards and are sent for every known family.
    """
    profile_key = profile if isinstance(profile, str) and profile in API_MODEL_PROFILES else None
    if profile_key is None:
        profile_key = infer_api_model_profile_from_endpoint(endpoint_url)
    if profile_key is None:
        profile_key = "pp-structurev3"
    family = API_MODEL_PROFILES[profile_key].get("request_family")
    options: dict[str, object] = {}
    options.update(PADDLE_COORD_STABILITY_FLAGS)
    if family == "ocr-word-box":
        options.update(PADDLE_OCR_WORD_BOX_PARAMS)
        return options
    return options


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
