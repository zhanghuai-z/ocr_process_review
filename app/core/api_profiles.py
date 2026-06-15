"""API model profiles and endpoint resolution.

The UI lets users store either a service root URL or a full model endpoint.
Core OCR/layout code must resolve that value consistently instead of blindly
appending the role-specific endpoint suffix.
"""
from __future__ import annotations

from typing import Any

PADDLE_V16_JOBS_PATH = "/api/v2/ocr/jobs"
LEGACY_LAYOUT_ENDPOINT_SUFFIX = "/layout-parsing"
CURRENT_API_ENDPOINT_SUFFIXES = ("/ocr", PADDLE_V16_JOBS_PATH)
KNOWN_API_ENDPOINT_SUFFIXES = CURRENT_API_ENDPOINT_SUFFIXES + (LEGACY_LAYOUT_ENDPOINT_SUFFIX,)
FIXED_LAYOUT_PROFILE = "paddleocr-vl-1.6"
FIXED_OCR_PROFILE = "pp-ocrv5"
PADDLE_V16_OFFICIAL_ROOT = "https://paddleocr.aistudio-app.com"
LEGACY_OFFICIAL_LAYOUT_ROOTS = frozenset({
    "https://fbv8f7s7v9u9hbk7.aistudio-app.com",
    "https://c92fu3s8m4y5i0je.aistudio-app.com",
    "https://15j75bd0964dzbwe.aistudio-app.com",
})

PADDLE_COORD_STABILITY_FLAGS: dict[str, bool] = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}

PADDLE_OCR_TEXT_DET_PARAMS: dict[str, object] = {
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
        "request_family": "ocr-text",
        "returns": ("line",),
        "max_text_bbox_granularity": "line",
        "layout": False,
    },
    "paddleocr-vl-1.6": {
        "label": "PaddleOCR-VL-1.6",
        "url": f"{PADDLE_V16_OFFICIAL_ROOT}{PADDLE_V16_JOBS_PATH}",
        "desc": "VL 1.6 官方 jobs API",
        "endpoint_suffix": PADDLE_V16_JOBS_PATH,
        "request_family": "vl-layout-v2",
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
    return str(API_MODEL_PROFILES[FIXED_LAYOUT_PROFILE]["url"])


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized:
            return key
    return None


def normalize_api_base_url(api_url: str | None) -> str:
    """Store API addresses as service roots, never concrete role endpoints."""
    normalized = (api_url or "").strip().rstrip("/")
    for suffix in KNOWN_API_ENDPOINT_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if normalized in LEGACY_OFFICIAL_LAYOUT_ROOTS:
        return PADDLE_V16_OFFICIAL_ROOT
    return normalized


def match_api_model_profile_from_base_url(api_url: str | None) -> str | None:
    base = normalize_api_base_url(api_url)
    if not base:
        return None
    for key, spec in API_MODEL_PROFILES.items():
        if normalize_api_base_url(str(spec["url"])) == base:
            return key
    return None


def default_endpoint_suffix_for_profile(profile: str | None, fallback: str = PADDLE_V16_JOBS_PATH) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return str(API_MODEL_PROFILES[profile].get("endpoint_suffix") or fallback)
    return fallback


def resolve_api_endpoint(
    api_url: str | None,
    *,
    default_suffix: str = PADDLE_V16_JOBS_PATH,
    profile: str | None = None,
) -> str:
    url = (api_url or "").strip().rstrip("/")
    if not url:
        return ""
    if any(url.endswith(suffix) for suffix in CURRENT_API_ENDPOINT_SUFFIXES):
        return url
    base_url = normalize_api_base_url(url)
    suffix = default_endpoint_suffix_for_profile(profile, default_suffix)
    return f"{base_url}{suffix}"


def _profile_from_explicit_or_profile_url(api_url: str, profile: str | None) -> str | None:
    matched = match_api_model_profile_from_url(api_url)
    if matched:
        return matched
    if not isinstance(profile, str) or profile not in API_MODEL_PROFILES:
        return None
    profile_url = get_api_model_profile_url(profile).rstrip("/")
    profile_root = profile_url
    for suffix in KNOWN_API_ENDPOINT_SUFFIXES:
        if profile_root.endswith(suffix):
            profile_root = profile_root[: -len(suffix)]
            break
    if api_url == profile_root:
        return profile
    return None


# 主链 layout 角色固定走 PaddleOCR-VL-1.6；OCR proof 角色固定走 PP-OCRv5。
# 旧 /layout-parsing 只在 normalize_api_base_url() 中作为历史配置后缀剥离，
# 不再作为可请求的 layout endpoint。
LAYOUT_DEFAULT_PROFILE = "paddleocr-vl-1.6"


def resolve_api_endpoint_for_role(
    api_url: str | None,
    *,
    profile: str | None = None,
    role: str,
) -> str:
    """Resolve the concrete endpoint for the model role used by the main app.

    The proof workflow is intentionally dual-model:
    - layout role -> PaddleOCR-VL-1.6 `/api/v2/ocr/jobs`
    - OCR proof role -> PP-OCRv5 `/ocr`

    Official AiStudio presets use different hosts, so exact preset URLs are
    switched to their paired role endpoint.  Custom self-hosted URLs keep the
    same root and use the role endpoint suffix.
    """
    normalized = (api_url or "").strip().rstrip("/")
    if not normalized:
        return ""

    base_url = normalize_api_base_url(normalized)
    profile_key = (
        match_api_model_profile_from_url(normalized)
        or match_api_model_profile_from_base_url(base_url)
        or _profile_from_explicit_or_profile_url(normalized, profile)
    )
    if profile_key:
        fixed_profile = FIXED_OCR_PROFILE if role == "ocr" else FIXED_LAYOUT_PROFILE if role == "layout" else ""
        if fixed_profile:
            return get_api_model_profile_url(fixed_profile)

    if role == "ocr":
        return resolve_api_endpoint(
            base_url,
            default_suffix="/ocr",
            profile=FIXED_OCR_PROFILE,
        )

    if role == "layout":
        if normalized.endswith("/ocr"):
            return resolve_api_endpoint(
                normalized[: -len("/ocr")],
                default_suffix=PADDLE_V16_JOBS_PATH,
                profile=LAYOUT_DEFAULT_PROFILE,
            )
        return resolve_api_endpoint(
            base_url,
            default_suffix=PADDLE_V16_JOBS_PATH,
            profile=LAYOUT_DEFAULT_PROFILE,
        )

    raise ValueError(f"Unknown API endpoint role: {role}")


def infer_api_model_profile_from_endpoint(endpoint_url: str | None) -> str | None:
    normalized = (endpoint_url or "").strip().rstrip("/")
    matched = match_api_model_profile_from_url(normalized)
    if matched:
        return matched
    if normalized.endswith(PADDLE_V16_JOBS_PATH):
        return "paddleocr-vl-1.6"
    if normalized.endswith("/ocr"):
        return "pp-ocrv5"
    return None


def get_api_request_options(profile: str | None, endpoint_url: str | None = None) -> dict[str, object]:
    """Return request options supported by the selected model family.

    OCR detector/recognizer tuning is meaningful for PP-OCRv5.  VL layout
    endpoints may ignore or reject these fields, so they intentionally do not
    receive them.  The orientation/unwarping flags are coordinate-space guards
    and are sent for every known family.
    """
    profile_key = profile if isinstance(profile, str) and profile in API_MODEL_PROFILES else None
    if profile_key is None:
        profile_key = infer_api_model_profile_from_endpoint(endpoint_url)
    if profile_key is None:
        profile_key = LAYOUT_DEFAULT_PROFILE
    family = API_MODEL_PROFILES[profile_key].get("request_family")
    options: dict[str, object] = {}
    options.update(PADDLE_COORD_STABILITY_FLAGS)
    if family == "ocr-text":
        options.update(PADDLE_OCR_TEXT_DET_PARAMS)
        return options
    return options
