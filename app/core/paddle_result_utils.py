"""Helpers for PaddleOCR 3.x result objects and local model profiles."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable


DEFAULT_LOCAL_LANG = "ch"
DEFAULT_LOCAL_MODEL_PROFILE = "standard"
LOCAL_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        "label": "标准（推荐）",
        "description": "PP-OCRv5 Server + PP-DocLayout-M，稳定优先",
        "ocr_text_detection_model_name": "PP-OCRv5_server_det",
        "ocr_text_recognition_model_name": "PP-OCRv5_server_rec",
        "ocr_text_det_limit_side_len": 960,
        "ocr_text_det_limit_type": "max",
        "ocr_use_textline_orientation": False,
        "layout_detection_model_name": "PP-DocLayout-M",
        "layout_text_detection_model_name": "PP-OCRv5_mobile_det",
        "layout_text_recognition_model_name": "PP-OCRv5_mobile_rec",
    },
    "fast": {
        "label": "快速（更省内存）",
        "description": "PP-OCRv5 Server + PP-DocLayout-M，更小检测边长",
        "ocr_text_detection_model_name": "PP-OCRv5_server_det",
        "ocr_text_recognition_model_name": "PP-OCRv5_server_rec",
        "ocr_text_det_limit_side_len": 736,
        "ocr_text_det_limit_type": "max",
        "ocr_use_textline_orientation": False,
        "layout_detection_model_name": "PP-DocLayout-M",
        "layout_text_detection_model_name": "PP-OCRv5_mobile_det",
        "layout_text_recognition_model_name": "PP-OCRv5_mobile_rec",
    },
}


def prepare_paddle_runtime_env() -> None:
    """Use stable local defaults for PaddleX model discovery."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")


def normalize_local_model_profile(profile: str | None) -> str:
    if isinstance(profile, str) and profile in LOCAL_MODEL_PROFILES:
        return profile
    return DEFAULT_LOCAL_MODEL_PROFILE


def get_local_model_profile(profile: str | None) -> dict[str, Any]:
    return LOCAL_MODEL_PROFILES[normalize_local_model_profile(profile)]


def get_local_model_profile_options() -> list[tuple[str, str]]:
    return [
        (key, spec["label"])
        for key, spec in LOCAL_MODEL_PROFILES.items()
    ]


def get_local_model_profile_label(profile: str | None) -> str:
    return get_local_model_profile(profile)["label"]


def get_local_ocr_init_kwargs(profile: str | None) -> dict[str, Any]:
    spec = get_local_model_profile(profile)
    return {
        "text_detection_model_name": spec["ocr_text_detection_model_name"],
        "text_recognition_model_name": spec["ocr_text_recognition_model_name"],
        "engine": "paddle_dynamic",
        "text_det_limit_side_len": spec["ocr_text_det_limit_side_len"],
        "text_det_limit_type": spec["ocr_text_det_limit_type"],
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": spec["ocr_use_textline_orientation"],
    }


def _maybe_decode_payload(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        nested = value.get("res")
        if isinstance(nested, dict):
            return nested
        return value
    return None


def result_to_dict(result: Any) -> dict:
    """Convert PaddleOCR/PaddleX result objects to a plain dict."""
    candidates: list[Any] = [result]

    for attr in ("json", "res"):
        if not hasattr(result, attr):
            continue
        value = getattr(result, attr)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        candidates.append(value)

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            candidates.append(to_dict())
        except TypeError:
            pass

    if not isinstance(result, dict):
        try:
            candidates.append(dict(result))
        except Exception:
            pass

    for candidate in candidates:
        payload = _maybe_decode_payload(candidate)
        if payload is not None:
            return payload
    return {}


def results_to_dicts(results: Any) -> list[dict]:
    if results is None:
        return []
    if isinstance(results, Iterable) and not isinstance(results, (str, bytes, dict)):
        items = list(results)
    else:
        items = [results]
    return [payload for payload in (result_to_dict(item) for item in items) if payload]
