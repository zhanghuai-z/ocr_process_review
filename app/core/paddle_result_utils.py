"""Helpers for normalizing PaddleOCR 3.x result objects."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable


DEFAULT_LOCAL_OCR_VERSION = "PP-OCRv5"
DEFAULT_LOCAL_LAYOUT_MODEL = "PP-DocLayout-M"
DEFAULT_LOCAL_LANG = "ch"
DEFAULT_LOCAL_LAYOUT_TEXT_DET_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_LOCAL_LAYOUT_TEXT_REC_MODEL = "PP-OCRv5_mobile_rec"


def prepare_paddle_runtime_env() -> None:
    """Use stable local defaults for PaddleX model discovery."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")


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
