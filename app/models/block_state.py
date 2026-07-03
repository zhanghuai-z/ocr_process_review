"""Typed state helpers for layout blocks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .project import PaddleBinding


def paddle_binding_dict(block: object) -> dict[str, Any]:
    binding = getattr(block, "paddle_binding", None)
    if isinstance(binding, PaddleBinding):
        return binding.to_dict()
    return {}


def set_paddle_binding(block: object, binding: Mapping[str, Any] | PaddleBinding | None) -> None:
    if isinstance(binding, PaddleBinding):
        setattr(block, "paddle_binding", binding)
        return
    setattr(block, "paddle_binding", PaddleBinding.from_dict(dict(binding or {})))


def clear_paddle_binding(block: object) -> None:
    setattr(block, "paddle_binding", None)


def is_ocr_text_invalidated(block: object) -> bool:
    reason = str(getattr(block, "ocr_invalidated_reason", "") or "")
    return bool(reason)


def ocr_invalidation_reason(block: object) -> str:
    return str(getattr(block, "ocr_invalidated_reason", "") or "")


def mark_ocr_text_invalidated(block: object, kind: str | None = None) -> dict[str, Any]:
    reason = str(kind or "layout_changed")
    setattr(block, "ocr_invalidated_reason", reason)
    return {}


def clear_ocr_text_invalidation(block: object) -> dict[str, Any]:
    setattr(block, "ocr_invalidated_reason", "")
    return {}
