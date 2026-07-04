"""Shared proof geometry quality predicates.

The same OCR bbox metadata drives different decisions:

- user warnings should report estimated/unavailable geometry;
- VProof indexing should hide geometry that cannot point at a stable glyph.

Keep these predicates named by use case so callers do not collapse distinct
semantics into a vague "fallback" check.
"""
from __future__ import annotations


_ESTIMATED_SOURCES = {"fallback", "unavailable"}
_ESTIMATED_GRANULARITIES = {"fallback", "unavailable", "line"}
_TRUSTED_INDEX_SOURCES = {"ocr", "paddle_inline_formula"}


def normalize_geometry_label(value: object) -> str:
    return str(value or "").strip().lower()


def is_estimated_or_unavailable_geometry(
    bbox_source: object,
    bbox_granularity: object,
) -> bool:
    """Return True when a user-facing warning should mention this bbox."""

    source = normalize_geometry_label(bbox_source)
    granularity = normalize_geometry_label(bbox_granularity)
    return source in _ESTIMATED_SOURCES or granularity in _ESTIMATED_GRANULARITIES


def is_char_index_hidden_geometry(
    bbox_source: object,
    bbox_granularity: object,
) -> bool:
    """Return True when VProof should hide this entry by default.

    Hanwang char-level fallback sources are still real OCR-derived char boxes
    in the current pipeline, so they remain indexable when granularity is
    ``char``. Line/fallback/unavailable granularity is too coarse for same-char
    indexing and stays hidden unless callers explicitly include fallback data.
    """

    source = normalize_geometry_label(bbox_source)
    granularity = normalize_geometry_label(bbox_granularity)
    if source not in _TRUSTED_INDEX_SOURCES and not source.startswith("hanwang:"):
        return True
    return granularity in _ESTIMATED_GRANULARITIES
