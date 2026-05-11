"""Replaceable OCR Inspector core seam.

UI code should consume IR and diagnostics produced here instead of deriving
Paddle semantics directly from raw JSON.
"""
from .paddle_seam import (
    CoreDiagnostic,
    PaddleCoreResult,
    build_paddle_document,
    canonical_bbox,
    canonical_polygon,
    raw_contains_word_regions,
)
from .search_index import SearchIndexEntry, build_text_search_index, query_text_search_index

__all__ = [
    "CoreDiagnostic",
    "PaddleCoreResult",
    "SearchIndexEntry",
    "build_paddle_document",
    "build_text_search_index",
    "canonical_bbox",
    "canonical_polygon",
    "query_text_search_index",
    "raw_contains_word_regions",
]
