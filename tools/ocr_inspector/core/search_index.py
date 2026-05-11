"""Stable text-to-IR search index for OCR Inspector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.ocr_inspector.models.ir import BBox, CharNode, DocumentNode, LineNode, PageNode


@dataclass(frozen=True)
class SearchIndexEntry:
    identity: str
    node: CharNode | LineNode
    page: PageNode
    text: str
    kind: str
    bbox: BBox | None
    bbox_source: str
    bbox_granularity: str


def build_text_search_index(doc: DocumentNode, *, include_chars: bool = True, include_lines: bool = True) -> list[SearchIndexEntry]:
    entries: list[SearchIndexEntry] = []
    for page_idx, page in enumerate(doc.pages):
        for line_idx, line in enumerate(page.all_lines):
            if include_chars:
                seen_tokens: set[tuple[Any, str]] = set()
                for char_idx, char in enumerate(line.chars):
                    if getattr(char, "bbox_source", "") != "ocr":
                        continue
                    token = char.token_text or char.char
                    key = (id(line), token)
                    if key in seen_tokens:
                        continue
                    seen_tokens.add(key)
                    entries.append(SearchIndexEntry(
                        identity=f"p{page_idx}:l{line_idx}:token{char_idx}:{token}",
                        node=char,
                        page=page,
                        text=token,
                        kind="char",
                        bbox=char.bbox,
                        bbox_source=char.bbox_source,
                        bbox_granularity=char.bbox_granularity,
                    ))
            if include_lines:
                entries.append(SearchIndexEntry(
                    identity=f"p{page_idx}:l{line_idx}:line",
                    node=line,
                    page=page,
                    text=line.text,
                    kind="line",
                    bbox=line.bbox,
                    bbox_source=getattr(line, "source_field", "") or "overall_ocr_res",
                    bbox_granularity="line",
                ))
    return entries


def query_text_search_index(
    doc: DocumentNode,
    query: str,
    *,
    case_sensitive: bool = False,
    include_chars: bool = True,
    include_lines: bool = True,
) -> list[SearchIndexEntry]:
    if not query:
        return []
    cmp_query = query if case_sensitive else query.lower()
    results: list[SearchIndexEntry] = []
    for entry in build_text_search_index(doc, include_chars=include_chars, include_lines=include_lines):
        cmp_text = entry.text if case_sensitive else entry.text.lower()
        if cmp_query in cmp_text:
            results.append(entry)
    return results
