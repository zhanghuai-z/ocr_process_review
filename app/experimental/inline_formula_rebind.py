"""Experimental formula span diagnostics.

Production formula crop OCR lives in ``app.services.formula_crop_ocr_service``.
This module only compares recognized crop text against parent paragraph spans.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable, Sequence

from app.core.text_classification import is_formula_marker_token
from app.services.formula_crop_ocr_service import FormulaRecognition, normalize_formula_key

XYXY = tuple[int, int, int, int]
FORMULA_PATTERN = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class FormulaSpan:
    span_index: int
    text: str
    normalized_key: str
    start: int
    end: int


@dataclass(frozen=True)
class FormulaBindingSuggestion:
    crop_index: int
    bbox: XYXY
    recognized_text: str
    span_index: int | None
    span_text: str
    status: str
    score: float = 0.0


def suggest_formula_bindings(
    parent_text: str,
    recognitions: Iterable[FormulaRecognition],
) -> list[FormulaBindingSuggestion]:
    """Map crop OCR evidence to parent ``$...$`` spans for diagnostics."""
    spans = formula_spans(parent_text)
    spans_by_key: dict[str, list[FormulaSpan]] = {}
    for span in spans:
        if span.normalized_key:
            spans_by_key.setdefault(span.normalized_key, []).append(span)

    used_span_indices: set[int] = set()
    occurrence_by_key: dict[str, int] = {}
    suggestions: list[FormulaBindingSuggestion] = []
    for recognition in sorted(recognitions, key=lambda item: item.crop_index):
        key = recognition.normalized_key
        candidates = [span for span in spans_by_key.get(key, []) if span.span_index not in used_span_indices]
        if candidates:
            offset = occurrence_by_key.get(key, 0)
            span = candidates[min(offset, len(candidates) - 1)]
            occurrence_by_key[key] = offset + 1
            used_span_indices.add(span.span_index)
            status = "exact" if len(spans_by_key.get(key, [])) == 1 else "duplicate_key_ordered"
            suggestions.append(_bound_suggestion(recognition, span, status, 1.0))
            continue

        fuzzy = _best_fuzzy_span(key, spans, used_span_indices)
        if fuzzy is not None:
            span, score = fuzzy
            used_span_indices.add(span.span_index)
            suggestions.append(_bound_suggestion(recognition, span, "fuzzy", score))
            continue

        suggestions.append(
            FormulaBindingSuggestion(
                crop_index=recognition.crop_index,
                bbox=recognition.bbox,
                recognized_text=recognition.text,
                span_index=None,
                span_text="",
                status="unresolved",
                score=0.0,
            )
        )
    return suggestions


def formula_spans(parent_text: str) -> list[FormulaSpan]:
    spans: list[FormulaSpan] = []
    for match in FORMULA_PATTERN.finditer(parent_text or ""):
        text = match.group(0)
        if is_formula_marker_token(text):
            continue
        spans.append(
            FormulaSpan(
                span_index=len(spans),
                text=text,
                normalized_key=normalize_formula_key(text),
                start=match.start(),
                end=match.end(),
            )
        )
    return spans


def _best_fuzzy_span(
    key: str,
    spans: Sequence[FormulaSpan],
    used_span_indices: set[int],
) -> tuple[FormulaSpan, float] | None:
    if not key:
        return None
    scored: list[tuple[float, FormulaSpan]] = []
    for span in spans:
        if span.span_index in used_span_indices or not span.normalized_key:
            continue
        score = SequenceMatcher(a=key, b=span.normalized_key, autojunk=False).ratio()
        scored.append((score, span))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 0.86:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.04:
        return None
    return scored[0][1], scored[0][0]


def _bound_suggestion(
    recognition: FormulaRecognition,
    span: FormulaSpan,
    status: str,
    score: float,
) -> FormulaBindingSuggestion:
    return FormulaBindingSuggestion(
        crop_index=recognition.crop_index,
        bbox=recognition.bbox,
        recognized_text=recognition.text,
        span_index=span.span_index,
        span_text=span.text,
        status=status,
        score=score,
    )


__all__ = [
    "FormulaBindingSuggestion",
    "FormulaSpan",
    "formula_spans",
    "suggest_formula_bindings",
]
