"""Helpers for reconstructing proof display text from OCR char carriers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.block_attributes import normalize_source_label
from app.models import Char


@dataclass(frozen=True)
class CharDisplaySpan:
    text: str
    start: int
    end: int
    char_indices: tuple[int, ...]


def chars_display_spans(chars: Sequence[Char]) -> list[CharDisplaySpan]:
    """Return display spans represented by ``chars``.

    A single OCR ``Char`` may carry a word or inline formula token. Multiple
    word-level chars can also share the same token/bbox.  Proofing code should
    compare and edit the display text represented by those carriers, not just
    ``len(chars)``.
    """
    spans: list[CharDisplaySpan] = []
    idx = 0
    cursor = 0
    while idx < len(chars):
        char = chars[idx]
        token = char_display_text(char)
        end_idx = idx + 1
        if _is_shared_word_carrier(char, token):
            bbox_key = _bbox_key(char)
            while end_idx < len(chars):
                other = chars[end_idx]
                if not _is_shared_word_carrier(other, token):
                    break
                if char_display_text(other) != token:
                    break
                if _bbox_key(other) != bbox_key:
                    break
                end_idx += 1
        span_end = cursor + len(token)
        spans.append(
            CharDisplaySpan(
                text=token,
                start=cursor,
                end=span_end,
                char_indices=tuple(range(idx, end_idx)),
            )
        )
        cursor = span_end
        idx = end_idx
    return spans


def chars_display_text(chars: Sequence[Char]) -> str:
    return "".join(span.text for span in chars_display_spans(chars))


def char_display_text(char: Char) -> str:
    """Return the text this char contributes to the visible proof line."""
    if is_display_carrier(char):
        return str(char.token_text or char.char or "")
    return str(char.char or "")


def is_display_carrier(char: Char) -> bool:
    """Return True when one Char carries a multi-character display token.

    EngCut exact keeps the source word in ``token_text`` for each char-level
    bbox.  That token is metadata; it must not replace the single-character
    ``char`` in proof display/sync logic.
    """
    char_text = str(char.char or "")
    granularity = normalize_source_label(char.bbox_granularity)
    return granularity in {"word", "formula"} or len(char_text) > 1


def _is_shared_word_carrier(char: Char, token: str) -> bool:
    return bool(token) and len(token) > 1 and normalize_source_label(char.bbox_granularity) == "word"


def _bbox_key(char: Char):
    return char.bbox.to_dict() if char.bbox is not None else None
