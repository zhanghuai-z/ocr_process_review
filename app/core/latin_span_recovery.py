"""Exact Latin-token geometry recovery from EngCut/Eng20 character boxes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

XYXY = tuple[int, int, int, int]

LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\-]{1,}")
TEXT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\-]{1,}|[0-9]{2,}(?:[./\-][0-9A-Za-z]+)*")
FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)

LATIN_ENGCUT_BBOX_SOURCE = "hanwang:EngCut:latin_exact"
LATIN_ENGCUT_BBOX_GRANULARITY = "char"
LATIN_ENGCUT_REVIEW_FLAG = "hanwang_latin_engcut_review"
LATIN_ENGCUT_EXACT_STATUS = "latin_token_engcut_line_exact"
LATIN_ENGCUT_VARIANT_STATUS = "latin_token_engcut_variant_exact"
LATIN_ENGCUT_REVERSE_STATUS = "latin_token_engcut_reverse_exact"
LATIN_ENGCUT_MULTILINE_STATUS = "latin_token_engcut_multiline_review"
LATIN_ENGCUT_NOT_FOUND_STATUS = "not_found_in_engcut_line_exact"
LATIN_ENGCUT_INCOMPLETE_BBOX_STATUS = "found_without_complete_engcut_char_bbox"
SLASH_VARIANTS = ("/", "f", "!", "l", "I", "1", "|")


@dataclass(frozen=True)
class LatinToken:
    text: str
    start: int
    end: int
    kind: str = "latin"


@dataclass(frozen=True)
class EngcutChar:
    text: str
    bbox: XYXY | None = None
    code: int = 0
    line_index: int = 0
    group_index: int = 0
    char_index: int = 0


@dataclass(frozen=True)
class LatinBinding:
    token: LatinToken
    status: str
    engcut_start: int = -1
    engcut_end: int = -1
    bbox: XYXY | None = None
    char_bboxes: tuple[XYXY, ...] = ()
    reason: str = ""


def _formula_ranges(text: str) -> list[range]:
    return [range(match.start(), match.end()) for match in FORMULA_SPAN_RE.finditer(text or "")]


def _overlaps_any_formula(start: int, end: int, formula_ranges: Iterable[range]) -> bool:
    return any(start < value.stop and end > value.start for value in formula_ranges)


def latin_token_spans(text: str, *, skip_formula_spans: bool = True) -> list[LatinToken]:
    """Return Latin-like tokens in source order.

    Single letters are intentionally ignored here. They are too common in prose
    and formula-adjacent text to be a safe automatic geometry target.
    """
    value = text or ""
    formula_ranges = _formula_ranges(value) if skip_formula_spans else []
    tokens: list[LatinToken] = []
    for match in LATIN_TOKEN_RE.finditer(value):
        if skip_formula_spans and _overlaps_any_formula(match.start(), match.end(), formula_ranges):
            continue
        tokens.append(LatinToken(text=match.group(0), start=match.start(), end=match.end()))
    return tokens


def has_latin_token(text: str) -> bool:
    return bool(latin_token_spans(text))


def normalize_text_token(text: str) -> str:
    value = (text or "").strip()
    value = value.translate(str.maketrans({
        "／": "/",
        "⁄": "/",
        "∕": "/",
        "－": "-",
        "—": "-",
        "–": "-",
        "＋": "+",
        "＆": "&",
        "．": ".",
    }))
    return value.strip(" \t\r\n,，.。;；:：()（）[]【】{}")


def text_token_spans(text: str, *, skip_formula_spans: bool = True) -> list[LatinToken]:
    value = text or ""
    formula_ranges = _formula_ranges(value) if skip_formula_spans else []
    tokens: list[LatinToken] = []
    for match in TEXT_TOKEN_RE.finditer(value):
        if skip_formula_spans and _overlaps_any_formula(match.start(), match.end(), formula_ranges):
            continue
        token = normalize_text_token(match.group(0))
        if len(token) < 2:
            continue
        compact = token.replace(".", "").replace("/", "").replace("-", "")
        kind = "number" if compact.isdigit() else "latin"
        tokens.append(LatinToken(text=token, start=match.start(), end=match.end(), kind=kind))
    return tokens


def token_variants(token: str) -> list[str]:
    token = normalize_text_token(token)
    variants = [token]
    if "/" in token:
        expanded = [""]
        for char in token:
            replacements = SLASH_VARIANTS if char == "/" else (char,)
            expanded = [prefix + replacement for prefix in expanded for replacement in replacements]
        variants.extend(expanded)
        variants.append(token.replace("/", ""))
    seen: set[str] = set()
    result: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _code_to_text(code: int) -> str:
    if 32 <= code <= 126:
        return chr(code)
    return "~"


def _code_to_int(code: object) -> int:
    if isinstance(code, str):
        return int(code, 0) if code.startswith(("0x", "0X")) else int(code, 16)
    return int(code)


def _bbox_from_raw(raw: object) -> XYXY | None:
    if not isinstance(raw, dict):
        return None
    try:
        left = int(raw.get("left", raw.get("x", 0)))
        top = int(raw.get("top", raw.get("y", 0)))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw.get("width", 0))
            bottom = top + int(raw.get("height", 0))
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def engcut_chars_from_payload(payload: dict[str, Any]) -> list[EngcutChar]:
    """Parse the JSON emitted by scripts/eng20_probe.cs."""
    chars: list[EngcutChar] = []
    for line_index, line in enumerate(payload.get("lines") or []):
        if not isinstance(line, dict):
            continue
        for group_index, group in enumerate(line.get("groups") or []):
            if not isinstance(group, dict):
                continue
            for char_index, char in enumerate(group.get("chars") or []):
                if not isinstance(char, dict):
                    continue
                codes = char.get("codes") or []
                try:
                    code = _code_to_int(codes[0]) if codes else 0
                except (TypeError, ValueError):
                    code = 0
                bbox = _bbox_from_raw(char.get("bbox"))
                chars.append(
                    EngcutChar(
                        text=_code_to_text(code),
                        bbox=bbox,
                        code=code,
                        line_index=line_index,
                        group_index=group_index,
                        char_index=int(char.get("index", char_index) or 0),
                    )
                )
    return chars


def offset_engcut_chars(chars: Iterable[EngcutChar], *, dx: int, dy: int) -> list[EngcutChar]:
    shifted: list[EngcutChar] = []
    for char in chars:
        bbox = None
        if char.bbox is not None:
            x1, y1, x2, y2 = char.bbox
            bbox = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        shifted.append(
            EngcutChar(
                text=char.text,
                bbox=bbox,
                code=char.code,
                line_index=char.line_index,
                group_index=char.group_index,
                char_index=char.char_index,
            )
        )
    return shifted


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def bind_latin_tokens_to_engcut_chars(
    line_text: str,
    engcut_chars: list[EngcutChar],
) -> list[LatinBinding]:
    """Bind line Latin tokens to EngCut chars by exact substring only."""
    tokens = latin_token_spans(line_text, skip_formula_spans=False)
    if not tokens:
        return []

    formula_ranges = _formula_ranges(line_text)
    engcut_text = "".join(char.text for char in engcut_chars)
    cursor = 0
    bindings: list[LatinBinding] = []
    for token in tokens:
        visible_token = not _overlaps_any_formula(token.start, token.end, formula_ranges)
        found = engcut_text.find(token.text, cursor)
        if found < 0:
            if visible_token:
                bindings.append(
                    LatinBinding(
                        token=token,
                        status=LATIN_ENGCUT_NOT_FOUND_STATUS,
                        reason="token_not_found_after_source_cursor",
                    )
                )
            continue

        end = found + len(token.text)
        char_slice = engcut_chars[found:end]
        boxes = [char.bbox for char in char_slice if char.bbox is not None]
        if not visible_token:
            cursor = end
            continue
        if len(char_slice) != len(token.text) or len(boxes) != len(token.text):
            bindings.append(
                LatinBinding(
                    token=token,
                    status=LATIN_ENGCUT_INCOMPLETE_BBOX_STATUS,
                    engcut_start=found,
                    engcut_end=end,
                    reason="engcut_match_has_missing_char_bbox",
                )
            )
            cursor = end
            continue

        binding_boxes = tuple(box for box in boxes if box is not None)
        bindings.append(
            LatinBinding(
                token=token,
                status=LATIN_ENGCUT_EXACT_STATUS,
                engcut_start=found,
                engcut_end=end,
                bbox=_union(list(binding_boxes)),
                char_bboxes=binding_boxes,
            )
        )
        cursor = end
    return bindings
