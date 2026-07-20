"""Pure Unicode text classification shared by layout and OCR routing."""
from __future__ import annotations

import re
import unicodedata


_FORMULA_SYMBOLS = set("=+-−*/×÷^_()[]{}<>≤≥±√∑∫∞≈≠πΠαβγδθλμσΩω|")
_SUPERSCRIPT_MARKER_RE = re.compile(r"^\^\{(?:\*{1,3}|[①②③④⑤⑥⑦⑧⑨⑩])\}$")


def is_cjk_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def is_formula_char(char: str) -> bool:
    if len(char) != 1 or char.isspace() or is_cjk_char(char):
        return False
    if char in _FORMULA_SYMBOLS:
        return True
    category = unicodedata.category(char)
    return (char.isascii() and (char.isalpha() or char.isdigit())) or category.startswith("S")


def is_formula_token(text: str) -> bool:
    compact = "".join(char for char in str(text) if not char.isspace())
    if not compact or compact.isdigit() or any(is_cjk_char(char) for char in compact):
        return False
    return any(char.isalpha() for char in compact) and all(
        is_formula_char(char) or unicodedata.category(char).startswith("P")
        for char in compact
    )


def is_formula_marker_token(text: str) -> bool:
    compact = "".join(char for char in str(text or "") if not char.isspace()).strip("$")
    return bool(compact and _SUPERSCRIPT_MARKER_RE.fullmatch(compact))


def classify_text(text: str) -> str:
    compact = "".join(char for char in str(text) if not char.isspace())
    if not compact:
        return "other"
    if compact.isdigit():
        return "digit"
    if is_formula_token(compact):
        return "formula"
    categories = tuple(unicodedata.category(char) for char in compact)
    if all(category.startswith("P") for category in categories):
        return "punct"
    if all(category.startswith("S") for category in categories):
        return "symbol"
    if any(category.startswith("L") for category in categories):
        return "text"
    return "other"


__all__ = [
    "classify_text",
    "is_cjk_char",
    "is_formula_char",
    "is_formula_marker_token",
    "is_formula_token",
]
