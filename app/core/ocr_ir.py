"""OCR intermediate representation helpers.

OCR_IR separates raw engine fields from the proof-facing objects so adapters can
preserve source semantics before converting into project models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import unicodedata
import re
from typing import Any, Literal, Optional

from app.models import BBox
from app.models.entity_id import ensure_entity_uid

OCR_IR_SOURCE_REC_TEXT = "overall_ocr_res.rec_texts"

OcrIrKind = Literal["text", "digit", "formula", "punct", "symbol", "other"]

_FORMULA_SYMBOLS = set("=+-−*/×÷^_()[]{}<>≤≥±√∑∫∞≈≠πΠαβγδθλμσΩω|")
_SUPERSCRIPT_MARKER_RE = re.compile(r"^\^\{(?:\*{1,3}|[①②③④⑤⑥⑦⑧⑨⑩])\}$")


def is_cjk_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def is_formula_char(char: str) -> bool:
    if not char or char.isspace() or is_cjk_char(char):
        return False
    if char in _FORMULA_SYMBOLS:
        return True
    category = unicodedata.category(char)
    return (
        char.isascii() and (char.isalpha() or char.isdigit())
    ) or category.startswith("S")


def is_formula_token(text: str) -> bool:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if not compact or compact.isdigit():
        return False
    if any(is_cjk_char(ch) for ch in compact):
        return False
    return any(ch.isalpha() for ch in compact) and all(
        is_formula_char(ch) or unicodedata.category(ch).startswith("P")
        for ch in compact
    )


def is_formula_marker_token(text: str) -> bool:
    """Return True for visual footnote/significance markers misreported as formula.

    Paddle can emit inline_formula boxes for superscript markers such as
    ``$ ^{*} $`` or ``$ ^{②} $``.  They should stay in parent text/table
    content, but not drive formula-debug routing or editable formula boxes.
    """
    compact = "".join(ch for ch in str(text or "") if not ch.isspace())
    if not compact:
        return False
    compact = compact.strip("$")
    return bool(_SUPERSCRIPT_MARKER_RE.fullmatch(compact))


def classify_ir_text(text: str) -> OcrIrKind:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if not compact:
        return "other"
    if compact.isdigit():
        return "digit"
    if is_formula_token(compact):
        return "formula"
    categories = [unicodedata.category(ch) for ch in compact]
    if all(category.startswith("P") for category in categories):
        return "punct"
    if all(category.startswith("S") for category in categories):
        return "symbol"
    if any(category.startswith("L") for category in categories):
        return "text"
    return "other"


@dataclass(frozen=True)
class OcrIrToken:
    text: str
    bbox: Optional[BBox]
    row_index: int
    token_index: int
    raw_region: Any = None
    confidence: Optional[float] = None
    kind: OcrIrKind = "other"
    bbox_source: str = "ocr"
    bbox_granularity: str = "char"


@dataclass
class OcrIrLine:
    text: str
    confidence: float
    bbox: BBox
    source_text: str
    tokens: list[OcrIrToken] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)


@dataclass
class OcrRun:
    """A typed OCR observation run.

    ``OcrIrLine``/``OcrIrToken`` remain the line/token payload.  The run adds
    identity and engine/layout metadata so OCR observations can be carried
    without pretending that proof-facing ``Line`` is the source object.
    """
    engine: str
    lines: list[OcrIrLine] = field(default_factory=list)
    page_uid: str = ""
    block_uid: str = ""
    engine_version: str = ""
    input_layout_revision: int = 0
    created_at: float = field(default_factory=time.time)
    uid: str = ""

    def __post_init__(self) -> None:
        self.uid = ensure_entity_uid(self.uid, "ocrrun")


OcrToken = OcrIrToken
OcrLine = OcrIrLine


def build_ocr_run(
    *,
    engine: str,
    lines: list[OcrIrLine],
    page_uid: str = "",
    block_uid: str = "",
    engine_version: str = "",
    input_layout_revision: int = 0,
) -> OcrRun:
    return OcrRun(
        engine=engine,
        lines=list(lines),
        page_uid=page_uid,
        block_uid=block_uid,
        engine_version=engine_version,
        input_layout_revision=input_layout_revision,
    )
