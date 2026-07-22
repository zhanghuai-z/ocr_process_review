"""Read-only confidence projections used by the proof views.

This module deliberately depends on application proof DTOs only.  It does not
look up OCR records or proof state and it cannot refresh or mutate a session.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from app.application.proof_workspace import ProofAtomView, ProofLineView, ProofPageView


def normalize_confidence(raw: object) -> float | None:
    """Return a finite score in ``0..1`` or ``None`` when unavailable."""

    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value <= 0:
        return None
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    if value > 1.0:
        return None
    return max(0.0, value)


@dataclass(frozen=True, slots=True)
class ProofCharView:
    """One proof character and the active OCR evidence available for it."""

    proof_uid: str
    text_unit_uid: str
    char_index: int
    text: str
    page_uid: str
    page_number: int
    image_path: str
    line_uid: str | None
    region_uid: str | None
    atom_uid: str | None
    atom_index: int | None
    bbox: tuple[int, int, int, int] | None
    confidence: float | None
    ocr_char: str | None
    available: bool
    line_bbox: tuple[int, int, int, int] | None = None

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid


def build_char_views(
    line: ProofLineView,
    page: ProofPageView,
) -> tuple[ProofCharView, ...]:
    """Project proof characters onto the atoms already present in a line view.

    The explicit ``atom.char_span`` carried by the immutable view is the
    only character-to-atom relation used here.  Atoms without a span claim
    no characters.  A multi-character carrier may establish text ownership,
    but its shared bbox is not single-character geometry and therefore is not
    exposed as a character crop.  Sequential guessing or splitting a word
    bbox is deliberately not a fallback: either would invent geometry.
    """

    if line.page_uid != page.page_uid:
        raise ValueError("line and page views must reference the same page UID")

    by_char_index: dict[int, tuple[ProofAtomView, int]] = {}
    for atom in sorted(line.atoms, key=lambda item: (item.atom_index, item.atom_uid)):
        span = atom.char_span
        if span is None:
            continue
        start, end = span
        for offset, char_index in enumerate(range(start, end)):
            if char_index >= len(line.proof_text):
                raise ValueError(
                    f"atom {atom.atom_uid!r} char span exceeds proof text length"
                )
            if char_index in by_char_index:
                raise ValueError(
                    f"proof character index {char_index} maps to multiple atoms"
                )
            by_char_index[char_index] = (atom, offset)

    result: list[ProofCharView] = []
    for char_index, text_char in enumerate(line.proof_text):
        mapped = by_char_index.get(char_index)
        atom, offset = mapped if mapped is not None else (None, -1)
        exact_geometry = (
            atom is not None
            and atom.char_span is not None
            and atom.char_span[1] - atom.char_span[0] == 1
        )
        ocr_char = (
            atom.text[offset]
            if atom is not None and 0 <= offset < len(atom.text)
            else None
        )
        result.append(
            ProofCharView(
                proof_uid=line.proof_uid,
                text_unit_uid=line.text_unit_uid,
                char_index=char_index,
                text=text_char,
                page_uid=page.page_uid,
                page_number=page.page_number,
                image_path=page.image_path,
                line_uid=atom.line_uid if atom is not None else line.line_uid,
                region_uid=atom.region_uid if atom is not None else line.region_uid,
                atom_uid=atom.atom_uid if atom is not None else None,
                atom_index=atom.atom_index if atom is not None else None,
                bbox=atom.bbox if exact_geometry else None,
                confidence=normalize_confidence(atom.confidence) if atom is not None else None,
                ocr_char=ocr_char,
                available=exact_geometry,
                line_bbox=line.bbox,
            )
        )

    return tuple(result)


def char_confidence(view: ProofCharView | None) -> float | None:
    """Return the score carried by one immutable character view."""

    return None if view is None else normalize_confidence(view.confidence)


def line_confidence(view: ProofLineView | None) -> float | None:
    """Return the aggregate score carried by one immutable line view."""

    return None if view is None else normalize_confidence(view.confidence)


__all__ = [
    "ProofCharView",
    "build_char_views",
    "char_confidence",
    "line_confidence",
    "normalize_confidence",
]
