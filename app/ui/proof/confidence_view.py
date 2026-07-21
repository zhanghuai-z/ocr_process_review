"""Read-only confidence projections used by the proof views.

This module deliberately depends on application proof DTOs only.  It does not
look up OCR records or proof state and it cannot refresh or mutate a session.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from app.application.proof_workspace import ProofLineView, ProofPageView


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

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid


def build_char_views(
    line: ProofLineView,
    page: ProofPageView,
) -> tuple[ProofCharView, ...]:
    """Project proof characters onto the atoms already present in a line view.

    Atom text is the only character-to-atom relation carried by the immutable
    application view.  Characters beyond that relation remain visible but are
    explicitly marked unavailable; no OCR record or legacy index is queried.
    """

    if line.page_uid != page.page_uid:
        raise ValueError("line and page views must reference the same page UID")

    atoms = tuple(sorted(line.atoms, key=lambda atom: (atom.atom_index, atom.atom_uid)))
    result: list[ProofCharView] = []
    atom_position = 0
    atom_offset = 0
    for char_index, text_char in enumerate(line.proof_text):
        atom = None
        while atom_position < len(atoms):
            candidate = atoms[atom_position]
            if atom_offset < len(candidate.text):
                atom = candidate
                break
            atom_position += 1
            atom_offset = 0
        if atom is None:
            result.append(
                ProofCharView(
                    proof_uid=line.proof_uid,
                    text_unit_uid=line.text_unit_uid,
                    char_index=char_index,
                    text=text_char,
                    page_uid=page.page_uid,
                    page_number=page.page_number,
                    image_path=page.image_path,
                    line_uid=line.line_uid,
                    region_uid=line.region_uid,
                    atom_uid=None,
                    atom_index=None,
                    bbox=None,
                    confidence=None,
                    ocr_char=None,
                    available=False,
                )
            )
            continue
        ocr_char = atom.text[atom_offset]
        result.append(
            ProofCharView(
                proof_uid=line.proof_uid,
                text_unit_uid=line.text_unit_uid,
                char_index=char_index,
                text=text_char,
                page_uid=page.page_uid,
                page_number=page.page_number,
                image_path=page.image_path,
                line_uid=atom.line_uid,
                region_uid=atom.region_uid,
                atom_uid=atom.atom_uid,
                atom_index=atom.atom_index,
                bbox=atom.bbox,
                confidence=normalize_confidence(atom.confidence),
                ocr_char=ocr_char,
                available=True,
            )
        )
        atom_offset += 1

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
