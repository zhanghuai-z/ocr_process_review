"""Immutable proof character-index value objects.

The index is derived from one active OCR batch and one proof state. It is never
a write surface and it deliberately carries an explicit unavailable geometry
state instead of inventing coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


GEOMETRY_AVAILABLE = "available"
GEOMETRY_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CharIndexEntry:
    """One proof character backed by an OCR atom, or explicitly unavailable."""

    proof_uid: str
    text_unit_uid: str
    char_index: int
    text: str
    page_uid: str
    page_number: int
    image_path: str
    scope_uid: str
    batch_uid: str
    line_uid: str
    atom_uid: str | None
    atom_index: int | None
    bbox: tuple[int, int, int, int] | None
    geometry_status: str
    unavailable_reason: str = ""
    line_order: int = 0
    atom_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.proof_uid.strip():
            raise ValueError("proof_uid must be non-empty")
        if not self.text_unit_uid.strip():
            raise ValueError("text_unit_uid must be non-empty")
        if self.char_index < 0:
            raise ValueError("char_index must be non-negative")
        if not self.text:
            raise ValueError("text must be non-empty")
        if self.geometry_status not in {GEOMETRY_AVAILABLE, GEOMETRY_UNAVAILABLE}:
            raise ValueError("invalid geometry status")
        if self.geometry_status == GEOMETRY_AVAILABLE:
            if self.atom_uid is None or self.bbox is None:
                raise ValueError("available geometry requires an OCR atom and bbox")
        elif self.bbox is not None:
            raise ValueError("unavailable geometry must not carry a bbox")

    @property
    def available(self) -> bool:
        return self.geometry_status == GEOMETRY_AVAILABLE

    @property
    def key(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class CharIndex:
    """Sorted, read-only character index for one proof state."""

    proof_uid: str
    state_revision: int
    state_fingerprint: str
    batch_uid: str
    entries: tuple[CharIndexEntry, ...] = ()
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(entry.proof_uid != self.proof_uid for entry in entries):
            raise ValueError("index entry belongs to another proof state")
        if self.state_revision < 0:
            raise ValueError("state_revision must be non-negative")
        object.__setattr__(self, "entries", entries)

    def query(self, text: str, *, include_unavailable: bool = False) -> tuple[CharIndexEntry, ...]:
        result = tuple(entry for entry in self.entries if entry.text == text)
        if include_unavailable:
            return result
        return tuple(entry for entry in result if entry.available)

    def same_text(self, text: str, *, include_unavailable: bool = False) -> tuple[CharIndexEntry, ...]:
        return self.query(text, include_unavailable=include_unavailable)

    def unavailable(self) -> tuple[CharIndexEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.available)

    @property
    def unique_texts(self) -> tuple[str, ...]:
        return tuple(sorted({entry.text for entry in self.entries}))

    @property
    def unique_char_count(self) -> int:
        return len(self.unique_texts)

    @property
    def total_char_count(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterable[CharIndexEntry]:
        return iter(self.entries)


__all__ = [
    "CharIndex",
    "CharIndexEntry",
    "GEOMETRY_AVAILABLE",
    "GEOMETRY_UNAVAILABLE",
]
