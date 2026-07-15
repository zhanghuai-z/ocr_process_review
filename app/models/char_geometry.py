"""Typed boundary between native character proposals and trusted geometry."""
from __future__ import annotations

from dataclasses import dataclass


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class NativeGeometryProposal:
    """One native bbox proposal. Recognized text is deliberately excluded."""

    index: int
    bbox: XYXY
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("geometry proposal index must be non-negative")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("geometry proposal requires non-empty bbox")


@dataclass(frozen=True)
class GeometryAtom:
    """One geometrically coherent unit carried into OCR observations."""

    bbox: XYXY
    proposal_indices: tuple[int, ...]
    granularity: str = "char"
    component_indices: tuple[int, ...] = ()
    reason: str = "native_proposal"

    def __post_init__(self) -> None:
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("geometry atom requires non-empty bbox")
        if not self.proposal_indices:
            raise ValueError("geometry atom requires at least one proposal")
        if tuple(sorted(set(self.proposal_indices))) != self.proposal_indices:
            raise ValueError("geometry atom proposal indices must be unique and sorted")
        if self.granularity not in {"char", "token"}:
            raise ValueError(f"unsupported geometry granularity: {self.granularity!r}")


@dataclass(frozen=True)
class GeometryReconcileResult:
    atoms: tuple[GeometryAtom, ...]


__all__ = [
    "GeometryAtom",
    "GeometryReconcileResult",
    "NativeGeometryProposal",
    "XYXY",
]
