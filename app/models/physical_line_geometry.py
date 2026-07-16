"""Typed contracts for page-local physical text-line geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


XYXY = tuple[int, int, int, int]
TEXT_AXIS_HORIZONTAL = "horizontal"
TEXT_AXIS_VERTICAL = "vertical"
_TEXT_AXES = frozenset({TEXT_AXIS_HORIZONTAL, TEXT_AXIS_VERTICAL})


@dataclass(frozen=True)
class LineGeometrySeed:
    """One immutable external line proposal stripped of OCR text."""

    source_index: int
    bbox: XYXY
    text_axis: str = TEXT_AXIS_HORIZONTAL
    orientation_angle: int = -1
    has_word_geometry: bool = False

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("line geometry seed requires a non-negative source index")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("line geometry seed requires a non-empty bbox")
        if self.text_axis not in _TEXT_AXES:
            raise ValueError(f"unsupported line text axis: {self.text_axis!r}")
        if self.orientation_angle not in {-1, 0, 180}:
            raise ValueError(
                f"unsupported line orientation angle: {self.orientation_angle}"
            )


@dataclass(frozen=True)
class LineGeometryContext:
    """Layout ownership and exclusions used to resolve one group of rows."""

    block_uid: str
    block_bbox: XYXY
    seeds: tuple[LineGeometrySeed, ...]
    excluded_bboxes: tuple[XYXY, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "excluded_bboxes", tuple(self.excluded_bboxes))
        if not self.block_uid:
            raise ValueError("line geometry context requires a block uid")
        if self.block_bbox[2] <= self.block_bbox[0] or self.block_bbox[3] <= self.block_bbox[1]:
            raise ValueError("line geometry context requires a non-empty block bbox")


@dataclass(frozen=True)
class ResolvedPhysicalLine:
    """One physical row backed by uniquely selected foreground components."""

    owner_block_uid: str
    representative_index: int
    source_indices: tuple[int, ...]
    bbox: XYXY
    text_axis: str
    orientation_angle: int
    has_word_geometry: bool = False
    component_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_indices", tuple(self.source_indices))
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        if not self.owner_block_uid:
            raise ValueError("resolved physical line requires an owner block uid")
        if not self.source_indices:
            raise ValueError("resolved physical line requires source indices")
        if self.representative_index not in self.source_indices:
            raise ValueError("representative index must belong to the resolved source indices")
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ValueError("resolved physical line source indices must be unique")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("resolved physical line requires a non-empty bbox")
        if self.text_axis not in _TEXT_AXES:
            raise ValueError(f"unsupported resolved text axis: {self.text_axis!r}")
        if self.orientation_angle not in {-1, 0, 180}:
            raise ValueError(
                f"unsupported resolved orientation angle: {self.orientation_angle}"
            )


@dataclass(frozen=True)
class PhysicalLineResolution:
    """Immutable page result with source-index lookup."""

    rows: tuple[ResolvedPhysicalLine, ...]
    _rows_by_source_index: Mapping[int, ResolvedPhysicalLine] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        object.__setattr__(self, "rows", rows)
        by_source_index: dict[int, ResolvedPhysicalLine] = {}
        component_owners: dict[int, int] = {}
        for row in rows:
            for source_index in row.source_indices:
                if source_index in by_source_index:
                    raise ValueError(f"duplicate physical source index: {source_index}")
                by_source_index[source_index] = row
            for component_index in row.component_indices:
                owner = component_owners.setdefault(component_index, row.representative_index)
                if owner != row.representative_index:
                    raise ValueError(
                        "foreground component has multiple physical-line owners: "
                        f"component={component_index} owners={owner},{row.representative_index}"
                    )
        object.__setattr__(self, "_rows_by_source_index", MappingProxyType(by_source_index))

    def row_for_source_index(self, source_index: int) -> ResolvedPhysicalLine | None:
        return self._rows_by_source_index.get(source_index)


__all__ = [
    "LineGeometryContext",
    "LineGeometrySeed",
    "PhysicalLineResolution",
    "ResolvedPhysicalLine",
    "TEXT_AXIS_HORIZONTAL",
    "TEXT_AXIS_VERTICAL",
    "XYXY",
]
