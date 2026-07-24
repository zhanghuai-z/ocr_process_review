"""Immutable observations used to compile one page-local OCR routing run.

These values are external evidence and derived alignment facts.  They are not
layout truth and must not be written back into ``Page`` or ``LayoutSnapshot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from .entity_id import ensure_entity_uid
from .layout_snapshot import LayoutSnapshot


XYXY = tuple[int, int, int, int]
TPrepass = TypeVar("TPrepass")


class BlockVlObservationStatus(str, Enum):
    OBSERVED = "observed"
    EMPTY = "empty"


class BlockAlignmentStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    EXACT = "exact"
    MATCHED = "matched"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


class InlineFormulaTextObservationStatus(str, Enum):
    OBSERVED = "observed"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class BlockVlTextRegion:
    index: int
    label: str
    text: str
    bbox: XYXY


@dataclass(frozen=True)
class BlockVlObservation:
    page_uid: str
    block_uid: str
    block_bbox: XYXY
    image_hash: str
    layout_fingerprint: str
    status: BlockVlObservationStatus
    regions: tuple[BlockVlTextRegion, ...] = ()
    source_artifact_uid: str = ""
    source_run_id: str = ""
    raw_response_ref: str = ""
    attempts: int = 0
    uid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "uid", ensure_entity_uid(self.uid, "vlobs"))
        object.__setattr__(self, "regions", tuple(self.regions))
        if not self.page_uid or not self.block_uid:
            raise ValueError("block VL observation requires page_uid and block_uid")
        if self.block_bbox[2] <= self.block_bbox[0] or self.block_bbox[3] <= self.block_bbox[1]:
            raise ValueError("block VL observation requires a non-empty bbox")
        if self.status == BlockVlObservationStatus.EMPTY and self.regions:
            raise ValueError("empty block VL observation cannot contain text regions")

    @property
    def text(self) -> str:
        return "\n".join(region.text for region in self.regions if region.text).strip()


@dataclass(frozen=True)
class LineCutOwnershipDirective:
    block_uid: str
    line_index: int
    bbox: XYXY
    reason: str
    vl_text: str = ""


@dataclass(frozen=True)
class BlockObservationAlignment:
    block_uid: str
    status: BlockAlignmentStatus
    vl_observation_uid: str
    pp_source_indices: tuple[int, ...] = ()
    directives: tuple[LineCutOwnershipDirective, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "pp_source_indices", tuple(self.pp_source_indices))
        object.__setattr__(self, "directives", tuple(self.directives))


@dataclass(frozen=True)
class InlineFormulaTextObservation:
    page_uid: str
    block_uid: str
    bbox: XYXY
    status: InlineFormulaTextObservationStatus
    text: str
    source: str
    source_artifact_uid: str = ""
    raw_response_ref: str = ""
    attempts: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        if not self.page_uid or not self.block_uid:
            raise ValueError("inline formula observation requires page_uid and block_uid")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("inline formula observation requires non-empty geometry")
        if not self.source:
            raise ValueError("inline formula observation requires an explicit source")
        if self.status is InlineFormulaTextObservationStatus.OBSERVED and not self.text:
            raise ValueError("observed inline formula requires text")
        if self.status is InlineFormulaTextObservationStatus.UNRESOLVED and self.text:
            raise ValueError("unresolved inline formula cannot contain text")


@dataclass(frozen=True)
class RoutingObservationBundle(Generic[TPrepass]):
    run_uid: str
    snapshot: LayoutSnapshot
    prepass: TPrepass
    image_hash: str
    layout_fingerprint: str
    block_vl_observations: tuple[BlockVlObservation, ...]
    inline_formula_observations: tuple[InlineFormulaTextObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_vl_observations", tuple(self.block_vl_observations))
        object.__setattr__(
            self,
            "inline_formula_observations",
            tuple(self.inline_formula_observations),
        )
        if not self.run_uid or not self.image_hash or not self.layout_fingerprint:
            raise ValueError("routing observation bundle requires run and scope identity")
        if any(item.page_uid != self.snapshot.page_uid for item in self.block_vl_observations):
            raise ValueError("routing observation bundle contains a cross-page VL observation")
        block_uids = [item.block_uid for item in self.block_vl_observations]
        if len(set(block_uids)) != len(block_uids):
            raise ValueError("routing observation bundle contains duplicate block observations")
        formula_uids = [item.block_uid for item in self.inline_formula_observations]
        if len(set(formula_uids)) != len(formula_uids):
            raise ValueError("routing observation bundle contains duplicate formula observations")
        if any(item.page_uid != self.snapshot.page_uid for item in self.inline_formula_observations):
            raise ValueError("routing observation bundle contains a cross-page formula observation")

    def vl_observation_for(self, block_uid: str) -> BlockVlObservation | None:
        return next(
            (item for item in self.block_vl_observations if item.block_uid == block_uid),
            None,
        )

    def inline_formula_observation_for(
        self,
        block_uid: str,
    ) -> InlineFormulaTextObservation | None:
        return next(
            (item for item in self.inline_formula_observations if item.block_uid == block_uid),
            None,
        )


__all__ = [
    "BlockAlignmentStatus",
    "BlockObservationAlignment",
    "BlockVlObservation",
    "BlockVlObservationStatus",
    "BlockVlTextRegion",
    "LineCutOwnershipDirective",
    "InlineFormulaTextObservation",
    "InlineFormulaTextObservationStatus",
    "RoutingObservationBundle",
]
