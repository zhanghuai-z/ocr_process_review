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
class RoutingObservationBundle(Generic[TPrepass]):
    run_uid: str
    snapshot: LayoutSnapshot
    prepass: TPrepass
    image_hash: str
    layout_fingerprint: str
    block_vl_observations: tuple[BlockVlObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_vl_observations", tuple(self.block_vl_observations))
        if not self.run_uid or not self.image_hash or not self.layout_fingerprint:
            raise ValueError("routing observation bundle requires run and scope identity")
        if any(item.page_uid != self.snapshot.page_uid for item in self.block_vl_observations):
            raise ValueError("routing observation bundle contains a cross-page VL observation")
        block_uids = [item.block_uid for item in self.block_vl_observations]
        if len(set(block_uids)) != len(block_uids):
            raise ValueError("routing observation bundle contains duplicate block observations")

    def vl_observation_for(self, block_uid: str) -> BlockVlObservation | None:
        return next(
            (item for item in self.block_vl_observations if item.block_uid == block_uid),
            None,
        )


__all__ = [
    "BlockAlignmentStatus",
    "BlockObservationAlignment",
    "BlockVlObservation",
    "BlockVlObservationStatus",
    "BlockVlTextRegion",
    "LineCutOwnershipDirective",
    "RoutingObservationBundle",
]
