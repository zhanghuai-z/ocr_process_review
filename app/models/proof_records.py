"""Immutable proof records independent from OCR observations."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
from typing import Any


def _required_uid(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable UID")
    return value


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("revision must be a non-negative integer")
    return value


def _range(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _range_pair(start: int, end: int, field_name: str) -> None:
    if end < start:
        raise ValueError(f"{field_name} end must not precede start")


def _canonical(value: object) -> object:
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
            if item.name not in {"revision", "fingerprint"}
        }
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot fingerprint value of type {type(value).__name__}")


def _set_fingerprint(record: object) -> None:
    payload = _canonical(record)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    object.__setattr__(record, "fingerprint", hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class ProofAnchorSnapshot:
    """The proof-facing anchor version used to build alignment."""

    project_uid: str
    uid: str
    scope_uid: str
    layout_fingerprint: str
    source_fingerprint: str
    anchor_revision: int
    geometry_fingerprint: str = ""
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.scope_uid, "scope_uid")
        _required_text(self.layout_fingerprint, "layout_fingerprint")
        _required_text(self.source_fingerprint, "source_fingerprint")
        _range(self.anchor_revision, "anchor_revision")
        _text(self.geometry_fingerprint, "geometry_fingerprint")
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)

    @property
    def anchor_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class ProofAlignmentSegment:
    """A coarse source-to-proof alignment interval."""

    project_uid: str
    uid: str
    anchor_uid: str
    text_unit_uid: str
    source_line_uids: tuple[str, ...]
    source_start: int
    source_end: int
    proof_start: int
    proof_end: int
    kind: str = "text"
    status: str = "aligned"
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.anchor_uid, "anchor_uid")
        _required_uid(self.text_unit_uid, "text_unit_uid")
        source_line_uids = tuple(
            _required_uid(item, "source_line_uid") for item in self.source_line_uids
        )
        if not source_line_uids:
            raise ValueError("source_line_uids must contain at least one OCR line UID")
        if len(set(source_line_uids)) != len(source_line_uids):
            raise ValueError("source_line_uids contain duplicate UIDs")
        object.__setattr__(self, "source_line_uids", source_line_uids)
        source_start = _range(self.source_start, "source_start")
        source_end = _range(self.source_end, "source_end")
        proof_start = _range(self.proof_start, "proof_start")
        proof_end = _range(self.proof_end, "proof_end")
        _range_pair(source_start, source_end, "source range")
        _range_pair(proof_start, proof_end, "proof range")
        _required_text(self.kind, "kind")
        _required_text(self.status, "status")
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)

    @property
    def segment_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class ProofAlignmentSlice:
    """A fine-grained slice belonging to one alignment segment."""

    project_uid: str
    uid: str
    segment_uid: str
    source_start: int
    source_end: int
    proof_start: int
    proof_end: int
    source_text: str
    proof_text: str
    status: str = "aligned"
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.segment_uid, "segment_uid")
        source_start = _range(self.source_start, "source_start")
        source_end = _range(self.source_end, "source_end")
        proof_start = _range(self.proof_start, "proof_start")
        proof_end = _range(self.proof_end, "proof_end")
        _range_pair(source_start, source_end, "source range")
        _range_pair(proof_start, proof_end, "proof range")
        _text(self.source_text, "source_text")
        _text(self.proof_text, "proof_text")
        _required_text(self.status, "status")
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)

    @property
    def slice_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class ProofTextUnit:
    """One stable, human-editable text unit independent from OCR geometry."""

    project_uid: str
    uid: str
    order: int
    text: str
    status: str = "unchecked"
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _range(self.order, "order")
        _text(self.text, "text")
        _required_text(self.status, "status")
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)


@dataclass(frozen=True, slots=True)
class ProofState:
    """Human proof state whose source anchor is explicit and versioned."""

    project_uid: str
    uid: str
    anchor_snapshot: ProofAnchorSnapshot
    text_units: tuple[ProofTextUnit, ...] = ()
    alignment_segments: tuple[ProofAlignmentSegment, ...] = ()
    alignment_slices: tuple[ProofAlignmentSlice, ...] = ()
    rebind_required: bool = False
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        if not isinstance(self.anchor_snapshot, ProofAnchorSnapshot):
            raise TypeError("anchor_snapshot requires ProofAnchorSnapshot")
        if self.anchor_snapshot.project_uid != self.project_uid:
            raise ValueError("anchor_snapshot belongs to another project")
        text_units = tuple(self.text_units)
        segments = tuple(self.alignment_segments)
        slices = tuple(self.alignment_slices)
        if any(not isinstance(item, ProofTextUnit) for item in text_units):
            raise TypeError("text_units require ProofTextUnit values")
        if any(not isinstance(item, ProofAlignmentSegment) for item in segments):
            raise TypeError("alignment_segments require ProofAlignmentSegment values")
        if any(not isinstance(item, ProofAlignmentSlice) for item in slices):
            raise TypeError("alignment_slices require ProofAlignmentSlice values")
        if len({item.uid for item in text_units}) != len(text_units):
            raise ValueError("text_units contain duplicate UIDs")
        if len({item.order for item in text_units}) != len(text_units):
            raise ValueError("text_units contain duplicate order values")
        if len({item.uid for item in segments}) != len(segments):
            raise ValueError("alignment_segments contain duplicate UIDs")
        if len({item.uid for item in slices}) != len(slices):
            raise ValueError("alignment_slices contain duplicate UIDs")
        segment_uids = {item.uid for item in segments}
        text_unit_uids = {item.uid for item in text_units}
        for item in text_units:
            if item.project_uid != self.project_uid:
                raise ValueError("proof text unit belongs to another project")
        for segment in segments:
            if segment.project_uid != self.project_uid:
                raise ValueError("alignment segment belongs to another project")
            if segment.anchor_uid != self.anchor_snapshot.uid:
                raise ValueError("alignment segment is bound to another anchor")
            if segment.text_unit_uid not in text_unit_uids:
                raise ValueError("alignment segment references an unknown text unit")
        aligned_unit_uids = [segment.text_unit_uid for segment in segments]
        if len(set(aligned_unit_uids)) != len(aligned_unit_uids):
            raise ValueError("multiple alignment segments reference one text unit")
        for item in slices:
            if item.project_uid != self.project_uid:
                raise ValueError("alignment slice belongs to another project")
            if item.segment_uid not in segment_uids:
                raise ValueError("alignment slice references an unknown segment")
        if not isinstance(self.rebind_required, bool):
            raise TypeError("rebind_required must be bool")
        object.__setattr__(self, "text_units", tuple(sorted(text_units, key=lambda item: item.order)))
        object.__setattr__(self, "alignment_segments", segments)
        object.__setattr__(self, "alignment_slices", slices)
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)

    @property
    def proof_uid(self) -> str:
        return self.uid


__all__ = [
    "ProofAlignmentSegment",
    "ProofAlignmentSlice",
    "ProofAnchorSnapshot",
    "ProofState",
    "ProofTextUnit",
]
