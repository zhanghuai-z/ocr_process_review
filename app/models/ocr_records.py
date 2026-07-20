"""Immutable records for a project-scoped OCR observation store.

The records in this module are machine observations.  They deliberately use
stable string identifiers and value types only; no runtime layout or proof
model is part of the contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math


XYXY = tuple[int, int, int, int]


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


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")
    return result


def _bbox(value: object, field_name: str = "bbox") -> XYXY:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError(f"{field_name} must contain four integer coordinates")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{field_name} coordinates must be integers")
    result = tuple(value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field_name} must be non-empty")
    return result  # type: ignore[return-value]


def _optional_bbox(value: object, field_name: str = "bbox") -> XYXY | None:
    if value is None:
        return None
    return _bbox(value, field_name)


def _uid_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a sequence of UIDs")
    result = tuple(_required_uid(item, f"{field_name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate UIDs")
    return result


def _metadata(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("metadata must be a sequence of string pairs")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("metadata must contain string pairs")
        key, item_value = item
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(item_value, str):
            raise TypeError("metadata values must be strings")
        result.append((key, item_value))
    if len({key for key, _value in result}) != len(result):
        raise ValueError("metadata must not contain duplicate keys")
    return tuple(result)


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
class OcrRun:
    """One immutable OCR execution and its input scope."""

    project_uid: str
    uid: str
    engine: str
    layout_fingerprint: str
    input_fingerprint: str
    status: str = "completed"
    engine_version: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_text(self.engine, "engine")
        _required_text(self.layout_fingerprint, "layout_fingerprint")
        _required_text(self.input_fingerprint, "input_fingerprint")
        _required_text(self.status, "status")
        _text(self.engine_version, "engine_version")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        _set_fingerprint(self)

    @property
    def run_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrRegion:
    """One OCR input region observed during a run."""

    project_uid: str
    uid: str
    run_uid: str
    page_uid: str
    bbox: XYXY
    kind: str
    order: int = 0
    label: str = ""
    text_hint: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.run_uid, "run_uid")
        _required_uid(self.page_uid, "page_uid")
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        _required_text(self.kind, "kind")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
        _text(self.label, "label")
        _text(self.text_hint, "text_hint")
        _set_fingerprint(self)

    @property
    def region_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One OCR line observation, independent from any proof line object."""

    project_uid: str
    uid: str
    run_uid: str
    region_uid: str
    page_uid: str
    text: str
    bbox: XYXY
    confidence: float
    order: int = 0
    atom_uids: tuple[str, ...] = ()
    source_fingerprint: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.run_uid, "run_uid")
        _required_uid(self.region_uid, "region_uid")
        _required_uid(self.page_uid, "page_uid")
        _text(self.text, "text")
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
        object.__setattr__(self, "atom_uids", _uid_tuple(self.atom_uids, "atom_uids"))
        _text(self.source_fingerprint, "source_fingerprint")
        _set_fingerprint(self)

    @property
    def line_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrAtom:
    """One atomic OCR observation within a line."""

    project_uid: str
    uid: str
    run_uid: str
    region_uid: str
    line_uid: str
    index: int
    text: str
    bbox: XYXY
    confidence: float
    candidate_uids: tuple[str, ...] = ()
    source_fingerprint: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.run_uid, "run_uid")
        _required_uid(self.region_uid, "region_uid")
        _required_uid(self.line_uid, "line_uid")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        _text(self.text, "text")
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "candidate_uids", _uid_tuple(self.candidate_uids, "candidate_uids"))
        _text(self.source_fingerprint, "source_fingerprint")
        _set_fingerprint(self)

    @property
    def atom_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrCandidate:
    """One ranked machine candidate for an OCR atom."""

    project_uid: str
    uid: str
    run_uid: str
    region_uid: str
    line_uid: str
    atom_uid: str
    batch_uid: str
    text: str
    confidence: float
    rank: int
    source: str
    bbox: XYXY | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.run_uid, "run_uid")
        _required_uid(self.region_uid, "region_uid")
        _required_uid(self.line_uid, "line_uid")
        _required_uid(self.atom_uid, "atom_uid")
        _required_uid(self.batch_uid, "batch_uid")
        _text(self.text, "text")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        _required_text(self.source, "source")
        object.__setattr__(self, "bbox", _optional_bbox(self.bbox))
        _set_fingerprint(self)

    @property
    def candidate_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrBatch:
    """One complete append-only observation unit for an OCR scope."""

    project_uid: str
    uid: str
    run_uid: str
    scope_uid: str
    input_fingerprint: str
    layout_fingerprint: str
    region_uids: tuple[str, ...] = ()
    line_uids: tuple[str, ...] = ()
    atom_uids: tuple[str, ...] = ()
    candidate_uids: tuple[str, ...] = ()
    status: str = "complete"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.run_uid, "run_uid")
        _required_uid(self.scope_uid, "scope_uid")
        _required_text(self.input_fingerprint, "input_fingerprint")
        _required_text(self.layout_fingerprint, "layout_fingerprint")
        object.__setattr__(self, "region_uids", _uid_tuple(self.region_uids, "region_uids"))
        object.__setattr__(self, "line_uids", _uid_tuple(self.line_uids, "line_uids"))
        object.__setattr__(self, "atom_uids", _uid_tuple(self.atom_uids, "atom_uids"))
        object.__setattr__(self, "candidate_uids", _uid_tuple(self.candidate_uids, "candidate_uids"))
        _required_text(self.status, "status")
        _set_fingerprint(self)

    @property
    def batch_uid(self) -> str:
        return self.uid


@dataclass(frozen=True, slots=True)
class OcrActivePointer:
    """The active OCR batch for one stable project scope."""

    project_uid: str
    uid: str
    scope_uid: str
    batch_uid: str
    run_uid: str
    batch_fingerprint: str
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.scope_uid, "scope_uid")
        _required_uid(self.batch_uid, "batch_uid")
        _required_uid(self.run_uid, "run_uid")
        _required_text(self.batch_fingerprint, "batch_fingerprint")
        object.__setattr__(self, "revision", _revision(self.revision))
        _set_fingerprint(self)

    @property
    def active_batch_uid(self) -> str:
        return self.batch_uid

    @property
    def active_run_uid(self) -> str:
        return self.run_uid


__all__ = [
    "OcrActivePointer",
    "OcrAtom",
    "OcrBatch",
    "OcrCandidate",
    "OcrLine",
    "OcrRegion",
    "OcrRun",
    "XYXY",
]
