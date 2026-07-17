"""Immutable audit records for one OCR routing run.

These records describe derived observations and routing decisions. They are
not layout, OCR, or proof truth and are intentionally not attached to Page.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .entity_id import ensure_entity_uid


XYXY = tuple[int, int, int, int]


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _optional_bbox(value: object) -> XYXY | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("audit message bbox must contain four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("audit message bbox coordinates must be integers")
    return tuple(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class BlockVlObservationSummary:
    """Persistable summary of one block-scoped VL observation."""

    block_uid: str
    status: str
    observation_uid: str = ""
    text_length: int = 0
    record_count: int = 0
    raw_refs: tuple[str, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_uid", _required_text(self.block_uid, "block_uid"))
        object.__setattr__(self, "status", _required_text(self.status, "status"))
        object.__setattr__(self, "observation_uid", _text(self.observation_uid, "observation_uid"))
        object.__setattr__(
            self,
            "text_length",
            _non_negative_int(self.text_length, "text_length"),
        )
        object.__setattr__(
            self,
            "record_count",
            _non_negative_int(self.record_count, "record_count"),
        )
        if not isinstance(self.raw_refs, (list, tuple)):
            raise TypeError("raw_refs must be a sequence of strings")
        if any(not isinstance(ref, str) for ref in self.raw_refs):
            raise TypeError("raw_refs must contain strings")
        refs = tuple(ref.strip() for ref in self.raw_refs)
        if any(not ref for ref in refs):
            raise ValueError("raw_refs cannot contain empty values")
        object.__setattr__(self, "raw_refs", refs)
        object.__setattr__(self, "message", _text(self.message, "message"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_uid": self.block_uid,
            "status": self.status,
            "observation_uid": self.observation_uid,
            "text_length": self.text_length,
            "record_count": self.record_count,
            "raw_refs": list(self.raw_refs),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BlockVlObservationSummary":
        if not isinstance(value, dict):
            raise TypeError("block VL observation summary must be a dict")
        refs = value.get("raw_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise TypeError("block VL observation raw_refs must be a list of strings")
        return cls(
            block_uid=value.get("block_uid", ""),
            status=value.get("status", ""),
            observation_uid=value.get("observation_uid", ""),
            text_length=value.get("text_length", 0),
            record_count=value.get("record_count", 0),
            raw_refs=tuple(refs),
            message=value.get("message", ""),
        )


@dataclass(frozen=True)
class BlockAlignmentAuditSummary:
    """Result of aligning external observations inside one layout block."""

    block_uid: str
    status: str
    matched_span_count: int = 0
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_uid", _required_text(self.block_uid, "block_uid"))
        object.__setattr__(self, "status", _required_text(self.status, "status"))
        object.__setattr__(
            self,
            "matched_span_count",
            _non_negative_int(self.matched_span_count, "matched_span_count"),
        )
        object.__setattr__(self, "message", _text(self.message, "message"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_uid": self.block_uid,
            "status": self.status,
            "matched_span_count": self.matched_span_count,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BlockAlignmentAuditSummary":
        if not isinstance(value, dict):
            raise TypeError("block alignment status must be a dict")
        return cls(
            block_uid=value.get("block_uid", ""),
            status=value.get("status", ""),
            matched_span_count=value.get("matched_span_count", 0),
            message=value.get("message", ""),
        )


@dataclass(frozen=True)
class RoutingAuditMessage:
    """Serializable validation issue or non-blocking route diagnostic."""

    code: str
    message: str
    block_uid: str = ""
    line_index: int | None = None
    bbox: XYXY | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_text(self.code, "code"))
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(self, "block_uid", _text(self.block_uid, "block_uid"))
        if self.line_index is not None and (
            isinstance(self.line_index, bool) or not isinstance(self.line_index, int)
        ):
            raise TypeError("line_index must be an integer or null")
        object.__setattr__(self, "bbox", _optional_bbox(self.bbox))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "block_uid": self.block_uid,
            "line_index": self.line_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RoutingAuditMessage":
        if not isinstance(value, dict):
            raise TypeError("routing audit message must be a dict")
        return cls(
            code=value.get("code", ""),
            message=value.get("message", ""),
            block_uid=value.get("block_uid", ""),
            line_index=value.get("line_index"),
            bbox=value.get("bbox"),
        )


@dataclass(frozen=True)
class RouteAuditSummary:
    """Compact derived-plan summary retained for diagnostics and replay audit."""

    block_count: int
    line_count: int
    segment_count: int
    is_dispatchable: bool
    validation_issues: tuple[RoutingAuditMessage, ...] = ()
    diagnostics: tuple[RoutingAuditMessage, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("block_count", "line_count", "segment_count"):
            object.__setattr__(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        if not isinstance(self.is_dispatchable, bool):
            raise TypeError("is_dispatchable must be bool")
        object.__setattr__(self, "validation_issues", tuple(self.validation_issues))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(not isinstance(item, RoutingAuditMessage) for item in self.validation_issues):
            raise TypeError("validation_issues require RoutingAuditMessage values")
        if any(not isinstance(item, RoutingAuditMessage) for item in self.diagnostics):
            raise TypeError("diagnostics require RoutingAuditMessage values")
        if self.is_dispatchable == bool(self.validation_issues):
            raise ValueError("is_dispatchable must be false exactly when validation issues exist")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_count": self.block_count,
            "line_count": self.line_count,
            "segment_count": self.segment_count,
            "is_dispatchable": self.is_dispatchable,
            "validation_issues": [item.to_dict() for item in self.validation_issues],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: object) -> "RouteAuditSummary":
        if not isinstance(value, dict):
            raise TypeError("route audit summary must be a dict")
        issues = value.get("validation_issues", [])
        diagnostics = value.get("diagnostics", [])
        if not isinstance(issues, list) or not isinstance(diagnostics, list):
            raise TypeError("route audit messages must be lists")
        return cls(
            block_count=value.get("block_count", 0),
            line_count=value.get("line_count", 0),
            segment_count=value.get("segment_count", 0),
            is_dispatchable=value.get("is_dispatchable", False),
            validation_issues=tuple(RoutingAuditMessage.from_dict(item) for item in issues),
            diagnostics=tuple(RoutingAuditMessage.from_dict(item) for item in diagnostics),
        )


@dataclass(frozen=True)
class OcrRoutingRunAuditDraft:
    """Page-scoped audit before the application binds it to a stored project."""

    page_uid: str
    image_hash: str
    layout_fingerprint: str
    pp_run_id: str
    route_summary: RouteAuditSummary
    block_vl_observations: tuple[BlockVlObservationSummary, ...] = ()
    alignment_statuses: tuple[BlockAlignmentAuditSummary, ...] = ()
    created_at: float = field(default_factory=time.time)
    run_uid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.run_uid, str):
            raise TypeError("run_uid must be str")
        object.__setattr__(self, "run_uid", ensure_entity_uid(self.run_uid, "ocrrun"))
        for field_name in ("page_uid", "image_hash", "layout_fingerprint", "pp_run_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        _validate_audit_payload(self)

    def bind_to_project(self, project_id: int) -> "OcrRoutingRunAudit":
        return OcrRoutingRunAudit(
            project_id=project_id,
            page_uid=self.page_uid,
            image_hash=self.image_hash,
            layout_fingerprint=self.layout_fingerprint,
            pp_run_id=self.pp_run_id,
            route_summary=self.route_summary,
            block_vl_observations=self.block_vl_observations,
            alignment_statuses=self.alignment_statuses,
            created_at=self.created_at,
            run_uid=self.run_uid,
        )


@dataclass(frozen=True)
class OcrRoutingRunAudit:
    """Append-only audit package for one page-scoped routing compilation run."""

    project_id: int
    page_uid: str
    image_hash: str
    layout_fingerprint: str
    pp_run_id: str
    route_summary: RouteAuditSummary
    block_vl_observations: tuple[BlockVlObservationSummary, ...] = ()
    alignment_statuses: tuple[BlockAlignmentAuditSummary, ...] = ()
    created_at: float = field(default_factory=time.time)
    run_uid: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.project_id, bool) or not isinstance(self.project_id, int) or self.project_id <= 0:
            raise ValueError("project_id must be a positive integer")
        if not isinstance(self.run_uid, str):
            raise TypeError("run_uid must be str")
        object.__setattr__(self, "run_uid", ensure_entity_uid(self.run_uid, "ocrrun"))
        for field_name in ("page_uid", "image_hash", "layout_fingerprint", "pp_run_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        _validate_audit_payload(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_uid": self.run_uid,
            "project_id": self.project_id,
            "page_uid": self.page_uid,
            "image_hash": self.image_hash,
            "layout_fingerprint": self.layout_fingerprint,
            "pp_run_id": self.pp_run_id,
            "block_vl_observations": [item.to_dict() for item in self.block_vl_observations],
            "alignment_statuses": [item.to_dict() for item in self.alignment_statuses],
            "route_summary": self.route_summary.to_dict(),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> "OcrRoutingRunAudit":
        if not isinstance(value, dict):
            raise TypeError("OCR routing run audit must be a dict")
        observations = value.get("block_vl_observations", [])
        alignments = value.get("alignment_statuses", [])
        if not isinstance(observations, list) or not isinstance(alignments, list):
            raise TypeError("OCR routing audit observation fields must be lists")
        return cls(
            run_uid=value.get("run_uid", ""),
            project_id=value.get("project_id", 0),
            page_uid=value.get("page_uid", ""),
            image_hash=value.get("image_hash", ""),
            layout_fingerprint=value.get("layout_fingerprint", ""),
            pp_run_id=value.get("pp_run_id", ""),
            block_vl_observations=tuple(
                BlockVlObservationSummary.from_dict(item) for item in observations
            ),
            alignment_statuses=tuple(
                BlockAlignmentAuditSummary.from_dict(item) for item in alignments
            ),
            route_summary=RouteAuditSummary.from_dict(value.get("route_summary")),
            created_at=value.get("created_at", 0.0),
        )


def _validate_audit_payload(audit: OcrRoutingRunAudit | OcrRoutingRunAuditDraft) -> None:
    if (
        not isinstance(audit.created_at, (int, float))
        or isinstance(audit.created_at, bool)
        or not math.isfinite(audit.created_at)
        or audit.created_at < 0
    ):
        raise ValueError("created_at must be a finite non-negative number")
    if not isinstance(audit.route_summary, RouteAuditSummary):
        raise TypeError("route_summary requires RouteAuditSummary")
    object.__setattr__(audit, "block_vl_observations", tuple(audit.block_vl_observations))
    object.__setattr__(audit, "alignment_statuses", tuple(audit.alignment_statuses))
    if any(not isinstance(item, BlockVlObservationSummary) for item in audit.block_vl_observations):
        raise TypeError("block_vl_observations require typed summaries")
    if any(not isinstance(item, BlockAlignmentAuditSummary) for item in audit.alignment_statuses):
        raise TypeError("alignment_statuses require typed values")
    observation_uids = [item.block_uid for item in audit.block_vl_observations]
    alignment_uids = [item.block_uid for item in audit.alignment_statuses]
    if len(set(observation_uids)) != len(observation_uids):
        raise ValueError("block_vl_observations contain duplicate block_uid values")
    if len(set(alignment_uids)) != len(alignment_uids):
        raise ValueError("alignment_statuses contain duplicate block_uid values")


__all__ = [
    "BlockAlignmentAuditSummary",
    "BlockVlObservationSummary",
    "OcrRoutingRunAudit",
    "OcrRoutingRunAuditDraft",
    "RouteAuditSummary",
    "RoutingAuditMessage",
]
