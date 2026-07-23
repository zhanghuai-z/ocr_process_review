"""Immutable request and result contracts for the CharOCR native boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .charocr_routing import PageRoutingPlan
from .enums import BlockSource, OcrPolicy
from .layout_snapshot import LayoutSnapshot
from .project_session import PageRecord


XYXY = tuple[int, int, int, int]


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CharOcrInputRow:
    """One layout-owned native input row; never a mutable layout object."""

    block_uid: str
    label: str
    bbox: XYXY
    content: str
    ocr_policy: OcrPolicy
    authorship: BlockSource
    order: int

    def __post_init__(self) -> None:
        if not self.block_uid or not self.label:
            raise ValueError("CharOCR input row requires block UID and label")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("CharOCR input row requires non-empty geometry")
        if self.order < 0:
            raise ValueError("CharOCR input row order must be non-negative")

@dataclass(frozen=True, slots=True)
class CharOcrPageRequest:
    """One complete immutable native execution input."""

    project_uid: str
    page: PageRecord
    layout: LayoutSnapshot
    routing_plan: PageRoutingPlan
    rows: tuple[CharOcrInputRow, ...]
    input_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.page.project_uid != self.project_uid:
            raise ValueError("CharOCR page belongs to another project")
        if self.layout.page_uid != self.page.uid:
            raise ValueError("CharOCR layout belongs to another page")
        if self.routing_plan.page_uid != self.page.uid:
            raise ValueError("CharOCR routing plan belongs to another page")
        rows = tuple(self.rows)
        if tuple(row.order for row in rows) != tuple(range(len(rows))):
            raise ValueError("CharOCR rows must use stable layout order")
        if len({row.block_uid for row in rows}) != len(rows):
            raise ValueError("CharOCR rows contain duplicate block UIDs")
        layout_uids = {block.uid for block in self.layout.blocks}
        if any(row.block_uid not in layout_uids for row in rows):
            raise ValueError("CharOCR row is outside the adopted layout")
        payload = {
            "project_uid": self.project_uid,
            "page_uid": self.page.uid,
            "page_fingerprint": self.page.fingerprint,
            "layout_revision": self.layout.revision,
            "routing_run_uid": self.routing_plan.routing_run_uid,
            "rows": [
                {
                    "uid": row.block_uid,
                    "label": row.label,
                    "bbox": row.bbox,
                    "content": row.content,
                    "policy": row.ocr_policy.value,
                    "authorship": row.authorship.value,
                    "order": row.order,
                }
                for row in rows
            ],
        }
        computed = _fingerprint(payload)
        if self.input_fingerprint and self.input_fingerprint != computed:
            raise ValueError("CharOCR request input fingerprint is invalid")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "input_fingerprint", computed)


@dataclass(frozen=True, slots=True)
class CharOcrCandidateObservation:
    text: str
    confidence: float
    source: str = ""
    bbox: XYXY | None = None


@dataclass(frozen=True, slots=True)
class CharOcrAtomObservation:
    text: str
    bbox: XYXY
    confidence: float
    source: str
    granularity: str = "char"
    token_text: str = ""
    candidates: tuple[CharOcrCandidateObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class CharOcrLineObservation:
    text: str
    bbox: XYXY
    confidence: float
    source: str
    atoms: tuple[CharOcrAtomObservation, ...] = ()
    review_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CharOcrRegionObservation:
    block_uid: str
    label: str
    bbox: XYXY
    source: str
    lines: tuple[CharOcrLineObservation, ...] = ()
    audit_json: str = "{}"


@dataclass(frozen=True, slots=True)
class CharOcrPageResult:
    """Native output before it is appended to the OCR repository."""

    page_uid: str
    input_fingerprint: str
    regions: tuple[CharOcrRegionObservation, ...]
    metrics: tuple[tuple[str, str], ...] = ()


__all__ = [
    "CharOcrAtomObservation",
    "CharOcrCandidateObservation",
    "CharOcrInputRow",
    "CharOcrLineObservation",
    "CharOcrPageRequest",
    "CharOcrPageResult",
    "CharOcrRegionObservation",
]
