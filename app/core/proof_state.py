"""Typed proof state contracts shared by proof UI, services, and stats."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app.models import BBox, Block, Line, Page


TOPIC_LINE_PROOF_CHANGED = "line.proof_changed"
TOPIC_PROBE_OBSERVED = "probe.observed"


@dataclass(frozen=True)
class ProofSelection:
    """Current proof selection identity without Qt widget references."""

    page_number: int
    page_id: Optional[int] = None
    page_uid: Optional[str] = None
    block_order: Optional[int] = None
    line_id: Optional[int] = None
    line_uid: Optional[str] = None
    line_index: Optional[int] = None
    char_index: Optional[int] = None
    source: str = ""

    @classmethod
    def for_line(
        cls,
        *,
        page: Page,
        block: Block,
        line: Line,
        line_index: int,
        source: str = "",
    ) -> "ProofSelection":
        return cls(
            page_number=int(page.page_number),
            page_id=page.id,
            page_uid=page.uid,
            block_order=int(block.order),
            line_id=line.id,
            line_uid=line.uid,
            line_index=int(line_index),
            source=source,
        )

    @classmethod
    def for_char_entry(cls, entry: Any, *, source: str = "") -> "ProofSelection":
        line = getattr(entry, "line", None)
        return cls(
            page_number=int(getattr(entry, "page_number", 1)),
            page_id=getattr(entry, "page_id", None),
            page_uid=getattr(entry, "page_uid", None),
            block_order=int(getattr(entry, "block_order", 0)),
            line_id=getattr(line, "id", None),
            line_uid=getattr(line, "uid", None),
            line_index=int(getattr(entry, "line_idx", 0)),
            char_index=int(getattr(entry, "char_idx", 0)),
            source=source,
        )


@dataclass(frozen=True)
class ProofLineViewModel:
    """Line-level view state consumed by proof panels."""

    selection: ProofSelection
    text: str
    ocr_text: str
    proof_status: str
    confidence: float
    bbox: BBox
    review_flags: tuple[str, ...] = ()
    char_count: int = 0

    @classmethod
    def from_model(
        cls,
        *,
        page: Page,
        block: Block,
        line: Line,
        line_index: int,
        display_text: Optional[str] = None,
        source: str = "",
    ) -> "ProofLineViewModel":
        text = display_text if display_text is not None else line.display_text
        return cls(
            selection=ProofSelection.for_line(
                page=page,
                block=block,
                line=line,
                line_index=line_index,
                source=source,
            ),
            text=text,
            ocr_text=line.ocr_text or "",
            proof_status=line.proof_status.value,
            confidence=float(line.confidence),
            bbox=line.bbox,
            review_flags=tuple(line.review_flags or ()),
            char_count=len(text),
        )


@dataclass(frozen=True)
class ProofUpdateRequest:
    """One proof edit/status update event."""

    page_id: Optional[int]
    line_id: Optional[int]
    status: str
    page_uid: Optional[str] = None
    line_uid: Optional[str] = None
    origin: Optional[int] = None
    selection: Optional[ProofSelection] = None
    source: str = ""

    @classmethod
    def from_legacy(cls, payload: Optional[Any] = None, **kwargs: Any) -> "ProofUpdateRequest":
        if isinstance(payload, cls):
            return payload
        data: dict[str, Any] = {}
        if isinstance(payload, dict):
            data.update(payload)
        data.update(kwargs)
        return cls(
            page_id=data.get("page_id"),
            line_id=data.get("line_id"),
            status=str(data.get("status", "")),
            page_uid=data.get("page_uid"),
            line_uid=data.get("line_uid"),
            origin=data.get("origin"),
            selection=data.get("selection"),
            source=str(data.get("source", "")),
        )

    def to_legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_id": self.page_id,
            "line_id": self.line_id,
            "status": self.status,
            "origin": self.origin,
            "source": self.source,
        }
        if self.selection is not None:
            payload["selection"] = self.selection
        return payload


def proof_request_matches_page(request: ProofUpdateRequest, page: Page) -> bool:
    """Return whether a proof event targets the given page."""
    page_uid = request.page_uid or None
    if page_uid is not None:
        return page.uid == page_uid
    page_id = request.page_id
    return page_id is None or page.id == page_id


def proof_request_matches_line(request: ProofUpdateRequest, line: Line) -> bool:
    """Return whether a proof event targets the given line, preferring stable UID."""
    line_uid = request.line_uid or None
    if line_uid is not None:
        return line.uid == line_uid
    line_id = request.line_id
    return line_id is not None and line.id == line_id


@dataclass(frozen=True)
class CandidateOption:
    text: str
    rank: int
    source: str = ""


@dataclass(frozen=True)
class CandidateSet:
    selection: Optional[ProofSelection]
    options: tuple[CandidateOption, ...] = field(default_factory=tuple)

    @classmethod
    def from_values(
        cls,
        *,
        selection: Optional[ProofSelection],
        values: Iterable[str],
        source: str = "",
    ) -> "CandidateSet":
        return cls(
            selection=selection,
            options=tuple(
                CandidateOption(text=value, rank=index, source=source)
                for index, value in enumerate(values)
            ),
        )

    @property
    def texts(self) -> list[str]:
        return [option.text for option in self.options]


@dataclass(frozen=True)
class ProbeObservation:
    """Quality-probe observation event."""

    page_number: int
    block_index: int
    line_index: int
    char_index: int
    true_char: str
    fake_char: str
    observation: str = "corrected"

    @classmethod
    def from_legacy(cls, payload: Optional[Any] = None, **kwargs: Any) -> "ProbeObservation":
        if isinstance(payload, cls):
            return payload
        data: dict[str, Any] = {}
        if isinstance(payload, dict):
            data.update(payload)
        data.update(kwargs)
        return cls(
            page_number=int(data.get("page_number", 0)),
            block_index=int(data.get("block_index", 0)),
            line_index=int(data.get("line_index", 0)),
            char_index=int(data.get("char_index", 0)),
            true_char=str(data.get("true_char", "")),
            fake_char=str(data.get("fake_char", "")),
            observation=str(data.get("observation", "corrected")),
        )

    def to_legacy_payload(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "block_index": self.block_index,
            "line_index": self.line_index,
            "char_index": self.char_index,
            "true_char": self.true_char,
            "fake_char": self.fake_char,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class QualityStatsState:
    enabled: bool
    total_probes: int = 0
    corrected: int = 0
    pending: int = 0
    detect_ratio: float = 0.0
    grade: str = "INSUFFICIENT"
    grade_label: str = ""
    sampled_from_chars: int = 0
    target_probes: int = 0
    density_text: str = ""
