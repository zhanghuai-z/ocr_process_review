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
    block_order: Optional[int] = None
    line_id: Optional[int] = None
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
            block_order=int(block.order),
            line_id=line.id,
            line_index=int(line_index),
            source=source,
        )

    @classmethod
    def for_char_entry(cls, entry: Any, *, source: str = "") -> "ProofSelection":
        line = getattr(entry, "line", None)
        return cls(
            page_number=int(getattr(entry, "page_number", 1)),
            block_order=int(getattr(entry, "block_order", 0)),
            line_id=getattr(line, "id", None),
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
        text = display_text if display_text is not None else (line.final_text or line.text or "")
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

