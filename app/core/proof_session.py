"""Pure values shared by the proof session service and its callers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.proof_records import ProofState


class ProofOperation(str, Enum):
    CREATE = "create"
    TEXT = "text"
    BATCH_TEXT = "batch_text"
    STATUS = "status"
    SPANS = "spans"
    UNDO = "undo"
    REDO = "redo"
    MARK_REBIND = "mark_rebind"
    REBIND = "rebind"
    REFRESH = "refresh"


class ProofOutcome(str, Enum):
    APPLIED = "applied"
    NOOP = "noop"
    UNCHANGED = "unchanged"
    DIRTY_REBASE = "dirty_rebase"
    REBOUND = "rebound"
    CONFLICT = "conflict"
    REBIND_REQUIRED = "rebind_required"


@dataclass(frozen=True, slots=True)
class ProofRevision:
    uid: str
    revision: int
    fingerprint: str

    @classmethod
    def from_record(cls, record: object) -> "ProofRevision":
        return cls(
            uid=str(getattr(record, "uid")),
            revision=int(getattr(record, "revision")),
            fingerprint=str(getattr(record, "fingerprint")),
        )


@dataclass(frozen=True, slots=True)
class ProofSessionResult:
    """Typed result for every proof mutation.

    Callers must inspect ``outcome`` and ``changed``.  This object intentionally
    has no boolean conversion and is not a compatibility replacement for the
    old ``changed: bool`` write APIs.
    """

    operation: ProofOperation
    outcome: ProofOutcome
    state: ProofState
    changed: bool
    changed_text_unit_uids: tuple[str, ...] = ()
    message: str = ""

    @property
    def revision(self) -> int:
        return self.state.revision

    @property
    def fingerprint(self) -> str:
        return self.state.fingerprint


@dataclass(frozen=True, slots=True)
class ProofEditorSnapshot:
    """CAS token and baseline text for one editor binding."""

    proof_uid: str
    text_unit_uid: str
    text: str
    status: str
    state_revision: int
    state_fingerprint: str
    text_unit_revision: int
    text_unit_fingerprint: str


@dataclass(frozen=True, slots=True)
class ProofRefreshResult:
    """Result of comparing an editor binding with a newer proof snapshot."""

    outcome: ProofOutcome
    snapshot: ProofEditorSnapshot | None
    current_state: ProofState
    external_changed: bool
    text_changed: bool
    message: str = ""

    @property
    def conflict(self) -> bool:
        return self.outcome == ProofOutcome.CONFLICT


@dataclass(frozen=True, slots=True)
class ProofSpanReplacement:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("span must satisfy 0 <= start <= end")
        if not isinstance(self.text, str):
            raise TypeError("replacement text must be str")


__all__ = [
    "ProofEditorSnapshot",
    "ProofOperation",
    "ProofOutcome",
    "ProofRefreshResult",
    "ProofRevision",
    "ProofSessionResult",
    "ProofSpanReplacement",
]
