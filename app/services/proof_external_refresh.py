"""Shared external-refresh state for proof views."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.core.proof_state import ProofUpdateRequest
from app.models import Line


@dataclass(frozen=True)
class ProofExternalLineRef:
    """Stable line identity carried by proof bus refresh events."""

    line_uid: str = ""
    line_id: int | None = None

    @property
    def is_valid(self) -> bool:
        return bool(self.line_uid) or self.line_id is not None

    @classmethod
    def from_request(cls, request: ProofUpdateRequest) -> "ProofExternalLineRef":
        return cls(
            line_uid=request.line_uid or "",
            line_id=request.line_id,
        )

    @classmethod
    def from_key(cls, key: int | str) -> "ProofExternalLineRef":
        if isinstance(key, int):
            return cls(line_id=key)
        value = str(key or "")
        return cls(line_uid=value)

    def matches_line(self, line: Line) -> bool:
        if self.line_uid:
            return line.uid == self.line_uid
        return self.line_id is not None and line.id == self.line_id


@dataclass(frozen=True)
class ProofExternalRefreshBatch:
    line_refs: tuple[ProofExternalLineRef, ...] = tuple()
    page_keys: tuple[tuple[object, ...], ...] = tuple()

    @property
    def has_work(self) -> bool:
        return bool(self.line_refs or self.page_keys)


@dataclass(frozen=True)
class ProofExternalRefreshPlan:
    """Shared UI-free external refresh plan for proof views.

    Horizontal proof consumes projection indexes. Vertical proof consumes page
    keys and a current-page reload flag. Keeping one plan type prevents H/V
    refresh semantics from drifting while still allowing each view to execute
    its own UI update.
    """

    line_refs: tuple[ProofExternalLineRef, ...] = tuple()
    affected_page_keys: tuple[tuple[object, ...], ...] = tuple()
    touched_projection_indexes: tuple[int, ...] = tuple()
    reload_current_page: bool = False

    @property
    def has_work(self) -> bool:
        return bool(
            self.affected_page_keys
            or self.touched_projection_indexes
            or self.reload_current_page
        )


@dataclass
class ProofExternalRefreshQueue:
    _line_refs: list[ProofExternalLineRef] = field(default_factory=list)
    _page_keys: set[tuple[object, ...]] = field(default_factory=set)

    @property
    def has_pending(self) -> bool:
        return bool(self._line_refs or self._page_keys)

    def clear(self) -> None:
        self._line_refs.clear()
        self._page_keys.clear()

    def queue_request(
        self,
        request: ProofUpdateRequest,
        *,
        page_keys: Iterable[tuple[object, ...]] = (),
    ) -> bool:
        return self.queue_line_ref(
            ProofExternalLineRef.from_request(request),
            page_keys=page_keys,
        )

    def queue_line_key(
        self,
        line_key: int | str,
        *,
        page_keys: Iterable[tuple[object, ...]] = (),
    ) -> bool:
        return self.queue_line_ref(
            ProofExternalLineRef.from_key(line_key),
            page_keys=page_keys,
        )

    def queue_line_ref(
        self,
        line_ref: ProofExternalLineRef,
        *,
        page_keys: Iterable[tuple[object, ...]] = (),
    ) -> bool:
        if not line_ref.is_valid:
            return False
        self._line_refs.append(line_ref)
        self._page_keys.update(tuple(page_key) for page_key in page_keys)
        return True

    def consume(self) -> ProofExternalRefreshBatch:
        if not self.has_pending:
            return ProofExternalRefreshBatch()
        batch = ProofExternalRefreshBatch(
            line_refs=tuple(self._line_refs),
            page_keys=tuple(self._page_keys),
        )
        self.clear()
        return batch
