"""Project-scoped proof session write and read boundary.

``ProofSessionService`` is the only proof business service.  Human text and
status are changed by replacing immutable ``ProofState`` values in the
project's proof repository.  OCR records, layout snapshots, and page records
are read-only inputs to this service.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from app.core.char_index import CharIndex, CharIndexEntry
from app.core.proof_session import (
    ProofEditorSnapshot,
    ProofOperation,
    ProofOutcome,
    ProofRefreshResult,
    ProofSessionResult,
    ProofSpanReplacement,
)
from app.models.ocr_records import OcrAtom, OcrBatch, OcrLine
from app.models.layout_snapshot import LayoutSnapshot
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.models.project_session import (
    InvalidRecordError,
    PageRecord,
    ProjectSession,
    RecordNotFoundError,
    RevisionConflictError,
)
from app.services.char_index_service import CharIndexService


class ProofSessionError(RuntimeError):
    """Base class for proof-session contract failures."""


class ProofTargetError(ProofSessionError, KeyError):
    """A proof or text-unit UID is not present in the requested state."""


class ProofConflictError(ProofSessionError):
    """An editor binding cannot be rebased without losing user input."""


class ProofRefreshKind(str, Enum):
    UNCHANGED = "unchanged"
    REBIND_REQUIRED = "rebind_required"


HISTORY_LIMIT = 5


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    proof_uid: str
    before: ProofState
    after: ProofState
    operation: ProofOperation
    expected_revision: int


@dataclass(frozen=True, slots=True)
class _ActiveObservation:
    page: PageRecord
    layout: LayoutSnapshot
    batch: OcrBatch
    lines: tuple[OcrLine, ...]
    atoms: tuple[OcrAtom, ...]


class ProofSessionService:
    """Own all proof mutations, refresh decisions, rebinds, and indexing."""

    def __init__(self, session: ProjectSession, *, history_limit: int = HISTORY_LIMIT) -> None:
        if history_limit != HISTORY_LIMIT:
            raise ValueError("proof session history is fixed at five operations")
        self._session = session
        self._history_limit = history_limit
        self._undo: dict[str, list[_HistoryEntry]] = {}
        self._redo: dict[str, list[_HistoryEntry]] = {}
        self._indexer = CharIndexService()

    @property
    def project_uid(self) -> str:
        return self._session.project_uid

    @property
    def history_limit(self) -> int:
        return self._history_limit

    def create_state(self, state: ProofState) -> ProofSessionResult:
        stored = self._session.proof_repository.create_state(state)
        self._clear_history(stored.uid)
        return ProofSessionResult(
            operation=ProofOperation.CREATE,
            outcome=ProofOutcome.APPLIED,
            state=stored,
            changed=True,
        )

    def get_state(
        self,
        proof_uid: str,
        *,
        expected_revision: int | None = None,
        expected_fingerprint: str | None = None,
    ) -> ProofState:
        try:
            return self._session.proof_repository.get_state(
                proof_uid,
                revision=expected_revision,
                fingerprint=expected_fingerprint,
            )
        except (KeyError, RecordNotFoundError) as exc:
            raise ProofTargetError(proof_uid) from exc

    def editor_snapshot(self, proof_uid: str, text_unit_uid: str) -> ProofEditorSnapshot:
        state = self.get_state(proof_uid)
        unit = self._text_unit(state, text_unit_uid)
        return self._editor_snapshot(state, unit)

    def replace_text(
        self,
        proof_uid: str,
        text_unit_uid: str,
        text: str,
        *,
        expected_revision: int,
        expected_fingerprint: str,
        expected_unit_revision: int | None = None,
        expected_unit_fingerprint: str | None = None,
        status: str | None = "modified",
    ) -> ProofSessionResult:
        """Replace one human text unit using a state and unit CAS token."""

        if not isinstance(text, str):
            raise TypeError("text must be str")
        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        unit = self._text_unit(current, text_unit_uid)
        self._cas_unit(
            unit,
            expected_revision=expected_unit_revision,
            expected_fingerprint=expected_unit_fingerprint,
        )
        next_status = unit.status if status is None else _status_value(status)
        if unit.text == text and unit.status == next_status:
            return self._result(
                ProofOperation.TEXT,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        next_unit = replace(
            unit,
            text=text,
            status=next_status,
            revision=unit.revision + 1,
        )
        units = tuple(next_unit if item.uid == unit.uid else item for item in current.text_units)
        stored = self._replace_state(current, text_units=units)
        self._record_history(current, stored, ProofOperation.TEXT)
        return self._result(
            ProofOperation.TEXT,
            ProofOutcome.APPLIED,
            stored,
            changed=True,
            changed_text_unit_uids=(unit.uid,),
        )

    def replace_text_units(
        self,
        proof_uid: str,
        replacements: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        expected_revision: int,
        expected_fingerprint: str,
        expected_unit_revisions: Mapping[str, int] | None = None,
        expected_unit_fingerprints: Mapping[str, str] | None = None,
        status: str | None = "modified",
    ) -> ProofSessionResult:
        """Atomically replace several text units in one state revision."""

        values = dict(replacements)
        if any(not isinstance(value, str) for value in values.values()):
            raise TypeError("replacement texts must be str")
        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        units_by_uid = {unit.uid: unit for unit in current.text_units}
        unknown = set(values) - set(units_by_uid)
        if unknown:
            raise ProofTargetError(f"unknown text unit UID(s): {sorted(unknown)!r}")
        next_status = None if status is None else _status_value(status)
        changed: list[str] = []
        next_units: list[ProofTextUnit] = []
        for unit in current.text_units:
            if unit.uid not in values:
                next_units.append(unit)
                continue
            self._cas_unit(
                unit,
                expected_revision=(expected_unit_revisions or {}).get(unit.uid),
                expected_fingerprint=(expected_unit_fingerprints or {}).get(unit.uid),
            )
            unit_status = unit.status if next_status is None else next_status
            text = values[unit.uid]
            if text == unit.text and unit_status == unit.status:
                next_units.append(unit)
                continue
            changed.append(unit.uid)
            next_units.append(
                replace(
                    unit,
                    text=text,
                    status=unit_status,
                    revision=unit.revision + 1,
                )
            )
        if not changed:
            return self._result(
                ProofOperation.BATCH_TEXT,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        stored = self._replace_state(current, text_units=tuple(next_units))
        self._record_history(current, stored, ProofOperation.BATCH_TEXT)
        return self._result(
            ProofOperation.BATCH_TEXT,
            ProofOutcome.APPLIED,
            stored,
            changed=True,
            changed_text_unit_uids=tuple(changed),
        )

    def replace_spans(
        self,
        proof_uid: str,
        text_unit_uid: str,
        replacements: Iterable[ProofSpanReplacement],
        *,
        expected_revision: int,
        expected_fingerprint: str,
        expected_unit_revision: int | None = None,
        expected_unit_fingerprint: str | None = None,
        status: str | None = "modified",
    ) -> ProofSessionResult:
        """Apply validated, non-overlapping spans as one proof mutation."""

        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        unit = self._text_unit(current, text_unit_uid)
        self._cas_unit(
            unit,
            expected_revision=expected_unit_revision,
            expected_fingerprint=expected_unit_fingerprint,
        )
        spans = tuple(replacements)
        if not spans:
            return self._result(
                ProofOperation.SPANS,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        ordered = tuple(sorted(spans, key=lambda item: (item.start, item.end)))
        previous_end = 0
        for item in ordered:
            if item.end > len(unit.text) or item.start < previous_end:
                raise ValueError("replacement spans overlap or exceed text unit")
            previous_end = item.end
        updated = unit.text
        for item in reversed(ordered):
            updated = updated[:item.start] + item.text + updated[item.end:]
        result = self.replace_text(
            proof_uid,
            text_unit_uid,
            updated,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
            expected_unit_revision=unit.revision,
            expected_unit_fingerprint=unit.fingerprint,
            status=status,
        )
        return ProofSessionResult(
            operation=ProofOperation.SPANS,
            outcome=result.outcome,
            state=result.state,
            changed=result.changed,
            changed_text_unit_uids=result.changed_text_unit_uids,
            message=result.message,
        )

    def set_status(
        self,
        proof_uid: str,
        text_unit_uid: str,
        status: str,
        *,
        expected_revision: int,
        expected_fingerprint: str,
        expected_unit_revision: int | None = None,
        expected_unit_fingerprint: str | None = None,
    ) -> ProofSessionResult:
        """Set a human status without touching OCR observations."""

        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        unit = self._text_unit(current, text_unit_uid)
        self._cas_unit(
            unit,
            expected_revision=expected_unit_revision,
            expected_fingerprint=expected_unit_fingerprint,
        )
        normalized = _status_value(status)
        if unit.status == normalized:
            return self._result(
                ProofOperation.STATUS,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        next_unit = replace(unit, status=normalized, revision=unit.revision + 1)
        stored = self._replace_state(
            current,
            text_units=tuple(next_unit if item.uid == unit.uid else item for item in current.text_units),
        )
        self._record_history(current, stored, ProofOperation.STATUS)
        return self._result(
            ProofOperation.STATUS,
            ProofOutcome.APPLIED,
            stored,
            changed=True,
            changed_text_unit_uids=(unit.uid,),
        )

    def undo(
        self,
        proof_uid: str,
        *,
        expected_revision: int | None = None,
        expected_fingerprint: str | None = None,
    ) -> ProofSessionResult:
        return self._apply_history(
            proof_uid,
            undo=True,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def redo(
        self,
        proof_uid: str,
        *,
        expected_revision: int | None = None,
        expected_fingerprint: str | None = None,
    ) -> ProofSessionResult:
        return self._apply_history(
            proof_uid,
            undo=False,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def can_undo(self, proof_uid: str) -> bool:
        return bool(self._undo.get(proof_uid))

    def can_redo(self, proof_uid: str) -> bool:
        return bool(self._redo.get(proof_uid))

    def refresh_editor(
        self,
        snapshot: ProofEditorSnapshot,
        *,
        editor_text: str,
    ) -> ProofRefreshResult:
        """Compare an editor baseline with current proof state.

        A clean editor can rebind to a changed state. A dirty editor conflicts
        only when the same text unit changed externally; status-only changes
        produce a new CAS token and remain submit-able.
        """

        current = self.get_state(snapshot.proof_uid)
        unit = self._text_unit(current, snapshot.text_unit_uid)
        state_changed = (
            current.revision != snapshot.state_revision
            or current.fingerprint != snapshot.state_fingerprint
        )
        unit_text_changed = unit.text != snapshot.text
        if not state_changed:
            return ProofRefreshResult(
                outcome=ProofOutcome.UNCHANGED,
                snapshot=snapshot,
                current_state=current,
                external_changed=False,
                text_changed=False,
            )
        current_snapshot = self._editor_snapshot(current, unit)
        if unit_text_changed and editor_text not in {unit.text, snapshot.text}:
            return ProofRefreshResult(
                outcome=ProofOutcome.CONFLICT,
                snapshot=None,
                current_state=current,
                external_changed=True,
                text_changed=True,
                message="proof text changed while the editor was dirty",
            )
        if unit_text_changed:
            return ProofRefreshResult(
                outcome=ProofOutcome.REBOUND,
                snapshot=current_snapshot,
                current_state=current,
                external_changed=True,
                text_changed=True,
            )
        if editor_text != snapshot.text:
            return ProofRefreshResult(
                outcome=ProofOutcome.DIRTY_REBASE,
                snapshot=current_snapshot,
                current_state=current,
                external_changed=True,
                text_changed=False,
            )
        return ProofRefreshResult(
            outcome=ProofOutcome.REBOUND,
            snapshot=current_snapshot,
            current_state=current,
            external_changed=True,
            text_changed=False,
        )

    def rebind_editor(self, snapshot: ProofEditorSnapshot) -> ProofRefreshResult:
        """Explicitly discard the old editor baseline and bind to current state."""

        current = self.get_state(snapshot.proof_uid)
        unit = self._text_unit(current, snapshot.text_unit_uid)
        return ProofRefreshResult(
            outcome=ProofOutcome.REBOUND,
            snapshot=self._editor_snapshot(current, unit),
            current_state=current,
            external_changed=(
                current.revision != snapshot.state_revision
                or current.fingerprint != snapshot.state_fingerprint
            ),
            text_changed=unit.text != snapshot.text,
        )

    def refresh_active_observation(
        self,
        proof_uid: str,
        *,
        expected_revision: int,
        expected_fingerprint: str,
    ) -> ProofSessionResult:
        """Compare the active OCR batch with the proof anchor and mark rebind."""

        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        active = self._active_observation(current.anchor_snapshot.scope_uid)
        anchor = current.anchor_snapshot
        if current.rebind_required:
            return self._result(
                ProofOperation.REFRESH,
                ProofOutcome.REBIND_REQUIRED,
                current,
                changed=False,
                message="proof state already requires an explicit rebind",
            )
        if (
            anchor.source_fingerprint == active.batch.fingerprint
            and anchor.layout_fingerprint == active.batch.layout_fingerprint
        ):
            return self._result(
                ProofOperation.REFRESH,
                ProofOutcome.UNCHANGED,
                current,
                changed=False,
            )
        next_anchor = replace(
            anchor,
            source_fingerprint=active.batch.fingerprint,
            layout_fingerprint=active.batch.layout_fingerprint,
            anchor_revision=anchor.anchor_revision + 1,
            revision=anchor.revision + 1,
        )
        stored = self._mark_rebind(current, next_anchor)
        self._clear_history(proof_uid)
        return self._result(
            ProofOperation.MARK_REBIND,
            ProofOutcome.REBIND_REQUIRED,
            stored,
            changed=True,
            message="active OCR observation changed; explicit rebind is required",
        )

    def mark_rebind_required(
        self,
        proof_uid: str,
        anchor_snapshot: ProofAnchorSnapshot,
        *,
        expected_revision: int,
        expected_fingerprint: str,
    ) -> ProofSessionResult:
        """Record an explicit new anchor and invalidate old alignment."""

        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        if anchor_snapshot.scope_uid != current.anchor_snapshot.scope_uid:
            raise InvalidRecordError("rebind anchor scope does not match proof state")
        if current.rebind_required and current.anchor_snapshot.fingerprint == anchor_snapshot.fingerprint:
            return self._result(
                ProofOperation.MARK_REBIND,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        stored = self._mark_rebind(current, anchor_snapshot)
        self._clear_history(proof_uid)
        return self._result(
            ProofOperation.MARK_REBIND,
            ProofOutcome.REBIND_REQUIRED,
            stored,
            changed=True,
        )

    def rebind(
        self,
        proof_uid: str,
        anchor_snapshot: ProofAnchorSnapshot,
        alignment_segments: Sequence[ProofAlignmentSegment],
        alignment_slices: Sequence[ProofAlignmentSlice],
        *,
        expected_revision: int,
        expected_fingerprint: str,
    ) -> ProofSessionResult:
        """Install new alignment only after the state explicitly requires it."""

        current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        if not current.rebind_required:
            raise InvalidRecordError("proof state does not require rebind")
        if anchor_snapshot.scope_uid != current.anchor_snapshot.scope_uid:
            raise InvalidRecordError("rebind anchor scope does not match proof state")
        candidate = replace(
            current,
            anchor_snapshot=anchor_snapshot,
            alignment_segments=tuple(alignment_segments),
            alignment_slices=tuple(alignment_slices),
            rebind_required=False,
            revision=current.revision,
        )
        stored = self._session.proof_repository.replace_state(
            candidate,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
        )
        self._clear_history(proof_uid)
        return self._result(
            ProofOperation.REBIND,
            ProofOutcome.APPLIED,
            stored,
            changed=True,
        )

    def build_char_index(
        self,
        proof_uid: str,
        *,
        expected_revision: int | None = None,
        expected_fingerprint: str | None = None,
    ) -> CharIndex:
        """Build from the active OCR batch and the current proof state only."""

        state = self.get_state(
            proof_uid,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        active = self._active_observation(state.anchor_snapshot.scope_uid)
        return self._indexer.build(
            page=active.page,
            batch=active.batch,
            lines=active.lines,
            atoms=active.atoms,
            state=state,
        )

    def same_text(
        self,
        proof_uid: str,
        text: str,
        *,
        include_unavailable: bool = False,
        expected_revision: int | None = None,
        expected_fingerprint: str | None = None,
    ) -> tuple[CharIndexEntry, ...]:
        index = self.build_char_index(
            proof_uid,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        return index.same_text(text, include_unavailable=include_unavailable)

    def _active_observation(self, scope_uid: str) -> _ActiveObservation:
        pages = self._session.page_repository
        layouts = self._session.layout_repository
        page = pages.get(scope_uid)
        layout = layouts.get(scope_uid)
        if layout.page_uid != page.uid:
            raise InvalidRecordError("layout snapshot belongs to another page")
        ocr = self._session.ocr_observation_repository
        pointer = ocr.get_active_pointer(scope_uid)
        batch = ocr.get_batch(pointer.batch_uid, fingerprint=pointer.batch_fingerprint)
        lines = tuple(ocr.get_line(uid) for uid in batch.line_uids)
        atoms = tuple(ocr.get_atom(uid) for uid in batch.atom_uids)
        if batch.project_uid != self.project_uid:
            raise InvalidRecordError("active OCR batch belongs to another project")
        return _ActiveObservation(
            page=page,
            layout=layout,
            batch=batch,
            lines=lines,
            atoms=atoms,
        )

    def _cas_state(self, proof_uid: str, revision: int, fingerprint: str) -> ProofState:
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("expected_fingerprint must be non-empty")
        return self.get_state(
            proof_uid,
            expected_revision=revision,
            expected_fingerprint=fingerprint,
        )

    @staticmethod
    def _cas_unit(
        unit: ProofTextUnit,
        *,
        expected_revision: int | None,
        expected_fingerprint: str | None,
    ) -> None:
        if expected_revision is not None and unit.revision != expected_revision:
            raise RevisionConflictError(
                f"text unit revision mismatch: expected {expected_revision}, current {unit.revision}"
            )
        if expected_fingerprint is not None and unit.fingerprint != expected_fingerprint:
            raise RevisionConflictError("text unit fingerprint mismatch")

    @staticmethod
    def _text_unit(state: ProofState, text_unit_uid: str) -> ProofTextUnit:
        for unit in state.text_units:
            if unit.uid == text_unit_uid:
                return unit
        raise ProofTargetError(text_unit_uid)

    def _replace_state(self, current: ProofState, **changes: object) -> ProofState:
        candidate = replace(current, revision=current.revision, **changes)
        return self._session.proof_repository.replace_state(
            candidate,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
        )

    def _mark_rebind(
        self,
        current: ProofState,
        anchor_snapshot: ProofAnchorSnapshot,
    ) -> ProofState:
        candidate = replace(
            current,
            anchor_snapshot=anchor_snapshot,
            alignment_segments=(),
            alignment_slices=(),
            rebind_required=True,
            revision=current.revision,
        )
        return self._session.proof_repository.replace_state(
            candidate,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
        )

    def _apply_history(
        self,
        proof_uid: str,
        *,
        undo: bool,
        expected_revision: int | None,
        expected_fingerprint: str | None,
    ) -> ProofSessionResult:
        source = self._undo if undo else self._redo
        target = self._redo if undo else self._undo
        stack = source.setdefault(proof_uid, [])
        target_stack = target.setdefault(proof_uid, [])
        current = self.get_state(proof_uid)
        if expected_revision is not None or expected_fingerprint is not None:
            if expected_revision is None or expected_fingerprint is None:
                raise ValueError("history CAS requires revision and fingerprint together")
            current = self._cas_state(proof_uid, expected_revision, expected_fingerprint)
        if not stack:
            return self._result(
                ProofOperation.UNDO if undo else ProofOperation.REDO,
                ProofOutcome.NOOP,
                current,
                changed=False,
            )
        action = stack[-1]
        expected_content = action.after if undo else action.before
        if (
            current.revision != action.expected_revision
            or current.fingerprint != expected_content.fingerprint
        ):
            raise RevisionConflictError("proof history target is stale after an external change")
        target_content = action.before if undo else action.after
        stored = self._restore_content(current, target_content)
        stack.pop()
        if stack:
            stack[-1] = replace(stack[-1], expected_revision=stored.revision)
        target_stack.append(replace(action, expected_revision=stored.revision))
        if len(target_stack) > self._history_limit:
            del target_stack[:-self._history_limit]
        return self._result(
            ProofOperation.UNDO if undo else ProofOperation.REDO,
            ProofOutcome.APPLIED,
            stored,
            changed=True,
            changed_text_unit_uids=tuple(unit.uid for unit in stored.text_units),
        )

    def _restore_content(self, current: ProofState, target: ProofState) -> ProofState:
        target_by_uid = {unit.uid: unit for unit in target.text_units}
        current_by_uid = {unit.uid: unit for unit in current.text_units}
        next_units: list[ProofTextUnit] = []
        for target_unit in target.text_units:
            old_unit = current_by_uid.get(target_unit.uid)
            if old_unit is None:
                next_units.append(target_unit)
                continue
            if target_unit.text == old_unit.text and target_unit.status == old_unit.status:
                next_units.append(old_unit)
            else:
                next_units.append(replace(target_unit, revision=old_unit.revision + 1))
        if set(target_by_uid) != set(current_by_uid):
            raise ProofSessionError("proof history changed text-unit membership")
        candidate = replace(
            current,
            anchor_snapshot=target.anchor_snapshot,
            text_units=tuple(next_units),
            alignment_segments=target.alignment_segments,
            alignment_slices=target.alignment_slices,
            rebind_required=target.rebind_required,
            revision=current.revision,
        )
        return self._session.proof_repository.replace_state(
            candidate,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
        )

    def _record_history(
        self,
        before: ProofState,
        after: ProofState,
        operation: ProofOperation,
    ) -> None:
        stack = self._undo.setdefault(before.uid, [])
        stack.append(
            _HistoryEntry(
                before.uid,
                before,
                after,
                operation,
                expected_revision=after.revision,
            )
        )
        if len(stack) > self._history_limit:
            del stack[:-self._history_limit]
        self._redo.pop(before.uid, None)

    def _clear_history(self, proof_uid: str) -> None:
        self._undo.pop(proof_uid, None)
        self._redo.pop(proof_uid, None)

    @staticmethod
    def _editor_snapshot(
        state: ProofState,
        unit: ProofTextUnit,
    ) -> ProofEditorSnapshot:
        return ProofEditorSnapshot(
            proof_uid=state.uid,
            text_unit_uid=unit.uid,
            text=unit.text,
            status=unit.status,
            state_revision=state.revision,
            state_fingerprint=state.fingerprint,
            text_unit_revision=unit.revision,
            text_unit_fingerprint=unit.fingerprint,
        )

    @staticmethod
    def _result(
        operation: ProofOperation,
        outcome: ProofOutcome,
        state: ProofState,
        *,
        changed: bool,
        changed_text_unit_uids: tuple[str, ...] = (),
        message: str = "",
    ) -> ProofSessionResult:
        return ProofSessionResult(
            operation=operation,
            outcome=outcome,
            state=state,
            changed=changed,
            changed_text_unit_uids=changed_text_unit_uids,
            message=message,
        )


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("status must be non-empty text")
    return value


__all__ = [
    "HISTORY_LIMIT",
    "ProofConflictError",
    "ProofRefreshKind",
    "ProofSessionError",
    "ProofSessionService",
    "ProofTargetError",
]
