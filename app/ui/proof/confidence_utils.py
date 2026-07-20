"""Read-only confidence and proof-context helpers for the v2 proof UI."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Optional

from app.core.char_index import CharIndex
from app.models.ocr_records import OcrAtom, OcrBatch, OcrLine
from app.models.proof_records import ProofState
from app.models.project_session import PageRecord, ProjectSession
from app.services.proof_session_service import ProofSessionService


def normalize_confidence(raw: object) -> Optional[float]:
    """Return a finite confidence in ``0..1`` or ``None`` when unavailable."""

    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value <= 0:
        return None
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    if value > 1.0:
        return None
    return max(0.0, value)


def line_confidence(
    observation: OcrLine,
    atoms: Iterable[OcrAtom] = (),
) -> Optional[float]:
    """Average the supplied atom scores, falling back to the line score."""

    scores = [
        score
        for atom in atoms
        if (score := normalize_confidence(atom.confidence)) is not None
    ]
    if scores:
        return sum(scores) / len(scores)
    return normalize_confidence(observation.confidence)


def char_confidence(
    atom: OcrAtom | None,
    observation: OcrLine | None = None,
) -> Optional[float]:
    """Read one atom score, with an explicit line-level fallback."""

    if atom is not None:
        score = normalize_confidence(atom.confidence)
        if score is not None:
            return score
    if observation is None:
        return None
    return normalize_confidence(observation.confidence)


@dataclass(frozen=True, slots=True)
class ProofContext:
    """Derived read-only context for one proof state and page scope."""

    page: PageRecord
    state: ProofState
    batch: OcrBatch
    lines: tuple[OcrLine, ...]
    atoms: tuple[OcrAtom, ...]
    index: CharIndex

    @property
    def lines_by_uid(self) -> dict[str, OcrLine]:
        return {item.uid: item for item in self.lines}

    @property
    def atoms_by_uid(self) -> dict[str, OcrAtom]:
        return {item.uid: item for item in self.atoms}


def build_proof_contexts(
    session: ProjectSession,
    proof_service: ProofSessionService,
) -> tuple[ProofContext, ...]:
    """Build UI projections from the active OCR pointer and proof aggregate.

    The service may mark a stale anchor as requiring rebind while refreshing the
    projection. No OCR or page record is mutated here.
    """

    if proof_service.project_uid != session.project_uid:
        raise ValueError("proof service and project session must share a project UID")

    pages = {page.uid: page for page in session.page_repository.all()}
    ocr = session.ocr_observation_repository
    contexts: list[ProofContext] = []
    states = sorted(
        session.proof_repository.all_states(),
        key=lambda state: (state.anchor_snapshot.scope_uid, state.uid),
    )
    for state in states:
        page = pages.get(state.anchor_snapshot.scope_uid)
        if page is None:
            continue
        refreshed = proof_service.refresh_active_observation(
            state.uid,
            expected_revision=state.revision,
            expected_fingerprint=state.fingerprint,
        )
        current = refreshed.state
        pointer = ocr.get_active_pointer(page.uid)
        batch = ocr.get_batch(pointer.batch_uid, fingerprint=pointer.batch_fingerprint)
        lines = tuple(ocr.get_line(uid) for uid in batch.line_uids)
        atoms = tuple(ocr.get_atom(uid) for uid in batch.atom_uids)
        index = proof_service.build_char_index(
            current.uid,
            expected_revision=current.revision,
            expected_fingerprint=current.fingerprint,
        )
        contexts.append(
            ProofContext(
                page=page,
                state=current,
                batch=batch,
                lines=lines,
                atoms=atoms,
                index=index,
            )
        )
    return tuple(contexts)


__all__ = [
    "ProofContext",
    "build_proof_contexts",
    "char_confidence",
    "line_confidence",
    "normalize_confidence",
]
