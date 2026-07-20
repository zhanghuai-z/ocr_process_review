"""Read-only character indexing from an active OCR batch and proof state."""
from __future__ import annotations

from collections import Counter

from app.core.char_index import (
    GEOMETRY_AVAILABLE,
    GEOMETRY_UNAVAILABLE,
    CharIndex,
    CharIndexEntry,
)
from app.core.proof_alignment import atom_span_for_source_index, parse_alignment
from app.models.ocr_records import OcrAtom, OcrBatch, OcrLine
from app.models.proof_records import ProofState, ProofTextUnit
from app.models.project_session import PageRecord


class CharIndexService:
    """Compile one immutable index; all input records are read-only snapshots."""

    def build(
        self,
        *,
        page: PageRecord,
        batch: OcrBatch,
        lines: tuple[OcrLine, ...],
        atoms: tuple[OcrAtom, ...],
        state: ProofState,
    ) -> CharIndex:
        if page.uid != batch.scope_uid:
            raise ValueError("page and active OCR batch scope do not match")
        if state.anchor_snapshot.scope_uid != batch.scope_uid:
            raise ValueError("proof state and active OCR batch scope do not match")
        if {line.uid for line in lines} != set(batch.line_uids):
            raise ValueError("active batch line membership is incomplete")
        if {atom.uid for atom in atoms} != set(batch.atom_uids):
            raise ValueError("active batch atom membership is incomplete")
        if any(line.page_uid != page.uid for line in lines):
            raise ValueError("OCR line belongs to another page")
        if any(atom.line_uid not in {line.uid for line in lines} for atom in atoms):
            raise ValueError("OCR atom is outside the active batch lines")

        observation_mismatch = (
            state.anchor_snapshot.source_fingerprint != batch.fingerprint
            or state.anchor_snapshot.layout_fingerprint != batch.layout_fingerprint
        )
        alignment = parse_alignment(
            state,
            lines,
            atoms,
            force_unavailable_reason=(
                "active_observation_changed_requires_rebind"
                if observation_mismatch
                else ""
            ),
        )
        unit_offsets = _unit_offsets(state)
        entries: list[CharIndexEntry] = []
        for unit, start in unit_offsets:
            for offset, text in enumerate(unit.text):
                proof_index = start + offset
                source_index = alignment.source_index_for(proof_index)
                atom_span = (
                    atom_span_for_source_index(alignment, source_index)
                    if source_index is not None
                    else None
                )
                reason = alignment.reason_for(proof_index)
                if atom_span is None:
                    entries.append(
                        CharIndexEntry(
                            proof_uid=state.uid,
                            text_unit_uid=unit.uid,
                            char_index=offset,
                            text=text,
                            page_uid=page.uid,
                            page_number=page.page_number,
                            image_path=page.image_path,
                            scope_uid=batch.scope_uid,
                            batch_uid=batch.uid,
                            line_uid="",
                            atom_uid=None,
                            atom_index=None,
                            bbox=None,
                            geometry_status=GEOMETRY_UNAVAILABLE,
                            unavailable_reason=reason or "no_ocr_atom_for_alignment",
                        )
                    )
                    continue
                atom = atom_span.atom
                entries.append(
                    CharIndexEntry(
                        proof_uid=state.uid,
                        text_unit_uid=unit.uid,
                        char_index=offset,
                        text=text,
                        page_uid=page.uid,
                        page_number=page.page_number,
                        image_path=page.image_path,
                        scope_uid=batch.scope_uid,
                        batch_uid=batch.uid,
                        line_uid=atom_span.line.uid,
                        atom_uid=atom.uid,
                        atom_index=atom.index,
                        bbox=atom.bbox,
                        geometry_status=GEOMETRY_AVAILABLE,
                        line_order=atom_span.line.order,
                        atom_fingerprint=atom.fingerprint,
                    )
                )
        entries.sort(
            key=lambda item: (
                item.page_number,
                item.line_order,
                item.atom_index if item.atom_index is not None else 10**9,
                item.char_index,
                item.text_unit_uid,
            )
        )
        return CharIndex(
            proof_uid=state.uid,
            state_revision=state.revision,
            state_fingerprint=state.fingerprint,
            batch_uid=batch.uid,
            entries=tuple(entries),
            unavailable_reason=(
                "proof_state_requires_rebind"
                if state.rebind_required
                else (
                    "active_observation_changed_requires_rebind"
                    if observation_mismatch
                    else ""
                )
            ),
        )

    @staticmethod
    def char_frequency(index: CharIndex) -> tuple[tuple[str, int], ...]:
        counts = Counter(entry.text for entry in index.entries if entry.available)
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @staticmethod
    def same_text(
        index: CharIndex,
        text: str,
        *,
        include_unavailable: bool = False,
    ) -> tuple[CharIndexEntry, ...]:
        return index.same_text(text, include_unavailable=include_unavailable)


def _unit_offsets(state: ProofState) -> tuple[tuple[ProofTextUnit, int], ...]:
    offset = 0
    result: list[tuple[ProofTextUnit, int]] = []
    for unit in sorted(state.text_units, key=lambda item: item.order):
        result.append((unit, offset))
        offset += len(unit.text)
    return tuple(result)


def char_entry_display_text(entry: CharIndexEntry) -> str:
    """Return the proof text represented by an index entry."""

    return entry.text


__all__ = [
    "CharIndex",
    "CharIndexEntry",
    "CharIndexService",
    "GEOMETRY_AVAILABLE",
    "GEOMETRY_UNAVAILABLE",
    "char_entry_display_text",
]
