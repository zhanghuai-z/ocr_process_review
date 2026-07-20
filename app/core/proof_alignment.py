"""Explicit source/proof alignment parsing for proof character geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.models.ocr_records import OcrAtom, OcrLine
from app.models.proof_records import ProofState


@dataclass(frozen=True, slots=True)
class SourceAtomSpan:
    atom: OcrAtom
    line: OcrLine
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class AlignmentParse:
    source_text: str
    proof_text: str
    atom_spans: tuple[SourceAtomSpan, ...]
    proof_to_source: Mapping[int, int]
    unavailable_by_proof_index: Mapping[int, str]

    def source_index_for(self, proof_index: int) -> int | None:
        return self.proof_to_source.get(proof_index)

    def reason_for(self, proof_index: int) -> str:
        return self.unavailable_by_proof_index.get(proof_index, "")


def parse_alignment(
    state: ProofState,
    lines: tuple[OcrLine, ...],
    atoms: tuple[OcrAtom, ...],
    *,
    force_unavailable_reason: str = "",
) -> AlignmentParse:
    """Map proof positions to atom source positions without fuzzy inference.

    Source offsets are the concatenation of the active OCR line text values.
    A slice is usable only when both its declared ranges and its text values
    agree exactly.  Any malformed, missing, overlapping, or unequal-length
    mapping is marked unavailable at the affected proof positions.
    """

    ordered_lines = tuple(sorted(lines, key=lambda item: (item.order, item.uid)))
    atom_by_uid = {atom.uid: atom for atom in atoms}
    source_parts: list[str] = []
    atom_spans: list[SourceAtomSpan] = []
    source_cursor = 0
    for line in ordered_lines:
        line_atoms = [atom_by_uid[uid] for uid in line.atom_uids if uid in atom_by_uid]
        line_atoms.sort(key=lambda item: (item.index, item.uid))
        line_text = "".join(atom.text for atom in line_atoms)
        line_offset = source_cursor
        source_parts.append(line.text)
        source_cursor += len(line.text)
        if line_text != line.text:
            continue
        atom_cursor = line_offset
        for atom in line_atoms:
            atom_spans.append(
                SourceAtomSpan(
                    atom=atom,
                    line=line,
                    source_start=atom_cursor,
                    source_end=atom_cursor + len(atom.text),
                )
            )
            atom_cursor += len(atom.text)

    proof_units = tuple(sorted(state.text_units, key=lambda item: item.order))
    proof_text = "".join(unit.text for unit in proof_units)
    source_text = "".join(source_parts)
    proof_to_source: dict[int, int] = {}
    unavailable: dict[int, str] = {}
    for item in state.alignment_slices:
        proof_length = item.proof_end - item.proof_start
        source_length = item.source_end - item.source_start
        affected = range(
            max(0, item.proof_start),
            min(len(proof_text), max(item.proof_start, item.proof_end)),
        )
        reason = ""
        if proof_length != len(item.proof_text) or source_length != len(item.source_text):
            reason = "alignment_range_length_mismatch"
        elif proof_length != source_length:
            reason = "non_positional_alignment"
        elif item.proof_start < 0 or item.proof_end > len(proof_text):
            reason = "proof_range_out_of_bounds"
        elif item.source_start < 0 or item.source_end > len(source_text):
            reason = "source_range_out_of_bounds"
        elif proof_text[item.proof_start:item.proof_end] != item.proof_text:
            reason = "proof_slice_text_mismatch"
        elif source_text[item.source_start:item.source_end] != item.source_text:
            reason = "source_slice_text_mismatch"
        if reason:
            for proof_index in affected:
                unavailable[proof_index] = reason
            continue
        for offset in range(proof_length):
            proof_index = item.proof_start + offset
            if proof_index in proof_to_source:
                unavailable[proof_index] = "overlapping_alignment_slices"
                proof_to_source.pop(proof_index, None)
                continue
            proof_to_source[proof_index] = item.source_start + offset

    if force_unavailable_reason:
        proof_to_source.clear()
        for proof_index in range(len(proof_text)):
            unavailable[proof_index] = force_unavailable_reason
    elif state.rebind_required:
        for proof_index in range(len(proof_text)):
            proof_to_source.pop(proof_index, None)
            unavailable[proof_index] = "proof_state_requires_rebind"
    else:
        for proof_index in range(len(proof_text)):
            if proof_index not in proof_to_source and proof_index not in unavailable:
                unavailable[proof_index] = "no_alignment_for_proof_position"

    return AlignmentParse(
        source_text=source_text,
        proof_text=proof_text,
        atom_spans=tuple(atom_spans),
        proof_to_source=proof_to_source,
        unavailable_by_proof_index=unavailable,
    )


def atom_span_for_source_index(
    alignment: AlignmentParse,
    source_index: int,
) -> SourceAtomSpan | None:
    for span in alignment.atom_spans:
        if span.source_start <= source_index < span.source_end:
            return span
    return None


__all__ = [
    "AlignmentParse",
    "SourceAtomSpan",
    "atom_span_for_source_index",
    "parse_alignment",
]
