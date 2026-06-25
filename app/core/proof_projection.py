"""Canonical proof projections compiled from Page/Block/Line facts."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.proof_atom import ProofAtom, build_line_proof_atoms
from app.core.proof_line_facts import proof_display_text
from app.core.proof_occurrence import line_signature, proof_line_identity_key
from app.core.proof_state import ProofLineViewModel
from app.models import Block, Line, Page


@dataclass(frozen=True)
class ProofLineProjection:
    """One line-level proof fact consumed by proof views.

    This is runtime-only. It does not persist and it does not own mutations.
    HProof/VProof may render it differently, but they should not reinterpret the
    same line's display text, atom sequence, or identity independently.
    """

    uid: str
    identity_key: tuple[object, ...]
    page: Page
    block: Block
    line: Line
    line_index: int
    display_text: str
    line_signature: str
    atoms: tuple[ProofAtom, ...]
    editable: bool = True

    def view_model(self, *, source: str = "") -> ProofLineViewModel:
        return ProofLineViewModel.from_model(
            page=self.page,
            block=self.block,
            line=self.line,
            line_index=self.line_index,
            display_text=self.display_text,
            source=source,
        )


def build_proof_line_projection(
    page: Page,
    block: Block,
    line: Line,
    line_index: int,
    *,
    display_text: str | None = None,
    editable: bool | None = None,
) -> ProofLineProjection:
    identity_key = proof_line_identity_key(page, block, line, line_index)
    text = proof_display_text(line) if display_text is None else display_text
    return ProofLineProjection(
        uid=_projection_uid(identity_key),
        identity_key=identity_key,
        page=page,
        block=block,
        line=line,
        line_index=line_index,
        display_text=text,
        line_signature=line_signature(line),
        atoms=tuple(build_line_proof_atoms(block, line)),
        editable=(line_index >= 0) if editable is None else bool(editable),
    )


def _projection_uid(identity_key: tuple[object, ...]) -> str:
    return ":".join(str(part) for part in identity_key)
