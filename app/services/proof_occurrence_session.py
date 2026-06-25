"""Runtime session state for occurrence-based proof views."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.proof_occurrence import (
    ProofOccurrence,
    proof_occurrence_key,
    proof_page_identity_key,
)
from app.models import Page
from app.services.proof_reference_context import (
    ProofReferenceContext,
    build_proof_reference_context,
)


@dataclass
class VProofOccurrenceSession:
    """UI-free session state for vertical proof occurrence views."""

    pages: list[Page] = field(default_factory=list)
    current_page_key: tuple[object, ...] | None = None
    reference_context: ProofReferenceContext | None = None
    loaded_text: str = ""
    selected_tokens: list[str] = field(default_factory=list)
    selected_occurrence_key: tuple[object, ...] | None = None
    selected_occurrence_keys: list[tuple[object, ...]] = field(default_factory=list)
    pending_external_lines: set[int | str] = field(default_factory=set)
    pending_external_page_keys: set[tuple[object, ...]] = field(default_factory=set)

    def reset(self) -> None:
        self.pages = []
        self.current_page_key = None
        self.reference_context = None
        self.loaded_text = ""
        self.selected_tokens = []
        self.selected_occurrence_key = None
        self.selected_occurrence_keys = []
        self.pending_external_lines.clear()
        self.pending_external_page_keys.clear()

    def set_pages(
        self,
        pages: list[Page],
        *,
        current_page_key: tuple[object, ...] | None = None,
    ) -> None:
        self.pages = pages
        page_keys = [proof_page_identity_key(page) for page in pages]
        if current_page_key in page_keys:
            self.current_page_key = current_page_key
        elif pages:
            self.current_page_key = page_keys[0]
        else:
            self.current_page_key = None
        if self.current_page_key is None:
            self.reference_context = None
            self.loaded_text = ""

    def current_page_index(self) -> int:
        if not self.pages:
            return 0
        if self.current_page_key is None:
            return 0
        for idx, page in enumerate(self.pages):
            if proof_page_identity_key(page) == self.current_page_key:
                return idx
        return 0

    def current_page(self) -> Page | None:
        if not self.pages:
            return None
        return self.pages[self.current_page_index()]

    def set_current_page_by_index(self, idx: int) -> Page | None:
        if not self.pages:
            self.current_page_key = None
            self.reference_context = None
            self.loaded_text = ""
            return None
        idx = max(0, min(int(idx), len(self.pages) - 1))
        page = self.pages[idx]
        self.current_page_key = proof_page_identity_key(page)
        return page

    def find_page_index(self, target: Page) -> int:
        if not self.pages:
            return 0
        target_key = proof_page_identity_key(target)
        for idx, page in enumerate(self.pages):
            if proof_page_identity_key(page) == target_key:
                return idx
        for idx, page in enumerate(self.pages):
            if page is target:
                return idx
        return self.current_page_index()

    def load_reference_context(self, page: Page) -> ProofReferenceContext:
        context = build_proof_reference_context(page)
        self.reference_context = context
        self.loaded_text = context.text
        self.current_page_key = proof_page_identity_key(page)
        return context

    def mark_selected_occurrence(self, occurrence: ProofOccurrence) -> None:
        self.set_selected_occurrences([occurrence])

    def set_selected_occurrences(self, occurrences: list[ProofOccurrence]) -> None:
        keys = [proof_occurrence_key(occurrence) for occurrence in occurrences]
        self.selected_occurrence_key = keys[0] if keys else None
        self.selected_occurrence_keys = keys

    def clear_selection(self) -> None:
        self.selected_occurrence_key = None
        self.selected_occurrence_keys = []
