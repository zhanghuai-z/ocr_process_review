"""Runtime session state for the horizontal proof view.

The HProof panel owns Qt widgets and user interaction. This object owns the
non-widget runtime facts that must survive small rebuilds: loaded pages,
current projection index, page filter, debug filter, and pending external
updates.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.proof_projection import ProofLineProjection
from app.core.proof_state import ProofUpdateRequest
from app.models import Page
from app.services.proof_external_refresh import (
    ProofExternalRefreshQueue,
    ProofExternalLineRef,
)


PagePredicate = Callable[[Page], bool]


@dataclass(frozen=True)
class HProofExternalRefreshPlan:
    """External line updates collapsed into HProof projection indexes."""

    touched_projection_indexes: tuple[int, ...] = tuple()

    @property
    def has_work(self) -> bool:
        return bool(self.touched_projection_indexes)


@dataclass
class HProofRuntimeSession:
    pages: list[Page] = field(default_factory=list)
    projections: list[ProofLineProjection] = field(default_factory=list)
    current_projection_index: int = 0
    filter_updating: bool = False
    selected_page_number: int | None = None
    show_formula_debug: bool = False
    show_table_debug: bool = False
    _external_refresh_queue: ProofExternalRefreshQueue = field(default_factory=ProofExternalRefreshQueue)

    def reset(self) -> None:
        self.pages = []
        self.projections.clear()
        self.current_projection_index = 0
        self.filter_updating = False
        self.selected_page_number = None
        self.show_formula_debug = False
        self.show_table_debug = False
        self.clear_pending_external_refresh()

    def set_pages(
        self,
        pages: list[Page],
        *,
        page_has_lines: PagePredicate | None = None,
    ) -> None:
        self.pages = pages
        self.ensure_selected_page_exists(page_has_lines)

    def usable_pages(self, page_has_lines: PagePredicate | None = None) -> list[Page]:
        if page_has_lines is None:
            return list(self.pages)
        return [page for page in self.pages if page_has_lines(page)]

    def ensure_selected_page_exists(
        self,
        page_has_lines: PagePredicate | None = None,
    ) -> None:
        if self.selected_page_number is None:
            return
        page_numbers = {
            page.page_number for page in self.usable_pages(page_has_lines)
        }
        if self.selected_page_number not in page_numbers:
            self.selected_page_number = None

    def filtered_pages(self) -> list[Page]:
        if self.selected_page_number is None:
            return list(self.pages)
        return [
            page for page in self.pages
            if page.page_number == self.selected_page_number
        ]

    def set_debug_flags(self, *, formula: bool, table: bool) -> None:
        self.show_formula_debug = bool(formula)
        self.show_table_debug = bool(table)

    @property
    def debug_enabled(self) -> bool:
        return self.show_formula_debug or self.show_table_debug

    @property
    def debug_label(self) -> str:
        labels: list[str] = []
        if self.show_formula_debug:
            labels.append("公式")
        if self.show_table_debug:
            labels.append("表格")
        return " / ".join(labels)

    def clear_pending_external_refresh(self) -> None:
        self._external_refresh_queue.clear()

    @property
    def has_pending_external_refresh(self) -> bool:
        return self._external_refresh_queue.has_pending

    def has_projection_for_request(self, request: ProofUpdateRequest) -> bool:
        line_ref = ProofExternalLineRef.from_request(request)
        if not line_ref.is_valid:
            return False
        return any(
            line_ref.matches_line(projection.line)
            for projection in self.projections
        )

    def queue_external_refresh(self, request: ProofUpdateRequest) -> None:
        self._external_refresh_queue.queue_request(request)

    def consume_external_refresh_plan(self) -> HProofExternalRefreshPlan:
        batch = self._external_refresh_queue.consume()
        if not batch.line_refs:
            return HProofExternalRefreshPlan()
        touched_indexes = {
            index
            for index, projection in enumerate(self.projections)
            if any(
                line_ref.matches_line(projection.line)
                for line_ref in batch.line_refs
            )
        }
        return HProofExternalRefreshPlan(
            touched_projection_indexes=tuple(sorted(touched_indexes)),
        )


@dataclass
class HProofLineEditSession:
    """Non-widget edit state for one horizontal proof row."""

    editable: bool
    loaded_display_text: str
    loaded_line_signature: str
    external_conflict: bool = False

    def is_dirty(self, editor_text: str) -> bool:
        return editor_text != self.loaded_display_text

    def dirty_snapshot(self, editor_text: str) -> tuple[str, str]:
        return editor_text, self.loaded_display_text

    def mark_external_conflict(self) -> None:
        self.external_conflict = True

    def mark_saved(self, loaded_display_text: str, loaded_line_signature: str) -> None:
        self.loaded_display_text = loaded_display_text
        self.loaded_line_signature = loaded_line_signature
        self.external_conflict = False

    def clear_conflict_if_editor_matches_model(
        self,
        *,
        editor_text: str,
        model_display_text: str,
        model_line_signature: str,
    ) -> bool:
        if not self.external_conflict or editor_text != model_display_text:
            return False
        self.mark_saved(model_display_text, model_line_signature)
        return True

    def restore_dirty_editor_text(
        self,
        *,
        editor_text: str,
        previous_loaded_text: str | None,
    ) -> None:
        self.external_conflict = bool(
            previous_loaded_text is not None
            and previous_loaded_text != self.loaded_display_text
            and editor_text != self.loaded_display_text
        )

    def rebind_to_model(
        self,
        *,
        editor_text: str,
        model_display_text: str,
        model_line_signature: str,
    ) -> bool:
        old_loaded_text = self.loaded_display_text
        was_dirty = editor_text != old_loaded_text
        external_changed = model_display_text != old_loaded_text
        self.loaded_display_text = model_display_text
        self.loaded_line_signature = model_line_signature
        self.external_conflict = bool(
            was_dirty
            and external_changed
            and editor_text != model_display_text
        )
        return was_dirty
