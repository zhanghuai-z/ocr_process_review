"""Persistence boundary for proof edits."""
from __future__ import annotations

import logging

from app.core import quality_probe as qp
from app.core.proof_change import ProofChangeSet, ProofLineRef
from app.core.project_store import ProjectStore
from app.models import Block, Line, OcrProject, Page
from app.models.ocr_observation import iter_project_ocr_line_occurrences

logger = logging.getLogger(__name__)


class ProofPersistenceService:
    """Persist proof-side mutations described by ``ProofChangeSet``.

    The service intentionally does not infer change types from line status.
    Panels and proof services own mutation detection; this service owns the
    storage side effects.
    """

    def __init__(self, store: ProjectStore, project: OcrProject) -> None:
        self._store = store
        self._project = project

    def persist(self, change: ProofChangeSet | None = None) -> bool:
        full_project_save = change is None
        if change is None:
            change = ProofChangeSet(text_changed=True, status_changed=True, probe_changed=True)
        if change.blocked or not change.needs_persist:
            return False

        persisted = False
        if change.text_changed or change.status_changed:
            if full_project_save:
                persisted = self._persist_all_lines() or persisted
            else:
                persisted = self._persist_scoped_lines(change) or persisted
        if change.probe_changed:
            persisted = self.persist_quality_probe_sidecar() or persisted
        return persisted

    def persist_quality_probe_sidecar(self) -> bool:
        if not self._project.db_path:
            return False
        store = qp.get_active_store()
        if store is None:
            return False
        side = qp.sidecar_path_for_project(self._project.db_path)
        if not side:
            return False
        return qp.save_store_to_path(store, side)

    def _persist_all_lines(self) -> bool:
        updates: list[tuple[Line, bool]] = []
        for occurrence in iter_project_ocr_line_occurrences(self._project):
            if not occurrence.line.id:
                continue
            updates.append((occurrence.line, True))
        if not updates:
            return False
        try:
            self._store.update_proof_lines(updates)
        except Exception:
            logger.exception("Failed to persist proof lines")
            raise
        return True

    def _persist_scoped_lines(self, change: ProofChangeSet) -> bool:
        if not change.line_refs:
            logger.warning(
                "Ignoring unscoped proof line change: text=%s status=%s",
                change.text_changed,
                change.status_changed,
            )
            return False
        updates: list[tuple[Line, bool]] = []
        for ref in change.line_refs:
            line = self._resolve_line_ref(ref)
            if line is None or not line.id:
                logger.warning(
                    "Ignoring stale proof line ref: line_id=%s line_uid=%s",
                    ref.line_id,
                    ref.line_uid,
                )
                continue
            updates.append((line, ref.write_chars))
        if not updates:
            return False
        try:
            self._store.update_proof_lines(updates)
        except Exception:
            logger.exception("Failed to persist scoped proof lines")
            raise
        return True

    def _resolve_line_ref(self, ref: ProofLineRef) -> Line | None:
        for occurrence in iter_project_ocr_line_occurrences(self._project):
            page = occurrence.page
            block = occurrence.block
            line = occurrence.line
            if not self._matches_container(ref, page, block):
                continue
            if ref.line is not None and line is ref.line:
                return line
            if ref.line_uid and line.uid == ref.line_uid:
                return line
            if ref.line_id is not None and line.id == ref.line_id:
                return line
        return None

    def _matches_container(self, ref: ProofLineRef, page: Page, block: Block) -> bool:
        if ref.page_uid and page.uid != ref.page_uid:
            return False
        if ref.page_id is not None and page.id != ref.page_id:
            return False
        if ref.block_uid and block.uid != ref.block_uid:
            return False
        if ref.block_id is not None and block.id != ref.block_id:
            return False
        return True
