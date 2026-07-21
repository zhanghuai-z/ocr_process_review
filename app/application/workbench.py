"""Single application boundary for the active OCR workbench session."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.application.contracts import (
    LayoutEditCommand,
    LayoutEditResult,
    LayoutWorkspaceView,
    PageView,
    ProofEditCommand,
    ProofEditResult,
)
from app.application.layout_workspace import build_layout_workspace_view
from app.application.ocr_workspace import OcrWorkspaceView, build_ocr_workspace_view
from app.application.proof_workspace import ProofWorkspaceView, build_proof_workspace_view
from app.core.workflow_state import (
    PageGateInfo,
    active_line_count,
    compute_max_step,
    page_gate_info,
    pending_ocr_page_uids,
)
from app.core.ocr_currentness import current_ocr_observation
from app.models.export_snapshot import ExportProjectSnapshot
from app.models.project_session import (
    ProjectRecord,
    ProjectSession,
    RecordNotFoundError,
)
from app.services.export_service import capture_export_snapshot
from app.services.import_service import ImportJobRequest, ImportResult, ImportService
from app.services.layout_analysis_service import (
    LayoutAnalysisCommit,
    LayoutAnalysisService,
    LayoutPageJobRequest,
    LayoutPageJobResult,
)
from app.services.layout_edit_service import (
    LayoutEditCommand as DomainLayoutEditCommand,
    LayoutEditService,
)
from app.services.ocr_job_service import (
    OcrJobService,
    OcrPageCommit,
    OcrPageJobRequest,
    OcrPageJobResult,
)
from app.services.project_file_service import ProjectFileService
from app.services.proof_session_service import ProofSessionService


class WorkbenchApplication:
    """Own use-case services and expose no repository to UI/controller callers."""

    def __init__(
        self,
        *,
        session: ProjectSession | None = None,
        import_service: ImportService,
        project_file_service: ProjectFileService | None = None,
        layout_service: LayoutAnalysisService | None = None,
        ocr_service: OcrJobService | None = None,
    ) -> None:
        if session is not None and not isinstance(session, ProjectSession):
            raise TypeError("session must be ProjectSession or None")
        self._session = session
        self._import_service = import_service
        self._project_file_service = project_file_service or ProjectFileService()
        self._layout_service = layout_service
        self._ocr_service = ocr_service
        self._layout_edit_service = LayoutEditService()
        self._proof_service = ProofSessionService(session) if session is not None else None
        self._dirty = False

    @property
    def has_project(self) -> bool:
        return self._session is not None

    @property
    def project_uid(self) -> str:
        return self._require_session().project_uid if self._session is not None else ""

    @property
    def project_name(self) -> str:
        return self._require_session().project_record.name if self._session is not None else ""

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def is_bound_project(self) -> bool:
        return bool(self._session is not None and self._session.save_path)

    def ensure_project(self, project_uid: str, name: str) -> None:
        if self._session is not None:
            return
        self._replace_session(ProjectSession(ProjectRecord(project_uid, name=name)))

    def pages(self) -> tuple[PageView, ...]:
        if self._session is None:
            return ()
        return self.layout_workspace().pages

    def require_page(self, page_uid: str) -> PageView:
        for page in self.pages():
            if page.page_uid == page_uid:
                return page
        raise RecordNotFoundError(f"page not found: {page_uid!r}")

    def max_step(self) -> int:
        return compute_max_step(self._session)

    def active_line_count(self) -> int:
        return active_line_count(self._require_session()) if self._session is not None else 0

    def page_gate(self, page_uid: str) -> PageGateInfo:
        return page_gate_info(self._require_session(), page_uid)

    def pending_ocr_page_uids(self) -> tuple[str, ...]:
        return pending_ocr_page_uids(self._session)

    def has_layout(self, page_uid: str) -> bool:
        try:
            self._require_session().layout_repository.get(page_uid)
        except RecordNotFoundError:
            return False
        return True

    def layout_revision(self, page_uid: str) -> int:
        try:
            return self._require_session().layout_repository.get(page_uid).revision
        except RecordNotFoundError:
            return 0

    def has_ocr(self, page_uid: str) -> bool:
        return current_ocr_observation(self._require_session(), page_uid) is not None

    def layout_workspace(self) -> LayoutWorkspaceView:
        return build_layout_workspace_view(self._require_session())

    def ocr_workspace(self) -> OcrWorkspaceView:
        return build_ocr_workspace_view(self._require_session())

    def proof_workspace(self, *, proof_uid: str | None = None) -> ProofWorkspaceView:
        return build_proof_workspace_view(self._require_session(), proof_uid=proof_uid)

    def prepare_import(self, paths: Iterable[str | Path]) -> ImportJobRequest:
        session = self._require_session()
        return ImportJobRequest(
            project_uid=session.project_uid,
            paths=tuple(str(path) for path in paths),
            first_page_number=len(session.page_repository.all()) + 1,
            expected_page_uids=tuple(page.uid for page in session.page_repository.all()),
        )

    def execute_import(self, request: ImportJobRequest) -> ImportResult:
        return self._import_service.execute(request)

    def commit_import(self, result: ImportResult) -> ImportResult:
        committed = self._import_service.commit(self._require_session(), result)
        if committed.pages:
            self._dirty = True
        return committed

    def configure_layout_service(self, service: LayoutAnalysisService) -> None:
        self._layout_service = service

    def prepare_layout_page(self, page_uid: str) -> LayoutPageJobRequest:
        service = self._require_layout_service()
        return service.prepare_page(
            self._require_session(),
            page_uid,
            expected_revision=self.layout_revision(page_uid),
        )

    def execute_layout_page(self, request: LayoutPageJobRequest) -> LayoutPageJobResult:
        return self._require_layout_service().execute_page(request)

    def commit_layout_page(self, result: LayoutPageJobResult) -> LayoutAnalysisCommit:
        commit = self._require_layout_service().commit_page(self._require_session(), result)
        self._dirty = True
        return commit

    def configure_ocr_service(self, service: OcrJobService) -> None:
        self._ocr_service = service

    def prepare_ocr_page(self, page_uid: str) -> OcrPageJobRequest:
        return self._require_ocr_service().prepare_page(self._require_session(), page_uid)

    def execute_ocr_page(self, request: OcrPageJobRequest, *, progress_callback=None) -> OcrPageJobResult:
        return self._require_ocr_service().execute_page(
            request,
            progress_callback=progress_callback,
        )

    def commit_ocr_page(self, result: OcrPageJobResult) -> OcrPageCommit:
        commit = self._require_ocr_service().commit_page(self._require_session(), result)
        self._dirty = True
        return commit

    def apply_layout_edit(self, command: LayoutEditCommand) -> LayoutEditResult:
        if not isinstance(command, LayoutEditCommand):
            raise TypeError("layout edit requires application LayoutEditCommand")
        session = self._require_session()
        current = session.layout_repository.get(command.page_uid)
        domain_command = self._domain_layout_command(command)
        domain_result = self._layout_edit_service.apply(current, domain_command)
        session.layout_repository.put(domain_result.snapshot, expected_revision=current.revision)
        self._dirty = True
        page = next(
            page for page in self.layout_workspace().pages if page.page_uid == command.page_uid
        )
        return LayoutEditResult(
            command=command,
            page_uid=command.page_uid,
            revision_before=current.revision,
            revision_after=domain_result.snapshot.revision,
            affected_block_uids=domain_result.affected_block_uids,
            page_view=page,
            reason=domain_result.ocr_invalidation.reason,
        )

    def apply_proof_edit(self, command: ProofEditCommand) -> ProofEditResult:
        if not isinstance(command, ProofEditCommand):
            raise TypeError("proof edit requires ProofEditCommand")
        service = self._require_proof_service()
        common = {
            "expected_revision": command.expected_revision,
            "expected_fingerprint": command.expected_fingerprint,
        }
        if command.op == "replace_text":
            result = service.replace_text(
                command.proof_uid,
                command.text_unit_uid,
                command.text,
                expected_unit_revision=command.expected_unit_revision,
                expected_unit_fingerprint=command.expected_unit_fingerprint,
                status=command.status,
                **common,
            )
        elif command.op == "replace_many":
            result = service.replace_text_units(
                command.proof_uid,
                command.replacements,
                status=command.status,
                **common,
            )
        elif command.op == "set_status":
            result = service.set_status(
                command.proof_uid,
                command.text_unit_uid,
                command.status,
                expected_unit_revision=command.expected_unit_revision,
                expected_unit_fingerprint=command.expected_unit_fingerprint,
                **common,
            )
        elif command.op == "undo":
            result = service.undo(command.proof_uid, **common)
        else:
            result = service.redo(command.proof_uid, **common)
        if result.changed:
            self._dirty = True
        return ProofEditResult(
            command=command,
            changed=result.changed,
            proof_uid=result.state.uid,
            revision=result.state.revision,
            fingerprint=result.state.fingerprint,
            changed_text_unit_uids=result.changed_text_unit_uids,
        )

    def open_project(self, path: str | Path) -> None:
        self._replace_session(self._project_file_service.open_project(path).session)
        self._dirty = False

    def save_as(self, path: str | Path) -> None:
        self._replace_session(self._project_file_service.save_as(self._require_session(), path).session)
        self._dirty = False

    def save(self) -> None:
        self._replace_session(self._project_file_service.save_session(self._require_session()).session)
        self._dirty = False

    def export_snapshot(self) -> ExportProjectSnapshot:
        return capture_export_snapshot(self._require_session())

    def close_project(self) -> None:
        self._session = None
        self._proof_service = None
        self._dirty = False

    def _replace_session(self, session: ProjectSession) -> None:
        self._session = session
        self._proof_service = ProofSessionService(session)

    def _require_session(self) -> ProjectSession:
        if self._session is None:
            raise RuntimeError("no active project session")
        return self._session

    def _require_layout_service(self) -> LayoutAnalysisService:
        if self._layout_service is None:
            raise RuntimeError("layout analysis service is not configured")
        return self._layout_service

    def _require_ocr_service(self) -> OcrJobService:
        if self._ocr_service is None:
            raise RuntimeError("OCR service is not configured")
        return self._ocr_service

    def _require_proof_service(self) -> ProofSessionService:
        if self._proof_service is None:
            raise RuntimeError("proof service has no active project")
        return self._proof_service

    @staticmethod
    def _domain_layout_command(command: LayoutEditCommand) -> DomainLayoutEditCommand:
        op = command.op
        if op == "draw":
            return DomainLayoutEditCommand.create_block(
                command.page_uid,
                command.expected_revision,
                command.bbox,
                command.block_type,
                command.source_label,
                new_block_uid=command.new_block_uid,
            )
        if op == "delete":
            return DomainLayoutEditCommand.delete_block(
                command.page_uid, command.expected_revision, command.block_uid
            )
        if op in {"move", "resize"}:
            return DomainLayoutEditCommand.resize_block(
                command.page_uid,
                command.expected_revision,
                command.block_uid,
                bbox=command.bbox,
            )
        if op == "change_type":
            return DomainLayoutEditCommand.change_kind(
                command.page_uid,
                command.expected_revision,
                command.block_uid,
                block_type=command.block_type,
                source_label=command.source_label,
            )
        return DomainLayoutEditCommand.merge_blocks(
            command.page_uid,
            command.expected_revision,
            command.block_uids,
            command.bbox,
            block_type=command.block_type,
            source_label=command.source_label,
            primary_block_uid=command.primary_block_uid,
        )


__all__ = ["WorkbenchApplication"]
