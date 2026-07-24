"""Session-scoped application orchestration for the OCR workbench."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import tempfile
from threading import Lock

from PySide6.QtCore import QObject, QThread, Signal

from app.application import (
    ImportCompletionView,
    ImportFailureView,
    LayoutEditCommand,
    LayoutEditResult,
    PageView,
    ProofBatchEditCommand,
    ProofBatchEditResult,
    ProofEditCommand,
    ProofEditResult,
    WorkbenchApplication,
)
from app.core.app_config import get_config
from app.core.workflow_state import (
    STEP_HPROOF,
    STEP_IMPORT,
    STEP_LAYOUT,
    STEP_OCR,
    STEP_VPROOF,
    WorkflowProgressState,
    WorkflowViewState,
)
from app.models.entity_id import new_ulid
from app.models.export_snapshot import ExportProjectSnapshot
from app.services import (
    ImportResult,
    ImportJobRequest,
    ImportService,
    LayoutAnalysisCommit,
    LayoutAnalysisService,
    LayoutPageJobRequest,
    LayoutPageJobResult,
    OcrJobService,
    OcrPageCommit,
    OcrPageJobFailure,
    OcrPageJobRequest,
    OcrPageJobResult,
    ProjectFileService,
)


class _TaskCancelled(RuntimeError):
    """Internal worker control flow for cooperative cancellation."""


def _configured_page_workers(
    key: str,
    *,
    default: int,
    cap: int,
    total: int,
) -> int:
    """Resolve one bounded page-worker count from application settings."""

    if total <= 1:
        return 1
    try:
        configured = int(get_config().get(key, default))
    except (TypeError, ValueError):
        configured = default
    return max(1, min(total, cap, configured))


class _ImportServiceWorker(QThread):
    """Decode immutable import input without access to ProjectSession."""

    completed = Signal(str, object)
    failed = Signal(str)

    def __init__(
        self,
        service: ImportService,
        request: ImportJobRequest,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._request = request

    def run(self) -> None:
        try:
            result = self._service.execute(self._request)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(self._request.project_uid, result)


class _LayoutServiceWorker(QThread):
    """Execute immutable layout requests without access to ProjectSession."""

    committed = Signal(object)
    progress = Signal(int, int)
    stage = Signal(str, int, int, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        service: LayoutAnalysisService,
        requests: tuple[LayoutPageJobRequest, ...],
        *,
        max_workers: int = 1,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._requests = requests
        self._max_workers = max(1, min(len(requests) or 1, int(max_workers)))

    def run(self) -> None:
        try:
            total = len(self._requests)
            if self._max_workers <= 1:
                results: list[LayoutPageJobResult] = []
                for index, request in enumerate(self._requests):
                    if self.isInterruptionRequested():
                        raise _TaskCancelled
                    self.stage.emit(request.page_uid, index + 1, total, "版面分析")
                    results.append(self._service.execute_page(request))
                    self.progress.emit(index + 1, total)
                if self.isInterruptionRequested():
                    raise _TaskCancelled
                self.committed.emit(tuple(results))
                return

            indexed_results: list[LayoutPageJobResult | None] = [None] * total
            executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="layout-page",
            )
            futures = {
                executor.submit(self._service.execute_page, request): index
                for index, request in enumerate(self._requests)
            }
            pending = set(futures)
            completed = 0
            try:
                while pending:
                    if self.isInterruptionRequested():
                        raise _TaskCancelled
                    done, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        index = futures[future]
                        request = self._requests[index]
                        indexed_results[index] = future.result()
                        completed += 1
                        self.progress.emit(completed, total)
                        self.stage.emit(request.page_uid, completed, total, "版面分析")
                if self.isInterruptionRequested():
                    raise _TaskCancelled
            finally:
                if self.isInterruptionRequested():
                    for future in pending:
                        future.cancel()
                executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            if any(result is None for result in indexed_results):
                raise RuntimeError("layout worker completed without every page result")
            self.committed.emit(tuple(indexed_results))
        except _TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return


class _OcrServiceWorker(QThread):
    """Execute immutable OCR requests without access to ProjectSession."""

    committed = Signal(object)
    progress = Signal(object)
    cancelled = Signal()
    failed = Signal(str)
    page_failed = Signal(object)

    def __init__(
        self,
        service: OcrJobService,
        requests: tuple[OcrPageJobRequest, ...],
        *,
        max_workers: int = 1,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._requests = requests
        self._max_workers = max(1, min(len(requests) or 1, int(max_workers)))

    def run(self) -> None:
        total = len(self._requests)
        indexed_results: list[OcrPageJobResult | None] = [None] * total
        completed_pages = 0
        completed_lock = Lock()
        emit_lock = Lock()

        def completed_count() -> int:
            with completed_lock:
                return completed_pages

        def execute(request: OcrPageJobRequest) -> OcrPageJobResult:
            page_uid = request.page.uid
            with emit_lock:
                self.progress.emit(WorkflowProgressState(
                    phase="ocr",
                    current=0,
                    total=0,
                    completed_pages=completed_count(),
                    total_pages=total,
                    message="准备识别",
                    page_uid=page_uid,
                ))

            def on_progress(current: int, block_total: int, message: str) -> None:
                if self.isInterruptionRequested():
                    raise _TaskCancelled
                with emit_lock:
                    self.progress.emit(WorkflowProgressState(
                        phase="ocr",
                        current=current,
                        total=block_total,
                        completed_pages=completed_count(),
                        total_pages=total,
                        message=message,
                        page_uid=page_uid,
                    ))

            return self._service.execute_page(
                request,
                progress_callback=on_progress,
            )

        try:
            executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="charocr-page",
            )
            futures = {
                executor.submit(execute, request): index
                for index, request in enumerate(self._requests)
            }
            pending = set(futures)
            try:
                while pending:
                    if self.isInterruptionRequested():
                        raise _TaskCancelled
                    done, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        index = futures[future]
                        request = self._requests[index]
                        page_succeeded = False
                        try:
                            result = future.result()
                        except _TaskCancelled:
                            raise
                        except Exception as exc:
                            message = str(exc).strip() or type(exc).__name__
                            self.page_failed.emit(OcrPageJobFailure(request, message))
                        else:
                            if not isinstance(result, OcrPageJobResult):
                                self.page_failed.emit(OcrPageJobFailure(
                                    request,
                                    "OcrJobService returned an invalid result",
                                ))
                            else:
                                indexed_results[index] = result
                                page_succeeded = True
                        with completed_lock:
                            completed_pages += 1
                            completed = completed_pages
                        self.progress.emit(WorkflowProgressState(
                            phase="ocr",
                            current=1,
                            total=1,
                            completed_pages=completed,
                            total_pages=total,
                            message="已完成" if page_succeeded else "识别失败",
                            page_uid=request.page.uid,
                        ))
                if self.isInterruptionRequested():
                    raise _TaskCancelled
            finally:
                if self.isInterruptionRequested():
                    for future in pending:
                        future.cancel()
                executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
        except _TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.committed.emit(tuple(
            result for result in indexed_results if result is not None
        ))


class WorkflowController(QObject):
    """Qt orchestration facade over the single workbench application boundary."""

    session_identity_changed = Signal(str, str)  # project UID, display name
    layout_workspace_changed = Signal(object)
    ocr_workspace_changed = Signal(object)
    proof_workspace_changed = Signal(object)
    proof_workspace_patched = Signal(object)
    step_enabled_changed = Signal(int)
    step_requested = Signal(int)
    layout_finished = Signal()
    layout_progress = Signal(int, int)
    layout_stage = Signal(str, int, int, str)  # page UID, current, total, message
    ocr_finished = Signal()
    ocr_progress = Signal(object)  # WorkflowProgressState
    worker_error = Signal(str)
    layout_cancelled = Signal()
    ocr_cancelled = Signal()
    import_finished = Signal(object)  # ImportCompletionView
    status_message = Signal(str)
    current_step_changed = Signal(int)
    current_page_uid_changed = Signal(str)
    layout_run_enabled_changed = Signal(bool)
    view_state_changed = Signal(object)  # WorkflowViewState
    progress_state_changed = Signal(object)  # WorkflowProgressState
    focus_page = Signal(str)
    page_gate_state = Signal(str, str, bool, str, str)
    primary_action = Signal(str, str, str, bool)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        application: WorkbenchApplication | None = None,
        import_service: ImportService | None = None,
        layout_analysis_service: LayoutAnalysisService | None = None,
        ocr_job_service: OcrJobService | None = None,
        project_file_service: ProjectFileService | None = None,
    ) -> None:
        super().__init__(parent)
        if application is not None and not isinstance(application, WorkbenchApplication):
            raise TypeError("application must be WorkbenchApplication or None")
        self._session_dir = Path(tempfile.mkdtemp(prefix="ocr-process-session-"))
        self._import_service = import_service or ImportService(
            cache_dir=self._session_dir / "imports"
        )
        self._layout_analysis_service = layout_analysis_service
        self._ocr_job_service = ocr_job_service
        self._project_file_service = project_file_service or ProjectFileService()
        self._application = application or WorkbenchApplication(
            import_service=self._import_service,
            project_file_service=self._project_file_service,
            layout_service=layout_analysis_service,
            ocr_service=ocr_job_service,
        )
        self._max_step = self._application.max_step()
        self._current_step = STEP_IMPORT
        self._current_page_uid = ""
        self._layout_run_enabled = bool(self._application.pages())
        self._import_worker: _ImportServiceWorker | None = None
        self._layout_worker: _LayoutServiceWorker | None = None
        self._ocr_worker: _OcrServiceWorker | None = None
        self._closed = False

        if self._application.pages():
            self._current_page_uid = self._application.pages()[0].uid

    # ------------------------------------------------------------------ session state

    @property
    def has_project(self) -> bool:
        return self._application.has_project

    @property
    def project_uid(self) -> str:
        return self._application.project_uid

    @property
    def project_name(self) -> str:
        return self._application.project_name

    @property
    def pages(self) -> tuple[PageView, ...]:
        return self._application.pages()

    @property
    def has_pages(self) -> bool:
        return bool(self.pages)

    @property
    def total_line_count(self) -> int:
        return self._application.active_line_count()

    @property
    def is_fully_analyzed(self) -> bool:
        pages = self.pages
        return bool(pages) and all(self._has_layout(page.uid) for page in pages)

    @property
    def has_any_ocr_result(self) -> bool:
        return any(self._has_ocr(page.uid) for page in self.pages)

    @property
    def all_pages_ocr_done(self) -> bool:
        pages = self.pages
        return bool(pages) and all(self._has_ocr(page.uid) for page in pages)

    @property
    def is_dirty(self) -> bool:
        return self._application.is_dirty

    @property
    def is_bound_project(self) -> bool:
        return self._application.is_bound_project

    @property
    def import_dir(self) -> Path:
        path = self._session_dir / "imports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_step(self) -> int:
        return self._max_step

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current_page_uid(self) -> str:
        return self._current_page_uid

    @property
    def layout_run_enabled(self) -> bool:
        return self._layout_run_enabled

    def ensure_session(self, name: str = "未命名项目") -> None:
        if not self._application.has_project:
            self._application.ensure_project(f"project_{new_ulid()}", name)
            self._emit_session_state()

    def workflow_view_state(self) -> WorkflowViewState:
        return WorkflowViewState(
            max_step=self._max_step,
            current_step=self._current_step,
            current_page_uid=self._current_page_uid,
            layout_run_enabled=self._layout_run_enabled,
            has_project=self._application.has_project,
            total_pages=len(self.pages),
            total_lines=self.total_line_count,
        )

    def publish_state(self) -> None:
        """Publish the current immutable identity, records, and view state."""
        self._emit_session_state()

    def _emit_session_state(self) -> None:
        self._max_step = self._application.max_step()
        self.step_enabled_changed.emit(self._max_step)
        self.session_identity_changed.emit(self.project_uid, self.project_name)
        if self._application.has_project:
            self.ocr_workspace_changed.emit(self._application.ocr_workspace())
            self.layout_workspace_changed.emit(self._application.layout_workspace())
            if self.has_any_ocr_result:
                self.proof_workspace_changed.emit(self._application.proof_workspace())
            else:
                self.proof_workspace_changed.emit(None)
        else:
            self.layout_workspace_changed.emit(None)
            self.ocr_workspace_changed.emit(None)
            self.proof_workspace_changed.emit(None)
        self.view_state_changed.emit(self.workflow_view_state())

    def _emit_view_state(self) -> None:
        self._max_step = self._application.max_step()
        self.step_enabled_changed.emit(self._max_step)
        self.view_state_changed.emit(self.workflow_view_state())

    def apply_layout_edit(self, command: LayoutEditCommand) -> LayoutEditResult:
        """Commit one layout intent and publish the resulting immutable view."""

        result = self._application.apply_layout_edit(command)
        self.refresh_page_gate_states()
        self._emit_session_state()
        return result

    def apply_proof_edit(
        self,
        command: ProofEditCommand | ProofBatchEditCommand,
    ) -> ProofEditResult | ProofBatchEditResult:
        """Commit one proof intent and publish the resulting immutable view."""

        result = self._application.apply_proof_edit(command)
        results = result.results if isinstance(result, ProofBatchEditResult) else (result,)
        changes: dict[str, set[str]] = {}
        for item in results:
            if item.changed:
                changes.setdefault(item.proof_uid, set()).update(item.changed_text_unit_uids)
        if changes:
            self.proof_workspace_patched.emit(self._application.proof_workspace_patch(changes))
        self._emit_view_state()
        return result

    # ------------------------------------------------------------------ navigation

    def get_open_step(self) -> int:
        if self._max_step >= STEP_HPROOF:
            return STEP_HPROOF
        if self._max_step >= STEP_OCR:
            return STEP_OCR
        if self._max_step >= STEP_LAYOUT:
            return STEP_LAYOUT
        return STEP_IMPORT

    def can_enter_step(self, step: int) -> bool:
        if step == STEP_IMPORT:
            return True
        if not self._application.has_project or not self.has_pages:
            return False
        if step == STEP_LAYOUT:
            return True
        if step == STEP_OCR:
            return any(self._has_layout(page.uid) for page in self.pages)
        if step in (STEP_HPROOF, STEP_VPROOF):
            return self.has_any_ocr_result
        return False

    def request_step(self, step: int) -> bool:
        if not self.can_enter_step(step):
            self.status_message.emit("当前项目尚未满足进入该步骤的条件")
            return False
        self.set_current_step(step)
        self.step_requested.emit(step)
        return True

    def set_current_step(self, step: int) -> None:
        if step not in {STEP_IMPORT, STEP_LAYOUT, STEP_OCR, STEP_HPROOF, STEP_VPROOF}:
            raise ValueError(f"unknown workflow step: {step}")
        if not self.can_enter_step(step):
            raise ValueError(f"workflow step is not available: {step}")
        if self._current_step == step:
            return
        self._current_step = step
        self.current_step_changed.emit(step)
        self._emit_view_state()

    def set_current_page_uid(self, page_uid: str) -> None:
        if not self._application.has_project:
            if page_uid:
                raise RuntimeError("cannot select a page without an active project")
            return
        self._application.require_page(page_uid)
        if self._current_page_uid == page_uid:
            return
        self._current_page_uid = page_uid
        self.current_page_uid_changed.emit(page_uid)
        self.focus_page.emit(page_uid)
        self._emit_view_state()

    def page_uid_at(self, index: int) -> str | None:
        pages = self.pages
        if index < 0 or index >= len(pages):
            return None
        return pages[index].uid

    def refresh_page_gate_states(self) -> None:
        if not self._application.has_project:
            return
        for page in self.pages:
            gate = self._application.page_gate(page.uid)
            self.page_gate_state.emit(
                page.uid,
                gate.page_state,
                gate.is_pending,
                gate.reason_code,
                gate.reason_text,
            )
            self.primary_action.emit(
                page.uid,
                gate.action_key,
                gate.action_label,
                gate.action_enabled,
            )

    # ------------------------------------------------------------------ import

    def start_import(self, paths: Iterable[str | Path], *, name: str = "未命名项目") -> bool:
        if self.has_running_workers():
            raise RuntimeError("cannot import while another application task is running")
        if self._application.has_project and self.has_pages:
            raise RuntimeError("import requires an empty project")
        self.ensure_session(name)
        normalized = tuple(str(path) for path in paths)
        if not normalized:
            raise ValueError("import requires at least one source path")
        request = self._application.prepare_import(normalized)
        worker = _ImportServiceWorker(self._import_service, request, self)
        self._import_worker = worker
        worker.completed.connect(self._on_import_completed)
        worker.failed.connect(self._on_import_failed)
        worker.finished.connect(lambda: self._clear_worker("_import_worker", worker))
        worker.start()
        return True

    def import_paths(
        self,
        paths: Iterable[str | Path],
        *,
        name: str = "未命名项目",
    ) -> ImportCompletionView:
        """Synchronous service entry point used by focused tests and scripts."""
        if self.has_running_workers():
            raise RuntimeError("cannot import while another application task is running")
        if self._application.has_project and self.has_pages:
            raise RuntimeError("import requires an empty project")
        self.ensure_session(name)
        normalized = tuple(str(path) for path in paths)
        if not normalized:
            raise ValueError("import requires at least one source path")
        result = self._application.execute_import(self._application.prepare_import(normalized))
        self._adopt_import_result(self.project_uid, result)
        return self._import_completion(result)

    def _on_import_completed(self, project_uid: str, result: ImportResult) -> None:
        try:
            self._adopt_import_result(project_uid, result)
        except Exception as exc:
            self.worker_error.emit(str(exc))

    def _adopt_import_result(self, project_uid: str, result: ImportResult) -> None:
        if not self._application.has_project or self.project_uid != project_uid:
            raise RuntimeError("import result belongs to an inactive project")
        if not isinstance(result, ImportResult):
            raise TypeError("ImportService must return ImportResult")
        self._application.commit_import(result)
        if result.pages:
            if not self._current_page_uid:
                self._current_page_uid = result.pages[0].uid
            self._set_layout_run_enabled(True)
        self._emit_session_state()
        self.import_finished.emit(self._import_completion(result))

    def _on_import_failed(self, message: str) -> None:
        self.worker_error.emit(message)

    # ------------------------------------------------------------------ layout

    def start_layout_analysis(self, page_uids: Iterable[str] | None = None) -> bool:
        if not self._application.has_project:
            raise RuntimeError("layout analysis requires an active project")
        if self.has_running_workers():
            return False
        selected = tuple(page_uids) if page_uids is not None else tuple(
            page.uid for page in self.pages
        )
        if not selected:
            raise ValueError("layout analysis requires at least one page UID")
        service = self._layout_analysis_service or self._build_default_layout_analysis_service()
        self._layout_analysis_service = service
        self._application.configure_layout_service(service)
        requests = tuple(
            self._application.prepare_layout_page(page_uid)
            for page_uid in selected
        )
        worker = _LayoutServiceWorker(
            service,
            requests,
            max_workers=_configured_page_workers(
                "layout_concurrency",
                default=8,
                cap=10,
                total=len(requests),
            ),
            parent=self,
        )
        self._layout_worker = worker
        self._set_layout_run_enabled(False)
        worker.committed.connect(self._on_layout_committed)
        worker.progress.connect(self._on_layout_progress)
        worker.stage.connect(self._on_layout_stage)
        worker.cancelled.connect(self._on_layout_cancelled)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(lambda: self._clear_worker("_layout_worker", worker))
        self._emit_view_state()
        worker.start()
        return True

    def _build_default_layout_analysis_service(self) -> LayoutAnalysisService:
        """Compose the layout use case from the current application settings."""
        from app.core.api_profiles import FIXED_LAYOUT_PROFILE, resolve_api_endpoint_for_role
        from app.core.paddle_v16_client import is_paddle_v16_endpoint
        from app.integrations.paddle import PaddleVLClient

        config = get_config()
        jobs_url = resolve_api_endpoint_for_role(
            config.get("api_url", ""),
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        if not jobs_url:
            raise RuntimeError("请先在“更多 → 设置”中配置 Paddle API 地址")
        if not is_paddle_v16_endpoint(jobs_url):
            raise RuntimeError("版面分析需要 PaddleOCR-VL-1.6 jobs API 地址")
        configured_timeout = max(1, int(config.get("api_timeout", 180)))
        client = PaddleVLClient(
            jobs_url=jobs_url,
            token=str(config.get("api_token", "") or ""),
            request_timeout=min(max(10, configured_timeout), 30),
            poll_timeout=max(configured_timeout, 180),
            network_mode=str(config.get("paddle_api_network_mode", "auto") or "auto"),
        )
        return LayoutAnalysisService(client)

    def _on_layout_progress(self, current: int, total: int) -> None:
        self.layout_progress.emit(current, total)
        self.progress_state_changed.emit(WorkflowProgressState(
            phase="layout",
            current=current,
            total=total,
            completed_pages=current,
            total_pages=total,
            message="版面分析",
        ))

    def _on_layout_stage(self, page_uid: str, current: int, total: int, message: str) -> None:
        self.layout_stage.emit(page_uid, current, total, message)

    def _on_layout_committed(self, results: object) -> None:
        if not isinstance(results, tuple) or any(
            not isinstance(item, LayoutPageJobResult) for item in results
        ):
            self._on_worker_failed("LayoutAnalysisService returned invalid job results")
            return
        if not self._application.has_project or self._layout_analysis_service is None:
            self._on_worker_failed("layout results have no active application context")
            return
        commits: list[LayoutAnalysisCommit] = []
        for result in results:
            try:
                commits.append(self._application.commit_layout_page(result))
            except Exception as exc:
                self.worker_error.emit(f"page {result.request.page.uid}: {exc}")
        if not commits:
            self._set_layout_run_enabled(True)
            self._emit_view_state()
            return
        self._set_layout_run_enabled(True)
        self.layout_finished.emit()
        self.refresh_page_gate_states()
        self._emit_session_state()

    def _on_layout_cancelled(self) -> None:
        self._set_layout_run_enabled(True)
        self.layout_cancelled.emit()
        self._emit_view_state()

    # ------------------------------------------------------------------ OCR

    def pending_ocr_page_uids(self, preferred_page_uid: str = "") -> tuple[str, ...]:
        pending = self._application.pending_ocr_page_uids()
        if not preferred_page_uid or preferred_page_uid not in pending:
            return pending
        return (
            preferred_page_uid,
            *(page_uid for page_uid in pending if page_uid != preferred_page_uid),
        )

    def start_ocr(self, page_uids: Iterable[str] | None = None) -> bool:
        if not self._application.has_project:
            raise RuntimeError("OCR requires an active project")
        if self.has_running_workers():
            return False
        selected = tuple(page_uids) if page_uids is not None else self.pending_ocr_page_uids()
        if not selected:
            raise ValueError("OCR has no pending page UIDs")
        for page_uid in selected:
            self._application.require_page(page_uid)
            if not self._has_layout(page_uid):
                raise ValueError(f"OCR requires an adopted layout for page {page_uid!r}")
        service = self._ocr_job_service or self._build_default_ocr_job_service()
        self._ocr_job_service = service
        self._application.configure_ocr_service(service)
        requests: list[OcrPageJobRequest] = []
        for page_uid in selected:
            try:
                requests.append(self._application.prepare_ocr_page(page_uid))
            except Exception as exc:
                self.worker_error.emit(f"page {page_uid}: {exc}")
        if not requests:
            raise RuntimeError("OCR has no page that can be prepared")
        worker = _OcrServiceWorker(
            service,
            tuple(requests),
            max_workers=_configured_page_workers(
                "ocr_page_concurrency",
                default=2,
                cap=20,
                total=len(requests),
            ),
            parent=self,
        )
        self._ocr_worker = worker
        worker.progress.connect(self._on_ocr_progress)
        worker.committed.connect(self._on_ocr_committed)
        worker.cancelled.connect(self._on_ocr_cancelled)
        worker.failed.connect(self._on_worker_failed)
        worker.page_failed.connect(self._on_ocr_page_failed)
        worker.finished.connect(lambda: self._clear_worker("_ocr_worker", worker))
        self._emit_view_state()
        worker.start()
        return True

    def _on_ocr_progress(self, progress: WorkflowProgressState) -> None:
        if not isinstance(progress, WorkflowProgressState):
            self._on_worker_failed("OcrJobService emitted an invalid progress record")
            return
        self.ocr_progress.emit(progress)
        self.progress_state_changed.emit(progress)

    def _on_ocr_committed(self, results: object) -> None:
        if not isinstance(results, tuple) or any(
            not isinstance(item, OcrPageJobResult) for item in results
        ):
            self._on_worker_failed("OcrJobService returned invalid job results")
            return
        if not self._application.has_project or self._ocr_job_service is None:
            self._on_worker_failed("OCR results have no active application context")
            return
        commits: list[OcrPageCommit] = []
        for result in results:
            try:
                commits.append(self._application.commit_ocr_page(result))
            except Exception as exc:
                self.worker_error.emit(f"page {result.request.page.uid}: {exc}")
        if not commits:
            self._emit_view_state()
            return
        self.ocr_finished.emit()
        self.refresh_page_gate_states()
        self._emit_session_state()

    def _on_ocr_page_failed(self, failure: object) -> None:
        if not isinstance(failure, OcrPageJobFailure):
            self._on_worker_failed("OcrJobService emitted an invalid page failure")
            return
        page_uid = failure.request.page.uid
        if not self._application.has_project or self._ocr_job_service is None:
            self._on_worker_failed("OCR failure has no active application context")
            return
        try:
            self._application.commit_ocr_failure(failure)
        except Exception as exc:
            self.worker_error.emit(f"page {page_uid}: {exc}")
            return
        self.worker_error.emit(f"page {page_uid}: {failure.message}")
        self.refresh_page_gate_states()
        self._emit_session_state()

    def _on_ocr_cancelled(self) -> None:
        self.ocr_cancelled.emit()
        self._emit_view_state()

    def _build_default_ocr_job_service(self) -> OcrJobService:
        config = get_config()
        mode = str(config.get("mode", "") or "").strip().lower()
        if mode != "hanwang":
            raise RuntimeError(
                "the configured OCR mode has no OcrJobService adapter; "
                "select hanwang explicitly"
            )
        from app.core.api_profiles import FIXED_LAYOUT_PROFILE, resolve_api_endpoint_for_role
        from app.core.paddle_v16_client import PaddleV16LayoutClient, is_paddle_v16_endpoint
        from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6PrepassClient
        from app.engines.hanwang.micro_recblock import HanwangMicroRecBlockEngine

        jobs_url = resolve_api_endpoint_for_role(
            config.get("api_url", ""),
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        if not jobs_url or not is_paddle_v16_endpoint(jobs_url):
            raise RuntimeError("hanwang OcrJobService requires a Paddle jobs endpoint")
        configured_timeout = max(1, int(config.get("api_timeout", 180)))
        transport = PaddleV16LayoutClient(
            jobs_url=jobs_url,
            token=str(config.get("api_token", "") or ""),
            request_timeout=min(max(10, configured_timeout), 30),
            poll_timeout=max(configured_timeout, 180),
            network_mode=str(config.get("paddle_api_network_mode", "auto") or "auto"),
        )
        return OcrJobService(
            prepass_client=PpOcrV6PrepassClient(transport),
            vl_client=transport,
            engine=HanwangMicroRecBlockEngine(),
        )

    @staticmethod
    def _import_completion(result: ImportResult) -> ImportCompletionView:
        return ImportCompletionView(
            page_uids=tuple(page.uid for page in result.pages),
            failures=tuple(
                ImportFailureView(source_path=item.source_path, error=item.error)
                for item in result.failures
            ),
        )

    # ------------------------------------------------------------------ files and export

    def open_project(self, file_path: str | Path) -> bool:
        if self.has_running_workers():
            raise RuntimeError("cannot open a project while an application task is running")
        self._application.open_project(file_path)
        self._current_step = STEP_IMPORT
        self._current_page_uid = self.pages[0].uid if self.pages else ""
        self._set_layout_run_enabled(bool(self.pages))
        self._emit_session_state()
        self.refresh_page_gate_states()
        return True

    def save_project_as(self, file_path: str | Path) -> bool:
        if not self._application.has_project:
            raise RuntimeError("cannot save without an active project")
        self._application.save_as(file_path)
        self._emit_session_state()
        return True

    def save_project(self) -> bool:
        if not self._application.has_project:
            raise RuntimeError("cannot save without an active project")
        if not self._application.is_bound_project:
            raise RuntimeError("active project has no save path")
        self._application.save()
        self._emit_session_state()
        return True

    def capture_export_snapshot(self) -> ExportProjectSnapshot:
        if not self._application.has_project:
            raise RuntimeError("cannot export without an active project")
        return self._application.export_snapshot()

    def close_project(self) -> bool:
        if self.has_running_workers() and not self.cancel_running_workers():
            return False
        self._application.close_project()
        self._current_step = STEP_IMPORT
        self._current_page_uid = ""
        self._set_layout_run_enabled(False)
        self._emit_session_state()
        return True

    # ------------------------------------------------------------------ lifecycle and diagnostics

    def has_running_workers(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self._import_worker, self._layout_worker, self._ocr_worker)
        )

    def cancel_layout_analysis(self, *, wait_ms: int = 1200) -> bool:
        worker = self._layout_worker
        if worker is None or not worker.isRunning():
            return False
        worker.requestInterruption()
        stopped = worker.wait(max(0, int(wait_ms)))
        return bool(stopped)

    def cancel_running_workers(self, *, wait_ms: int = 1200, force: bool = False) -> bool:
        del force  # hard termination is intentionally not part of the service boundary
        stopped = True
        for worker in (self._import_worker, self._layout_worker, self._ocr_worker):
            if worker is None or not worker.isRunning():
                continue
            worker.requestInterruption()
            stopped = bool(worker.wait(max(0, int(wait_ms)))) and stopped
        return stopped

    def ocr_engine_description(self) -> str:
        mode = str(get_config().get("mode", "") or "").strip() or "未配置"
        return f"OCR service：{mode}"

    def _has_layout(self, page_uid: str) -> bool:
        return self._application.has_project and self._application.has_layout(page_uid)

    def _has_ocr(self, page_uid: str) -> bool:
        return self._application.has_project and self._application.has_ocr(page_uid)

    def _layout_revision(self, page_uid: str) -> int:
        if not self._application.has_project:
            raise RuntimeError("no active project")
        return self._application.layout_revision(page_uid)

    def _clear_worker(self, attr_name: str, worker: QThread) -> None:
        if getattr(self, attr_name, None) is worker:
            setattr(self, attr_name, None)

    def _set_layout_run_enabled(self, enabled: bool) -> None:
        value = bool(enabled)
        if self._layout_run_enabled == value:
            return
        self._layout_run_enabled = value
        self.layout_run_enabled_changed.emit(value)

    def _on_worker_failed(self, message: str) -> None:
        self._set_layout_run_enabled(bool(self.pages))
        self.worker_error.emit(message)
        self._emit_view_state()

    def close(self) -> None:
        if self._closed:
            return
        if not self.cancel_running_workers():
            raise RuntimeError("application tasks did not stop before controller close")
        self._closed = True
        self._session_dir = Path(tempfile.mkdtemp(prefix="ocr-process-closed-"))


__all__ = [
    "STEP_HPROOF",
    "STEP_IMPORT",
    "STEP_LAYOUT",
    "STEP_OCR",
    "STEP_VPROOF",
    "WorkflowController",
]
