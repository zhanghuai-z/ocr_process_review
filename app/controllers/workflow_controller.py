"""Session-scoped application orchestration for the OCR workbench."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import tempfile

from PySide6.QtCore import QObject, QThread, Signal

from app.core.app_config import get_config
from app.core.workflow_state import (
    STEP_HPROOF,
    STEP_IMPORT,
    STEP_LAYOUT,
    STEP_OCR,
    STEP_VPROOF,
    WorkflowProgressState,
    WorkflowViewState,
    active_line_count,
    compute_max_step,
    page_gate_info,
    pending_ocr_page_uids,
)
from app.models.entity_id import new_ulid
from app.models.export_snapshot import ExportProjectSnapshot
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession, RecordNotFoundError
from app.services import (
    ImportResult,
    ImportService,
    LayoutAnalysisCommit,
    LayoutAnalysisService,
    OcrJobService,
    OcrPageCommit,
    ProjectFileService,
    capture_export_snapshot,
)


class _TaskCancelled(RuntimeError):
    """Internal worker control flow for cooperative cancellation."""


class _ImportServiceWorker(QThread):
    """Run exactly one ImportService call outside the UI thread."""

    completed = Signal(str, object)
    failed = Signal(str)

    def __init__(
        self,
        service: ImportService,
        session: ProjectSession,
        paths: tuple[str, ...],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._session = session
        self._paths = paths

    def run(self) -> None:
        try:
            result = self._service.import_paths(self._session, self._paths)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(self._session.project_uid, result)


class _LayoutServiceWorker(QThread):
    """Run LayoutAnalysisService and emit only immutable layout commits."""

    committed = Signal(object)
    progress = Signal(int, int)
    stage = Signal(str, int, int, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        service: LayoutAnalysisService,
        session: ProjectSession,
        page_uids: tuple[str, ...],
        expected_revisions: Mapping[str, int],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._session = session
        self._page_uids = page_uids
        self._expected_revisions = dict(expected_revisions)

    def run(self) -> None:
        try:
            def on_progress(current: int, total: int, page_uid: str) -> None:
                if self.isInterruptionRequested():
                    raise _TaskCancelled
                self.progress.emit(current, total)
                self.stage.emit(page_uid, current, total, "版面分析")

            commits = self._service.analyze_pages(
                self._session,
                self._page_uids,
                expected_revisions=self._expected_revisions,
                progress_callback=on_progress,
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
        except _TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.committed.emit(tuple(commits))


class _OcrServiceWorker(QThread):
    """Load page pixels and invoke OcrJobService for each stable page UID."""

    committed = Signal(object)
    progress = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        service: OcrJobService,
        session: ProjectSession,
        page_uids: tuple[str, ...],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._session = session
        self._page_uids = page_uids

    def run(self) -> None:
        commits: list[OcrPageCommit] = []
        total = len(self._page_uids)
        try:
            for index, page_uid in enumerate(self._page_uids):
                if self.isInterruptionRequested():
                    raise _TaskCancelled
                self.progress.emit(WorkflowProgressState(
                    phase="ocr",
                    current=0,
                    total=0,
                    completed_pages=index,
                    total_pages=total,
                    message="准备识别",
                    page_uid=page_uid,
                ))
                page = self._session.page_repository.get(page_uid)
                image = _read_page_image(page)

                def on_progress(current: int, block_total: int, message: str) -> None:
                    if self.isInterruptionRequested():
                        raise _TaskCancelled
                    self.progress.emit(WorkflowProgressState(
                        phase="ocr",
                        current=current,
                        total=block_total,
                        completed_pages=index,
                        total_pages=total,
                        message=message,
                        page_uid=page_uid,
                    ))

                commit = self._service.run_page(
                    self._session,
                    page_uid,
                    image,
                    progress_callback=on_progress,
                )
                if not isinstance(commit, OcrPageCommit):
                    raise TypeError("OcrJobService must return OcrPageCommit")
                commits.append(commit)
                self.progress.emit(WorkflowProgressState(
                    phase="ocr",
                    current=1,
                    total=1,
                    completed_pages=index + 1,
                    total_pages=total,
                    message="已完成",
                    page_uid=page_uid,
                ))
        except _TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.committed.emit(tuple(commits))


def _read_page_image(page: PageRecord):
    """Read one immutable page record into the service's pixel contract."""
    import cv2

    path = Path(page.cache_image_path or page.image_path)
    if not path.is_file():
        raise FileNotFoundError(f"page image is missing: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"page image cannot be decoded: {path}")
    return image


class WorkflowController(QObject):
    """Own the active ProjectSession and coordinate application services."""

    session_identity_changed = Signal(str, str)  # project UID, display name
    page_records_changed = Signal(object)  # tuple[PageRecord, ...]
    step_enabled_changed = Signal(int)
    step_requested = Signal(int)
    layout_finished = Signal(object)  # tuple[LayoutAnalysisCommit, ...]
    layout_progress = Signal(int, int)
    layout_stage = Signal(str, int, int, str)  # page UID, current, total, message
    ocr_finished = Signal(object)  # tuple[OcrPageCommit, ...]
    ocr_progress = Signal(object)  # WorkflowProgressState
    worker_error = Signal(str)
    layout_cancelled = Signal()
    ocr_cancelled = Signal()
    import_finished = Signal(object)  # ImportResult
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
        session: ProjectSession | None = None,
        import_service: ImportService | None = None,
        layout_analysis_service: LayoutAnalysisService | None = None,
        ocr_job_service: OcrJobService | None = None,
        project_file_service: ProjectFileService | None = None,
    ) -> None:
        super().__init__(parent)
        if session is not None and not isinstance(session, ProjectSession):
            raise TypeError("WorkflowController session must be ProjectSession")
        self._session: ProjectSession | None = session
        self._session_dir = Path(tempfile.mkdtemp(prefix="ocr-process-session-"))
        self._import_service = import_service or ImportService(
            cache_dir=self._session_dir / "imports"
        )
        self._layout_analysis_service = layout_analysis_service or LayoutAnalysisService()
        self._ocr_job_service = ocr_job_service
        self._project_file_service = project_file_service or ProjectFileService()
        self._dirty = False
        self._max_step = compute_max_step(session)
        self._current_step = STEP_IMPORT
        self._current_page_uid = ""
        self._layout_run_enabled = bool(session and session.page_repository.all())
        self._import_worker: _ImportServiceWorker | None = None
        self._layout_worker: _LayoutServiceWorker | None = None
        self._ocr_worker: _OcrServiceWorker | None = None
        self._closed = False

        if session is not None and session.page_repository.all():
            self._current_page_uid = session.page_repository.all()[0].uid

    # ------------------------------------------------------------------ session state

    @property
    def session(self) -> ProjectSession | None:
        return self._session

    @property
    def project_uid(self) -> str:
        return self._session.project_uid if self._session is not None else ""

    @property
    def project_name(self) -> str:
        return self._session.project_record.name if self._session is not None else ""

    @property
    def page_records(self) -> tuple[PageRecord, ...]:
        return self._session.page_repository.all() if self._session is not None else ()

    @property
    def has_pages(self) -> bool:
        return bool(self.page_records)

    @property
    def total_line_count(self) -> int:
        return active_line_count(self._session) if self._session is not None else 0

    @property
    def is_fully_analyzed(self) -> bool:
        pages = self.page_records
        return bool(pages) and all(self._has_layout(page.uid) for page in pages)

    @property
    def has_any_ocr_result(self) -> bool:
        return any(self._has_ocr(page.uid) for page in self.page_records)

    @property
    def all_pages_ocr_done(self) -> bool:
        pages = self.page_records
        return bool(pages) and all(self._has_ocr(page.uid) for page in pages)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def is_bound_project(self) -> bool:
        return bool(self._session and self._session.save_path)

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

    def ensure_session(self, name: str = "未命名项目") -> ProjectSession:
        if self._session is None:
            self._session = ProjectSession(
                ProjectRecord(f"project_{new_ulid()}", name=name)
            )
            self._dirty = False
            self._emit_session_state()
        return self._session

    def workflow_view_state(self) -> WorkflowViewState:
        return WorkflowViewState(
            max_step=self._max_step,
            current_step=self._current_step,
            current_page_uid=self._current_page_uid,
            layout_run_enabled=self._layout_run_enabled,
            has_project=self._session is not None,
            total_pages=len(self.page_records),
            total_lines=self.total_line_count,
        )

    def publish_state(self) -> None:
        """Publish the current immutable identity, records, and view state."""
        self._emit_session_state()

    def _emit_session_state(self) -> None:
        self._max_step = compute_max_step(self._session)
        self.step_enabled_changed.emit(self._max_step)
        self.session_identity_changed.emit(self.project_uid, self.project_name)
        self.page_records_changed.emit(self.page_records)
        self.view_state_changed.emit(self.workflow_view_state())

    def _emit_view_state(self) -> None:
        self._max_step = compute_max_step(self._session)
        self.step_enabled_changed.emit(self._max_step)
        self.view_state_changed.emit(self.workflow_view_state())

    def mark_dirty(self) -> None:
        if self._session is None:
            raise RuntimeError("cannot mark a missing ProjectSession dirty")
        self._dirty = True
        self._emit_view_state()

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
        if self._session is None or not self.has_pages:
            return False
        if step == STEP_LAYOUT:
            return True
        if step == STEP_OCR:
            return any(self._has_layout(page.uid) for page in self.page_records)
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
        if self._session is None:
            if page_uid:
                raise RuntimeError("cannot select a page without a ProjectSession")
            return
        self._session.page_repository.get(page_uid)
        if self._current_page_uid == page_uid:
            return
        self._current_page_uid = page_uid
        self.current_page_uid_changed.emit(page_uid)
        self.focus_page.emit(page_uid)
        self._emit_view_state()

    def page_record(self, page_uid: str) -> PageRecord:
        if self._session is None:
            raise RuntimeError("no active ProjectSession")
        return self._session.page_repository.get(page_uid)

    def page_uid_at(self, index: int) -> str | None:
        records = self.page_records
        if index < 0 or index >= len(records):
            return None
        return records[index].uid

    def refresh_page_gate_states(self) -> None:
        if self._session is None:
            return
        for page in self.page_records:
            gate = page_gate_info(self._session, page.uid)
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
        if self._session is not None and self.has_pages:
            raise RuntimeError("import requires an empty ProjectSession")
        session = self.ensure_session(name)
        normalized = tuple(str(path) for path in paths)
        if not normalized:
            raise ValueError("import requires at least one source path")
        worker = _ImportServiceWorker(self._import_service, session, normalized, self)
        self._import_worker = worker
        worker.completed.connect(self._on_import_completed)
        worker.failed.connect(self._on_import_failed)
        worker.finished.connect(lambda: self._clear_worker("_import_worker", worker))
        worker.start()
        return True

    def import_paths(self, paths: Iterable[str | Path], *, name: str = "未命名项目") -> ImportResult:
        """Synchronous service entry point used by focused tests and scripts."""
        if self.has_running_workers():
            raise RuntimeError("cannot import while another application task is running")
        if self._session is not None and self.has_pages:
            raise RuntimeError("import requires an empty ProjectSession")
        session = self.ensure_session(name)
        normalized = tuple(str(path) for path in paths)
        if not normalized:
            raise ValueError("import requires at least one source path")
        result = self._import_service.import_paths(session, normalized)
        self._adopt_import_result(session.project_uid, result)
        return result

    def _on_import_completed(self, project_uid: str, result: ImportResult) -> None:
        try:
            self._adopt_import_result(project_uid, result)
        except Exception as exc:
            self.worker_error.emit(str(exc))

    def _adopt_import_result(self, project_uid: str, result: ImportResult) -> None:
        if self._session is None or self._session.project_uid != project_uid:
            raise RuntimeError("import result belongs to an inactive ProjectSession")
        if not isinstance(result, ImportResult):
            raise TypeError("ImportService must return ImportResult")
        if result.pages:
            self._dirty = True
            if not self._current_page_uid:
                self._current_page_uid = result.pages[0].uid
                self.current_page_uid_changed.emit(self._current_page_uid)
            self._set_layout_run_enabled(True)
        self.import_finished.emit(result)
        self._emit_session_state()

    def _on_import_failed(self, message: str) -> None:
        self.worker_error.emit(message)

    # ------------------------------------------------------------------ layout

    def start_layout_analysis(self, page_uids: Iterable[str] | None = None) -> bool:
        if self._session is None:
            raise RuntimeError("layout analysis requires a ProjectSession")
        if self.has_running_workers():
            return False
        selected = tuple(page_uids) if page_uids is not None else tuple(
            page.uid for page in self.page_records
        )
        if not selected:
            raise ValueError("layout analysis requires at least one page UID")
        expected = {
            page_uid: self._layout_revision(page_uid)
            for page_uid in selected
        }
        worker = _LayoutServiceWorker(
            self._layout_analysis_service,
            self._session,
            selected,
            expected,
            self,
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

    def _on_layout_committed(self, commits: object) -> None:
        if not isinstance(commits, tuple) or any(
            not isinstance(item, LayoutAnalysisCommit) for item in commits
        ):
            self._on_worker_failed("LayoutAnalysisService returned invalid commits")
            return
        self._dirty = True
        self._set_layout_run_enabled(True)
        self.layout_finished.emit(commits)
        self.refresh_page_gate_states()
        self._emit_view_state()

    def _on_layout_cancelled(self) -> None:
        self._set_layout_run_enabled(True)
        self.layout_cancelled.emit()
        self._emit_view_state()

    # ------------------------------------------------------------------ OCR

    def start_ocr(self, page_uids: Iterable[str] | None = None) -> bool:
        if self._session is None:
            raise RuntimeError("OCR requires a ProjectSession")
        if self.has_running_workers():
            return False
        selected = tuple(page_uids) if page_uids is not None else pending_ocr_page_uids(self._session)
        if not selected:
            raise ValueError("OCR has no pending page UIDs")
        for page_uid in selected:
            self._session.page_repository.get(page_uid)
            if not self._has_layout(page_uid):
                raise ValueError(f"OCR requires an adopted layout for page {page_uid!r}")
        service = self._ocr_job_service or self._build_default_ocr_job_service()
        worker = _OcrServiceWorker(service, self._session, selected, self)
        self._ocr_worker = worker
        worker.progress.connect(self._on_ocr_progress)
        worker.committed.connect(self._on_ocr_committed)
        worker.cancelled.connect(self._on_ocr_cancelled)
        worker.failed.connect(self._on_worker_failed)
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

    def _on_ocr_committed(self, commits: object) -> None:
        if not isinstance(commits, tuple) or any(
            not isinstance(item, OcrPageCommit) for item in commits
        ):
            self._on_worker_failed("OcrJobService returned invalid commits")
            return
        self._dirty = True
        self.ocr_finished.emit(commits)
        self.refresh_page_gate_states()
        self._emit_view_state()

    def _on_ocr_cancelled(self) -> None:
        self.ocr_cancelled.emit()
        self._emit_view_state()

    def _build_default_ocr_job_service(self) -> OcrJobService:
        config = get_config()
        mode = str(config.get("mode", "") or "").strip().lower()
        if mode != "hanwang":
            raise RuntimeError(
                "the configured OCR mode has no ProjectSession OcrJobService adapter; "
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

    # ------------------------------------------------------------------ files and export

    def open_project(self, file_path: str | Path) -> bool:
        if self.has_running_workers():
            raise RuntimeError("cannot open a project while an application task is running")
        bound = self._project_file_service.open_project(file_path)
        if not isinstance(bound.session, ProjectSession):
            raise TypeError("project file service returned an invalid ProjectSession")
        self._session = bound.session
        self._dirty = False
        self._current_step = STEP_IMPORT
        self._current_page_uid = self.page_records[0].uid if self.page_records else ""
        self._set_layout_run_enabled(bool(self.page_records))
        self._emit_session_state()
        self.refresh_page_gate_states()
        return True

    def save_project_as(self, file_path: str | Path) -> bool:
        if self._session is None:
            raise RuntimeError("cannot save without a ProjectSession")
        bound = self._project_file_service.save_as(self._session, file_path)
        self._session = bound.session
        self._dirty = False
        self._emit_session_state()
        return True

    def save_project(self) -> bool:
        if self._session is None:
            raise RuntimeError("cannot save without a ProjectSession")
        if not self._session.save_path:
            raise RuntimeError("ProjectSession has no save path")
        bound = self._project_file_service.save_session(self._session)
        self._session = bound.session
        self._dirty = False
        self._emit_session_state()
        return True

    def capture_export_snapshot(self) -> ExportProjectSnapshot:
        if self._session is None:
            raise RuntimeError("cannot export without a ProjectSession")
        return capture_export_snapshot(self._session)

    def close_project(self) -> bool:
        if self.has_running_workers() and not self.cancel_running_workers():
            return False
        self._session = None
        self._dirty = False
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
        return f"ProjectSession OCR service：{mode}"

    def _has_layout(self, page_uid: str) -> bool:
        if self._session is None:
            return False
        try:
            self._session.layout_repository.get(page_uid)
        except RecordNotFoundError:
            return False
        return True

    def _has_ocr(self, page_uid: str) -> bool:
        if self._session is None:
            return False
        try:
            self._session.ocr_observation_repository.get_active_pointer(page_uid)
        except RecordNotFoundError:
            return False
        return True

    def _layout_revision(self, page_uid: str) -> int:
        if self._session is None:
            raise RuntimeError("no active ProjectSession")
        try:
            return self._session.layout_repository.get(page_uid).revision
        except RecordNotFoundError:
            return 0

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
        self._set_layout_run_enabled(bool(self.page_records))
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
