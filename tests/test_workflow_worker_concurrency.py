from __future__ import annotations

from threading import Lock
import time
from types import SimpleNamespace

from app.application import WorkbenchApplication
from app.controllers import workflow_controller as workflow
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.services.import_service import ImportService
from app.services.ocr_job_service import OcrPageJobRequest, OcrPageJobResult


class _OverlapProbe:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def test_configured_page_workers_consumes_and_bounds_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "get_config",
        lambda: {"layout_concurrency": 7, "ocr_page_concurrency": 30},
    )

    assert workflow._configured_page_workers(
        "layout_concurrency", default=8, cap=10, total=4
    ) == 4
    assert workflow._configured_page_workers(
        "ocr_page_concurrency", default=2, cap=20, total=30
    ) == 20
    assert workflow._configured_page_workers(
        "layout_concurrency", default=8, cap=10, total=1
    ) == 1


def test_controller_passes_configured_concurrency_to_production_workers(monkeypatch) -> None:
    session = ProjectSession(ProjectRecord("project-1", "Concurrency"))
    for index in range(1, 6):
        page = PageRecord(
            project_uid=session.project_uid,
            uid=f"page-{index}",
            image_path=f"page-{index}.png",
            source_path=f"page-{index}.png",
            cache_image_path="",
            thumbnail_path="",
            width=100,
            height=80,
            page_number=index,
            source_page_index=index - 1,
            status="imported",
            error="",
            image_hash=f"hash-{index}",
            image_revision=1,
        )
        session.page_repository.put(page, expected_revision=0)
        artifact = PaddleArtifact(
            project_uid=session.project_uid,
            uid=f"artifact-{index}",
            page_uid=page.uid,
            source_engine="test",
            source_run_id=f"layout-run-{index}",
            image_hash=page.image_hash,
            payload_json="{}",
        )
        session.paddle_artifact_repository.append(artifact)
        session.layout_repository.put(
            LayoutSnapshot(
                page_uid=page.uid,
                revision=1,
                artifact_uid=artifact.uid,
                source_engine="test",
                source_run_id=artifact.source_run_id,
                blocks=(),
            ),
            expected_revision=0,
        )

    class LayoutService:
        def prepare_page(self, _session, page_uid, *, expected_revision=0, **_kwargs):
            return SimpleNamespace(page_uid=page_uid, expected_revision=expected_revision)

    class OcrService:
        def prepare_page(self, current_session, page_uid):
            page = current_session.page_repository.get(page_uid)
            layout = current_session.layout_repository.get(page_uid)
            artifact = current_session.paddle_artifact_repository.get(layout.artifact_uid)
            return OcrPageJobRequest(
                project_uid=current_session.project_uid,
                page=page,
                layout=layout,
                artifact=artifact,
                image_bytes=b"image",
                expected_pointer_revision=0,
                expected_pointer_fingerprint=None,
            )

    layout_service = LayoutService()
    ocr_service = OcrService()
    application = WorkbenchApplication(
        session=session,
        import_service=ImportService(),
        layout_service=layout_service,
        ocr_service=ocr_service,
    )
    controller = workflow.WorkflowController(
        application=application,
        layout_analysis_service=layout_service,
        ocr_job_service=ocr_service,
    )
    monkeypatch.setattr(
        workflow,
        "get_config",
        lambda: {"layout_concurrency": 3, "ocr_page_concurrency": 4},
    )
    monkeypatch.setattr(workflow._LayoutServiceWorker, "start", lambda self: None)
    monkeypatch.setattr(workflow._OcrServiceWorker, "start", lambda self: None)

    assert controller.start_layout_analysis() is True
    assert controller._layout_worker is not None
    assert controller._layout_worker._max_workers == 3

    assert controller.start_ocr() is True
    assert controller._ocr_worker is not None
    assert controller._ocr_worker._max_workers == 4


def test_layout_worker_executes_pages_concurrently_and_preserves_request_order() -> None:
    probe = _OverlapProbe()

    class Service:
        def execute_page(self, request):
            probe.enter()
            try:
                time.sleep(0.05 if request.page_uid == "page-1" else 0.01)
                return request.page_uid
            finally:
                probe.leave()

    requests = tuple(
        SimpleNamespace(page_uid=f"page-{index}")
        for index in range(1, 5)
    )
    worker = workflow._LayoutServiceWorker(Service(), requests, max_workers=3)
    committed: list[tuple[str, ...]] = []
    progress: list[tuple[int, int]] = []
    worker.committed.connect(committed.append)
    worker.progress.connect(lambda current, total: progress.append((current, total)))

    worker.run()

    assert probe.peak == 3
    assert committed == [("page-1", "page-2", "page-3", "page-4")]
    assert [current for current, _total in progress] == [1, 2, 3, 4]


def test_ocr_worker_executes_pages_concurrently_and_preserves_request_order() -> None:
    probe = _OverlapProbe()

    class Service:
        def execute_page(self, request, progress_callback=None):
            probe.enter()
            try:
                if progress_callback is not None:
                    progress_callback(1, 2, "字符识别")
                time.sleep(0.05 if request.page.uid == "page-1" else 0.01)
                return OcrPageJobResult(request=request, records=object())
            finally:
                probe.leave()

    requests = tuple(
        OcrPageJobRequest(
            project_uid="project-1",
            page=SimpleNamespace(uid=f"page-{index}"),
            layout=None,
            artifact=None,
            image_bytes=b"image",
            expected_pointer_revision=0,
            expected_pointer_fingerprint=None,
        )
        for index in range(1, 5)
    )
    worker = workflow._OcrServiceWorker(Service(), requests, max_workers=2)
    committed: list[tuple[OcrPageJobResult, ...]] = []
    failures: list[object] = []
    worker.committed.connect(committed.append)
    worker.page_failed.connect(failures.append)

    worker.run()

    assert probe.peak == 2
    assert failures == []
    assert len(committed) == 1
    assert tuple(result.request.page.uid for result in committed[0]) == (
        "page-1",
        "page-2",
        "page-3",
        "page-4",
    )


def test_ocr_worker_keeps_successes_and_reports_page_failure() -> None:
    class Service:
        def execute_page(self, request, progress_callback=None):
            del progress_callback
            if request.page.uid == "page-2":
                raise RuntimeError("page failed")
            return OcrPageJobResult(request=request, records=object())

    requests = tuple(
        OcrPageJobRequest(
            project_uid="project-1",
            page=SimpleNamespace(uid=f"page-{index}"),
            layout=None,
            artifact=None,
            image_bytes=b"image",
            expected_pointer_revision=0,
            expected_pointer_fingerprint=None,
        )
        for index in range(1, 4)
    )
    worker = workflow._OcrServiceWorker(Service(), requests, max_workers=3)
    committed: list[tuple[OcrPageJobResult, ...]] = []
    failures: list[object] = []
    worker.committed.connect(committed.append)
    worker.page_failed.connect(failures.append)

    worker.run()

    assert [failure.request.page.uid for failure in failures] == ["page-2"]
    assert tuple(result.request.page.uid for result in committed[0]) == (
        "page-1",
        "page-3",
    )
