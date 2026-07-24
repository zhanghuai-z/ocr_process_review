from __future__ import annotations

import ast
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from app.controllers.workflow_controller import WorkflowController
from app.application import OcrWorkspaceView, PageView, WorkbenchApplication
from app.core.workflow_state import (
    STEP_LAYOUT,
    STEP_OCR,
    WorkflowProgressState,
    compute_max_step,
    page_gate_info,
    pending_ocr_page_uids,
)
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.services import ImportService
from app.ui.main_window import MainWindow, _ShellProgress


def _page(
    session: ProjectSession,
    *,
    uid: str = "page-1",
    page_number: int = 1,
) -> PageRecord:
    record = PageRecord(
        project_uid=session.project_uid,
        uid=uid,
        image_path=f"/tmp/{uid}.png",
        source_path="source.png",
        cache_image_path="",
        thumbnail_path="",
        width=100,
        height=80,
        page_number=page_number,
        source_page_index=page_number - 1,
        status="imported",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )
    session.page_repository.put(record, expected_revision=0)
    return record


def _layout(session: ProjectSession, page: PageRecord) -> None:
    artifact = session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=session.project_uid,
            uid=f"artifact-{page.uid}",
            page_uid=page.uid,
            source_engine="test",
            source_run_id=f"layout-run-{page.uid}",
            image_hash=page.image_hash,
            payload_json="{}",
        )
    )
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid=artifact.uid,
            source_engine="test",
            source_run_id=f"layout-run-{page.uid}",
            blocks=(),
        ),
        expected_revision=0,
    )


def test_workflow_state_uses_adopted_session_facts() -> None:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = _page(session)

    assert compute_max_step(session) == STEP_LAYOUT
    gate = page_gate_info(session, page.uid)
    assert gate.reason_code == "layout_not_done"
    assert pending_ocr_page_uids(session) == ()

    _layout(session, page)

    assert compute_max_step(session) == STEP_OCR
    assert pending_ocr_page_uids(session) == (page.uid,)
    assert page_gate_info(session, page.uid).action_enabled is True


def test_controller_orders_all_pending_ocr_pages_with_submitted_page_first() -> None:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    pages = tuple(
        _page(session, uid=f"page-{index}", page_number=index)
        for index in range(1, 4)
    )
    for page in pages:
        _layout(session, page)
    controller = WorkflowController(
        application=WorkbenchApplication(session=session, import_service=ImportService())
    )

    assert controller.pending_ocr_page_uids("page-2") == (
        "page-2",
        "page-1",
        "page-3",
    )
    controller.close()


def test_controller_import_emits_immutable_records(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)

    controller = WorkflowController()
    workspace_events: list[object] = []
    controller.ocr_workspace_changed.connect(workspace_events.append)

    result = controller.import_paths([source], name="Imported book")

    assert result.success_count == 1
    assert controller.has_project is True
    assert controller.project_name == "Imported book"
    assert tuple(page.uid for page in controller.pages) == result.page_uids
    assert isinstance(controller.pages, tuple)
    assert isinstance(workspace_events[-1], OcrWorkspaceView)
    assert all(isinstance(page.page, PageView) for page in workspace_events[-1].pages)
    assert not hasattr(controller, "_project")

    controller.close()


def test_import_publishes_pages_before_current_page_and_completion(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)
    controller = WorkflowController()
    events: list[tuple[str, object]] = []
    controller.ocr_workspace_changed.connect(
        lambda workspace: events.append(("ocr", workspace.page_uids))
    )
    controller.view_state_changed.connect(
        lambda state: events.append(("view", state.current_page_uid))
    )
    controller.import_finished.connect(
        lambda result: events.append(("finished", result.page_uids[0]))
    )

    result = controller.import_paths([source])

    page_uid = result.page_uids[0]
    assert events[-3:] == [
        ("ocr", (page_uid,)),
        ("view", page_uid),
        ("finished", page_uid),
    ]
    controller.close()


def test_main_window_binds_ocr_pages_before_selecting_imported_uid(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)
    controller = WorkflowController()
    window = MainWindow(controller)

    result = controller.import_paths([source])

    page_uid = result.page_uids[0]
    assert tuple(window._ocr_panel._pages_by_uid) == (page_uid,)
    assert isinstance(window._ocr_panel._pages_by_uid[page_uid], PageView)
    assert window._ocr_panel._current_page_uid == page_uid
    controller.close()
    window.deleteLater()
    app.processEvents()


def test_main_window_keeps_ocr_stage_on_layout_workbench(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = _page(session)
    _layout(session, page)
    controller = WorkflowController(
        application=WorkbenchApplication(session=session, import_service=ImportService())
    )
    window = MainWindow(controller)

    window._go_to_step(STEP_OCR)

    assert window._stack.currentWidget() is window._layout_panel
    assert window._stack_by_step[STEP_LAYOUT] is window._layout_panel
    assert window._stack_by_step[STEP_OCR] is window._layout_panel
    controller.close()
    window.deleteLater()
    app.processEvents()


def test_shell_progress_combines_page_and_in_page_ocr_progress() -> None:
    app = QApplication.instance() or QApplication([])
    progress = _ShellProgress()

    progress.update_ocr(WorkflowProgressState(
        phase="ocr",
        current=5,
        total=10,
        completed_pages=1,
        total_pages=2,
        message="Hanwang OCR 识别中 5/10",
        page_uid="page-2",
    ))

    assert progress.active is True
    assert progress._title.text() == "OCR"
    assert progress._detail.text() == "字符识别"
    assert progress._count.text() == "2/2 页"
    assert progress._bar.value() == 85
    progress.finish()
    assert progress.active is False
    progress.deleteLater()
    app.processEvents()


def test_shell_progress_uses_one_based_layout_page_count() -> None:
    app = QApplication.instance() or QApplication([])
    progress = _ShellProgress()

    progress.update_layout_stage(1, 3, "版面分析")

    assert progress._count.text() == "1/3 页"
    assert progress._bar.value() == 2
    progress.deleteLater()
    app.processEvents()


def test_main_window_more_menu_opens_ocr_settings(monkeypatch) -> None:
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    app = QApplication.instance() or QApplication([])
    controller = WorkflowController()
    window = MainWindow(controller)
    opened: list[ApiSettingsDialog] = []
    monkeypatch.setattr(
        ApiSettingsDialog,
        "exec",
        lambda dialog: opened.append(dialog) or 0,
    )

    menu = window._top_bar._btn_more_menu.menu()
    settings_action = next(action for action in menu.actions() if action.text() == "设置…")
    settings_action.trigger()

    assert len(opened) == 1
    controller.close()
    window.deleteLater()
    app.processEvents()


def test_controller_composes_layout_client_from_current_settings(monkeypatch) -> None:
    import app.controllers.workflow_controller as controller_module

    monkeypatch.setattr(
        controller_module,
        "get_config",
        lambda: {
            "api_url": "https://paddleocr.aistudio-app.com",
            "api_token": "configured-token",
            "api_timeout": 45,
            "paddle_api_network_mode": "env_proxy",
        },
    )
    controller = WorkflowController()

    service = controller._build_default_layout_analysis_service()
    client = service._client

    assert client.jobs_url == "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    assert client.token == "configured-token"
    assert client.request_timeout == 30
    assert client.poll_timeout == 180
    assert client.network_mode == "env_proxy"
    controller.close()


def test_controller_save_and_open_rebinds_only_the_session(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)
    target = tmp_path / "book.ocrproj"

    controller = WorkflowController()
    controller.import_paths([source], name="Saved book")
    page = controller.pages[0]
    controller.save_project_as(target)

    assert controller.is_bound_project is True
    assert controller.has_project is True
    assert not hasattr(controller, "_store")
    saved_uid = controller.project_uid
    controller.open_project(target)
    assert controller.project_uid == saved_uid
    assert controller.pages[0].uid == page.uid
    controller.close()


def test_owned_application_boundaries_contain_no_retired_runtime_flow() -> None:
    paths = (
        Path("app/controllers/workflow_controller.py"),
        Path("app/ui/main_window.py"),
        Path("app/ui/recognize/ocr_panel.py"),
        Path("app/ui/export/export_dialog.py"),
        Path("app/core/workflow_state.py"),
        Path("app/services/__init__.py"),
    )
    forbidden = {
        "OcrProject",
        "Page",
        "Block",
        "Line",
        "Char",
        "LayoutWorker",
        "OcrPipeline",
        "ProjectStore",
        "ProofPersistenceService",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        imported = {
            alias.name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not names & forbidden, path
        assert not imported & forbidden, path
