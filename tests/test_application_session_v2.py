from __future__ import annotations

import ast
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from app.controllers.workflow_controller import WorkflowController
from app.core.workflow_state import (
    STEP_LAYOUT,
    STEP_OCR,
    WorkflowProgressState,
    compute_max_step,
    page_gate_info,
    pending_ocr_page_uids,
)
from app.models.layout_snapshot import LayoutSnapshot
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession
from app.ui.main_window import MainWindow, _ShellProgress


def _page(session: ProjectSession) -> PageRecord:
    record = PageRecord(
        project_uid=session.project_uid,
        uid="page-1",
        image_path="/tmp/page-1.png",
        source_path="source.png",
        cache_image_path="",
        thumbnail_path="",
        width=100,
        height=80,
        page_number=1,
        source_page_index=0,
        status="imported",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )
    session.page_repository.put(record, expected_revision=0)
    return record


def test_workflow_state_uses_adopted_session_facts() -> None:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = _page(session)

    assert compute_max_step(session) == STEP_LAYOUT
    gate = page_gate_info(session, page.uid)
    assert gate.reason_code == "layout_not_done"
    assert pending_ocr_page_uids(session) == ()

    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid="",
            source_engine="test",
            source_run_id="layout-run",
            blocks=(),
        ),
        expected_revision=0,
    )

    assert compute_max_step(session) == STEP_OCR
    assert pending_ocr_page_uids(session) == (page.uid,)
    assert page_gate_info(session, page.uid).action_enabled is True


def test_controller_import_emits_immutable_records_and_captures_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)

    controller = WorkflowController()
    page_events: list[object] = []
    controller.page_records_changed.connect(page_events.append)

    result = controller.import_paths([source], name="Imported book")

    assert result.success_count == 1
    assert controller.session is not None
    assert controller.project_name == "Imported book"
    assert controller.page_records == result.pages
    assert isinstance(controller.page_records, tuple)
    assert isinstance(page_events[-1], tuple)
    assert all(isinstance(page, PageRecord) for page in page_events[-1])
    assert not hasattr(controller, "_project")

    page = controller.page_records[0]
    controller.session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid="",
            source_engine="test",
            source_run_id="layout-run",
            blocks=(),
        ),
        expected_revision=0,
    )
    snapshot = controller.capture_export_snapshot()
    assert snapshot.project.project_uid == controller.project_uid
    assert snapshot.pages[0].page == page
    controller.close()


def test_import_publishes_pages_before_current_page_and_completion(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)
    controller = WorkflowController()
    events: list[tuple[str, object]] = []
    controller.page_records_changed.connect(
        lambda records: events.append(("pages", tuple(page.uid for page in records)))
    )
    controller.view_state_changed.connect(
        lambda state: events.append(("view", state.current_page_uid))
    )
    controller.import_finished.connect(
        lambda result: events.append(("finished", result.pages[0].uid))
    )

    result = controller.import_paths([source])

    page_uid = result.pages[0].uid
    assert events[-3:] == [
        ("pages", (page_uid,)),
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

    page_uid = result.pages[0].uid
    assert window._ocr_panel._pages_by_uid == {page_uid: result.pages[0]}
    assert window._ocr_panel._current_page_uid == page_uid
    controller.close()
    window.deleteLater()
    app.processEvents()


def test_main_window_keeps_ocr_stage_on_layout_workbench(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "page.png"
    Image.new("RGB", (24, 16), "white").save(source)
    controller = WorkflowController()
    window = MainWindow(controller)
    result = controller.import_paths([source])
    page = result.pages[0]
    assert controller.session is not None
    controller.session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid="",
            source_engine="test",
            source_run_id="layout-run",
            blocks=(),
        ),
        expected_revision=0,
    )

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
    page = controller.page_records[0]
    controller.session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid="",
            source_engine="test",
            source_run_id="layout-run",
            blocks=(),
        ),
        expected_revision=0,
    )
    controller.save_project_as(target)

    assert controller.is_bound_project is True
    assert controller.session is not None
    assert not hasattr(controller, "_store")
    saved_uid = controller.project_uid
    controller.open_project(target)
    assert controller.project_uid == saved_uid
    assert controller.page_records[0].uid == page.uid
    controller.close()


def test_owned_application_boundaries_contain_no_retired_runtime_flow() -> None:
    paths = (
        Path("app/controllers/workflow_controller.py"),
        Path("app/ui/main_window.py"),
        Path("app/ui/recognize/ocr_panel.py"),
        Path("app/ui/widgets/page_directory.py"),
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
