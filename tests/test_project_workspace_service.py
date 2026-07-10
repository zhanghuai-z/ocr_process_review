from __future__ import annotations

from pathlib import Path

import pytest

from app.controllers.workflow_controller import WorkflowController
from app.core.project_store import ProjectDataError, ProjectStore
from app.models import OcrProject, Page
from app.models.layout_snapshot_store import clear_layout_snapshot_for_page
from app.services.project_workspace_service import ProjectWorkspaceService


class _Settings:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


def _service(tmp_path: Path) -> ProjectWorkspaceService:
    return ProjectWorkspaceService(
        tmp_path / "workspaces",
        settings=_Settings(),
    )


def test_default_workspace_root_uses_ascii_local_appdata(monkeypatch, tmp_path: Path):
    local_appdata = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    assert ProjectWorkspaceService.default_workspace_root() == (
        local_appdata / "ocr_process" / "workspaces"
    )


def _imported_page(working_db: Path) -> Page:
    image_dir = working_db.parent / ".cache" / "images"
    thumb_dir = working_db.parent / ".cache" / "thumbnails"
    image_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / "page.png"
    thumbnail = thumb_dir / "page.png"
    image.write_bytes(b"image")
    thumbnail.write_bytes(b"thumbnail")
    page = Page(
        image_path=str(image),
        cache_image_path=str(image),
        thumbnail_path=str(thumbnail),
        width=30,
        height=20,
        page_number=1,
    )
    clear_layout_snapshot_for_page(page)
    return page


def test_project_store_persists_imported_page_before_layout(tmp_path: Path):
    db_path = tmp_path / "working.ocrproj"
    page = _imported_page(db_path)
    project = OcrProject(name="draft", pages=[page], db_path=str(db_path))

    with ProjectStore(str(db_path)) as store:
        project = store.save_project(project)
        persisted_snapshot_count = store.conn.execute(
            "SELECT COUNT(*) FROM layout_snapshot WHERE project_id=?",
            (project.id,),
        ).fetchone()[0]
        loaded = store.load_project(project.id)

    assert loaded is not None
    assert len(loaded.pages) == 1
    assert loaded.pages[0].blocks == []
    assert persisted_snapshot_count == 0


def test_snapshot_replaces_target_and_keeps_working_cache_active(tmp_path: Path):
    service = _service(tmp_path)
    working = service.create_working_project("first")
    first_page = _imported_page(Path(working.store.db_path))
    working.project.pages = [first_page]
    working.store.save_project(working.project)
    target = tmp_path / "exports" / "book.ocrproj"

    assert service.write_snapshot(
        active_store=working.store,
        project=working.project,
        target_path=target,
    ) == target
    first_cache_path = working.project.db_path

    second_page = _imported_page(Path(working.store.db_path))
    second_page.image_path = str(Path(second_page.image_path).with_name("replacement.png"))
    second_page.cache_image_path = second_page.image_path
    Path(second_page.image_path).write_bytes(b"replacement-image")
    working.project.pages = [second_page]
    working.project.name = "second"
    working.store.save_project(working.project)
    service.write_snapshot(
        active_store=working.store,
        project=working.project,
        target_path=target,
    )

    with ProjectStore(str(target)) as snapshot_store:
        projects = snapshot_store.list_projects()
        loaded = snapshot_store.load_project(projects[0]["id"])

    assert len(projects) == 1
    assert loaded is not None
    assert loaded.name == "second"
    assert loaded.db_path == str(target)
    assert len(loaded.pages) == 1
    assert loaded.pages[0].uid == second_page.uid
    assert Path(loaded.pages[0].display_image_path).is_file()
    assert "book.assets" in loaded.pages[0].display_image_path
    asset_images = list((target.parent / "book.assets" / "images").iterdir())
    assert [image.name for image in asset_images] == [f"{second_page.uid}.png"]
    assert working.project.db_path == first_cache_path
    assert working.project.db_path != str(target)
    working.store.close()


def test_open_snapshot_copies_it_into_a_new_working_cache(tmp_path: Path):
    service = _service(tmp_path)
    working = service.create_working_project("draft")
    working.project.pages = [_imported_page(Path(working.store.db_path))]
    working.store.save_project(working.project)
    snapshot = tmp_path / "published.ocrproj"
    service.write_snapshot(
        active_store=working.store,
        project=working.project,
        target_path=snapshot,
    )

    opened = service.open_snapshot_into_workspace(snapshot)

    assert Path(opened.store.db_path) != snapshot
    assert opened.project.db_path == opened.store.db_path
    assert len(opened.store.list_projects()) == 1
    assert Path(opened.project.pages[0].display_image_path).is_file()
    opened.store.close()
    working.store.close()


def test_resume_reopens_the_last_managed_working_cache(tmp_path: Path):
    settings = _Settings()
    service = ProjectWorkspaceService(tmp_path / "workspaces", settings=settings)
    working = service.create_working_project("draft")
    working.project.name = "resumable"
    working.store.save_project(working.project)
    working_db = Path(working.store.db_path)
    working.store.close()

    resumed = service.resume_working_project()

    assert resumed is not None
    assert Path(resumed.store.db_path) == working_db
    assert resumed.project.name == "resumable"
    resumed.store.close()


def test_workspace_rejects_multi_project_snapshot(tmp_path: Path):
    snapshot = tmp_path / "multiple.ocrproj"
    with ProjectStore(str(snapshot)) as store:
        store.save_project(OcrProject(name="one"))
        store.save_project(OcrProject(name="two"))

    with pytest.raises(ProjectDataError, match="exactly one project"):
        _service(tmp_path).open_snapshot_into_workspace(snapshot)


def test_controller_snapshot_never_rebinds_auto_save_to_user_file(tmp_path: Path):
    service = _service(tmp_path)
    controller = WorkflowController(workspace_service=service)
    try:
        project = controller.ensure_working_project("draft")
        active_db = project.db_path
        target = tmp_path / "user-copy.ocrproj"

        assert controller.save_project_snapshot(str(target)) is True
        project.name = "cache-only-change"
        assert controller.save_project() is True

        with ProjectStore(str(target)) as store:
            projects = store.list_projects()
            saved = store.load_project(projects[0]["id"])

        assert saved is not None
        assert saved.name == "draft"
        assert controller.project is project
        assert controller.project.db_path == active_db
        assert controller.store is not None
        assert controller.store.db_path == active_db
    finally:
        controller.close()


def test_snapshot_rejects_active_working_cache_even_without_extension(tmp_path: Path):
    service = _service(tmp_path)
    working = service.create_working_project("draft")
    try:
        working_db = Path(working.store.db_path)
        with pytest.raises(ProjectDataError, match="active working cache"):
            service.write_snapshot(
                active_store=working.store,
                project=working.project,
                target_path=working_db.with_suffix(""),
            )
    finally:
        working.store.close()
