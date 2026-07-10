from __future__ import annotations

from pathlib import Path

from app.controllers.workflow_controller import WorkflowController
from app.core.project_store import ProjectStore
from app.models import OcrProject, Page
from app.models.layout_snapshot_store import clear_layout_snapshot_for_page
from app.services.project_file_service import ProjectFileService


def _page(image: Path, *, number: int = 1) -> Page:
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(f"image-{number}".encode())
    page = Page(
        image_path=str(image),
        cache_image_path=str(image),
        width=30,
        height=20,
        page_number=number,
    )
    clear_layout_snapshot_for_page(page)
    return page


def _project(tmp_path: Path) -> OcrProject:
    return OcrProject(
        name="draft",
        pages=[_page(tmp_path / "session" / "page-1.png")],
    )


def test_first_bind_creates_portable_project_and_rebinds_live_model(tmp_path: Path):
    service = ProjectFileService()
    project = _project(tmp_path)
    target = tmp_path / "exports" / "book.ocrproj"

    bound = service.bind_project(project, target)
    try:
        stored_path = bound.store.conn.execute(
            "SELECT image_path FROM page WHERE project_id=?", (project.id,)
        ).fetchone()[0]
    finally:
        bound.store.close()

    assert project.db_path == str(target)
    assert Path(project.pages[0].display_image_path).is_file()
    assert stored_path == f"book.assets/images/{project.pages[0].uid}.png"

    opened = service.open_project(target)
    try:
        assert opened.store.db_path == str(target)
        assert opened.project.name == "draft"
        assert Path(opened.project.pages[0].display_image_path).is_file()
    finally:
        opened.store.close()


def test_bound_project_materializes_new_imports_before_incremental_save(tmp_path: Path):
    service = ProjectFileService()
    project = _project(tmp_path)
    target = tmp_path / "book.ocrproj"
    bound = service.bind_project(project, target)
    try:
        project.pages.append(_page(tmp_path / "session" / "page-2.png", number=2))
        service.materialize_bound_assets(project)
        bound.store.save_project(project)
    finally:
        bound.store.close()

    opened = service.open_project(target)
    try:
        assert len(opened.project.pages) == 2
        assert all(Path(page.display_image_path).is_file() for page in opened.project.pages)
        stored_paths = [
            row[0]
            for row in opened.store.conn.execute(
                "SELECT image_path FROM page ORDER BY page_number"
            ).fetchall()
        ]
    finally:
        opened.store.close()

    assert stored_paths == [
        f"book.assets/images/{page.uid}.png" for page in project.pages
    ]


def test_rebind_makes_new_file_the_only_future_persistence_target(tmp_path: Path):
    service = ProjectFileService()
    project = _project(tmp_path)
    first = service.bind_project(project, tmp_path / "first.ocrproj")
    first.store.close()

    project.name = "second-binding"
    second_path = tmp_path / "second.ocrproj"
    second = service.bind_project(project, second_path)
    try:
        project.name = "written-to-second"
        service.materialize_bound_assets(project)
        second.store.save_project(project)
    finally:
        second.store.close()

    with ProjectStore(str(tmp_path / "first.ocrproj")) as first_store:
        first_loaded = first_store.load_project(first_store.list_projects()[0]["id"])
    with ProjectStore(str(second_path)) as second_store:
        second_loaded = second_store.load_project(second_store.list_projects()[0]["id"])

    assert first_loaded is not None and first_loaded.name == "draft"
    assert second_loaded is not None and second_loaded.name == "written-to-second"
    assert project.db_path == str(second_path)


def test_controller_keeps_new_project_in_memory_until_user_binds_path(tmp_path: Path):
    controller = WorkflowController()
    try:
        page = _page(tmp_path / "session" / "page-1.png")
        controller.on_images_ready([page])

        assert controller.project is not None
        assert controller.store is None
        assert controller.project.db_path is None
        assert controller.is_dirty is True
        assert controller.save_project() is False

        target = tmp_path / "chosen-by-user.ocrproj"
        assert controller.save_project_as(str(target)) is True
        assert controller.store is not None
        assert controller.project.db_path == str(target)
        assert controller.is_dirty is False
        assert target.is_file()
    finally:
        controller.close()
