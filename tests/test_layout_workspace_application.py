from dataclasses import FrozenInstanceError

import pytest

from app.application.contracts import (
    BlockView,
    LayoutEditCommand,
    LayoutEditResult,
    LayoutWorkspaceView,
    PageView,
)
from app.application.layout_workspace import LayoutWorkspaceQuery, query_layout_workspace
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession


def _page(project_uid: str, uid: str, page_number: int) -> PageRecord:
    return PageRecord(
        project_uid=project_uid,
        uid=uid,
        image_path=f"/images/{uid}.png",
        source_path=f"/source/{uid}.pdf",
        cache_image_path=f"/cache/{uid}.png",
        thumbnail_path=f"/thumbs/{uid}.png",
        width=1200,
        height=1600,
        page_number=page_number,
        source_page_index=page_number - 1,
        status="imported",
        error="",
        image_hash=f"hash-{uid}",
        image_revision=1,
    )


def _block(uid: str, order: int = 0) -> LayoutBlockSnapshot:
    bbox = BBox(10 + order * 100, 20, 80, 50)
    return LayoutBlockSnapshot(
        uid=uid,
        block_type=BlockType.TEXT,
        bbox=bbox,
        order=order,
        source_label="text",
        origin=BlockOrigin(
            source_engine="paddle",
            source_run_id="run-1",
            vendor_label="text",
            original_bbox=bbox,
            original_kind=BlockType.TEXT,
            raw_artifact_uid="artifact-1",
            raw_index=order,
        ),
        ocr_policy=OcrPolicy.TEXT_OCR,
        authorship=BlockSource.AUTO_LAYOUT,
    )


def test_query_projects_session_into_immutable_layout_views() -> None:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    session.page_repository.put(_page("project-1", "page-2", 2), expected_revision=0)
    session.page_repository.put(_page("project-1", "page-1", 1), expected_revision=0)
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=1,
            artifact_uid="artifact-1",
            source_engine="paddle",
            source_run_id="run-1",
            blocks=(_block("block-1"), _block("block-2", order=1)),
        ),
        expected_revision=0,
    )

    view = query_layout_workspace(session)

    assert isinstance(view, LayoutWorkspaceView)
    assert view.project_uid == "project-1"
    assert view.project_name == "Book"
    assert tuple(page.uid for page in view.pages) == ("page-1", "page-2")
    assert isinstance(view.pages, tuple)
    assert isinstance(view.pages[0], PageView)
    assert view.pages[0].layout_revision == 1
    assert view.pages[0].artifact_uid == "artifact-1"
    assert view.pages[0].block_uids == ("block-1", "block-2")
    assert isinstance(view.pages[0].blocks[0], BlockView)
    assert view.pages[1].layout_revision is None
    assert view.pages[1].blocks == ()

    with pytest.raises(FrozenInstanceError):
        view.project_name = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        view.pages[0].blocks = ()  # type: ignore[misc]

    # The projection is detached from later repository state and has no runtime owner.
    assert not hasattr(view, "session")
    assert not hasattr(view, "layout_repository")
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=2,
            artifact_uid="artifact-2",
            source_engine="manual",
            source_run_id="run-2",
            blocks=(_block("block-1"),),
        ),
        expected_revision=1,
    )
    assert view.pages[0].layout_revision == 1
    assert LayoutWorkspaceQuery().execute(session).pages[0].layout_revision == 2


def test_edit_commands_cover_application_layout_intents_and_are_frozen() -> None:
    bbox = BBox(1, 2, 30, 40)
    commands = (
        LayoutEditCommand.draw("page-1", 3, bbox, BlockType.TEXT, "text", new_block_uid="new"),
        LayoutEditCommand.delete("page-1", 3, "block-1"),
        LayoutEditCommand.move("page-1", 3, "block-1", bbox),
        LayoutEditCommand.resize("page-1", 3, "block-1", bbox),
        LayoutEditCommand.change_type("page-1", 3, "block-1", BlockType.TITLE, "title"),
        LayoutEditCommand.merge(
            "page-1",
            3,
            ("block-1", "block-2"),
            bbox,
            BlockType.TEXT,
            "text",
            primary_block_uid="block-1",
        ),
    )

    assert tuple(command.op for command in commands) == (
        "draw",
        "delete",
        "move",
        "resize",
        "change_type",
        "merge",
    )
    assert commands[0].new_block_uid == "new"
    assert commands[-1].block_uids == ("block-1", "block-2")
    assert all(command.operation == command.op for command in commands)
    assert all(
        not hasattr(command, name)
        for command in commands
        for name in ("page", "block")
    )

    with pytest.raises(FrozenInstanceError):
        commands[0].op = "delete"  # type: ignore[misc]
    with pytest.raises(ValueError):
        LayoutEditCommand("page-1", 3, "create_block")
    with pytest.raises(ValueError):
        LayoutEditCommand.delete("page-1", 3, "")
    with pytest.raises(ValueError):
        LayoutEditCommand.merge("page-1", 3, ("block-1", "block-1"), bbox, BlockType.TEXT)


def test_edit_result_is_an_immutable_value_without_runtime_dependencies() -> None:
    command = LayoutEditCommand.resize("page-1", 4, "block-1", BBox(2, 3, 40, 50))
    result = LayoutEditResult(
        command=command,
        page_uid="page-1",
        revision_before=4,
        revision_after=5,
        affected_block_uids=("block-1",),
        accepted=True,
        reason="layout_block_resized",
    )

    assert result.op == "resize"
    assert result.operation == "resize"
    assert result.revision == 5
    assert result.affected_block_uids == ("block-1",)
    assert not hasattr(result, "session")
    assert not hasattr(result, "repository")
    assert not hasattr(result, "service")
    with pytest.raises(FrozenInstanceError):
        result.revision_after = 6  # type: ignore[misc]
