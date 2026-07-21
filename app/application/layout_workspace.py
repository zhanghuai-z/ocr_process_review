"""Read-only layout workspace application query."""
from __future__ import annotations

from app.application.contracts import LayoutWorkspaceView, PageView
from app.models.project_session import (
    ProjectSession,
    RecordNotFoundError,
)


def build_layout_workspace_view(session: ProjectSession) -> LayoutWorkspaceView:
    """Build an immutable layout projection from current session facts only."""

    if not isinstance(session, ProjectSession):
        raise TypeError("layout workspace query requires ProjectSession")

    pages: list[PageView] = []
    for page in sorted(
        session.page_repository.all(),
        key=lambda item: (item.page_number, item.uid),
    ):
        try:
            layout = session.layout_repository.get(page.uid)
        except RecordNotFoundError:
            layout = None
        pages.append(PageView.from_records(page, layout))

    return LayoutWorkspaceView(
        project_uid=session.project_uid,
        project_name=session.project_record.name,
        pages=tuple(pages),
    )


def query_layout_workspace(session: ProjectSession) -> LayoutWorkspaceView:
    """Return an immutable layout workspace projection for ``session``."""

    return build_layout_workspace_view(session)


class LayoutWorkspaceQuery:
    """Stateless callable facade for the layout workspace query."""

    __slots__ = ()

    def __call__(self, session: ProjectSession) -> LayoutWorkspaceView:
        return build_layout_workspace_view(session)

    def execute(self, session: ProjectSession) -> LayoutWorkspaceView:
        return self(session)


__all__ = [
    "LayoutWorkspaceQuery",
    "build_layout_workspace_view",
    "query_layout_workspace",
]
