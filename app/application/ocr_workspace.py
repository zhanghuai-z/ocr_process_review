"""Immutable read model for the OCR workspace."""
from __future__ import annotations

from dataclasses import dataclass

from app.application.contracts import PageView
from app.application.layout_workspace import build_layout_workspace_view
from app.core.ocr_currentness import current_ocr_observation
from app.models.project_session import ProjectSession


BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class OcrAtomView:
    atom_uid: str
    text: str
    bbox: BBox
    confidence: float
    index: int


@dataclass(frozen=True, slots=True)
class OcrLineView:
    line_uid: str
    region_uid: str
    text: str
    bbox: BBox
    confidence: float
    order: int
    atoms: tuple[OcrAtomView, ...]


@dataclass(frozen=True, slots=True)
class OcrRegionView:
    region_uid: str
    block_uid: str | None
    kind: str
    label: str
    bbox: BBox
    order: int
    lines: tuple[OcrLineView, ...]


@dataclass(frozen=True, slots=True)
class OcrPageView:
    page: PageView
    batch_uid: str | None
    regions: tuple[OcrRegionView, ...]

    @property
    def page_uid(self) -> str:
        return self.page.page_uid

    @property
    def has_current_ocr(self) -> bool:
        return self.batch_uid is not None

    @property
    def line_count(self) -> int:
        return sum(len(region.lines) for region in self.regions)


@dataclass(frozen=True, slots=True)
class OcrWorkspaceView:
    project_uid: str
    project_name: str
    pages: tuple[OcrPageView, ...]

    @property
    def page_uids(self) -> tuple[str, ...]:
        return tuple(page.page_uid for page in self.pages)

    @property
    def line_count(self) -> int:
        return sum(page.line_count for page in self.pages)


def build_ocr_workspace_view(session: ProjectSession) -> OcrWorkspaceView:
    """Join current layout and active OCR observations without mutating either."""

    if not isinstance(session, ProjectSession):
        raise TypeError("OCR workspace query requires ProjectSession")
    layout_workspace = build_layout_workspace_view(session)
    ocr = session.ocr_observation_repository
    bindings = session.binding_repository.all()
    page_views: list[OcrPageView] = []

    for page in layout_workspace.pages:
        observation = current_ocr_observation(session, page.page_uid)
        if observation is None:
            page_views.append(OcrPageView(page=page, batch_uid=None, regions=()))
            continue
        batch = observation.batch

        block_uid_by_region = {
            binding.target_uid: binding.source_uid
            for binding in bindings
            if binding.target_uid in batch.region_uids
            and binding.source_uid in page.block_uids
        }
        lines_by_region: dict[str, list[OcrLineView]] = {
            region_uid: [] for region_uid in batch.region_uids
        }
        for line_uid in batch.line_uids:
            line = ocr.get_line(line_uid)
            atoms = tuple(
                OcrAtomView(
                    atom_uid=atom.uid,
                    text=atom.text,
                    bbox=atom.bbox,
                    confidence=atom.confidence,
                    index=atom.index,
                )
                for atom in sorted(
                    (ocr.get_atom(atom_uid) for atom_uid in line.atom_uids),
                    key=lambda item: (item.index, item.uid),
                )
            )
            lines_by_region.setdefault(line.region_uid, []).append(
                OcrLineView(
                    line_uid=line.uid,
                    region_uid=line.region_uid,
                    text=line.text,
                    bbox=line.bbox,
                    confidence=line.confidence,
                    order=line.order,
                    atoms=atoms,
                )
            )

        regions = tuple(
            OcrRegionView(
                region_uid=region.uid,
                block_uid=block_uid_by_region.get(region.uid),
                kind=region.kind,
                label=region.label,
                bbox=region.bbox,
                order=region.order,
                lines=tuple(sorted(
                    lines_by_region.get(region.uid, ()),
                    key=lambda item: (item.order, item.line_uid),
                )),
            )
            for region in sorted(
                (ocr.get_region(region_uid) for region_uid in batch.region_uids),
                key=lambda item: (item.order, item.uid),
            )
        )
        page_views.append(OcrPageView(page=page, batch_uid=batch.uid, regions=regions))

    return OcrWorkspaceView(
        project_uid=session.project_uid,
        project_name=session.project_record.name,
        pages=tuple(page_views),
    )


__all__ = [
    "OcrAtomView",
    "OcrLineView",
    "OcrPageView",
    "OcrRegionView",
    "OcrWorkspaceView",
    "build_ocr_workspace_view",
]
