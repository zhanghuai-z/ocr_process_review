"""Immutable point-in-time input for all export adapters."""
from __future__ import annotations

from dataclasses import dataclass

from .layout_snapshot import LayoutSnapshot
from .ocr_records import OcrAtom, OcrBatch, OcrCandidate, OcrLine, OcrRegion
from .project_session import BindingRecord, PageRecord, ProjectRecord, TableTextRecord
from .proof_records import ProofState


@dataclass(frozen=True, slots=True)
class ExportPageSnapshot:
    """All adopted facts needed to export one page, without runtime objects."""

    page: PageRecord
    layout: LayoutSnapshot
    active_ocr_batch: OcrBatch | None = None
    ocr_regions: tuple[OcrRegion, ...] = ()
    ocr_lines: tuple[OcrLine, ...] = ()
    ocr_atoms: tuple[OcrAtom, ...] = ()
    ocr_candidates: tuple[OcrCandidate, ...] = ()
    proof_states: tuple[ProofState, ...] = ()
    bindings: tuple[BindingRecord, ...] = ()
    table_texts: tuple[TableTextRecord, ...] = ()

    def __post_init__(self) -> None:
        project_uid = self.page.project_uid
        page_uid = self.page.uid
        if self.layout.page_uid != page_uid:
            raise ValueError("export layout belongs to another page")
        records = (
            *self.ocr_regions,
            *self.ocr_lines,
            *self.ocr_atoms,
            *self.ocr_candidates,
            *self.proof_states,
            *self.bindings,
            *self.table_texts,
        )
        if any(record.project_uid != project_uid for record in records):
            raise ValueError("export page contains a record from another project")
        if self.active_ocr_batch is not None:
            if self.active_ocr_batch.project_uid != project_uid:
                raise ValueError("active OCR batch belongs to another project")
            if self.active_ocr_batch.scope_uid != page_uid:
                raise ValueError("active OCR batch belongs to another page")
        if any(record.page_uid != page_uid for record in self.ocr_regions):
            raise ValueError("OCR region belongs to another page")
        if any(record.page_uid != page_uid for record in self.ocr_lines):
            raise ValueError("OCR line belongs to another page")
        if any(record.page_uid != page_uid for record in self.table_texts):
            raise ValueError("table text belongs to another page")


@dataclass(frozen=True, slots=True)
class ExportProjectSnapshot:
    """One coherent export transaction input captured from a ProjectSession."""

    project: ProjectRecord
    pages: tuple[ExportPageSnapshot, ...]

    def __post_init__(self) -> None:
        pages = tuple(self.pages)
        if any(page.page.project_uid != self.project.project_uid for page in pages):
            raise ValueError("export page belongs to another project")
        page_uids = tuple(page.page.uid for page in pages)
        if len(set(page_uids)) != len(page_uids):
            raise ValueError("export snapshot contains duplicate page UIDs")
        object.__setattr__(
            self,
            "pages",
            tuple(sorted(pages, key=lambda page: page.page.page_number)),
        )


__all__ = ["ExportPageSnapshot", "ExportProjectSnapshot"]
