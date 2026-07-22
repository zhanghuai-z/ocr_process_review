from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.infrastructure.project_store import FORMAT_GENERATION, ProjectStoreDataError, load_session
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrCandidate,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import (
    BindingRecord,
    PageRecord,
    ProjectRecord,
    ProjectSession,
    TableTextRecord,
)
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.services import project_file_service as project_file_service_module
from app.services.project_file_service import BoundProject, ProjectFileService


def _page(tmp_path: Path, project_uid: str, uid: str, page_number: int) -> PageRecord:
    image = tmp_path / "imports" / f"{uid}.png"
    thumbnail = tmp_path / "imports" / f"{uid}.thumb.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(f"image-{uid}".encode())
    thumbnail.write_bytes(f"thumbnail-{uid}".encode())
    return PageRecord(
        project_uid=project_uid,
        uid=uid,
        image_path=str(image),
        source_path="input.pdf",
        cache_image_path=str(image),
        thumbnail_path=str(thumbnail),
        width=120,
        height=240,
        page_number=page_number,
        source_page_index=page_number - 1,
        status="imported",
        error="",
        image_hash=f"hash-{uid}",
        image_revision=1,
    )


def _session(tmp_path: Path, project_uid: str = "project-1") -> ProjectSession:
    session = ProjectSession(ProjectRecord(project_uid, "Draft"))
    page = session.page_repository.put(
        _page(tmp_path, project_uid, "page-1", 1),
        expected_revision=0,
    )
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=page.uid,
            revision=1,
            artifact_uid="paddle-artifact-1",
            source_engine="test-layout",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=BBox(1, 2, 30, 40),
                    order=0,
                    source_label="text",
                    origin=BlockOrigin(
                        created_by=BlockSource.AUTO_LAYOUT.value,
                        source_engine="test-layout",
                        source_run_id="layout-run-1",
                        vendor_label="text",
                        original_bbox=BBox(1, 2, 30, 40),
                        original_kind=BlockType.TEXT,
                        raw_artifact_uid="layout-artifact-1",
                        raw_json_path="records[0]",
                        raw_index=0,
                    ),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    authorship=BlockSource.AUTO_LAYOUT,
                    uid="layout-block-1",
                ),
            ),
        ),
        expected_revision=0,
    )
    session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=project_uid,
            uid="paddle-artifact-1",
            page_uid=page.uid,
            source_engine="test-layout",
            source_run_id="layout-run-1",
            image_hash=page.image_hash,
            payload_json='{"blocks": []}',
        )
    )

    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=project_uid,
            uid="run-1",
            engine="test-ocr",
            layout_fingerprint="layout-fingerprint",
            input_fingerprint="input-fingerprint",
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=project_uid,
            uid="region-1",
            run_uid=run.uid,
            page_uid=page.uid,
            bbox=(1, 2, 31, 42),
            kind="text",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=project_uid,
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid=page.uid,
            text="observed",
            bbox=(1, 2, 31, 12),
            confidence=0.9,
            atom_uids=("atom-1",),
        )
    )
    atom = ocr.append_atom(
        OcrAtom(
            project_uid=project_uid,
            uid="atom-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=0,
            text="o",
            bbox=(1, 2, 5, 12),
            confidence=0.9,
            candidate_uids=("candidate-1",),
            source="test",
            granularity="char",
            token_text="o",
        )
    )
    batch = ocr.append_batch(
        OcrBatch(
            project_uid=project_uid,
            uid="batch-1",
            run_uid=run.uid,
            scope_uid=page.uid,
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=run.layout_fingerprint,
            region_uids=(region.uid,),
            line_uids=(line.uid,),
            atom_uids=(atom.uid,),
            candidate_uids=("candidate-1",),
        )
    )
    candidate = ocr.append_candidate(
        OcrCandidate(
            project_uid=project_uid,
            uid="candidate-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            atom_uid=atom.uid,
            batch_uid=batch.uid,
            text="o",
            confidence=0.9,
            rank=0,
            source="test-ocr",
            bbox=(1, 2, 5, 12),
        )
    )
    ocr.switch_active_pointer(
        OcrActivePointer(
            project_uid=project_uid,
            uid="active-page-1",
            scope_uid=page.uid,
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )

    proof = session.proof_repository
    anchor = ProofAnchorSnapshot(
            project_uid=project_uid,
            uid="anchor-1",
            scope_uid=page.uid,
            layout_fingerprint=run.layout_fingerprint,
            source_fingerprint=batch.fingerprint,
            anchor_revision=1,
    )
    segment = ProofAlignmentSegment(
            project_uid=project_uid,
            uid="segment-1",
            anchor_uid=anchor.uid,
            text_unit_uid="proof-text-1",
            source_line_uids=("line-1",),
            source_start=0,
            source_end=8,
            proof_start=0,
            proof_end=7,
    )
    alignment_slice = ProofAlignmentSlice(
            project_uid=project_uid,
            uid="slice-1",
            segment_uid=segment.uid,
            source_start=0,
            source_end=8,
            proof_start=0,
            proof_end=7,
            source_text="observed",
            proof_text="correct",
    )
    proof.create_state(
        ProofState(
            project_uid=project_uid,
            uid="proof-1",
            anchor_snapshot=anchor,
            text_units=(
                ProofTextUnit(
                    project_uid=project_uid,
                    uid="proof-text-1",
                    order=0,
                    text="correct",
                    status="modified",
                ),
            ),
            alignment_segments=(segment,),
            alignment_slices=(alignment_slice,),
        )
    )

    binding = session.binding_repository.create(
        BindingRecord(
            project_uid=project_uid,
            uid="binding-1",
            source_uid="layout-block-1",
            target_uid=region.uid,
            relation="observed_by",
            source_fingerprint="layout-fingerprint",
            target_fingerprint=region.fingerprint,
        )
    )
    session.binding_repository.replace(
        replace(binding, attributes=(("reviewed", "yes"),)),
        expected_revision=binding.revision,
        expected_fingerprint=binding.fingerprint,
    )
    table_text = session.table_text_repository.create(
        TableTextRecord(
            project_uid=project_uid,
            uid="cell-1",
            page_uid=page.uid,
            table_uid="table-1",
            row_index=0,
            column_index=0,
            text="cell",
            source_fingerprint="table-source",
        )
    )
    session.table_text_repository.replace(
        replace(table_text, text="corrected cell"),
        expected_revision=table_text.revision,
        expected_fingerprint=table_text.fingerprint,
    )
    return session


def _records(session: ProjectSession) -> tuple[object, ...]:
    ocr = session.ocr_observation_repository
    proof = session.proof_repository
    return (
        session.project_record,
        session.page_repository.all(),
        session.layout_repository.all(),
        session.paddle_artifact_repository.all(),
        ocr.all_runs(),
        ocr.all_regions(),
        ocr.all_lines(),
        ocr.all_atoms(),
        ocr.all_candidates(),
        ocr.all_batches(),
        ocr.all_active_pointers(),
        proof.all_states(),
        session.binding_repository.all(),
        session.table_text_repository.all(),
    )


def _with_extra_page(session: ProjectSession, tmp_path: Path) -> ProjectSession:
    pages = session.page_repository.all() + (
        _page(tmp_path, session.project_uid, "page-2", 2),
    )
    return ProjectSession.from_records(
        session.project_record,
        pages=pages,
        paddle_artifacts=session.paddle_artifact_repository.all(),
        layouts=session.layout_repository.all(),
        ocr_runs=session.ocr_observation_repository.all_runs(),
        ocr_regions=session.ocr_observation_repository.all_regions(),
        ocr_lines=session.ocr_observation_repository.all_lines(),
        ocr_atoms=session.ocr_observation_repository.all_atoms(),
        ocr_candidates=session.ocr_observation_repository.all_candidates(),
        ocr_batches=session.ocr_observation_repository.all_batches(),
        ocr_active_pointers=session.ocr_observation_repository.all_active_pointers(),
        proof_states=session.proof_repository.all_states(),
        bindings=session.binding_repository.all(),
        table_texts=session.table_text_repository.all(),
        save_path=session.save_path,
    )


def test_open_uses_strict_store_and_validates_page_assets(tmp_path: Path):
    service = ProjectFileService()
    service.bind_project(_session(tmp_path), tmp_path / "book.ocrproj")

    opened = service.open_project(tmp_path / "book.ocrproj")

    assert isinstance(opened, BoundProject)
    assert isinstance(opened.store, project_file_service_module.ProjectStore)
    assert isinstance(opened.session, ProjectSession)
    page = opened.session.page_repository.get("page-1")
    assert Path(page.image_path).is_file()
    assert Path(page.thumbnail_path).is_file()
    assert opened.session.save_path == str((tmp_path / "book.ocrproj").resolve())
    assert "save_path" not in (tmp_path / "book.ocrproj").read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_first_bind_stages_assets_and_preserves_every_repository(tmp_path: Path):
    service = ProjectFileService()
    source = _session(tmp_path)
    target = tmp_path / "exports" / "book.ocrproj"

    bound = service.bind_project(source, target)
    stored = load_session(target)

    assert source.save_path is None
    assert bound.session.save_path == str(target.resolve())
    assert (target.parent / "book.assets" / "images" / "page-1.png").is_file()
    assert (target.parent / "book.assets" / "thumbnails" / "page-1.png").is_file()
    stored_records = _records(stored)
    source_records = _records(source)
    assert stored_records[:1] == source_records[:1]
    assert stored_records[2:] == source_records[2:]
    assert stored.page_repository.get("page-1").image_path == "book.assets/images/page-1.png"
    assert stored.page_repository.get("page-1").thumbnail_path == (
        "book.assets/thumbnails/page-1.png"
    )
    assert FORMAT_GENERATION.encode() in target.read_bytes()


def test_normal_save_materializes_new_page_and_persists_active_session(tmp_path: Path):
    service = ProjectFileService()
    target = tmp_path / "book.ocrproj"
    first = service.bind_project(_session(tmp_path), target)
    changed = _with_extra_page(first.session, tmp_path)

    saved = service.save_project(BoundProject(first.store, changed))
    opened = service.open_project(target)

    assert saved.session.save_path == str(target.resolve())
    assert len(opened.session.page_repository.all()) == 2
    second = opened.session.page_repository.get("page-2")
    assert Path(second.image_path).is_file()
    assert Path(second.thumbnail_path).is_file()
    persisted = load_session(target)
    assert persisted.page_repository.get("page-2").image_path == "book.assets/images/page-2.png"


def test_save_as_rebinds_to_new_path_without_changing_old_file(tmp_path: Path):
    service = ProjectFileService()
    first_path = tmp_path / "first.ocrproj"
    second_path = tmp_path / "second.ocrproj"
    first = service.bind_project(_session(tmp_path), first_path)
    changed = replace(first.session.project_record, name="saved elsewhere")
    changed_session = ProjectSession.from_records(
        changed,
        pages=first.session.page_repository.all(),
        paddle_artifacts=first.session.paddle_artifact_repository.all(),
        layouts=first.session.layout_repository.all(),
        ocr_runs=first.session.ocr_observation_repository.all_runs(),
        ocr_regions=first.session.ocr_observation_repository.all_regions(),
        ocr_lines=first.session.ocr_observation_repository.all_lines(),
        ocr_atoms=first.session.ocr_observation_repository.all_atoms(),
        ocr_candidates=first.session.ocr_observation_repository.all_candidates(),
        ocr_batches=first.session.ocr_observation_repository.all_batches(),
        ocr_active_pointers=first.session.ocr_observation_repository.all_active_pointers(),
        proof_states=first.session.proof_repository.all_states(),
        bindings=first.session.binding_repository.all(),
        table_texts=first.session.table_text_repository.all(),
        save_path=first.session.save_path,
    )

    second = service.save_as(BoundProject(first.store, changed_session), second_path)

    assert first.session.save_path == str(first_path.resolve())
    assert second.session.save_path == str(second_path.resolve())
    assert load_session(first_path).project_record.name == "Draft"
    assert load_session(second_path).project_record.name == "saved elsewhere"
    assert first_path.parent.joinpath("first.assets").is_dir()
    assert second_path.parent.joinpath("second.assets").is_dir()


def test_bind_rejects_missing_page_asset_before_creating_target(tmp_path: Path):
    service = ProjectFileService()
    source = _session(tmp_path)
    page = source.page_repository.get("page-1")
    missing = replace(page, thumbnail_path=str(tmp_path / "missing-thumb.png"))
    source = ProjectSession.from_records(
        source.project_record,
        pages=(missing,),
        paddle_artifacts=source.paddle_artifact_repository.all(),
        layouts=source.layout_repository.all(),
        ocr_runs=source.ocr_observation_repository.all_runs(),
        ocr_regions=source.ocr_observation_repository.all_regions(),
        ocr_lines=source.ocr_observation_repository.all_lines(),
        ocr_atoms=source.ocr_observation_repository.all_atoms(),
        ocr_candidates=source.ocr_observation_repository.all_candidates(),
        ocr_batches=source.ocr_observation_repository.all_batches(),
        ocr_active_pointers=source.ocr_observation_repository.all_active_pointers(),
        proof_states=source.proof_repository.all_states(),
        bindings=source.binding_repository.all(),
        table_texts=source.table_text_repository.all(),
    )
    target = tmp_path / "missing.ocrproj"

    with pytest.raises(ProjectStoreDataError, match="thumbnail_path"):
        service.bind_project(source, target)

    assert source.save_path is None
    assert not target.exists()
    assert not target.with_name("missing.assets").exists()


def test_atomic_rollback_restores_existing_db_and_assets(tmp_path: Path, monkeypatch):
    service = ProjectFileService()
    target = tmp_path / "book.ocrproj"
    original = service.bind_project(_session(tmp_path), target)
    before_db = target.read_bytes()
    before_asset = (tmp_path / "book.assets" / "images" / "page-1.png").read_bytes()
    changed = _with_extra_page(original.session, tmp_path)

    real_replace = project_file_service_module.os.replace
    failed = False

    def fail_database_install(source: Path, destination: Path) -> None:
        nonlocal failed
        if destination == target and not failed:
            failed = True
            raise OSError("forced database install failure")
        real_replace(source, destination)

    monkeypatch.setattr(project_file_service_module.os, "replace", fail_database_install)
    with pytest.raises(OSError, match="forced database install failure"):
        service.save_project(BoundProject(original.store, changed))

    assert target.read_bytes() == before_db
    assert (tmp_path / "book.assets" / "images" / "page-1.png").read_bytes() == before_asset
    assert not (tmp_path / "book.assets" / "images" / "page-2.png").exists()
    assert changed.save_path == str(target.resolve())
    assert not list(tmp_path.glob(".book.save-stage-*"))
