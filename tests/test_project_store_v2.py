from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sqlite3

import pytest

from app.infrastructure.project_store import (
    FORMAT_GENERATION,
    InvalidProjectFormatError,
    ProjectStoreDataError,
    ProjectUidMismatchError,
    load_session,
    save_session,
)
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


def _page(project_uid: str, uid: str = "page-1") -> PageRecord:
    return PageRecord(
        project_uid=project_uid,
        uid=uid,
        image_path=f"/images/{uid}.png",
        source_path="/sources/input.pdf",
        cache_image_path=f"/cache/{uid}.png",
        thumbnail_path=f"/thumbs/{uid}.png",
        width=120,
        height=240,
        page_number=1 if uid == "page-1" else 2,
        source_page_index=0 if uid == "page-1" else 1,
        status="imported",
        error="",
        image_hash=f"hash-{uid}-1",
        image_revision=1,
    )


def _full_session(project_uid: str = "project-a") -> ProjectSession:
    session = ProjectSession(
        ProjectRecord(project_uid=project_uid, name="Persistence V2"),
        save_path="/ui/state/not-domain-data.ocrproj",
    )

    first_page = session.page_repository.put(_page(project_uid), expected_revision=0)
    session.page_repository.put(
        replace(first_page, image_hash="hash-page-1-2", image_revision=2),
        expected_revision=1,
    )

    block = LayoutBlockSnapshot(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 2, 30, 40),
        order=0,
        source_label="text",
        origin=BlockOrigin(
            created_by=BlockSource.AUTO_LAYOUT.value,
            source_engine="layout-engine",
            source_run_id="layout-run-1",
            vendor_label="text",
            source_confidence=0.95,
            original_bbox=BBox(1, 2, 30, 40),
            original_kind=BlockType.TEXT,
            raw_artifact_uid="artifact-1",
            raw_json_path="records[0]",
            raw_index=0,
        ),
        ocr_policy=OcrPolicy.TEXT_OCR,
        authorship=BlockSource.AUTO_LAYOUT,
        uid="block-1",
    )
    session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=project_uid,
            uid="artifact-1",
            page_uid="page-1",
            source_engine="layout-engine",
            source_run_id="layout-run-1",
            image_hash="hash-page-1-2",
            payload_json='{"parsing_res_list": []}',
        )
    )
    session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=project_uid,
            uid="artifact-2",
            page_uid="page-1",
            source_engine="layout-engine",
            source_run_id="layout-run-2",
            image_hash="hash-page-1-2",
            payload_json='{"parsing_res_list": [{"block_label": "text"}]}',
        )
    )
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=1,
            artifact_uid="artifact-1",
            source_engine="layout-engine",
            source_run_id="layout-run-1",
            blocks=(block,),
        ),
        expected_revision=0,
    )
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=2,
            artifact_uid="artifact-2",
            source_engine="layout-engine",
            source_run_id="layout-run-2",
            blocks=(replace(block, source_label="body"),),
        ),
        expected_revision=1,
    )

    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=project_uid,
            uid="run-1",
            engine="ocr-engine",
            engine_version="1.0",
            layout_fingerprint="layout-fingerprint",
            input_fingerprint="input-fingerprint",
            metadata=(("language", "zh"),),
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=project_uid,
            uid="region-1",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(1, 2, 31, 42),
            kind="text",
            label="body",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=project_uid,
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="observed",
            bbox=(1, 2, 31, 12),
            confidence=0.91,
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
            scope_uid="page-1",
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=run.layout_fingerprint,
            region_uids=(region.uid,),
            line_uids=(line.uid,),
            atom_uids=(atom.uid,),
            candidate_uids=("candidate-1",),
        )
    )
    ocr.append_candidate(
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
            source="ocr-engine",
            bbox=(1, 2, 5, 12),
        )
    )
    ocr.switch_active_pointer(
        OcrActivePointer(
            project_uid=project_uid,
            uid="active-page-1",
            scope_uid="page-1",
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
            scope_uid="page-1",
            layout_fingerprint=run.layout_fingerprint,
            source_fingerprint=batch.fingerprint,
            anchor_revision=1,
            geometry_fingerprint="geometry-1",
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
                    revision=3,
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
            source_uid="block-1",
            target_uid="region-1",
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
            page_uid="page-1",
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


def _repository_snapshot(session: ProjectSession) -> tuple[object, ...]:
    ocr = session.ocr_observation_repository
    proof = session.proof_repository
    return (
        session.project_record,
        session.page_repository.all(),
        session.paddle_artifact_repository.all(),
        session.layout_repository.all(),
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


def test_new_project_session_roundtrip_persists_every_new_record_family(tmp_path: Path):
    path = tmp_path / "roundtrip.ocrproj"
    session = _full_session()

    save_session(path, session)
    loaded = load_session(path, expected_project_uid="project-a")

    assert _repository_snapshot(loaded) == _repository_snapshot(session)
    assert session.save_path == str(path)
    assert loaded.save_path == str(path)
    assert b"/ui/state/not-domain-data.ocrproj" not in path.read_bytes()
    with sqlite3.connect(path) as connection:
        generation = connection.execute(
            "SELECT generation FROM project_format WHERE singleton = 1"
        ).fetchone()[0]
        assert generation == FORMAT_GENERATION


def test_space_atom_without_geometry_roundtrips(tmp_path: Path) -> None:
    project_uid = "project-space"
    session = ProjectSession(ProjectRecord(project_uid, "Space"))
    session.page_repository.put(_page(project_uid), expected_revision=0)
    ocr = session.ocr_observation_repository
    run = ocr.append_run(OcrRun(
        project_uid=project_uid,
        uid="run-space",
        engine="ocr-engine",
        layout_fingerprint="layout",
        input_fingerprint="input",
    ))
    region = ocr.append_region(OcrRegion(
        project_uid=project_uid,
        uid="region-space",
        run_uid=run.uid,
        page_uid="page-1",
        bbox=(1, 2, 31, 42),
        kind="text",
    ))
    line = ocr.append_line(OcrLine(
        project_uid=project_uid,
        uid="line-space",
        run_uid=run.uid,
        region_uid=region.uid,
        page_uid="page-1",
        text=" ",
        bbox=(1, 2, 31, 12),
        confidence=0.0,
        atom_uids=("atom-space",),
    ))
    ocr.append_atom(OcrAtom(
        project_uid=project_uid,
        uid="atom-space",
        run_uid=run.uid,
        region_uid=region.uid,
        line_uid=line.uid,
        index=0,
        text=" ",
        bbox=None,
        confidence=0.0,
        source="hanwang:EngCut:latin_route",
        granularity="space",
        token_text=" ",
    ))
    path = tmp_path / "space.ocrproj"

    save_session(path, session)
    loaded = load_session(path, expected_project_uid=project_uid)

    atom = loaded.ocr_observation_repository.get_atom("atom-space")
    assert atom.text == " "
    assert atom.bbox is None
    assert atom.granularity == "space"


def _invalid_sqlite(path: Path, kind: str) -> None:
    with sqlite3.connect(path) as connection:
        if kind == "legacy":
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO meta VALUES ('schema_version', '25')")
        else:
            connection.execute(
                "CREATE TABLE project_format (singleton INTEGER PRIMARY KEY, generation TEXT)"
            )
            connection.execute(
                "INSERT INTO project_format VALUES (1, 'ocr_process.project_session.v3')"
            )
    timestamp_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(timestamp_ns, timestamp_ns))


@pytest.mark.parametrize("kind", ["legacy", "unknown"])
@pytest.mark.parametrize("operation", ["load", "save"])
def test_invalid_sqlite_is_rejected_without_changing_bytes_or_mtime(
    tmp_path: Path,
    kind: str,
    operation: str,
):
    path = tmp_path / f"{kind}-{operation}.ocrproj"
    _invalid_sqlite(path, kind)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    with pytest.raises(InvalidProjectFormatError):
        if operation == "load":
            load_session(path)
        else:
            save_session(path, ProjectSession("project-a"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert path.stat().st_mtime_ns == before_mtime


def test_save_rejects_a_different_project_uid_without_modifying_file(tmp_path: Path):
    path = tmp_path / "one-project.ocrproj"
    save_session(path, ProjectSession(ProjectRecord("project-a", "A")))
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    with pytest.raises(ProjectUidMismatchError):
        save_session(path, ProjectSession(ProjectRecord("project-b", "B")))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert path.stat().st_mtime_ns == before_mtime


def test_save_session_rolls_back_all_rows_when_one_write_fails(tmp_path: Path):
    path = tmp_path / "rollback.ocrproj"
    original_page = _page("project-a")
    original = ProjectSession.from_records(
        ProjectRecord("project-a", "Original"),
        pages=(original_page,),
    )
    save_session(path, original)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_second_page
            BEFORE INSERT ON page
            WHEN NEW.uid = 'page-2'
            BEGIN
                SELECT RAISE(ABORT, 'forced page failure');
            END
            """
        )

    changed = ProjectSession.from_records(
        ProjectRecord("project-a", "Changed before failure"),
        pages=(original_page, _page("project-a", "page-2")),
        save_path="/ui/unchanged-after-failure.ocrproj",
    )
    with pytest.raises(ProjectStoreDataError):
        save_session(path, changed)

    loaded = load_session(path)
    assert loaded.project_record.name == "Original"
    assert loaded.page_repository.all() == (original_page,)
    assert changed.save_path == "/ui/unchanged-after-failure.ocrproj"
