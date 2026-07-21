from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from app.core.layout_scope import layout_snapshot_fingerprint
from app.application.proof_workspace import (
    ProofAtomView,
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspaceView,
    build_proof_workspace_view,
    query_proof_workspace,
)
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.layout_snapshot import LayoutSnapshot
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.models.project_session import PageRecord, ProjectSession


PROJECT_UID = "proof-application-project"


def _page(uid: str, page_number: int) -> PageRecord:
    return PageRecord(
        project_uid=PROJECT_UID,
        uid=uid,
        image_path=f"{uid}.png",
        source_path="book.pdf",
        cache_image_path=f"{uid}.cache.png",
        thumbnail_path=f"{uid}.thumb.png",
        width=200,
        height=100,
        page_number=page_number,
        source_page_index=page_number - 1,
        status="imported",
        error="",
        image_hash=f"hash-{uid}",
        image_revision=1,
    )


def _session(
    *,
    proof_text: str = "ac",
    source_end: int = 2,
) -> tuple[ProjectSession, ProofState, OcrBatch, OcrActivePointer]:
    session = ProjectSession(PROJECT_UID)
    session.page_repository.put(_page("page-1", 1), expected_revision=0)
    session.page_repository.put(_page("page-2", 2), expected_revision=0)
    page = session.page_repository.get("page-1")
    artifact = PaddleArtifact(
        project_uid=PROJECT_UID,
        uid="artifact-1",
        page_uid=page.uid,
        source_engine="test",
        source_run_id="layout-run-1",
        image_hash=page.image_hash,
        payload_json="{}",
    )
    session.paddle_artifact_repository.append(artifact)
    layout = LayoutSnapshot(
        page_uid=page.uid,
        revision=1,
        artifact_uid=artifact.uid,
        source_engine="test",
        source_run_id="layout-run-1",
        blocks=(),
    )
    session.layout_repository.put(layout, expected_revision=0)
    layout_fingerprint = layout_snapshot_fingerprint(layout)

    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=PROJECT_UID,
            uid="run-1",
            engine="test-engine",
            layout_fingerprint=layout_fingerprint,
            input_fingerprint="input-1",
            metadata=(("page_fingerprint", page.fingerprint),),
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=PROJECT_UID,
            uid="region-1",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(0, 0, 100, 60),
            kind="text",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=PROJECT_UID,
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="ab",
            bbox=(10, 10, 50, 30),
            confidence=0.90,
            order=0,
            atom_uids=("atom-1", "atom-2"),
        )
    )
    atom_a = ocr.append_atom(
        OcrAtom(
            project_uid=PROJECT_UID,
            uid="atom-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=0,
            text="a",
            bbox=(10, 10, 30, 30),
            confidence=0.96,
            source="ocr:test",
            granularity="char",
            token_text="a",
        )
    )
    atom_b = ocr.append_atom(
        OcrAtom(
            project_uid=PROJECT_UID,
            uid="atom-2",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=1,
            text="b",
            bbox=(30, 10, 50, 30),
            confidence=0.72,
            source="ocr:test",
            granularity="char",
            token_text="b",
        )
    )
    batch = ocr.append_batch(
        OcrBatch(
            project_uid=PROJECT_UID,
            uid="batch-1",
            run_uid=run.uid,
            scope_uid="page-1",
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=run.layout_fingerprint,
            region_uids=(region.uid,),
            line_uids=(line.uid,),
            atom_uids=(atom_a.uid, atom_b.uid),
        )
    )
    pointer = ocr.switch_active_pointer(
        OcrActivePointer(
            project_uid=PROJECT_UID,
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
    anchor = ProofAnchorSnapshot(
        project_uid=PROJECT_UID,
        uid="anchor-1",
        scope_uid="page-1",
        layout_fingerprint=batch.layout_fingerprint,
        source_fingerprint=batch.fingerprint,
        anchor_revision=1,
    )
    segment = ProofAlignmentSegment(
        project_uid=PROJECT_UID,
        uid="segment-1",
        anchor_uid=anchor.uid,
        source_start=0,
        source_end=source_end,
        proof_start=0,
        proof_end=len(proof_text),
    )
    alignment = ProofAlignmentSlice(
        project_uid=PROJECT_UID,
        uid="slice-1",
        segment_uid=segment.uid,
        source_start=0,
        source_end=source_end,
        proof_start=0,
        proof_end=len(proof_text),
        source_text="ab"[:source_end],
        proof_text=proof_text,
    )
    state = session.proof_repository.create_state(
        ProofState(
            project_uid=PROJECT_UID,
            uid="proof-1",
            anchor_snapshot=anchor,
            text_units=(
                ProofTextUnit(
                    project_uid=PROJECT_UID,
                    uid="unit-1",
                    order=0,
                    text=proof_text,
                    status="modified",
                ),
            ),
            alignment_segments=(segment,),
            alignment_slices=(alignment,),
        )
    )
    return session, state, batch, pointer


def _assert_dto_graph_is_value_only(value: object) -> None:
    if is_dataclass(value):
        assert not hasattr(value, "session")
        assert not hasattr(value, "repository")
        assert not hasattr(value, "service")
        for field in fields(value):
            _assert_dto_graph_is_value_only(getattr(value, field.name))
        return
    if isinstance(value, tuple):
        for item in value:
            _assert_dto_graph_is_value_only(item)
        return
    assert not isinstance(value, (dict, list, set, bytearray))


def test_query_projects_active_ocr_and_proof_state_into_stable_read_views() -> None:
    session, state, batch, pointer = _session()

    view = build_proof_workspace_view(session)

    assert isinstance(view, ProofWorkspaceView)
    assert view.project_uid == PROJECT_UID
    assert tuple(page.page_uid for page in view.pages) == ("page-1", "page-2")
    assert view.pages[0].image_path == "page-1.png"
    assert not hasattr(view.pages[0], "proof_text")

    state_view = view.proof_states[0]
    assert isinstance(state_view, ProofStateView)
    assert state_view.proof_state_uid == state.uid
    assert state_view.active_batch_uid == batch.uid
    assert state_view.active_pointer_uid == pointer.uid
    assert state_view.revision == state.revision
    assert state_view.fingerprint == state.fingerprint
    assert state_view.rebind_required is False

    unit_view = state_view.text_units[0]
    assert isinstance(unit_view, ProofTextUnitView)
    assert unit_view.text_unit_uid == "unit-1"
    assert unit_view.text == "ac"
    assert unit_view.status == "modified"
    assert unit_view.revision == state.text_units[0].revision

    line = view.lines[0]
    assert isinstance(line, ProofLineView)
    assert line.proof_state_uid == "proof-1"
    assert line.line_uid == "line-1"
    assert line.region_uid == "region-1"
    assert line.text_unit_uid == "unit-1"
    assert line.ocr_text == "ab"
    assert line.proof_text == "ac"
    assert line.confidence == pytest.approx(0.84)
    assert line.bbox == (10, 10, 50, 30)
    assert tuple(atom.atom_uid for atom in line.atoms) == ("atom-1", "atom-2")

    atom = line.atoms[0]
    assert isinstance(atom, ProofAtomView)
    assert atom.page_uid == "page-1"
    assert atom.region_uid == "region-1"
    assert atom.line_uid == "line-1"
    assert atom.atom_uid == "atom-1"
    assert atom.text == "a"
    assert atom.bbox == (10, 10, 30, 30)
    assert atom.source == "ocr:test"
    assert atom.granularity == "char"
    assert atom.token_text == "a"
    assert atom.render_kind == "text"
    assert atom.char_span == (0, 1)
    assert atom.geometry_available is True
    assert view.atoms == line.atoms
    _assert_dto_graph_is_value_only(view)


def test_line_view_keeps_complete_mapped_line_atoms_with_explicit_unmapped_metadata() -> None:
    session, state, _batch, _pointer = _session(proof_text="a", source_end=1)

    view = build_proof_workspace_view(session, proof_uid=state.uid)
    line = view.lines[0]

    assert line.line_uid == "line-1"
    assert line.ocr_text == "ab"
    assert tuple(atom.atom_uid for atom in line.atoms) == ("atom-1", "atom-2")
    assert line.atoms[0].char_span == (0, 1)
    assert line.atoms[0].geometry_available is True
    assert line.atoms[1].char_span is None
    assert line.atoms[1].geometry_available is False
    assert line.atoms[1].source == "ocr:test"
    assert line.atoms[1].granularity == "char"
    assert line.atoms[1].token_text == "b"


def test_query_hides_historical_proof_after_layout_changes() -> None:
    session, state, _batch, _pointer = _session()
    current = session.layout_repository.get("page-1")
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=current.page_uid,
            revision=2,
            artifact_uid=current.artifact_uid,
            source_engine=current.source_engine,
            source_run_id=current.source_run_id,
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=BBox(1, 1, 50, 30),
                    order=0,
                    source_label="text",
                    origin=BlockOrigin(created_by="manual_edit"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    authorship=BlockSource.USER_EDITED,
                    uid="changed-block",
                ),
            ),
        ),
        expected_revision=1,
    )

    view = build_proof_workspace_view(session)

    assert session.proof_repository.get_state(state.uid) == state
    assert view.proof_states == ()
    assert view.lines == ()


def test_query_is_read_only_and_exposes_cas_tokens_without_repository_handles() -> None:
    session, state, _batch, _pointer = _session()
    before_state = session.proof_repository.get_state(state.uid)
    before_pointer = session.ocr_observation_repository.get_active_pointer("page-1")

    first = query_proof_workspace(session, proof_uid=state.uid)
    second = query_proof_workspace(session, proof_uid=state.uid)

    assert first == second
    assert first.proof_states[0].fingerprint == before_state.fingerprint
    assert first.lines[0].state_fingerprint == before_state.fingerprint
    assert first.lines[0].text_unit_fingerprint == before_state.text_units[0].fingerprint
    assert session.proof_repository.get_state(state.uid) == before_state
    assert session.ocr_observation_repository.get_active_pointer("page-1") == before_pointer

    with pytest.raises(FrozenInstanceError):
        first.lines[0].proof_text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.pages[0].image_path = "changed.png"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.lines[0].atoms[0] = first.lines[0].atoms[0]  # type: ignore[index]


def test_query_does_not_reuse_old_line_or_atom_when_active_observation_changes() -> None:
    session, state, _batch, pointer = _session()
    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=PROJECT_UID,
            uid="run-2",
            engine="test-engine",
            layout_fingerprint=layout_snapshot_fingerprint(
                session.layout_repository.get("page-1")
            ),
            input_fingerprint="input-2",
            metadata=((
                "page_fingerprint",
                session.page_repository.get("page-1").fingerprint,
            ),),
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=PROJECT_UID,
            uid="region-2",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(0, 0, 100, 60),
            kind="text",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=PROJECT_UID,
            uid="line-2",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="xy",
            bbox=(10, 10, 50, 30),
            confidence=0.8,
            atom_uids=("atom-3", "atom-4"),
        )
    )
    atoms = (
        ocr.append_atom(
            OcrAtom(
                project_uid=PROJECT_UID,
                uid="atom-3",
                run_uid=run.uid,
                region_uid=region.uid,
                line_uid=line.uid,
                index=0,
                text="x",
                bbox=(10, 10, 30, 30),
                confidence=0.8,
                source="test",
                granularity="char",
                token_text="x",
            )
        ),
        ocr.append_atom(
            OcrAtom(
                project_uid=PROJECT_UID,
                uid="atom-4",
                run_uid=run.uid,
                region_uid=region.uid,
                line_uid=line.uid,
                index=1,
                text="y",
                bbox=(30, 10, 50, 30),
                confidence=0.8,
                source="test",
                granularity="char",
                token_text="y",
            )
        ),
    )
    batch = ocr.append_batch(
        OcrBatch(
            project_uid=PROJECT_UID,
            uid="batch-2",
            run_uid=run.uid,
            scope_uid="page-1",
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=run.layout_fingerprint,
            region_uids=(region.uid,),
            line_uids=(line.uid,),
            atom_uids=tuple(atom.uid for atom in atoms),
        )
    )
    ocr.switch_active_pointer(
        OcrActivePointer(
            project_uid=PROJECT_UID,
            uid=pointer.uid,
            scope_uid=pointer.scope_uid,
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=pointer.revision + 1,
        ),
        expected_revision=pointer.revision,
        expected_fingerprint=pointer.fingerprint,
    )

    view = build_proof_workspace_view(session, proof_uid=state.uid)
    line_view = view.lines[0]

    assert view.proof_states[0].active_batch_uid == "batch-2"
    assert line_view.proof_text == "ac"
    assert line_view.line_uid is None
    assert line_view.ocr_text == ""
    assert line_view.atoms == ()
