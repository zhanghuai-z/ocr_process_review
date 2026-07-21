from dataclasses import FrozenInstanceError, fields, replace

import pytest

import app.models.project_session as project_session_module
from app.models.layout_snapshot import LayoutSnapshot
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrCandidate,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.models.project_session import (
    BindingRepository,
    DuplicateUidError,
    FingerprintConflictError,
    LayoutRepository,
    OcrObservationRepository,
    PageRecord,
    PageRepository,
    ProjectScopeError,
    ProjectRecord,
    ProjectSession,
    RecordNotFoundError,
    RevisionConflictError,
    TableTextRepository,
)


def _run(project_uid: str, uid: str, *, input_fingerprint: str = "input-a") -> OcrRun:
    return OcrRun(
        project_uid=project_uid,
        uid=uid,
        engine="test-engine",
        layout_fingerprint="layout-a",
        input_fingerprint=input_fingerprint,
    )


def _batch(
    project_uid: str,
    uid: str,
    run: OcrRun,
    *,
    scope_uid: str = "page-1",
) -> OcrBatch:
    return OcrBatch(
        project_uid=project_uid,
        uid=uid,
        run_uid=run.uid,
        scope_uid=scope_uid,
        input_fingerprint=run.input_fingerprint,
        layout_fingerprint=run.layout_fingerprint,
    )


def _layout(page_uid: str, revision: int, artifact_uid: str) -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid=page_uid,
        revision=revision,
        artifact_uid=artifact_uid,
        source_engine="test-layout",
        source_run_id=artifact_uid,
        blocks=(),
    )


def _page(project_uid: str, *, uid: str = "page-1", image_revision: int = 1) -> PageRecord:
    return PageRecord(
        project_uid=project_uid,
        uid=uid,
        image_path=f"/images/{uid}.png",
        source_path=f"/sources/{uid}.pdf",
        cache_image_path=f"/cache/{uid}.png",
        thumbnail_path=f"/thumbs/{uid}.png",
        width=100,
        height=200,
        page_number=1,
        source_page_index=1,
        status="imported",
        error="",
        image_hash=f"hash-{image_revision}",
        image_revision=image_revision,
    )


def _anchor(
    project_uid: str,
    *,
    source_fingerprint: str = "source-a",
    layout_fingerprint: str = "layout-a",
    anchor_revision: int = 1,
) -> ProofAnchorSnapshot:
    return ProofAnchorSnapshot(
        project_uid=project_uid,
        uid="anchor-1",
        scope_uid="page-1",
        layout_fingerprint=layout_fingerprint,
        source_fingerprint=source_fingerprint,
        anchor_revision=anchor_revision,
    )


def _segment(project_uid: str, anchor: ProofAnchorSnapshot) -> ProofAlignmentSegment:
    return ProofAlignmentSegment(
        project_uid=project_uid,
        uid="segment-1",
        anchor_uid=anchor.uid,
        source_start=0,
        source_end=1,
        proof_start=0,
        proof_end=1,
    )


def _slice(project_uid: str, segment: ProofAlignmentSegment) -> ProofAlignmentSlice:
    return ProofAlignmentSlice(
        project_uid=project_uid,
        uid="slice-1",
        segment_uid=segment.uid,
        source_start=0,
        source_end=1,
        proof_start=0,
        proof_end=1,
        source_text="a",
        proof_text="a",
    )


def test_project_session_owns_project_scoped_repositories_and_layout_truth():
    first = ProjectSession(
        ProjectRecord(project_uid="project-a", name="Project A"),
        save_path="/projects/project-a.ocrproj",
    )
    second = ProjectSession("project-b")

    assert isinstance(first.project_record, ProjectRecord)
    assert first.project_record.name == "Project A"
    assert first.save_path == "/projects/project-a.ocrproj"
    first.save_path = "/projects/project-a-copy.ocrproj"
    assert first.save_path.endswith("copy.ocrproj")
    assert "save_path" not in {item.name for item in fields(first.project_record)}
    with pytest.raises(FrozenInstanceError):
        first.project_record.name = "mutated"  # type: ignore[misc]

    assert isinstance(first.layout_repository, LayoutRepository)
    assert isinstance(first.page_repository, PageRepository)
    assert isinstance(first.ocr_observation_repository, OcrObservationRepository)
    assert isinstance(first.binding_repository, BindingRepository)
    assert isinstance(first.table_text_repository, TableTextRepository)
    assert first.layout_repository.project_uid == "project-a"
    assert first.page_repository.project_uid == "project-a"
    assert second.layout_repository.project_uid == "project-b"

    page = first.page_repository.put(_page("project-a"), expected_revision=0)
    assert first.page_repository.get("page-1", image_revision=1, fingerprint=page.fingerprint) == page
    assert "blocks" not in {item.name for item in fields(page)}
    next_page = first.page_repository.put(
        replace(page, image_hash="hash-2", image_revision=2),
        expected_revision=1,
    )
    assert next_page.image_revision == 2
    with pytest.raises(RevisionConflictError):
        first.page_repository.put(
            replace(next_page, image_hash="hash-3", image_revision=3),
            expected_revision=1,
        )
    with pytest.raises(ProjectScopeError):
        first.page_repository.put(_page("project-b"), expected_revision=0)
    second_page = second.page_repository.put(_page("project-b"), expected_revision=0)
    assert second_page.project_uid != page.project_uid

    first_snapshot = first.layout_repository.put(
        _layout("page-1", 1, "artifact-a1"),
        expected_revision=0,
    )
    assert first.layout_repository.get("page-1") == first_snapshot
    with pytest.raises(RecordNotFoundError):
        second.layout_repository.get("page-1")

    second_snapshot = second.layout_repository.put(
        _layout("page-1", 1, "artifact-b1"),
        expected_revision=0,
    )
    assert second_snapshot.artifact_uid != first_snapshot.artifact_uid

    first_run = first.ocr_observation_repository.append_run(_run("project-a", "run-1"))
    with pytest.raises(RecordNotFoundError):
        second.ocr_observation_repository.get_run("run-1")
    with pytest.raises(ProjectScopeError):
        second.ocr_observation_repository.append_run(first_run)
    second_run = second.ocr_observation_repository.append_run(_run("project-b", "run-1"))
    assert second_run.project_uid != first_run.project_uid


def test_layout_repository_requires_sequential_snapshot_revision_and_no_layout_record():
    repository = LayoutRepository("project-a")
    first = repository.put(_layout("page-1", 1, "artifact-1"), expected_revision=0)

    with pytest.raises(RevisionConflictError):
        repository.put(_layout("page-1", 3, "artifact-3"), expected_revision=1)
    with pytest.raises(RevisionConflictError):
        repository.put(_layout("page-1", 2, "artifact-2"), expected_revision=0)

    second = repository.put(_layout("page-1", 2, "artifact-2"), expected_revision=1)
    fingerprint = repository.fingerprint_for(second)
    assert repository.get("page-1", revision=2, fingerprint=fingerprint) == second
    assert repository.fingerprint_for(first) != fingerprint
    assert not hasattr(repository, "replace")
    assert not hasattr(project_session_module, "LayoutRecord")


def test_ocr_facts_are_immutable_append_only_and_have_no_revision_or_replace_api():
    session = ProjectSession("project-a")
    repository = session.ocr_observation_repository
    run = repository.append_run(_run("project-a", "run-1"))
    region = repository.append_region(
        OcrRegion(
            project_uid="project-a",
            uid="region-1",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(0, 0, 10, 10),
            kind="text",
        )
    )
    line = repository.append_line(
        OcrLine(
            project_uid="project-a",
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="a",
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            atom_uids=("atom-1",),
        )
    )
    atom = repository.append_atom(
        OcrAtom(
            project_uid="project-a",
            uid="atom-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=0,
            text="a",
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            candidate_uids=("candidate-1",),
            source="test",
            granularity="char",
            token_text="a",
        )
    )
    batch = repository.append_batch(_batch("project-a", "batch-1", run))
    candidate = repository.append_candidate(
        OcrCandidate(
            project_uid="project-a",
            uid="candidate-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            atom_uid=atom.uid,
            batch_uid=batch.uid,
            text="a",
            confidence=0.9,
            rank=0,
            source="test-engine",
        )
    )

    for record in (run, region, line, atom, candidate, batch):
        assert "revision" not in {item.name for item in fields(record)}
        assert record.fingerprint
    with pytest.raises(FrozenInstanceError):
        run.engine = "mutated"  # type: ignore[misc]
    assert isinstance(batch.region_uids, tuple)
    assert repository.get_run(run.uid, fingerprint=run.fingerprint) == run
    assert repository.get_candidate(candidate.uid, fingerprint=candidate.fingerprint) == candidate

    assert not hasattr(repository, "create")
    assert not hasattr(repository, "replace")
    assert not hasattr(repository, "replace_run")
    with pytest.raises(DuplicateUidError):
        repository.append_run(replace(run, engine="another-engine"))


def test_only_active_pointer_supports_cas_switch_by_pointer_revision_and_fingerprint():
    repository = ProjectSession("project-a").ocr_observation_repository
    run_a = repository.append_run(_run("project-a", "run-a"))
    run_b = repository.append_run(_run("project-a", "run-b", input_fingerprint="input-b"))
    batch_a = repository.append_batch(_batch("project-a", "batch-a", run_a))
    batch_b = repository.append_batch(_batch("project-a", "batch-b", run_b))

    pointer = repository.switch_active_pointer(
        OcrActivePointer(
            project_uid="project-a",
            uid="active-page-1",
            scope_uid="page-1",
            batch_uid=batch_a.uid,
            run_uid=run_a.uid,
            batch_fingerprint=batch_a.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )
    next_pointer = replace(
        pointer,
        batch_uid=batch_b.uid,
        run_uid=run_b.uid,
        batch_fingerprint=batch_b.fingerprint,
        revision=2,
    )

    with pytest.raises(RevisionConflictError):
        repository.switch_active_pointer(
            next_pointer,
            expected_revision=0,
            expected_fingerprint=None,
        )
    with pytest.raises(FingerprintConflictError):
        repository.switch_active_pointer(
            next_pointer,
            expected_revision=pointer.revision,
            expected_fingerprint="stale-fingerprint",
        )

    active = repository.switch_active_pointer(
        next_pointer,
        expected_revision=pointer.revision,
        expected_fingerprint=pointer.fingerprint,
    )
    assert active.revision == 2
    assert repository.get_active_pointer("page-1") == active
    assert repository.get_active_pointer_by_uid(active.uid) == active
    assert not hasattr(repository, "create_active_pointer")
    assert not hasattr(repository, "replace_active_pointer")


def test_proof_rebind_marks_old_alignment_stale_then_requires_explicit_rebind():
    repository = ProjectSession("project-a").proof_repository
    anchor = _anchor("project-a")
    segment = _segment("project-a", anchor)
    alignment_slice = _slice("project-a", segment)
    state = repository.create_state(
        ProofState(
            project_uid="project-a",
            uid="proof-1",
            anchor_snapshot=anchor,
            text_units=(
                ProofTextUnit(
                    project_uid="project-a",
                    uid="proof-text-1",
                    order=0,
                    text="人工终稿",
                    status="modified",
                ),
            ),
            alignment_segments=(segment,),
            alignment_slices=(alignment_slice,),
        )
    )

    new_anchor = _anchor(
        "project-a",
        source_fingerprint="source-b",
        layout_fingerprint="layout-b",
        anchor_revision=2,
    )
    marked = repository.mark_rebind_required(
        state.uid,
        new_anchor,
        expected_revision=state.revision,
        expected_fingerprint=state.fingerprint,
    )
    assert marked.rebind_required is True
    assert marked.alignment_segments == ()
    assert marked.alignment_slices == ()
    assert tuple(item.text for item in marked.text_units) == ("人工终稿",)
    assert marked.anchor_snapshot.source_fingerprint == "source-b"

    rebound_segment = _segment("project-a", new_anchor)
    rebound_slice = _slice("project-a", rebound_segment)
    rebound = repository.rebind(
        marked.uid,
        new_anchor,
        (rebound_segment,),
        (rebound_slice,),
        expected_revision=marked.revision,
        expected_fingerprint=marked.fingerprint,
    )
    assert rebound.rebind_required is False
    assert rebound.revision == marked.revision + 1
    assert rebound.text_units == marked.text_units
    assert rebound.alignment_segments == (rebound_segment,)
    assert rebound.alignment_slices == (rebound_slice,)
