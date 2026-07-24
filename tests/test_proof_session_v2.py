from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.char_index import GEOMETRY_UNAVAILABLE
from app.models.layout_snapshot import LayoutSnapshot
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.project_session import PageRecord, ProjectSession, RevisionConflictError
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.core.proof_session import ProofSpanReplacement
from app.services.char_index_service import CharIndexService
from app.services.proof_session_service import ProofSessionService


ROOT = Path(__file__).resolve().parents[1]
SCOPED = tuple(
    sorted(
        [
            *((ROOT / "app/services").glob("proof_*.py")),
            ROOT / "app/services/char_index_service.py",
            *((ROOT / "app/core").glob("proof_*.py")),
            ROOT / "app/core/char_index.py",
        ]
    )
)


def _session() -> tuple[ProjectSession, ProofState]:
    project_uid = "project-proof-v2"
    session = ProjectSession(project_uid)
    session.page_repository.put(
        PageRecord(
            project_uid=project_uid,
            uid="page-1",
            image_path="page-1.png",
            source_path="book.pdf",
            cache_image_path="page-1.png",
            thumbnail_path="page-1-thumb.png",
            width=200,
            height=100,
            page_number=1,
            source_page_index=0,
            status="imported",
            error="",
            image_hash="page-hash",
            image_revision=1,
        ),
        expected_revision=0,
    )
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=1,
            artifact_uid="layout-1",
            source_engine="layout-test",
            source_run_id="layout-run-1",
            blocks=(),
        ),
        expected_revision=0,
    )

    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=project_uid,
            uid="run-1",
            engine="ocr-test",
            layout_fingerprint="layout-1",
            input_fingerprint="input-1",
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=project_uid,
            uid="region-1",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(0, 0, 100, 50),
            kind="text",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=project_uid,
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="ab",
            bbox=(10, 10, 40, 30),
            confidence=0.9,
            atom_uids=("atom-1", "atom-2"),
        )
    )
    atom_a = ocr.append_atom(
        OcrAtom(
            project_uid=project_uid,
            uid="atom-1",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=0,
            text="a",
            bbox=(10, 10, 20, 30),
            confidence=0.9,
            source="test",
            granularity="char",
            token_text="a",
        )
    )
    atom_b = ocr.append_atom(
        OcrAtom(
            project_uid=project_uid,
            uid="atom-2",
            run_uid=run.uid,
            region_uid=region.uid,
            line_uid=line.uid,
            index=1,
            text="b",
            bbox=(20, 10, 30, 30),
            confidence=0.8,
            source="test",
            granularity="char",
            token_text="b",
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
            atom_uids=(atom_a.uid, atom_b.uid),
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

    anchor = ProofAnchorSnapshot(
        project_uid=project_uid,
        uid="anchor-1",
        scope_uid="page-1",
        layout_fingerprint=run.layout_fingerprint,
        source_fingerprint=batch.fingerprint,
        anchor_revision=1,
    )
    segment = ProofAlignmentSegment(
        project_uid=project_uid,
        uid="segment-1",
        anchor_uid=anchor.uid,
        text_unit_uid="unit-1",
        source_line_uids=("line-1",),
        source_start=0,
        source_end=2,
        proof_start=0,
        proof_end=2,
    )
    alignment = ProofAlignmentSlice(
        project_uid=project_uid,
        uid="slice-1",
        segment_uid=segment.uid,
        source_start=0,
        source_end=2,
        proof_start=0,
        proof_end=2,
        source_text="ab",
        proof_text="ac",
    )
    state = ProofState(
        project_uid=project_uid,
        uid="proof-1",
        anchor_snapshot=anchor,
        text_units=(
            ProofTextUnit(
                project_uid=project_uid,
                uid="unit-1",
                order=0,
                text="ac",
            ),
            ProofTextUnit(
                project_uid=project_uid,
                uid="unit-2",
                order=1,
                text="z",
            ),
        ),
        alignment_segments=(segment,),
        alignment_slices=(alignment,),
    )
    return session, state


def test_cross_state_text_batch_is_atomic_when_later_cas_fails() -> None:
    session, state = _session()
    service = ProofSessionService(session)
    first = service.create_state(state).state
    second_unit = replace(state.text_units[0], uid="unit-3", text="de")
    second_state = replace(
        state,
        uid="proof-2",
        anchor_snapshot=replace(state.anchor_snapshot, uid="anchor-2"),
        text_units=(second_unit,),
        alignment_segments=(),
        alignment_slices=(),
    )
    second = service.create_state(second_state).state

    with pytest.raises(RevisionConflictError):
        service.replace_text_units_across_states(
            (
                (
                    first.uid,
                    (("unit-1", "xc"),),
                    first.revision,
                    first.fingerprint,
                    "modified",
                ),
                (
                    second.uid,
                    (("unit-3", "ye"),),
                    second.revision + 1,
                    second.fingerprint,
                    "modified",
                ),
            )
        )

    assert service.get_state(first.uid).text_units[0].text == "ac"
    assert service.get_state(second.uid).text_units[0].text == "de"

    results = service.replace_text_units_across_states(
        (
            (
                first.uid,
                (("unit-1", "xc"),),
                first.revision,
                first.fingerprint,
                "modified",
            ),
            (
                second.uid,
                (("unit-3", "ye"),),
                second.revision,
                second.fingerprint,
                "modified",
            ),
        )
    )
    assert all(result.changed for result in results)
    assert service.get_state(first.uid).text_units[0].text == "xc"
    assert service.get_state(second.uid).text_units[0].text == "ye"


def test_proof_scope_has_one_write_service_and_no_retired_entry_points() -> None:
    assert (ROOT / "app/services/proof_session_service.py").exists()
    for retired in (
        "proof_edit_service.py",
        "proof_persistence_service.py",
        "proof_hproof_session.py",
        "proof_occurrence_session.py",
        "proof_probe_text_service.py",
        "proof_external_refresh.py",
    ):
        assert not (ROOT / "app/services" / retired).exists()
    for retired in ("proof_projection.py", "proof_state_bus.py", "proof_line_mutation.py"):
        assert not (ROOT / "app/core" / retired).exists()

    forbidden_imports = {"Page", "Block", "Line", "Char", "OcrProject"}
    forbidden_names = {
        "runtime_block",
        "ProofStateBus",
        "update_line",
        "final_text",
        "sidecar",
    }
    for path in SCOPED:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app.models"):
                imported = {alias.name for alias in node.names}
                assert not imported & forbidden_imports, (path, imported)
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names, (path, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, (path, node.attr)
    service_source = (ROOT / "app/services/proof_session_service.py").read_text(encoding="utf-8")
    assert "proof_repository.rebind" not in service_source
    assert "create_anchor_snapshot" not in service_source
    assert "get_anchor_snapshot" not in service_source
    assert "create_alignment_segment" not in service_source
    assert "get_alignment_slice" not in service_source


def test_proof_writes_use_aggregate_cas_and_ocr_is_not_mutated() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    unit = created.state.text_units[0]
    changed = service.replace_text(
        initial.uid,
        unit.uid,
        "edited",
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
        expected_unit_revision=unit.revision,
        expected_unit_fingerprint=unit.fingerprint,
    )
    assert changed.changed is True
    assert changed.state.text_units[0].text == "edited"
    assert session.ocr_observation_repository.get_line("line-1").text == "ab"
    with pytest.raises(RevisionConflictError):
        service.set_status(
            initial.uid,
            unit.uid,
            "checked",
            expected_revision=created.revision,
            expected_fingerprint=created.fingerprint,
        )


def test_equal_length_text_edit_updates_explicit_alignment() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    unit = created.state.text_units[0]

    changed = service.replace_text(
        initial.uid,
        unit.uid,
        "ax",
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
        expected_unit_revision=unit.revision,
        expected_unit_fingerprint=unit.fingerprint,
    )

    assert changed.state.alignment_slices[0].proof_text == "ax"
    index = service.build_char_index(changed.state.uid)
    assert [entry.available for entry in index.entries[:2]] == [True, True]


def test_length_change_retires_only_changed_unit_positional_slices() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    unit = created.state.text_units[0]

    changed = service.replace_text(
        initial.uid,
        unit.uid,
        "formula",
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
        expected_unit_revision=unit.revision,
        expected_unit_fingerprint=unit.fingerprint,
    )

    assert changed.state.alignment_slices == ()
    assert changed.state.alignment_segments[0].proof_end == len("formula")


def test_batch_status_and_five_step_undo_redo_are_session_operations() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    current = service.create_state(initial)
    replacement = service.replace_text_units(
        initial.uid,
        {"unit-1": "one", "unit-2": "two"},
        expected_revision=current.revision,
        expected_fingerprint=current.fingerprint,
    )
    assert replacement.changed_text_unit_uids == ("unit-1", "unit-2")
    current = replacement
    status = service.set_status(
        initial.uid,
        "unit-1",
        "checked",
        expected_revision=current.revision,
        expected_fingerprint=current.fingerprint,
    )
    assert status.state.text_units[0].status == "checked"

    current_state = status.state
    for value in ("b", "c", "d", "e", "f", "g"):
        result = service.replace_text(
            initial.uid,
            "unit-1",
            value,
            expected_revision=current_state.revision,
            expected_fingerprint=current_state.fingerprint,
            status=None,
        )
        current_state = result.state
    for _ in range(5):
        current_state = service.undo(initial.uid).state
    assert current_state.text_units[0].text == "b"
    assert service.can_undo(initial.uid) is False
    for _ in range(5):
        current_state = service.redo(initial.uid).state
    assert current_state.text_units[0].text == "g"


def test_span_replacement_is_one_cas_mutation() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    result = service.replace_spans(
        initial.uid,
        "unit-1",
        (ProofSpanReplacement(0, 1, "x"),),
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
    )
    assert result.changed is True
    assert result.state.text_units[0].text == "xc"
def test_editor_refresh_preserves_clean_rebind_and_reports_dirty_conflict() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    snapshot = service.editor_snapshot(initial.uid, "unit-1")
    external = service.replace_text(
        initial.uid,
        "unit-1",
        "external",
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
    )
    conflict = service.refresh_editor(snapshot, editor_text="local")
    assert conflict.conflict is True
    rebound = service.rebind_editor(snapshot)
    assert rebound.snapshot is not None
    assert rebound.snapshot.state_revision == external.revision
    assert rebound.snapshot.text == "external"


def test_index_uses_proof_alignment_and_atom_geometry_only() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    index = service.build_char_index(
        initial.uid,
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
    )
    first = index.query("a")
    second = index.query("c")
    unavailable = index.query("z", include_unavailable=True)
    assert first[0].bbox == (10, 10, 20, 30)
    assert second[0].bbox == (20, 10, 30, 30)
    assert unavailable[0].geometry_status == GEOMETRY_UNAVAILABLE
    assert unavailable[0].bbox is None
    assert unavailable[0].unavailable_reason == "no_alignment_for_proof_position"
    assert service.same_text(initial.uid, "c")[0].atom_uid == "atom-2"


def test_index_keeps_whitespace_text_but_marks_its_geometry_unavailable() -> None:
    session, initial = _session()
    ocr = session.ocr_observation_repository
    page = session.page_repository.get("page-1")
    original_batch = ocr.get_batch("batch-1")
    original_line = ocr.get_line("line-1")
    atom_a = ocr.get_atom("atom-1")
    atom_b = replace(ocr.get_atom("atom-2"), index=2)
    space = OcrAtom(
        project_uid=initial.project_uid,
        uid="atom-space",
        run_uid=atom_a.run_uid,
        region_uid=atom_a.region_uid,
        line_uid=original_line.uid,
        index=1,
        text=" ",
        bbox=original_line.bbox,
        confidence=0.0,
        source="hanwang:EngCut:latin_route",
        granularity="space",
        token_text=" ",
    )
    line = replace(
        original_line,
        text="a b",
        atom_uids=(atom_a.uid, space.uid, atom_b.uid),
    )
    batch = replace(
        original_batch,
        atom_uids=(atom_a.uid, space.uid, atom_b.uid),
    )
    unit = replace(initial.text_units[0], text="a b")
    segment = replace(
        initial.alignment_segments[0],
        source_end=3,
        proof_end=3,
    )
    alignment = replace(
        initial.alignment_slices[0],
        source_end=3,
        proof_end=3,
        source_text="a b",
        proof_text="a b",
    )
    state = replace(
        initial,
        anchor_snapshot=replace(
            initial.anchor_snapshot,
            source_fingerprint=batch.fingerprint,
        ),
        text_units=(unit,),
        alignment_segments=(segment,),
        alignment_slices=(alignment,),
    )

    index = CharIndexService().build(
        page=page,
        batch=batch,
        lines=(line,),
        atoms=(atom_a, space, atom_b),
        state=state,
    )

    whitespace = index.query(" ", include_unavailable=True)
    assert len(whitespace) == 1
    assert whitespace[0].bbox is None
    assert whitespace[0].geometry_status == GEOMETRY_UNAVAILABLE
    assert whitespace[0].unavailable_reason == "whitespace_has_no_glyph_geometry"
    assert "".join(entry.text for entry in index.entries) == "a b"


def test_rebind_is_an_aggregate_replace_and_invalidates_alignment_until_rebound() -> None:
    session, initial = _session()
    service = ProofSessionService(session)
    created = service.create_state(initial)
    next_anchor = replace(
        initial.anchor_snapshot,
        source_fingerprint="new-source",
        layout_fingerprint="new-layout",
        anchor_revision=2,
    )
    marked = service.mark_rebind_required(
        initial.uid,
        next_anchor,
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
    )
    assert marked.state.rebind_required is True
    assert marked.state.alignment_slices == ()
    unavailable_index = service.build_char_index(initial.uid)
    assert unavailable_index.unavailable_reason == "proof_state_requires_rebind"

    segment = replace(initial.alignment_segments[0], anchor_uid=next_anchor.uid)
    rebound = service.rebind(
        initial.uid,
        next_anchor,
        (segment,),
        initial.alignment_slices,
        expected_revision=marked.revision,
        expected_fingerprint=marked.fingerprint,
    )
    assert rebound.state.rebind_required is False
    assert rebound.state.alignment_segments == (segment,)


def test_active_observation_refresh_marks_stale_anchor_without_writing_ocr() -> None:
    session, initial = _session()
    stale = replace(
        initial,
        anchor_snapshot=replace(initial.anchor_snapshot, source_fingerprint="stale-source"),
    )
    service = ProofSessionService(session)
    created = service.create_state(stale)
    refreshed = service.refresh_active_observation(
        stale.uid,
        expected_revision=created.revision,
        expected_fingerprint=created.fingerprint,
    )
    assert refreshed.outcome.value == "rebind_required"
    assert refreshed.state.rebind_required is True
    assert session.ocr_observation_repository.get_active_pointer("page-1").batch_uid == "batch-1"
