from __future__ import annotations

import json

import numpy as np
import pytest

import app.services.ocr_job_service as module
from app.application.ocr_workspace import build_ocr_workspace_view
from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.charocr_execution import (
    CharOcrAtomObservation,
    CharOcrCandidateObservation,
    CharOcrLineObservation,
    CharOcrPageResult,
    CharOcrRegionObservation,
)
from app.models.charocr_routing import PageRoutingPlan
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.ocr_records import OcrActivePointer
from app.models.project_session import (
    BindingRecord, PageRecord, ProjectRecord, ProjectSession, RecordNotFoundError,
)
from app.models.proof_records import ProofAnchorSnapshot, ProofState, ProofTextUnit
from app.services.ocr_job_service import OcrJobService


def _session(*, with_proof: bool = True) -> ProjectSession:
    session = ProjectSession(ProjectRecord("project-1", "Book"))
    page = PageRecord(
        project_uid="project-1", uid="page-1", image_path="page.png",
        source_path="book.pdf", cache_image_path="page.png", thumbnail_path="",
        width=100, height=80, page_number=1, source_page_index=0,
        status="layout_done", error="", image_hash="hash", image_revision=1,
    )
    session.page_repository.put(page, expected_revision=0)
    artifact = PaddleArtifact(
        project_uid="project-1", uid="artifact-1", page_uid="page-1",
        source_engine="paddle-vl", source_run_id="layout-run", image_hash="hash",
        payload_json=json.dumps({"result": {"layoutParsingResults": []}}),
    )
    session.paddle_artifact_repository.append(artifact)
    layout = LayoutSnapshot(
        page_uid="page-1", revision=1, artifact_uid="artifact-1",
        source_engine="paddle-vl", source_run_id="layout-run",
        blocks=(LayoutBlockSnapshot(
            uid="block-1", block_type=BlockType.TEXT, bbox=BBox(1, 2, 40, 20),
            order=0, source_label="text",
            origin=BlockOrigin(created_by=BlockSource.MANUAL_DRAW.value),
            ocr_policy=OcrPolicy.TEXT_OCR,
            authorship=BlockSource.MANUAL_DRAW,
        ),),
    )
    session.layout_repository.put(layout, expected_revision=0)
    anchor = ProofAnchorSnapshot(
        project_uid="project-1", uid="anchor-1", scope_uid="page-1",
        layout_fingerprint="old-layout", source_fingerprint="old-source",
        anchor_revision=1,
    )
    if with_proof:
        session.proof_repository.create_state(ProofState(
            project_uid="project-1", uid="proof-1", anchor_snapshot=anchor,
            text_units=(ProofTextUnit(
                project_uid="project-1", uid="proof-text-1", order=0,
                text="human correction",
            ),),
        ))
    return session


class _Prepass:
    def analyze_page(self, _image, *, page_uid):
        return object()


class _Engine:
    engine_id = "charocr-test"

    def __init__(self, *, include_pp_symbol: bool = False) -> None:
        self._include_pp_symbol = include_pp_symbol

    def recognize_page(self, _image, request, progress_callback=None):
        atoms = [
            CharOcrAtomObservation(
                text="machine", bbox=(2, 3, 30, 15), confidence=0.8, source="test",
                granularity="word", token_text="machine",
                candidates=(CharOcrCandidateObservation(
                    text="rnachine",
                    confidence=0.0,
                    source="ppocrv6:latin_token_text_alignment",
                    bbox=(3, 4, 29, 14),
                ),),
            ),
        ]
        if self._include_pp_symbol:
            atoms.append(CharOcrAtomObservation(
                text="、", bbox=(32, 9, 38, 15), confidence=0.0,
                source="ppocrv6:symbol_foreground_observation",
                token_text="、",
                candidates=(CharOcrCandidateObservation(
                    text="、",
                    confidence=0.0,
                    source="ppocrv6:symbol_foreground_observation",
                    bbox=(32, 9, 38, 15),
                ),),
            ))
        return CharOcrPageResult(
            page_uid=request.page.uid,
            input_fingerprint=request.input_fingerprint,
            regions=(CharOcrRegionObservation(
                block_uid="block-1", label="text", bbox=(1, 2, 41, 22), source="test",
                lines=(CharOcrLineObservation(
                    text="machine、" if self._include_pp_symbol else "machine",
                    bbox=(2, 3, 38, 15) if self._include_pp_symbol else (2, 3, 30, 15),
                    confidence=0.9, source="test", atoms=tuple(atoms),
                ),),
            ),),
        )


def test_page_job_appends_batch_switches_pointer_and_preserves_proof(monkeypatch) -> None:
    session = _session()
    layout = session.layout_repository.get("page-1")
    routing = PageRoutingPlan(
        page_uid="page-1", routing_run_uid="routing-1",
        layout_fingerprint=layout_snapshot_fingerprint(layout),
        prepass_run_id="prepass-1", blocks=(),
    )
    monkeypatch.setattr(module, "acquire_routing_observation_bundle", lambda **_kwargs: object())
    monkeypatch.setattr(module, "compile_page_routing_plan", lambda *_args, **_kwargs: routing)
    service = OcrJobService(
        prepass_client=_Prepass(), vl_client=object(),
        engine=_Engine(include_pp_symbol=True),
    )

    commit = service.run_page(session, "page-1", np.zeros((80, 100, 3), dtype=np.uint8))

    pointer = session.ocr_observation_repository.get_active_pointer("page-1")
    assert pointer.batch_uid == commit.batch_uid
    assert len(session.ocr_observation_repository.all_lines()) == 1
    batch = session.ocr_observation_repository.get_batch(commit.batch_uid)
    atom = session.ocr_observation_repository.get_atom(batch.atom_uids[0])
    assert atom.source == "test"
    assert atom.granularity == "word"
    assert atom.token_text == "machine"
    candidate = session.ocr_observation_repository.get_candidate(atom.candidate_uids[0])
    assert candidate.text == "rnachine"
    assert candidate.source == "ppocrv6:latin_token_text_alignment"
    assert candidate.bbox == (3, 4, 29, 14)
    symbol = session.ocr_observation_repository.get_atom(batch.atom_uids[1])
    assert symbol.text == "、"
    assert symbol.bbox == (32, 9, 38, 15)
    assert symbol.source == "ppocrv6:symbol_foreground_observation"
    proof = session.proof_repository.get_state("proof-1")
    assert proof.text_units[0].text == "human correction"
    assert proof.rebind_required is True
    assert proof.alignment_segments == ()
    assert proof.anchor_snapshot.source_fingerprint == pointer.batch_fingerprint
    binding = session.binding_repository.get("ocrbind_block-1")
    assert binding.source_uid == "block-1"
    assert binding.target_uid == session.ocr_observation_repository.all_regions()[0].uid
    workspace = build_ocr_workspace_view(session)
    assert workspace.line_count == 1
    assert workspace.pages[0].batch_uid == commit.batch_uid
    assert workspace.pages[0].regions[0].block_uid == "block-1"
    assert workspace.pages[0].regions[0].lines[0].text == "machine、"
    assert workspace.pages[0].regions[0].lines[0].atoms[0].text == "machine"


def test_first_page_ocr_creates_editable_proof_state_with_exact_alignment(monkeypatch) -> None:
    from app.services.char_index_service import CharIndexService

    session = _session(with_proof=False)
    layout = session.layout_repository.get("page-1")
    routing = PageRoutingPlan(
        page_uid="page-1", routing_run_uid="routing-1",
        layout_fingerprint=layout_snapshot_fingerprint(layout),
        prepass_run_id="prepass-1", blocks=(),
    )
    monkeypatch.setattr(module, "acquire_routing_observation_bundle", lambda **_kwargs: object())
    monkeypatch.setattr(module, "compile_page_routing_plan", lambda *_args, **_kwargs: routing)
    service = OcrJobService(prepass_client=_Prepass(), vl_client=object(), engine=_Engine())

    commit = service.run_page(session, "page-1", np.zeros((80, 100, 3), dtype=np.uint8))

    assert commit.proof_states_created == 1
    state = session.proof_repository.get_state("proof_page-1")
    assert [unit.text for unit in state.text_units] == ["machine"]
    assert state.rebind_required is False
    pointer = session.ocr_observation_repository.get_active_pointer("page-1")
    batch = session.ocr_observation_repository.get_batch(pointer.batch_uid)
    lines = tuple(
        session.ocr_observation_repository.get_line(uid) for uid in batch.line_uids
    )
    atoms = tuple(
        session.ocr_observation_repository.get_atom(uid) for uid in batch.atom_uids
    )
    index = CharIndexService().build(
        page=session.page_repository.get("page-1"),
        batch=batch,
        lines=lines,
        atoms=atoms,
        state=state,
    )
    assert "".join(entry.text for entry in index.entries) == "machine"
    assert all(entry.available for entry in index.entries)


def test_first_page_ocr_is_visible_in_both_proof_panels(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    from app.application.proof_workspace import build_proof_workspace_view
    from app.ui.proof.h_proof import HProofPanel
    from app.ui.proof.v_proof import VProofPanel

    app = QApplication.instance() or QApplication([])
    session = _session(with_proof=False)
    layout = session.layout_repository.get("page-1")
    routing = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routing-1",
        layout_fingerprint=layout_snapshot_fingerprint(layout),
        prepass_run_id="prepass-1",
        blocks=(),
    )
    monkeypatch.setattr(module, "acquire_routing_observation_bundle", lambda **_kwargs: object())
    monkeypatch.setattr(module, "compile_page_routing_plan", lambda *_args, **_kwargs: routing)
    service = OcrJobService(prepass_client=_Prepass(), vl_client=object(), engine=_Engine())

    service.run_page(session, "page-1", np.zeros((80, 100, 3), dtype=np.uint8))
    workspace = build_proof_workspace_view(session)
    horizontal = HProofPanel(workspace=workspace)
    vertical = VProofPanel(workspace=workspace)

    assert [row.unit.text for row in horizontal._rows] == ["machine"]
    assert vertical._entries
    assert "".join(entry.text for entry in vertical._entries) == "machine"
    horizontal.close()
    vertical.close()
    app.processEvents()


def test_observation_batch_validation_is_all_or_nothing() -> None:
    session = ProjectSession("project-1")
    repository = session.ocr_observation_repository
    result = CharOcrPageResult(
        page_uid="page-1", input_fingerprint="input",
        regions=(CharOcrRegionObservation(
            block_uid="block-1", label="text", bbox=(1, 1, 2, 2), source="test",
        ),),
    )
    records = module._observation_records(
        project_uid="project-1", page_uid="page-1", engine_id="test",
            layout_fingerprint="layout", page_fingerprint="page", result=result,
    )
    bad_batch = records.batch
    invalid_batch = type(bad_batch)(
        project_uid=bad_batch.project_uid, uid=bad_batch.uid, run_uid=bad_batch.run_uid,
        scope_uid=bad_batch.scope_uid, input_fingerprint=bad_batch.input_fingerprint,
        layout_fingerprint=bad_batch.layout_fingerprint,
    )

    try:
        repository.append_observation_batch(
            run=records.run, regions=records.regions, lines=records.lines,
            atoms=records.atoms, candidates=records.candidates, batch=invalid_batch,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete batch should be rejected")
    assert repository.all_runs() == ()
    assert repository.all_regions() == ()


def test_page_adoption_rolls_back_observation_pointer_and_binding_on_proof_failure() -> None:
    session = ProjectSession("project-1")
    result = CharOcrPageResult(
        page_uid="page-1", input_fingerprint="input",
        regions=(CharOcrRegionObservation(
            block_uid="block-1", label="text", bbox=(1, 1, 2, 2), source="test",
        ),),
    )
    records = module._observation_records(
        project_uid="project-1", page_uid="page-1", engine_id="test",
        layout_fingerprint="layout", page_fingerprint="page", result=result,
    )
    pointer = OcrActivePointer(
        project_uid="project-1", uid="ocrptr_page-1", scope_uid="page-1",
        batch_uid=records.batch.uid, run_uid=records.run.uid,
        batch_fingerprint=records.batch.fingerprint, revision=1,
    )
    region = records.regions[0]
    binding = BindingRecord(
        project_uid="project-1", uid="ocrbind_block-1", source_uid="block-1",
        target_uid=region.uid, relation="observed_by", source_fingerprint="layout",
        target_fingerprint=region.fingerprint,
    )
    missing_state = ProofState(
        project_uid="project-1", uid="missing-proof",
        anchor_snapshot=ProofAnchorSnapshot(
            project_uid="project-1", uid="anchor-missing", scope_uid="page-1",
            layout_fingerprint="layout", source_fingerprint=records.batch.fingerprint,
            anchor_revision=1,
        ),
    )

    with pytest.raises(RecordNotFoundError):
        session.adopt_ocr_page_observation(
            run=records.run, regions=records.regions, lines=records.lines,
            atoms=records.atoms, candidates=records.candidates, batch=records.batch,
            pointer=pointer, expected_pointer_revision=0,
            expected_pointer_fingerprint=None, bindings=(binding,),
            proof_states=(missing_state,),
        )

    assert session.ocr_observation_repository.all_runs() == ()
    assert session.ocr_observation_repository.all_regions() == ()
    assert session.ocr_observation_repository.all_active_pointers() == ()
    assert session.binding_repository.all() == ()
