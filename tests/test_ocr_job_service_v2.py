from __future__ import annotations

import json

import numpy as np
import pytest

import app.services.ocr_job_service as module
from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.charocr_execution import (
    CharOcrAtomObservation,
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


def _session() -> ProjectSession:
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

    def recognize_page(self, _image, request, progress_callback=None):
        return CharOcrPageResult(
            page_uid=request.page.uid,
            input_fingerprint=request.input_fingerprint,
            regions=(CharOcrRegionObservation(
                block_uid="block-1", label="text", bbox=(1, 2, 41, 22), source="test",
                lines=(CharOcrLineObservation(
                    text="machine", bbox=(2, 3, 30, 15), confidence=0.9, source="test",
                    atoms=(CharOcrAtomObservation(
                        text="机", bbox=(2, 3, 8, 15), confidence=0.8, source="test",
                    ),),
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
    service = OcrJobService(prepass_client=_Prepass(), vl_client=object(), engine=_Engine())

    commit = service.run_page(session, "page-1", np.zeros((80, 100, 3), dtype=np.uint8))

    pointer = session.ocr_observation_repository.get_active_pointer("page-1")
    assert pointer.batch_uid == commit.batch_uid
    assert len(session.ocr_observation_repository.all_lines()) == 1
    proof = session.proof_repository.get_state("proof-1")
    assert proof.text_units[0].text == "human correction"
    assert proof.rebind_required is True
    assert proof.alignment_segments == ()
    assert proof.anchor_snapshot.source_fingerprint == pointer.batch_fingerprint
    binding = session.binding_repository.get("ocrbind_block-1")
    assert binding.source_uid == "block-1"
    assert binding.target_uid == session.ocr_observation_repository.all_regions()[0].uid


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
        layout_fingerprint="layout", result=result,
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
        layout_fingerprint="layout", result=result,
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
