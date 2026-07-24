from __future__ import annotations

import json

import numpy as np
import pytest

import app.services.ocr_routing_observation_service as module
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_routing_observation import InlineFormulaTextObservationStatus
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord
from app.services.formula_crop_ocr_service import FormulaCropOcrOutcome


def _response(parent_text: str) -> dict:
    return {
        "result": {
            "layoutParsingResults": [{
                "prunedResult": {
                    "width": 100,
                    "height": 80,
                    "parsing_res_list": [{
                        "block_label": "text",
                        "block_bbox": [1, 2, 90, 30],
                        "block_content": parent_text,
                    }],
                    "layout_det_res": {
                        "boxes": [{
                            "label": "inline_formula",
                            "coordinate": [45, 5, 55, 20],
                            "score": 0.9,
                        }]
                    },
                }
            }]
        }
    }


def _inputs(parent_text: str):
    page = PageRecord(
        project_uid="project-1",
        uid="page-1",
        image_path="page.png",
        source_path="book.pdf",
        cache_image_path="page.png",
        thumbnail_path="",
        width=100,
        height=80,
        page_number=1,
        source_page_index=0,
        status="layout_done",
        error="",
        image_hash="hash",
        image_revision=1,
    )
    artifact = PaddleArtifact(
        project_uid="project-1",
        uid="artifact-1",
        page_uid="page-1",
        source_engine="paddle-vl",
        source_run_id="layout-run",
        image_hash="hash",
        payload_json=json.dumps(_response(parent_text)),
    )
    block = LayoutBlockSnapshot(
        uid="formula-1",
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(45, 5, 55, 20),
        order=0,
        source_label="inline_formula",
        origin=BlockOrigin(
            created_by=BlockSource.AUTO_LAYOUT.value,
            raw_artifact_uid=artifact.uid,
            raw_json_path=(
                "layoutParsingResults[0].prunedResult.layout_det_res.boxes[0]"
            ),
        ),
        ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
        authorship=BlockSource.AUTO_LAYOUT,
    )
    snapshot = LayoutSnapshot(
        page_uid=page.uid,
        revision=1,
        artifact_uid=artifact.uid,
        source_engine="paddle-vl",
        source_run_id="layout-run",
        blocks=(block,),
    )
    return page, artifact, snapshot


def _edited_text_inputs():
    page, artifact, _snapshot = _inputs("")
    original_bbox = BBox.from_xyxy(1, 2, 90, 30)
    block = LayoutBlockSnapshot(
        uid="text-1",
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(2, 2, 90, 30),
        order=0,
        source_label="text",
        origin=BlockOrigin(
            created_by=BlockSource.AUTO_LAYOUT.value,
            raw_artifact_uid=artifact.uid,
            raw_index=0,
            original_bbox=original_bbox,
            original_kind=BlockType.TEXT,
        ),
        ocr_policy=OcrPolicy.TEXT_OCR,
        authorship=BlockSource.USER_EDITED,
    )
    snapshot = LayoutSnapshot(
        page_uid=page.uid,
        revision=2,
        artifact_uid=artifact.uid,
        source_engine="layout_edit",
        source_run_id="layout-edit:2",
        blocks=(block,),
    )
    return page, artifact, snapshot


def test_exact_parent_formula_text_does_not_call_crop_ocr(monkeypatch) -> None:
    page, artifact, snapshot = _inputs("left $x$ right")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("exact parent binding must not call formula crop OCR")

    monkeypatch.setattr(module, "recognize_formula_bboxes_with_retry", unexpected)
    bundle = module.acquire_routing_observation_bundle(
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        image_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        prepass=object(),
        vl_client=object(),
    )

    observation = bundle.inline_formula_observations[0]
    assert observation.status is InlineFormulaTextObservationStatus.OBSERVED
    assert observation.text == "$x$"
    assert observation.source == module.INLINE_FORMULA_PARENT_SOURCE


def test_count_mismatch_uses_formula_crop_ocr_and_normalizes_inline_text(monkeypatch) -> None:
    page, artifact, snapshot = _inputs("left $x$ and missing $y$")
    monkeypatch.setattr(
        module,
        "recognize_formula_bboxes_with_retry",
        lambda *_args, **_kwargs: FormulaCropOcrOutcome(
            texts_by_index={0: r"\(z_1\)"},
            attempted_indices=(0,),
            failed_indices=(),
            attempts=2,
        ),
    )

    bundle = module.acquire_routing_observation_bundle(
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        image_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        prepass=object(),
        vl_client=object(),
    )

    observation = bundle.inline_formula_observations[0]
    assert observation.text == "$z_1$"
    assert observation.source == module.INLINE_FORMULA_CROP_SOURCE
    assert observation.attempts == 2


def test_failed_formula_crop_is_an_explicit_nonblocking_observation(monkeypatch) -> None:
    page, artifact, snapshot = _inputs("left $x$ and missing $y$")
    monkeypatch.setattr(
        module,
        "recognize_formula_bboxes_with_retry",
        lambda *_args, **_kwargs: FormulaCropOcrOutcome(
            texts_by_index={},
            attempted_indices=(0,),
            failed_indices=(0,),
            attempts=3,
            error="service unavailable",
        ),
    )

    bundle = module.acquire_routing_observation_bundle(
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        image_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        prepass=object(),
        vl_client=object(),
    )

    observation = bundle.inline_formula_observations[0]
    assert observation.status is InlineFormulaTextObservationStatus.UNRESOLVED
    assert observation.text == ""
    assert observation.error == "service unavailable"


def test_block_vl_refresh_retries_without_optional_batch_id() -> None:
    page, artifact, snapshot = _edited_text_inputs()

    class Client:
        def __init__(self) -> None:
            self.batch_ids: list[str] = []

        def analyze_image(self, _crop, *, optional_payload, batch_id):
            self.batch_ids.append(batch_id)
            if batch_id:
                raise RuntimeError("vendor rejected batch metadata")
            response = _response("fresh text")
            response["paddle_v16"] = {"jobId": "job-refresh"}
            return response

    client = Client()
    bundle = module.acquire_routing_observation_bundle(
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        image_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        prepass=object(),
        vl_client=client,
    )

    assert len(client.batch_ids) == 2
    assert client.batch_ids[0]
    assert client.batch_ids[1] == ""
    assert bundle.block_vl_observations[0].source_run_id == "job-refresh"
    assert bundle.block_vl_observations[0].attempts == 2


def test_block_vl_refresh_failure_contains_copyable_page_block_and_attempt_details() -> None:
    page, artifact, snapshot = _edited_text_inputs()

    class Client:
        def analyze_image(self, _crop, *, optional_payload, batch_id):
            mode = "batched" if batch_id else "unbatched"
            raise RuntimeError(f"{mode} request failed")

    with pytest.raises(module.BlockVlObservationRefreshError) as captured:
        module.acquire_routing_observation_bundle(
            page=page,
            snapshot=snapshot,
            artifact=artifact,
            image_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
            prepass=object(),
            vl_client=Client(),
        )

    message = str(captured.value)
    assert "page='book.pdf'" in message
    assert "block=text-1" in message
    assert "label='text'" in message
    assert "bbox=(2, 2, 90, 30)" in message
    assert "attempt 1: RuntimeError: batched request failed" in message
    assert "attempt 2: RuntimeError: unbatched request failed" in message
