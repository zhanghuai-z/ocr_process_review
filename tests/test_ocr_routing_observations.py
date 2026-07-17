from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.models import BBox, BlockOrigin, BlockSource, BlockType, OcrPolicy, Page
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.services.ocr_routing_observation_service import (
    BlockVlObservationRefreshError,
    acquire_routing_observation_bundle,
)


def _snapshot(*, bbox=(10, 20, 80, 60), origin=None):
    return LayoutSnapshot(
        page_uid="page-test",
        artifact_uid="layout-artifact-test",
        source_engine="test",
        source_run_id="layout-run-test",
        blocks=(LayoutBlockSnapshot(
            uid="block-test",
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(*bbox),
            order=0,
            source_label="text",
            origin=origin,
            ocr_policy=OcrPolicy.TEXT_OCR,
        ),),
    )


def _page():
    return Page(image_path="", width=100, height=80, uid="page-test")


def _response(text="fresh text", bbox=(0, 0, 70, 40)):
    records = [] if not text else [{
        "block_label": "text",
        "block_bbox": list(bbox),
        "block_content": text,
    }]
    return {
        "result": {
            "layoutParsingResults": [{
                "prunedResult": {"parsing_res_list": records},
            }],
        },
        "paddle_v16": {"jobId": "vl-job-test"},
    }


class _VlClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.crops = []

    def analyze_image(self, image, **_kwargs):
        self.crops.append(image.copy())
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_changed_layout_block_refreshes_vl_against_current_bbox():
    client = _VlClient([_response()])
    bundle = acquire_routing_observation_bundle(
        page=_page(),
        snapshot=_snapshot(),
        image_bgr=np.full((80, 100, 3), 255, dtype=np.uint8),
        prepass=object(),
        vl_client=client,
    )

    assert client.crops[0].shape[:2] == (40, 70)
    observation = bundle.block_vl_observations[0]
    assert observation.block_bbox == (10, 20, 80, 60)
    assert observation.regions[0].bbox == (10, 20, 80, 60)
    assert observation.text == "fresh text"
    assert observation.attempts == 1


def test_successful_empty_vl_observation_is_explicit_and_nonfatal():
    client = _VlClient([_response(text="")])
    bundle = acquire_routing_observation_bundle(
        page=_page(),
        snapshot=_snapshot(),
        image_bgr=np.full((80, 100, 3), 255, dtype=np.uint8),
        prepass=object(),
        vl_client=client,
    )

    observation = bundle.block_vl_observations[0]
    assert observation.status.value == "empty"
    assert observation.regions == ()


def test_vl_refresh_retries_once_then_fails_page_local():
    client = _VlClient([RuntimeError("first"), RuntimeError("second")])
    with pytest.raises(BlockVlObservationRefreshError, match="after retry"):
        acquire_routing_observation_bundle(
            page=_page(),
            snapshot=_snapshot(),
            image_bgr=np.full((80, 100, 3), 255, dtype=np.uint8),
            prepass=object(),
            vl_client=client,
        )
    assert len(client.crops) == 2


def test_malformed_vl_response_is_retried_and_not_treated_as_empty():
    client = _VlClient([{"result": {}}, {"unexpected": []}])
    with pytest.raises(BlockVlObservationRefreshError, match="after retry"):
        acquire_routing_observation_bundle(
            page=_page(),
            snapshot=_snapshot(),
            image_bgr=np.full((80, 100, 3), 255, dtype=np.uint8),
            prepass=object(),
            vl_client=client,
        )
    assert len(client.crops) == 2


def test_unchanged_auto_block_reuses_exact_original_vl_observation(monkeypatch):
    origin = BlockOrigin(
        created_by=BlockSource.AUTO_LAYOUT.value,
        source_label="text",
        original_bbox=BBox.from_xyxy(10, 20, 80, 60),
        original_kind=BlockType.TEXT,
        raw_artifact_uid="rawocr-test",
        raw_index=3,
        source_run_id="layout-job-test",
    )
    artifact = SimpleNamespace(
        artifact_uid="rawocr-test",
        regions=(SimpleNamespace(
            index=3,
            label="text",
            text="original text",
            bbox=(10, 20, 80, 60),
        ),),
    )
    monkeypatch.setattr(
        "app.services.ocr_routing_observation_service.normalized_layout_artifact_from_page",
        lambda _page: artifact,
    )
    client = _VlClient([])

    bundle = acquire_routing_observation_bundle(
        page=_page(),
        snapshot=_snapshot(origin=origin),
        image_bgr=np.full((80, 100, 3), 255, dtype=np.uint8),
        prepass=object(),
        vl_client=client,
    )

    assert client.crops == []
    assert bundle.block_vl_observations[0].text == "original text"
    assert bundle.block_vl_observations[0].attempts == 0
