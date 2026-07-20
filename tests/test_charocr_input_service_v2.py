from __future__ import annotations

import json

from app.models.charocr_routing import PageRoutingPlan
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord
from app.services.charocr_input_service import compile_charocr_page_request


def _page() -> PageRecord:
    return PageRecord(
        project_uid="project-1",
        uid="page-1",
        image_path="page.png",
        source_path="source.pdf",
        cache_image_path="page.png",
        thumbnail_path="thumb.png",
        width=100,
        height=80,
        page_number=1,
        source_page_index=0,
        status="layout_done",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )


def _artifact() -> PaddleArtifact:
    payload = {
        "result": {
            "layoutParsingResults": [{
                "prunedResult": {
                    "parsing_res_list": [{
                        "block_label": "text",
                        "block_bbox": [5, 6, 50, 30],
                        "block_content": "Paddle observed text",
                    }]
                }
            }]
        }
    }
    return PaddleArtifact(
        project_uid="project-1",
        uid="artifact-1",
        page_uid="page-1",
        source_engine="paddle-vl-1.6",
        source_run_id="run-layout-1",
        image_hash="image-hash",
        payload_json=json.dumps(payload),
    )


def _layout() -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid="page-1",
        revision=1,
        artifact_uid="artifact-1",
        source_engine="paddle-vl-1.6",
        source_run_id="run-layout-1",
        blocks=(LayoutBlockSnapshot(
            uid="block-1",
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(5, 6, 50, 30),
            order=0,
            source_label="text",
            origin=BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                raw_artifact_uid="artifact-1",
                raw_index=0,
                original_bbox=BBox.from_xyxy(5, 6, 50, 30),
                original_kind=BlockType.TEXT,
            ),
            ocr_policy=OcrPolicy.TEXT_OCR,
        ),),
    )


def test_compiler_uses_layout_identity_and_paddle_content_only() -> None:
    layout = _layout()
    routing = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routing-1",
        layout_fingerprint="layout-fingerprint",
        prepass_run_id="prepass-1",
        blocks=(),
    )
    request = compile_charocr_page_request(
        project_uid="project-1",
        page=_page(),
        layout=layout,
        artifact=_artifact(),
        routing_plan=routing,
    )

    assert request.rows[0].block_uid == "block-1"
    assert request.rows[0].content == "Paddle observed text"
    assert request.rows[0].native_payload()["_layout_block_uid"] == "block-1"
    assert request.input_fingerprint


def test_compiler_rejects_an_artifact_not_adopted_by_layout() -> None:
    wrong = PaddleArtifact(
        project_uid="project-1",
        uid="artifact-2",
        page_uid="page-1",
        source_engine="paddle-vl-1.6",
        source_run_id="run-layout-2",
        image_hash="image-hash",
        payload_json="{}",
    )
    routing = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routing-1",
        layout_fingerprint="layout-fingerprint",
        prepass_run_id="prepass-1",
        blocks=(),
    )

    try:
        compile_charocr_page_request(
            project_uid="project-1",
            page=_page(),
            layout=_layout(),
            artifact=wrong,
            routing_plan=routing,
        )
    except ValueError as exc:
        assert "does not reference" in str(exc)
    else:
        raise AssertionError("foreign artifact should be rejected")
