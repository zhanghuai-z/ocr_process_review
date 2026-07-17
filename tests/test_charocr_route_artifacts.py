from __future__ import annotations

import json

import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6PrepassArtifact
from app.diagnostics.charocr_route_artifacts import write_charocr_route_artifacts
from app.models import BBox, Block, BlockType, Page
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RouteDiagnostic,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection
from app.services.ocr_pipeline import OcrPipeline


def test_route_artifact_emits_only_text_route_crops(tmp_path):
    plan = PageRoutingPlan(
        page_uid="page-test",
        prepass_run_id="prepass-test",
        blocks=(BlockRoutingPlan(
            block_uid="block-test",
            plan=RoutingPlan(
                lines=(RoutingLine(
                    index=7,
                    bbox=(5, 5, 75, 25),
                    segments=(
                        RoutingSegment(kind="text_other", bbox=(5, 5, 25, 25)),
                        RoutingSegment(kind="formula", bbox=(25, 5, 45, 25), label="formula"),
                        RoutingSegment(kind="text_latin", bbox=(45, 5, 75, 25), text="ABC"),
                    ),
                    source="ppocrv6_prepass",
                ),),
                text_slices=(
                    TextSliceRoute(7, 0, (5, 5, 25, 25), True, kind="text_other"),
                    TextSliceRoute(7, 2, (45, 5, 75, 25), True, kind="text_latin"),
                ),
                has_layout_routes=True,
            ),
        ),),
        diagnostics=(RouteDiagnostic(
            code="missing_latin_token_ink",
            message="foreground remains on LineCut",
            line_index=7,
            bbox=(25, 5, 35, 25),
        ),),
    )

    output = write_charocr_route_artifacts(
        image_bgr=np.full((40, 90, 3), 255, dtype=np.uint8),
        source_image_path="page.png",
        routing_plan=plan,
        output_root=tmp_path,
    )

    assert output is not None
    assert (output / "route-overlay.png").is_file()
    crop_names = sorted(path.name for path in (output / "crops").glob("*.png"))
    assert crop_names == [
        "line_0007_segment_00_text_other_5_5_25_25.png",
        "line_0007_segment_02_text_latin_45_5_75_25.png",
    ]
    payload = json.loads((output / "route-plan.json").read_text(encoding="utf-8"))
    segments = payload["routes"][0]["segments"]
    assert [segment["native_branch"] for segment in segments] == ["linecut", "excluded", "engcut"]
    assert segments[0]["crop_role"] == "route_segment_visualization"
    assert segments[2]["crop_role"] == "route_segment_visualization"
    assert segments[1].get("crop_file") is None
    assert payload["diagnostics"] == [{
        "code": "missing_latin_token_ink",
        "message": "foreground remains on LineCut",
        "line_index": 7,
        "bbox": [25, 5, 35, 25],
    }]
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "不是\n对 native 调用的逐像素复刻" in readme


def test_route_artifact_can_skip_crop_visualizations(tmp_path):
    plan = PageRoutingPlan(
        page_uid="page-test",
        prepass_run_id="prepass-test",
        blocks=(BlockRoutingPlan(
            block_uid="block-test",
            plan=RoutingPlan(
                lines=(RoutingLine(
                    index=0,
                    bbox=(5, 5, 25, 25),
                    segments=(RoutingSegment(kind="text_other", bbox=(5, 5, 25, 25)),),
                ),),
                text_slices=(TextSliceRoute(0, 0, (5, 5, 25, 25), True, kind="text_other"),),
                has_layout_routes=True,
            ),
        ),),
    )

    output = write_charocr_route_artifacts(
        image_bgr=np.full((40, 40, 3), 255, dtype=np.uint8),
        source_image_path="page.png",
        routing_plan=plan,
        output_root=tmp_path,
        write_crops=False,
    )

    assert output is not None
    assert not (output / "crops").exists()
    payload = json.loads((output / "route-plan.json").read_text(encoding="utf-8"))
    assert payload["routes"][0]["segments"][0]["native_branch"] == "linecut"
    assert payload["routes"][0]["segments"][0].get("crop_file") is None


def test_hybrid_pipeline_writes_current_route_plan_before_native_dispatch(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), np.full((40, 90, 3), 255, dtype=np.uint8))
    page = Page(
        image_path=str(image_path),
        width=90,
        height=40,
        blocks=[Block(BlockType.TEXT, BBox.from_xyxy(0, 0, 90, 40), source_label="text")],
    )
    sync_page_layout_snapshot_from_projection(page, source_engine="test")

    class FakePrepass:
        def analyze_page(self, _image, *, page_uid: str):
            return PpOcrV6PrepassArtifact(
                page_uid=page_uid,
                run_id="route-debug-test",
                lines=(PpOcrV6LineHint(0, "正文", (0, 0, 90, 40), ()),),
            )

    class FakeNativeEngine:
        prefer_page_hybrid_blocks = True
        bbox_space = "page"

        def __init__(self):
            self.plan = None

        def recognize_page_blocks(self, _image, _page, **kwargs):
            self.plan = kwargs["routing_plan"]

    hook_root = tmp_path / "hook"
    monkeypatch.setenv("CHAROCR_ROUTE_HOOK_DIR", str(hook_root))
    engine = FakeNativeEngine()
    pipeline = OcrPipeline(engine=engine, hybrid_prepass_engine=FakePrepass())

    pipeline._process_page_with_hybrid_blocks(
        np.full((40, 90, 3), 255, dtype=np.uint8),
        page,
    )

    assert engine.plan is not None
    artifact_dirs = list(hook_root.iterdir())
    assert len(artifact_dirs) == 1
    assert (artifact_dirs[0] / "route-overlay.png").is_file()
    payload = json.loads((artifact_dirs[0] / "route-plan.json").read_text(encoding="utf-8"))
    assert payload["prepass_run_id"] == "route-debug-test"
    assert payload["routes"][0]["segments"][0]["native_branch"] == "linecut"
