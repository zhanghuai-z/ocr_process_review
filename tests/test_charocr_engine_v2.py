from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

from app.engines.hanwang.micro_recblock import (
    HanwangMicroRecBlockEngine,
    RunStats,
)
from app.models.charocr_execution import (
    CharOcrAtomObservation,
    CharOcrCandidateObservation,
    CharOcrInputRow,
    CharOcrLineObservation,
    CharOcrPageRequest,
    CharOcrPageResult,
    CharOcrRegionObservation,
)
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
)
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.project_session import PageRecord


def _page() -> PageRecord:
    return PageRecord(
        project_uid="project-1",
        uid="page-1",
        image_path="page.png",
        source_path="source.pdf",
        cache_image_path="page.png",
        thumbnail_path="thumb.png",
        width=120,
        height=80,
        page_number=1,
        source_page_index=0,
        status="layout_done",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )


def _layout() -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid="page-1",
        revision=1,
        artifact_uid="artifact-1",
        source_engine="test-layout",
        source_run_id="layout-run-1",
        blocks=(
            LayoutBlockSnapshot(
                uid="row-text",
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(5, 5, 70, 30),
                order=0,
                source_label="text",
                origin=BlockOrigin(raw_artifact_uid="artifact-1", raw_index=0),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
            LayoutBlockSnapshot(
                uid="row-formula",
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(75, 5, 115, 30),
                order=1,
                source_label="formula",
                origin=BlockOrigin(raw_artifact_uid="artifact-1", raw_index=1),
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
            ),
        ),
    )


def _request() -> CharOcrPageRequest:
    return CharOcrPageRequest(
        project_uid="project-1",
        page=_page(),
        layout=_layout(),
        routing_plan=PageRoutingPlan(
            page_uid="page-1",
            routing_run_uid="routing-run-1",
            layout_fingerprint="layout-fingerprint-1",
            prepass_run_id="prepass-run-1",
            blocks=(
                BlockRoutingPlan(
                    block_uid="row-text",
                    plan=RoutingPlan(
                        lines=(RoutingLine(
                            index=0,
                            bbox=(5, 5, 70, 30),
                            segments=(RoutingSegment(
                                kind="text_other",
                                bbox=(5, 5, 70, 30),
                                text="route text",
                            ),),
                        ),),
                        text_slices=(),
                        has_layout_routes=True,
                    ),
                ),
                BlockRoutingPlan(
                    block_uid="row-formula",
                    plan=RoutingPlan(
                        lines=(RoutingLine(
                            index=0,
                            bbox=(75, 5, 115, 30),
                            segments=(RoutingSegment(
                                kind="formula",
                                bbox=(75, 5, 115, 30),
                                text="",
                                content_bbox=(75, 5, 115, 30),
                            ),),
                        ),),
                        text_slices=(),
                        has_layout_routes=True,
                    ),
                ),
            ),
        ),
        rows=(
            CharOcrInputRow(
                block_uid="row-text",
                label="text",
                bbox=(5, 5, 70, 30),
                content="input text",
                ocr_policy=OcrPolicy.TEXT_OCR,
                authorship=BlockSource.AUTO_LAYOUT,
                order=0,
            ),
            CharOcrInputRow(
                block_uid="row-formula",
                label="formula",
                bbox=(75, 5, 115, 30),
                content="stale formula text must not be routed",
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                authorship=BlockSource.AUTO_LAYOUT,
                order=1,
            ),
        ),
    )


def test_hanwang_engine_uses_only_native_request_rows_and_returns_observations() -> None:
    request = _request()
    captured: dict[str, object] = {}

    def fake_runner(image_bgr, native_rows, **kwargs):
        captured["image"] = image_bgr
        captured["rows"] = native_rows
        captured["routing_plan"] = kwargs["routing_plan"]
        assert "page" not in kwargs
        assert [row.block_uid for row in native_rows] == [
            "row-text",
            "row-formula",
        ]
        assert all(isinstance(row, CharOcrInputRow) for row in native_rows)
        assert (
            kwargs["routing_plan"]
            .for_block("row-formula")
            .lines[0]
            .segments[0]
            .text
            == ""
        )
        text_region = CharOcrRegionObservation(
            block_uid="row-text",
            label="text",
            bbox=(5, 5, 70, 30),
            source="fake-native",
            lines=(CharOcrLineObservation(
                text="正文",
                bbox=(5, 5, 70, 30),
                confidence=0.9,
                source="fake-native",
                atoms=(CharOcrAtomObservation(
                    text="正",
                    bbox=(5, 5, 35, 30),
                    confidence=0.8,
                    source="fake-native",
                    candidates=(CharOcrCandidateObservation("正", 0.0),),
                ),),
            ),),
        )
        empty_formula_region = CharOcrRegionObservation(
            block_uid="row-formula",
            label="formula",
            bbox=(75, 5, 115, 30),
            source="fake-native",
            lines=(CharOcrLineObservation(
                text="",
                bbox=(75, 5, 115, 30),
                confidence=0.0,
                source="fake-native",
            ),),
        )
        return (empty_formula_region, text_region), RunStats(n_blocks_total=2)

    result = HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page(
        np.zeros((80, 120, 3), dtype=np.uint8),
        request,
    )

    assert captured["routing_plan"] is request.routing_plan
    assert [region.block_uid for region in result.regions] == [
        "row-text",
        "row-formula",
    ]
    assert result.regions[0].lines[0].atoms[0].text == "正"
    assert result.regions[0].lines[0].atoms[0].candidates[0].confidence == 0.0
    assert result.regions[1].lines[0].text == ""
    assert result.regions[1].lines[0].atoms == ()
    assert result.page_uid == request.page.uid
    assert result.input_fingerprint == request.input_fingerprint


def test_native_text_only_candidate_defaults_to_zero_confidence() -> None:
    import app.engines.hanwang.micro_recblock as module

    atom = module._char_result({
        "text": "candidate-only",
        "bbox": {"left": 5, "top": 5, "right": 30, "bottom": 25},
    })
    line = module._native_line_observation(module._NativeLineResult(
        text="candidate-only",
        bbox=(5, 5, 30, 25),
        chars=[atom],
    ))

    assert line.atoms[0].text == "candidate-only"
    assert line.atoms[0].confidence == 0.0
    assert line.atoms[0].candidates[0].confidence == 0.0


def test_hanwang_engine_has_no_legacy_model_or_page_write_entry() -> None:
    import app.engines.hanwang.micro_recblock as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    forbidden_modules = {
        "app.models",
        "app.models.project",
        "app.models.layout_block_view",
    }
    forbidden_names = {
        "Page",
        "Block",
        "Line",
        "Char",
        "OcrProject",
        "LayoutBlockView",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
            assert not forbidden_names.intersection(
                alias.name for alias in node.names
            )

    assert "runtime_block" not in source
    assert "recognize_page_blocks" not in source
    assert "_compile_layout_ocr_input_plan" not in source
    assert not hasattr(module.HanwangMicroRecBlockEngine, "recognize_page_blocks")
    assert list(inspect.signature(
        module.HanwangMicroRecBlockEngine.recognize_page,
    ).parameters) == ["self", "image_bgr", "request", "progress_callback"]

    with pytest.raises(FrozenInstanceError):
        request = _request()
        request.rows[0].block_uid = "changed"

    assert CharOcrPageResult.__dataclass_params__.frozen
