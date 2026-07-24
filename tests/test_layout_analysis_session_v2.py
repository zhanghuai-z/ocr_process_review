from __future__ import annotations

import ast
from dataclasses import fields, replace
import json
from pathlib import Path

import pytest

from app.adapters.paddle.inline_formula_observations import (
    inline_formula_detector_observations,
)
from app.core.layout_analyzer import LayoutAnalysisError, LayoutAnalyzer
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import (
    DuplicateUidError,
    PageRecord,
    ProjectRecord,
    ProjectSession,
    RecordNotFoundError,
    RevisionConflictError,
)
from app.services.import_service import ImportJobRequest, ImportService
from app.services.layout_analysis_service import LayoutAnalysisService


def _page(session: ProjectSession, path: Path, *, uid: str, page_number: int) -> PageRecord:
    path.write_bytes(f"image:{uid}".encode("ascii"))
    page = PageRecord(
        project_uid=session.project_uid,
        uid=uid,
        image_path=str(path),
        source_path=str(path),
        cache_image_path="",
        thumbnail_path="",
        width=200,
        height=300,
        page_number=page_number,
        source_page_index=page_number - 1,
        status="imported",
        error="",
        image_hash=f"hash-{uid}",
        image_revision=1,
    )
    return session.page_repository.put(page, expected_revision=0)


def _response(*, label: str = "text", x: int = 10) -> dict[str, object]:
    return {
        "logId": "vendor-log",
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {
                                "block_label": label,
                                "block_bbox": [x, 20, x + 80, 80],
                                "block_content": "vendor text must stay in the artifact",
                                "score": 0.91,
                            }
                        ]
                    }
                }
            ]
        },
    }


class FakePaddle:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[bytes, str, str, str]] = []

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        page_uid: str,
        source_run_id: str,
    ) -> dict[str, object]:
        self.calls.append((image_bytes, filename, page_uid, source_run_id))
        return self.responses.pop(0)


def test_import_writes_only_page_records_to_the_project_session(tmp_path: Path):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 48), "white").save(source)
    session = ProjectSession(ProjectRecord("project-import"))

    service = ImportService(tmp_path / "assets")
    result = service.execute(ImportJobRequest(session.project_uid, (str(source),)))
    service.commit(session, result)

    assert result.failure_count == 0
    assert len(result.pages) == 1
    page = result.pages[0]
    assert session.page_repository.get(page.uid) == page
    assert page.width == 32
    assert page.height == 48
    assert page.page_number == 1
    assert page.source_page_index == 0
    assert page.image_revision == 1
    assert "blocks" not in {item.name for item in fields(page)}
    assert "payload_json" not in {item.name for item in fields(page)}


def test_layout_analysis_is_page_scoped_and_keeps_raw_vendor_json_append_only(tmp_path: Path):
    session = ProjectSession(ProjectRecord("project-layout"))
    page_one = _page(session, tmp_path / "one.bin", uid="page-1", page_number=1)
    page_two = _page(session, tmp_path / "two.bin", uid="page-2", page_number=2)
    fake = FakePaddle([_response(x=10), _response(x=100)])
    service = LayoutAnalysisService(fake)

    commits = service.analyze_pages(session, [page_one.uid, page_two.uid])

    assert [commit.page_uid for commit in commits] == ["page-1", "page-2"]
    assert [commit.snapshot.revision for commit in commits] == [1, 1]
    assert len(session.paddle_artifact_repository.all()) == 2
    assert {artifact.page_uid for artifact in session.paddle_artifact_repository.all()} == {
        "page-1",
        "page-2",
    }
    for commit in commits:
        artifact = session.paddle_artifact_repository.get(commit.artifact_uid)
        snapshot = session.layout_repository.get(commit.page_uid, revision=1)
        assert snapshot.artifact_uid == artifact.uid
        assert snapshot.blocks[0].origin.raw_artifact_uid == artifact.uid
        assert json.loads(artifact.payload_json)["result"] == _response(x=10 if commit.page_uid == "page-1" else 100)["result"]
        assert "block_content" not in {item.name for item in fields(snapshot.blocks[0])}

    assert fake.calls[0][2] == "page-1"
    assert fake.calls[1][2] == "page-2"


def test_real_paddle_inline_formula_detectors_enter_the_same_layout_snapshot() -> None:
    fixture = json.loads(
        Path("tests/fixtures/layout/120166-layout-api-fixture.json").read_text(
            encoding="utf-8"
        )
    )
    response = fixture["response"]

    observations = inline_formula_detector_observations(
        response,
        page_width=2356,
        page_height=3424,
    )
    assert len(observations) == 7
    assert observations[0].bbox == BBox.from_xyxy(1466, 1818, 1518, 1869)
    assert observations[0].raw_json_path.endswith("layout_det_res.boxes[10]")
    assert observations[0].parent_raw_index == 9
    assert observations[0].exact_parent_text == ""

    snapshot = LayoutAnalyzer().analyze(
        response,
        page_uid="page-real",
        page_width=2356,
        page_height=3424,
        artifact_uid="artifact-real",
        source_run_id="layout-real",
    )
    inline_blocks = [
        block for block in snapshot.blocks if block.source_label == "inline_formula"
    ]
    assert len(inline_blocks) == 7
    assert all(block.block_type is BlockType.EQUATION for block in inline_blocks)
    assert all(block.ocr_policy is OcrPolicy.PRESERVE_AS_FORMULA for block in inline_blocks)
    assert all(block.origin.raw_index is None for block in inline_blocks)
    assert inline_blocks[0].origin.raw_json_path.endswith("layout_det_res.boxes[10]")


def test_inline_formula_parent_text_requires_an_exact_per_parent_count() -> None:
    response = _response()
    pruned = response["result"]["layoutParsingResults"][0]["prunedResult"]
    pruned["parsing_res_list"][0]["block_content"] = "left $x$ middle $y$ right"
    pruned["layout_det_res"] = {
        "boxes": [
            {"label": "inline_formula", "coordinate": [20, 30, 30, 50]},
            {"label": "inline_formula", "coordinate": [50, 30, 60, 50]},
        ]
    }

    observations = inline_formula_detector_observations(
        response,
        page_width=200,
        page_height=300,
    )

    assert [item.exact_parent_text for item in observations] == ["$x$", "$y$"]


def test_error_response_does_not_partially_persist_vendor_fact(tmp_path: Path):
    session = ProjectSession(ProjectRecord("project-error"))
    page = _page(session, tmp_path / "page.bin", uid="page-1", page_number=1)
    response = {
        "errorCode": 4001,
        "errorMsg": "invalid image",
        "result": {"layoutParsingResults": []},
    }
    service = LayoutAnalysisService(FakePaddle([response]))

    with pytest.raises(LayoutAnalysisError, match="invalid image"):
        service.analyze_page(session, page.uid, source_run_id="failed-run")

    assert session.paddle_artifact_repository.all() == ()
    with pytest.raises(RecordNotFoundError):
        session.layout_repository.get(page.uid)


def test_duplicate_artifact_uid_is_rejected_without_replacing_the_first_fact():
    session = ProjectSession(ProjectRecord("project-artifact"))
    artifact = PaddleArtifact(
        project_uid=session.project_uid,
        uid="artifact-1",
        page_uid="page-1",
        source_engine="paddleocr-vl-1.6",
        source_run_id="run-1",
        image_hash="image-hash",
        payload_json='{"result": {"layoutParsingResults": []}}',
    )
    repository = session.paddle_artifact_repository
    repository.append(artifact)

    with pytest.raises(DuplicateUidError):
        repository.append(artifact)
    assert repository.get("artifact-1") is artifact
    assert repository.all() == (artifact,)


def test_rerun_and_manual_adoption_require_layout_cas_and_keep_artifact_reference(tmp_path: Path):
    session = ProjectSession(ProjectRecord("project-cas"))
    page = _page(session, tmp_path / "page.bin", uid="page-1", page_number=1)
    first_response = _response(x=10)
    second_response = _response(label="table", x=20)
    fake = FakePaddle([first_response, second_response])
    service = LayoutAnalysisService(fake)

    first = service.analyze_page(session, page.uid, expected_revision=0, source_run_id="run-1")
    with pytest.raises(RevisionConflictError):
        service.analyze_page(session, page.uid, expected_revision=0, source_run_id="stale-run")
    assert len(session.paddle_artifact_repository.all()) == 1

    edited_block = replace(
        first.snapshot.blocks[0],
        bbox=BBox(30, 40, 70, 60),
        authorship=BlockSource.USER_EDITED,
    )
    edited_snapshot = replace(
        first.snapshot,
        revision=2,
        source_engine="layout-edit",
        blocks=(edited_block,),
    )
    manual = service.adopt_snapshot(session, edited_snapshot, expected_revision=1)
    assert manual.revision == 2
    assert manual.artifact_uid == first.artifact_uid
    assert manual.blocks[0].authorship is BlockSource.USER_EDITED

    second = service.analyze_page(session, page.uid, expected_revision=2, source_run_id="run-2")
    assert second.snapshot.revision == 3
    assert second.artifact_uid != first.artifact_uid
    assert session.layout_repository.get(page.uid, revision=3).artifact_uid == second.artifact_uid

    adopted = service.adopt_artifact(
        session,
        page.uid,
        first.artifact_uid,
        expected_revision=3,
    )
    assert adopted.revision == 4
    assert adopted.artifact_uid == first.artifact_uid
    assert adopted.blocks[0].origin.raw_artifact_uid == first.artifact_uid


def test_layout_snapshot_fields_are_current_layout_plus_provenance_only():
    assert {item.name for item in fields(LayoutBlockSnapshot)} == {
        "block_type",
        "bbox",
        "order",
        "source_label",
        "origin",
        "ocr_policy",
        "authorship",
        "uid",
    }
    assert {item.name for item in fields(BlockOrigin)} == {
        "created_by",
        "source_engine",
        "source_run_id",
        "vendor_label",
        "source_confidence",
        "original_bbox",
        "original_kind",
        "raw_artifact_uid",
        "raw_json_path",
        "raw_index",
    }


def test_layout_boundary_has_no_qt_legacy_model_or_projection_imports():
    paths = [
        Path("app/core/layout_analyzer.py"),
        Path("app/integrations/paddle/vl_client.py"),
        Path("app/services/import_service.py"),
        Path("app/services/layout_analysis_service.py"),
    ]
    forbidden_text = (
        "PySide6",
        "QThread",
        "Signal",
        "layout_snapshot_store",
        "layout_snapshot_projection",
        "raw_ocr_artifact",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden_text), path
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.models":
                pytest.fail(f"aggregate legacy model import in {path}")
            if isinstance(node, ast.ImportFrom) and node.module == "app.models.project":
                pytest.fail(f"legacy project model import in {path}")
            if isinstance(node, ast.Import):
                assert all(alias.name != "PySide6" for alias in node.names)
