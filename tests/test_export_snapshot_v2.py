from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.export.ir_builder import build_export_ir
from app.models.enums import BlockType, OcrPolicy
from app.models.export_snapshot import ExportPageSnapshot, ExportProjectSnapshot
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_records import OcrLine, OcrRegion, OcrRun
from app.models.project_session import (
    BindingRecord, PageRecord, ProjectRecord, ProjectSession, TableTextRecord,
)
from app.services.export_service import capture_export_snapshot
from app.models.proof_records import ProofAnchorSnapshot, ProofState, ProofTextUnit


PROJECT_UID = "project-1"
PAGE_UID = "page-1"
RUN_UID = "run-1"


def _page() -> PageRecord:
    return PageRecord(
        project_uid=PROJECT_UID,
        uid=PAGE_UID,
        image_path="/input/page-1.png",
        source_path="/input/book.pdf",
        cache_image_path="",
        thumbnail_path="",
        width=1000,
        height=1400,
        page_number=1,
        source_page_index=0,
        status="proof_done",
        error="",
        image_hash="image-hash",
        image_revision=1,
    )


def _block(
    uid: str,
    order: int,
    block_type: BlockType,
    *,
    source_label: str = "",
    policy: OcrPolicy = OcrPolicy.TEXT_OCR,
) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        block_type=block_type,
        bbox=BBox(10, 20 + order * 50, 300, 40),
        order=order,
        source_label=source_label,
        origin=BlockOrigin(created_by="manual_draw", vendor_label=source_label),
        ocr_policy=policy,
        uid=uid,
    )


def _snapshot(
    blocks: tuple[LayoutBlockSnapshot, ...],
    *,
    lines: tuple[OcrLine, ...] = (),
    regions: tuple[OcrRegion, ...] = (),
    bindings: tuple[BindingRecord, ...] = (),
    proof_states: tuple[ProofState, ...] = (),
    table_texts: tuple[TableTextRecord, ...] = (),
) -> ExportProjectSnapshot:
    layout = LayoutSnapshot(
        page_uid=PAGE_UID,
        revision=1,
        artifact_uid="",
        source_engine="test",
        source_run_id="layout-run-1",
        blocks=blocks,
    )
    page = ExportPageSnapshot(
        page=_page(),
        layout=layout,
        ocr_regions=regions,
        ocr_lines=lines,
        bindings=bindings,
        proof_states=proof_states,
        table_texts=table_texts,
    )
    return ExportProjectSnapshot(
        project=ProjectRecord(project_uid=PROJECT_UID, name="Snapshot book"),
        pages=(page,),
    )


def _ocr_facts(block_uid: str = "block-1") -> tuple[OcrLine, OcrRegion, BindingRecord]:
    run = OcrRun(
        project_uid=PROJECT_UID,
        uid=RUN_UID,
        engine="test-ocr",
        layout_fingerprint="layout-fingerprint",
        input_fingerprint="input-fingerprint",
    )
    region = OcrRegion(
        project_uid=PROJECT_UID,
        uid="region-1",
        run_uid=run.uid,
        page_uid=PAGE_UID,
        bbox=(10, 20, 310, 60),
        kind="text",
    )
    line = OcrLine(
        project_uid=PROJECT_UID,
        uid="line-1",
        run_uid=run.uid,
        region_uid=region.uid,
        page_uid=PAGE_UID,
        text="OCR source text",
        bbox=(10, 20, 310, 60),
        confidence=0.91,
    )
    binding = BindingRecord(
        project_uid=PROJECT_UID,
        uid="binding-1",
        source_uid=block_uid,
        target_uid=region.uid,
        relation="observed_by",
        source_fingerprint="layout-fingerprint",
        target_fingerprint=region.fingerprint,
    )
    return line, region, binding


def _proof_state(text: str, status: str = "modified") -> ProofState:
    anchor = ProofAnchorSnapshot(
        project_uid=PROJECT_UID,
        uid="anchor-1",
        scope_uid=PAGE_UID,
        layout_fingerprint="layout-fingerprint",
        source_fingerprint="source-fingerprint",
        anchor_revision=1,
    )
    unit = ProofTextUnit(
        project_uid=PROJECT_UID,
        uid="proof-text-1",
        order=0,
        text=text,
        status=status,
    )
    return ProofState(
        project_uid=PROJECT_UID,
        uid="proof-1",
        anchor_snapshot=anchor,
        text_units=(unit,),
    )


def test_export_modules_have_no_runtime_model_or_projection_dependency():
    paths = sorted(Path("app/export").glob("*.py")) + [Path("app/services/export_service.py")]
    forbidden_names = {
        "OcrProject",
        "Page",
        "Block",
        "Line",
        "Char",
        "LayoutBlockView",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        imported_names = {
            alias.name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not names & forbidden_names, path
        assert not imported_names & forbidden_names, path
        assert "runtime_block" not in source, path
        assert "layout_projection" not in source, path


def test_build_export_ir_rejects_the_retired_runtime_input():
    with pytest.raises(TypeError, match="ExportProjectSnapshot"):
        build_export_ir(object())


def test_capture_export_snapshot_reads_session_once_without_runtime_projection():
    session = ProjectSession(ProjectRecord(PROJECT_UID, "Snapshot book"))
    session.page_repository.put(_page(), expected_revision=0)
    block = _block("block-1", 0, BlockType.TEXT, source_label="text")
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid=PAGE_UID,
            revision=1,
            artifact_uid="",
            source_engine="test",
            source_run_id="layout-run-1",
            blocks=(block,),
        ),
        expected_revision=0,
    )

    captured = capture_export_snapshot(session)

    assert captured.project.project_uid == PROJECT_UID
    assert captured.pages[0].layout.blocks == (block,)


def test_proof_text_has_priority_over_ocr_text():
    block = _block("block-1", 0, BlockType.TEXT, source_label="text")
    line, region, binding = _ocr_facts()
    snapshot = _snapshot(
        (block,),
        lines=(line,),
        regions=(region,),
        bindings=(binding,),
        proof_states=(_proof_state("proof final text"),),
    )

    document = build_export_ir(snapshot, "json").to_dict()
    element = document["pages"][0]["elements"][0]
    payload_line = element["payload"]["lines"][0]

    assert element["payload"]["text"] == "proof final text"
    assert payload_line["text"] == "proof final text"
    assert payload_line["ocr_text"] == "OCR source text"
    assert element["proof"]["corrected"] is True
    assert element["proof"]["status"] == "modified"


def test_layout_snapshot_order_controls_export_order():
    first = _block("block-first", 0, BlockType.TITLE, source_label="heading_1")
    second = _block("block-second", 1, BlockType.TEXT, source_label="text")
    line, region, binding = _ocr_facts(block_uid=second.uid)
    document = build_export_ir(_snapshot(
        (first, second),
        lines=(line,),
        regions=(region,),
        bindings=(binding,),
    ), "json").to_dict()

    elements = document["pages"][0]["elements"]
    assert [element["source"]["block_ids"][0] for element in elements] == [
        "block-first",
        "block-second",
    ]
    assert [element["order"] for element in elements] == [0, 1]
    assert elements[0]["kind"] == "title"
    assert elements[1]["payload"]["text"] == "OCR source text"


def test_structural_elements_survive_without_runtime_blocks_or_lines():
    blocks = (
        _block("figure-1", 0, BlockType.FIGURE, source_label="figure", policy=OcrPolicy.MANUAL_ONLY),
        _block("table-1", 1, BlockType.TABLE, source_label="table", policy=OcrPolicy.PRESERVE_AS_TABLE),
        _block("equation-1", 2, BlockType.EQUATION, source_label="equation", policy=OcrPolicy.PRESERVE_AS_FORMULA),
    )
    table_text = TableTextRecord(
        project_uid=PROJECT_UID,
        uid="cell-1",
        page_uid=PAGE_UID,
        table_uid="table-1",
        row_index=0,
        column_index=0,
        text="table cell",
        source_fingerprint="table-source",
    )

    document = build_export_ir(_snapshot(blocks, table_texts=(table_text,)), "json").to_dict()
    elements = document["pages"][0]["elements"]

    assert [element["kind"] for element in elements] == ["figure", "table", "equation"]
    assert elements[0]["payload"]["asset_ref"] == "asset-el-p1-0001"
    assert elements[1]["payload"]["text"] == "table cell"
    assert elements[1]["payload"]["table_text_layer_cells"][0]["text"] == "table cell"
    assert elements[2]["payload"]["mode"] == "image_fallback"
    assert elements[2]["fallback"]["used"] is True


@pytest.mark.parametrize("fmt", ("txt", "rtf", "docx", "html", "md", "xml", "json", "pdf-single", "pdf-dual"))
def test_snapshot_builds_every_export_format(fmt: str):
    block = _block("block-1", 0, BlockType.TEXT, source_label="text")
    line, region, binding = _ocr_facts()
    snapshot = _snapshot((block,), lines=(line,), regions=(region,), bindings=(binding,))

    document = build_export_ir(snapshot, fmt)

    assert document.profile.format == fmt
    assert document.pages[0].elements[0].payload["text"] == "OCR source text"
