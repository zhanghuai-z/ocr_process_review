"""基础单元测试：无需 OCR 引擎，不启动 GUI。

覆盖：
- 模型基础行为（含新增字段）
- BBox 工具函数
- ProjectStore 保存/加载/迁移/脏数据清理
- ProofAutoFlagService 低置信标记
- TXT/XML/HTML 导出
- Fake OCR/Layout 引擎
- OcrPipeline
- ExportService
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import OcrPolicy
from app.models.ocr_observation import (
    page_has_ocr_result,
    page_ocr_line_count,
    project_all_pages_ocr_done,
    project_has_any_ocr_result,
    project_ocr_line_count,
)

from app.core.line_text_contract import ensure_line_text_contract
from app.core.proof_line_facts import (
    proof_display_text,
    proof_final_text,
    proof_final_text_set,
    proof_runtime_state,
    proof_status,
)
from app.core.proof_line_mutation import apply_line_proof_state, set_line_proof_status, set_line_proof_text


def _paddle_layout_artifact(records):
    from app.models import RawOcrArtifact

    return RawOcrArtifact.from_paddle_layout_records(records)


def _attach_raw_layout_records(page, records):
    from app.core.raw_ocr_artifact import set_paddle_raw_layout_records

    return set_paddle_raw_layout_records(page, records)


def _raw_layout_records(page):
    from app.core.raw_ocr_artifact import raw_layout_records

    return raw_layout_records(page)


def test_layout_projection_boundary_tracks_current_page_blocks():
    from app.models import BBox, Block, BlockType, Page
    from app.models.layout_projection import (
        append_page_layout_block,
        block_belongs_to_page,
        find_page_layout_block_index,
        iter_page_layout_block_occurrences,
        page_has_layout_blocks,
        page_layout_block_count,
        page_layout_blocks,
        replace_page_layout_blocks,
    )

    page = Page(image_path="/tmp/layout-projection.png", width=100, height=80)
    text = Block(block_type=BlockType.TEXT, bbox=BBox(1, 2, 30, 10), order=0)
    formula = Block(block_type=BlockType.EQUATION, bbox=BBox(40, 2, 20, 10), order=1)

    assert not page_has_layout_blocks(page)
    append_page_layout_block(page, text)
    assert page_layout_blocks(page) == [text]
    assert page_layout_block_count(page) == 1
    assert find_page_layout_block_index(page, text) == 0
    assert block_belongs_to_page(page, text)

    replace_page_layout_blocks(page, [formula, text])
    assert page_layout_blocks(page) == [formula, text]
    assert find_page_layout_block_index(page, text) == 1
    assert [
        (occurrence.block, occurrence.block_index)
        for occurrence in iter_page_layout_block_occurrences(page)
    ] == [(formula, 0), (text, 1)]


def test_ocr_character_observation_boundary_tracks_current_line_chars():
    from app.models import BBox, Char, Line
    from app.models.ocr_character_observation import (
        clear_line_ocr_chars,
        iter_line_ocr_char_occurrences,
        line_has_ocr_chars,
        line_ocr_char_at,
        line_ocr_char_count,
        line_ocr_chars,
        replace_line_ocr_char_span,
        replace_line_ocr_chars,
    )

    line = Line(text="甲乙", confidence=0.95, bbox=BBox(0, 0, 20, 10))
    first = Char(char="甲", confidence=0.95, bbox=BBox(0, 0, 10, 10))
    second = Char(char="乙", confidence=0.96, bbox=BBox(10, 0, 10, 10))

    assert not line_has_ocr_chars(line)
    replace_line_ocr_chars(line, [first, second])
    assert line_ocr_chars(line) == [first, second]
    assert line_ocr_char_count(line) == 2
    assert line_ocr_char_at(line, 1) is second
    assert [
        (occurrence.char, occurrence.char_index)
        for occurrence in iter_line_ocr_char_occurrences(line)
    ] == [(first, 0), (second, 1)]

    clear_line_ocr_chars(line)
    assert line_ocr_chars(line) == []
    assert line.chars == []

    replace_line_ocr_chars(line, [first, second])
    third = Char(char="丙", confidence=0.97, bbox=BBox(10, 0, 10, 10))
    replace_line_ocr_char_span(line, 1, 2, [third])
    assert line_ocr_chars(line) == [first, third]
    assert line.chars is line_ocr_chars(line)


def test_ocr_text_observation_boundary_tracks_current_line_text_projection():
    from app.models import BBox, Line
    from app.models.ocr_text_observation import (
        append_line_ocr_review_flag_once,
        line_has_ocr_review_flag,
        line_ocr_review_flags,
        line_ocr_text,
        line_ocr_text_observation,
        set_line_ocr_text_observation,
    )
    from app.models.ocr_text_observation_store import OcrTextObservation

    line = Line(text="OCR", confidence=0.8, bbox=BBox(0, 0, 20, 10), ocr_text="OCR源")

    observation = line_ocr_text_observation(line)
    assert observation.text == "OCR"
    assert observation.ocr_text == "OCR源"
    assert line_ocr_text(line) == "OCR源"

    set_line_ocr_text_observation(
        line,
        OcrTextObservation(
            text="新OCR",
            ocr_text="新OCR源",
            confidence=0.9,
            review_flags=("flag-a", "flag-a", "flag-b"),
        ),
    )
    assert line.text == "新OCR"
    assert line.ocr_text == "新OCR源"
    assert line.confidence == 0.9
    assert line.review_flags == ["flag-a", "flag-b"]
    assert line_ocr_review_flags(line) == ("flag-a", "flag-b")

    line.text = "旧投影直改"
    line.ocr_text = "旧投影源"
    line.review_flags = ["legacy"]
    adopted = line_ocr_text_observation(line)
    assert adopted.text == "旧投影直改"
    assert adopted.ocr_text == "旧投影源"
    assert line_ocr_review_flags(line) == ("legacy",)

    append_line_ocr_review_flag_once(line, "legacy")
    append_line_ocr_review_flag_once(line, "new-flag")
    assert line_has_ocr_review_flag(line, "new-flag")
    assert line.review_flags == ["legacy", "new-flag"]


def test_ocr_observation_runtime_store_keeps_block_projection_in_sync():
    from app.models import BBox, Block, BlockType, Line
    from app.models.ocr_observation import (
        append_block_ocr_line,
        block_ocr_lines,
        clear_block_ocr_lines,
        replace_block_ocr_lines,
    )

    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40))
    first = Line(text="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10))
    second = Line(text="乙", confidence=0.8, bbox=BBox(0, 12, 10, 10))

    replace_block_ocr_lines(block, [first])
    assert block_ocr_lines(block) == [first]
    assert block.lines is block_ocr_lines(block)

    append_block_ocr_line(block, second)
    assert block_ocr_lines(block) == [first, second]
    assert block.lines == [first, second]

    clear_block_ocr_lines(block)
    assert block_ocr_lines(block) == []
    assert block.lines == []


def test_ocr_observation_store_adopts_replaced_legacy_projections():
    from app.models import BBox, Block, BlockType, Char, Line
    from app.models.ocr_character_observation import line_ocr_chars, replace_line_ocr_chars
    from app.models.ocr_observation import block_ocr_lines, replace_block_ocr_lines

    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40))
    line1 = Line(text="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10))
    line2 = Line(text="乙", confidence=0.8, bbox=BBox(0, 12, 10, 10))
    replace_block_ocr_lines(block, [line1])

    block.lines = [line2]
    assert block_ocr_lines(block) == [line2]

    char1 = Char(char="甲", confidence=0.9)
    char2 = Char(char="乙", confidence=0.8)
    replace_line_ocr_chars(line2, [char1])

    line2.chars = [char2]
    assert line_ocr_chars(line2) == [char2]


def test_raw_block_payload_prefers_page_artifact_origin_record():
    from app.core.raw_ocr_artifact import raw_block_payload, raw_block_text_values
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    page = Page(image_path="/tmp/raw-origin.png", width=100, height=80)
    _attach_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_content": "artifact text",
            "block_bbox": [1, 2, 30, 20],
        },
        {
            "block_label": "table",
            "block_content": "<table><tr><td>A</td></tr></table>",
            "block_bbox": [5, 6, 40, 30],
        },
    ])
    block = Block(
        block_type=BlockType.TABLE,
        bbox=BBox.from_xyxy(5, 6, 40, 30),
        origin=BlockOrigin(source_label="table", raw_index=1),
    )

    assert raw_block_payload(block, page)["block_content"] == "<table><tr><td>A</td></tr></table>"
    assert raw_block_text_values(block, ("block_content",), page) == ["<table><tr><td>A</td></tr></table>"]
    assert raw_block_payload(block) == {}

    print("test_raw_block_payload_prefers_page_artifact_origin_record PASSED")


# =====================================================================
# 模型测试
# =====================================================================

def test_models():
    from app.models import (
        BBox, Block, BlockSource, BlockType, Char, Line,
        OcrProject, Page, PageStatus, ProofLineState, ProofStatus,
    )
    from app.models.page_state import (
        clear_page_ocr_invalidation,
        invalidate_page_ocr,
        page_is_ocr_done,
        page_needs_ocr_rerun,
    )

    # BBox
    bb = BBox(10, 20, 100, 30)
    assert bb.to_xyxy() == (10, 20, 110, 50)
    assert BBox.from_xyxy(0, 0, 50, 80) == BBox(0, 0, 50, 80)
    assert bb.area == 3000
    assert bb.x1 == 10 and bb.y1 == 20 and bb.x2 == 110 and bb.y2 == 50

    # BBox clamp
    clamped = BBox(-10, -5, 200, 300).clamp(100, 100)
    assert clamped.x >= 0 and clamped.y >= 0
    assert clamped.x2 <= 100 and clamped.y2 <= 100

    # BBox normalize
    norm = BBox(50, 60, -20, -30).normalize()
    assert norm.w == 20 and norm.h == 30
    assert norm.x == 30 and norm.y == 30

    # BBox expand
    expanded = BBox(10, 10, 100, 50).expand(5)
    assert expanded.x == 5 and expanded.y == 5
    assert expanded.w == 110 and expanded.h == 60

    # BBox iou
    a = BBox(0, 0, 100, 100)
    b = BBox(50, 50, 100, 100)
    assert 0.0 < a.iou(b) < 1.0
    assert a.iou(a) == 1.0
    assert BBox(0, 0, 10, 10).iou(BBox(100, 100, 10, 10)) == 0.0

    # Line
    line = Line(text="测试文字", confidence=0.95, bbox=bb)
    assert line.uid.startswith("line_")
    assert proof_status(line) == ProofStatus.UNCHECKED
    assert line.review_flags == []
    assert not hasattr(line, "proof_state")
    set_line_proof_text(line, "修改文字")
    assert proof_status(line) == ProofStatus.MODIFIED
    state = proof_runtime_state(line)
    assert state.final_text == "修改文字"
    assert state.final_text_set is True
    assert state.proof_status == ProofStatus.MODIFIED
    apply_line_proof_state(
        line,
        ProofLineState(
            line_uid=line.uid,
            final_text="状态对象终稿",
            final_text_set=True,
            proof_status=ProofStatus.MODIFIED,
        )
    )
    ensure_line_text_contract(line)
    assert proof_display_text(line) == "状态对象终稿"
    assert proof_final_text(line) == "状态对象终稿"
    assert not hasattr(line, "final_text")
    assert not hasattr(line, "proof_state")
    assert proof_runtime_state(line).final_text == "状态对象终稿"
    assert proof_display_text(line) == "状态对象终稿"
    set_line_proof_text(line, "修改文字")

    # Line with new fields
    line2 = Line(
        text="终稿", confidence=0.85, bbox=bb,
        ocr_text="OCR原文",
        review_flags=["low_confidence"],
    )
    assert line2.ocr_text == "OCR原文"

    char = Char(char="测", confidence=0.9, bbox=bb)
    assert char.uid.startswith("char_")

    proof_state = ProofLineState(line_uid=line.uid, final_text="修改文字", final_text_set=True)
    assert proof_state.line_uid == line.uid
    assert proof_state.proof_status == ProofStatus.UNCHECKED

    # Block
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line, line2])
    assert block.uid.startswith("block_")
    from app.core.proof_line_facts import proof_block_text

    assert proof_block_text(block) == "修改文字\n终稿"
    assert block.source == BlockSource.AUTO_LAYOUT
    assert block.ocr_policy == OcrPolicy.TEXT_OCR

    # Block new fields
    block2 = Block(
        block_type=BlockType.TABLE, bbox=bb,
        source=BlockSource.MANUAL_DRAW,
        ocr_policy=OcrPolicy.MANUAL_ONLY, note="测试备注",
    )
    assert block2.source == BlockSource.MANUAL_DRAW
    assert block2.ocr_policy != OcrPolicy.TEXT_OCR
    assert block2.note == "测试备注"

    # Page
    page = Page(image_path="/tmp/test.jpg", width=800, height=1200)
    assert page.uid.startswith("page_")
    page.blocks.append(block)
    formula_block = Block(block_type=BlockType.EQUATION, bbox=bb, ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA)
    page.blocks.append(formula_block)
    from app.models.layout_projection import page_has_layout_blocks

    assert page_has_layout_blocks(page)
    assert page.status == PageStatus.IMPORTED
    assert page.source_type == "image"
    from app.services.ocr_dispatch_plan import build_text_ocr_dispatch_plan

    assert list(build_text_ocr_dispatch_plan(page).text_block_models) == [block]

    # Page new fields
    page2 = Page(
        image_path="/tmp/cache.png", width=800, height=1200,
        source_path="/tmp/original.pdf", source_type="pdf",
        source_page_index=1, cache_image_path="/tmp/cache.png",
        status=PageStatus.LAYOUT_DONE,
    )
    assert page2.source_type == "pdf"
    assert page2.status == PageStatus.LAYOUT_DONE
    assert page2.display_image_path == "/tmp/cache.png"

    # Project
    project = OcrProject(name="测试项目", pages=[page])
    assert project.page_count == 1
    assert project_ocr_line_count(project) == 2
    assert project_has_any_ocr_result(project) is True
    assert project_all_pages_ocr_done(project) is False
    page.status = PageStatus.OCR_DONE
    assert page_has_ocr_result(page) is True
    assert page_is_ocr_done(page) is True
    page.status = PageStatus.PROOFING
    assert page_is_ocr_done(page) is True
    assert project_all_pages_ocr_done(project) is True
    invalidate_page_ocr(page, "block_moved")
    assert page_needs_ocr_rerun(page) is True
    assert page.ocr_invalidated_reason == "block_moved"
    clear_page_ocr_invalidation(page)
    assert page_needs_ocr_rerun(page) is False

    # Export summary lives outside the project model; proof stats are service-owned.
    from app.services.export_service import build_export_summary

    summary = build_export_summary(project)
    assert summary["total_pages"] == 1
    assert summary["total_lines"] >= 0
    assert summary["unrecognized_blocks"] == 0

    print("test_models PASSED")


def test_proof_line_facts_rejects_legacy_field_only_objects():
    import pytest
    from types import SimpleNamespace

    from app.core.proof_line_facts import proof_runtime_state
    from app.models import ProofStatus

    legacy_like = SimpleNamespace(
        uid="line_legacy",
        final_text="旧终稿",
        final_text_set=True,
        proof_status=ProofStatus.MODIFIED,
    )

    with pytest.raises(TypeError, match="proof_state"):
        proof_runtime_state(legacy_like)


def test_proof_state_store_rejects_retired_line_proof_state_attr():
    import pytest

    from app.core.proof_line_facts import proof_runtime_state
    from app.models import BBox, Line, ProofLineState

    line = Line(text="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10))
    line.__dict__["proof_state"] = ProofLineState(line_uid=line.uid)

    with pytest.raises(TypeError, match="retired"):
        proof_runtime_state(line)


def test_workflow_state_keeps_project_and_page_ocr_state_separate():
    from app.core.workflow_state import (
        STEP_OCR,
        STEP_VPROOF,
        compute_max_step,
        page_gate_info,
        pending_ocr_pages,
    )
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
    from app.models.page_state import invalidate_page_ocr, page_is_ocr_done

    done_page = Page(
        image_path="/tmp/done.png",
        width=100,
        height=100,
        status=PageStatus.OCR_DONE,
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(0, 0, 80, 20),
                lines=[Line(text="已识别", confidence=0.9, bbox=BBox(0, 0, 80, 20))],
            )
        ],
    )
    pending_page = Page(
        image_path="/tmp/pending.png",
        width=100,
        height=100,
        status=PageStatus.LAYOUT_DONE,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 80, 20))],
    )
    project = OcrProject(name="mixed", pages=[done_page, pending_page])

    assert project_has_any_ocr_result(project) is True
    assert project_all_pages_ocr_done(project) is False
    assert compute_max_step(project) == STEP_VPROOF
    assert page_gate_info(done_page).is_pending is False
    assert page_gate_info(pending_page).page_state == "ocr_ready"
    assert pending_ocr_pages(project) == [pending_page]

    lines_without_done_status = Page(
        image_path="/tmp/legacy-ish.png",
        width=100,
        height=100,
        status=PageStatus.LAYOUT_DONE,
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(0, 0, 80, 20),
                lines=[Line(text="有行但状态未完成", confidence=0.9, bbox=BBox(0, 0, 80, 20))],
            )
        ],
    )
    assert page_has_ocr_result(lines_without_done_status) is True
    assert page_is_ocr_done(lines_without_done_status) is False
    assert compute_max_step(OcrProject(name="lines-only", pages=[lines_without_done_status])) == STEP_OCR
    assert page_gate_info(lines_without_done_status).page_state == "ocr_ready"

    errored_page = Page(
        image_path="/tmp/error.png",
        width=100,
        height=100,
        status=PageStatus.LAYOUT_DONE,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 80, 20))],
        error_message="layout failed",
    )
    errored_gate = page_gate_info(errored_page)
    assert errored_gate.page_state == "error"
    assert errored_gate.is_pending is False
    assert errored_gate.action_enabled is False
    assert pending_ocr_pages(OcrProject(name="error-page", pages=[errored_page])) == []

    ocr_error_page = Page(
        image_path="/tmp/ocr-error.png",
        width=100,
        height=100,
        status=PageStatus.ERROR,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 80, 20))],
        error_message="OCR 失败：micro-recblock failed",
    )
    ocr_error_gate = page_gate_info(ocr_error_page)
    assert ocr_error_gate.page_state == "ocr_error"
    assert ocr_error_gate.is_pending is True
    assert ocr_error_gate.action_enabled is True
    assert pending_ocr_pages(OcrProject(name="ocr-error-page", pages=[ocr_error_page])) == [ocr_error_page]

    invalidate_page_ocr(pending_page, "block_moved")
    invalidated = page_gate_info(pending_page)
    assert invalidated.page_state == "ocr_invalidated"
    assert invalidated.action_enabled is True

    print("test_workflow_state_keeps_project_and_page_ocr_state_separate PASSED")


# =====================================================================
# BBox 工具测试
# =====================================================================

def test_bbox_tools():
    import cv2
    import numpy as np

    from app.core.bbox_utils import (
        bbox_from_xyxy, is_crop_relative_bbox, project_line_bbox, sanitize_xyxy_bbox, scale_bbox,
    )
    from app.core.char_bbox_utils import (
        LINE_DIRECTION_HORIZONTAL, LINE_DIRECTION_VERTICAL,
        ensure_line_char_bboxes, infer_bbox_direction, infer_line_direction,
        refine_line_bbox, refine_line_char_bboxes,
        split_line_bbox_into_char_bboxes,
    )
    from app.core.coordinate_seam import (
        BBOX_SPACE_CROP, BBOX_SPACE_PAGE, CropCoordinateSeam,
    )
    from app.models import BBox, Char, Line

    # from_dict / to_dict round-trip
    d = {"x": 10, "y": 20, "w": 100, "h": 50}
    bb = BBox.from_dict(d)
    assert bb.to_dict() == d

    # clamp edge cases
    bb = BBox(-10, -5, 50, 60)
    c = bb.clamp(100, 100)
    assert c.x == 0 and c.y == 0

    bb2 = BBox(80, 90, 50, 60).clamp(100, 100)
    assert bb2.x2 <= 100 and bb2.y2 <= 100

    parsed = bbox_from_xyxy([120.4, 30.2, 10.1, 60.9])
    assert parsed == BBox(10, 30, 110, 31)

    sanitized = sanitize_xyxy_bbox([-10, -5, 120, 60], 100, 50)
    assert sanitized == BBox(0, 0, 100, 50)

    assert is_crop_relative_bbox(BBox(5, 5, 40, 10), 100, 50) is True
    assert is_crop_relative_bbox(BBox(105, 5, 40, 10), 100, 50) is False

    page_bbox = project_line_bbox(
        BBox(10, 5, 40, 10),
        crop_origin_x=100,
        crop_origin_y=50,
        crop_w=80,
        crop_h=40,
        page_w=400,
        page_h=300,
        source_space=BBOX_SPACE_CROP,
    )
    assert page_bbox == BBox(110, 55, 40, 10)

    seam = CropCoordinateSeam.from_page_bbox(BBox(100, 50, 120, 80), page_w=400, page_h=300)
    assert seam.to_page_bbox(BBox(12, 8, 40, 12), source_space=BBOX_SPACE_CROP) == BBox(112, 58, 40, 12)
    assert seam.to_page_bbox(BBox(120, 70, 40, 12), source_space=BBOX_SPACE_PAGE) == BBox(120, 70, 40, 12)
    assert seam.to_crop_bbox(BBox(120, 70, 40, 12)) == BBox(20, 20, 40, 12)

    assert infer_line_direction(BBox(10, 20, 160, 24), 4) == LINE_DIRECTION_HORIZONTAL
    assert infer_line_direction(BBox(10, 20, 24, 160), 4) == LINE_DIRECTION_VERTICAL
    assert infer_bbox_direction(BBox(10, 20, 160, 24)) == LINE_DIRECTION_HORIZONTAL
    assert infer_bbox_direction(BBox(10, 20, 24, 160)) == LINE_DIRECTION_VERTICAL

    horizontal_chars = split_line_bbox_into_char_bboxes(BBox(10, 20, 160, 24), "天地玄黄")
    assert horizontal_chars[0] == BBox(10, 20, 40, 24)
    assert horizontal_chars[-1] == BBox(130, 20, 40, 24)

    vertical_chars = split_line_bbox_into_char_bboxes(BBox(10, 20, 24, 160), "天地玄黄")
    assert vertical_chars[0] == BBox(10, 20, 24, 40)
    assert vertical_chars[-1] == BBox(10, 140, 24, 40)

    line = Line(text="天地玄黄", confidence=0.9, bbox=BBox(10, 20, 24, 160))
    ensure_line_char_bboxes(line)
    assert len(line.chars) == 4
    assert line.chars[1].bbox == BBox(10, 60, 24, 40)

    shifted = Line(
        text="甲乙丙",
        confidence=0.9,
        bbox=BBox(10, 20, 90, 30),
        chars=[
            Char(
                char="甲", confidence=0.9, bbox=BBox(40, 20, 30, 30),
                bbox_source="ocr", bbox_granularity="char", token_text="甲",
            ),
            Char(
                char="乙", confidence=0.9, bbox=BBox(40, 20, 30, 30),
                bbox_source="ocr", bbox_granularity="char", token_text="乙",
            ),
            Char(
                char="丙", confidence=0.9, bbox=BBox(70, 20, 30, 30),
                bbox_source="ocr", bbox_granularity="char", token_text="丙",
            ),
        ],
    )
    ensure_line_char_bboxes(shifted)
    assert shifted.chars[0].bbox == BBox(40, 20, 30, 30)
    assert shifted.chars[0].bbox_source == "ocr"
    assert shifted.chars[1].bbox == BBox(40, 20, 30, 30)
    assert shifted.chars[1].bbox_source == "ocr"

    fallback_shifted = Line(
        text="甲乙丙",
        confidence=0.9,
        bbox=BBox(10, 20, 90, 30),
        chars=[
            Char(
                char="甲", confidence=0.9, bbox=BBox(40, 20, 30, 30),
                bbox_source="fallback", bbox_granularity="char", token_text="甲",
            ),
            Char(
                char="乙", confidence=0.9, bbox=BBox(40, 20, 30, 30),
                bbox_source="fallback", bbox_granularity="char", token_text="乙",
            ),
            Char(
                char="丙", confidence=0.9, bbox=BBox(70, 20, 30, 30),
                bbox_source="fallback", bbox_granularity="char", token_text="丙",
            ),
        ],
    )
    ensure_line_char_bboxes(fallback_shifted)
    assert fallback_shifted.chars[0].bbox == BBox(10, 20, 30, 30)
    assert fallback_shifted.chars[0].bbox_source == "fallback"

    img = np.full((100, 180, 3), 255, dtype=np.uint8)
    glyph_boxes = [
        BBox(16, 34, 12, 20),
        BBox(43, 34, 24, 20),
        BBox(84, 34, 8, 20),
        BBox(108, 34, 28, 20),
    ]
    for bb in glyph_boxes:
        cv2.rectangle(img, (bb.x, bb.y), (bb.x2, bb.y2), (0, 0, 0), -1)

    refined = refine_line_char_bboxes(BBox(10, 28, 132, 32), "甲乙丙丁", img)
    for got, expected in zip(refined, glyph_boxes):
        assert abs(got.x - expected.x) <= 3
        assert abs(got.w - expected.w) <= 4
        assert abs(got.y - expected.y) <= 3
        assert abs(got.h - expected.h) <= 4

    line = Line(text="甲乙丙丁", confidence=0.95, bbox=BBox(10, 28, 132, 32))
    ensure_line_char_bboxes(line, page_image=img)
    for got, expected in zip(line.chars, glyph_boxes):
        assert abs(got.bbox.x - expected.x) <= 3
        assert abs(got.bbox.w - expected.w) <= 4

    line_img = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(line_img, (18, 18), (158, 30), (0, 0, 0), -1)
    cv2.rectangle(line_img, (22, 62), (150, 76), (0, 0, 0), -1)
    refined_line = refine_line_bbox(BBox(10, 44, 160, 40), line_img)
    assert abs(refined_line.y - 61) <= 3
    assert refined_line.h <= 18
    assert refined_line.w >= 120

    scaled = scale_bbox(BBox(10, 20, 30, 40), 2.0, 1.5)
    assert scaled == BBox(20, 30, 60, 60)

    print("test_bbox_tools PASSED")


def test_block_type_mapping():
    from app.adapters.paddle import map_paddle_label_to_block_type
    from app.models import BlockType
    from app.core.paddle_labels import normalize_paddle_label

    assert map_paddle_label_to_block_type("paragraph") == BlockType.TEXT
    assert map_paddle_label_to_block_type("doc_title") == BlockType.TITLE
    assert map_paddle_label_to_block_type("section_title") == BlockType.TITLE
    assert map_paddle_label_to_block_type("heading_1") == BlockType.TITLE
    assert map_paddle_label_to_block_type("heading_6") == BlockType.TITLE
    assert map_paddle_label_to_block_type("image_caption") == BlockType.FIGURE_CAPTION
    assert map_paddle_label_to_block_type("table_caption_text") == BlockType.TABLE_CAPTION
    assert map_paddle_label_to_block_type("table_body") == BlockType.TABLE
    assert map_paddle_label_to_block_type("graphic") == BlockType.FIGURE
    assert map_paddle_label_to_block_type("display_formula") == BlockType.EQUATION
    assert map_paddle_label_to_block_type("isolated_formula") == BlockType.EQUATION
    assert map_paddle_label_to_block_type("math_formula") == BlockType.EQUATION
    assert map_paddle_label_to_block_type("bibliography") == BlockType.REFERENCE
    assert map_paddle_label_to_block_type("vision_footnote") == BlockType.TEXT
    assert normalize_paddle_label("vision_footnote") == "footnote"
    assert map_paddle_label_to_block_type("custom_formula_noise") == BlockType.UNKNOWN
    assert map_paddle_label_to_block_type("not_table") == BlockType.UNKNOWN
    assert map_paddle_label_to_block_type("untitled_region") == BlockType.UNKNOWN
    assert map_paddle_label_to_block_type("reference_like") == BlockType.UNKNOWN

    print("test_block_type_mapping PASSED")


def test_block_state_helpers_use_typed_state_only():
    from app.models.block_state import (
        clear_ocr_text_invalidation,
        is_ocr_text_invalidated,
        mark_ocr_text_invalidated,
        ocr_invalidation_reason,
        set_paddle_binding,
    )
    from app.models import BBox, Block, BlockType, PaddleBinding

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 10, 10),
    )

    set_paddle_binding(block, {"status": "paddle_geometry_hit", "text": "$ A $"})
    assert isinstance(block.paddle_binding, PaddleBinding)
    assert block.paddle_binding.text == "$ A $"

    mark_ocr_text_invalidated(block, "block_moved")
    assert is_ocr_text_invalidated(block) is True
    assert ocr_invalidation_reason(block) == "block_moved"
    assert block.ocr_invalidated_reason == "block_moved"
    clear_ocr_text_invalidation(block)
    assert is_ocr_text_invalidated(block) is False

    print("test_block_state_helpers_use_typed_state_only PASSED")


def test_paddle_layout_schema_normalizes_record_fields():
    from app.core.paddle_layout_schema import (
        normalize_paddle_layout_record,
        paddle_record_text,
        raw_bbox_max_from_record,
        route_subblock_payload,
    )
    from app.core.paddle_line_routing import block_text

    record = {
        "block_label": "inline_formula",
        "label": "text",
        "block_bbox": [10, 20, 110, 60],
        "block_content": " $ A $ ",
        "score": "0.91",
        "custom_raw": {"keep": True},
    }

    normalized = normalize_paddle_layout_record(
        record,
        page_width=500,
        page_height=500,
        scale_x=2.0,
        scale_y=0.5,
    )

    assert normalized is not None
    assert normalized.label == "inline_formula"
    assert normalized.normalized_label == "inline_formula"
    assert normalized.bbox.to_xyxy() == (20, 10, 220, 30)
    assert normalized.text == "$ A $"
    assert normalized.score == 0.91
    assert normalized.raw["custom_raw"]["keep"] is True
    assert normalized.signature == ("inline_formula", 20, 10, 200, 20)
    assert raw_bbox_max_from_record(record) == (110, 60)

    payload = route_subblock_payload(normalized)
    assert payload["block_label"] == "inline_formula"
    assert payload["block_bbox"] == [20, 10, 220, 30]
    assert payload["raw_payload"]["label"] == "text"
    assert paddle_record_text({"block_content": " preview "}) == "preview"
    assert paddle_record_text({"markdown": " preview "}) == ""
    assert block_text({"markdown": "must not route"}) == ""

    print("test_paddle_layout_schema_normalizes_record_fields PASSED")


def test_ocr_run_wraps_ir_lines_without_proof_model():
    from app.core.ocr_ir import OcrIrLine, OcrIrToken, OcrLine, OcrToken, build_ocr_run
    from app.models import BBox

    token = OcrIrToken(
        text="甲",
        bbox=BBox(1, 2, 3, 4),
        row_index=0,
        token_index=0,
        confidence=0.9,
        kind="text",
    )
    line = OcrIrLine(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 2, 20, 10),
        source_text="test",
        tokens=[token],
    )
    run = build_ocr_run(
        engine="hanwang",
        engine_version="native",
        page_uid="page_1",
        block_uid="block_1",
        lines=[line],
        input_layout_revision=7,
    )

    assert run.uid.startswith("ocrrun_")
    assert run.page_uid == "page_1"
    assert run.block_uid == "block_1"
    assert run.input_layout_revision == 7
    assert run.lines == [line]
    assert OcrLine is OcrIrLine
    assert OcrToken is OcrIrToken
    assert not hasattr(run.lines[0], "proof_state")

    print("test_ocr_run_wraps_ir_lines_without_proof_model PASSED")


def test_block_origin_label_is_authoritative_for_attributes_and_dispatch():
    from app.core.block_attributes import block_attributes
    from app.core.ocr_dispatch_policy import authoritative_block_label
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 100, 20),
        source_label="stale_text",
        origin=BlockOrigin(source_label="footnote"),
    )

    assert authoritative_block_label(block) == "footnote"
    attrs = block_attributes(block)
    assert attrs.source_label == "footnote"
    assert attrs.raw_label == ""

    block.source = BlockSource.USER_EDITED
    block.source_label = "inline_formula"
    attrs = block_attributes(block)
    assert attrs.source_label == "footnote"
    assert attrs.origin_label == "footnote"
    assert attrs.current_label == "inline_formula"
    assert attrs.semantic_label == "inline_formula"
    assert authoritative_block_label(block) == "inline_formula"

    print("test_block_origin_label_is_authoritative_for_attributes_and_dispatch PASSED")


# =====================================================================
# ProjectStore 测试
# =====================================================================

def test_project_store():
    from app.models import BBox, Block, BlockType, Char, Line, OcrPolicy, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(
            text="Hello OCR",
            confidence=0.92,
            bbox=bb,
            chars=[
                Char(
                    char="H",
                    confidence=0.92,
                    bbox=BBox(0, 0, 12, 20),
                    bbox_source="ocr",
                    bbox_granularity="char",
                    token_text="H",
                ),
                Char(
                    char="e",
                    confidence=0.92,
                    bbox=BBox(12, 0, 18, 20),
                    bbox_source="ocr",
                    bbox_granularity="word",
                    token_text="ello",
                ),
            ],
        )
        block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
        page = Page(image_path="/tmp/img.jpg", width=800, height=600)
        page.blocks.append(block)
        project = OcrProject(name="存储测试", pages=[page])

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            assert saved.id == 1
            assert saved.pages[0].id is not None
            loaded = store.load_project(project_id=1)
            assert loaded.name == "存储测试"
            assert loaded.pages[0].blocks[0].lines[0].text == "Hello OCR"
            loaded_chars = loaded.pages[0].blocks[0].lines[0].chars
            assert loaded_chars[0].bbox_source == "ocr"
            assert loaded_chars[0].bbox_granularity == "char"
            assert loaded_chars[1].token_text == "ello"

        print("test_project_store PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_raw_layout_artifact():
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.core.raw_ocr_artifact import (
        layout_records_with_route_attachments,
        layout_route_attachments,
        raw_block_payload,
    )
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockOrigin, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        parsing_res_list = [
            {
                "block_label": "text",
                "block_bbox": [10, 20, 110, 60],
                "block_content": "天地玄黄",
                ROUTE_SUBBLOCKS_FIELD: [
                    {"block_label": "inline_formula", "block_bbox": [30, 24, 50, 42]},
                ],
            },
            {
                "block_label": "display_formula",
                "block_bbox": [20, 80, 180, 120],
                "block_content": "$$x+y$$",
            },
        ]
        project = OcrProject(
            name="paddle-raw-artifact",
            pages=[
                Page(
                    image_path="/tmp/img.jpg",
                    width=800,
                    height=600,
                    raw_layout_artifact=_paddle_layout_artifact(parsing_res_list),
                    blocks=[
                        Block(
                            block_type=BlockType.TEXT,
                            bbox=BBox(10, 20, 100, 40),
                            source_label="paragraph_title",
                            origin=BlockOrigin(source_label="text", raw_index=0),
                        )
                    ],
                )
            ],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)

        artifact = loaded.pages[0].raw_layout_artifact
        assert artifact is not None
        assert artifact.engine == "paddleocr-vl"
        assert artifact.engine_version == "1.6"
        expected_records = [dict(record) for record in parsing_res_list]
        expected_records[0].pop(ROUTE_SUBBLOCKS_FIELD)
        assert _raw_layout_records(loaded.pages[0]) == expected_records
        assert layout_route_attachments(loaded.pages[0])[0] == [
            {"block_label": "inline_formula", "block_bbox": [30, 24, 50, 42]},
        ]
        assert layout_records_with_route_attachments(loaded.pages[0])[0][ROUTE_SUBBLOCKS_FIELD] == [
            {"block_label": "inline_formula", "block_bbox": [30, 24, 50, 42]},
        ]
        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.source_label == "paragraph_title"
        assert raw_block_payload(loaded_block, loaded.pages[0])["block_content"] == "天地玄黄"

        print("test_project_store_persists_raw_layout_artifact PASSED")
    finally:
        os.unlink(db_path)


def test_raw_ocr_artifact_rejects_malformed_route_subblocks():
    import pytest

    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.models import RawOcrArtifact

    with pytest.raises(ValueError, match=r"records\[0\]\._route_subblocks\[0\] must be dict"):
        RawOcrArtifact.from_paddle_layout_records([
            {
                "block_label": "text",
                "block_bbox": [10, 20, 110, 60],
                ROUTE_SUBBLOCKS_FIELD: ["not-a-dict"],
            }
        ])

    with pytest.raises(ValueError, match=r"route_attachments\[0\]\[0\] must be dict"):
        RawOcrArtifact.from_paddle_layout_records(
            [{"block_label": "text", "block_bbox": [10, 20, 110, 60]}],
            route_attachments={0: ["not-a-dict"]},
        )

    print("test_raw_ocr_artifact_rejects_malformed_route_subblocks PASSED")


def test_project_store_rejects_malformed_route_attachments_on_save_and_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page, RawOcrArtifact

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bad_artifact = RawOcrArtifact(
            engine="paddleocr-vl",
            engine_version="1.6",
            records=[{"block_label": "text", "block_bbox": [10, 20, 110, 60]}],
            route_attachments={0: ["not-a-dict"]},
        )
        bad_project = OcrProject(
            name="bad-route-attachments-save",
            pages=[
                Page(
                    image_path="/tmp/bad-route-save.png",
                    width=120,
                    height=80,
                    raw_layout_artifact=bad_artifact,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 100, 40))],
                )
            ],
        )
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"raw_ocr_artifact\.route_attachments\[0\]\[0\] must be dict"):
                store.save_project(bad_project)

        good_project = OcrProject(
            name="bad-route-attachments-load",
            pages=[
                Page(
                    image_path="/tmp/bad-route-load.png",
                    width=120,
                    height=80,
                    raw_layout_artifact=RawOcrArtifact.from_paddle_layout_records([
                        {"block_label": "text", "block_bbox": [10, 20, 110, 60]}
                    ]),
                    blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 100, 40))],
                )
            ],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(good_project)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE raw_ocr_artifact SET route_attachments_json=?",
                (json.dumps({"0": ["not-a-dict"]}, ensure_ascii=False),),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"raw_ocr_artifact\.route_attachments_json\['0'\]\[0\] must be dict"):
                store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_malformed_route_attachments_on_save_and_load PASSED")


def test_project_store_rejects_non_dict_raw_layout_records_on_save_and_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page, RawOcrArtifact

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bad_project = OcrProject(
            name="bad-raw-records-save",
            pages=[
                Page(
                    image_path="/tmp/bad-record-save.png",
                    width=120,
                    height=80,
                    raw_layout_artifact=RawOcrArtifact(
                        engine="paddleocr-vl",
                        engine_version="1.6",
                        records=["not-a-dict"],
                    ),
                    blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 100, 40))],
                )
            ],
        )
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"raw_ocr_artifact\.records\[0\] must be dict"):
                store.save_project(bad_project)

        good_project = OcrProject(
            name="bad-raw-records-load",
            pages=[
                Page(
                    image_path="/tmp/bad-record-load.png",
                    width=120,
                    height=80,
                    raw_layout_artifact=RawOcrArtifact.from_paddle_layout_records([
                        {"block_label": "text", "block_bbox": [10, 20, 110, 60]}
                    ]),
                    blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 100, 40))],
                )
            ],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(good_project)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE raw_ocr_artifact SET records_json=?",
                (json.dumps(["not-a-dict"], ensure_ascii=False),),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"raw_ocr_artifact\.records_json\[0\] must be dict"):
                store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_non_dict_raw_layout_records_on_save_and_load PASSED")


def test_project_store_rejects_non_dict_table_text_layer_cells_on_save_and_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bad_block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox(10, 20, 100, 40),
            table_text_layer_cells=["not-a-dict"],
        )
        bad_project = OcrProject(
            name="bad-table-cells-save",
            pages=[Page(image_path="/tmp/bad-table-save.png", width=120, height=80, blocks=[bad_block])],
        )
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"block\.table_text_layer_cells\[0\] must be dict"):
                store.save_project(bad_project)

        good_project = OcrProject(
            name="bad-table-cells-load",
            pages=[
                Page(
                    image_path="/tmp/bad-table-load.png",
                    width=120,
                    height=80,
                    blocks=[
                        Block(
                            block_type=BlockType.TABLE,
                            bbox=BBox(10, 20, 100, 40),
                            table_text_layer_cells=[{"text": "A", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
                        )
                    ],
                )
            ],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(good_project)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE block SET table_text_layer_cells_json=?",
                (json.dumps(["not-a-dict"], ensure_ascii=False),),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"block\.table_text_layer_cells_json\[0\] must be dict"):
                store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_non_dict_table_text_layer_cells_on_save_and_load PASSED")


def test_project_store_persists_typed_paddle_binding():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, PaddleBinding, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        binding_payload = {
            "status": "paddle_geometry_hit",
            "source": "paddle_geometry",
            "block_type": "equation",
            "source_label": "inline_formula",
            "text": "$ A $",
            "parent_index": 3,
            "candidate_index": 1,
            "score": 0.75,
            "manual_bbox": [10, 20, 30, 40],
            "candidate_bbox": [11, 21, 31, 41],
            "review_flags": ["manual_paddle_binding"],
        }
        block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox.from_xyxy(10, 20, 30, 40),
            paddle_binding=PaddleBinding.from_dict(binding_payload),
        )
        project = OcrProject(
            name="paddle-binding",
            pages=[Page(image_path="/tmp/binding.png", width=100, height=100, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            loaded = store.load_project(saved.id)

        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.paddle_binding is not None
        assert loaded_block.paddle_binding.to_dict()["text"] == "$ A $"
        assert loaded_block.paddle_binding.parent_index == 3

        print("test_project_store_persists_typed_paddle_binding PASSED")
    finally:
        os.unlink(db_path)


def test_paddle_binding_rejects_malformed_typed_fields():
    import pytest

    from app.models import PaddleBinding

    with pytest.raises(ValueError, match="parent_index must be int"):
        PaddleBinding.from_dict({"status": "hit", "parent_index": "3"})

    with pytest.raises(ValueError, match=r"candidate_bbox\[1\] must be int"):
        PaddleBinding.from_dict({"status": "hit", "candidate_bbox": [1, "x", 3, 4]})

    with pytest.raises(ValueError, match=r"review_flags\[0\] must be str"):
        PaddleBinding.from_dict({"status": "hit", "review_flags": [1]})

    print("test_paddle_binding_rejects_malformed_typed_fields PASSED")


def test_project_store_rejects_invalid_typed_block_state_on_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, PaddleBinding, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        binding = PaddleBinding.from_dict({
            "status": "paddle_geometry_hit",
            "manual_bbox": [10, 20, 30, 40],
        })
        block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox.from_xyxy(10, 20, 30, 40),
            paddle_binding=binding,
        )
        project = OcrProject(
            name="invalid-typed-block-state",
            pages=[Page(image_path="/tmp/invalid-typed-state.png", width=100, height=100, blocks=[block])],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(project)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE block SET paddle_binding_json=? WHERE uid=?",
                (json.dumps({"status": "hit", "manual_bbox": [10, "bad", 30, 40]}), block.uid),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"block\.paddle_binding_json manual_bbox\[1\] must be int"):
                store.load_project(saved.id)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE block SET paddle_binding_json=? WHERE uid=?",
                (json.dumps(binding.to_dict(), ensure_ascii=False), block.uid),
            )
            conn.execute(
                "UPDATE block_origin SET original_kind=? WHERE block_uid=?",
                ("not-a-kind", block.uid),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match="block_origin.original_kind invalid value"):
                store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_invalid_typed_block_state_on_load PASSED")


def test_project_store_persists_block_ocr_invalidation():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockOrigin, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 40, 20),
            ocr_invalidated_reason="block_moved",
        )
        project = OcrProject(
            name="typed-ocr-invalidation",
            pages=[Page(image_path="/tmp/ocr-invalidated.png", width=100, height=100, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            loaded = store.load_project(saved.id)

        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.ocr_invalidated_reason == "block_moved"

        print("test_project_store_persists_block_ocr_invalidation PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_inline_formula_origin():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox.from_xyxy(12, 4, 44, 24),
            source_label="inline_formula",
            origin=BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                source_engine="paddleocr-vl",
                source_label="inline_formula",
                original_bbox=BBox.from_xyxy(10, 3, 40, 23),
                original_kind=BlockType.EQUATION,
            ),
        )
        project = OcrProject(
            name="inline-formula-origin",
            pages=[Page(image_path="/tmp/inline-formula.png", width=100, height=80, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            loaded = store.load_project(saved.id)

        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.origin is not None
        assert loaded_block.origin.source_label == "inline_formula"
        assert loaded_block.origin.original_bbox == BBox.from_xyxy(10, 3, 40, 23)
        assert loaded_block.origin.original_kind == BlockType.EQUATION

        print("test_project_store_persists_inline_formula_origin PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_ocr_audit():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        audit = {
            "schema": "hanwang_bbox_audit.v1",
            "layout_block_bbox": [0, 0, 40, 20],
            "hanwang_recog_group_failed_count": 1,
        }
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 40, 20),
            ocr_audit=dict(audit),
        )
        project = OcrProject(
            name="ocr-audit",
            pages=[Page(image_path="/tmp/ocr-audit.png", width=100, height=80, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            loaded = store.load_project(saved.id)

        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.ocr_audit["schema"] == "hanwang_bbox_audit.v1"
        assert loaded_block.ocr_audit["layout_block_bbox"] == [0, 0, 40, 20]

        print("test_project_store_persists_ocr_audit PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_table_text_layer_cells():
    from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        cells = [
            {
                "text": "A",
                "bbox": {"x": 10, "y": 20, "w": 30, "h": 12},
                "row": 0,
                "col": 0,
            }
        ]
        block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox.from_xyxy(0, 0, 80, 40),
            table_text_layer_cells=list(cells),
        )
        project = OcrProject(
            name="table-text-layer-cells",
            pages=[Page(image_path="/tmp/table-cells.png", width=100, height=80, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
            loaded = store.load_project(saved.id)

        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.table_text_layer_cells == cells

        print("test_project_store_persists_table_text_layer_cells PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_block_origin_separately_from_current_layout():
    from app.core.project_store import ProjectStore
    from app.models import (
        BBox, Block, BlockOrigin, BlockSource, BlockType, OcrProject, Page,
    )
    from app.services.layout_edit_service import LayoutEditCommand, LayoutEditService

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        origin_bbox = BBox(10, 20, 100, 40)
        current_bbox = BBox(15, 25, 120, 44)
        block = Block(
            block_type=BlockType.TEXT,
            bbox=current_bbox,
            source=BlockSource.USER_EDITED,
            source_label="paragraph_title",
            origin=BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                source_engine="paddleocr-vl",
                source_run_id="run-1",
                source_label="paragraph_title",
                source_confidence=0.88,
                original_bbox=origin_bbox,
                original_kind=BlockType.TITLE,
                raw_artifact_uid="rawocr_1",
                raw_json_path=".cache/paddle_artifacts/page/run-1.json",
                raw_index=3,
            ),
        )
        project = OcrProject(
            name="block-origin",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)
            loaded_block = loaded.pages[0].blocks[0]
            assert loaded_block.bbox == current_bbox
            assert loaded_block.origin is not None
            assert loaded_block.origin.original_bbox == origin_bbox
            assert loaded_block.origin.original_kind == BlockType.TITLE
            assert loaded_block.origin.source_label == "paragraph_title"
            assert loaded_block.origin.source_confidence == 0.88
            assert loaded_block.origin.raw_index == 3

            LayoutEditService().apply(LayoutEditCommand.update_geometry(
                loaded.pages[0],
                loaded_block,
                bbox=BBox(30, 40, 130, 50),
            ))
            store.save_project(loaded)
            reloaded = store.load_project(project_id=1)
            reloaded_block = reloaded.pages[0].blocks[0]
            assert reloaded_block.bbox == BBox(30, 40, 130, 50)
            assert reloaded_block.origin is not None
            assert reloaded_block.origin.original_bbox == origin_bbox

        print("test_project_store_persists_block_origin_separately_from_current_layout PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_persists_layout_edit_events():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, LayoutEditEvent, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 20))
        page = Page(
            image_path="/tmp/img.jpg",
            width=800,
            height=600,
            blocks=[block],
            layout_edit_events=[
                LayoutEditEvent(
                    page_uid="",
                    target_uid=block.uid,
                    op="resize_block",
                    before={"bbox": [0, 0, 100, 20]},
                    after={"bbox": [5, 5, 110, 25]},
                )
            ],
        )
        project = OcrProject(name="layout-events", pages=[page])

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)
            events = loaded.pages[0].layout_edit_events
            assert len(events) == 1
            assert events[0].page_uid == loaded.pages[0].uid
            assert events[0].target_uid == block.uid
            assert events[0].op == "resize_block"
            assert events[0].before == {"bbox": [0, 0, 100, 20]}
            assert events[0].after == {"bbox": [5, 5, 110, 25]}

            store.save_project(loaded)
            reloaded = store.load_project(project_id=1)
            assert [event.uid for event in reloaded.pages[0].layout_edit_events] == [events[0].uid]

        print("test_project_store_persists_layout_edit_events PASSED")
    finally:
        os.unlink(db_path)


def test_line_final_text_contract_and_project_store_roundtrip():
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page
    from app.models.ocr_text_observation import set_line_ocr_text_observation
    from app.models.ocr_text_observation_store import OcrTextObservation
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(text="OCR text", confidence=0.9, bbox=bb)
        set_line_proof_text(line, "人工终稿")
        assert line.text == "OCR text"
        assert proof_final_text(line) == "人工终稿"
        assert proof_display_text(line) == "人工终稿"
        set_line_ocr_text_observation(
            line,
            OcrTextObservation(
                text="观察边界写入",
                ocr_text="OCR text",
                confidence=0.9,
            ),
        )
        assert proof_final_text(line) == "人工终稿"
        ensure_line_text_contract(line)
        assert line.text == "观察边界写入"
        assert proof_final_text(line) == "人工终稿"
        assert line.ocr_text == "OCR text"
        assert not hasattr(line, "final_text")
        assert line.text == "观察边界写入"
        assert proof_display_text(line) == "人工终稿"
        assert not hasattr(line, "final_text_set")
        assert proof_display_text(line) == "人工终稿"
        set_line_proof_text(line, "最终真值")
        set_line_proof_text(line, "")
        ensure_line_text_contract(line)
        assert proof_final_text(line) == ""
        assert proof_final_text_set(line) is True
        assert proof_display_text(line) == ""
        set_line_proof_text(line, "最终真值")

        line_without_text = Line(
            text="",
            ocr_text="OCR补全文本",
            confidence=0.8,
            bbox=bb,
        )
        ensure_line_text_contract(line_without_text)
        assert line_without_text.text == "OCR补全文本"
        assert proof_display_text(line_without_text) == "OCR补全文本"
        assert line_without_text.ocr_text == "OCR补全文本"

        final_only = Line(text="", confidence=0.8, bbox=bb)
        set_line_proof_text(final_only, "人工终稿")
        final_only.ocr_text = ""
        ensure_line_text_contract(final_only)
        assert proof_display_text(final_only) == "人工终稿"
        assert final_only.ocr_text == ""

        project = OcrProject(
            name="final_text",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600,
                        blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)
            loaded_line = loaded.pages[0].blocks[0].lines[0]
            assert proof_final_text(loaded_line) == "最终真值"
            assert proof_final_text_set(loaded_line) is True
            assert loaded_line.text == "观察边界写入"
            assert loaded_line.ocr_text == "OCR text"

        print("test_line_final_text_contract_and_project_store_roundtrip PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_preserves_empty_final_text_roundtrip():
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(text="OCR原文", confidence=0.9, bbox=bb)
        set_line_proof_text(line, "")
        ensure_line_text_contract(line)
        assert proof_final_text(line) == ""
        assert proof_display_text(line) == ""

        project = OcrProject(
            name="empty-final-text",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600,
                        blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)
            loaded_line = loaded.pages[0].blocks[0].lines[0]
            assert loaded_line.text == "OCR原文"
            assert proof_final_text(loaded_line) == ""
            assert proof_final_text_set(loaded_line) is True
            assert proof_display_text(loaded_line) == ""
    finally:
        os.unlink(db_path)


def test_project_store_serializes_line_contract_without_mutating_line():
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(text="", ocr_text="OCR补全文本", confidence=0.9, bbox=bb)
        assert not hasattr(line, "proof_state")
        project = OcrProject(
            name="line-contract-no-mutate",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600,
                        blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(project)
            assert line.text == ""
            assert line.ocr_text == "OCR补全文本"
            assert not hasattr(line, "proof_state")

            loaded = store.load_project(project_id=1)
            loaded_line = loaded.pages[0].blocks[0].lines[0]
            assert loaded_line.text == "OCR补全文本"
            assert loaded_line.ocr_text == "OCR补全文本"
            assert proof_display_text(loaded_line) == "OCR补全文本"
    finally:
        os.unlink(db_path)


def test_project_store_clean_on_resave():
    """重新保存时旧 block 不残留。"""
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)

        # 第一次保存：block A
        line_a = Line(text="BlockA", confidence=0.9, bbox=bb)
        block_a = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line_a])
        page = Page(image_path="/tmp/img.jpg", width=800, height=600)
        page.blocks = [block_a]
        project = OcrProject(name="清理测试", pages=[page])

        with ProjectStore(db_path) as store:
            store.save_project(project)
            page_id = project.pages[0].id

            # 第二次保存：只保留 block B
            line_b = Line(text="BlockB", confidence=0.95, bbox=bb)
            block_b = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line_b])
            project.pages[0].blocks = [block_b]
            store.save_project(project)

        # 重新打开，断言只有 BlockB
        with ProjectStore(db_path) as store:
            loaded = store.load_project(project_id=1)
            assert len(loaded.pages[0].blocks) == 1
            assert loaded.pages[0].blocks[0].lines[0].text == "BlockB"

        print("test_project_store_clean_on_resave PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_current_schema_omits_retired_block_payload_columns():
    import sqlite3

    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        with ProjectStore(db_path) as store:
            store.conn.execute("SELECT 1").fetchone()
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(block)").fetchall()}
        assert "raw_payload_json" not in columns
        assert "app_payload_json" not in columns

    finally:
        os.unlink(db_path)

    print("test_project_store_current_schema_omits_retired_block_payload_columns PASSED")


def test_project_store_migration_drops_empty_retired_block_payload_columns():
    import sqlite3

    from app.core.logging import SCHEMA_VERSION
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(0, 0, 40, 20))
        clean_project = OcrProject(
            name="retired payload empty migration",
            pages=[Page(image_path="/tmp/empty-retired-payload.png", width=80, height=40, blocks=[block])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(clean_project)
        with sqlite3.connect(db_path) as conn:
            conn.execute("ALTER TABLE block ADD COLUMN raw_payload_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute("ALTER TABLE block ADD COLUMN app_payload_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION - 1),),
            )
            conn.commit()
        with ProjectStore(db_path) as store:
            store.conn.execute("SELECT 1").fetchone()
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(block)").fetchall()}
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        assert "raw_payload_json" not in columns
        assert "app_payload_json" not in columns
        assert int(meta["schema_version"]) == SCHEMA_VERSION
    finally:
        os.unlink(db_path)

    print("test_project_store_migration_drops_empty_retired_block_payload_columns PASSED")


def test_project_store_migration_rejects_nonempty_retired_block_payload_columns():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        clean_block = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(0, 0, 40, 20))
        clean_project = OcrProject(
            name="retired payload nonempty migration",
            pages=[Page(image_path="/tmp/nonempty-retired-payload.png", width=80, height=40, blocks=[clean_block])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(clean_project)
        with sqlite3.connect(db_path) as conn:
            conn.execute("ALTER TABLE block ADD COLUMN raw_payload_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute("ALTER TABLE block ADD COLUMN app_payload_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute(
                "UPDATE block SET raw_payload_json=? WHERE id=?",
                (json.dumps({"paddle_binding": {"source_label": "text"}}, ensure_ascii=False), clean_block.id),
            )
            conn.execute("UPDATE meta SET value='22' WHERE key='schema_version'")
            conn.commit()
        with pytest.raises(ProjectDataError, match="retired payload"):
            store = ProjectStore(db_path)
            store.open()
    finally:
        try:
            store.close()
        except Exception:
            pass
        os.unlink(db_path)

    print("test_project_store_migration_rejects_nonempty_retired_block_payload_columns PASSED")


def test_project_store_rejects_invalid_review_flags_json_on_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        line = Line(text="正文", confidence=0.9, bbox=BBox.from_xyxy(1, 2, 30, 20))
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 40, 20),
            lines=[line],
        )
        project = OcrProject(
            name="invalid payload json",
            pages=[Page(image_path="/tmp/invalid-payload-json.png", width=80, height=40, blocks=[block])],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(project)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE line SET review_flags_json=? WHERE id=?", ("[", line.id))
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match="line.review_flags_json invalid json"):
                store.load_project(saved.id)

        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE line SET review_flags_json=? WHERE id=?", ("{}", line.id))
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match="line.review_flags_json must be list"):
                store.load_project(saved.id)

        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE line SET review_flags_json=? WHERE id=?", (json.dumps([1]), line.id))
            conn.commit()
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match=r"line\.review_flags_json\[0\] must be str"):
                store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_invalid_review_flags_json_on_load PASSED")


def test_project_store_rejects_invalid_current_schema_enums_on_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        line = Line(text="正文", confidence=0.9, bbox=BBox.from_xyxy(1, 2, 30, 20))
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 40, 20),
            lines=[line],
        )
        project = OcrProject(
            name="invalid current schema enum",
            pages=[Page(image_path="/tmp/invalid-current-schema-enum.png", width=80, height=40, blocks=[block])],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(project)

        cases = [
            ("UPDATE page SET status=? WHERE id=?", ("not-a-status", project.pages[0].id), "page.status invalid value"),
            ("UPDATE page SET status=? WHERE id=?", ("", project.pages[0].id), "page.status is empty"),
            ("UPDATE block SET block_type=? WHERE id=?", ("not-a-block", block.id), "block.block_type invalid value"),
            ("UPDATE block SET source=? WHERE id=?", ("not-a-source", block.id), "block.source invalid value"),
            ("UPDATE block SET ocr_policy=? WHERE id=?", ("", block.id), "block.ocr_policy is empty"),
            ("UPDATE block SET ocr_policy=? WHERE id=?", ("legacy-wordbox", block.id), "block.ocr_policy invalid value"),
        ]

        for sql, params, match in cases:
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE page SET status='imported' WHERE id=?", (project.pages[0].id,))
                conn.execute("UPDATE block SET block_type='text', source='auto_layout', ocr_policy='text_ocr' WHERE id=?", (block.id,))
                conn.execute(sql, params)
                conn.commit()
            with ProjectStore(db_path) as store:
                with pytest.raises(ProjectDataError, match=match):
                    store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_invalid_current_schema_enums_on_load PASSED")


def test_project_store_rejects_malformed_persisted_geometry_on_load():
    import pytest
    import sqlite3

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Char, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        char = Char(char="字", confidence=0.9, bbox=BBox.from_xyxy(2, 3, 8, 12))
        line = Line(text="字", confidence=0.9, bbox=BBox.from_xyxy(1, 2, 20, 12), chars=[char])
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 40, 20),
            lines=[line],
            origin=BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                original_bbox=BBox.from_xyxy(0, 0, 40, 20),
                original_kind=BlockType.TEXT,
            ),
        )
        project = OcrProject(
            name="malformed persisted geometry",
            pages=[Page(image_path="/tmp/malformed-geometry.png", width=80, height=40, blocks=[block])],
        )
        with ProjectStore(db_path) as store:
            saved = store.save_project(project)

        cases = [
            ("UPDATE block SET x=? WHERE id=?", ("bad-x", block.id), "block.x must be int"),
            ("UPDATE line SET h=? WHERE id=?", ("bad-h", line.id), "line.h must be int"),
            ("UPDATE char_ SET y=? WHERE id=?", (None, char.id), "char bbox is partial"),
            (
                "UPDATE block_origin SET original_w=? WHERE block_uid=?",
                ("bad-w", block.uid),
                r"block_origin\.original_bbox\.original_w must be int",
            ),
            (
                "UPDATE block_origin SET original_y=? WHERE block_uid=?",
                (None, block.uid),
                "block_origin.original_bbox bbox is partial",
            ),
        ]

        for sql, params, match in cases:
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE block SET x=0, y=0, w=40, h=20 WHERE id=?", (block.id,))
                conn.execute("UPDATE line SET x=1, y=2, w=19, h=10 WHERE id=?", (line.id,))
                conn.execute("UPDATE char_ SET x=2, y=3, w=6, h=9 WHERE id=?", (char.id,))
                conn.execute(
                    "UPDATE block_origin SET original_x=0, original_y=0, original_w=40, original_h=20 WHERE block_uid=?",
                    (block.uid,),
                )
                conn.execute(sql, params)
                conn.commit()
            with ProjectStore(db_path) as store:
                with pytest.raises(ProjectDataError, match=match):
                    store.load_project(saved.id)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_malformed_persisted_geometry_on_load PASSED")


def test_model_validation_rejects_legacy_page_and_block_payload_attr():
    import pytest

    from app.core.model_validation import (
        ModelValidationError,
        validate_block_model,
        validate_page_model,
    )
    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(0, 0, 20, 20),
    )
    block.raw_payload = {LAYOUT_LINE_ROUTES_FIELD: []}
    with pytest.raises(ModelValidationError, match="must not expose raw_payload"):
        validate_block_model(block)

    page = Page(image_path="/tmp/model-validation.png", width=20, height=20)
    page.ppvl_parsing_res_list = []
    with pytest.raises(ModelValidationError, match="ppvl_parsing_res_list"):
        validate_page_model(page)

    print("test_model_validation_rejects_legacy_page_and_block_payload_attr PASSED")


def test_project_store_rejects_legacy_page_model_on_save():
    import os
    import tempfile

    import pytest

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        page = Page(image_path="/tmp/legacy-page-model.png", width=20, height=20)
        page.ppvl_parsing_res_list = []
        project = OcrProject(name="legacy page", pages=[page])
        with ProjectStore(db_path) as store:
            with pytest.raises(ProjectDataError, match="ppvl_parsing_res_list"):
                store.save_project(project)
    finally:
        os.unlink(db_path)

    print("test_project_store_rejects_legacy_page_model_on_save PASSED")


def test_project_store_save_project_preserves_child_rowids():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line1 = Line(
            text="第一行",
            confidence=0.9,
            bbox=bb,
            chars=[Char(char="第", confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        line2 = Line(
            text="第二行",
            confidence=0.8,
            bbox=BBox(0, 30, 100, 20),
            chars=[Char(char="第", confidence=0.8, bbox=BBox(0, 30, 10, 10))],
        )
        block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1, line2])
        project = OcrProject(
            name="stable ids",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block])],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            ids = {
                "page": project.pages[0].id,
                "block": block.id,
                "line": line1.id,
                "char": line1.chars[0].id,
                "removed_line": line2.id,
                "removed_char": line2.chars[0].id,
            }
            uids = {
                "page": project.pages[0].uid,
                "block": block.uid,
                "line": line1.uid,
                "char": line1.chars[0].uid,
            }

            set_line_proof_text(line1, "第一行已校对")
            line1.chars[0].char = "一"
            block.note = "updated without id churn"
            block.lines = [line1]
            store.save_project(project)
            loaded = store.load_project(project_id=project.id)

        loaded_page = loaded.pages[0]
        loaded_block = loaded_page.blocks[0]
        loaded_line = loaded_block.lines[0]
        loaded_char = loaded_line.chars[0]

        assert loaded_page.id == ids["page"]
        assert loaded_block.id == ids["block"]
        assert loaded_line.id == ids["line"]
        assert loaded_char.id == ids["char"]
        assert loaded_page.uid == uids["page"]
        assert loaded_block.uid == uids["block"]
        assert loaded_line.uid == uids["line"]
        assert loaded_char.uid == uids["char"]
        assert proof_final_text(loaded_line) == "第一行已校对"
        assert loaded_char.char == "一"
        assert loaded_block.note == "updated without id churn"
        assert len(loaded_block.lines) == 1

        with ProjectStore(db_path) as store:
            assert store.conn.execute(
                "SELECT COUNT(*) FROM line WHERE id=?",
                (ids["removed_line"],),
            ).fetchone()[0] == 0
            assert store.conn.execute(
                "SELECT COUNT(*) FROM char_ WHERE id=?",
                (ids["removed_char"],),
            ).fetchone()[0] == 0
    finally:
        os.unlink(db_path)

    print("test_project_store_save_project_preserves_child_rowids PASSED")


def test_project_store_upsert_rejects_foreign_parent_rowids():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    def make_project(name: str, text: str) -> OcrProject:
        bb = BBox(0, 0, 100, 20)
        line = Line(
            text=text,
            confidence=0.9,
            bbox=bb,
            chars=[Char(char=text[:1], confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        return OcrProject(
            name=name,
            pages=[
                Page(
                    image_path=f"/tmp/{name}.jpg",
                    width=800,
                    height=600,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])],
                )
            ],
        )

    try:
        project1 = make_project("p1", "甲")
        project2 = make_project("p2", "乙")

        with ProjectStore(db_path) as store:
            store.save_project(project1)
            store.save_project(project2)

            p1_page = project1.pages[0]
            p1_block = p1_page.blocks[0]
            p1_line = p1_block.lines[0]
            p1_char = p1_line.chars[0]

            p2_page = project2.pages[0]
            p2_block = p2_page.blocks[0]
            p2_line = p2_block.lines[0]
            p2_char = p2_line.chars[0]
            original_p2_ids = (p2_page.id, p2_block.id, p2_line.id, p2_char.id)
            original_p2_uids = (p2_page.uid, p2_block.uid, p2_line.uid, p2_char.uid)

            p2_page.id = p1_page.id
            p2_block.id = p1_block.id
            p2_line.id = p1_line.id
            p2_char.id = p1_char.id
            set_line_proof_text(p2_line, "乙已改")
            p2_char.char = "乙"

            store.save_project(project2)

            reloaded1 = store.load_project(project_id=project1.id)
            reloaded2 = store.load_project(project_id=project2.id)

        assert reloaded1.pages[0].id == p1_page.id
        assert reloaded1.pages[0].blocks[0].id == p1_block.id
        assert reloaded1.pages[0].blocks[0].lines[0].id == p1_line.id
        assert reloaded1.pages[0].blocks[0].lines[0].chars[0].id == p1_char.id
        assert reloaded1.pages[0].blocks[0].lines[0].text == "甲"
        assert reloaded1.pages[0].blocks[0].lines[0].chars[0].char == "甲"

        assert reloaded2.pages[0].id != p1_page.id
        assert reloaded2.pages[0].blocks[0].id != p1_block.id
        assert reloaded2.pages[0].blocks[0].lines[0].id != p1_line.id
        assert reloaded2.pages[0].blocks[0].lines[0].chars[0].id != p1_char.id
        assert reloaded2.pages[0].id == original_p2_ids[0]
        assert reloaded2.pages[0].blocks[0].id == original_p2_ids[1]
        assert reloaded2.pages[0].blocks[0].lines[0].id == original_p2_ids[2]
        assert reloaded2.pages[0].blocks[0].lines[0].chars[0].id == original_p2_ids[3]
        assert reloaded2.pages[0].uid == original_p2_uids[0]
        assert reloaded2.pages[0].blocks[0].uid == original_p2_uids[1]
        assert reloaded2.pages[0].blocks[0].lines[0].uid == original_p2_uids[2]
        assert reloaded2.pages[0].blocks[0].lines[0].chars[0].uid == original_p2_uids[3]
        assert proof_final_text(reloaded2.pages[0].blocks[0].lines[0]) == "乙已改"
        assert proof_display_text(reloaded2.pages[0].blocks[0].lines[0]) == "乙已改"
        assert reloaded2.pages[0].blocks[0].lines[0].chars[0].char == "乙"
    finally:
        os.unlink(db_path)

    print("test_project_store_upsert_rejects_foreign_parent_rowids PASSED")


def test_project_store_cross_project_uid_collision_remints_without_stealing():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    def make_project(name: str, text: str) -> OcrProject:
        bb = BBox(0, 0, 100, 20)
        line = Line(
            text=text,
            confidence=0.9,
            bbox=bb,
            chars=[Char(char=text[:1], confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        return OcrProject(
            name=name,
            pages=[
                Page(
                    image_path=f"/tmp/{name}.jpg",
                    width=800,
                    height=600,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])],
                )
            ],
        )

    try:
        project1 = make_project("p1", "甲")
        project2 = make_project("p2", "乙")

        with ProjectStore(db_path) as store:
            store.save_project(project1)
            store.save_project(project2)

            p1_page = project1.pages[0]
            p1_block = p1_page.blocks[0]
            p1_line = p1_block.lines[0]
            p1_char = p1_line.chars[0]
            p1_ids = (p1_page.id, p1_block.id, p1_line.id, p1_char.id)
            p1_uids = (p1_page.uid, p1_block.uid, p1_line.uid, p1_char.uid)

            p2_page = project2.pages[0]
            p2_block = p2_page.blocks[0]
            p2_line = p2_block.lines[0]
            p2_char = p2_line.chars[0]

            p2_page.id, p2_page.uid = p1_page.id, p1_page.uid
            p2_block.id, p2_block.uid = p1_block.id, p1_block.uid
            p2_line.id, p2_line.uid = p1_line.id, p1_line.uid
            p2_char.id, p2_char.uid = p1_char.id, p1_char.uid
            set_line_proof_text(p2_line, "乙已改")
            p2_char.char = "乙"

            store.save_project(project2)

            reloaded1 = store.load_project(project_id=project1.id)
            reloaded2 = store.load_project(project_id=project2.id)

        p1_loaded_page = reloaded1.pages[0]
        p1_loaded_block = p1_loaded_page.blocks[0]
        p1_loaded_line = p1_loaded_block.lines[0]
        p1_loaded_char = p1_loaded_line.chars[0]
        assert (p1_loaded_page.id, p1_loaded_block.id, p1_loaded_line.id, p1_loaded_char.id) == p1_ids
        assert (
            p1_loaded_page.uid,
            p1_loaded_block.uid,
            p1_loaded_line.uid,
            p1_loaded_char.uid,
        ) == p1_uids
        assert proof_display_text(p1_loaded_line) == "甲"
        assert p1_loaded_char.char == "甲"

        p2_loaded_page = reloaded2.pages[0]
        p2_loaded_block = p2_loaded_page.blocks[0]
        p2_loaded_line = p2_loaded_block.lines[0]
        p2_loaded_char = p2_loaded_line.chars[0]
        assert (p2_loaded_page.id, p2_loaded_block.id, p2_loaded_line.id, p2_loaded_char.id) != p1_ids
        assert p2_loaded_page.uid != p1_uids[0]
        assert p2_loaded_block.uid != p1_uids[1]
        assert p2_loaded_line.uid != p1_uids[2]
        assert p2_loaded_char.uid != p1_uids[3]
        assert proof_display_text(p2_loaded_line) == "乙已改"
        assert p2_loaded_char.char == "乙"
    finally:
        os.unlink(db_path)

    print("test_project_store_cross_project_uid_collision_remints_without_stealing PASSED")


def test_project_store_cross_project_uid_pollution_preserves_valid_rowids():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    def make_project(name: str, text: str) -> OcrProject:
        bb = BBox(0, 0, 100, 20)
        line = Line(
            text=text,
            confidence=0.9,
            bbox=bb,
            chars=[Char(char=text[:1], confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        return OcrProject(
            name=name,
            pages=[
                Page(
                    image_path=f"/tmp/{name}.jpg",
                    width=800,
                    height=600,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])],
                )
            ],
        )

    try:
        project1 = make_project("p1", "甲")
        project2 = make_project("p2", "乙")

        with ProjectStore(db_path) as store:
            store.save_project(project1)
            store.save_project(project2)

            p1_page = project1.pages[0]
            p1_block = p1_page.blocks[0]
            p1_line = p1_block.lines[0]
            p1_char = p1_line.chars[0]
            p1_ids = (p1_page.id, p1_block.id, p1_line.id, p1_char.id)

            p2_page = project2.pages[0]
            p2_block = p2_page.blocks[0]
            p2_line = p2_block.lines[0]
            p2_char = p2_line.chars[0]
            p2_ids = (p2_page.id, p2_block.id, p2_line.id, p2_char.id)
            p2_uids = (p2_page.uid, p2_block.uid, p2_line.uid, p2_char.uid)

            p2_page.uid = p1_page.uid
            p2_block.uid = p1_block.uid
            p2_line.uid = p1_line.uid
            p2_char.uid = p1_char.uid
            set_line_proof_text(p2_line, "乙已改")
            p2_char.char = "乙"

            store.save_project(project2)

            reloaded1 = store.load_project(project_id=project1.id)
            reloaded2 = store.load_project(project_id=project2.id)

        p1_loaded_page = reloaded1.pages[0]
        p1_loaded_block = p1_loaded_page.blocks[0]
        p1_loaded_line = p1_loaded_block.lines[0]
        p1_loaded_char = p1_loaded_line.chars[0]
        assert (p1_loaded_page.id, p1_loaded_block.id, p1_loaded_line.id, p1_loaded_char.id) == p1_ids
        assert proof_display_text(p1_loaded_line) == "甲"
        assert p1_loaded_char.char == "甲"

        p2_loaded_page = reloaded2.pages[0]
        p2_loaded_block = p2_loaded_page.blocks[0]
        p2_loaded_line = p2_loaded_block.lines[0]
        p2_loaded_char = p2_loaded_line.chars[0]
        assert (p2_loaded_page.id, p2_loaded_block.id, p2_loaded_line.id, p2_loaded_char.id) == p2_ids
        assert (
            p2_loaded_page.uid,
            p2_loaded_block.uid,
            p2_loaded_line.uid,
            p2_loaded_char.uid,
        ) == p2_uids
        assert proof_display_text(p2_loaded_line) == "乙已改"
        assert p2_loaded_char.char == "乙"
    finally:
        os.unlink(db_path)

    print("test_project_store_cross_project_uid_pollution_preserves_valid_rowids PASSED")


def test_project_store_uid_recovers_same_parent_stale_rowid():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line1 = Line(
            text="第一行",
            confidence=0.9,
            bbox=bb,
            chars=[Char(char="一", confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        line2 = Line(
            text="第二行",
            confidence=0.9,
            bbox=BBox(0, 30, 100, 20),
            chars=[Char(char="二", confidence=0.9, bbox=BBox(0, 30, 10, 10))],
        )
        block1 = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1])
        block2 = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 60, 100, 20),
            lines=[line2],
            order=1,
        )
        project = OcrProject(
            name="same parent stale rowid",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block1, block2])],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            original = {
                "block1_id": block1.id,
                "block1_uid": block1.uid,
                "block2_id": block2.id,
                "block2_uid": block2.uid,
                "line1_id": line1.id,
                "line1_uid": line1.uid,
                "line2_id": line2.id,
                "line2_uid": line2.uid,
            }

            block2.id = block1.id
            line2.id = line1.id
            block2.note = "第二块已更新"
            set_line_proof_text(line2, "第二行已更新")
            store.save_project(project)
            loaded = store.load_project(project_id=project.id)

        loaded_blocks = {block.uid: block for block in loaded.pages[0].blocks}
        loaded_block1 = loaded_blocks[original["block1_uid"]]
        loaded_block2 = loaded_blocks[original["block2_uid"]]
        loaded_line1 = loaded_block1.lines[0]
        loaded_line2 = loaded_block2.lines[0]

        assert loaded_block1.id == original["block1_id"]
        assert loaded_block2.id == original["block2_id"]
        assert loaded_block1.note == ""
        assert loaded_block2.note == "第二块已更新"
        assert loaded_line1.id == original["line1_id"]
        assert loaded_line2.id == original["line2_id"]
        assert loaded_line1.uid == original["line1_uid"]
        assert loaded_line2.uid == original["line2_uid"]
        assert proof_display_text(loaded_line1) == "第一行"
        assert proof_display_text(loaded_line2) == "第二行已更新"
    finally:
        os.unlink(db_path)

    print("test_project_store_uid_recovers_same_parent_stale_rowid PASSED")


def test_project_store_duplicate_sibling_uids_are_reminted():
    import copy
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line1 = Line(
            text="甲",
            confidence=0.9,
            bbox=bb,
            chars=[Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        )
        line2 = copy.deepcopy(line1)
        line2.text = "乙"
        line2.ocr_text = "乙"
        set_line_proof_text(line2, "乙")
        line2.chars[0].char = "乙"

        line_block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1, line2])
        block_copy = copy.deepcopy(line_block)
        block_copy.order = 1
        block_copy.lines[0].text = "丙"
        block_copy.lines[0].ocr_text = "丙"
        set_line_proof_text(block_copy.lines[0], "丙")
        block_copy.lines[0].chars[0].char = "丙"

        project = OcrProject(
            name="duplicate sibling uids",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[line_block, block_copy])],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=project.id)

        assert len(loaded.pages[0].blocks) == 2
        assert len({block.uid for block in loaded.pages[0].blocks}) == 2
        first_block, second_block = loaded.pages[0].blocks
        assert [proof_display_text(line) for line in first_block.lines] == ["甲", "乙"]
        assert len({line.uid for line in first_block.lines}) == 2
        assert [proof_display_text(line) for line in second_block.lines] == ["丙", "乙"]
        assert len({line.uid for block in loaded.pages[0].blocks for line in block.lines}) == 4
        assert len({
            char.uid
            for block in loaded.pages[0].blocks
            for line in block.lines
            for char in line.chars
        }) == 4
    finally:
        os.unlink(db_path)

    print("test_project_store_duplicate_sibling_uids_are_reminted PASSED")


def test_project_store_cross_parent_moves_preserve_uids_regardless_of_save_order():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    def make_project() -> tuple[OcrProject, dict[str, str]]:
        bb = BBox(0, 0, 100, 20)
        moved_block_line = Line(text="跨页块", confidence=0.9, bbox=bb)
        moved_block = Block(
            block_type=BlockType.TEXT,
            bbox=bb,
            lines=[moved_block_line],
            order=0,
        )
        source_line = Line(
            text="跨块行",
            confidence=0.9,
            bbox=BBox(0, 30, 100, 20),
            chars=[Char(char="行", confidence=0.9, bbox=BBox(0, 30, 10, 10))],
        )
        line_source_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 30, 100, 20),
            lines=[source_line],
            order=1,
        )
        line_target_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 60, 100, 20),
            lines=[],
            order=2,
        )
        char_source_line = Line(
            text="源",
            confidence=0.9,
            bbox=BBox(0, 90, 100, 20),
            chars=[Char(char="源", confidence=0.9, bbox=BBox(0, 90, 10, 10))],
        )
        char_target_line = Line(
            text="目标",
            confidence=0.9,
            bbox=BBox(0, 120, 100, 20),
            chars=[],
        )
        char_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 90, 100, 50),
            lines=[char_source_line, char_target_line],
            order=3,
        )
        page1 = Page(
            image_path="/tmp/p1.jpg",
            width=800,
            height=600,
            page_number=1,
            blocks=[moved_block, line_source_block, line_target_block, char_block],
        )
        page2 = Page(
            image_path="/tmp/p2.jpg",
            width=800,
            height=600,
            page_number=2,
            blocks=[],
        )
        project = OcrProject(name="cross parent moves", pages=[page1, page2])
        uids = {
            "block": moved_block.uid,
            "line": source_line.uid,
            "char": char_source_line.chars[0].uid,
            "line_target_block": line_target_block.uid,
            "char_target_line": char_target_line.uid,
        }
        return project, uids

    def exercise(page_order: str, block_order: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
            db_path = f.name
        try:
            project, uids = make_project()
            with ProjectStore(db_path) as store:
                store.save_project(project)
                page1, page2 = project.pages
                moved_block = next(block for block in page1.blocks if block.uid == uids["block"])
                line_source_block = next(block for block in page1.blocks if block.order == 1)
                line_target_block = next(block for block in page1.blocks if block.uid == uids["line_target_block"])
                char_block = next(block for block in page1.blocks if block.order == 3)
                source_line = line_source_block.lines.pop(0)
                char_source_line = char_block.lines[0]
                char_target_line = next(line for line in char_block.lines if line.uid == uids["char_target_line"])
                moved_char = char_source_line.chars.pop(0)

                page1.blocks.remove(moved_block)
                page2.blocks.append(moved_block)
                line_target_block.lines.append(source_line)
                char_target_line.chars.append(moved_char)

                if block_order == "new_parent_first":
                    page1.blocks.remove(line_target_block)
                    page1.blocks.insert(0, line_target_block)
                    char_block.lines = [char_target_line, char_source_line]
                else:
                    char_block.lines = [char_source_line, char_target_line]
                if page_order == "new_parent_first":
                    project.pages = [page2, page1]
                else:
                    project.pages = [page1, page2]

                store.save_project(project)
                loaded = store.load_project(project_id=project.id)

            loaded_pages = {page.page_number: page for page in loaded.pages}
            assert any(block.uid == uids["block"] for block in loaded_pages[2].blocks)
            assert all(block.uid != uids["block"] for block in loaded_pages[1].blocks)

            loaded_blocks = [block for page in loaded.pages for block in page.blocks]
            loaded_line_target = next(block for block in loaded_blocks if block.uid == uids["line_target_block"])
            assert [line.uid for line in loaded_line_target.lines] == [uids["line"]]

            loaded_lines = [line for block in loaded_blocks for line in block.lines]
            loaded_char_target = next(line for line in loaded_lines if line.uid == uids["char_target_line"])
            assert [char.uid for char in loaded_char_target.chars] == [uids["char"]]
        finally:
            os.unlink(db_path)

    exercise("old_parent_first", "old_parent_first")
    exercise("new_parent_first", "new_parent_first")

    print("test_project_store_cross_parent_moves_preserve_uids_regardless_of_save_order PASSED")


def test_project_store_persists_page_ocr_invalidation_reason():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
    from app.models.page_state import invalidate_page_ocr, page_needs_ocr_rerun
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        page = Page(
            image_path="/tmp/img.jpg",
            width=800,
            height=600,
            status=PageStatus.LAYOUT_DONE,
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=bb,
                    lines=[Line(text="旧 OCR", confidence=0.9, bbox=bb)],
                )
            ],
        )
        invalidate_page_ocr(page, "block_type_changed")
        project = OcrProject(name="invalidate", pages=[page])

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)

        assert loaded.pages[0].ocr_invalidated_reason == "block_type_changed"
        assert page_needs_ocr_rerun(loaded.pages[0]) is True
        assert loaded.pages[0].status == PageStatus.LAYOUT_DONE
    finally:
        os.unlink(db_path)

    print("test_project_store_persists_page_ocr_invalidation_reason PASSED")


def test_project_store_update_proof_lines_rolls_back_as_single_transaction():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line1 = Line(text="第一行", confidence=0.9, bbox=bb)
        line2 = Line(text="第二行", confidence=0.9, bbox=bb)
        project = OcrProject(
            name="batch",
            pages=[
                Page(
                    image_path="/tmp/img.jpg",
                    width=800,
                    height=600,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1, line2])],
                )
            ],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            set_line_proof_text(line1, "第一行已改")
            set_line_proof_text(line2, "第二行已改")
            line2.id = -999999
            try:
                store.update_proof_lines([(line1, False), (line2, False)])
            except Exception:
                pass
            else:
                raise AssertionError("update_proof_lines should fail on invalid line id")

            loaded = store.load_project(project_id=1)

        loaded_lines = loaded.pages[0].blocks[0].lines
        assert [proof_display_text(line) for line in loaded_lines] == ["第一行", "第二行"]
        assert [proof_status(line) for line in loaded_lines] == [
            ProofStatus.UNCHECKED,
            ProofStatus.UNCHECKED,
        ]
    finally:
        os.unlink(db_path)

    print("test_project_store_update_proof_lines_rolls_back_as_single_transaction PASSED")


def test_project_store_update_proof_lines_requires_stable_uid_match():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line1 = Line(text="第一行", confidence=0.9, bbox=bb)
        line2 = Line(text="第二行", confidence=0.9, bbox=bb)
        project = OcrProject(
            name="line uid guard",
            pages=[
                Page(
                    image_path="/tmp/img.jpg",
                    width=800,
                    height=600,
                    blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1, line2])],
                )
            ],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            line1_id = line1.id
            line2_id = line2.id
            line1_uid = line1.uid
            line2_uid = line2.uid

            line2.id = line1_id
            set_line_proof_text(line2, "不应写入第一行")
            try:
                store.update_proof_lines([(line2, False)])
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_proof_lines should reject stale rowid with mismatched uid")

            loaded = store.load_project(project_id=project.id)
            loaded_lines = loaded.pages[0].blocks[0].lines
            assert loaded_lines[0].id == line1_id
            assert loaded_lines[0].uid == line1_uid
            assert proof_display_text(loaded_lines[0]) == "第一行"
            assert loaded_lines[1].id == line2_id
            assert loaded_lines[1].uid == line2_uid
            assert proof_display_text(loaded_lines[1]) == "第二行"

            line2.id = line1_id
            line2.uid = ""
            set_line_proof_text(line2, "仍不应写入第一行")
            try:
                store.update_proof_lines([(line2, False)])
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_proof_lines should reject missing uid even when rowid exists")

            loaded = store.load_project(project_id=project.id)
            loaded_lines = loaded.pages[0].blocks[0].lines
            assert loaded_lines[0].id == line1_id
            assert loaded_lines[0].uid == line1_uid
            assert proof_display_text(loaded_lines[0]) == "第一行"
            assert loaded_lines[1].id == line2_id
            assert loaded_lines[1].uid == line2_uid
            assert proof_display_text(loaded_lines[1]) == "第二行"

            line1.id = line1_id
            line1.uid = ""
            set_line_proof_text(line1, "第一行已改")
            try:
                store.update_proof_lines([(line1, False)])
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_proof_lines should reject missing uid on a valid rowid")

            line1.uid = line1_uid
            store.update_proof_lines([(line1, False)])
            loaded = store.load_project(project_id=project.id)

        assert loaded.pages[0].blocks[0].lines[0].id == line1_id
        assert loaded.pages[0].blocks[0].lines[0].uid == line1_uid
        assert proof_display_text(loaded.pages[0].blocks[0].lines[0]) == "第一行已改"
    finally:
        os.unlink(db_path)

    print("test_project_store_update_proof_lines_requires_stable_uid_match PASSED")


def test_project_store_new_db_records_current_schema_version():
    import sqlite3

    from app.core.logging import APP_VERSION, SCHEMA_VERSION
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        with ProjectStore(db_path) as store:
            store.log_operation(
                project_id=1,
                action="proof_edit",
                object_type="line",
                object_id=7,
                object_uid="line_test_uid",
                payload={"field": "final_text"},
            )
            store.log_operation(
                1,
                "legacy_edit",
                "line",
                8,
                2,
                {"field": "legacy"},
            )

        conn = sqlite3.connect(db_path)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            operation_cols = conn.execute("PRAGMA table_info(operation_log)").fetchall()
            log_rows = conn.execute(
                "SELECT object_id, object_uid, page_id, action, payload_json "
                "FROM operation_log ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

        assert int(meta["schema_version"]) == SCHEMA_VERSION
        assert meta["app_version"] == APP_VERSION
        assert any(col[1] == "object_uid" for col in operation_cols)
        assert log_rows == [
            (7, "line_test_uid", None, "proof_edit", '{"field": "final_text"}'),
            (8, "", 2, "legacy_edit", '{"field": "legacy"}'),
        ]

        print("test_project_store_new_db_records_current_schema_version PASSED")
    finally:
        os.unlink(db_path)


# =====================================================================
# ProofAutoFlagService 测试
# =====================================================================

def test_proof_auto_flag_service():
    from app.models import BBox, Block, BlockType, Line, Page, ProofStatus
    from app.services.proof_auto_flag_service import ProofAutoFlagService

    bb = BBox(0, 0, 100, 20)
    lines = [
        Line(text="高置信", confidence=0.95, bbox=bb),
        Line(text="低置信", confidence=0.65, bbox=bb),
        Line(text="边界值", confidence=0.80, bbox=bb),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=lines)
    page = Page(image_path="/tmp/x.jpg", width=800, height=600, blocks=[block])

    service = ProofAutoFlagService(threshold=0.80)
    count = service.auto_flag([page])
    assert count == 1
    assert proof_status(lines[1]) == ProofStatus.AUTO_FLAGGED
    print("test_proof_auto_flag_service PASSED")


# =====================================================================
# 导出测试
# =====================================================================

def test_export_txt():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.export.txt import TxtExporter, txt_output_paths
    bb = BBox(0, 0, 100, 20)
    line = Line(text="导出测试行", confidence=0.9, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block])
    project = OcrProject(name="TxtTest", pages=[page])
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out_path = f.name
    try:
        TxtExporter().export(project, out_path)
        utf8_path, gbk_path = txt_output_paths(out_path)
        assert "=== 第 1 页 ===" in open(utf8_path, encoding="utf-8").read()
        assert "导出测试行" in open(utf8_path, encoding="utf-8").read()
        assert "导出测试行" in open(gbk_path, encoding="gbk").read()
        print("test_export_txt PASSED")
    finally:
        for path in set((out_path, *txt_output_paths(out_path))):
            if os.path.exists(path):
                os.unlink(path)


def test_txt_dual_encoding_outputs_and_layout_contract():
    from app.export.txt import TxtExporter, txt_output_paths
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    page = Page(image_path="/tmp/img.jpg", width=800, height=600, page_number=1, blocks=[
        Block(block_type=BlockType.TITLE, bbox=BBox(0, 0, 100, 20), order=0, lines=[
            Line(text="标题", confidence=0.95, bbox=BBox(0, 0, 100, 20)),
        ]),
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 100, 40), order=1, lines=[
            Line(text="Hello", confidence=0.9, bbox=BBox(0, 30, 100, 20)),
            Line(text="world", confidence=0.9, bbox=BBox(0, 50, 100, 20)),
        ]),
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 80, 100, 40), order=2, lines=[
            Line(text="中文", confidence=0.9, bbox=BBox(0, 80, 100, 20)),
            Line(text="正文", confidence=0.9, bbox=BBox(0, 100, 100, 20)),
        ]),
        Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(0, 130, 100, 20), order=3, lines=[
            Line(text="图注", confidence=0.9, bbox=BBox(0, 130, 100, 20)),
        ]),
        Block(block_type=BlockType.REFERENCE, bbox=BBox(0, 160, 100, 40), order=4, lines=[
            Line(text="参考一", confidence=0.9, bbox=BBox(0, 160, 100, 20)),
            Line(text="参考二", confidence=0.9, bbox=BBox(0, 180, 100, 20)),
        ]),
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 210, 100, 20), order=5, lines=[
            Line(text="E=mc^2", confidence=0.9, bbox=BBox(0, 210, 100, 20)),
        ]),
    ])
    project = OcrProject(name="TxtDual", pages=[page])
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "TxtDual.txt")
        utf8_path, gbk_path = txt_output_paths(out_path)
        TxtExporter().export(project, out_path)
        assert os.path.exists(utf8_path)
        assert os.path.exists(gbk_path)
        utf8_bytes = open(utf8_path, "rb").read()
        gbk_bytes = open(gbk_path, "rb").read()
        assert not utf8_bytes.startswith(b"\xef\xbb\xbf")
        content = utf8_bytes.decode("utf-8")
        assert gbk_bytes.decode("gbk") == content
        assert content == (
            "=== 第 1 页 ===\n"
            "标题\n\n"
            "Hello world\n\n"
            "中文正文\n\n"
            "图注\n\n"
            "参考一\n参考二\n\n"
            "E=mc^2\n"
        )
        assert "图注：" not in content
        assert "参考文献" not in content
        assert "公式：" not in content

    print("test_txt_dual_encoding_outputs_and_layout_contract PASSED")


def test_export_xml():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.export.xml import XmlExporter
    from lxml import etree
    bb = BBox(10, 20, 200, 30)
    line = Line(text="XML测试", confidence=0.88, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/img.jpg", width=800, height=600,
                blocks=[block], page_number=1)
    project = OcrProject(name="XmlTest", pages=[page])
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        out_path = f.name
    try:
        XmlExporter().export(project, out_path)
        root = etree.parse(out_path).getroot()
        assert root.tag == "OcrArchive"
        assert root.get("archive_role") == "primary_authority"
        assert root.get("authority") == "xml"
        ln = root.findall(".//Line")[0]
        assert ln.text == "XML测试"
        assert ln.get("x") is None
        assert ln.find("BBox").get("x") == "10"
        print("test_export_xml PASSED")
    finally:
        os.unlink(out_path)


def test_export_html():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.export.html import HtmlExporter
    bb = BBox(0, 0, 100, 20)
    line = Line(text="HTML测试内容", confidence=0.9, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block])
    project = OcrProject(name="HtmlTest", pages=[page])
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out_path = f.name
    try:
        HtmlExporter().export(project, out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "HTML测试内容" in content
        assert "<!DOCTYPE html>" in content
        print("test_export_html PASSED")
    finally:
        os.unlink(out_path)


def test_export_markdown_structure():
    from app.export.markdown import MarkdownExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "assets", "figure.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        open(image_path, "wb").write(b"not-a-real-image")
        out_path = os.path.join(tmpdir, "out.md")
        bb = BBox(10, 20, 200, 30)
        page = Page(
            image_path=image_path,
            width=800,
            height=600,
            page_number=1,
            blocks=[
                Block(block_type=BlockType.TITLE, bbox=bb, order=0, lines=[
                    Line(text="章节标题", confidence=0.95, bbox=bb),
                ]),
                Block(block_type=BlockType.TEXT, bbox=BBox(10, 80, 200, 80), order=1, lines=[
                    Line(text="1. 这不是列表", confidence=0.90, bbox=BBox(10, 80, 200, 24)),
                    Line(text="again", confidence=0.91, bbox=BBox(10, 110, 200, 24)),
                ]),
                Block(block_type=BlockType.REFERENCE, bbox=BBox(10, 150, 200, 30), order=2, lines=[
                    Line(text="1) Ref not list", confidence=0.93, bbox=BBox(10, 150, 200, 30)),
                ]),
                Block(block_type=BlockType.FIGURE, bbox=BBox(10, 180, 200, 30), order=3, note="Figure Alt"),
                Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(10, 230, 200, 30), order=4, lines=[
                    Line(text="1. 图一 *示例*", confidence=0.93, bbox=BBox(10, 230, 200, 30)),
                ]),
                Block(block_type=BlockType.TABLE, bbox=BBox(10, 260, 200, 30), order=5, lines=[
                    Line(text="Cell <1>", confidence=0.93, bbox=BBox(10, 260, 200, 30)),
                ]),
                Block(block_type=BlockType.TABLE_CAPTION, bbox=BBox(10, 290, 200, 30), order=6, lines=[
                    Line(text="表一", confidence=0.93, bbox=BBox(10, 290, 200, 30)),
                ]),
                Block(block_type=BlockType.EQUATION, bbox=BBox(10, 320, 200, 30), order=7, lines=[
                    Line(text="E = mc^2", confidence=0.88, bbox=BBox(10, 320, 200, 30)),
                ]),
                Block(block_type=BlockType.EQUATION, bbox=BBox(10, 350, 200, 30), order=8),
                Block(block_type=BlockType.UNKNOWN, bbox=BBox(10, 380, 200, 30), order=9, lines=[
                    Line(text="Unknown <block>", confidence=0.88, bbox=BBox(10, 380, 200, 30)),
                ]),
            ],
        )
        project = OcrProject(name="MdTest", pages=[page])
        MarkdownExporter().export(project, out_path)
        raw = open(out_path, "rb").read()
        assert not raw.startswith(b"\xef\xbb\xbf")
        content = open(out_path, encoding="utf-8").read()
        assert "## 第 1 页" not in content
        assert "<!--" not in content
        assert content.startswith("# 章节标题")
        assert "1\\. 这不是列表 again" in content
        assert "### 参考文献" not in content
        assert "1\\) Ref not list" in content
        assert "![Figure Alt](assets/figure.png)" in content
        assert "*1\\. 图一 \\*示例\\**" in content
        assert "<table>" in content
        assert "<td>Cell &lt;1&gt;</td>" in content
        assert "*表一*" in content
        assert "$$\nE = mc^2\n$$" in content
        assert "![equation](assets/figure.png)" in content
        assert "background-color:#fff3cd" in content
        assert "Unknown &lt;block&gt;" in content
        assert content.index("# 章节标题") < content.index("1\\. 这不是列表") < content.index("1\\) Ref not list")
        print("test_export_markdown_structure PASSED")


def test_markdown_fallback_assets_are_cropped_regions():
    from PIL import Image

    from app.export.markdown import MarkdownExporter
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        Image.new("RGB", (120, 90), "white").save(image_path)
        out_path = os.path.join(tmpdir, "out.md")
        page = Page(
            image_path=image_path,
            width=120,
            height=90,
            page_number=1,
            blocks=[
                Block(block_type=BlockType.TABLE, bbox=BBox(10, 20, 30, 15), order=0),
                Block(block_type=BlockType.EQUATION, bbox=BBox(50, 30, 20, 10), order=1),
            ],
        )
        project = OcrProject(name="MdCrop", pages=[page])

        MarkdownExporter().export(project, out_path)

        content = open(out_path, encoding="utf-8").read()
        assert "image_fallback" not in content
        assert "../page.png" not in content
        assert "![table](out_assets/asset-el-p1-0001_table_crop.png)" in content
        assert "![equation](out_assets/asset-el-p1-0002_equation_crop.png)" in content
        table_crop = os.path.join(tmpdir, "out_assets", "asset-el-p1-0001_table_crop.png")
        equation_crop = os.path.join(tmpdir, "out_assets", "asset-el-p1-0002_equation_crop.png")
        assert os.path.exists(table_crop)
        assert os.path.exists(equation_crop)
        assert Image.open(table_crop).size == (30, 15)
        assert Image.open(equation_crop).size == (20, 10)

    print("test_markdown_fallback_assets_are_cropped_regions PASSED")


def test_markdown_export_settings_filter_and_merge_layout_fragments():
    from app.export.markdown import MarkdownExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        open(image_path, "wb").write(b"not-a-real-image")
        out_path = os.path.join(tmpdir, "out.md")
        page = Page(
            image_path=image_path,
            width=800,
            height=600,
            page_number=197,
            blocks=[
                Block(block_type=BlockType.TEXT, source_label="header", bbox=BBox(0, 0, 200, 20), order=0, lines=[
                    Line(text="魏薇等：页眉", confidence=0.9, bbox=BBox(0, 0, 200, 20)),
                ]),
                Block(block_type=BlockType.TEXT, bbox=BBox(10, 80, 200, 40), order=1, lines=[
                    Line(text="正文内容", confidence=0.9, bbox=BBox(10, 80, 200, 40)),
                ]),
                Block(block_type=BlockType.TABLE_CAPTION, source_label="table_title", bbox=BBox(10, 130, 200, 20), order=2, lines=[
                    Line(text="表3", confidence=0.9, bbox=BBox(10, 130, 200, 20)),
                ]),
                Block(block_type=BlockType.TABLE_CAPTION, source_label="table_title", bbox=BBox(10, 154, 300, 20), order=3, lines=[
                    Line(text="内生性检验一工具变量回归与剥离同期政策影响", confidence=0.9, bbox=BBox(10, 154, 300, 20)),
                ]),
                Block(block_type=BlockType.TABLE, source_label="table", bbox=BBox(10, 180, 300, 80), order=4, lines=[
                    Line(text="<table><tr><td>A</td></tr></table>", confidence=0.9, bbox=BBox(10, 180, 300, 80)),
                ]),
                Block(block_type=BlockType.EQUATION, source_label="display_formula", bbox=BBox(10, 280, 300, 40), order=5, lines=[
                    Line(text="$$ \\begin{aligned}x=y\\end{aligned} $$", confidence=0.9, bbox=BBox(10, 280, 300, 40)),
                ]),
                Block(block_type=BlockType.EQUATION, source_label="formula_number", bbox=BBox(330, 280, 40, 20), order=6, lines=[
                    Line(text="（9）", confidence=0.9, bbox=BBox(330, 280, 40, 20)),
                ]),
                Block(block_type=BlockType.TEXT, source_label="number", bbox=BBox(380, 560, 40, 20), order=7, lines=[
                    Line(text="197", confidence=0.9, bbox=BBox(380, 560, 40, 20)),
                ]),
            ],
        )
        project = OcrProject(name="MdPolicy", pages=[page])

        MarkdownExporter().export(project, out_path)

        content = open(out_path, encoding="utf-8").read()
        assert "魏薇等：页眉" not in content
        assert "\n197\n" not in f"\n{content}\n"
        assert "*表3 内生性检验一工具变量回归与剥离同期政策影响*" in content
        assert "*表3*\n\n*内生性检验" not in content
        assert "<table><tr><td>A</td></tr></table>" in content
        assert "&lt;table&gt;" not in content
        assert "$$\n\\begin{aligned}x=y\\end{aligned} \\tag{9}\n$$" in content
        assert "$$\n$$" not in content

    print("test_markdown_export_settings_filter_and_merge_layout_fragments PASSED")


def test_markdown_filter_ignores_raw_payload_labels():
    from app.export.markdown import MarkdownExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        open(image_path, "wb").write(b"not-a-real-image")
        out_path = os.path.join(tmpdir, "out.md")
        page = Page(
            image_path=image_path,
            width=200,
            height=120,
            page_number=1,
                blocks=[
                    Block(
                        block_type=BlockType.TEXT,
                        bbox=BBox(0, 0, 120, 24),
                        order=0,
                        lines=[Line(text="正文不能被 raw header 过滤", confidence=0.9, bbox=BBox(0, 0, 120, 24))],
                    ),
                Block(
                    block_type=BlockType.TEXT,
                    source_label="header",
                    bbox=BBox(0, 30, 120, 24),
                    order=1,
                    lines=[Line(text="结构化页眉应被过滤", confidence=0.9, bbox=BBox(0, 30, 120, 24))],
                ),
            ],
        )
        MarkdownExporter().export(OcrProject(name="MdRawLabelIgnored", pages=[page]), out_path)
        content = open(out_path, encoding="utf-8").read()

    assert "正文不能被 raw header 过滤" in content
    assert "结构化页眉应被过滤" not in content

    print("test_markdown_filter_ignores_raw_payload_labels PASSED")


def test_export_formats_share_structured_blocks():
    from app.export import get_exporter
    from app.export.markdown import MarkdownExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    text_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 60, 100, 20),
        order=2,
        lines=[Line(text="正文", confidence=0.9, bbox=BBox(0, 60, 100, 20))],
    )
    caption_block = Block(
        block_type=BlockType.TABLE_CAPTION,
        bbox=BBox(0, 20, 100, 20),
        order=1,
        lines=[Line(text="表注文字", confidence=0.9, bbox=BBox(0, 20, 100, 20))],
    )
    page = Page(
        image_path="/tmp/img.jpg",
        width=800,
        height=600,
        blocks=[text_block, caption_block],
    )
    project = OcrProject(name="StructuredExport", pages=[page])

    assert isinstance(get_exporter("markdown"), MarkdownExporter)
    for fmt in ("txt", "html", "xml", "md"):
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            out_path = f.name
        try:
            exporter = get_exporter(fmt)
            exporter.export(project, out_path)
            read_path = out_path[:-len(".txt")] + ".utf8.txt" if fmt == "txt" else out_path
            content = open(read_path, encoding="utf-8").read()
            assert "表注文字" in content
            assert content.index("表注文字") < content.index("正文")
        finally:
            cleanup_paths = {out_path}
            if fmt == "txt":
                cleanup_paths.update({out_path[:-len(".txt")] + ".utf8.txt", out_path[:-len(".txt")] + ".gbk.txt"})
            for path in cleanup_paths:
                if os.path.exists(path):
                    os.unlink(path)

    print("test_export_formats_share_structured_blocks PASSED")


def test_export_ir_rules_load_and_validate():
    from app.export.rules import load_export_rules

    rules = load_export_rules()
    assert rules.version == "export_ir.v1"
    assert rules.profile_for("json").format == "json"
    assert rules.profile_for("json").archive_role == "semantic_mirror"
    assert rules.profile_for("xml").archive_role == "primary_authority"
    assert rules.profile_for("pdf").format == "pdf-single"
    assert rules.profile_for("pdf-single").mode == "page-faithful"
    assert rules.profile_for("pdf-single").options["pdf_layer"] == "image-only"
    assert rules.profile_for("markdown").format == "md"
    assert rules.profile_for("md").format == "md"
    assert rules.profile_for("pdf-dual").mode == "page-faithful"
    assert rules.profile_for("pdf-dual").options["text_layer"] == "invisible-char"
    assert rules.formats["pdf-single"]["pdf_layer"] == "image-only"
    assert rules.formats["pdf-single"]["text_layer"] == "none"
    assert rules.formats["pdf-dual"]["text_layer"] == "invisible-char"
    assert rules.formats["pdf-dual"]["dpi"] == 300
    assert rules.kind_for_block_type("text") == "paragraph"
    assert rules.rule_for_kind("table")["asset_kind"] == "table_crop"
    assert rules.archive["authority_format"] == "xml"
    assert rules.archive["mirror_format"] == "json"
    assert "elements.source" in rules.archive["mandatory_parity"]
    for kind in (
        "title", "paragraph", "reference", "figure", "figure_caption",
        "table", "table_caption", "equation", "unknown",
    ):
        assert kind in rules.kind_rules

    print("test_export_ir_rules_load_and_validate PASSED")


def test_project_to_export_ir_builder_maps_final_text_and_fallbacks():
    from app.export.ir_builder import build_export_ir
    from app.export.rules import load_export_rules
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page, ProofStatus

    bb = BBox(1, 2, 30, 40)
    edited = Line(text="OCR原文", confidence=0.9, bbox=bb)
    set_line_proof_status(edited, ProofStatus.MODIFIED)
    edited.ocr_text = "OCR原文"
    edited.chars = [Char(char="O", confidence=0.9, bbox=bb)]
    set_line_proof_text(edited, "人工终审")
    table_line = Line(text="表格文字", confidence=0.8, bbox=bb)
    page = Page(
        image_path="/tmp/page.png",
        cache_image_path="/tmp/cache.png",
        source_path="/tmp/source.pdf",
        source_type="pdf",
        source_page_index=1,
        width=100,
        height=200,
        blocks=[
            Block(block_type=BlockType.TITLE, bbox=bb, order=0, lines=[Line(text="标题", confidence=0.95, bbox=bb)]),
            Block(block_type=BlockType.TEXT, bbox=bb, order=1, lines=[edited]),
            Block(block_type=BlockType.REFERENCE, bbox=bb, order=2, lines=[Line(text="参考", confidence=0.9, bbox=bb)]),
            Block(block_type=BlockType.FIGURE, bbox=bb, order=3),
            Block(block_type=BlockType.FIGURE_CAPTION, bbox=bb, order=4, lines=[Line(text="图注", confidence=0.9, bbox=bb)]),
            Block(block_type=BlockType.TABLE, bbox=bb, order=5, lines=[table_line]),
            Block(block_type=BlockType.TABLE_CAPTION, bbox=bb, order=6, lines=[Line(text="表注", confidence=0.9, bbox=bb)]),
            Block(block_type=BlockType.EQUATION, bbox=bb, order=7, lines=[Line(text="E=mc^2", confidence=0.9, bbox=bb)]),
            Block(block_type=BlockType.UNKNOWN, bbox=bb, order=8),
        ],
    )
    project = OcrProject(name="IRProject", pages=[page])
    rules = load_export_rules()
    rules.kind_rules["table"]["asset_kind"] = "json_controlled_table_crop"
    rules.fallback_strategies["image_fallback"]["reason"] = "json_controlled_missing_structure"
    document = build_export_ir(project, "json", rules=rules)
    data = document.to_dict()
    kinds = [element["kind"] for element in data["pages"][0]["elements"]]
    assert kinds == [
        "title", "paragraph", "reference", "figure", "figure_caption",
        "table", "table_caption", "equation", "unknown",
    ]
    paragraph = data["pages"][0]["elements"][1]
    assert paragraph["source"]["block_ids"] == [page.blocks[1].uid]
    assert paragraph["source"]["line_ids"] == [edited.uid]
    assert paragraph["source"]["char_ids"] == [edited.chars[0].uid]
    assert paragraph["payload"]["text"] == "人工终审"
    assert paragraph["payload"]["lines"][0]["line_id"] == edited.uid
    assert paragraph["payload"]["lines"][0]["chars"][0]["char_id"] == edited.chars[0].uid
    assert paragraph["payload"]["lines"][0]["ocr_text"] == "OCR原文"
    assert paragraph["proof"]["corrected"] is True
    table = data["pages"][0]["elements"][5]
    assert table["payload"]["mode"] == "image_fallback"
    assert table["fallback"]["mode"] == "image_fallback"
    assert table["fallback"]["reason"] == "json_controlled_missing_structure"
    assert any(d["code"] == "table_fallback_to_image" for d in data["diagnostics"])
    assert any(asset["kind"] == "json_controlled_table_crop" for asset in data["assets"])
    assert data["pages"][0]["source_image"] == "cache.png"
    assert data["pages"][0]["source_meta"]["source_path"] == "source.pdf"
    assert all(asset["path"] == "cache.png" for asset in data["assets"])

    print("test_project_to_export_ir_builder_maps_final_text_and_fallbacks PASSED")


def test_export_ir_keeps_render_paths_only_for_rendering_formats(tmp_path):
    from app.export.ir_builder import build_export_ir
    from app.models import BBox, Block, BlockType, OcrProject, Page

    image_path = tmp_path / "cache" / "page.png"
    source_path = tmp_path / "source" / "book.pdf"
    image_path.parent.mkdir()
    source_path.parent.mkdir()
    image_path.write_bytes(b"fake")
    source_path.write_bytes(b"fake")
    page = Page(
        image_path=str(image_path),
        cache_image_path=str(image_path),
        source_path=str(source_path),
        source_type="pdf",
        source_page_index=1,
        width=100,
        height=200,
        blocks=[Block(block_type=BlockType.FIGURE, bbox=BBox(1, 2, 30, 40), order=1)],
    )
    project = OcrProject(name="private-paths", pages=[page])

    json_data = build_export_ir(project, "json").to_dict()
    xml_data = build_export_ir(project, "xml").to_dict()
    html_data = build_export_ir(project, "html").to_dict()
    md_data = build_export_ir(project, "md").to_dict()

    assert str(tmp_path) not in json.dumps(json_data, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(xml_data, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(html_data, ensure_ascii=False)
    assert json_data["pages"][0]["source_image"] == "page.png"
    assert json_data["pages"][0]["source_meta"]["source_path"] == "book.pdf"
    assert md_data["pages"][0]["source_image"] == str(image_path)
    assert md_data["pages"][0]["source_meta"]["source_path"] == str(source_path)

    page.cache_image_path = r"D:\project\ocr_process\.cache\images\page.png"
    page.source_path = r"D:\project\ocr_process\file\book.pdf"
    windows_data = build_export_ir(project, "json").to_dict()
    assert windows_data["pages"][0]["source_image"] == "page.png"
    assert windows_data["pages"][0]["source_meta"]["source_path"] == "book.pdf"
    assert "D:" not in json.dumps(windows_data, ensure_ascii=False)

    print("test_export_ir_keeps_render_paths_only_for_rendering_formats PASSED")


def test_export_ir_char_source_fallbacks_are_unique_across_lines():
    from app.export.ir_builder import build_export_ir
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    bb = BBox(1, 2, 30, 10)
    lines = [
        Line(
            text="甲乙",
            confidence=0.9,
            bbox=bb,
            chars=[
                Char(char="甲", confidence=0.9, bbox=bb),
                Char(char="乙", confidence=0.9, bbox=bb),
            ],
        ),
        Line(
            text="丙丁",
            confidence=0.9,
            bbox=BBox(1, 20, 30, 10),
            chars=[
                Char(char="丙", confidence=0.9, bbox=bb),
                Char(char="丁", confidence=0.9, bbox=bb),
            ],
        ),
    ]
    for line in lines:
        line.uid = ""
        for char in line.chars:
            char.uid = ""
    block = Block(block_type=BlockType.TEXT, bbox=bb, order=0, lines=lines)
    block.uid = ""
    project = OcrProject(
        name="fallback source ids",
        pages=[Page(image_path="/tmp/page.png", width=100, height=100, blocks=[block])],
    )

    document = build_export_ir(project, "json")
    element = document.to_dict()["pages"][0]["elements"][0]
    source = element["source"]
    assert source["line_ids"] == [0, 1]
    assert source["char_ids"] == [
        "line-0-char-0",
        "line-0-char-1",
        "line-1-char-0",
        "line-1-char-1",
    ]
    payload_char_ids = [
        char["char_id"]
        for line_payload in element["payload"]["lines"]
        for char in line_payload["chars"]
    ]
    assert payload_char_ids == source["char_ids"]

    print("test_export_ir_char_source_fallbacks_are_unique_across_lines PASSED")


def test_export_ir_preserves_structured_block_attributes():
    from app.export.ir_builder import build_export_ir
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 2, 30, 40),
        order=0,
        lines=[Line(text="标题文本", confidence=0.95, bbox=BBox(1, 2, 30, 10))],
        source_label="text",
        origin=BlockOrigin(source_label="paragraph_title"),
    )
    page = Page(image_path="/tmp/attrs-page.png", width=100, height=100, blocks=[block])
    document = build_export_ir(OcrProject(name="AttrIR", pages=[page]), "json")
    element = document.to_dict()["pages"][0]["elements"][0]

    assert element["kind"] == "title"
    assert element["source"]["block_type"] == "text"
    assert element["source"]["source_label"] == "paragraph_title"
    assert element["source"]["semantic_label"] == "paragraph_title"
    assert element["source"]["semantic_block_type"] == "title"
    assert "raw_payload" not in element["source"]
    assert "raw_payload" not in element["layout_attributes"]
    assert element["layout_attributes"]["semantic_block_type"] == "title"

    print("test_export_ir_preserves_structured_block_attributes PASSED")


def test_export_ir_reads_layout_facts_from_snapshot_view():
    from app.export.ir_builder import build_export_ir
    from app.models import (
        BBox, Block, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot,
        Line, OcrPolicy, OcrProject, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page

    runtime_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 10, 10),
        order=0,
        source_label="text",
        lines=[Line(text="<table><tr><td>A</td></tr></table>", confidence=1.0, bbox=BBox(50, 50, 500, 160))],
    )
    page = Page(image_path="/tmp/snapshot-export.png", width=700, height=320, blocks=[runtime_block])
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="",
        source_engine="test",
        source_run_id="",
        blocks=(
            LayoutBlockSnapshot(
                uid=runtime_block.uid,
                block_type=BlockType.TABLE,
                bbox=BBox(50, 50, 500, 160),
                order=0,
                source_label="table",
                origin=BlockOrigin(source_label="table", original_bbox=BBox(50, 50, 500, 160), original_kind=BlockType.TABLE),
                ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
            ),
        ),
    ))

    document = build_export_ir(OcrProject(name="SnapshotExport", pages=[page]), "json")
    element = document.to_dict()["pages"][0]["elements"][0]

    assert element["kind"] == "table"
    assert element["bbox"] == {"x": 50, "y": 50, "w": 500, "h": 160}
    assert element["source"]["block_ids"] == [runtime_block.uid]
    assert element["source"]["block_type"] == "table"
    assert element["layout_attributes"]["semantic_block_type"] == "table"
    assert any(asset["bbox"] == {"x": 50, "y": 50, "w": 500, "h": 160} for asset in document.to_dict()["assets"])

    print("test_export_ir_reads_layout_facts_from_snapshot_view PASSED")


def test_pdf_page_faithful_plans_use_image_and_char_layer():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans, pixel_bbox_to_pdf_rect
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    title_char = Char(char="题", confidence=0.99, bbox=BBox(10, 20, 5, 10), bbox_source="ocr", bbox_granularity="char")
    table_char = Char(char="表", confidence=0.98, bbox=BBox(30, 50, 8, 12), bbox_source="ocr", bbox_granularity="char")
    equation_char = Char(char="E", confidence=0.97, bbox=BBox(60, 80, 6, 9), bbox_source="ocr", bbox_granularity="char")
    project = OcrProject(name="PdfPlan", pages=[
        Page(image_path="/tmp/page1.png", width=100, height=200, blocks=[
            Block(block_type=BlockType.TITLE, bbox=BBox(10, 20, 20, 10), order=0, lines=[
                Line(text="题", confidence=0.99, bbox=BBox(10, 20, 5, 10), chars=[title_char]),
            ]),
            Block(block_type=BlockType.TEXT, bbox=BBox(10, 35, 20, 10), order=1, lines=[
                Line(text="正", confidence=0.96, bbox=BBox(10, 35, 10, 10)),
            ]),
            Block(block_type=BlockType.TABLE, bbox=BBox(30, 50, 20, 12), order=2, lines=[
                Line(text="表", confidence=0.98, bbox=BBox(30, 50, 8, 12), chars=[table_char]),
            ]),
            Block(block_type=BlockType.EQUATION, bbox=BBox(60, 80, 20, 9), order=3, lines=[
                Line(text="E", confidence=0.97, bbox=BBox(60, 80, 6, 9), chars=[equation_char]),
            ]),
        ]),
        Page(image_path="/tmp/page2.png", width=100, height=200, blocks=[
            Block(block_type=BlockType.FIGURE, bbox=BBox(5, 5, 50, 50), order=0),
        ]),
    ])

    document = build_export_ir(project, "pdf-dual")
    assert document.profile.options["dpi"] == 300
    assert document.profile.options["text_layer"] == "invisible-char"
    assert document.pages[0].elements[0].payload["lines"][0]["chars"][0]["bbox"] == {"x": 10, "y": 20, "w": 5, "h": 10}

    single_plans = build_pdf_page_plans(document, include_text=False, dpi=100)
    dual_plans = build_pdf_page_plans(document, include_text=True, dpi=100)
    assert len(single_plans) == 2
    assert single_plans[0].image_path == "/tmp/page1.png"
    assert single_plans[0].text_items == []
    assert [item.text for item in dual_plans[0].text_items] == ["题", "正", "表", "E"]
    assert dual_plans[1].image_path == "/tmp/page2.png"
    assert dual_plans[1].text_items == []

    x, y, w, h = pixel_bbox_to_pdf_rect({"x": 10, "y": 20, "w": 5, "h": 10}, 200, dpi=100)
    assert (round(x, 2), round(y, 2), round(w, 2), round(h, 2)) == (7.2, 122.4, 3.6, 7.2)
    first = dual_plans[0].text_items[0]
    assert (round(first.x, 2), round(first.y, 2), round(first.w, 2), round(first.h, 2)) == (7.2, 122.4, 3.6, 7.2)

    print("test_pdf_page_faithful_plans_use_image_and_char_layer PASSED")


def test_pdf_dual_textless_page_degrades_without_text_font():
    import app.export.pdf as pdf_module
    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, OcrProject, Page

    original_find_font = pdf_module._find_font
    pdf_module._find_font = lambda: None
    try:
        project = OcrProject(name="PdfTextless", pages=[
            Page(image_path="/tmp/textless-page.png", width=100, height=200, blocks=[
                Block(block_type=BlockType.FIGURE, bbox=BBox(5, 5, 50, 50), order=0),
            ]),
        ])
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            out_path = f.name
        try:
            PdfExporter("pdf-dual").export(project, out_path)
            assert os.path.getsize(out_path) > 0
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
    finally:
        pdf_module._find_font = original_find_font

    print("test_pdf_dual_textless_page_degrades_without_text_font PASSED")


def test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_region_fonts():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (120, 160), "white").save(image_path)
        project = OcrProject(name="PdfSearch", pages=[
            Page(image_path=image_path, width=120, height=160, blocks=[
                Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 40, 20), order=0, lines=[
                    Line(text="大小", confidence=0.99, bbox=BBox(10, 20, 40, 20), chars=[
                        Char(char="大", confidence=0.99, bbox=BBox(10, 20, 20, 20), bbox_source="ocr", bbox_granularity="char"),
                        Char(char="小", confidence=0.99, bbox=BBox(30, 20, 20, 20), bbox_source="ocr", bbox_granularity="char"),
                    ]),
                ]),
                Block(block_type=BlockType.EQUATION, bbox=BBox(10, 60, 40, 10), order=1, lines=[
                    Line(text="多少", confidence=0.99, bbox=BBox(10, 60, 40, 10), chars=[
                        Char(char="多", confidence=0.99, bbox=BBox(10, 60, 20, 10), bbox_source="ocr", bbox_granularity="char"),
                        Char(char="少", confidence=0.99, bbox=BBox(30, 60, 20, 10), bbox_source="ocr", bbox_granularity="char"),
                    ]),
                ]),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            assert len(page.search_for("大")) == 1
            assert len(page.search_for("小")) == 1
            assert len(page.search_for("大小")) == 1
            assert len(page.search_for("多少")) == 1
            first_rect = page.search_for("大小")[0]
            second_rect = page.search_for("多少")[0]
            expected_x0 = 10 * 72 / 300
            expected_x1 = 50 * 72 / 300
            expected_w = 40 * 72 / 300
            assert abs(first_rect.x0 - expected_x0) < 0.3
            assert abs(first_rect.x1 - expected_x1) < 0.3
            assert abs(first_rect.width - expected_w) < 0.3
            assert abs(second_rect.x0 - expected_x0) < 0.3
            assert abs(second_rect.x1 - expected_x1) < 0.3
            assert abs(second_rect.width - expected_w) < 0.3
            assert abs(first_rect.y0 - (20 * 72 / 300)) < 0.3
            assert abs(second_rect.y0 - (60 * 72 / 300)) < 0.3
            sizes = set()
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.add(round(float(span["size"]), 2))
            assert len(sizes) == 2
            assert min(sizes) < max(sizes)
        finally:
            doc.close()

    print("test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_region_fonts PASSED")


def test_pdf_dual_positions_mixed_chars_without_copy_spaces():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    text = "资本积累（Whited和Wu，2006；Chen"
    char_specs = [
        ("资", 100, 100, 42, 46),
        ("本", 152, 100, 42, 46),
        ("积", 204, 100, 42, 46),
        ("累", 256, 100, 40, 46),
        ("（", 308, 100, 12, 46),
        ("W", 332, 107, 40, 34),
        ("h", 376, 106, 21, 35),
        ("i", 401, 108, 10, 33),
        ("t", 414, 112, 11, 29),
        ("e", 428, 113, 19, 28),
        ("d", 451, 106, 21, 35),
        ("和", 488, 101, 42, 45),
        ("W", 544, 107, 41, 34),
        ("u", 589, 113, 21, 28),
        ("，", 622, 136, 7, 13),
        ("2", 646, 107, 22, 34),
        ("0", 671, 107, 22, 34),
        ("0", 696, 107, 22, 34),
        ("6", 721, 107, 21, 34),
        ("；", 755, 121, 8, 28),
        ("C", 777, 107, 25, 34),
        ("h", 807, 106, 21, 35),
        ("e", 831, 113, 18, 28),
        ("n", 853, 113, 21, 28),
    ]
    chars = [
        Char(char=ch, confidence=0.99, bbox=BBox(x, y, w, h), bbox_source="ocr", bbox_granularity="char")
        for ch, x, y, w, h in char_specs
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (1000, 240), "white").save(image_path)
        project = OcrProject(name="PdfMixed", pages=[
            Page(image_path=image_path, width=1000, height=240, blocks=[
                Block(block_type=BlockType.TEXT, bbox=BBox(100, 100, 780, 50), order=0, lines=[
                    Line(text=text, confidence=0.99, bbox=BBox(100, 100, 780, 50), chars=chars),
                ]),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            extracted = page.get_text()
            assert text in extracted
            assert "Whited 和" not in extracted
            assert len(page.search_for("（Whited和Wu，2006；Chen")) == 1

            raw_chars = []
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        raw_chars.extend(span.get("chars", []))
            raw_text = "".join(char["c"] for char in raw_chars)
            w_idx = raw_text.index("W")
            i_idx = raw_text.index("i")
            scale = 72 / 300
            assert abs(raw_chars[w_idx]["bbox"][0] - 332 * scale) < 0.4
            assert abs(raw_chars[i_idx]["bbox"][0] - 401 * scale) < 0.4
        finally:
            doc.close()

    print("test_pdf_dual_positions_mixed_chars_without_copy_spaces PASSED")


def test_pdf_dual_inline_formula_item_does_not_shift_text_baseline():
    from app.export.pdf import (
        PDF_TEXT_ASCENDER_RATIO,
        PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO,
        PdfPagePlan,
        PdfTextItem,
        PdfTextSpan,
        _page_text_font_size,
        _span_baseline_top_y,
        _span_font_size,
    )

    text_left = PdfTextItem("甲", x=10, y=30, w=8, h=6, source="left", bbox_granularity="char")
    formula = PdfTextItem(
        "$ A_i $",
        x=18,
        y=22,
        w=22,
        h=18,
        source="formula",
        bbox_source="paddle_inline_formula",
        bbox_granularity="formula",
    )
    text_right = PdfTextItem("乙", x=40, y=30, w=8, h=6, source="right", bbox_granularity="char")
    span = PdfTextSpan(
        text="甲$ A_i $乙",
        x=10,
        y=22,
        w=38,
        h=18,
        source="mixed",
        items=(text_left, formula, text_right),
    )
    plan = PdfPagePlan(
        page_number=1,
        image_path="/tmp/pdf-formula-baseline.png",
        page_width_px=200,
        page_height_px=200,
        width_pt=50,
        height_pt=60,
        text_items=[text_left, formula, text_right],
        text_spans=[span],
    )

    font_size = _page_text_font_size(plan)
    expected = plan.height_pt - (text_left.y + text_left.h - font_size * PDF_TEXT_ASCENDER_RATIO)

    assert round(font_size, 2) == round(text_left.h * PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO, 2)
    assert round(_span_baseline_top_y(plan, span, font_size), 4) == round(expected, 4)

    formula_only_span = PdfTextSpan(
        text="$ A_i $",
        x=formula.x,
        y=formula.y,
        w=formula.w,
        h=formula.h,
        source="formula-only",
        items=(formula,),
    )
    formula_only_font_size = _span_font_size(plan, formula_only_span, font_size)
    formula_only_expected = plan.height_pt - (formula.y + formula.h - formula_only_font_size * PDF_TEXT_ASCENDER_RATIO)
    assert round(formula_only_font_size, 2) == round(formula.h * PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO, 2)
    assert round(_span_baseline_top_y(plan, formula_only_span, formula_only_font_size), 4) == round(formula_only_expected, 4)

    print("test_pdf_dual_inline_formula_item_does_not_shift_text_baseline PASSED")


def test_pdf_dual_text_bbox_ratio_can_be_profile_tuned():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    page = Page(image_path="/tmp/pdf-ratio-tuning.png", width=120, height=100, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 40, 20), order=0, lines=[
            Line(text="字", confidence=0.99, bbox=BBox(10, 20, 20, 20), chars=[
                Char(char="字", confidence=0.99, bbox=BBox(10, 20, 20, 20), bbox_source="ocr", bbox_granularity="char"),
            ]),
        ]),
    ])
    document = build_export_ir(OcrProject(name="PdfRatio", pages=[page]), "pdf-dual")
    document.profile.options["pdf_text_font_size_to_bbox_ratio"] = 0.72

    tuned = build_pdf_page_plans(
        document,
        include_text=True,
        dpi=100,
        text_font_size_to_bbox_ratio=document.profile.options["pdf_text_font_size_to_bbox_ratio"],
    )[0]
    default = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert tuned.text_font_size_to_bbox_ratio == 0.72
    assert default.text_font_size_to_bbox_ratio == 0.80

    print("test_pdf_dual_text_bbox_ratio_can_be_profile_tuned PASSED")


def test_pdf_dual_text_layer_splits_inline_formula_as_atomic_span():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    formula = "$ GGF_{it}^{Post-short} $"
    line = Line(text=f"甲{formula}乙", confidence=0.9, bbox=BBox(10, 20, 160, 30), chars=[
        Char(char="甲", confidence=0.9, bbox=BBox(10, 20, 20, 30), bbox_source="ocr", bbox_granularity="char"),
        Char(
            char=formula,
            confidence=1.0,
            bbox=BBox(35, 18, 90, 36),
            bbox_source="paddle_inline_formula",
            bbox_granularity="formula",
            token_text=formula,
        ),
        Char(char="乙", confidence=0.9, bbox=BBox(130, 20, 20, 30), bbox_source="ocr", bbox_granularity="char"),
    ])
    page = Page(image_path="/tmp/pdf-inline-formula-split.png", width=220, height=120, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(10, 20, 160, 40), lines=[line]),
    ])

    document = build_export_ir(OcrProject(name="PdfFormulaSplit", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["甲", formula, "乙"]
    assert plan.text_spans[1].items[0].bbox_source == "paddle_inline_formula"
    assert plan.text_spans[1].items[0].bbox_granularity == "formula"

    print("test_pdf_dual_text_layer_splits_inline_formula_as_atomic_span PASSED")


def test_pdf_dual_skips_duplicate_inline_formula_equation_element():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    formula = "$ A $"
    formula_bbox = BBox(30, 18, 50, 24)
    text_line = Line(text=f"甲{formula}乙", confidence=0.9, bbox=BBox(10, 20, 120, 24), chars=[
        Char(char="甲", confidence=0.9, bbox=BBox(10, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
        Char(
            char=formula,
            confidence=1.0,
            bbox=formula_bbox,
            bbox_source="paddle_inline_formula",
            bbox_granularity="formula",
            token_text=formula,
        ),
        Char(char="乙", confidence=0.9, bbox=BBox(85, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
    ])
    duplicate_formula_line = Line(
        text="$$ A $$ $$ A $$",
        confidence=1.0,
        bbox=formula_bbox,
    )
    page = Page(image_path="/tmp/pdf-inline-formula-dedupe.png", width=180, height=120, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(10, 18, 120, 30), order=0, lines=[text_line]),
        Block(
            block_type=BlockType.EQUATION,
            bbox=formula_bbox,
            order=1,
            lines=[duplicate_formula_line],
            source_label="inline_formula",
        ),
    ])

    document = build_export_ir(OcrProject(name="PdfFormulaDedupe", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["甲", formula, "乙"]
    assert [item.text for item in plan.text_items].count(formula) == 1
    assert all(item.text != "$$ A $$ $$ A $$" for item in plan.text_items)

    print("test_pdf_dual_skips_duplicate_inline_formula_equation_element PASSED")


def test_pdf_dual_dedup_ignores_raw_payload_inline_formula_label():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    formula = "$ A $"
    formula_bbox = BBox(30, 18, 50, 24)
    text_line = Line(text=f"甲{formula}乙", confidence=0.9, bbox=BBox(10, 20, 120, 24), chars=[
        Char(char="甲", confidence=0.9, bbox=BBox(10, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
        Char(
            char=formula,
            confidence=1.0,
            bbox=formula_bbox,
            bbox_source="paddle_inline_formula",
            bbox_granularity="formula",
            token_text=formula,
        ),
        Char(char="乙", confidence=0.9, bbox=BBox(85, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
    ])
    raw_only_formula_line = Line(
        text="$$ A $$ $$ A $$",
        confidence=1.0,
        bbox=formula_bbox,
    )
    page = Page(image_path="/tmp/pdf-raw-inline-label-ignored.png", width=180, height=120, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(10, 18, 120, 30), order=0, lines=[text_line]),
        Block(
            block_type=BlockType.EQUATION,
            bbox=formula_bbox,
            order=1,
            lines=[raw_only_formula_line],
            source_label="display_formula",
        ),
    ])

    document = build_export_ir(OcrProject(name="PdfRawInlineIgnored", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["甲", formula, "乙", "$$ A $$"]
    assert [item.text for item in plan.text_items].count(formula) == 1
    assert "$$ A $$" in [item.text for item in plan.text_items]

    print("test_pdf_dual_dedup_ignores_raw_payload_inline_formula_label PASSED")


def test_pdf_dual_equation_text_collapses_identical_formula_repeat():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    page = Page(image_path="/tmp/pdf-equation-collapse-repeat.png", width=180, height=120, blocks=[
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(20, 30, 100, 30),
            order=0,
            lines=[Line(text="$$ A $$ $$ A $$", confidence=1.0, bbox=BBox(20, 30, 100, 30))],
            source_label="display_formula",
        ),
    ])

    document = build_export_ir(OcrProject(name="PdfEquationCollapse", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["$$ A $$"]
    assert [item.text for item in plan.text_items] == ["$$ A $$"]

    print("test_pdf_dual_equation_text_collapses_identical_formula_repeat PASSED")


def test_pdf_dual_keeps_display_equation_even_if_it_overlaps_inline_formula_bbox():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    formula = "$ A $"
    bbox = BBox(30, 18, 50, 24)
    inline_line = Line(text=f"甲{formula}乙", confidence=0.9, bbox=BBox(10, 20, 120, 24), chars=[
        Char(char="甲", confidence=0.9, bbox=BBox(10, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
        Char(
            char=formula,
            confidence=1.0,
            bbox=bbox,
            bbox_source="paddle_inline_formula",
            bbox_granularity="formula",
            token_text=formula,
        ),
        Char(char="乙", confidence=0.9, bbox=BBox(85, 20, 18, 22), bbox_source="ocr", bbox_granularity="char"),
    ])
    display_formula = "$$ B $$"
    page = Page(image_path="/tmp/pdf-display-formula-overlap.png", width=180, height=120, blocks=[
        Block(block_type=BlockType.TEXT, bbox=BBox(10, 18, 120, 30), order=0, lines=[inline_line]),
        Block(
            block_type=BlockType.EQUATION,
            bbox=bbox,
            order=1,
            lines=[Line(text=display_formula, confidence=1.0, bbox=bbox)],
            source_label="display_formula",
        ),
    ])

    document = build_export_ir(OcrProject(name="PdfDisplayFormulaOverlap", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["甲", formula, "乙", display_formula]

    print("test_pdf_dual_keeps_display_equation_even_if_it_overlaps_inline_formula_bbox PASSED")


def test_pdf_dual_table_text_layer_uses_atomic_rows():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    page = Page(image_path="/tmp/pdf-table-rows.png", width=220, height=160, blocks=[
        Block(block_type=BlockType.TABLE, bbox=BBox(20, 30, 140, 70), order=0, lines=[
            Line(text="A1 B1", confidence=0.9, bbox=BBox(22, 34, 120, 18), chars=[
                Char(char="A", confidence=0.9, bbox=BBox(22, 34, 12, 18), bbox_source="ocr", bbox_granularity="char"),
                Char(char="1", confidence=0.9, bbox=BBox(36, 34, 10, 18), bbox_source="ocr", bbox_granularity="char"),
            ]),
            Line(text="A2 B2", confidence=0.9, bbox=BBox(22, 62, 120, 18), chars=[
                Char(char="A", confidence=0.9, bbox=BBox(22, 62, 12, 18), bbox_source="ocr", bbox_granularity="char"),
                Char(char="2", confidence=0.9, bbox=BBox(36, 62, 10, 18), bbox_source="ocr", bbox_granularity="char"),
            ]),
        ]),
    ])

    document = build_export_ir(OcrProject(name="PdfTableRows", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == ["A1 B1", "A2 B2"]
    assert [item.text for item in plan.text_items] == ["A1 B1", "A2 B2"]
    assert all(len(span.items) == 1 for span in plan.text_spans)
    assert all(span.items[0].bbox_source == "export_table" for span in plan.text_spans)
    assert all(span.items[0].bbox_granularity == "table_row" for span in plan.text_spans)

    print("test_pdf_dual_table_text_layer_uses_atomic_rows PASSED")


def test_pdf_dual_html_table_text_layer_splits_cells_without_tags():
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    html = (
        '<table><tr><td rowspan="2">变量</td><td>(1)</td><td>(2)</td></tr>'
        '<tr><td>固定资产投资</td><td>生产性支出占比</td></tr>'
        '<tr><td>Incentive $ \\times $Post</td><td>$ 0.522^{{***}} $</td><td>(0.139)</td></tr></table>'
    )
    page = Page(image_path="/tmp/pdf-table-html.png", width=260, height=180, blocks=[
        Block(block_type=BlockType.TABLE, bbox=BBox(20, 30, 180, 90), order=0, lines=[
            Line(text=html, confidence=0.9, bbox=BBox(20, 30, 180, 90)),
        ]),
    ])

    document = build_export_ir(OcrProject(name="PdfHtmlTableRows", pages=[page]), "pdf-dual")
    plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

    assert [span.text for span in plan.text_spans] == [
        "变量",
        "(1)",
        "(2)",
        "固定资产投资",
        "生产性支出占比",
        "Incentive $ \\times $Post",
        "$ 0.522^{{***}} $",
        "(0.139)",
    ]
    assert all("<" not in span.text and ">" not in span.text for span in plan.text_spans)
    assert [span.items[0].bbox_granularity for span in plan.text_spans] == [
        "table_cell",
        "table_cell",
        "table_cell",
        "table_cell",
        "table_cell",
        "table_cell",
        "table_formula_cell",
        "table_cell",
    ]
    assert round(plan.text_spans[0].h, 2) == round((90 / 3 * 2) * 72 / 100, 2)
    assert round(plan.text_spans[1].w, 2) == round((180 / 3) * 72 / 100, 2)

    print("test_pdf_dual_html_table_text_layer_splits_cells_without_tags PASSED")


def test_pdf_dual_generated_table_rows_stay_inside_table_lines():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (300, 200), "white").save(image_path)
        project = OcrProject(name="PdfTableBBox", pages=[
            Page(image_path=image_path, width=300, height=200, blocks=[
                Block(block_type=BlockType.TABLE, bbox=BBox(50, 50, 180, 80), order=0, lines=[
                    Line(text="A1 B1", confidence=1.0, bbox=BBox(55, 60, 160, 20)),
                    Line(text="A2 B2", confidence=1.0, bbox=BBox(55, 95, 160, 20)),
                ]),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            scale = 72 / 300
            first_rect = page.search_for("A1 B1")[0]
            second_rect = page.search_for("A2 B2")[0]
            assert abs(first_rect.x0 - 55 * scale) < 0.5
            assert abs(first_rect.x1 - (55 + 160) * scale) < 1.0
            assert abs(second_rect.x0 - 55 * scale) < 0.5
            assert abs(second_rect.x1 - (55 + 160) * scale) < 1.0
            assert first_rect.y1 < second_rect.y0
            assert abs(first_rect.y0 - 60 * scale) < 0.6
            assert abs(second_rect.y0 - 95 * scale) < 0.6
        finally:
            doc.close()

    print("test_pdf_dual_generated_table_rows_stay_inside_table_lines PASSED")


def test_pdf_dual_generated_html_table_cells_stay_inside_cells():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    html = (
        '<table><tr><td rowspan="2">变量</td><td>(1)</td><td>(2)</td></tr>'
        '<tr><td>固定资产投资</td><td>生产性支出占比</td></tr>'
        '<tr><td>R</td><td>$ 0.522^{{***}} $</td><td>(0.139)</td></tr></table>'
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (320, 180), "white").save(image_path)
        project = OcrProject(name="PdfHtmlTableCells", pages=[
            Page(image_path=image_path, width=320, height=180, blocks=[
                Block(block_type=BlockType.TABLE, bbox=BBox(30, 30, 210, 90), order=0, lines=[
                    Line(text=html, confidence=1.0, bbox=BBox(30, 30, 210, 90)),
                ]),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            scale = 72 / 300
            variable_rect = page.search_for("变量")[0]
            formula_rect = page.search_for("$ 0.522^{{***}} $")[0]

            assert abs(variable_rect.x0 - 30 * scale) < 0.8
            assert variable_rect.x1 <= (30 + 70) * scale + 1.0
            assert variable_rect.y0 >= 30 * scale - 0.8
            assert variable_rect.y1 <= (30 + 60) * scale + 1.5

            assert formula_rect.x0 >= (30 + 70) * scale - 0.8
            assert formula_rect.x1 <= (30 + 140) * scale + 1.0
            assert formula_rect.y0 >= (30 + 60) * scale - 0.8
            assert formula_rect.y1 <= (30 + 90) * scale + 1.5
            assert "<table>" not in page.get_text()
        finally:
            doc.close()

    print("test_pdf_dual_generated_html_table_cells_stay_inside_cells PASSED")


def test_pdf_dual_html_table_cells_prefer_image_text_clusters_over_equal_grid():
    from PIL import Image, ImageDraw, ImageFont

    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        image = Image.new("RGB", (700, 320), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        # Deliberately non-uniform columns. Equal grid would place the second
        # column near x=216; the image cluster is near x=300.
        for y in (80, 150):
            for x, text in ((70, "AAA"), (300, "BBB"), (480, "CCC")):
                draw.text((x, y), text, fill="black", font=font)
        image.save(image_path)

        html = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td><td>F</td></tr></table>"
        page = Page(image_path=image_path, width=700, height=320, blocks=[
            Block(block_type=BlockType.TABLE, bbox=BBox(50, 50, 500, 160), order=0, lines=[
                Line(text=html, confidence=1.0, bbox=BBox(50, 50, 500, 160)),
            ]),
        ])

        document = build_export_ir(OcrProject(name="PdfClusterTable", pages=[page]), "pdf-dual")
        plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]

        spans_by_text = {span.text: span for span in plan.text_spans}
        second_cell = spans_by_text["B"].items[0]
        second_row_cell = spans_by_text["E"].items[0]
        scale = 72 / 100

        assert abs(second_cell.x - 300 * scale) < 2.0
        assert abs(second_row_cell.x - 300 * scale) < 2.0
        assert second_cell.w < 40 * scale

    print("test_pdf_dual_html_table_cells_prefer_image_text_clusters_over_equal_grid PASSED")


def test_table_text_layer_service_writes_hidden_cells_for_table_block():
    from PIL import Image, ImageDraw, ImageFont

    from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.table_text_layer_service import TableTextLayerService

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        image = Image.new("RGB", (700, 320), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for y in (80, 150):
            for x, text in ((70, "AAA"), (300, "BBB"), (480, "CCC")):
                draw.text((x, y), text, fill="black", font=font)
        image.save(image_path)

        html = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td><td>F</td></tr></table>"
        block = Block(block_type=BlockType.TABLE, bbox=BBox(50, 50, 500, 160), order=0, lines=[
            Line(text=html, confidence=1.0, bbox=BBox(50, 50, 500, 160)),
        ])
        page = Page(image_path=image_path, width=700, height=320, blocks=[block])

        updated = TableTextLayerService().enrich_page(page)

        cells = block.table_text_layer_cells
        assert updated == 1
        assert len(cells) == 6
        assert cells[1]["text"] == "B"
        assert cells[1]["bbox_source"] == "image_text_cluster"
        assert abs(cells[1]["bbox"]["x"] - 300) < 3
        assert cells[1]["bbox"]["w"] < 40

    print("test_table_text_layer_service_writes_hidden_cells_for_table_block PASSED")


def test_table_text_layer_service_reads_table_bbox_from_layout_snapshot():
    from PIL import Image, ImageDraw, ImageFont

    from app.models import (
        BBox, Block, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot,
        Line, OcrPolicy, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.services.table_text_layer_service import TableTextLayerService

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        image = Image.new("RGB", (700, 320), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((300, 80), "BBB", fill="black", font=font)
        image.save(image_path)

        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 10, 10), order=0, lines=[
            Line(text=html, confidence=1.0, bbox=BBox(50, 50, 500, 160)),
        ])
        page = Page(image_path=image_path, width=700, height=320, blocks=[block])
        snapshot = LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="",
            source_engine="test",
            source_run_id="",
            blocks=(
                LayoutBlockSnapshot(
                    uid=block.uid,
                    block_type=BlockType.TABLE,
                    bbox=BBox(50, 50, 500, 160),
                    order=0,
                    source_label="table",
                    origin=BlockOrigin(original_bbox=BBox(50, 50, 500, 160), original_kind=BlockType.TABLE),
                    ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
                ),
            ),
        )
        set_layout_snapshot_for_page(page, snapshot)

        updated = TableTextLayerService().enrich_page(page)

        assert updated == 1
        assert block.table_text_layer_cells
        assert abs(block.table_text_layer_cells[1]["bbox"]["x"] - 300) < 3

    print("test_table_text_layer_service_reads_table_bbox_from_layout_snapshot PASSED")


def test_table_text_layer_service_clears_stale_cells_without_table_html_source():
    from PIL import Image

    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.services.table_text_layer_service import TableTextLayerService

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        Image.new("RGB", (240, 160), "white").save(image_path)
        block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox(20, 30, 160, 70),
            order=0,
            table_text_layer_cells=[
                {
                    "text": "<table><tr><td>A</td><td>B</td></tr></table>",
                    "bbox": {"x": 20, "y": 30, "w": 160, "h": 70},
                    "row": 0,
                    "col": 0,
                }
            ],
        )
        page = Page(image_path=image_path, width=240, height=160, blocks=[block])

        updated = TableTextLayerService().enrich_page(page)

        assert updated == 0
        assert block.table_text_layer_cells == []

    print("test_table_text_layer_service_clears_stale_cells_without_table_html_source PASSED")


def test_table_text_layer_service_accepts_raw_vendor_table_html_source():
    from PIL import Image

    from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.services.table_text_layer_service import TableTextLayerService

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        Image.new("RGB", (240, 160), "white").save(image_path)
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox(20, 30, 160, 70),
            order=0,
            origin=BlockOrigin(source_label="table", raw_index=0),
        )
        page = Page(image_path=image_path, width=240, height=160, blocks=[block])
        _attach_raw_layout_records(page, [
            {"block_label": "table", "block_bbox": [20, 30, 180, 100], "block_content": html}
        ])

        updated = TableTextLayerService().enrich_page(page)

        assert updated == 1
        assert block.table_text_layer_cells[0]["text"] == "A"

    print("test_table_text_layer_service_accepts_raw_vendor_table_html_source PASSED")


def test_pdf_dual_table_cells_use_ocr_stage_payload_before_image_inference():
    from PIL import Image

    from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
    from app.export.ir_builder import build_export_ir
    from app.export.pdf import build_pdf_page_plans
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        Image.new("RGB", (400, 220), "white").save(image_path)
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        block = Block(block_type=BlockType.TABLE, bbox=BBox(20, 40, 300, 80), order=0, lines=[
            Line(text=html, confidence=1.0, bbox=BBox(20, 40, 300, 80)),
        ])
        block.table_text_layer_cells = [
            {
                "text": "A",
                "bbox": {"x": 33, "y": 55, "w": 22, "h": 12},
                "row": 0,
                "col": 0,
                "row_span": 1,
                "col_span": 1,
                "bbox_source": "image_text_cluster",
                "bbox_granularity": "table_cell",
            },
            {
                "text": "B",
                "bbox": {"x": 260, "y": 55, "w": 20, "h": 12},
                "row": 0,
                "col": 1,
                "row_span": 1,
                "col_span": 1,
                "bbox_source": "image_text_cluster",
                "bbox_granularity": "table_cell",
            },
        ]
        page = Page(image_path=image_path, width=400, height=220, blocks=[block])
        document = build_export_ir(OcrProject(name="PdfPayloadTable", pages=[page]), "pdf-dual")
        payload = document.pages[0].elements[0].payload
        assert payload[TABLE_TEXT_LAYER_CELLS_KEY][1]["bbox"]["x"] == 260

        plan = build_pdf_page_plans(document, include_text=True, dpi=100)[0]
        spans = {span.text: span for span in plan.text_spans}
        scale = 72 / 100
        assert abs(spans["B"].items[0].x - 260 * scale) < 0.1
        assert spans["B"].items[0].bbox_source == "image_text_cluster"

    print("test_pdf_dual_table_cells_use_ocr_stage_payload_before_image_inference PASSED")


def test_pdf_dual_generated_formula_text_bbox_stays_inside_formula_block():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    formula = "$$ \\begin{aligned} GGF_{it}^{Post-short}+GGF_{it}^{Post-long}=1 \\end{aligned} $$"
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (500, 180), "white").save(image_path)
        project = OcrProject(name="PdfFormulaBBox", pages=[
            Page(image_path=image_path, width=500, height=180, blocks=[
                Block(block_type=BlockType.EQUATION, bbox=BBox(100, 60, 180, 40), order=0, lines=[
                    Line(text=formula, confidence=1.0, bbox=BBox(100, 60, 180, 40)),
                ]),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            scale = 72 / 300
            expected_x0 = 100 * scale
            expected_x1 = (100 + 180) * scale
            formula_spans = []
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = "".join(char.get("c", "") for char in span.get("chars", []))
                        if "GGF" in text:
                            formula_spans.append(span)
            assert len(formula_spans) == 1
            x0, _y0, x1, _y1 = formula_spans[0]["bbox"]
            assert x0 >= expected_x0 - 0.5
            assert x1 <= expected_x1 + 0.8
        finally:
            doc.close()

    print("test_pdf_dual_generated_formula_text_bbox_stays_inside_formula_block PASSED")


def test_pdf_dual_generated_pdf_deduplicates_inline_formula_equation_text():
    import fitz
    from PIL import Image

    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    formula = "$ GGF_{it} $"
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        pdf_path = os.path.join(tmpdir, "dual.pdf")
        Image.new("RGB", (320, 140), "white").save(image_path)
        text_line = Line(text=f"甲{formula}乙", confidence=0.9, bbox=BBox(20, 40, 220, 30), chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(20, 42, 24, 26), bbox_source="ocr", bbox_granularity="char"),
            Char(
                char=formula,
                confidence=1.0,
                bbox=BBox(50, 35, 100, 42),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text=formula,
            ),
            Char(char="乙", confidence=0.9, bbox=BBox(160, 42, 24, 26), bbox_source="ocr", bbox_granularity="char"),
        ])
        project = OcrProject(name="PdfFormulaGeneratedDedupe", pages=[
            Page(image_path=image_path, width=320, height=140, blocks=[
                Block(block_type=BlockType.TEXT, bbox=BBox(20, 35, 200, 45), order=0, lines=[text_line]),
                Block(
                    block_type=BlockType.EQUATION,
                    bbox=BBox(50, 35, 100, 42),
                    order=1,
                    lines=[Line(text="$ GGF_{it} $ $ GGF_{it} $", confidence=1.0, bbox=BBox(50, 35, 100, 42))],
                    source_label="inline_formula",
                ),
            ]),
        ])

        PdfExporter("pdf-dual").export(project, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            assert len(page.search_for("GGF")) == 1
            assert len(page.search_for("甲")) == 1
            assert len(page.search_for("乙")) == 1
        finally:
            doc.close()

    print("test_pdf_dual_generated_pdf_deduplicates_inline_formula_equation_text PASSED")


def test_pdf_invisible_text_layer_resets_render_mode_on_font_size_error():
    import app.export.pdf as pdf_module
    from app.export.pdf import PdfPagePlan, PdfTextItem, PdfTextSpan, _write_invisible_text_layer

    class _FakePdf:
        def __init__(self):
            self.ops = []

        def set_text_color(self, *args):
            pass

        def _out(self, command):
            self.ops.append(command)

        def set_font(self, *args, **kwargs):
            raise AssertionError("set_font should not run after font-size failure")

    plan = PdfPagePlan(
        page_number=1,
        image_path="/tmp/page.png",
        page_width_px=100,
        page_height_px=100,
        width_pt=24.0,
        height_pt=24.0,
        text_items=[PdfTextItem(text="字", x=1.0, y=1.0, w=2.0, h=2.0, source="probe")],
        text_spans=[PdfTextSpan(text="字", x=1.0, y=1.0, w=2.0, h=2.0, source="probe")],
    )
    pdf = _FakePdf()
    original = pdf_module._page_text_font_size
    pdf_module._page_text_font_size = lambda plan: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        try:
            _write_invisible_text_layer(pdf, plan)
            raise AssertionError("expected RuntimeError from _page_text_font_size")
        except RuntimeError as exc:
            assert str(exc) == "boom"
    finally:
        pdf_module._page_text_font_size = original

    assert pdf.ops == ["3 Tr", "0 Tr"]

    print("test_pdf_invisible_text_layer_resets_render_mode_on_font_size_error PASSED")


def test_xml_authority_and_json_mirror_archive_parity():
    import json
    import tempfile

    from lxml import etree

    from app.export import get_exporter
    from app.export.archive import json_archive_projection, xml_archive_projection
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus

    bb = BBox(5, 6, 70, 30)
    edited = Line(
        text="OCR旧文",
        confidence=0.7,
        bbox=bb,
        review_flags=["low_confidence"],
    )
    set_line_proof_status(edited, ProofStatus.AUTO_FLAGGED)
    edited.ocr_text = "OCR旧文"
    set_line_proof_text(edited, "人工终文")
    table_line = Line(text="表格文本", confidence=0.8, bbox=bb)
    page = Page(
        image_path="/tmp/archive.png",
        cache_image_path="/tmp/archive-cache.png",
        width=800,
        height=600,
        page_number=2,
        source_path="/tmp/source.pdf",
        source_type="pdf",
        source_page_index=2,
        error_message="page warning",
        blocks=[
            Block(block_type=BlockType.TEXT, bbox=bb, order=1, id=10, lines=[edited]),
            Block(block_type=BlockType.TABLE, bbox=bb, order=2, id=11, lines=[table_line]),
            Block(block_type=BlockType.FIGURE, bbox=bb, order=3, id=12),
        ],
    )
    project = OcrProject(name="ArchiveProject", pages=[page])

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = f"{tmpdir}/archive.xml"
        json_path = f"{tmpdir}/archive.json"
        get_exporter("xml").export(project, xml_path)
        get_exporter("json").export(project, json_path)

        root = etree.parse(xml_path).getroot()
        assert root.get("archive_role") == "primary_authority"
        assert root.get("authority") == "xml"
        data = json.loads(open(json_path, encoding="utf-8").read())
        assert data["profile"]["archive_role"] == "semantic_mirror"
        assert data["profile"]["authority"] == "xml"

        xml_projection = xml_archive_projection(root)
        json_projection = json_archive_projection(data)
        assert xml_projection == json_projection

        first_element = xml_projection["pages"][0]["elements"][0]
        assert first_element["source"]["block_ids"] == [page.blocks[0].uid]
        assert first_element["payload"]["lines"][0]["line_id"] == edited.uid
        assert first_element["proof"]["status"] == "modified"
        assert first_element["proof"]["flags"] == ["low_confidence"]
        assert first_element["payload"]["lines"][0]["ocr_text"] == "OCR旧文"
        table_element = xml_projection["pages"][0]["elements"][1]
        assert table_element["fallback"]["mode"] == "image_fallback"
        figure_element = xml_projection["pages"][0]["elements"][2]
        assert figure_element["proof"]["confidence"] == 0.0
        assert xml_projection["assets"][0]["kind"] == "table_crop"
        assert any(item["code"] == "page_error" for item in xml_projection["diagnostics"])
        assert any(item["code"] == "table_fallback_to_image" for item in xml_projection["diagnostics"])

    print("test_xml_authority_and_json_mirror_archive_parity PASSED")


def test_ir_based_exporters_and_pdf_profiles():
    import json
    import os
    import tempfile

    from app.export import get_exporter
    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.export_service import build_export_path

    bb = BBox(0, 0, 100, 20)
    project = OcrProject(name="IRExport", pages=[
        Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
            Block(block_type=BlockType.TEXT, bbox=bb, order=0, lines=[
                Line(text="正文", confidence=0.9, bbox=bb),
            ]),
            Block(block_type=BlockType.TABLE, bbox=bb, order=1, lines=[
                Line(text="表格", confidence=0.8, bbox=bb),
            ]),
        ])
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = build_export_path(tmpdir, project.name, "json")
        txt_path = build_export_path(tmpdir, project.name, "txt")
        xml_path = build_export_path(tmpdir, project.name, "xml")
        pdf_single_path = build_export_path(tmpdir, project.name, "pdf-single")
        pdf_dual_path = build_export_path(tmpdir, project.name, "pdf-dual")

        get_exporter("json").export(project, str(json_path))
        get_exporter("txt").export(project, str(txt_path))
        get_exporter("xml").export(project, str(xml_path))
        get_exporter("pdf-single").export(project, str(pdf_single_path))
        get_exporter("pdf-dual").export(project, str(pdf_dual_path))

        data = json.loads(open(json_path, encoding="utf-8").read())
        txt = open(txt_path, encoding="utf-8").read()
        assert data["version"] == "export_ir.v1"
        assert data["profile"]["format"] == "json"
        assert data["pages"][0]["elements"][1]["fallback"]["mode"] == "image_fallback"
        assert "正文" in txt
        assert "bbox=" not in txt
        assert "=== 第 1 页 ===" in txt
        assert "[正文" not in txt
        assert os.path.exists(os.path.join(tmpdir, "IRExport.gbk.txt"))
        assert "<Element" in open(xml_path, encoding="utf-8").read()
        assert os.path.getsize(pdf_single_path) > 0
        assert os.path.getsize(pdf_dual_path) > 0
        assert pdf_single_path.name == "IRExport.pdf-single.pdf"
        assert pdf_dual_path.name == "IRExport.pdf-dual.pdf"

    assert isinstance(get_exporter("pdf"), PdfExporter)
    assert get_exporter("pdf").profile == "pdf-single"
    assert get_exporter("pdf-dual").profile == "pdf-dual"

    print("test_ir_based_exporters_and_pdf_profiles PASSED")


def test_export_dialog_offers_markdown():
    from app.models import OcrProject
    from app.ui.export.export_dialog import ExportDialog

    _get_qapp()
    dialog = ExportDialog(OcrProject(name="DialogExport"))
    try:
        assert "md" in dialog._checkboxes
        assert "json" in dialog._checkboxes
        assert "pdf-single" in dialog._checkboxes
        assert "pdf-dual" in dialog._checkboxes
        assert "pdf" not in dialog._checkboxes
        assert dialog._checkboxes["pdf-single"].text() == "PDF 原图单层 (.pdf)"
        assert dialog._checkboxes["pdf-dual"].text() == "PDF 原图+可搜索文本 (.pdf)"
        assert dialog._checkboxes["md"].isChecked()
        assert dialog._checkboxes["json"].isChecked()
    finally:
        dialog.close()

    print("test_export_dialog_offers_markdown PASSED")


def test_export_default_styles_map_to_html_docx_and_pdf():
    from docx import Document

    from app.export.docx_exporter import DocxExporter
    from app.export.html import HtmlExporter
    from app.export.pdf import PdfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    title_bb = BBox(0, 0, 100, 20)
    body_bb = BBox(0, 30, 100, 20)
    equation_bb = BBox(0, 60, 100, 20)
    project = OcrProject(name="StyleExport", pages=[
        Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
            Block(block_type=BlockType.TITLE, bbox=title_bb, order=0, lines=[
                Line(text="章节标题", confidence=0.95, bbox=title_bb),
            ]),
            Block(block_type=BlockType.TEXT, bbox=body_bb, order=1, lines=[
                Line(text="正文内容", confidence=0.90, bbox=body_bb),
            ]),
            Block(block_type=BlockType.EQUATION, bbox=equation_bb, order=2, lines=[
                Line(text="E = mc^2", confidence=0.88, bbox=equation_bb),
            ]),
        ]),
    ])

    paths = []
    try:
        for suffix in (".html", ".docx", ".pdf"):
            f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            paths.append(f.name)
            f.close()
        html_path, docx_path, pdf_path = paths

        HtmlExporter().export(project, html_path)
        html = open(html_path, encoding="utf-8").read()
        assert 'class="block block-title"' in html
        assert 'class="block block-body"' in html
        assert 'class="block block-equation"' in html

        DocxExporter().export(project, docx_path)
        doc = Document(docx_path)
        styled = {p.text.strip(): p.style.name for p in doc.paragraphs if p.text.strip()}
        assert styled["章节标题"].startswith("Heading")
        assert styled["正文内容"] == "Normal"

        PdfExporter().export(project, pdf_path)
        assert os.path.getsize(pdf_path) > 0
        print("test_export_default_styles_map_to_html_docx_and_pdf PASSED")
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)


def test_rich_reflow_contract_shared_by_html_docx_and_rtf():
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    from app.export.docx_exporter import DocxExporter
    from app.export.html import HtmlExporter
    from app.export.ir_builder import build_export_ir
    from app.export.rendering import iter_rich_reflow_blocks
    from app.export.rtf import RtfExporter
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    page = Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
        Block(block_type=BlockType.TITLE, bbox=BBox(0, 0, 100, 20), order=0, lines=[
            Line(text="Chapter", confidence=0.95, bbox=BBox(0, 0, 100, 20)),
        ]),
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 100, 40), order=1, lines=[
            Line(text="Hello", confidence=0.90, bbox=BBox(0, 30, 100, 20)),
            Line(text="world", confidence=0.90, bbox=BBox(0, 50, 100, 20)),
        ]),
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 75, 100, 40), order=2, lines=[
            Line(text="中文", confidence=0.90, bbox=BBox(0, 75, 100, 20)),
            Line(text="正文", confidence=0.90, bbox=BBox(0, 95, 100, 20)),
        ]),
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 125, 100, 40), order=3, lines=[
            Line(text="Mixed中文", confidence=0.90, bbox=BBox(0, 125, 100, 20)),
            Line(text="and English", confidence=0.90, bbox=BBox(0, 145, 100, 20)),
        ]),
        Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(0, 175, 100, 20), order=4, lines=[
            Line(text="Figure cap", confidence=0.90, bbox=BBox(0, 80, 100, 20)),
        ]),
        Block(block_type=BlockType.REFERENCE, bbox=BBox(0, 205, 100, 40), order=5, lines=[
            Line(text="Ref one", confidence=0.90, bbox=BBox(0, 110, 100, 20)),
            Line(text="Ref two", confidence=0.90, bbox=BBox(0, 130, 100, 20)),
        ]),
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 255, 100, 20), order=6, lines=[
            Line(text="E=mc^2", confidence=0.90, bbox=BBox(0, 160, 100, 20)),
        ]),
    ])
    project = OcrProject(name="RichReflow", pages=[page])
    document = build_export_ir(project, "html")
    blocks = list(iter_rich_reflow_blocks(document))
    assert [(block.role, block.lines) for block in blocks] == [
        ("heading", ["Chapter"]),
        ("body", ["Hello world"]),
        ("body", ["中文正文"]),
        ("body", ["Mixed中文 and English"]),
        ("caption", ["Figure cap"]),
        ("reference", ["Ref one", "Ref two"]),
        ("equation", ["E=mc^2"]),
    ]

    paths = []
    try:
        for suffix in (".html", ".docx", ".rtf"):
            f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            paths.append(f.name)
            f.close()
        html_path, docx_path, rtf_path = paths
        HtmlExporter().export(project, html_path)
        DocxExporter().export(project, docx_path)
        RtfExporter().export(project, rtf_path)

        html = open(html_path, encoding="utf-8").read()
        assert 'data-role="body"' in html
        assert 'class="block block-title"' in html
        assert "Helloworld" not in html
        assert "Hello world" in html
        assert "中文 正文" not in html
        assert "中文正文" in html
        assert "Mixed中文and English" not in html
        assert "Mixed中文 and English" in html
        assert html.index("Chapter") < html.index("Hello world") < html.index("中文正文") < html.index("Mixed中文 and English") < html.index("Figure cap")
        assert html.index("Ref one") < html.index("Ref two") < html.index("E=mc^2")

        doc = Document(docx_path)
        paragraphs = [p for p in doc.paragraphs if p.text.strip()]
        texts = [p.text.strip() for p in paragraphs]
        assert "Hello world" in texts
        assert "中文正文" in texts
        assert "Mixed中文 and English" in texts
        assert "Helloworld" not in texts
        assert "Mixed中文and English" not in texts
        assert texts.index("Chapter") < texts.index("Hello world") < texts.index("中文正文") < texts.index("Mixed中文 and English") < texts.index("Figure cap")
        assert paragraphs[texts.index("Chapter")].style.name.startswith("Heading")
        assert paragraphs[texts.index("Figure cap")].style.name == "Caption"
        assert paragraphs[texts.index("E=mc^2")].alignment == WD_ALIGN_PARAGRAPH.CENTER

        rtf = open(rtf_path, encoding="ascii").read()
        assert r"\pard\sb120\b\fs32 Chapter\b0\fs24\par" in rtf
        assert r"\pard Hello world\par" in rtf
        assert r"\pard Helloworld\par" not in rtf
        assert r"\pard Mixed\u20013?\u25991? and English\par" in rtf
        assert r"\pard\i\fs20 Figure cap\i0\fs24\par" in rtf
        assert r"\pard\fs22 Ref one\fs24\par" in rtf
        assert r"\pard\qc E=mc^2\par" in rtf
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)

    print("test_rich_reflow_contract_shared_by_html_docx_and_rtf PASSED")


def test_export_worker_reports_completion_progress():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.ui.export.export_dialog import ExportWorker

    _get_qapp()
    bb = BBox(0, 0, 100, 20)
    project = OcrProject(name="WorkerExport", pages=[
        Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
            Block(block_type=BlockType.TEXT, bbox=bb, lines=[
                Line(text="导出内容", confidence=0.9, bbox=bb),
            ]),
        ]),
    ])
    events = []
    completed = []
    with tempfile.TemporaryDirectory() as tmpdir:
        worker = ExportWorker(project, ["txt"], tmpdir)
        worker.progress.connect(lambda msg, current, total: events.append((msg, current, total)))
        worker.completed.connect(completed.append)
        worker.run()
        assert os.path.exists(os.path.join(tmpdir, "WorkerExport.utf8.txt"))
        assert os.path.exists(os.path.join(tmpdir, "WorkerExport.gbk.txt"))
    assert events[-1] == ("TXT 导出完成", 1, 1)
    assert len(completed) == 1
    assert completed[0].all_ok is True
    assert completed[0].successes[0].fmt == "txt"

    print("test_export_worker_reports_completion_progress PASSED")


def test_export_filename_sanitizes_invalid_project_name():
    from app.services.export_service import build_export_path, sanitize_export_filename

    assert sanitize_export_filename(' 卷/一:测试*? ') == "卷_一_测试"
    assert sanitize_export_filename("CON") == "CON_"
    assert sanitize_export_filename("CON.txt") == "CON.txt_"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = build_export_path(tmpdir, '卷/一:测试*?', "txt")
        assert out_path.parent.exists()
        assert out_path.name == "卷_一_测试.utf8.txt"

    print("test_export_filename_sanitizes_invalid_project_name PASSED")


def test_export_worker_sanitizes_project_name_for_all_formats():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.export_service import build_export_path
    from app.ui.export.export_dialog import ExportWorker

    _get_qapp()
    bb = BBox(0, 0, 100, 20)
    project = OcrProject(name='卷/一:测试*?', pages=[
        Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
            Block(block_type=BlockType.TITLE, bbox=bb, lines=[
                Line(text="标题", confidence=0.9, bbox=bb),
            ]),
            Block(block_type=BlockType.TEXT, bbox=bb, lines=[
                Line(text="正文", confidence=0.9, bbox=bb),
            ]),
        ]),
    ])
    completed = []
    formats = ["txt", "json", "md", "rtf", "pdf", "pdf-single", "pdf-dual", "xml", "html", "docx"]
    with tempfile.TemporaryDirectory() as tmpdir:
        worker = ExportWorker(project, formats, tmpdir)
        worker.completed.connect(completed.append)
        worker.run()
        for fmt in formats:
            assert os.path.exists(build_export_path(tmpdir, project.name, fmt))
        assert os.path.exists(os.path.join(tmpdir, "卷_一_测试.gbk.txt"))

    assert len(completed) == 1
    assert completed[0].all_ok is True
    assert {result.fmt for result in completed[0].successes} == set(formats)

    print("test_export_worker_sanitizes_project_name_for_all_formats PASSED")


def test_export_worker_keeps_formats_independent_when_one_fails():
    import app.export as export_module
    from app.export.base import ExporterBase
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.export_service import build_export_path, sanitize_export_filename
    from app.ui.export.export_dialog import ExportWorker

    class WritingExporter(ExporterBase):
        def __init__(self, label: str):
            self._label = label

        def export(self, project, out_path: str) -> None:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self._label)

    class FailingExporter(ExporterBase):
        def export(self, project, out_path: str) -> None:
            raise RuntimeError("pdf failed")

    original_get_exporter = export_module.get_exporter
    export_module.get_exporter = lambda fmt: FailingExporter() if fmt == "pdf" else WritingExporter(fmt)
    try:
        bb = BBox(0, 0, 100, 20)
        project = OcrProject(name="IndependentExport", pages=[
            Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
                Block(block_type=BlockType.TEXT, bbox=bb, lines=[
                    Line(text="导出内容", confidence=0.9, bbox=bb),
                ]),
            ]),
        ])
        completed = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ExportWorker(project, ["txt", "pdf", "md"], tmpdir)
            worker.completed.connect(completed.append)
            worker.run()
            base = sanitize_export_filename(project.name)
            assert open(build_export_path(tmpdir, project.name, "txt"), encoding="utf-8").read() == "txt"
            assert open(os.path.join(tmpdir, f"{base}.md"), encoding="utf-8").read() == "md"
            assert not os.path.exists(os.path.join(tmpdir, f"{base}.pdf"))

        result = completed[0]
        assert result.any_success is True
        assert result.all_ok is False
        assert [item.fmt for item in result.successes] == ["txt", "md"]
        assert [item.fmt for item in result.failures] == ["pdf"]
        assert "pdf failed" in result.summary()
    finally:
        export_module.get_exporter = original_get_exporter

    print("test_export_worker_keeps_formats_independent_when_one_fails PASSED")


def test_export_dialog_reports_partial_success_without_critical_error():
    from PySide6.QtWidgets import QMessageBox

    from app.ui.export.export_dialog import ExportDialog, ExportFormatResult, ExportRunResult
    from app.models import OcrProject

    _get_qapp()
    warnings = []
    criticals = []
    original_warning = QMessageBox.warning
    original_critical = QMessageBox.critical
    QMessageBox.warning = lambda *args, **kwargs: warnings.append(args)
    QMessageBox.critical = lambda *args, **kwargs: criticals.append(args)
    dialog = ExportDialog(OcrProject(name="PartialDialog"))
    try:
        dialog._progress_bar.setVisible(True)
        dialog._progress_bar.setRange(0, 2)
        result = ExportRunResult([
            ExportFormatResult("txt", "/tmp/out.txt", True),
            ExportFormatResult("pdf", "/tmp/out.pdf", False, "pdf failed"),
        ])
        dialog._on_finished(result)

        assert dialog._progress_lbl.text() == "部分导出完成"
        assert warnings and "部分导出完成" in warnings[0][1]
        assert criticals == []
    finally:
        QMessageBox.warning = original_warning
        QMessageBox.critical = original_critical
        dialog.close()

    print("test_export_dialog_reports_partial_success_without_critical_error PASSED")


def test_export_dialog_surfaces_output_path_failure_from_real_worker():
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QMessageBox

    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.ui.export.export_dialog import ExportDialog

    app = _get_qapp()
    warnings = []
    criticals = []
    original_warning = QMessageBox.warning
    original_critical = QMessageBox.critical
    QMessageBox.warning = lambda *args, **kwargs: warnings.append(args)
    QMessageBox.critical = lambda *args, **kwargs: criticals.append(args)
    bb = BBox(0, 0, 100, 20)
    project = OcrProject(name="PathFailureExport", pages=[
        Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[
            Block(block_type=BlockType.TEXT, bbox=bb, lines=[
                Line(text="导出内容", confidence=0.9, bbox=bb),
            ]),
        ]),
    ])
    dialog = ExportDialog(project)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            blocked_path = os.path.join(tmpdir, "not_a_directory")
            with open(blocked_path, "w", encoding="utf-8") as f:
                f.write("blocks directory creation")
            dialog._dir_edit.setText(blocked_path)
            dialog._start_export()
            worker = dialog._worker
            assert worker is not None
            if not worker.isFinished():
                loop = QEventLoop()
                worker.finished.connect(loop.quit)
                QTimer.singleShot(5000, loop.quit)
                loop.exec()
            app.processEvents()

            assert dialog._progress_lbl.text() == "导出失败"
            assert dialog._btn_start is not None and dialog._btn_start.isEnabled()
            assert warnings == []
            assert criticals and "导出失败" in criticals[0][1]
            assert "not_a_directory" in criticals[0][2]
    finally:
        QMessageBox.warning = original_warning
        QMessageBox.critical = original_critical
        dialog.close()

    print("test_export_dialog_surfaces_output_path_failure_from_real_worker PASSED")


def test_layout_panel_analysis_progress_lifecycle():
    from app.models import Page
    from app.ui.recognize.layout_panel import LayoutPanel

    _get_qapp()
    panel = LayoutPanel()
    try:
        panel.set_pages([Page(image_path="/tmp/img.jpg", width=100, height=100)])
        panel.start_analysis_progress(2)
        assert not panel._progress_bar.isHidden()
        assert panel._progress_bar.maximum() == 2
        panel.update_analysis_progress(0, 2)
        assert panel._progress_bar.value() == 1
        assert panel._progress_bar.format() == "%p%"
        assert panel._status_lbl.text() == "版面分析中…"
        panel.finish_analysis_progress("完成")
        assert panel._progress_bar.isHidden()
        assert panel._status_lbl.text() == "完成"
    finally:
        panel.close()

    print("test_layout_panel_analysis_progress_lifecycle PASSED")


def test_layout_panel_workbench_height_is_not_forced_by_sidebar():
    from app.ui.recognize.layout_panel import LayoutPanel

    _get_qapp()
    panel = LayoutPanel()
    try:
        assert panel.minimumSizeHint().height() <= 360
    finally:
        panel.close()

    print("test_layout_panel_workbench_height_is_not_forced_by_sidebar PASSED")


def test_layout_panel_splitter_keeps_sidebar_width_on_large_workbench():
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    panel = LayoutPanel()
    try:
        panel.resize(1536, 864)
        panel.show()
        app.processEvents()

        sizes = panel._splitter.sizes()
        assert sizes[2] >= 280
        assert sizes[1] > sizes[2]
    finally:
        panel.close()

    print("test_layout_panel_splitter_keeps_sidebar_width_on_large_workbench PASSED")


def test_layout_panel_right_sidebar_uses_project_stats_without_selection_inspector():
    from pathlib import Path
    import tempfile

    from PySide6.QtCore import Qt
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_a = Path(tmpdir) / "page-a.png"
        image_b = Path(tmpdir) / "page-b.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_a))
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_b))
        drifted_formula = Block(block_type=BlockType.TEXT, bbox=BBox(40, 10, 20, 10))
        pages = [
            Page(
                image_path=str(image_a),
                width=120,
                height=80,
                blocks=[
                    Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 20, 10)),
                    drifted_formula,
                ],
            ),
            Page(
                image_path=str(image_b),
                width=120,
                height=80,
                blocks=[Block(block_type=BlockType.TABLE, bbox=BBox(10, 10, 20, 10))],
            ),
        ]
        set_layout_snapshot_for_page(
            pages[0],
            LayoutSnapshot(
                page_uid=pages[0].uid,
                artifact_uid="artifact-1",
                source_engine="paddleocr-vl",
                source_run_id="layout-run-1",
                blocks=(
                    LayoutBlockSnapshot(
                        block_type=BlockType.TEXT,
                        bbox=pages[0].blocks[0].bbox,
                        order=0,
                        source_label="text",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                        ocr_policy=OcrPolicy.TEXT_OCR,
                        uid=pages[0].blocks[0].uid,
                    ),
                    LayoutBlockSnapshot(
                        block_type=BlockType.EQUATION,
                        bbox=drifted_formula.bbox,
                        order=1,
                        source_label="inline_formula",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="inline_formula"),
                        ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                        uid=drifted_formula.uid,
                    ),
                ),
            ),
        )
        panel = LayoutPanel()
        try:
            panel.set_pages(pages)
            app.processEvents()

            assert not hasattr(panel, "_inspector")
            assert not hasattr(panel, "_btn_merge")
            assert not hasattr(panel, "_btn_lock")
            stats = panel._project_stats_lbl.text()
            assert "页面：2/2 已分析" in stats
            assert "框：3 个" in stats
            assert "正文 1" in stats
            assert "公式 1" in stats
            assert "表格 1" in stats
        finally:
            panel.close()

    print("test_layout_panel_right_sidebar_uses_project_stats_without_selection_inspector PASSED")


def test_layout_panel_show_page_layers_uses_layout_snapshot_view_geometry():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 120, QImage.Format.Format_RGB888).save(str(image_path))
        block = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(90, 80, 140, 100), source_label="text")
        snapshot_bbox = BBox.from_xyxy(10, 20, 60, 40)
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[block])
        set_layout_snapshot_for_page(
            page,
            LayoutSnapshot(
                page_uid=page.uid,
                artifact_uid="artifact-1",
                source_engine="paddleocr-vl",
                source_run_id="layout-run-1",
                blocks=(
                    LayoutBlockSnapshot(
                        block_type=BlockType.TITLE,
                        bbox=snapshot_bbox,
                        order=0,
                        source_label="heading_1",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="heading_1"),
                        ocr_policy=OcrPolicy.TEXT_OCR,
                        uid=block.uid,
                    ),
                ),
            ),
        )

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            assert len(panel._viewer._block_items) == 1
            item, item_block = panel._viewer._block_items[0]
            assert item_block is block
            assert int(item.pos().x()) == snapshot_bbox.x
            assert int(item.pos().y()) == snapshot_bbox.y
            assert int(item.rect().width()) == snapshot_bbox.w
            assert int(item.rect().height()) == snapshot_bbox.h
        finally:
            panel.close()

    print("test_layout_panel_show_page_layers_uses_layout_snapshot_view_geometry PASSED")


def test_layout_panel_auto_text_blocks_are_editable_frames():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QGraphicsItem

    from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 40, 10),
            source=BlockSource.AUTO_LAYOUT,
        )
        formula_block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(60, 10, 20, 10),
            source=BlockSource.AUTO_LAYOUT,
        )
        page = Page(image_path=str(image_path), width=120, height=80, blocks=[text_block, formula_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            text_item = next(item for item, block in panel._viewer._block_items if block is text_block)
            formula_item = next(item for item, block in panel._viewer._block_items if block is formula_block)
            assert bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            assert bool(formula_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert bool(formula_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

            panel._on_block_clicked(text_block)
            assert panel._selected_block is text_block
        finally:
            panel.close()

    print("test_layout_panel_auto_text_blocks_are_editable_frames PASSED")


def test_layout_panel_pageup_pagedown_shortcuts_change_page():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_a = Path(tmpdir) / "page-a.png"
        image_b = Path(tmpdir) / "page-b.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_a))
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_b))
        pages = [
            Page(image_path=str(image_a), width=120, height=80, page_number=1),
            Page(image_path=str(image_b), width=120, height=80, page_number=2),
        ]

        panel = LayoutPanel()
        try:
            panel.set_pages(pages)
            app.processEvents()
            assert panel._current_page_idx == 0

            panel._page_down_shortcut.activated.emit()
            assert panel._current_page_idx == 1
            assert panel._lbl_page_no.text() == "2 / 2"

            panel._page_up_shortcut.activated.emit()
            assert panel._current_page_idx == 0
            assert panel._lbl_page_no.text() == "1 / 2"
        finally:
            panel.close()

    print("test_layout_panel_pageup_pagedown_shortcuts_change_page PASSED")


def test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page
    from app.models.page_state import invalidate_page_ocr
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 80, QImage.Format.Format_RGB888).save(str(image_path))
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox.from_xyxy(0, 0, 120, 40),
            source=BlockSource.AUTO_LAYOUT,
            lines=[Line(text="甲$ A $乙", confidence=0.9, bbox=BBox.from_xyxy(0, 0, 120, 40))],
            ocr_audit={
                "schema": "hanwang_bbox_audit.v1",
                "layout_block_bbox": [0, 0, 120, 40],
                "effective_block_bbox": [0, 0, 120, 40],
                "effective_block_bbox_source": "layout_line_routes_union",
                "layout_line_route_bboxes": [[0, 0, 120, 40]],
                "route_text_slice_bboxes": [[0, 0, 40, 40], [70, 0, 120, 40]],
                "hanwang_recog_group_bboxes": [[0, 2, 40, 38], [70, 2, 120, 38]],
                "hanwang_segimg_group_clipped_count": 1,
                "hanwang_segimg_group_dropped_count": 0,
                "route_text_slice_count": 2,
                "hanwang_recog_group_count": 2,
            },
        )
        page = Page(image_path=str(image_path), width=160, height=80, blocks=[text_block])
        skip_block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox.from_xyxy(10, 50, 80, 70),
            source=BlockSource.AUTO_LAYOUT,
            ocr_audit={
                "schema": "hanwang_bbox_audit.v1",
                "layout_block_bbox": [10, 50, 80, 70],
                "effective_block_bbox": [10, 50, 80, 70],
                "effective_block_bbox_source": "layout_block_bbox",
                "route_text_slice_count": 0,
                "hanwang_recog_group_count": 0,
            },
        )
        page.blocks.append(skip_block)
        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            assert not hasattr(panel, "_btn_hanwang_audit")
            assert len(panel._viewer._readonly_overlay_items) == 0
            panel._refresh_current_page_layers()
            assert len(panel._viewer._readonly_overlay_items) == 0

            assert not hasattr(panel, "_inspector")

            text_block.ocr_invalidated_reason = "block_moved"
            invalidate_page_ocr(page, "block_moved")
            panel._refresh_current_page_layers()
            assert len(panel._viewer._readonly_overlay_items) == 0
        finally:
            panel.close()

    print("test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle PASSED")


def test_layout_panel_draw_merge_uses_large_box_and_removes_overlap():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Line, PaddleBinding, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(140, 90, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        page = Page(image_path=str(image_path), width=140, height=90)
        page.blocks = [
            Block(block_type=BlockType.EQUATION, bbox=BBox(20, 20, 20, 10), lines=[
                Line(text="x", confidence=0.9, bbox=BBox(20, 20, 20, 10)),
            ], order=0),
            Block(block_type=BlockType.EQUATION, bbox=BBox(60, 20, 20, 10), lines=[
                Line(text="(1)", confidence=0.9, bbox=BBox(60, 20, 20, 10)),
            ], order=1),
        ]

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_type_buttons[BlockType.EQUATION].click()

            panel._on_block_created(BBox(10, 10, 90, 30))

            assert len(page.blocks) == 1
            assert page.blocks[0].bbox == BBox(10, 10, 90, 30)
            assert page.blocks[0].block_type == BlockType.EQUATION
            assert page.blocks[0].lines == []
            assert page.blocks[0].source == BlockSource.USER_EDITED
            assert page.blocks[0].ocr_invalidated_reason == "manual_draw_merge"
        finally:
            panel.close()

    print("test_layout_panel_draw_merge_uses_large_box_and_removes_overlap PASSED")


def test_layout_panel_bbox_hit_and_formula_label_use_layout_snapshot_view():
    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.layout_panel import LayoutPanel

    _get_qapp()
    text = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(120, 60, 150, 80), source_label="text")
    snapshot_bbox = BBox.from_xyxy(10, 10, 100, 40)
    page = Page(image_path="", width=160, height=100, blocks=[text])
    set_layout_snapshot_for_page(
        page,
        LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-1",
            source_engine="paddleocr-vl",
            source_run_id="layout-run-1",
            blocks=(
                LayoutBlockSnapshot(
                    block_type=BlockType.TEXT,
                    bbox=snapshot_bbox,
                    order=0,
                    source_label="text",
                    origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                    uid=text.uid,
                ),
            ),
        ),
    )
    panel = LayoutPanel()
    try:
        assert panel._blocks_intersecting_bbox(page, BBox.from_xyxy(95, 12, 105, 30)) == [text]
        assert panel._infer_formula_source_label_for_bbox(page, BBox.from_xyxy(30, 12, 40, 24)) == "inline_formula"
    finally:
        panel.close()

    print("test_layout_panel_bbox_hit_and_formula_label_use_layout_snapshot_view PASSED")


def test_layout_panel_draw_inside_text_frame_does_not_merge_parent():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(20, 20, 100, 60),
            lines=[Line(
                text="abc",
                confidence=0.9,
                bbox=BBox(20, 20, 100, 60),
                chars=[Char(char="a", confidence=0.9, bbox=BBox(20, 20, 18, 10))],
            )],
            source=BlockSource.AUTO_LAYOUT,
            order=0,
        )
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[text_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_type_buttons[BlockType.EQUATION].click()

            panel._on_block_created(BBox(55, 45, 20, 12))

            assert len(page.blocks) == 2
            assert page.blocks[0] is text_block
            assert page.blocks[1].block_type == BlockType.EQUATION
            assert page.blocks[1].source_label == "inline_formula"
            assert page.blocks[1].bbox == BBox(55, 45, 20, 12)
        finally:
            panel.close()

    print("test_layout_panel_draw_inside_text_frame_does_not_merge_parent PASSED")


def test_layout_panel_formula_draw_touching_text_frame_stays_separate():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Line, PaddleBinding, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(20, 20, 100, 60),
            lines=[Line(text="abc", confidence=0.9, bbox=BBox(20, 20, 100, 60))],
            source=BlockSource.AUTO_LAYOUT,
            order=0,
        )
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[text_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_type_buttons[BlockType.EQUATION].click()

            panel._on_block_created(BBox(55, 22, 20, 12))

            assert len(page.blocks) == 2
            assert page.blocks[0] is text_block
            assert page.blocks[0].lines
            formula = page.blocks[1]
            assert formula.block_type == BlockType.EQUATION
            assert formula.source_label == "inline_formula"
            assert formula.source == BlockSource.MANUAL_DRAW
            assert formula.note != "manual_draw_merge_requires_ocr_rerun"
        finally:
            panel.close()

    print("test_layout_panel_formula_draw_touching_text_frame_stays_separate PASSED")


def test_layout_panel_text_draw_does_not_absorb_inline_formula_block():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        formula_block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(50, 40, 20, 12),
            source=BlockSource.AUTO_LAYOUT,
            source_label="inline_formula",
            order=0,
        )
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[formula_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_type_buttons[BlockType.TEXT].click()

            panel._on_block_created(BBox(45, 36, 40, 20))

            assert len(page.blocks) == 2
            assert page.blocks[0] is formula_block
            assert page.blocks[0].block_type == BlockType.EQUATION
            assert page.blocks[0].source_label == "inline_formula"
            text_block = page.blocks[1]
            assert text_block.block_type == BlockType.TEXT
            assert text_block.source == BlockSource.MANUAL_DRAW
            assert text_block.note != "manual_draw_merge_requires_ocr_rerun"
        finally:
            panel.close()

    print("test_layout_panel_text_draw_does_not_absorb_inline_formula_block PASSED")


def test_layout_panel_drawn_block_is_selected_and_type_editable():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(140, 90, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(image_path=str(image_path), width=140, height=90)

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            panel._on_block_created(BBox(25, 20, 30, 18))

            assert len(page.blocks) == 1
            block = page.blocks[0]
            assert panel._selected_block is block
            assert panel._selected_type_buttons[BlockType.TABLE].isEnabled()
            assert any(item.isSelected() and item_block is block for item, item_block in panel._viewer._block_items)

            panel._selected_type_buttons[BlockType.TABLE].click()

            assert block.block_type == BlockType.TABLE
            assert block.source_label == "table"
        finally:
            panel.close()

    print("test_layout_panel_drawn_block_is_selected_and_type_editable PASSED")


def test_layout_panel_draw_snaps_to_image_ink_without_existing_blocks():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QColor, QImage, QPainter

    from app.models import BBox, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(140, 90, QImage.Format.Format_RGB888)
        image.fill(QColor("white"))
        painter = QPainter(image)
        painter.fillRect(20, 20, 20, 10, QColor("black"))
        painter.fillRect(44, 20, 8, 10, QColor("black"))
        painter.end()
        image.save(str(image_path))
        page = Page(image_path=str(image_path), width=140, height=90)

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_type_buttons[BlockType.EQUATION].click()

            panel._on_block_created(BBox(17, 19, 25, 13))

            assert len(page.blocks) == 1
            assert page.blocks[0].bbox == BBox(20, 20, 20, 10)
            assert page.blocks[0].block_type == BlockType.EQUATION
            assert panel._selected_block is page.blocks[0]
            assert any(item.isSelected() and item_block is page.blocks[0] for item, item_block in panel._viewer._block_items)
        finally:
            panel.close()

    print("test_layout_panel_draw_snaps_to_image_ink_without_existing_blocks PASSED")


def test_layout_panel_ink_snap_reuses_cached_image_mask():
    from pathlib import Path
    import tempfile

    import cv2
    from PySide6.QtGui import QColor, QImage, QPainter

    from app.models import BBox, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(140, 90, QImage.Format.Format_RGB888)
        image.fill(QColor("white"))
        painter = QPainter(image)
        painter.fillRect(20, 20, 20, 10, QColor("black"))
        painter.end()
        image.save(str(image_path))
        page = Page(image_path=str(image_path), width=140, height=90)
        panel = LayoutPanel()
        calls = []
        original_imread = cv2.imread

        def fake_imread(path, flags):
            calls.append((path, flags))
            return original_imread(path, flags)

        cv2.imread = fake_imread
        try:
            first = panel._snap_drawn_bbox(page, BBox(17, 19, 25, 13))
            second = panel._snap_drawn_bbox(page, BBox(18, 18, 25, 13))
        finally:
            cv2.imread = original_imread
            panel.close()

        assert first == BBox(20, 20, 20, 10)
        assert second == BBox(20, 20, 20, 10)
        assert len(calls) == 1

    print("test_layout_panel_ink_snap_reuses_cached_image_mask PASSED")


def test_layout_panel_delete_selected_removes_unlocked_box():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        formula_block = Block(block_type=BlockType.EQUATION, bbox=BBox(10, 10, 20, 20))
        page = Page(image_path=str(image_path), width=120, height=80, blocks=[formula_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            item, block = panel._viewer._block_items[0]
            assert block is formula_block
            item.setSelected(True)

            panel._delete_selected()

            assert page.blocks == []
            assert panel._btn_undo.isEnabled()
        finally:
            panel.close()

    print("test_layout_panel_delete_selected_removes_unlocked_box PASSED")


def test_layout_panel_readonly_char_boxes_do_not_block_formula_delete():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QGraphicsItem

    from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 100, QImage.Format.Format_RGB888).save(str(image_path))
        line = Line(
            text="A $x$ B",
            confidence=0.9,
            bbox=BBox(10, 10, 100, 20),
            chars=[
                Char(char="x", confidence=0.9, bbox=BBox(45, 12, 10, 12)),
            ],
        )
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 100, 20),
            lines=[line],
            source=BlockSource.AUTO_LAYOUT,
            order=0,
        )
        formula_block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(42, 10, 18, 18),
            source=BlockSource.MANUAL_DRAW,
            order=1,
        )
        page = Page(image_path=str(image_path), width=160, height=100, blocks=[text_block, formula_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            assert len(panel._viewer._char_items) == 1
            char_item, _ = panel._viewer._char_items[0]
            formula_item = next(item for item, block in panel._viewer._block_items if block is formula_block)
            assert not bool(char_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            assert not bool(char_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert char_item.zValue() > formula_item.zValue()

            formula_item.setSelected(True)
            panel._delete_selected()

            assert page.blocks == [text_block]
        finally:
            panel.close()

    print("test_layout_panel_readonly_char_boxes_do_not_block_formula_delete PASSED")


def test_layout_panel_hides_empty_and_invalidated_char_boxes():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.models.page_state import invalidate_page_ocr
    from app.ui.recognize.layout_panel import LayoutPanel

    _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 100, QImage.Format.Format_RGB888).save(str(image_path))
        line = Line(
            text="甲",
            confidence=0.9,
            bbox=BBox(10, 10, 60, 20),
            chars=[
                Char(char="甲", confidence=0.9, bbox=BBox(10, 10, 20, 20)),
                Char(char="", confidence=0.0, bbox=BBox(40, 10, 20, 20)),
            ],
        )
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 60, 20),
            lines=[line],
        )
        page = Page(image_path=str(image_path), width=160, height=100, blocks=[block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            assert len(panel._viewer._char_items) == 1

            block.ocr_invalidated_reason = "block_moved"
            panel._refresh_current_page_layers()
            assert panel._viewer._char_items == []

            block.ocr_invalidated_reason = ""
            invalidate_page_ocr(page, "block_moved")
            panel._refresh_current_page_layers()
            assert panel._viewer._char_items == []
        finally:
            panel.close()

    print("test_layout_panel_hides_empty_and_invalidated_char_boxes PASSED")


def test_layout_panel_excludes_inline_formula_carriers_from_char_boxes():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    line = Line(
        text="甲$ A $乙",
        confidence=0.9,
        bbox=BBox.from_xyxy(0, 0, 120, 30),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox.from_xyxy(0, 0, 20, 30), bbox_source="hanwang:micro_recblock"),
            Char(
                char="$ A $",
                confidence=0.0,
                bbox=BBox.from_xyxy(30, 0, 70, 30),
                bbox_source="paddle_inline_formula",
                bbox_granularity="word",
                token_text="$ A $",
            ),
            Char(char="乙", confidence=0.9, bbox=BBox.from_xyxy(80, 0, 100, 30), bbox_source="hanwang:micro_recblock"),
        ],
    )
    page = Page(
        image_path="",
        width=120,
        height=30,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(0, 0, 120, 30), lines=[line])],
    )

    chars = LayoutPanel._collect_page_chars(page)

    assert [char.char for char in chars] == ["甲", "乙"]
    assert all(char.bbox_source != "paddle_inline_formula" for char in chars)

    print("test_layout_panel_excludes_inline_formula_carriers_from_char_boxes PASSED")


def test_layout_panel_type_buttons_change_unlocked_block_type():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        formula_block = Block(block_type=BlockType.EQUATION, bbox=BBox(10, 10, 20, 20))
        page = Page(image_path=str(image_path), width=120, height=80, blocks=[formula_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._on_block_clicked(formula_block)
            panel._selected_type_buttons[BlockType.TABLE].click()

            assert formula_block.block_type == BlockType.TABLE
            assert formula_block.source_label == "table"
            assert formula_block.source == BlockSource.USER_EDITED
        finally:
            panel.close()

    print("test_layout_panel_type_buttons_change_unlocked_block_type PASSED")


def test_layout_panel_type_buttons_are_grouped():
    from PySide6.QtWidgets import QLabel, QPushButton

    from app.ui.recognize.layout_panel import (
        BLOCK_SUBTYPE_BUTTON_ORDER,
        BLOCK_TYPE_BUTTON_GROUPS,
        BLOCK_TYPE_BUTTON_ORDER,
        LayoutPanel,
    )

    _get_qapp()
    flat_specs = tuple(spec for _title, specs in BLOCK_TYPE_BUTTON_GROUPS for spec in specs)
    assert flat_specs == BLOCK_SUBTYPE_BUTTON_ORDER
    assert BLOCK_TYPE_BUTTON_ORDER == tuple(dict.fromkeys(spec.block_type for spec in BLOCK_SUBTYPE_BUTTON_ORDER))
    assert all(1 <= len(specs) <= 6 for _title, specs in BLOCK_TYPE_BUTTON_GROUPS)
    assert {
        "heading_1",
        "heading_2",
        "heading_3",
        "heading_4",
        "heading_5",
        "heading_6",
        "text",
        "abstract",
        "header",
        "footer",
        "number",
        "footnote",
        "formula",
        "table",
        "figure",
        "chart",
        "figure_title",
        "table_title",
        "reference_content",
    } == {spec.source_label for spec in BLOCK_SUBTYPE_BUTTON_ORDER}

    panel = LayoutPanel()
    try:
        titles = [
            label.text()
            for label in panel.findChildren(QLabel)
            if label.objectName() == "blockTypeGroupTitle"
        ]
        expected_titles = [title for title, _specs in BLOCK_TYPE_BUTTON_GROUPS]
        assert titles == expected_titles
        assert set(panel._new_type_buttons) == set(BLOCK_TYPE_BUTTON_ORDER)
        assert set(panel._selected_type_buttons) == set(BLOCK_TYPE_BUTTON_ORDER)
        expected_source_labels = {spec.source_label for spec in BLOCK_SUBTYPE_BUTTON_ORDER}
        assert set(panel._new_subtype_buttons) == expected_source_labels
        assert set(panel._selected_subtype_buttons) == expected_source_labels
        assert panel._new_subtype_buttons is panel._selected_subtype_buttons
        button_labels = [
            button.text()
            for button in panel.findChildren(QPushButton)
            if button.objectName() == "blockTypeButton"
        ]
        assert "其他" not in button_labels
        assert panel._type_context_title.text() == "新建框类型"
        assert panel._selection_mode_lbl.text() == "新建模式:"
        assert panel._selection_type_status.text() == "正文"
    finally:
        panel.close()

    print("test_layout_panel_type_buttons_are_grouped PASSED")


def test_layout_panel_subtype_buttons_write_paddle_source_label():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, BlockSource, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(image_path=str(image_path), width=120, height=80)

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_subtype_buttons["formula"].click()

            panel._on_block_created(BBox(10, 10, 20, 10))

            assert len(page.blocks) == 1
            block = page.blocks[0]
            assert block.block_type == BlockType.EQUATION
            assert block.source_label == "display_formula"
            assert block.source == BlockSource.MANUAL_DRAW
            assert block.ocr_policy != OcrPolicy.TEXT_OCR
            assert panel._type_context_title.text() == "选中框类型"
            assert panel._selection_mode_lbl.text() == "选中类型:"
            assert panel._selection_type_status.text() == "公式"
        finally:
            panel.close()

    print("test_layout_panel_subtype_buttons_write_paddle_source_label PASSED")


def test_layout_panel_formula_button_infers_inline_formula_inside_text_block():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 100, QImage.Format.Format_RGB888).save(str(image_path))
        text_block = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 120, 40), source_label="text")
        page = Page(image_path=str(image_path), width=160, height=100, blocks=[text_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._new_subtype_buttons["formula"].click()
            panel._on_block_created(BBox(40, 20, 20, 10))

            formula = page.blocks[-1]
            assert formula.block_type == BlockType.EQUATION
            assert formula.source_label == "inline_formula"
            assert panel._selection_type_status.text() == "公式"
        finally:
            panel.close()

    print("test_layout_panel_formula_button_infers_inline_formula_inside_text_block PASSED")


def test_layout_panel_search_results_can_batch_apply_heading_level():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 120, QImage.Format.Format_RGB888).save(str(image_path))
        target = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 120, 20),
            source_label="paragraph_title",
            lines=[Line(text="一、引言", confidence=0.9, bbox=BBox(10, 10, 120, 20))],
        )
        other = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 50, 120, 20),
            source_label="text",
            lines=[Line(text="普通正文", confidence=0.9, bbox=BBox(10, 50, 120, 20))],
        )
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[target, other])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._search_regex.setChecked(True)
            panel._search_input.setText("^一、")
            source_idx = panel._search_source_filter.findData("title")
            assert source_idx >= 0
            panel._search_source_filter.setCurrentIndex(source_idx)
            assert len(panel._block_search_matches) == 1

            target_idx = panel._search_target_combo.findData("heading_1")
            assert target_idx >= 0
            panel._search_target_combo.setCurrentIndex(target_idx)
            panel._apply_search_target_to_matches()

            assert target.block_type == BlockType.TITLE
            assert target.source_label == "heading_1"
            assert other.block_type == BlockType.TEXT
            assert panel._outline_tree.topLevelItemCount() == 1
            heading_item = panel._outline_tree.topLevelItem(0)
            assert heading_item.text(0) == "一、引言"
            assert "H1" in heading_item.toolTip(0)
            assert "第 1 页" in heading_item.toolTip(0)
            assert heading_item.childCount() == 0
        finally:
            panel.close()

    print("test_layout_panel_search_results_can_batch_apply_heading_level PASSED")


def test_layout_panel_search_source_filter_uses_layout_snapshot_view():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        Line,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 120, QImage.Format.Format_RGB888).save(str(image_path))
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 120, 20),
            source_label="text",
            lines=[Line(text="一、引言", confidence=0.9, bbox=BBox(10, 10, 120, 20))],
        )
        page = Page(image_path=str(image_path), width=160, height=120, blocks=[block])
        set_layout_snapshot_for_page(
            page,
            LayoutSnapshot(
                page_uid=page.uid,
                artifact_uid="artifact-1",
                source_engine="paddleocr-vl",
                source_run_id="layout-run-1",
                blocks=(
                    LayoutBlockSnapshot(
                        block_type=BlockType.TITLE,
                        bbox=block.bbox,
                        order=0,
                        source_label="heading_1",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="heading_1"),
                        ocr_policy=OcrPolicy.TEXT_OCR,
                        uid=block.uid,
                    ),
                ),
            ),
        )

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._search_regex.setChecked(True)
            panel._search_input.setText("^一、")
            source_idx = panel._search_source_filter.findData("title")
            assert source_idx >= 0
            panel._search_source_filter.setCurrentIndex(source_idx)

            assert len(panel._block_search_matches) == 1
            assert panel._block_search_matches[0] == (0, block)
            assert panel._search_results.item(0).text() == "一、引言"
        finally:
            panel.close()

    print("test_layout_panel_search_source_filter_uses_layout_snapshot_view PASSED")


def test_layout_panel_search_batch_apply_undo_restores_all_pages():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pages = []
        blocks = []
        for page_no in (1, 2):
            image_path = tmp / f"page-{page_no}.png"
            QImage(160, 120, QImage.Format.Format_RGB888).save(str(image_path))
            block = Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 10, 120, 20),
                source_label="paragraph_title",
                lines=[Line(text=f"一、标题{page_no}", confidence=0.9, bbox=BBox(10, 10, 120, 20))],
            )
            blocks.append(block)
            pages.append(Page(image_path=str(image_path), width=160, height=120, page_number=page_no, blocks=[block]))

        panel = LayoutPanel()
        try:
            panel.set_pages(pages)
            app.processEvents()
            panel._search_regex.setChecked(True)
            panel._search_input.setText("^一、")
            assert len(panel._block_search_matches) == 2

            target_idx = panel._search_target_combo.findData("heading_1")
            assert target_idx >= 0
            panel._search_target_combo.setCurrentIndex(target_idx)
            panel._apply_search_target_to_matches()

            assert [block.block_type for block in blocks] == [BlockType.TITLE, BlockType.TITLE]
            assert [block.source_label for block in blocks] == ["heading_1", "heading_1"]
            assert panel._btn_undo.isEnabled()

            panel._undo_last_edit()
            restored = [page.blocks[0] for page in pages]
            assert [block.block_type for block in restored] == [BlockType.TEXT, BlockType.TEXT]
            assert [block.source_label for block in restored] == ["paragraph_title", "paragraph_title"]
            assert not panel._btn_undo.isEnabled()
        finally:
            panel.close()

    print("test_layout_panel_search_batch_apply_undo_restores_all_pages PASSED")


def test_layout_panel_find_dialog_preset_matches_chinese_heading_forms():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(180, 160, QImage.Format.Format_RGB888).save(str(image_path))
        blocks = [
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 10, 120, 20),
                source_label="text",
                lines=[Line(text="一、研究背景", confidence=0.9, bbox=BBox(10, 10, 120, 20))],
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 45, 120, 20),
                source_label="text",
                lines=[Line(text="（一）基本问题", confidence=0.9, bbox=BBox(10, 45, 120, 20))],
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 80, 120, 20),
                source_label="text",
                lines=[Line(text="普通正文", confidence=0.9, bbox=BBox(10, 80, 120, 20))],
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 115, 120, 20),
                source_label="text",
                lines=[Line(text="1. 数字标题", confidence=0.9, bbox=BBox(10, 115, 120, 20))],
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox(10, 150, 120, 20),
                source_label="text",
                lines=[Line(text="正文包含2024数字", confidence=0.9, bbox=BBox(10, 150, 120, 20))],
            ),
        ]
        page = Page(image_path=str(image_path), width=180, height=200, blocks=blocks)

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            assert panel._find_dialog.isHidden()
            panel.show_find_dialog()
            assert not panel._find_dialog.isHidden()

            preset_index = next(
                idx for idx in range(panel._search_preset.count())
                if "中文序号标题" in panel._search_preset.itemText(idx)
            )
            panel._search_preset.setCurrentIndex(preset_index)

            assert panel._search_regex.isChecked()
            assert len(panel._block_search_matches) == 1
            assert [match[1] for match in panel._block_search_matches] == [blocks[0]]
            assert panel._search_results.item(0).text() == "一、研究背景"

            preset_index = next(
                idx for idx in range(panel._search_preset.count())
                if "括号中文标题" in panel._search_preset.itemText(idx)
            )
            panel._search_preset.setCurrentIndex(preset_index)
            assert len(panel._block_search_matches) == 1
            assert [match[1] for match in panel._block_search_matches] == [blocks[1]]

            preset_index = next(
                idx for idx in range(panel._search_preset.count())
                if "数字标题" in panel._search_preset.itemText(idx)
            )
            panel._search_preset.setCurrentIndex(preset_index)
            assert len(panel._block_search_matches) == 1
            assert [match[1] for match in panel._block_search_matches] == [blocks[3]]
        finally:
            panel.close()

    print("test_layout_panel_find_dialog_preset_matches_chinese_heading_forms PASSED")


def test_layout_panel_heading_outline_uses_nested_heading_levels():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    def heading(label: str, text: str, y: int) -> Block:
        return Block(
            block_type=BlockType.TITLE,
            bbox=BBox(10, y, 120, 18),
            source_label=label,
            lines=[Line(text=text, confidence=0.9, bbox=BBox(10, y, 120, 18))],
        )

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 160, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(
            image_path=str(image_path),
            width=160,
            height=160,
            blocks=[
                heading("heading_1", "第一章", 10),
                heading("heading_2", "第一节", 40),
                heading("heading_3", "一、背景", 70),
            ],
        )

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            assert panel._outline_tree.topLevelItemCount() == 1
            h1 = panel._outline_tree.topLevelItem(0)
            assert h1.text(0) == "第一章"
            assert "H1" in h1.toolTip(0)
            assert h1.childCount() == 1
            h2 = h1.child(0)
            assert h2.text(0) == "第一节"
            assert "H2" in h2.toolTip(0)
            assert h2.childCount() == 1
            h3 = h2.child(0)
            assert h3.text(0) == "一、背景"
            assert "H3" in h3.toolTip(0)
        finally:
            panel.close()

    print("test_layout_panel_heading_outline_uses_nested_heading_levels PASSED")


def test_layout_panel_heading_outline_uses_layout_snapshot_view():
    from pathlib import Path
    import tempfile

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        Line,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 100, QImage.Format.Format_RGB888).save(str(image_path))
        block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(10, 10, 120, 20),
            source_label="text",
            lines=[Line(text="第一章", confidence=0.9, bbox=BBox(10, 10, 120, 20))],
        )
        page = Page(image_path=str(image_path), width=160, height=100, blocks=[block])
        set_layout_snapshot_for_page(
            page,
            LayoutSnapshot(
                page_uid=page.uid,
                artifact_uid="artifact-1",
                source_engine="paddleocr-vl",
                source_run_id="layout-run-1",
                blocks=(
                    LayoutBlockSnapshot(
                        block_type=BlockType.TITLE,
                        bbox=block.bbox,
                        order=0,
                        source_label="heading_1",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="heading_1"),
                        ocr_policy=OcrPolicy.TEXT_OCR,
                        uid=block.uid,
                    ),
                ),
            ),
        )

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()

            assert panel._outline_tree.topLevelItemCount() == 1
            item = panel._outline_tree.topLevelItem(0)
            assert item.text(0) == "第一章"
            assert item.data(0, Qt.ItemDataRole.UserRole) == (0, block.uid)
            panel._on_outline_item_clicked(item)
            assert panel._selected_block is block
        finally:
            panel.close()

    print("test_layout_panel_heading_outline_uses_layout_snapshot_view PASSED")


def test_layout_panel_heading_outline_refreshes_after_ocr_text_arrives():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(160, 100, QImage.Format.Format_RGB888).save(str(image_path))
        heading = Block(
            block_type=BlockType.TITLE,
            bbox=BBox(10, 10, 120, 20),
            source_label="heading_1",
            note="Paddle 标题预览",
        )
        page = Page(image_path=str(image_path), width=160, height=100, blocks=[heading])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            assert panel._outline_tree.topLevelItem(0).text(0) == "Paddle 标题预览"

            heading.lines = [
                Line(text="Hanwang 标题文本", confidence=0.95, bbox=BBox(10, 10, 120, 20))
            ]
            panel.refresh_text_indexes()

            assert panel._outline_tree.topLevelItem(0).text(0) == "Hanwang 标题文本"
        finally:
            panel.close()

    print("test_layout_panel_heading_outline_refreshes_after_ocr_text_arrives PASSED")


def test_layout_panel_undo_restores_block_edits():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(image_path=str(image_path), width=120, height=80)
        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            assert len(page.blocks) == 0

            panel._on_block_created(BBox(10, 10, 20, 20))
            assert len(page.blocks) == 1
            assert page.blocks[0].block_type == BlockType.TEXT

            panel._undo_last_edit()
            assert len(page.blocks) == 0
            assert not panel._btn_undo.isEnabled()
        finally:
            panel.close()

    print("test_layout_panel_undo_restores_block_edits PASSED")


def test_layout_panel_undo_snapshot_uses_layout_snapshot_view_geometry():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import (
        BBox,
        Block,
        BlockOrigin,
        BlockType,
        LayoutBlockSnapshot,
        LayoutSnapshot,
        OcrPolicy,
        Page,
    )
    from app.models.layout_snapshot_store import layout_snapshot_for_page, set_layout_snapshot_for_page
    from app.services.layout_edit_service import LayoutEditCommand
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        block = Block(block_type=BlockType.TEXT, bbox=BBox.from_xyxy(80, 60, 110, 75), source_label="text")
        snapshot_bbox = BBox.from_xyxy(10, 10, 40, 25)
        changed_bbox = BBox.from_xyxy(45, 20, 75, 35)
        page = Page(image_path=str(image_path), width=120, height=80, blocks=[block])
        set_layout_snapshot_for_page(
            page,
            LayoutSnapshot(
                page_uid=page.uid,
                artifact_uid="artifact-1",
                source_engine="paddleocr-vl",
                source_run_id="layout-run-1",
                blocks=(
                    LayoutBlockSnapshot(
                        block_type=BlockType.TEXT,
                        bbox=snapshot_bbox,
                        order=0,
                        source_label="text",
                        origin=BlockOrigin(source_engine="paddleocr-vl", source_label="text"),
                        ocr_policy=OcrPolicy.TEXT_OCR,
                        uid=block.uid,
                    ),
                ),
            ),
        )

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._push_undo_snapshot_for_page(0)
            panel._layout_edit_service.apply(LayoutEditCommand.update_geometry(page, block, bbox=changed_bbox))
            assert layout_snapshot_for_page(page).blocks[0].bbox == changed_bbox

            panel._undo_last_edit()

            assert page.blocks[0].bbox == snapshot_bbox
            assert layout_snapshot_for_page(page).blocks[0].bbox == snapshot_bbox
        finally:
            panel.close()

    print("test_layout_panel_undo_snapshot_uses_layout_snapshot_view_geometry PASSED")


def test_layout_panel_undo_preserves_view_transform():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(240, 160, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(
            image_path=str(image_path),
            width=240,
            height=160,
            blocks=[Block(block_type=BlockType.EQUATION, bbox=BBox(10, 10, 20, 20))],
        )
        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            panel._viewer.scale(1.8, 1.8)
            before = panel._viewer.transform()

            panel._on_block_created(BBox(80, 40, 20, 20))
            assert len(page.blocks) == 2

            panel._undo_last_edit()
            after = panel._viewer.transform()

            assert len(page.blocks) == 1
            assert abs(after.m11() - before.m11()) < 0.000001
            assert abs(after.m22() - before.m22()) < 0.000001
        finally:
            panel.close()

    print("test_layout_panel_undo_preserves_view_transform PASSED")


def test_layout_panel_promotes_real_inline_formula_overlays_to_editable_blocks():
    import json
    from pathlib import Path

    from PySide6.QtWidgets import QGraphicsItem

    from app.core.inline_formula_edit_state import HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    project_root = Path(__file__).resolve().parents[1]
    layout_json = (
        project_root
        / "tests"
        / "fixtures"
        / "layout"
        / "120166-layout-api-fixture.json"
    )
    raw = json.loads(layout_json.read_text(encoding="utf-8"))
    page_info = raw["page"]
    sample = project_root / page_info["display_image_path"]
    page = Page(
        image_path=str(sample),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=1,
    )
    blocks, _raw_overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    page.blocks = blocks

    top_level_labels = [block.source_label for block in page.blocks]
    assert "display_formula" in top_level_labels
    assert "formula_number" in top_level_labels

    panel = LayoutPanel()
    try:
        overlays = panel._collect_readonly_layout_overlays(page)
        assert overlays == []

        panel.show_analysis_result([page])
        app.processEvents()

        inline_blocks = [block for block in page.blocks if block.source_label == "inline_formula"]
        assert len(inline_blocks) == 7
        assert all(block.block_type == BlockType.EQUATION for block in inline_blocks)
        assert len(panel._viewer._readonly_overlay_items) == 0
        assert len(panel._viewer._block_items) == len(page.blocks)
        inline_items = [
            item for item, block in panel._viewer._block_items
            if block.source_label == "inline_formula"
        ]
        assert len(inline_items) == 7
        assert all(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable for item in inline_items)
        assert all(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable for item in inline_items)

        first_inline = inline_blocks[0]
        first_item = next(item for item, block in panel._viewer._block_items if block is first_inline)
        first_item.setSelected(True)
        panel._delete_selected()
        assert first_inline not in page.blocks
        panel._show_page_layers(page)
        assert len([block for block in page.blocks if block.source_label == "inline_formula"]) == 6
        delete_events = [
            event for event in page.layout_edit_events
            if event.op == "delete_inline_formula"
            and event.after.get(HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY)
        ]
        assert len(delete_events) == 1
    finally:
        panel.close()

    print("test_layout_panel_promotes_real_inline_formula_overlays_to_editable_blocks PASSED")


def test_layout_panel_moved_generated_inline_formula_keeps_manual_geometry():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.core.inline_formula_edit_state import HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(220, 80, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        parent_record = {
            "block_label": "text",
            "block_bbox": [0, 0, 200, 40],
            "block_content": "甲 $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
            ],
            "_layout_line_routes": [
                {"bbox": [0, 0, 200, 40], "segments": [{"kind": "text", "bbox": [0, 0, 200, 40]}]},
            ],
        }
        page = Page(
            image_path=str(image_path),
            width=220,
            height=80,
            raw_layout_artifact=_paddle_layout_artifact([parent_record]),
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(0, 0, 200, 40),
                    origin=BlockOrigin(source_label="text", raw_index=0),
                )
            ],
        )

        panel = LayoutPanel()
        try:
            panel.show_analysis_result([page])
            app.processEvents()
            inline_blocks = [block for block in page.blocks if block.source_label == "inline_formula"]
            assert len(inline_blocks) == 1
            inline = inline_blocks[0]
            assert inline.bbox.to_xyxy() == (40, 0, 70, 30)
            assert inline.origin is not None
            assert inline.origin.source_label == "inline_formula"
            assert inline.origin.original_bbox == BBox.from_xyxy(40, 0, 70, 30)
            assert inline.origin.raw_index == 0

            inline.bbox = BBox.from_xyxy(45, 0, 75, 30)
            panel._on_block_moved(inline)
            from app.core.raw_ocr_artifact import layout_route_attachments

            assert ROUTE_SUBBLOCKS_FIELD not in _raw_layout_records(page)[0]
            raw_subblock = layout_route_attachments(page)[0][0]
            assert "_ui_deleted" not in raw_subblock
            claim_events = [
                event for event in page.layout_edit_events
                if event.op == "claim_inline_formula"
                and event.after.get(HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY) == [40, 0, 70, 30]
            ]
            assert len(claim_events) == 1

            # Real resize drags emit several geometry changes. Once the original
            # Paddle inline formula is marked handled, later drag events must
            # keep the already-established binding and only update manual_bbox.
            inline.bbox = BBox.from_xyxy(50, 0, 80, 30)
            panel._on_block_moved(inline)
            panel._refresh_current_page_layers()
            app.processEvents()

            inline_blocks = [block for block in page.blocks if block.source_label == "inline_formula"]
            assert len(inline_blocks) == 1
            assert inline_blocks[0] is inline
            assert inline.bbox.to_xyxy() == (50, 0, 80, 30)
            assert inline.source.value == "user_edited"
            assert inline.paddle_binding is not None
            assert inline.paddle_binding.manual_bbox == [50, 0, 80, 30]
            assert inline.origin is not None
            assert inline.origin.original_bbox == BBox.from_xyxy(40, 0, 70, 30)
            assert "_ui_deleted" not in layout_route_attachments(page)[0][0]
            claim_events = [
                event for event in page.layout_edit_events
                if event.op == "claim_inline_formula"
                and event.after.get(HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY) == [40, 0, 70, 30]
            ]
            assert len(claim_events) == 1

            ocr_blocks = _page_blocks_from_layout(page)
            subblocks = ocr_blocks[0][ROUTE_SUBBLOCKS_FIELD]
            assert [sub["block_bbox"] for sub in subblocks] == [[50, 0, 80, 30]]
        finally:
            panel.close()

    print("test_layout_panel_moved_generated_inline_formula_keeps_manual_geometry PASSED")


def test_layout_panel_corrected_inline_formula_releases_covered_text_slice():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block, text_slice_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        image = QImage(240, 80, QImage.Format.Format_RGB888)
        image.fill(0xFFFFFFFF)
        image.save(str(image_path))
        parent_record = {
            "block_label": "text",
            "block_bbox": [0, 0, 220, 40],
            "block_content": "甲0， $ A $ 乙",
            ROUTE_SUBBLOCKS_FIELD: [
                {"block_label": "inline_formula", "block_bbox": [40, 0, 120, 30]},
            ],
            "_layout_line_routes": [
                {"bbox": [0, 0, 220, 40], "segments": [{"kind": "text", "bbox": [0, 0, 220, 40]}]},
            ],
        }
        page = Page(
            image_path=str(image_path),
            width=240,
            height=80,
            raw_layout_artifact=_paddle_layout_artifact([parent_record]),
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(0, 0, 220, 40),
                    origin=BlockOrigin(source_label="text", raw_index=0),
                )
            ],
        )

        panel = LayoutPanel()
        try:
            panel.show_analysis_result([page])
            app.processEvents()
            inline = next(block for block in page.blocks if block.source_label == "inline_formula")
            assert inline.bbox.to_xyxy() == (40, 0, 120, 30)

            inline.bbox = BBox.from_xyxy(80, 0, 120, 30)
            panel._on_block_moved(inline)

            ocr_blocks = _page_blocks_from_layout(page)
            parent = ocr_blocks[0]
            assert [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]] == [[80, 0, 120, 30]]

            routes = line_routes_for_block(parent, 240, 80)
            formula_segments = [
                segment
                for route in routes
                for segment in route["segments"]
                if segment["kind"] == "formula"
            ]
            assert [segment["bbox"] for segment in formula_segments] == [[80, 0, 120, 30]]
            assert [route["bbox"] for route in text_slice_routes_for_block(parent, 240, 80)] == [
                [0, 0, 80, 30],
                [120, 0, 220, 30],
            ]
        finally:
            panel.close()

    print("test_layout_panel_corrected_inline_formula_releases_covered_text_slice PASSED")


def test_layout_panel_skips_superscript_marker_inline_formula_overlays_from_120169():
    import json
    from pathlib import Path

    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page
    from app.services.layout_overlay_service import LayoutOverlayService

    _get_qapp()
    project_root = Path(__file__).resolve().parents[1]
    layout_json = project_root / "file" / "244771纵校" / "120169.layout-api.json"
    raw = json.loads(layout_json.read_text(encoding="utf-8"))
    page_info = raw["page"]
    page = Page(
        image_path=str(project_root / "file" / "244771纵校" / "120169.tif"),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=169,
    )
    blocks, _raw_overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    page.blocks = blocks
    footnote = next(block for block in page.blocks if block.source_label == "footnote")
    assert footnote.block_type == BlockType.TEXT

    inline_bboxes = [
        overlay.bbox.to_xyxy()
        for overlay in LayoutOverlayService().iter_inline_formula_overlays(page)
    ]

    assert (372, 2022, 440, 2069) not in inline_bboxes  # $ ^{*} $
    assert (619, 2737, 672, 2785) not in inline_bboxes  # $ ^{②} $
    assert (434, 2658, 547, 2710) in inline_bboxes      # $ Time_{t} $
    assert (1273, 2740, 1325, 2793) in inline_bboxes    # $ \beta_{t} $

    print("test_layout_panel_skips_superscript_marker_inline_formula_overlays_from_120169 PASSED")


def test_workflow_controller_layout_progress_signal():
    from app.controllers.workflow_controller import WorkflowController

    controller = WorkflowController()
    events = []
    statuses = []
    controller.layout_progress.connect(lambda current, total: events.append((current, total)))
    controller.status_message.connect(statuses.append)

    controller._on_layout_progress(1, 3)

    assert events == [(1, 3)]
    assert statuses[-1] == "版面分析中…"

    print("test_workflow_controller_layout_progress_signal PASSED")


def test_workflow_controller_abstracts_internal_ocr_progress_messages():
    from app.controllers.workflow_controller import WorkflowController
    from app.services.ocr_run_result import OcrProgress

    controller = WorkflowController()
    statuses = []
    progress_states = []
    raw_progress = []
    controller.status_message.connect(statuses.append)
    controller.progress_state_changed.connect(progress_states.append)
    controller.ocr_progress.connect(raw_progress.append)

    controller._on_ocr_progress(OcrProgress(
        current_page=1,
        total_pages=2,
        current_block=1,
        total_blocks=8,
        completed_pages=0,
        message="Hanwang micro-recblock SegImg 分块中…",
    ))

    assert statuses[-1] == "文字识别中…"
    assert progress_states[-1].message == "文字识别中…"
    assert raw_progress[-1].message == "Hanwang micro-recblock SegImg 分块中…"

    print("test_workflow_controller_abstracts_internal_ocr_progress_messages PASSED")


def test_ocr_worker_emits_fine_grained_hanwang_progress_without_waiting():
    from app.controllers.workflow_controller import OcrPipelineWorker
    from app.models import Page
    from app.services.ocr_run_result import OcrProgress, OcrRunResult

    _get_qapp()
    page = Page(image_path="/tmp/progress.png", width=10, height=10)
    emitted = []

    class FastPipeline:
        def process_project(self, project, progress_callback=None):
            for current in (0, 1, 5, 10):
                progress_callback(OcrProgress(
                    current_page=1,
                    total_pages=1,
                    current_block=current,
                    total_blocks=10,
                    completed_pages=0,
                    message=f"Hanwang OCR 识别中 {current}/10",
                ))
            return OcrRunResult(pages=[page])

    worker = OcrPipelineWorker(FastPipeline(), [page])
    worker.progress_state.connect(emitted.append)
    worker.run()

    assert [event.current_block for event in emitted] == [0, 1, 5, 10]


def test_workflow_controller_clamps_ocr_page_concurrency_to_page_count():
    from app.controllers.workflow_controller import WorkflowController
    from app.models import Page

    class PageHybridEngine:
        prefer_page_hybrid_blocks = True

    controller = WorkflowController()
    pages = [
        Page(image_path="/tmp/concurrency-1.png", width=10, height=10),
        Page(image_path="/tmp/concurrency-2.png", width=10, height=10),
    ]
    controller._ocr_page_concurrency = lambda: 20  # type: ignore[method-assign]

    assert controller._effective_ocr_page_concurrency(pages, PageHybridEngine()) == 2
    assert controller._effective_ocr_page_concurrency(pages[:1], PageHybridEngine()) == 1
    assert controller._effective_ocr_page_concurrency(pages, object()) == 1

    print("test_workflow_controller_clamps_ocr_page_concurrency_to_page_count PASSED")


def test_main_window_layout_error_is_status_only():
    from PySide6.QtWidgets import QMessageBox

    from app.controllers.workflow_controller import STEP_LAYOUT, STEP_OCR
    from app.ui.main_window import MainWindow

    _get_qapp()
    calls = []
    original_critical = QMessageBox.critical
    QMessageBox.critical = lambda *args, **kwargs: calls.append(args)
    window = MainWindow()
    try:
        window._go_to_step(STEP_LAYOUT)
        window._on_worker_error("所有页面版面分析失败：网络错误")
        assert calls == []
        assert "网络错误" in window.statusBar().currentMessage()
        window._go_to_step(STEP_OCR)
        window._on_worker_error("OCR 自动衔接失败")
        assert calls == []
        assert "OCR 自动衔接失败" in window.statusBar().currentMessage()
    finally:
        QMessageBox.critical = original_critical
        window.close()

    print("test_main_window_layout_error_is_status_only PASSED")


def test_main_window_worker_error_finishes_background_ocr_progress_on_proof_step():
    from PySide6.QtWidgets import QMessageBox

    from app.controllers.workflow_controller import STEP_HPROOF
    from app.services.ocr_run_result import OcrProgress
    from app.ui.main_window import MainWindow

    _get_qapp()
    calls = []
    original_critical = QMessageBox.critical
    QMessageBox.critical = lambda *args, **kwargs: calls.append(args)
    window = MainWindow()
    try:
        window._on_ocr_progress(OcrProgress(
            current_page=1,
            total_pages=2,
            current_block=1,
            total_blocks=4,
            completed_pages=0,
            message="Hanwang OCR 识别中 1/4",
        ))
        assert window._ocr_placeholder.is_active()

        window._go_to_step(STEP_HPROOF)
        window._on_worker_error("OCR 后台失败")

        assert calls
        assert not window._ocr_placeholder.is_active()
        assert window._ocr_placeholder.isHidden()
        window._set_status_message("后续状态")
        assert window.statusBar().currentMessage() == "后续状态"
    finally:
        QMessageBox.critical = original_critical
        window.close()

    print("test_main_window_worker_error_finishes_background_ocr_progress_on_proof_step PASSED")


def test_layout_panel_status_label_elides_long_errors():
    from app.ui.recognize.layout_panel import LayoutPanel, STATUS_LABEL_MAX_CHARS

    _get_qapp()
    panel = LayoutPanel()
    try:
        long_error = "OCR 失败：" + "network-timeout-" * 30
        panel.finish_analysis_progress(long_error)

        assert len(panel._status_lbl.text()) <= STATUS_LABEL_MAX_CHARS
        assert panel._status_lbl.text().endswith("…")
        assert panel._status_lbl.toolTip() == long_error
    finally:
        panel.close()

    print("test_layout_panel_status_label_elides_long_errors PASSED")


def test_main_window_centered_resize_expands_from_current_center():
    from PySide6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    app = _get_qapp()
    window = MainWindow()
    try:
        window.setGeometry(220, 180, 300, 240)
        window.show()
        app.processEvents()
        before = window.frameGeometry().center()

        window._set_centered_window_size(700, 400)
        app.processEvents()

        after = window.frameGeometry().center()
        assert window.width() >= 700
        assert window.height() >= 400
        assert window.width() <= max(700, window.minimumSizeHint().width())
        assert window.height() <= max(400, window.minimumSizeHint().height())
        screen = window.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        frame = window.frameGeometry()
        if available is None or (
            frame.width() <= available.width()
            and frame.height() <= available.height()
            and available.left() <= before.x() - frame.width() // 2
            and before.x() + frame.width() // 2 <= available.right() + 1
            and available.top() <= before.y() - frame.height() // 2
            and before.y() + frame.height() // 2 <= available.bottom() + 1
        ):
            assert abs(after.x() - before.x()) <= 1
            assert abs(after.y() - before.y()) <= 1
        else:
            assert frame.left() >= available.left()
            assert frame.top() >= available.top()
            assert frame.right() <= available.right()
            assert frame.bottom() <= available.bottom()
    finally:
        window.close()

    print("test_main_window_centered_resize_expands_from_current_center PASSED")


def test_main_window_initial_import_window_is_screen_centered():
    from PySide6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    app = _get_qapp()
    window = MainWindow()
    try:
        app.processEvents()
        screen = window.screen() or QApplication.primaryScreen()
        assert screen is not None
        available_center = screen.availableGeometry().center()
        frame_center = window.frameGeometry().center()
        assert abs(frame_center.x() - available_center.x()) <= 2
        assert abs(frame_center.y() - available_center.y()) <= 2
    finally:
        window.close()

    print("test_main_window_initial_import_window_is_screen_centered PASSED")


def test_main_window_maximize_state_is_not_forced_back_to_normal():
    from app.ui.main_window import MainWindow

    app = _get_qapp()
    window = MainWindow()
    try:
        window.show()
        app.processEvents()
        normal_size = window.size()

        window.showMaximized()
        app.processEvents()
        assert window.isMaximized()

        window.showNormal()
        app.processEvents()
        assert not window.isMaximized()
        assert window.size() == normal_size
    finally:
        window.close()

    print("test_main_window_maximize_state_is_not_forced_back_to_normal PASSED")


def test_main_window_file_menu_uses_close_project_action():
    from PySide6.QtGui import QKeySequence

    from app.ui.main_window import MainWindow

    app = _get_qapp()
    window = MainWindow()
    try:
        assert window.menuBar().isHidden()
        file_menu = window._top_bar._btn_file_menu.menu()
        assert file_menu is not None
        actions = [action for action in file_menu.actions() if not action.isSeparator()]
        action_texts = [action.text() for action in actions]
        assert any("关闭项目" in text for text in action_texts)
        assert not any("新建项目" in text for text in action_texts)
        assert not any("退出" in text for text in action_texts)
        close_action = next(action for action in actions if "关闭项目" in action.text())
        assert close_action.shortcut().toString(QKeySequence.SequenceFormat.PortableText) == "Ctrl+W"
        assert any(
            action is close_action
            for action in window.actions()
        )
    finally:
        window.close()

    print("test_main_window_file_menu_uses_close_project_action PASSED")


def test_main_window_close_project_prompts_save_and_resets_workspace():
    from PySide6.QtWidgets import QMessageBox

    from app.models import OcrProject
    from app.ui.main_window import MainWindow

    _get_qapp()
    warnings = []
    saves = []
    original_warning = QMessageBox.warning
    QMessageBox.warning = lambda *args, **kwargs: warnings.append(args)

    class FakeStore:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    window = MainWindow()
    store = FakeStore()
    try:
        window._controller._project = OcrProject(name="demo")
        window._controller._store = store
        window._controller.save_project = lambda: saves.append(True) or True
        window._ask_save_before_close = lambda title, message: "save"

        window._close_project()

        assert saves == [True]
        assert warnings == []
        assert store.closed is True
        assert window._controller.project is None
        assert window._stack.currentWidget() is window._import_panel
        assert window._layout_panel._pages == []
        assert window.statusBar().currentMessage() == "项目已关闭"
    finally:
        QMessageBox.warning = original_warning
        window.close()

    print("test_main_window_close_project_prompts_save_and_resets_workspace PASSED")


def test_main_window_close_project_save_uses_save_as_for_transient_project():
    from PySide6.QtWidgets import QMessageBox

    from app.models import OcrProject
    from app.ui.main_window import MainWindow

    _get_qapp()
    warnings = []
    save_as_calls = []
    original_warning = QMessageBox.warning
    QMessageBox.warning = lambda *args, **kwargs: warnings.append(args)

    window = MainWindow()
    try:
        window._controller._project = OcrProject(name="draft")
        window._controller._store = None
        window._ask_save_before_close = lambda title, message: "save"
        window._save_project_as = lambda: save_as_calls.append(True) or True

        window._close_project()

        assert save_as_calls == [True]
        assert warnings == []
        assert window._controller.project is None
        assert window._stack.currentWidget() is window._import_panel
        assert window.statusBar().currentMessage() == "项目已关闭"
    finally:
        QMessageBox.warning = original_warning
        window.close()

    print("test_main_window_close_project_save_uses_save_as_for_transient_project PASSED")


# =====================================================================
# Fake OCR 引擎测试
# =====================================================================

def test_fake_ocr_engine():
    import numpy as np
    from app.engines.fake_ocr_engine import FakeOcrEngine
    from app.engines import OcrContext

    engine = FakeOcrEngine()
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    context = OcrContext()
    lines = engine.recognize(img, context)

    assert len(lines) > 0
    assert lines[0].text == "测试OCR文本"
    assert lines[0].confidence == 0.95

    # 应该有低置信行
    flagged = [l for l in lines if l.confidence < 0.80]
    assert len(flagged) > 0
    assert flagged[0].text == "低置信文本"

    print("test_fake_ocr_engine PASSED")


def test_create_engine_hanwang_exposes_page_block_capability():
    from app.engines import OCR_BBOX_SPACE_PAGE, supports_page_block_ocr
    from app.engines.real_ocr_adapter import create_engine

    engine = create_engine("hanwang")

    assert supports_page_block_ocr(engine) is True
    assert getattr(engine, "engine_id", "") == "hanwang.micro_recblock"
    assert getattr(engine, "bbox_space", "") == OCR_BBOX_SPACE_PAGE

    class FlagOnly:
        prefer_page_hybrid_blocks = True

    assert supports_page_block_ocr(FlagOnly()) is False

    print("test_create_engine_hanwang_exposes_page_block_capability PASSED")


def test_confidence_normalization():
    from app.engines.real_ocr_adapter import normalize_confidence
    from app.ui.widgets.confidence_badge import normalize_badge_score

    assert normalize_confidence(0.87) == 0.87
    assert normalize_confidence(87) == 0.87
    assert normalize_confidence("95") == 0.95
    assert normalize_confidence(None) == 0.0

    assert normalize_badge_score(88) == 0.88
    assert normalize_badge_score("0.76") == 0.76

    print("test_confidence_normalization PASSED")


def test_api_ocr_engine_does_not_request_return_word_box():
    import base64

    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine

    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"layoutParsingResults": []}}

    def fake_post(url, json, headers, timeout, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["proxies"] = kwargs.get("proxies")
        return DummyResponse()

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(
        mode="api",
        api_url="https://example.com",
        api_token="demo",
        api_timeout=12,
        paddle_api_network_mode="direct",
    )

    original_post = requests.post
    requests.post = fake_post
    try:
        engine = ApiOcrEngine()
        lines = engine.recognize(np.zeros((20, 30, 3), dtype=np.uint8), OcrContext())
        assert lines == []
        assert captured["url"] == "https://example.com/ocr"
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
        assert base64.b64decode(captured["json"]["file"]).startswith(b"\x89PNG\r\n\x1a\n")
        assert "returnWordBox" not in captured["json"]
        assert captured["json"]["useDocOrientationClassify"] is False
        assert captured["json"]["useDocUnwarping"] is False
        assert captured["json"]["useTextlineOrientation"] is False
        assert captured["json"]["textDetLimitSideLen"] == 1536
        assert captured["json"]["textDetBoxThresh"] == 0.6
        assert captured["json"]["textDetUnclipRatio"] == 2.0
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_does_not_request_return_word_box PASSED")


def test_ocr_ir_builder_ignores_legacy_word_box_payload():
    from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
    from app.core.ocr_ir_builder import build_ir_lines_from_item
    from app.models import BBox

    item = {
        "prunedResult": {
            "overall_ocr_res": {
                "rec_texts": ["甲乙"],
                "rec_scores": [0.95],
            },
            "text_word": [["甲", "乙"]],
            "text_word_region": [[
                [10, 20, 30, 40],
                [30, 20, 50, 40],
            ]],
        }
    }

    lines = build_ir_lines_from_item(item, fallback_bbox=BBox(0, 0, 100, 40))

    assert len(lines) == 1
    assert lines[0].text == "甲乙"
    assert lines[0].source_text == "甲乙"
    assert lines[0].source_field == "overall_ocr_res.rec_texts"
    assert lines[0].bbox == BBox(0, 0, 100, 40)
    assert lines[0].tokens == []
    assert MISSING_LINE_BBOX_FLAG in lines[0].review_flags

    token_only = {
        "prunedResult": {
            "text_word": [["甲", "乙"]],
            "text_word_region": [[[10, 20, 30, 40], [30, 20, 50, 40]]],
        }
    }
    assert build_ir_lines_from_item(token_only, fallback_bbox=BBox(0, 0, 100, 40)) == []

    print("test_ocr_ir_builder_ignores_legacy_word_box_payload PASSED")


def test_api_ocr_engine_does_not_promote_block_content_to_line():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "layoutParsingResults": [
                        {
                            "prunedResult": {
                                "overall_ocr_res": {
                                    "rec_texts": ["行级OCR"],
                                    "rec_scores": [0.92],
                                    "rec_boxes": [[10, 20, 80, 44]],
                                },
                                "parsing_res_list": [
                                    {
                                        "block_label": "text",
                                        "block_bbox": [8, 18, 160, 90],
                                        "block_content": "块级结构化文本，不能直接抬成行级Line",
                                    },
                                ],
                            },
                        }
                    ],
                },
            }

    original_post = requests.post
    requests.post = lambda *args, **kwargs: DummyResponse()
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(mode="api", api_url="https://example.com", api_timeout=12)
    try:
        engine = ApiOcrEngine()
        lines = engine.recognize(np.zeros((120, 200, 3), dtype=np.uint8), OcrContext())
        assert len(lines) == 1
        assert lines[0].text == "行级OCR"
        assert lines[0].ocr_text == "行级OCR"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_does_not_promote_block_content_to_line PASSED")


def test_api_ocr_engine_reads_direct_pruned_ppocr_rows():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "rec_texts": ["PP行"],
                                "rec_scores": [0.96],
                                "rec_boxes": [[10, 20, 90, 50]],
                            },
                        }
                    ],
                },
            }

    original_post = requests.post
    requests.post = lambda *args, **kwargs: DummyResponse()
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(mode="api", api_url="https://example.com", api_timeout=12)
    try:
        engine = ApiOcrEngine()
        lines = engine.recognize(np.zeros((120, 200, 3), dtype=np.uint8), OcrContext())
        assert len(lines) == 1
        assert lines[0].text == "PP行"
        assert lines[0].bbox.to_xyxy() == (10, 20, 90, 50)
        assert lines[0].confidence == 0.96
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_reads_direct_pruned_ppocr_rows PASSED")


def test_api_ocr_engine_ignores_block_content_without_rec_rows():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "layoutParsingResults": [
                        {
                            "prunedResult": {
                                "parsing_res_list": [
                                    {
                                        "block_label": "text",
                                        "block_bbox": [10, 10, 180, 80],
                                        "block_content": "只有块级结构文本",
                                    },
                                ],
                            },
                        }
                    ],
                },
            }

    original_post = requests.post
    requests.post = lambda *args, **kwargs: DummyResponse()
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(mode="api", api_url="https://example.com", api_timeout=12)
    try:
        engine = ApiOcrEngine()
        lines = engine.recognize(np.zeros((120, 220, 3), dtype=np.uint8), OcrContext())
        assert lines == []
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_ignores_block_content_without_rec_rows PASSED")


def test_api_ocr_engine_preserves_rec_text_without_any_geometry():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox, ProofStatus

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "layoutParsingResults": [
                        {
                            "prunedResult": {
                                "overall_ocr_res": {
                                    "rec_texts": ["有效OCR文本"],
                                    "rec_scores": [0.91],
                                    "rec_boxes": [],
                                },
                            },
                        }
                    ],
                },
            }

    original_post = requests.post
    requests.post = lambda *args, **kwargs: DummyResponse()
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(mode="api", api_url="https://example.com", api_timeout=12)
    try:
        engine = ApiOcrEngine()
        lines = engine.recognize(np.zeros((80, 120, 3), dtype=np.uint8), OcrContext())
        assert len(lines) == 1
        assert lines[0].text == "有效OCR文本"
        assert lines[0].ocr_text == "有效OCR文本"
        assert lines[0].bbox == BBox(0, 0, 120, 80)
        assert MISSING_LINE_BBOX_FLAG in lines[0].review_flags
        assert proof_status(lines[0]) == ProofStatus.AUTO_FLAGGED
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_preserves_rec_text_without_any_geometry PASSED")


# =====================================================================
# OcrPipeline 测试
# =====================================================================

def test_ocr_pipeline():
    import tempfile
    from app.engines.fake_ocr_engine import FakeOcrEngine
    from app.models import BBox, Block, BlockType, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    # 创建测试图像
    import cv2
    import numpy as np
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        # 创建测试数据
        bb = BBox(10, 10, 200, 300)
        block = Block(block_type=BlockType.TEXT, bbox=bb)
        page = Page(image_path=img_path, width=600, height=400)
        page.error_message = "OCR 失败：上一轮失败"
        page.blocks = [block]
        project = OcrProject(name="PipelineTest", pages=[page])

        # 使用 fake engine
        pipeline = OcrPipeline(engine=FakeOcrEngine())
        result = pipeline.process_project(project)

        assert len(result.pages) == 1
        assert len(result.pages[0].blocks) > 0
        # fake OCR 应生成了 lines
        assert len(result.pages[0].blocks[0].lines) > 0
        assert result.pages[0].blocks[0].lines[0].text is not None
        assert result.pages[0].error_message == ""

        print("test_ocr_pipeline PASSED")
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_keeps_page_relative_boxes():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class PageCoordEngine:
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="整页坐标", confidence=0.92, bbox=BBox(120, 70, 80, 18))]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((240, 320, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(100, 60, 150, 80))
        page = Page(image_path=img_path, width=320, height=240, blocks=[block])
        project = OcrProject(name="PageCoords", pages=[page])

        result = OcrPipeline(engine=PageCoordEngine()).process_project(project)
        line = result.pages[0].blocks[0].lines[0]

        assert line.bbox == BBox(120, 70, 80, 18)
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_offsets_crop_relative_boxes():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class CropCoordEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(
                text="局部坐标",
                confidence=0.92,
                bbox=BBox(20, 10, 80, 18),
                chars=[
                    Char(char="局", confidence=0.95, bbox=BBox(20, 10, 20, 18)),
                    Char(char="部", confidence=0.94, bbox=BBox(40, 10, 20, 18)),
                ],
            )]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((240, 320, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(100, 60, 150, 80))
        page = Page(image_path=img_path, width=320, height=240, blocks=[block])
        project = OcrProject(name="CropCoords", pages=[page])

        result = OcrPipeline(engine=CropCoordEngine()).process_project(project)
        line = result.pages[0].blocks[0].lines[0]

        assert line.bbox == BBox(120, 70, 80, 18)
        assert len(line.chars) == len(line.text)
        assert line.chars[0].bbox == BBox(120, 70, 20, 18)
        assert line.chars[1].bbox == BBox(140, 70, 20, 18)
        assert line.chars[2].bbox == BBox(160, 70, 20, 18)
        assert line.chars[3].bbox == BBox(180, 70, 20, 18)
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_prefers_engine_char_boxes_and_only_falls_back_for_missing_chars():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class PartialCharBoxEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(
                text="甲乙丙",
                confidence=0.94,
                bbox=BBox(10, 20, 90, 24),
                chars=[
                    Char(
                        char="甲",
                        confidence=0.94,
                        bbox=BBox(12, 20, 14, 24),
                        bbox_source="ocr",
                        bbox_granularity="char",
                        token_text="甲",
                    ),
                    Char(
                        char="乙",
                        confidence=0.94,
                        bbox=BBox(34, 20, 18, 24),
                        bbox_source="ocr",
                        bbox_granularity="char",
                        token_text="乙",
                    ),
                ],
            )]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((120, 220, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(100, 60, 100, 40))
        page = Page(image_path=img_path, width=220, height=120, blocks=[block])
        result = OcrPipeline(engine=PartialCharBoxEngine()).process_project(
            OcrProject(name="PartialCharBox", pages=[page])
        )
        line = result.pages[0].blocks[0].lines[0]

        assert line.chars[0].bbox == BBox(112, 80, 14, 24)
        assert line.chars[0].bbox_source == "ocr"
        assert line.chars[1].bbox == BBox(134, 80, 18, 24)
        assert line.chars[1].bbox_source == "ocr"
        assert line.chars[2].bbox_source == "fallback"
        assert line.chars[2].bbox_granularity == "fallback"
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_normalizes_proof_geometry():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class LooseLineEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(text="甲乙", confidence=0.96, bbox=BBox(10, 44, 160, 40))]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.full((120, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (16, 18), (204, 30), (0, 0, 0), -1)
        cv2.rectangle(img, (18, 68), (196, 82), (0, 0, 0), -1)
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 220, 120))
        page = Page(image_path=img_path, width=220, height=120, blocks=[block])
        result = OcrPipeline(engine=LooseLineEngine()).process_project(
            OcrProject(name="ProofNormalize", pages=[page])
        )
        line = result.pages[0].blocks[0].lines[0]

        assert line.bbox == BBox(10, 44, 160, 40)
        assert len(line.chars) == 2
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_process_block_normalizes_proof_geometry():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line
    from app.services.ocr_pipeline import OcrPipeline

    class LooseBlockEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(text="甲乙", confidence=0.96, bbox=BBox(10, 12, 80, 24), chars=[])]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.full((80, 140, 3), 255, dtype=np.uint8)
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(20, 10, 100, 40))
        result = OcrPipeline(engine=LooseBlockEngine()).process_block(block, img_path)
        line = result.lines[0]

        assert line.bbox == BBox(30, 22, 80, 24)
        assert len(line.chars) == 2
        assert [char.char for char in line.chars] == ["甲", "乙"]
        assert [char.bbox_source for char in line.chars] == ["fallback", "fallback"]
        assert [char.bbox_granularity for char in line.chars] == ["fallback", "fallback"]
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_emits_nonblocking_warning_when_proof_fallback_triggers():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class FallbackLineEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(text="甲A1", confidence=0.86, bbox=BBox(10, 10, 90, 24), chars=[])]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.full((100, 180, 3), 255, dtype=np.uint8)
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(20, 30, 120, 50))
        page = Page(image_path=img_path, width=180, height=100, blocks=[block])
        progress_events = []

        result = OcrPipeline(engine=FallbackLineEngine()).process_project(
            OcrProject(name="FallbackWarn", pages=[page]),
            progress_callback=progress_events.append,
        )

        assert result.pages[0].error_message == ""
        assert result.pages[0].blocks[0].lines[0].chars
        warnings = [event.message for event in progress_events if "proof fallback" in event.message]
        assert warnings == ["警告：第 1/1 页触发 proof fallback，1 行/3 字使用估算或不可用字框"]
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_emits_nonblocking_warning_when_proof_fallback_triggers PASSED")


def test_ocr_pipeline_reports_real_page_progress():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class ProgressEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            return [Line(text="进度", confidence=0.96, bbox=BBox(5, 5, 30, 12))]

    with tempfile.TemporaryDirectory() as tmpdir:
        pages = []
        for idx in range(2):
            img_path = os.path.join(tmpdir, f"p{idx}.png")
            img = np.ones((200, 300, 3), dtype=np.uint8) * 255
            cv2.imwrite(img_path, img)
            page = Page(image_path=img_path, width=300, height=200)
            page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(20, 30, 120, 60))]
            pages.append(page)

        progress_events = []
        result = OcrPipeline(engine=ProgressEngine()).process_project(
            OcrProject(name="ProgressProject", pages=pages),
            progress_callback=progress_events.append,
        )

        assert len(result.pages) == 2
        page_progress_events = [
            event for event in progress_events
            if event.message.startswith("OCR 识别中")
        ]
        warning_events = [
            event for event in progress_events
            if "proof fallback" in event.message
        ]
        assert len(page_progress_events) == 2
        assert page_progress_events[0].completed_pages == 1
        assert page_progress_events[0].current_page == 1
        assert page_progress_events[1].completed_pages == 2
        assert "第 2/2 页" in page_progress_events[1].message
        assert len(warning_events) == 2
        assert [event.completed_pages for event in warning_events] == [1, 2]

    print("test_ocr_pipeline_reports_real_page_progress PASSED")


def test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once():
    import tempfile
    import cv2
    import numpy as np
    from app.models import (
        BBox, Block, BlockOrigin, BlockType, Char, LayoutBlockSnapshot, LayoutSnapshot,
        Line, OcrPolicy, OcrProject, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.services.ocr_pipeline import OcrPipeline

    class PageOcrEngine:
        bbox_space = "crop"
        prefer_page_ocr = True

        def recognize(self, image_bgr, context):
            assert context.block_id is None
            return [
                Line(
                    text="甲",
                    confidence=0.95,
                    bbox=BBox(20, 20, 20, 20),
                    chars=[Char(char="甲", confidence=0.95, bbox=BBox(20, 20, 20, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲")],
                ),
                Line(
                    text="乙",
                    confidence=0.96,
                    bbox=BBox(140, 20, 20, 20),
                    chars=[Char(char="乙", confidence=0.96, bbox=BBox(140, 20, 20, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙")],
                ),
                Line(
                    text="丙",
                    confidence=0.97,
                    bbox=BBox(260, 120, 20, 20),
                    chars=[Char(char="丙", confidence=0.97, bbox=BBox(260, 120, 20, 20), bbox_source="ocr", bbox_granularity="char", token_text="丙")],
                ),
            ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.full((200, 320, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (40, 40), (0, 0, 0), -1)
        cv2.rectangle(img, (140, 20), (160, 40), (0, 0, 0), -1)
        cv2.rectangle(img, (260, 120), (280, 140), (0, 0, 0), -1)
        cv2.imwrite(img_path, img)

    try:
        broad = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 220, 80), order=0)
        precise = Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 60, 50), order=1)
        page = Page(image_path=img_path, width=320, height=200, blocks=[broad, precise])
        set_layout_snapshot_for_page(page, LayoutSnapshot(
            page_uid=page.uid,
            artifact_uid="artifact-page-ocr-assign",
            source_engine="test",
            source_run_id="run-page-ocr-assign",
            blocks=(
                LayoutBlockSnapshot(
                    uid=broad.uid,
                    block_type=BlockType.TEXT,
                    bbox=broad.bbox,
                    order=4,
                    source_label="text",
                    origin=BlockOrigin(source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                ),
                LayoutBlockSnapshot(
                    uid=precise.uid,
                    block_type=BlockType.TEXT,
                    bbox=precise.bbox,
                    order=8,
                    source_label="text",
                    origin=BlockOrigin(source_label="text"),
                    ocr_policy=OcrPolicy.TEXT_OCR,
                ),
            ),
        ))
        result = OcrPipeline(engine=PageOcrEngine()).process_project(
            OcrProject(name="PageOcrAssign", pages=[page])
        )
        blocks = result.pages[0].blocks
        all_texts = [line.text for block in blocks for line in block.lines]

        assert all_texts.count("甲") == 1
        assert all_texts.count("乙") == 1
        assert all_texts.count("丙") == 1
        assert [line.text for line in precise.lines] == ["甲"]
        assert [line.text for line in broad.lines] == ["乙"]
        assert blocks[-1].note == "PP-OCRv5 unmatched proof lines"
        assert [line.text for line in blocks[-1].lines] == ["丙"]
        assert blocks[-1].block_type == BlockType.TEXT
        assert blocks[-1].order == 9
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once PASSED")


def test_ocr_dispatch_policy_blocks_structural_and_paddle_skip_labels():
    from app.core.ocr_dispatch_policy import (
        is_text_ocr_candidate,
        should_dispatch_to_text_ocr,
    )
    from app.models import BBox, Block, BlockType

    bb = BBox(0, 0, 100, 20)
    footnote = Block(block_type=BlockType.TEXT, bbox=bb, source_label="vision_footnote")
    formula_label = Block(block_type=BlockType.TEXT, bbox=bb, source_label="inline_formula")
    equation = Block(block_type=BlockType.EQUATION, bbox=bb, ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA)
    table_label = Block(block_type=BlockType.TEXT, bbox=bb, source_label="table")
    unknown_formula_like = Block(block_type=BlockType.TEXT, bbox=bb, source_label="custom_formula_noise")
    unknown_table_like = Block(block_type=BlockType.TEXT, bbox=bb, source_label="not_table")
    known_display_formula = Block(block_type=BlockType.TEXT, bbox=bb, source_label="display_formula")
    text_without_structured_label = Block(block_type=BlockType.TEXT, bbox=bb)
    disabled_text = Block(block_type=BlockType.TEXT, bbox=bb, ocr_policy=OcrPolicy.MANUAL_ONLY)
    from app.core.ocr_dispatch_policy import default_ocr_policy_for_block

    formula_label.ocr_policy = default_ocr_policy_for_block(formula_label)
    table_label.ocr_policy = default_ocr_policy_for_block(table_label)
    unknown_formula_like.ocr_policy = default_ocr_policy_for_block(unknown_formula_like)
    unknown_table_like.ocr_policy = default_ocr_policy_for_block(unknown_table_like)
    known_display_formula.ocr_policy = default_ocr_policy_for_block(known_display_formula)

    assert is_text_ocr_candidate(footnote) is True
    assert should_dispatch_to_text_ocr(footnote) is True
    assert should_dispatch_to_text_ocr(formula_label) is False
    assert should_dispatch_to_text_ocr(equation) is False
    assert should_dispatch_to_text_ocr(table_label) is False
    assert should_dispatch_to_text_ocr(known_display_formula) is False
    assert is_text_ocr_candidate(unknown_formula_like) is True
    assert should_dispatch_to_text_ocr(unknown_formula_like) is True
    assert is_text_ocr_candidate(unknown_table_like) is True
    assert should_dispatch_to_text_ocr(unknown_table_like) is True
    assert should_dispatch_to_text_ocr(text_without_structured_label) is True
    assert is_text_ocr_candidate(disabled_text) is True
    assert should_dispatch_to_text_ocr(disabled_text) is False

    print("test_ocr_dispatch_policy_blocks_structural_and_paddle_skip_labels PASSED")


def test_page_ocr_refills_caption_blocks_and_preserves_equation_blocks():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    def make_line(text, x, y):
        return Line(
            text=text,
            confidence=0.95,
            bbox=BBox(x, y, 30, 20),
            chars=[
                Char(
                    char=text,
                    confidence=0.95,
                    bbox=BBox(x, y, 30, 20),
                    bbox_source="ocr",
                    bbox_granularity="char",
                    token_text=text,
                )
            ],
        )

    class PageOcrEngine:
        bbox_space = "crop"
        prefer_page_ocr = True

        def recognize(self, image_bgr, context):
            return [
                make_line("式", 20, 20),
                make_line("图", 20, 80),
                make_line("表", 20, 140),
            ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.full((220, 180, 3), 255, dtype=np.uint8)
        for y in (20, 80, 140):
            cv2.rectangle(img, (20, y), (50, y + 20), (0, 0, 0), -1)
        cv2.imwrite(img_path, img)

    try:
        equation = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 0, 120, 60),
            lines=[make_line("旧公式", 5, 5)],
            order=0,
            ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
        )
        figure_caption = Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(0, 60, 120, 60), lines=[make_line("旧", 5, 65)], order=1)
        table_caption = Block(block_type=BlockType.TABLE_CAPTION, bbox=BBox(0, 120, 120, 60), lines=[make_line("旧", 5, 125)], order=2)
        page = Page(
            image_path=img_path,
            width=180,
            height=220,
            blocks=[equation, figure_caption, table_caption],
        )

        result = OcrPipeline(engine=PageOcrEngine()).process_project(
            OcrProject(name="CaptionEquationRefill", pages=[page])
        )
        blocks = result.pages[0].blocks

        assert [line.text for line in blocks[0].lines] == ["旧公式"]
        assert [line.text for line in blocks[1].lines] == ["图"]
        assert [line.text for line in blocks[2].lines] == ["表"]
        all_texts = [line.text for block in blocks for line in block.lines]
        assert "式" not in all_texts
        assert "旧" not in all_texts
    finally:
        os.unlink(img_path)

    print("test_page_ocr_refills_caption_blocks_and_preserves_equation_blocks PASSED")


def test_page_ocr_nested_equation_blocks_before_parent_text_assignment():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.ocr_pipeline import OcrPipeline

    text = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 200, 200), order=0)
    equation = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(50, 50, 50, 30),
        order=1,
        ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
    )
    normal_line = Line(text="文", confidence=0.95, bbox=BBox(10, 10, 10, 10))
    formula_line = Line(text="式", confidence=0.95, bbox=BBox(55, 55, 10, 10))
    page = Page(
        image_path="/tmp/nested-equation.png",
        width=200,
        height=200,
        blocks=[text, equation],
    )

    OcrPipeline().assign_page_ocr_lines_to_blocks(page, [normal_line, formula_line])

    assert [line.text for line in text.lines] == ["文"]
    assert equation.lines == []
    assert len(page.blocks) == 2

    print("test_page_ocr_nested_equation_blocks_before_parent_text_assignment PASSED")


def test_hanwang_prepass_keeps_line_hint_overlapping_nested_formula_block():
    import numpy as np

    from app.core.ocr_line_hints import is_ppocr_page_line_hint
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.ocr_pipeline import OcrPipeline

    class FakePrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [
                Line(
                    text="PP-OCRv5 hint text is ignored",
                    confidence=0.95,
                    bbox=BBox.from_xyxy(0, 10, 180, 40),
                )
            ]

    text = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 190, 50), order=0)
    equation = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(50, 10, 130, 40),
        order=1,
        ocr_policy=OcrPolicy.TEXT_OCR,
    )
    page = Page(
        image_path="/tmp/hybrid-prepass-hint.png",
        width=200,
        height=80,
        blocks=[text, equation],
    )

    OcrPipeline()._process_page_with_page_ocr(
        np.ones((80, 200, 3), dtype=np.uint8) * 255,
        page,
        0,
        engine=FakePrepassEngine(),
        mark_page_line_hints=True,
    )

    assert len(text.lines) == 1
    assert text.lines[0].bbox == BBox.from_xyxy(0, 10, 180, 40)
    assert is_ppocr_page_line_hint(text.lines[0]) is True
    assert equation.lines == []
    assert len(page.blocks) == 2

    print("test_hanwang_prepass_keeps_line_hint_overlapping_nested_formula_block PASSED")


def test_ocr_pipeline_skips_equation_block_ocr_when_policy_preserves_formula():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class RaisingTextEngine:
        def recognize(self, image_bgr, context):
            raise AssertionError("equation blocks must not enter text OCR")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        cv2.imwrite(img_path, np.full((80, 160, 3), 255, dtype=np.uint8))

    try:
        formula_line = Line(text="E=mc^2", confidence=0.0, bbox=BBox(10, 10, 80, 20))
        equation = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 0, 120, 40),
            lines=[formula_line],
            ocr_policy=OcrPolicy.TEXT_OCR,
        )
        page = Page(image_path=img_path, width=160, height=80, blocks=[equation])
        result = OcrPipeline(engine=RaisingTextEngine()).process_project(
            OcrProject(name="EquationSkip", pages=[page])
        )

        assert [line.text for line in result.pages[0].blocks[0].lines] == ["E=mc^2"]
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_skips_equation_block_ocr_when_policy_preserves_formula PASSED")


def test_ocr_pipeline_avoids_double_shift_for_page_space_boxes():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class LargeCropPageCoordEngine:
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="整页坐标", confidence=0.95, bbox=BBox(120, 70, 80, 18))]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((320, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(100, 60, 300, 200))
        page = Page(image_path=img_path, width=480, height=320, blocks=[block])
        project = OcrProject(name="NoDoubleShift", pages=[page])

        result = OcrPipeline(engine=LargeCropPageCoordEngine()).process_project(project)
        line = result.pages[0].blocks[0].lines[0]

        assert line.bbox == BBox(120, 70, 80, 18)
    finally:
        os.unlink(img_path)


def test_ocr_pipeline_preserves_hanwang_crop_lines_and_chars():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class HanwangLikeCropEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            assert context.expected_bbox_space == "crop"
            assert context.crop_bbox == BBox(40, 50, 120, 60)
            return [
                Line(
                    text="汉王",
                    confidence=0.91,
                    bbox=BBox(6, 8, 42, 16),
                    chars=[
                        Char(char="汉", confidence=0.93, bbox=BBox(6, 8, 18, 16), bbox_source="hanwang:CharRcg"),
                        Char(char="王", confidence=0.89, bbox=BBox(30, 8, 18, 16), bbox_source="hanwang:CharRcg"),
                    ],
                    ocr_text="汉王",
                )
            ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((180, 240, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(40, 50, 120, 60), order=0)
        page = Page(image_path=img_path, width=240, height=180, blocks=[block])
        result = OcrPipeline(engine=HanwangLikeCropEngine()).process_project(
            OcrProject(name="HanwangSharedPipeline", pages=[page])
        )

        line = result.pages[0].blocks[0].lines[0]
        assert line.text == "汉王"
        assert line.bbox == BBox(46, 58, 42, 16)
        assert [char.char for char in line.chars] == ["汉", "王"]
        assert line.chars[0].bbox == BBox(46, 58, 18, 16)
        assert line.chars[1].bbox == BBox(70, 58, 18, 16)
        assert all(char.bbox_source.startswith("hanwang:") for char in line.chars)
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_preserves_hanwang_crop_lines_and_chars PASSED")


def test_hanwang_micro_recblock_routes_and_fallbacks():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        assert recblocks_xyxy == [(10, 20, 110, 60), (20, 140, 160, 180)]
        return {
            "lines": [
                {"groups": [{"bbox": {"left": 12, "top": 22, "right": 108, "bottom": 58}}]},
                {"groups": [{"bbox": {"left": 22, "top": 142, "right": 158, "bottom": 178}}]},
            ]
        }

    batch_shapes = []

    def fake_recog(*_args, **_kwargs):
        raise AssertionError("batch-list path should avoid per-crop recog")

    def fake_recog_batch(images_bgr, *, with_charrcg=True, timeout=0, **_kwargs):
        batch_shapes.extend(tuple(image.shape[:2]) for image in images_bgr)
        assert batch_shapes == [(56, 112), (56, 152)]
        return [
            {"lines": [{"groups": [{
                "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 40},
                "chars": [
                    {"codes": [code("天")], "scores": [5], "bbox": {"left": 2, "top": 2, "right": 25, "bottom": 38}},
                    {"codes": [code("地")], "scores": [6], "bbox": {"left": 30, "top": 2, "right": 53, "bottom": 38}},
                ],
            }]}]},
            {"lines": [{"groups": [{
                "bbox": {"left": 0, "top": 0, "right": 140, "bottom": 40},
                "chars": [
                    {"codes": [code("短")], "scores": [12], "bbox": {"left": 2, "top": 2, "right": 30, "bottom": 38}},
                ],
            }]}]},
        ]

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_recog_batch = micro_module.native_bridge.run_linecut_recog_batch_list
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    original_batch_reason = micro_module._BATCH_DISABLE_REASON
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module.native_bridge.run_linecut_recog_batch_list = fake_recog_batch
    micro_module._BATCH_DISABLED_FOR_SESSION = False
    micro_module._BATCH_DISABLE_REASON = ""

    try:
        blocks = [
            {"block_label": "text", "block_bbox": [10, 20, 110, 60], "block_content": "天地"},
            {"block_label": "display_formula", "block_bbox": [10, 80, 180, 120], "block_content": "$$x+y$$"},
            {"block_label": "reference", "block_bbox": [20, 140, 160, 180], "block_content": "参考文献很长"},
        ]
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((220, 240, 3), dtype=np.uint8),
            blocks,
            include_chars=True,
        )

        assert [row.source for row in rows] == ["hanwang", "ppvl", "hanwang"]
        assert rows[0].text == "天地"
        assert rows[0].lines[0].chars[0].text == "天"
        assert rows[0].lines[0].chars[0].bbox == (6, 14, 29, 50)
        assert rows[1].text == "$$x+y$$"
        assert rows[2].text == "短"
        assert rows[2].fallback_reason == ""
        assert stats.n_blocks_hanwang == 2
        assert stats.n_blocks_ppvl == 1
        assert stats.n_blocks_fallback == 0
        assert batch_shapes == [(56, 112), (56, 152)]
        assert stats.recog_full_page_pixels == 220 * 240 * 2
        assert stats.recog_crop_pixels == 56 * 112 + 56 * 152
        assert stats.recog_probe_calls == 1
        assert stats.recog_batch_chunks == 1
        assert stats.recog_batch_failures == 0
        assert stats.recog_batch_disabled is False
        assert stats.recog_max_batch_crop_width == 152
        assert stats.recog_max_batch_crop_height == 56
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module.native_bridge.run_linecut_recog_batch_list = original_recog_batch
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled
        micro_module._BATCH_DISABLE_REASON = original_batch_reason

    print("test_hanwang_micro_recblock_routes_and_fallbacks PASSED")


def test_hanwang_engine_uses_user_edited_layout_for_manual_formula_boxes():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module
    from app.models import BBox, Block, BlockSource, BlockType, Page

    captured = {}

    def fake_runner(
        image_bgr,
        ppvl_blocks,
        *,
        seg_timeout=0,
        recog_timeout=0,
        include_chars=True,
        page_ocr_lines=None,
        progress_callback=None,
    ):
        captured["blocks"] = ppvl_blocks
        row = micro_module.BlockResult(
            block_idx=0,
            block_label="equation",
            block_bbox=(20, 30, 80, 54),
            source="ppvl",
            text="",
            ppvl_text="",
            lines=[
                micro_module.LineResult(
                    text="",
                    bbox=(20, 30, 80, 54),
                    confidence=0.0,
                    source="ppvl",
                )
            ],
            raw_block=dict(ppvl_blocks[0]),
        )
        return [row], micro_module.RunStats(n_blocks_total=1, n_blocks_ppvl=1)

    page = Page(image_path="/tmp/manual-formula.png", width=120, height=90, page_number=1)
    _attach_raw_layout_records(page, [
        {"block_label": "text", "block_bbox": [0, 0, 100, 20], "block_content": "stale paddle text"}
    ])
    page.blocks = [
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(20, 30, 60, 24),
            source=BlockSource.MANUAL_DRAW,
        )
    ]

    engine = micro_module.HanwangMicroRecBlockEngine(runner=fake_runner)
    engine.recognize_page_blocks(np.zeros((90, 120, 3), dtype=np.uint8), page)

    assert captured["blocks"][0]["block_label"] == "equation"
    assert captured["blocks"][0]["block_bbox"] == [20, 30, 80, 54]
    assert captured["blocks"][0]["_layout_block_source"] == "manual_draw"
    assert "stale paddle text" not in str(captured["blocks"][0])
    assert len(page.blocks) == 1
    assert page.blocks[0].block_type == BlockType.EQUATION
    assert page.blocks[0].ocr_policy != OcrPolicy.TEXT_OCR
    assert len(page.blocks[0].lines) == 1
    assert page.blocks[0].lines[0].text == ""
    assert "manual_formula_needs_text" in page.blocks[0].lines[0].review_flags

    print("test_hanwang_engine_uses_user_edited_layout_for_manual_formula_boxes PASSED")


def test_hanwang_layout_injects_manual_formula_binding_into_parent_route():
    from app.core.paddle_artifact_index import BINDING_PARENT_FORMULA_INFERRED
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block, text_slice_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page

    parent_record = {
        "block_label": "text",
        "block_bbox": [0, 0, 200, 40],
        "block_content": "甲 $ A $ 乙 $ B $ 丙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
        ],
        "_layout_line_routes": [
            {"bbox": [0, 0, 200, 40], "segments": [{"kind": "text", "bbox": [0, 0, 200, 40]}]},
        ],
    }
    page = Page(
        image_path="/tmp/manual-binding-route.png",
        width=220,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([dict(parent_record)]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 200, 40),
                lines=[Line(text="bad active ocr", confidence=0.0, bbox=BBox.from_xyxy(0, 0, 200, 40))],
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(110, 0, 140, 30),
                source=BlockSource.MANUAL_DRAW,
                source_label="inline_formula",
                paddle_binding=PaddleBinding.from_dict(
                    {
                        "status": BINDING_PARENT_FORMULA_INFERRED,
                        "block_type": "equation",
                        "source_label": "inline_formula",
                        "text": "$ B $",
                        "parent_index": 0,
                        "manual_bbox": [110, 0, 140, 30],
                    }
                ),
            ),
        ],
    )

    blocks = _page_blocks_from_layout(page)
    assert len(blocks) == 1
    parent = blocks[0]
    assert parent["block_content"] == "甲 $ A $ 乙 $ B $ 丙"
    assert "_layout_line_routes" not in parent
    assert [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]] == [
        [40, 0, 70, 30],
        [110, 0, 140, 30],
    ]

    routes = line_routes_for_block(parent, 220, 60)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    assert [segment["text"] for segment in formula_segments] == ["", "$ B $"]
    assert [segment["bbox"] for segment in formula_segments] == [
        [40, 0, 70, 30],
        [110, 0, 140, 30],
    ]
    assert [route["bbox"] for route in text_slice_routes_for_block(parent, 220, 60)] == [
        [0, 0, 40, 30],
        [70, 0, 110, 30],
        [140, 0, 200, 30],
    ]

    print("test_hanwang_layout_injects_manual_formula_binding_into_parent_route PASSED")


def test_hanwang_manual_formula_candidate_does_not_replace_manual_sibling_route():
    from app.core.paddle_artifact_index import BINDING_GEOMETRY_HIT
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page

    parent_text = "其中， $ GGF_{it}^{Post-short} $、 $ GGF_{it}^{Post-long} $ 均为虚拟变量， $ GGF_{it}^{Post-short} $ 在企业获得政府引导基金"
    page = Page(
        image_path="/tmp/manual-formula-sibling-route.png",
        width=2200,
        height=900,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [200, 550, 2050, 850],
                "block_content": parent_text,
                ROUTE_SUBBLOCKS_FIELD: [
                    {
                        "block_label": "inline_formula",
                        "block_bbox": [445, 556, 653, 620],
                        "block_content": "$ GGF_{it}^{Post-short} $",
                        "_layout_manual_route_subblock": True,
                    },
                    {
                        "block_label": "inline_formula",
                        "block_bbox": [1242, 562, 1453, 620],
                    },
                    {
                        "block_label": "inline_formula",
                        "block_bbox": [686, 565, 884, 614],
                        "_layout_manual_route_subblock": True,
                        "_layout_manual_unbound_route_subblock": True,
                    },
                ],
            },
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(200, 550, 2050, 850),
                lines=[Line(text="stale", confidence=0.0, bbox=BBox.from_xyxy(200, 550, 2050, 620))],
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(445, 556, 653, 620),
                source=BlockSource.USER_EDITED,
                source_label="inline_formula",
                paddle_binding=PaddleBinding.from_dict(
                    {
                        "status": BINDING_GEOMETRY_HIT,
                        "block_type": "equation",
                        "source_label": "inline_formula",
                        "text": "$ GGF_{it}^{Post-short} $",
                        "parent_index": 0,
                        "candidate_bbox": [445, 556, 884, 620],
                        "manual_bbox": [445, 556, 653, 620],
                    }
                ),
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(686, 565, 884, 614),
                source=BlockSource.USER_EDITED,
                source_label="inline_formula",
            ),
        ],
    )

    parent = _page_blocks_from_layout(page)[0]
    sub_bboxes = [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]]
    manual_left = next(sub for sub in parent[ROUTE_SUBBLOCKS_FIELD] if sub["block_bbox"] == [445, 556, 653, 620])

    assert sub_bboxes.count([445, 556, 653, 620]) == 1
    assert [686, 565, 884, 614] in sub_bboxes
    assert manual_left["block_content"] == ""
    assert manual_left["_layout_manual_binding_text_stale"] is True

    formula_segments = [
        segment
        for route in line_routes_for_block(parent, page.width, page.height)
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]

    assert [segment["bbox"] for segment in formula_segments] == [
        [445, 556, 653, 620],
        [686, 565, 884, 614],
        [1242, 562, 1453, 620],
    ]
    assert [segment["text"] for segment in formula_segments] == ["", "", ""]

    print("test_hanwang_manual_formula_candidate_does_not_replace_manual_sibling_route PASSED")


def test_hanwang_manual_formula_child_cannot_steal_parent_route_index():
    from app.core.paddle_artifact_index import BINDING_AMBIGUOUS, BINDING_GEOMETRY_HIT
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, PaddleBinding, Page

    parent_text = "其中， $ GGF_{it}^{Post-short} $、 $ GGF_{it}^{Post-long} $ 均为虚拟变量， $ GGF_{it}^{Post-short} $ 在企业获得政府引导基金"
    parent_record = {
        "block_label": "text",
        "block_bbox": [200, 550, 2050, 850],
        "block_content": parent_text,
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [445, 556, 884, 620]},
            {"block_label": "inline_formula", "block_bbox": [1242, 562, 1453, 620]},
        ],
    }
    page = Page(
        image_path="/tmp/manual-formula-parent-steal.png",
        width=2200,
        height=900,
        raw_layout_artifact=_paddle_layout_artifact([parent_record]),
        blocks=[
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(680, 562, 883, 613),
                source=BlockSource.USER_EDITED,
                source_label="inline_formula",
                paddle_binding=PaddleBinding.from_dict(
                    {
                        "status": BINDING_AMBIGUOUS,
                        "block_type": "equation",
                        "source_label": "inline_formula",
                        "parent_index": 0,
                        "manual_bbox": [680, 562, 883, 613],
                    }
                ),
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(200, 550, 2050, 850),
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(444, 556, 653, 620),
                source=BlockSource.USER_EDITED,
                source_label="inline_formula",
                paddle_binding=PaddleBinding.from_dict(
                    {
                        "status": BINDING_GEOMETRY_HIT,
                        "block_type": "equation",
                        "source_label": "inline_formula",
                        "text": "$ GGF_{it}^{Post-short} $",
                        "parent_index": 0,
                        "candidate_index": 0,
                        "candidate_bbox": [445, 556, 884, 620],
                        "manual_bbox": [444, 556, 653, 620],
                    }
                ),
            ),
        ],
    )

    parent = _page_blocks_from_layout(page)[0]
    sub_bboxes = [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]]

    assert [445, 556, 884, 620] not in sub_bboxes
    assert [444, 556, 653, 620] in sub_bboxes
    assert [680, 562, 883, 613] in sub_bboxes
    formula_segments = [
        segment
        for route in line_routes_for_block(parent, page.width, page.height)
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    assert [segment["bbox"] for segment in formula_segments] == [
        [444, 556, 653, 620],
        [680, 562, 883, 613],
        [1242, 562, 1453, 620],
    ]
    assert [segment["text"] for segment in formula_segments] == ["", "", ""]

    print("test_hanwang_manual_formula_child_cannot_steal_parent_route_index PASSED")


def test_hanwang_layout_routes_use_raw_parent_formula_text_not_stale_ocr_text():
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, Page

    page = Page(
        image_path="/tmp/raw-parent-formula-text.png",
        width=260,
        height=80,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 240, 40],
                "block_content": "甲 $ A $ 乙 $ B $ 丙",
                ROUTE_SUBBLOCKS_FIELD: [
                    {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
                    {"block_label": "inline_formula", "block_bbox": [120, 0, 150, 30]},
                ],
            },
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 240, 40),
                lines=[
                    Line(
                        text="甲乙丙",
                        confidence=0.9,
                        bbox=BBox.from_xyxy(0, 0, 240, 40),
                    )
                ],
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
        ],
    )

    parent = _page_blocks_from_layout(page)[0]
    assert parent["block_content"] == "甲 $ A $ 乙 $ B $ 丙"

    formula_segments = [
        segment
        for route in line_routes_for_block(parent, page.width, page.height)
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]

    assert [segment["text"] for segment in formula_segments] == ["$ A $", "$ B $"]

    print("test_hanwang_layout_routes_use_raw_parent_formula_text_not_stale_ocr_text PASSED")


def test_hanwang_layout_injects_unbound_manual_formula_into_parent_route():
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block, text_slice_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page

    parent_record = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 40],
        "block_content": "甲 $ A $ 乙 $ B $ 丙",
        ROUTE_SUBBLOCKS_FIELD: [
            {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
        ],
        "_layout_line_routes": [
            {"bbox": [0, 0, 220, 40], "segments": [{"kind": "text", "bbox": [0, 0, 220, 40]}]},
        ],
    }
    page = Page(
        image_path="/tmp/manual-unbound-route.png",
        width=240,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([dict(parent_record)]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 220, 40),
                lines=[Line(text="bad active ocr", confidence=0.0, bbox=BBox.from_xyxy(0, 0, 220, 40))],
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(150, 0, 180, 30),
                source=BlockSource.MANUAL_DRAW,
                source_label="inline_formula",
            ),
        ],
    )

    blocks = _page_blocks_from_layout(page)
    assert len(blocks) == 1
    parent = blocks[0]
    assert "_layout_line_routes" not in parent
    assert [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]] == [
        [40, 0, 70, 30],
        [150, 0, 180, 30],
    ]
    assert parent[ROUTE_SUBBLOCKS_FIELD][1]["_layout_manual_unbound_route_subblock"] is True

    routes = line_routes_for_block(parent, 240, 60)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    assert [segment["bbox"] for segment in formula_segments] == [
        [40, 0, 70, 30],
        [150, 0, 180, 30],
    ]
    assert [route["bbox"] for route in text_slice_routes_for_block(parent, 240, 60)] == [
        [0, 0, 40, 30],
        [70, 0, 150, 30],
        [180, 0, 220, 30],
    ]

    print("test_hanwang_layout_injects_unbound_manual_formula_into_parent_route PASSED")


def test_hanwang_recognize_preserves_parent_bound_manual_formula_block():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module
    from app.core.paddle_artifact_index import BINDING_PARENT_FORMULA_INFERRED
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page

    captured = {}

    def fake_runner(
        image_bgr,
        ppvl_blocks,
        *,
        seg_timeout=0,
        recog_timeout=0,
        include_chars=True,
        page_ocr_lines=None,
        progress_callback=None,
    ):
        captured["blocks"] = ppvl_blocks
        return [
            micro_module.BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(0, 0, 200, 40),
                source="hanwang",
                text="甲乙丙",
                ppvl_text="甲 $ B $ 乙",
                raw_block=dict(ppvl_blocks[0]),
                lines=[
                    micro_module.LineResult(
                        text="甲乙丙",
                        bbox=(0, 0, 200, 40),
                        confidence=0.9,
                    )
                ],
            )
        ], micro_module.RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    manual_formula = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(110, 0, 140, 30),
        source=BlockSource.MANUAL_DRAW,
        source_label="inline_formula",
        lines=[Line(text="$ B $", confidence=0.0, bbox=BBox.from_xyxy(110, 0, 140, 30))],
        paddle_binding=PaddleBinding.from_dict(
            {
                "status": BINDING_PARENT_FORMULA_INFERRED,
                "block_type": "equation",
                "source_label": "inline_formula",
                "text": "$ B $",
                "parent_index": 0,
                "manual_bbox": [110, 0, 140, 30],
            }
        ),
    )
    page = Page(
        image_path="/tmp/manual-formula-preserve.png",
        width=220,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 200, 40],
                "block_content": "甲 $ B $ 乙",
                ROUTE_SUBBLOCKS_FIELD: [],
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 200, 40),
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            manual_formula,
        ],
    )

    micro_module.HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((60, 220, 3), dtype=np.uint8),
        page,
    )

    assert len(captured["blocks"]) == 1
    assert captured["blocks"][0][ROUTE_SUBBLOCKS_FIELD][0]["block_bbox"] == [110, 0, 140, 30]
    assert [block.block_type for block in page.blocks] == [BlockType.TEXT, BlockType.EQUATION]
    assert page.blocks[1] is manual_formula
    assert page.blocks[1].source == BlockSource.MANUAL_DRAW
    assert page.blocks[1].paddle_binding is not None
    assert page.blocks[1].paddle_binding.text == "$ B $"

    print("test_hanwang_recognize_preserves_parent_bound_manual_formula_block PASSED")


def test_hanwang_recognize_rebinds_manual_formula_text_from_paddle_crop_ocr():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module
    from app.core.paddle_artifact_index import BINDING_FORMULA_CROP_OCR, BINDING_GEOMETRY_HIT
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, PaddleBinding, Page

    captured = {}

    class FakeFormulaClient:
        def __init__(self):
            self.calls = 0

        def analyze_image_bytes(self, image_bytes, *, optional_payload=None, batch_id="", filename="page.png"):
            self.calls += 1
            assert image_bytes
            return {
                "result": {
                    "layoutParsingResults": [
                        {
                            "block_label": "inline_formula",
                            "block_content": "$ B_{new} $",
                            "block_bbox": [24, 24, 70, 72],
                        }
                    ]
                }
            }

    def fake_runner(
        image_bgr,
        ppvl_blocks,
        *,
        seg_timeout=0,
        recog_timeout=0,
        include_chars=True,
        page_ocr_lines=None,
        progress_callback=None,
    ):
        captured["blocks"] = ppvl_blocks
        return [
            micro_module.BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(0, 0, 200, 40),
                source="hanwang",
                text="甲乙",
                ppvl_text="甲 $ B $ 乙",
                raw_block=dict(ppvl_blocks[0]),
                lines=[
                    micro_module.LineResult(
                        text="甲乙",
                        bbox=(0, 0, 200, 40),
                        confidence=0.9,
                    )
                ],
            )
        ], micro_module.RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    formula = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(110, 0, 140, 30),
        source=BlockSource.USER_EDITED,
        source_label="inline_formula",
        lines=[Line(text="$ B_{old} $", confidence=0.0, bbox=BBox.from_xyxy(110, 0, 140, 30))],
        ocr_invalidated_reason="layout_changed",
        paddle_binding=PaddleBinding.from_dict(
            {
                "status": BINDING_GEOMETRY_HIT,
                "block_type": "equation",
                "source_label": "inline_formula",
                "text": "$ B_{old} $",
                "parent_index": 0,
                "candidate_bbox": [100, 0, 150, 30],
                "manual_bbox": [110, 0, 140, 30],
            }
        ),
    )
    page = Page(
        image_path="/tmp/manual-formula-crop-ocr.png",
        width=220,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 200, 40],
                "block_content": "甲 $ B $ 乙",
                ROUTE_SUBBLOCKS_FIELD: [],
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 200, 40),
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            formula,
        ],
    )

    client = FakeFormulaClient()
    micro_module.HanwangMicroRecBlockEngine(
        runner=fake_runner,
        formula_rebind_client=client,
    ).recognize_page_blocks(
        np.zeros((60, 220, 3), dtype=np.uint8),
        page,
    )

    assert client.calls == 1
    subblocks = captured["blocks"][0][ROUTE_SUBBLOCKS_FIELD]
    assert subblocks[0]["block_bbox"] == [110, 0, 140, 30]
    assert subblocks[0]["block_content"] == "$ B_{new} $"
    assert formula.lines[0].text == "$ B_{new} $"
    assert formula.paddle_binding is not None
    assert formula.paddle_binding.status == BINDING_FORMULA_CROP_OCR
    assert formula.paddle_binding.text == "$ B_{new} $"

    print("test_hanwang_recognize_rebinds_manual_formula_text_from_paddle_crop_ocr PASSED")


def test_hanwang_recognize_preserves_parent_unbound_manual_formula_block():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.models import BBox, Block, BlockOrigin, BlockSource, BlockType, Line, Page

    captured = {}
    manual_formula = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(150, 0, 180, 30),
        source=BlockSource.MANUAL_DRAW,
        source_label="inline_formula",
    )

    def fake_runner(
        image_bgr,
        ppvl_blocks,
        *,
        seg_timeout=0,
        recog_timeout=0,
        include_chars=True,
        page_ocr_lines=None,
        progress_callback=None,
    ):
        captured["blocks"] = ppvl_blocks
        return [
            micro_module.BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(0, 0, 220, 40),
                source="hanwang",
                text="甲乙丙",
                ppvl_text="甲 $ A $ 乙 $ B $ 丙",
                raw_block=dict(ppvl_blocks[0]),
                lines=[
                    micro_module.LineResult(
                        text="甲乙丙",
                        bbox=(0, 0, 220, 40),
                        confidence=0.9,
                    )
                ],
            )
        ], micro_module.RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    page = Page(
        image_path="/tmp/manual-unbound-preserve.png",
        width=240,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 220, 40],
                "block_content": "甲 $ A $ 乙 $ B $ 丙",
                ROUTE_SUBBLOCKS_FIELD: [
                    {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 30]},
                ],
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 220, 40),
                origin=BlockOrigin(source_label="text", raw_index=0),
            ),
            manual_formula,
        ],
    )

    micro_module.HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((60, 240, 3), dtype=np.uint8),
        page,
    )

    assert len(captured["blocks"]) == 1
    assert [sub["block_bbox"] for sub in captured["blocks"][0][ROUTE_SUBBLOCKS_FIELD]] == [
        [40, 0, 70, 30],
        [150, 0, 180, 30],
    ]
    assert [block.block_type for block in page.blocks] == [BlockType.TEXT, BlockType.EQUATION]
    assert page.blocks[1] is manual_formula
    assert page.blocks[1].source == BlockSource.MANUAL_DRAW
    assert page.blocks[1].source_label == "inline_formula"

    print("test_hanwang_recognize_preserves_parent_unbound_manual_formula_block PASSED")


def test_hanwang_inline_formula_text_slices_keep_chars():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {
                    "groups": (
                        [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]
                        if y2 - y1 >= 20 else []
                    )
                }
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        chars_by_crop = {
            (78, 40): [
                ("甲", (0, 0, 20, 30)),
                ("甲", (24, 0, 44, 30)),
            ],
            (116, 40): [
                ("乙", (10, 0, 30, 30)),
                ("乙", (34, 0, 54, 30)),
                ("。", (86, 18, 96, 30)),
            ],
            (48, 50): [
                ("丙", (0, 10, 20, 40)),
            ],
            (66, 50): [
                ("丁", (10, 10, 30, 40)),
            ],
            (46, 50): [
                ("戊", (10, 10, 30, 40)),
            ],
        }
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        lines = []
        for x1, y1, x2, y2 in blocks:
            specs = chars_by_crop.get((x2 - x1, y2 - y1), [])
            if not specs:
                continue
            chars = []
            for ch, (left, top, right, bottom) in specs:
                chars.append({
                    "codes": [code(ch)],
                    "scores": [5],
                    "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                })
            lines.append({"groups": [{
                "bbox": {"left": 0, "top": 0, "right": x2 - x1, "bottom": y2 - y1},
                "chars": chars,
            }]})
        return {
            "lines": lines
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        blocks = [{
            "block_label": "text",
            "block_bbox": [0, 0, 210, 80],
            "block_content": "甲甲 $ Y_{ct} $乙乙。丙 $ Incentive_{c} $ 丁 $ Post_{t} $戊",
            "_route_subblocks": [
                {
                    "block_label": "inline_formula",
                    "block_bbox": [70, 0, 110, 30],
                    "raw_payload": {"block_label": "inline_formula"},
                },
                {
                    "block_label": "inline_formula",
                    "block_bbox": [40, 40, 90, 70],
                    "raw_payload": {"block_label": "inline_formula"},
                },
                {
                    "block_label": "inline_formula",
                    "block_bbox": [140, 40, 180, 70],
                    "raw_payload": {"block_label": "inline_formula"},
                },
            ],
        }]
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((100, 220, 3), dtype=np.uint8),
            blocks,
            include_chars=True,
        )

        assert seen_recblocks == [
            (0, 0, 70, 30),
            (110, 0, 210, 30),
            (0, 40, 40, 70),
            (90, 40, 140, 70),
            (180, 40, 210, 70),
        ]
        assert len(rows) == len(blocks)
        assert [row.source for row in rows] == ["hanwang"]
        assert rows[0].block_label == "text"
        assert rows[0].raw_block["_route_subblocks"][0]["block_label"] == "inline_formula"
        assert [line.text for line in rows[0].lines] == [
            "甲甲$ Y_{ct} $乙乙。",
            "丙$ Incentive_{c} $丁$ Post_{t} $戊",
        ]
        assert [char.text for char in rows[0].lines[0].chars] == ["甲", "甲", "$ Y_{ct} $", "乙", "乙", "。"]
        assert [char.text for char in rows[0].lines[1].chars] == [
            "丙",
            "$ Incentive_{c} $",
            "丁",
            "$ Post_{t} $",
            "戊",
        ]
        formula_chars = [
            char
            for line in rows[0].lines
            for char in line.chars
            if char.source == "paddle_inline_formula"
        ]
        assert [char.text for char in formula_chars] == [
            "$ Y_{ct} $",
            "$ Incentive_{c} $",
            "$ Post_{t} $",
        ]
        assert [char.bbox for char in formula_chars] == [
            (70, 0, 110, 30),
            (40, 40, 90, 70),
            (140, 40, 180, 70),
        ]
        assert all(char.bbox_granularity == "word" for char in formula_chars)
        assert all(char.token_text == char.text for char in formula_chars)
        assert not any(
            char.text.startswith("$") and char.source.startswith("hanwang:")
            for line in rows[0].lines
            for char in line.chars
        )
        assert all(
            micro_module.ROUTE_INLINE_FORMULA_FLAG in line.review_flags
            for line in rows[0].lines
        )
        assert stats.n_blocks_total == 1
        assert stats.n_blocks_hanwang == 1
        assert stats.n_blocks_ppvl == 0
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_inline_formula_text_slices_keep_chars PASSED")


def test_hanwang_recog_filters_empty_decoded_char_boxes():
    from app.engines.hanwang.micro_recblock import _line_results_from_recog

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    raw = {
        "lines": [
            {
                "groups": [
                    {
                        "bbox": {"left": 0, "top": 0, "right": 90, "bottom": 30},
                        "chars": [
                            {
                                "codes": [code("甲")],
                                "scores": [5],
                                "bbox": {"left": 0, "top": 0, "right": 30, "bottom": 30},
                            },
                            {
                                "codes": [0],
                                "scores": [100],
                                "bbox": {"left": 30, "top": 0, "right": 60, "bottom": 30},
                            },
                            {
                                "codes": [code("乙")],
                                "scores": [5],
                                "bbox": {"left": 60, "top": 0, "right": 90, "bottom": 30},
                            },
                        ],
                    }
                ]
            }
        ]
    }

    lines = _line_results_from_recog(
        raw,
        fallback_bbox=(0, 0, 90, 30),
        include_chars=True,
    )

    assert len(lines) == 1
    assert lines[0].text == "甲乙"
    assert [char.text for char in lines[0].chars] == ["甲", "乙"]
    assert [char.bbox for char in lines[0].chars] == [(0, 0, 30, 30), (60, 0, 90, 30)]

    print("test_hanwang_recog_filters_empty_decoded_char_boxes PASSED")


def test_hanwang_latin_engcut_updates_geometry_without_changing_text():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {
                                "left": 0,
                                "top": 0,
                                "right": image_bgr.shape[1],
                                "bottom": image_bgr.shape[0],
                            }
                        }
                    ]
                }
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        text = "甲PE/VC乙"
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {
                                "left": 0,
                                "top": 0,
                                "right": image_bgr.shape[1],
                                "bottom": image_bgr.shape[0],
                            },
                            "chars": [
                                {
                                    "codes": [code(ch)],
                                    "scores": [5],
                                    "bbox": {
                                        "left": idx * 20,
                                        "top": 0,
                                        "right": idx * 20 + 18,
                                        "bottom": 30,
                                    },
                                }
                                for idx, ch in enumerate(text)
                            ],
                        }
                    ]
                }
            ]
        }

    eng20_calls = []

    def fake_eng20(image_bgr, *, timeout=0):
        eng20_calls.append(image_bgr.shape[:2])
        text = "~PE/VC~"
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "chars": [
                                {
                                    "codes": [ord(ch)],
                                    "bbox": {
                                        "left": idx * 9,
                                        "top": 2,
                                        "right": idx * 9 + 7,
                                        "bottom": 22,
                                    },
                                }
                                for idx, ch in enumerate(text)
                            ]
                        }
                    ]
                }
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module.native_bridge.run_eng20_recogline = fake_eng20

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((40, 160, 3), dtype=np.uint8),
            [{
                "block_label": "text",
                "block_bbox": [0, 0, 160, 40],
                "block_content": "甲PE/VC乙",
            }],
        )

        line = rows[0].lines[0]
        assert line.text == "甲PE/VC乙"
        assert [char.text for char in line.chars] == list("甲PE/VC乙")
        assert eng20_calls == [(40, 160)]
        assert stats.latin_engcut_probe_calls == 1
        assert stats.latin_engcut_probe_failures == 0
        assert stats.latin_engcut_exact_tokens == 1
        assert stats.latin_engcut_review_tokens == 0
        assert micro_module.LATIN_ENGCUT_REVIEW_FLAG not in line.review_flags
        assert [
            (char.text, char.source, char.bbox, char.bbox_granularity, char.token_text)
            for char in line.chars[1:6]
        ] == [
            ("P", "hanwang:EngCut:latin_exact", (9, 2, 16, 22), "char", "PE/VC"),
            ("E", "hanwang:EngCut:latin_exact", (18, 2, 25, 22), "char", "PE/VC"),
            ("/", "hanwang:EngCut:latin_exact", (27, 2, 34, 22), "char", "PE/VC"),
            ("V", "hanwang:EngCut:latin_exact", (36, 2, 43, 22), "char", "PE/VC"),
            ("C", "hanwang:EngCut:latin_exact", (45, 2, 52, 22), "char", "PE/VC"),
        ]
        assert line.chars[0].source == "hanwang:micro_recblock"
        assert line.chars[-1].source == "hanwang:micro_recblock"
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_updates_geometry_without_changing_text PASSED")


def test_hanwang_latin_engcut_rejects_engcut_stream_span_for_line_chars():
    import app.engines.hanwang.micro_recblock as micro_module

    def chars_for(text):
        return [
            micro_module.CharResult(
                text=ch,
                confidence=0.9,
                bbox=(idx * 10, 0, idx * 10 + 8, 20),
            )
            for idx, ch in enumerate(text)
        ]

    line = micro_module.LineResult(
        text="综合2016年增值税",
        bbox=(0, 0, 130, 20),
        confidence=0.9,
        chars=chars_for("综合2016年增值税"),
    )
    record = micro_module._EngcutLine(
        line=line,
        bbox=(0, 0, 130, 20),
        chars=[],
        text="",
        order=0,
    )
    binding = micro_module._TokenBinding(
        token=micro_module.LatinToken(text="2016", start=0, end=4),
        status=micro_module.LATIN_ENGCUT_EXACT_STATUS,
        matched_text="2016",
        chars=[
            micro_module.EngcutChar(text=ch, bbox=((idx + 2) * 10, 1, (idx + 2) * 10 + 7, 19))
            for idx, ch in enumerate("2016")
        ],
        line_records=[record],
        line_span=(0, 4),
    )

    assert micro_module._replace_line_span_with_binding(line, binding)
    assert line.text == "综合2016年增值税"
    assert [char.source for char in line.chars[:2]] == ["hanwang:micro_recblock"] * 2
    assert [char.source for char in line.chars[2:6]] == ["hanwang:EngCut:latin_exact"] * 4
    assert [char.text for char in line.chars[2:6]] == list("2016")
    assert [char.source for char in line.chars[6:]] == ["hanwang:micro_recblock"] * 4

    print("test_hanwang_latin_engcut_rejects_engcut_stream_span_for_line_chars PASSED")


def test_hanwang_latin_engcut_failure_is_line_local():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def _line(text, y):
        return micro_module.LineResult(
            text=text,
            bbox=(0, y, 80, y + 30),
            chars=[
                micro_module.CharResult(
                    text=ch,
                    bbox=(idx * 20, y, idx * 20 + 18, y + 30),
                )
                for idx, ch in enumerate(text)
            ],
        )

    calls = []

    def fake_eng20(image_bgr, *, timeout=0):
        calls.append(image_bgr.shape[:2])
        if len(calls) == 1:
            raise RuntimeError("bad line crop")
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "chars": [
                                {
                                    "codes": [ord(ch)],
                                    "bbox": {
                                        "left": idx * 9,
                                        "top": 2,
                                        "right": idx * 9 + 7,
                                        "bottom": 22,
                                    },
                                }
                                for idx, ch in enumerate("~CD~")
                            ]
                        }
                    ]
                }
            ]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20

    try:
        stats = micro_module.RunStats()
        first = _line("AB", 0)
        second = _line("CD", 40)
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((80, 100, 3), dtype=np.uint8),
            [first, second],
            stats,
            timeout=1.0,
        )

        assert calls == [(32, 82), (34, 82)]
        assert stats.latin_engcut_probe_calls == 2
        assert stats.latin_engcut_probe_failures == 1
        assert stats.latin_engcut_exact_tokens == 1
        assert [char.source for char in first.chars] == ["hanwang:micro_recblock", "hanwang:micro_recblock"]
        assert [
            (char.text, char.source, char.bbox, char.token_text)
            for char in second.chars
        ] == [
            ("C", "hanwang:EngCut:latin_exact", (9, 40, 16, 60), "CD"),
            ("D", "hanwang:EngCut:latin_exact", (18, 40, 25, 60), "CD"),
        ]
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_failure_is_line_local PASSED")


def test_hanwang_latin_engcut_targets_token_lines_without_chinese_only_probe():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    lines = [
        micro_module.LineResult(
            text="纯中文",
            bbox=(0, 0, 80, 30),
            chars=[
                micro_module.CharResult(text=ch, bbox=(idx * 18, 0, idx * 18 + 16, 26))
                for idx, ch in enumerate("纯中文")
            ],
        ),
        micro_module.LineResult(
            text="PE/VC",
            bbox=(0, 40, 90, 72),
            chars=[
                micro_module.CharResult(text=ch, bbox=(idx * 14, 40, idx * 14 + 10, 68))
                for idx, ch in enumerate("PE/VC")
            ],
        ),
        micro_module.LineResult(
            text="2026",
            bbox=(0, 80, 90, 112),
            chars=[
                micro_module.CharResult(text=ch, bbox=(idx * 14, 80, idx * 14 + 10, 108))
                for idx, ch in enumerate("2026")
            ],
        ),
    ]
    engcut_texts = ["PE/VC", "2026"]
    calls = []

    def fake_eng20(image_bgr, *, timeout=0):
        calls.append(image_bgr.shape[:2])
        text = engcut_texts.pop(0)
        return {
            "lines": [{
                "groups": [{
                    "chars": [
                        {
                            "codes": [ord(ch)],
                            "bbox": {
                                "left": idx * 14,
                                "top": 2,
                                "right": idx * 14 + 10,
                                "bottom": 28,
                            },
                        }
                        for idx, ch in enumerate(text)
                    ]
                }]
            }]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        stats = micro_module.RunStats()
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((120, 120, 3), dtype=np.uint8),
            lines,
            stats,
            timeout=1.0,
            block_text="纯中文 PE/VC 2026",
        )

        assert calls == [(36, 92), (36, 92)]
        assert stats.latin_engcut_probe_calls == 2
        assert [char.source for char in lines[0].chars] == ["hanwang:micro_recblock"] * 3
        assert [char.source for char in lines[1].chars] == ["hanwang:EngCut:latin_exact"] * 5
        assert [char.source for char in lines[2].chars] == ["hanwang:EngCut:latin_exact"] * 4
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_targets_token_lines_without_chinese_only_probe PASSED")


def test_hanwang_latin_engcut_uses_paddle_token_to_repair_bad_span():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    line = micro_module.LineResult(
        text="Gua吨lia",
        bbox=(0, 0, 120, 32),
        chars=[
            micro_module.CharResult(text="G", bbox=(0, 0, 10, 28)),
            micro_module.CharResult(text="u", bbox=(12, 8, 22, 28)),
            micro_module.CharResult(text="a", bbox=(24, 8, 34, 28)),
            micro_module.CharResult(text="吨", confidence=0.19, bbox=(36, 0, 70, 32)),
            micro_module.CharResult(text="l", bbox=(72, 0, 82, 28)),
            micro_module.CharResult(text="i", bbox=(84, 0, 94, 28)),
            micro_module.CharResult(text="a", bbox=(96, 8, 106, 28)),
        ],
    )

    def fake_eng20(image_bgr, *, timeout=0):
        return {
            "lines": [{
                "groups": [{
                    "chars": [
                        {
                            "codes": [ord(ch)],
                            "bbox": {
                                "left": idx * 12,
                                "top": 2,
                                "right": idx * 12 + 10,
                                "bottom": 30,
                            },
                        }
                        for idx, ch in enumerate("Guariglia")
                    ]
                }]
            }]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        stats = micro_module.RunStats()
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((40, 140, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
            block_text="Chen和Guariglia，",
        )

        assert line.text == "Guariglia"
        assert [char.text for char in line.chars] == list("Guariglia")
        assert all(char.source == "hanwang:EngCut:latin_exact" for char in line.chars)
        assert all(char.token_text == "Guariglia" for char in line.chars)
        assert stats.latin_engcut_exact_tokens == 1
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_uses_paddle_token_to_repair_bad_span PASSED")


def test_hanwang_latin_engcut_uses_formula_letter_list_and_low_confidence_fallback():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    chars = [
        ("自", 0.91, (0, 0, 30, 46)),
        ("然", 0.92, (45, 0, 89, 45)),
        ("对", 0.84, (100, 0, 146, 45)),
        ("数", 0.80, (154, 0, 200, 47)),
        ("形", 0.93, (208, 0, 252, 45)),
        ("式", 0.91, (260, 0, 306, 46)),
        ("，", 0.52, (322, 35, 329, 48)),
        ("即", 0.82, (346, 0, 385, 44)),
        ("y", 0.19, (400, 20, 427, 51)),
        ("、", 0.56, (428, 35, 438, 47)),
        ("尼", 0.19, (450, 6, 470, 41)),
        ("、", 0.49, (475, 36, 484, 47)),
        ("Z", 0.19, (497, 6, 509, 41)),
        ("、", 0.50, (514, 36, 523, 48)),
        ("m", 0.19, (536, 21, 569, 42)),
        ("分", 0.97, (583, 1, 629, 47)),
        ("别", 0.93, (636, 1, 679, 48)),
    ]
    line = micro_module.LineResult(
        text="自然对数形式，即y、尼、Z、m分别",
        bbox=(0, 0, 690, 60),
        chars=[
            micro_module.CharResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
                candidates=[text],
                source="hanwang:micro_recblock",
                bbox_granularity="char",
                token_text=text,
            )
            for text, confidence, bbox in chars
        ],
    )

    def fake_eng20(image_bgr, *, timeout=0):
        return {
            "lines": [{
                "groups": [{
                    "chars": [
                        {"codes": [ord("l")], "bbox": {"left": 166, "top": 0, "right": 177, "bottom": 21}},
                        {"codes": [ord("y")], "bbox": {"left": 400, "top": 20, "right": 427, "bottom": 51}},
                        {"codes": [ord("k")], "bbox": {"left": 450, "top": 6, "right": 470, "bottom": 41}},
                        {"codes": [ord("l")], "bbox": {"left": 497, "top": 6, "right": 509, "bottom": 41}},
                        {"codes": [ord("m")], "bbox": {"left": 536, "top": 21, "right": 569, "bottom": 42}},
                    ]
                }]
            }]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        stats = micro_module.RunStats()
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((50, 260, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
            block_text="其中，即 $ y, k, l, m $ 分别指企业的产量",
        )

        assert line.text == "自然对数形式，即y、k、l、m分别"
        assert line.chars[3].text == "数"
        assert line.chars[3].source == "hanwang:micro_recblock"
        assert [char.text for char in line.chars[8:15]] == ["y", "、", "k", "、", "l", "、", "m"]
        assert line.chars[10].source == "hanwang:EngCut:latin_exact"
        assert line.chars[12].source == "hanwang:EngCut:latin_exact"
        assert stats.latin_engcut_probe_calls == 1
        assert stats.latin_engcut_exact_tokens == 4
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_uses_formula_letter_list_and_low_confidence_fallback PASSED")


def test_hanwang_latin_engcut_marks_slash_variant_for_review_without_text_rewrite():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    line = micro_module.LineResult(
        text="PE!VC",
        bbox=(0, 0, 90, 32),
        chars=[
            micro_module.CharResult(text=ch, bbox=(idx * 14, 0, idx * 14 + 10, 28))
            for idx, ch in enumerate("PE!VC")
        ],
    )

    def fake_eng20(image_bgr, *, timeout=0):
        return {
            "lines": [{
                "groups": [{
                    "chars": [
                        {
                            "codes": [ord(ch)],
                            "bbox": {
                                "left": idx * 14,
                                "top": 2,
                                "right": idx * 14 + 10,
                                "bottom": 30,
                            },
                        }
                        for idx, ch in enumerate("PE!VC")
                    ]
                }]
            }]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        stats = micro_module.RunStats()
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((40, 100, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
            block_text="其他PE/VC基金",
        )

        assert line.text == "PE!VC"
        assert [char.text for char in line.chars] == list("PE!VC")
        assert micro_module.LATIN_ENGCUT_REVIEW_FLAG in line.review_flags
        assert stats.latin_engcut_exact_tokens == 0
        assert stats.latin_engcut_review_tokens == 1
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_marks_slash_variant_for_review_without_text_rewrite PASSED")


def test_hanwang_latin_engcut_reverse_fallback_accepts_exact_after_source_order_guard():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    lines = [
        micro_module.LineResult(
            text="Other",
            bbox=(0, 0, 80, 30),
            chars=[
                micro_module.CharResult(text=ch, bbox=(idx * 12, 0, idx * 12 + 10, 26))
                for idx, ch in enumerate("Other")
            ],
        ),
        micro_module.LineResult(
            text="PE/VC",
            bbox=(0, 40, 90, 72),
            chars=[
                micro_module.CharResult(text=ch, bbox=(idx * 14, 40, idx * 14 + 10, 68))
                for idx, ch in enumerate("PE/VC")
            ],
        ),
    ]
    engcut_texts = ["Other", "PE/VC"]

    def fake_eng20(image_bgr, *, timeout=0):
        text = engcut_texts.pop(0)
        return {
            "lines": [{
                "groups": [{
                    "chars": [
                        {
                            "codes": [ord(ch)],
                            "bbox": {
                                "left": idx * 14,
                                "top": 2,
                                "right": idx * 14 + 10,
                                "bottom": 28,
                            },
                        }
                        for idx, ch in enumerate(text)
                    ]
                }]
            }]
        }

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        stats = micro_module.RunStats()
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((90, 120, 3), dtype=np.uint8),
            lines,
            stats,
            timeout=1.0,
            block_text="PE/VC Other",
        )

        assert lines[1].text == "PE/VC"
        assert [char.source for char in lines[1].chars] == ["hanwang:EngCut:latin_exact"] * 5
        assert micro_module.LATIN_ENGCUT_REVIEW_FLAG not in lines[1].review_flags
        assert stats.latin_engcut_exact_tokens == 2
        assert stats.latin_engcut_review_tokens == 0
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    print("test_hanwang_latin_engcut_reverse_fallback_accepts_exact_after_source_order_guard PASSED")


def test_hanwang_crop_padding_expands_without_leaving_page():
    import app.engines.hanwang.micro_recblock as micro_module

    assert micro_module._expand_xyxy((10, 20, 30, 40), 100, 100, pad_x=2, pad_y=3) == (8, 17, 32, 43)
    assert micro_module._expand_xyxy((0, 1, 99, 100), 100, 100, pad_x=5, pad_y=5) == (0, 0, 100, 100)

    print("test_hanwang_crop_padding_expands_without_leaving_page PASSED")


def test_hanwang_inline_formula_carrier_survives_model_and_proof_helpers():
    import os
    import tempfile

    import cv2
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    from app.models import BBox, Block, BlockType, Page
    from app.services.char_index_service import CharIndexService
    from app.services.proof_crop_service import ProofCropService

    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (2, 5), (18, 25), (0, 0, 0), -1)
    cv2.rectangle(image, (42, 5), (68, 25), (0, 0, 0), -1)
    cv2.rectangle(image, (92, 5), (108, 25), (0, 0, 0), -1)
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    cv2.imwrite(handle.name, image)

    try:
        line_result = micro_module.LineResult(
            text="甲$ A $乙",
            bbox=(0, 0, 110, 30),
            confidence=0.9,
            chars=[
                micro_module.CharResult(text="甲", confidence=0.9, bbox=(0, 0, 20, 30)),
                micro_module.CharResult(
                    text="$ A $",
                    confidence=0.0,
                    bbox=(40, 0, 70, 30),
                    candidates=["$ A $"],
                    source="paddle_inline_formula",
                    bbox_granularity="word",
                    token_text="$ A $",
                ),
                micro_module.CharResult(text="乙", confidence=0.9, bbox=(90, 0, 110, 30)),
            ],
            review_flags=[micro_module.ROUTE_INLINE_FORMULA_FLAG],
        )
        line = micro_module._line_to_model(line_result, 120, 40, [])
        formula_char = line.chars[1]
        assert formula_char.char == "$ A $"
        assert formula_char.token_text == "$ A $"
        assert formula_char.bbox == BBox(40, 0, 30, 30)
        assert formula_char.bbox_source == "paddle_inline_formula"
        assert formula_char.bbox_granularity == "word"

        page = Page(image_path=handle.name, width=120, height=40)
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 110, 30), lines=[line])]
        before = [
            (char.char, char.token_text, char.bbox, char.bbox_source, char.bbox_granularity)
            for char in line.chars
        ]

        ProofCropService().normalize_pages([page])
        CharIndexService().build([page])

        after = [
            (char.char, char.token_text, char.bbox, char.bbox_source, char.bbox_granularity)
            for char in line.chars
        ]
        assert after == before
        assert CharIndexService().build([page]).query("$ A $") == []
    finally:
        os.unlink(handle.name)

    print("test_hanwang_inline_formula_carrier_survives_model_and_proof_helpers PASSED")


def test_proof_crop_service_does_not_rewrite_existing_ocr_geometry(tmp_path):
    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.services.proof_crop_service import ProofCropService

    img = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 62), (160, 76), (0, 0, 0), -1)
    img_path = str(tmp_path / "proof-page.png")
    cv2.imwrite(img_path, img)

    line = Line(
        text="汉王",
        confidence=0.9,
        bbox=BBox(10, 44, 180, 46),
        chars=[
            Char(char="汉", confidence=0.9, bbox=BBox(20, 62, 30, 20), bbox_source="hanwang:CharRcg", bbox_granularity="char"),
            Char(char="王", confidence=0.9, bbox=BBox(70, 62, 30, 20), bbox_source="hanwang:CharRcg", bbox_granularity="char"),
        ],
    )
    page = Page(
        image_path=img_path,
        width=220,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 200, 100), lines=[line])],
    )
    before_line_bbox = line.bbox
    before_char_boxes = [char.bbox for char in line.chars]

    stats = ProofCropService().normalize_pages([page])

    assert stats.line_bbox_updates == 0
    assert line.bbox == before_line_bbox
    assert [char.bbox for char in line.chars] == before_char_boxes

    print("test_proof_crop_service_does_not_rewrite_existing_ocr_geometry PASSED")


def test_proof_crop_service_reports_fallback_geometry(tmp_path):
    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.proof_crop_service import ProofCropService

    img = np.full((80, 180, 3), 255, dtype=np.uint8)
    img_path = str(tmp_path / "fallback-page.png")
    cv2.imwrite(img_path, img)

    line = Line(
        text="甲A1",
        confidence=0.8,
        bbox=BBox(10, 20, 90, 24),
        chars=[],
    )
    page = Page(
        image_path=img_path,
        width=180,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 160, 60), lines=[line])],
    )

    stats = ProofCropService().normalize_pages([page])

    assert stats.fallback_lines == 1
    assert stats.fallback_chars == 3
    assert stats.unavailable_chars == 0
    assert [char.bbox_granularity for char in line.chars] == ["fallback", "fallback", "fallback"]

    print("test_proof_crop_service_reports_fallback_geometry PASSED")


def test_proof_fallback_warning_scans_existing_chars():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.services.proof_crop_service import ProofCropStats, proof_fallback_warning

    line = Line(
        text="甲乙",
        confidence=0.8,
        bbox=BBox(10, 20, 60, 24),
        chars=[
            Char(char="甲", confidence=0.8, bbox=BBox(10, 20, 30, 24), bbox_source="fallback", bbox_granularity="fallback"),
            Char(char="乙", confidence=0.8, bbox=BBox(40, 20, 30, 24), bbox_source="fallback", bbox_granularity="fallback"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 60), lines=[line])],
    )

    warning = proof_fallback_warning(ProofCropStats(), [page])

    assert warning == "警告：proof fallback 1 行/2 字，字框为估算或不可用"

    print("test_proof_fallback_warning_scans_existing_chars PASSED")


def test_hanwang_pre_page_ocr_lines_split_before_recog():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {"groups": [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]}
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    chars_by_crop = {
        (68, 40): [("甲", (0, 10, 20, 30))],
        (106, 40): [("乙", (10, 10, 30, 30))],
        (88, 40): [("丙", (0, 10, 20, 30))],
        (86, 40): [("丁", (12, 10, 32, 30))],
    }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        result_lines = []
        for x1, y1, x2, y2 in blocks:
            specs = chars_by_crop.get((x2 - x1, y2 - y1), [])
            result_lines.append({"groups": [{
                "bbox": {"left": 0, "top": 0, "right": x2 - x1, "bottom": y2 - y1},
                "chars": [
                    {
                        "codes": [code(ch)],
                        "scores": [5],
                        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                    }
                    for ch, (left, top, right, bottom) in specs
                ],
            }]})
        return {"lines": result_lines}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((100, 200, 3), dtype=np.uint8),
            [{
                "block_label": "text",
                "block_bbox": [0, 0, 190, 90],
                "block_content": "甲 $ A $ 乙丙 $ B $ 丁",
                "_route_subblocks": [
                    {"block_label": "inline_formula", "block_bbox": [60, 0, 90, 40]},
                    {"block_label": "inline_formula", "block_bbox": [80, 40, 110, 80]},
                ],
            }],
            page_ocr_lines=[
                {"text": "甲乙", "bbox": [0, 10, 180, 30]},
                {"text": "丙丁", "bbox": [0, 50, 180, 70]},
            ],
        )

        assert seen_recblocks == [
            (0, 10, 60, 30),
            (90, 10, 180, 30),
            (0, 50, 80, 70),
            (110, 50, 180, 70),
        ]
        assert [line.text for line in rows[0].lines] == ["甲$ A $乙", "丙$ B $丁"]
        assert [char.text for line in rows[0].lines for char in line.chars] == ["甲", "$ A $", "乙", "丙", "$ B $", "丁"]
        assert [
            (char.text, char.bbox, char.source, char.bbox_granularity)
            for line in rows[0].lines
            for char in line.chars
            if char.source == "paddle_inline_formula"
        ] == [
            ("$ A $", (60, 0, 90, 40), "paddle_inline_formula", "word"),
            ("$ B $", (80, 40, 110, 80), "paddle_inline_formula", "word"),
        ]
        assert stats.n_blocks_hanwang == 1
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_pre_page_ocr_lines_split_before_recog PASSED")


def test_hanwang_route_assembly_recovers_tiny_punctuation_after_formula():
    from app.engines.hanwang.micro_recblock import (
        CharResult,
        LineResult,
        _assemble_layout_route_line,
    )
    from app.services.layout_routing_plan import RoutingLine, RoutingSegment

    route = RoutingLine(
        index=0,
        bbox=(0, 0, 130, 50),
        segments=(
            RoutingSegment(kind="text", bbox=(0, 0, 40, 50)),
            RoutingSegment(kind="formula", bbox=(40, 0, 80, 50), text="$ A $"),
            RoutingSegment(kind="text", bbox=(80, 0, 130, 50)),
        ),
    )
    grouped_lines = {
        (0, 0, 0): [
            LineResult(
                text="甲",
                bbox=(0, 8, 30, 42),
                chars=[CharResult(text="甲", confidence=0.9, bbox=(0, 8, 30, 42))],
            )
        ],
        (0, 0, 2): [
            LineResult(
                text="，乙",
                bbox=(78, 10, 120, 40),
                chars=[
                    CharResult(text="，", confidence=0.8, bbox=(78, 18, 79, 21)),
                    CharResult(text="乙", confidence=0.9, bbox=(92, 10, 120, 40)),
                ],
            )
        ],
    }

    lines = _assemble_layout_route_line(
        block_idx=0,
        line_idx=0,
        route=route,
        grouped_lines=grouped_lines,
    )

    assert len(lines) == 1
    assert lines[0].text == "甲$ A $，乙"
    comma = lines[0].chars[2]
    assert comma.text == "，"
    assert comma.bbox == (80, 10, 92, 40)
    assert comma.source.endswith(":punct_bbox_recovered")

    print("test_hanwang_route_assembly_recovers_tiny_punctuation_after_formula PASSED")


def test_hanwang_route_assembly_drops_formula_boundary_punctuation_noise():
    from app.engines.hanwang.micro_recblock import (
        CharResult,
        LineResult,
        _assemble_layout_route_line,
    )
    from app.services.layout_routing_plan import RoutingLine, RoutingSegment

    route = RoutingLine(
        index=0,
        bbox=(199, 556, 1453, 620),
        segments=(
            RoutingSegment(kind="text", bbox=(199, 556, 445, 620)),
            RoutingSegment(kind="formula", bbox=(445, 556, 884, 620), text=""),
            RoutingSegment(kind="text", bbox=(884, 556, 1242, 620)),
            RoutingSegment(kind="formula", bbox=(1242, 562, 1453, 620), text=""),
        ),
    )
    grouped_lines = {
        (0, 0, 0): [
            LineResult(
                text="其中，",
                bbox=(197, 558, 437, 616),
                chars=[
                    CharResult(text="其", confidence=0.9, bbox=(197, 563, 240, 609)),
                    CharResult(text="中", confidence=0.9, bbox=(245, 563, 288, 609)),
                    CharResult(text="，", confidence=0.8, bbox=(292, 585, 300, 609)),
                ],
            )
        ],
        (0, 0, 2): [
            LineResult(
                text="，均",
                bbox=(882, 553, 940, 622),
                chars=[
                    CharResult(text="，", confidence=0.19, bbox=(875, 563, 876, 566)),
                    CharResult(text="均", confidence=0.81, bbox=(896, 563, 940, 609)),
                ],
            )
        ],
    }

    lines = _assemble_layout_route_line(
        block_idx=0,
        line_idx=0,
        route=route,
        grouped_lines=grouped_lines,
    )

    assert len(lines) == 1
    boundary_comma = lines[0].chars[3]
    assert boundary_comma.text == "，"
    assert boundary_comma.bbox is None
    assert boundary_comma.source.endswith(":punct_bbox_dropped_at_route_boundary")
    assert lines[0].chars[4].text == "均"
    assert lines[0].chars[4].bbox == (896, 563, 940, 609)

    print("test_hanwang_route_assembly_drops_formula_boundary_punctuation_noise PASSED")


def test_hanwang_micro_recblock_sends_inter_formula_punctuation_gap_to_segimg():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {"lines": [{"groups": []} for _ in (recblocks_xyxy or [])]}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module._BATCH_DISABLED_FOR_SESSION = True
    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((80, 200, 3), dtype=np.uint8),
            [
                {
                    "block_label": "text",
                    "block_bbox": [0, 0, 180, 50],
                    "block_content": "甲 $ A $、$ B $ 乙",
                    "_route_subblocks": [
                        {"block_label": "inline_formula", "block_bbox": [40, 0, 82, 40]},
                        {"block_label": "inline_formula", "block_bbox": [78, 0, 120, 40]},
                    ],
                }
            ],
            include_chars=True,
        )

        assert any(left < 82 and right > 78 and right - left >= 16 for left, _top, right, _bottom in seen_recblocks)
        assert stats.n_blocks_hanwang == 1
        assert rows[0].source == "hanwang"
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_micro_recblock_sends_inter_formula_punctuation_gap_to_segimg PASSED")


def test_hanwang_micro_recblock_drops_stale_cached_layout_routes_without_page_hints():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_recblocks = []
    recog_texts = iter(["甲", "乙"])

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {"groups": [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]}
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        result_lines = []
        for x1, y1, x2, y2 in blocks:
            text = next(recog_texts)
            result_lines.append({"groups": [{
                "bbox": {"left": 0, "top": 0, "right": x2 - x1, "bottom": y2 - y1},
                "chars": [{
                    "codes": [code(text)],
                    "scores": [9],
                    "bbox": {"left": 5, "top": 5, "right": 25, "bottom": 35},
                }],
            }]})
        return {"lines": result_lines}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        rows, _stats = micro_module.run_micro_recblock(
            np.zeros((60, 150, 3), dtype=np.uint8),
            [{
                "block_label": "text",
                "block_bbox": [0, 0, 130, 40],
                "block_content": "甲 $ A $ 乙",
                "_route_subblocks": [
                    {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 40]},
                ],
                "_layout_line_routes": [
                    {
                        "bbox": [80, 0, 130, 40],
                        "segments": [{"kind": "text", "bbox": [80, 0, 130, 40], "text": ""}],
                    },
                ],
            }],
            page_ocr_lines=[],
        )

        assert seen_recblocks == [(0, 0, 40, 40), (70, 0, 130, 40)]
        assert rows[0].text == "甲$ A $乙"
        assert [char.text for line in rows[0].lines for char in line.chars] == ["甲", "$ A $", "乙"]
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_micro_recblock_drops_stale_cached_layout_routes_without_page_hints PASSED")


def test_hanwang_layout_row_ignores_stale_persisted_layout_routes():
    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD, text_slice_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    page = Page(
        image_path="",
        width=150,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 130, 40],
                "block_content": "甲 $ A $ 乙",
                "_route_subblocks": [
                    {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 40]},
                ],
            },
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 130, 40),
                origin=BlockOrigin(source_label="text", raw_index=0),
            )
        ],
    )

    rows = _page_blocks_from_layout(page)

    assert LAYOUT_LINE_ROUTES_FIELD not in rows[0]
    assert [route["bbox"] for route in text_slice_routes_for_block(rows[0], 150, 60)] == [
        [0, 0, 40, 40],
        [70, 0, 130, 40],
    ]

    print("test_hanwang_layout_row_ignores_stale_persisted_layout_routes PASSED")


def test_hanwang_page_block_writeback_does_not_persist_layout_line_routes():
    import numpy as np

    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD
    from app.engines.hanwang.micro_recblock import (
        BlockResult,
        HanwangMicroRecBlockEngine,
        LineResult,
    )
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    def fake_runner(image_bgr, ppvl_blocks, **_kwargs):
        ppvl_blocks[0][LAYOUT_LINE_ROUTES_FIELD] = [
            {
                "bbox": [80, 0, 130, 40],
                "segments": [{"kind": "text", "bbox": [80, 0, 130, 40], "text": ""}],
            }
        ]
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(0, 0, 130, 40),
                layout_bbox=(0, 0, 130, 40),
                block_bbox_source="layout_block_bbox",
                source="hanwang",
                text="甲",
                ppvl_text="甲 $ A $ 乙",
                lines=[LineResult(text="甲", bbox=(0, 0, 30, 40), confidence=0.9)],
                raw_block=dict(ppvl_blocks[0]),
            )
        ], micro_module.RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    import app.engines.hanwang.micro_recblock as micro_module

    page = Page(
        image_path="",
        width=150,
        height=60,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [0, 0, 130, 40],
                "block_content": "甲 $ A $ 乙",
                LAYOUT_LINE_ROUTES_FIELD: [
                    {
                        "bbox": [80, 0, 130, 40],
                        "segments": [{"kind": "text", "bbox": [80, 0, 130, 40], "text": ""}],
                    }
                ],
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 130, 40),
                origin=BlockOrigin(source_label="text", raw_index=0),
            )
        ],
    )

    HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((60, 150, 3), dtype=np.uint8),
        page,
    )

    assert LAYOUT_LINE_ROUTES_FIELD not in _raw_layout_records(page)[0]
    assert not hasattr(page.blocks[0], "raw_payload")

    print("test_hanwang_page_block_writeback_does_not_persist_layout_line_routes PASSED")


def test_hanwang_bbox_audit_distinguishes_layout_route_and_recog_boxes():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {
                                "left": x1 - 2,
                                "top": y1 + 2,
                                "right": x2 + 2,
                                "bottom": y2 - 2,
                            }
                        }
                    ]
                }
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        h, w = image_bgr.shape[:2]
        chars_by_crop = {
            (48, 48): [("甲", (8, 0, 38, 40))],
            (66, 48): [("乙", (12, 0, 52, 40))],
        }
        specs = chars_by_crop.get((w, h), [])
        if not specs:
            return {"lines": []}
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                            "chars": [
                                {
                                    "codes": [code(ch)],
                                    "scores": [5],
                                    "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                                }
                                for ch, (left, top, right, bottom) in specs
                            ],
                        }
                    ]
                }
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((60, 140, 3), dtype=np.uint8),
            [
                {
                    "block_label": "text",
                    "block_bbox": [0, 0, 120, 40],
                    "block_content": "甲 $ A $ 乙",
                    "_route_subblocks": [
                        {"block_label": "inline_formula", "block_bbox": [40, 0, 70, 40]},
                    ],
                }
            ],
            include_chars=True,
        )

        assert seen_recblocks == [(0, 0, 40, 40), (70, 0, 120, 40)]
        assert stats.n_blocks_hanwang == 1
        assert rows[0].text == "甲$ A $乙"
        assert rows[0].layout_bbox == (0, 0, 120, 40)
        assert rows[0].block_bbox_source == "layout_line_routes_union"
        assert rows[0].route_text_slice_bboxes == [(0, 0, 40, 40), (70, 0, 120, 40)]
        assert rows[0].recog_group_bboxes == [(0, 0, 48, 48), (62, 0, 128, 48)]
        assert rows[0].segimg_group_audits == [
            {
                "route_text_slice_bbox": [0, 0, 40, 40],
                "segimg_group_bbox": [0, 2, 42, 38],
                "recog_group_bbox": [0, 0, 48, 48],
                "recog_group_bbox_before_padding": [0, 2, 40, 38],
                "recog_group_bbox_padded": True,
                "clipped": True,
                "dropped": False,
            },
            {
                "route_text_slice_bbox": [70, 0, 120, 40],
                "segimg_group_bbox": [68, 2, 122, 38],
                "recog_group_bbox": [62, 0, 128, 48],
                "recog_group_bbox_before_padding": [70, 2, 120, 38],
                "recog_group_bbox_padded": True,
                "clipped": True,
                "dropped": False,
            },
        ]
        assert [line.bbox_source for line in rows[0].lines] == ["layout_route_assembled"]
        audit = rows[0].raw_block["_hanwang_bbox_audit"]
        assert audit["schema"] == "hanwang_bbox_audit.v1"
        assert audit["layout_block_bbox"] == [0, 0, 120, 40]
        assert audit["effective_block_bbox"] == [0, 0, 120, 40]
        assert audit["effective_block_bbox_source"] == "layout_line_routes_union"
        assert audit["layout_line_route_bboxes"] == [[0, 0, 120, 40]]
        assert audit["route_text_slice_bboxes"] == [[0, 0, 40, 40], [70, 0, 120, 40]]
        assert audit["hanwang_recog_group_bboxes"] == [[0, 0, 48, 48], [62, 0, 128, 48]]
        assert audit["hanwang_segimg_group_clipped_count"] == 2
        assert audit["hanwang_segimg_group_dropped_count"] == 0
        assert audit["route_text_slice_count"] == 2
        assert audit["hanwang_recog_group_count"] == 2
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_bbox_audit_distinguishes_layout_route_and_recog_boxes PASSED")


def test_hanwang_micro_recblock_short_chinese_group_keeps_vertical_context():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_crops = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        assert recblocks_xyxy == [(1661, 1269, 1773, 1330)]
        return {
            "lines": [{
                "groups": [{
                    "bbox": {"left": 1663, "top": 1285, "right": 1763, "bottom": 1317}
                }]
            }]
        }

    def fake_recog(image_bgr, **_kwargs):
        seen_crops.append(tuple(image_bgr.shape[:2]))
        assert seen_crops[-1] == (52, 116)
        return {
            "lines": [{
                "groups": [{
                    "bbox": {"left": 0, "top": 0, "right": 116, "bottom": 52},
                    "chars": [
                        {
                            "codes": [code("，")],
                            "scores": [80],
                            "bbox": {"left": 0, "top": 35, "right": 4, "bottom": 49},
                        },
                        {
                            "codes": [code("表")],
                            "scores": [8],
                            "bbox": {"left": 12, "top": 10, "right": 55, "bottom": 50},
                        },
                        {
                            "codes": [code("示")],
                            "scores": [8],
                            "bbox": {"left": 65, "top": 10, "right": 108, "bottom": 50},
                        },
                    ],
                }]
            }]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        rows, _stats = micro_module.run_micro_recblock(
            np.zeros((1400, 1900, 3), dtype=np.uint8),
            [{"block_label": "text", "block_bbox": [1661, 1269, 1773, 1330], "block_content": "表示"}],
            include_chars=True,
        )

        assert seen_crops == [(52, 116)]
        assert rows[0].text == "表示"
        assert rows[0].recog_group_bboxes == [(1655, 1275, 1771, 1327)]
        assert [char.text for char in rows[0].lines[0].chars] == ["表", "示"]
        assert rows[0].lines[0].chars[0].bbox == (1667, 1285, 1710, 1325)
        assert rows[0].lines[0].chars[1].bbox == (1720, 1285, 1763, 1325)
        audit = rows[0].raw_block["_hanwang_bbox_audit"]
        group_audit = audit["hanwang_segimg_groups"][0]
        assert group_audit["recog_group_bbox_before_padding"] == [1663, 1285, 1763, 1317]
        assert group_audit["recog_group_bbox"] == [1655, 1275, 1771, 1327]
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_micro_recblock_short_chinese_group_keeps_vertical_context PASSED")


def test_hanwang_digitlike_numeric_context_normalizes_low_confidence_binary_values():
    import app.engines.hanwang.micro_recblock as micro_module

    def char(text, confidence=0.9):
        return micro_module.CharResult(
            text=text,
            confidence=confidence,
            bbox=(0, 0, 10, 10),
            candidates=[text],
            source="hanwang:micro_recblock",
            bbox_granularity="char",
            token_text=text,
        )

    lines = [
        micro_module.LineResult(
            text="否则取o；",
            bbox=(0, 0, 100, 20),
            chars=[char("否"), char("则"), char("取"), char("o", 0.19), char("；")],
        ),
        micro_module.LineResult(
            text="之后取l，",
            bbox=(0, 20, 100, 40),
            chars=[char("之"), char("后"), char("取"), char("l", 0.19), char("，")],
        ),
        micro_module.LineResult(
            text="取open",
            bbox=(0, 40, 100, 60),
            chars=[char("取"), char("o", 0.19), char("p"), char("e"), char("n")],
        ),
        micro_module.LineResult(
            text="取o；",
            bbox=(0, 60, 100, 80),
            chars=[char("取"), char("o", 0.8), char("；")],
        ),
    ]

    micro_module._normalize_digitlike_numeric_context_lines(lines)

    assert lines[0].text == "否则取0；"
    assert lines[0].chars[3].text == "0"
    assert lines[0].chars[3].token_text == "0"
    assert micro_module.DIGITLIKE_NUMERIC_CONTEXT_REVIEW_FLAG in lines[0].review_flags
    assert lines[1].text == "之后取1，"
    assert lines[1].chars[3].text == "1"
    assert lines[2].text == "取open"
    assert lines[3].text == "取o；"

    print("test_hanwang_digitlike_numeric_context_normalizes_low_confidence_binary_values PASSED")


def test_hanwang_micro_recblock_refines_formula_mixed_ppocr_line_text_bands():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module
    from app.core.paddle_line_routing import PageOcrLineHint

    seen_recblocks = []

    def fake_segimg(_image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {"lines": [{"groups": []} for _ in (recblocks_xyxy or [])]}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    micro_module.native_bridge.run_linecut_segimg = fake_segimg

    try:
        image = np.full((90, 320, 3), 255, dtype=np.uint8)
        image[12:42, 10:90] = 0
        image[12:42, 170:290] = 0
        image[25:70, 105:145] = 0
        rows, _stats = micro_module.run_micro_recblock(
            image,
            [
                {
                    "block_label": "text",
                    "block_bbox": [0, 0, 300, 80],
                    "block_content": "甲 $ A $ 乙",
                    "_route_subblocks": [
                        {"block_label": "inline_formula", "block_bbox": [100, 25, 150, 70]},
                    ],
                }
            ],
            include_chars=True,
            page_ocr_lines=[
                PageOcrLineHint(
                    text="甲A乙",
                    bbox=(0, 0, 300, 80),
                )
            ],
        )

        assert seen_recblocks == [
            (0, 6, 100, 48),
            (150, 6, 300, 48),
        ]
        assert rows[0].route_text_slice_bboxes == [
            (0, 6, 100, 48),
            (150, 6, 300, 48),
        ]
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg

    print("test_hanwang_micro_recblock_refines_formula_mixed_ppocr_line_text_bands PASSED")


def test_hanwang_recog_group_failure_is_visible_in_audit_without_ppvl_fallback():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [
                {
                    "groups": [
                        {"bbox": {"left": 10, "top": 10, "right": 80, "bottom": 40}},
                    ]
                }
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        raise RuntimeError("native recog timeout")

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [
                {
                    "block_label": "text",
                    "block_bbox": [0, 0, 100, 50],
                    "block_content": "不应回退",
                }
            ],
            include_chars=True,
        )

        assert rows[0].source == "hanwang"
        assert rows[0].text == ""
        assert rows[0].ppvl_text == "不应回退"
        assert rows[0].fallback_reason == ""
        assert rows[0].lines == []
        assert stats.recog_group_failures == 1
        audit = rows[0].raw_block["_hanwang_bbox_audit"]
        assert audit["hanwang_recog_group_failed_count"] == 1
        assert audit["hanwang_recog_group_count"] == 1
        assert audit["hanwang_segimg_groups"][0]["recog_failed"] is True
        assert audit["hanwang_segimg_groups"][0]["recog_error"] == "native recog timeout"
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_recog_group_failure_is_visible_in_audit_without_ppvl_fallback PASSED")


def test_hanwang_recog_group_failure_retries_with_top_trim_before_dropping_line():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [
                {
                    "groups": [
                        {"bbox": {"left": 10, "top": 10, "right": 80, "bottom": 40}},
                    ]
                }
            ]
        }

    calls = []

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        calls.append(tuple(image_bgr.shape[:2]))
        if len(calls) == 1:
            raise RuntimeError("native recog access violation")
        assert calls[-1] == (47, 86)
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {"left": 0, "top": 0, "right": 30, "bottom": 20},
                            "chars": [
                                {
                                    "codes": [code("补")],
                                    "scores": [8],
                                    "bbox": {"left": 2, "top": 3, "right": 22, "bottom": 19},
                                },
                            ],
                        }
                    ]
                }
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [
                {
                    "block_label": "text",
                    "block_bbox": [0, 0, 100, 50],
                    "block_content": "补",
                }
            ],
            include_chars=True,
        )

        assert calls == [(50, 86), (47, 86)]
        assert rows[0].text == "补"
        assert rows[0].lines[0].bbox == (2, 3, 32, 23)
        assert rows[0].lines[0].chars[0].bbox == (4, 6, 24, 22)
        assert stats.recog_probe_calls == 2
        assert stats.recog_group_failures == 0
        assert stats.recog_group_retry_attempts == 1
        assert stats.recog_group_retry_successes == 1
        assert stats.recog_group_retry_failures == 0
        audit = rows[0].raw_block["_hanwang_bbox_audit"]
        assert audit["hanwang_recog_group_failed_count"] == 0
        group_audit = audit["hanwang_segimg_groups"][0]
        assert group_audit["recog_group_bbox"] == [2, 0, 88, 50]
        assert group_audit["recog_retry_attempted"] is True
        assert group_audit["recog_retry_succeeded"] is True
        assert group_audit["recog_retry_bbox"] == [2, 3, 88, 50]
        assert "native recog access violation" in group_audit["recog_retry_original_error"]
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_recog_group_failure_retries_with_top_trim_before_dropping_line PASSED")


def test_ocr_pipeline_hybrid_prepass_lines_feed_hanwang_splitter():
    import os
    import tempfile

    import cv2
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    from app.engines.hanwang.micro_recblock import HanwangMicroRecBlockEngine
    from app.models import BBox, Block, BlockOrigin, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {"groups": [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]}
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        crop_h, crop_w = image_bgr.shape[:2]
        if recblocks_xyxy:
            blocks = [(x1, y1, x2, y2, "甲" if x2 - x1 <= 70 else "乙") for x1, y1, x2, y2 in recblocks_xyxy]
        else:
            blocks = [(0, 0, crop_w, crop_h, "甲" if crop_w <= 70 else "乙")]
        return {
            "lines": [
                {"groups": [{
                    "bbox": {"left": 0, "top": 0, "right": x2 - x1, "bottom": y2 - y1},
                    "chars": [{
                        "codes": [code(text)],
                        "scores": [5],
                        "bbox": {"left": 0, "top": 0, "right": 20, "bottom": 20},
                    }],
                }]}
                for x1, y1, x2, y2, text in blocks
            ]
        }

    class FakePrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="甲乙", bbox=BBox.from_xyxy(0, 10, 180, 30), confidence=0.9)]

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        cv2.imwrite(img_path, np.ones((80, 200, 3), dtype=np.uint8) * 255)

    try:
        route_subblocks = [
            {"block_label": "inline_formula", "block_bbox": [60, 0, 90, 40]},
        ]
        page = Page(
            image_path=img_path,
            width=200,
            height=80,
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(0, 0, 190, 50),
                    source_label="text",
                    origin=BlockOrigin(source_label="text", raw_index=0),
                )
            ],
            raw_layout_artifact=_paddle_layout_artifact([{
                "block_label": "text",
                "block_bbox": [0, 0, 190, 50],
                "block_content": "甲 $ A $ 乙",
                "_route_subblocks": route_subblocks,
            }]),
        )
        result = OcrPipeline(
            engine=HanwangMicroRecBlockEngine(),
            hybrid_prepass_engine=FakePrepassEngine(),
        ).process_project(OcrProject(name="hybrid-prepass", pages=[page]))

        assert seen_recblocks == [(0, 10, 60, 30), (90, 10, 180, 30)]
        assert result.pages[0].blocks[0].lines[0].text == "甲$ A $乙"
        assert [char.char for char in result.pages[0].blocks[0].lines[0].chars] == ["甲", "$ A $", "乙"]
        assert result.pages[0].blocks[0].lines[0].chars[1].bbox_source == "paddle_inline_formula"
        assert result.pages[0].blocks[0].lines[0].chars[1].bbox_granularity == "word"
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        os.unlink(img_path)

    print("test_ocr_pipeline_hybrid_prepass_lines_feed_hanwang_splitter PASSED")


def test_paddle_line_routing_builds_layout_line_routes_from_reading_order():
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        build_layout_line_routes,
    )

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 210, 80],
        "block_content": "甲甲 $ A $ 乙乙。丙 $ B $ 丁 $ C $ 戊",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [70, 0, 110, 30]},
            {"block_label": "inline_formula", "block_bbox": [40, 40, 90, 70]},
            {"block_label": "inline_formula", "block_bbox": [140, 40, 180, 70]},
        ],
    }
    routes = build_layout_line_routes(block, 220, 100)

    assert len(routes) >= 2
    assert [segment["text"] for segment in routes[0]["segments"] if segment["kind"] == "formula"] == ["$ A $"]
    assert [segment["text"] for segment in routes[1]["segments"] if segment["kind"] == "formula"] == ["$ B $", "$ C $"]
    assert sum(1 for segment in routes[1]["segments"] if segment["kind"] == "text") >= 3
    block[LAYOUT_LINE_ROUTES_FIELD] = routes
    assert block[LAYOUT_LINE_ROUTES_FIELD][0]["segments"][1]["text"] == "$ A $"

    print("test_paddle_line_routing_builds_layout_line_routes_from_reading_order PASSED")


def test_paddle_line_routing_display_formula_span_is_not_split_to_empty_pair():
    from app.core.paddle_line_routing import _formula_spans, build_layout_line_routes

    assert _formula_spans(r"price is \$5, formula $ x $ and display $$ y+z $$") == [
        "$ x $",
        "$$ y+z $$",
    ]

    block = {
        "block_label": "display_formula",
        "block_bbox": [0, 0, 120, 30],
        "block_content": " $$ x+y $$ ",
        "_route_subblocks": [
            {"block_label": "display_formula", "block_bbox": [0, 0, 120, 30]},
        ],
    }

    routes = build_layout_line_routes(block, 140, 50)
    formula_texts = [
        segment["text"]
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    assert formula_texts == ["$$ x+y $$"]

    print("test_paddle_line_routing_display_formula_span_is_not_split_to_empty_pair PASSED")


def test_paddle_line_routing_line_hint_formula_recovery_does_not_shift_next_line():
    from app.core.paddle_line_routing import (
        PaddleRouteLineHint,
        recover_inline_formula_segments,
    )

    recovered = recover_inline_formula_segments(
        parent_text="甲 $ A $ 乙 $ B $ 丙 $ C $ 丁",
        line_hints=[
            PaddleRouteLineHint(text="甲 $ A $ 乙 $ B $", bbox=(0, 0, 200, 30)),
            PaddleRouteLineHint(text="丙 $ C $ 丁", bbox=(0, 40, 200, 70)),
        ],
        subblocks=[
            {"label": "inline_formula", "bbox": [100, 0, 130, 30]},
            {"label": "inline_formula", "bbox": [60, 40, 90, 70]},
        ],
    )

    assert [(item.line_index, item.text, item.bbox) for item in recovered] == [
        (0, "$ A $", (100, 0, 130, 30)),
        (1, "$ C $", (60, 40, 90, 70)),
    ]

    print("test_paddle_line_routing_line_hint_formula_recovery_does_not_shift_next_line PASSED")


def test_paddle_line_routing_missing_formula_box_does_not_shift_later_rows():
    from app.core.paddle_line_routing import (
        PaddleRouteLineHint,
        recover_inline_formula_segments,
    )

    recovered = recover_inline_formula_segments(
        parent_text=(
            "被解释变量 $ Y_{ct} $为地级市商品供需适配程度变量。"
            "核心解释变量 $ Incentive_{c} \\times Post_{t} $为强度变量与政策时点变量的交乘项，其中 "
            "$ Incentive_{c} $表示2016年增值税分成改革下各地级市结构性财政激励程度，"
            "政策时点变量 $ Post_{t} $为改革年份虚拟变量。 $ X_{ct} $为地区层面控制变量。"
            "年龄结构等。 $ \\delta_{c} $和 $ \\varphi_{t} $分别表示城市固定效应和年份固定效应，"
            " $ \\varepsilon_{ct} $为随机扰动项。"
        ),
        line_hints=[
            PaddleRouteLineHint(text="被解释变量Y为地级市商品供需适配程度变量。", bbox=(0, 0, 200, 30)),
            PaddleRouteLineHint(text="核心解释变量Incentive×Post为强度变量与政策时点变量的交乘项，其中", bbox=(0, 40, 260, 70)),
            PaddleRouteLineHint(text="Incentive表示2016年增值税分成改革下各地级市结构性财政激励程度", bbox=(0, 80, 260, 110)),
            PaddleRouteLineHint(text="政策时点变量Post为改革年份虚拟变量。", bbox=(0, 120, 260, 150)),
            PaddleRouteLineHint(text="X为地区层面控制变量。", bbox=(0, 160, 260, 190)),
            PaddleRouteLineHint(text="年龄结构等。δ和φ分别表示城市固定效应和年份固定效应，ε为随机扰动项。", bbox=(0, 200, 300, 230)),
        ],
        subblocks=[
            {"label": "inline_formula", "bbox": [80, 0, 110, 30]},
            {"label": "inline_formula", "bbox": [95, 40, 155, 70]},
            # Missing "$ Incentive_{c} $" box on physical line 2.
            {"label": "inline_formula", "bbox": [90, 120, 130, 150]},
            {"label": "inline_formula", "bbox": [40, 160, 70, 190]},
            {"label": "inline_formula", "bbox": [50, 200, 80, 230]},
            {"label": "inline_formula", "bbox": [110, 200, 140, 230]},
            {"label": "inline_formula", "bbox": [230, 200, 270, 230]},
        ],
    )

    assert [(item.line_index, item.text) for item in recovered] == [
        (0, "$ Y_{ct} $"),
        (1, "$ Incentive_{c} \\times Post_{t} $"),
        (3, "$ Post_{t} $"),
        (4, "$ X_{ct} $"),
        (5, "$ \\delta_{c} $"),
        (5, "$ \\varphi_{t} $"),
        (5, "$ \\varepsilon_{ct} $"),
    ]

    print("test_paddle_line_routing_missing_formula_box_does_not_shift_later_rows PASSED")


def test_paddle_line_routing_marker_formula_does_not_cut_text_slice():
    from app.core.paddle_line_routing import build_layout_line_routes, text_slice_routes_for_block

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 300, 40],
        "block_content": "甲 $ ^{②} $ 乙 $ B $ 丙",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [70, 0, 100, 30]},
            {"block_label": "inline_formula", "block_bbox": [170, 0, 200, 30]},
        ],
    }

    routes = build_layout_line_routes(block, 320, 60)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    text_slice_bboxes = [
        route["bbox"]
        for route in text_slice_routes_for_block(block, 320, 60)
    ]

    assert [segment["text"] for segment in formula_segments] == ["$ B $"]
    assert [70, 0, 100, 30] not in text_slice_bboxes
    assert [0, 0, 170, 30] in text_slice_bboxes

    print("test_paddle_line_routing_marker_formula_does_not_cut_text_slice PASSED")


def test_paddle_line_routing_cached_marker_formula_routes_are_rebuilt():
    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD, line_routes_for_block

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 300, 40],
        "block_content": "甲 $ ^{②} $ 乙 $ B $ 丙",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [70, 0, 100, 30]},
            {"block_label": "inline_formula", "block_bbox": [170, 0, 200, 30]},
        ],
        LAYOUT_LINE_ROUTES_FIELD: [
            {
                "bbox": [0, 0, 300, 40],
                "segments": [
                    {"kind": "text", "bbox": [0, 0, 70, 40]},
                    {"kind": "formula", "bbox": [70, 0, 100, 40], "text": "$ ^{②} $"},
                    {"kind": "text", "bbox": [100, 0, 170, 40]},
                    {"kind": "formula", "bbox": [170, 0, 200, 40], "text": "$ B $"},
                    {"kind": "text", "bbox": [200, 0, 300, 40]},
                ],
            }
        ],
    }

    routes = line_routes_for_block(block, 320, 60)
    formula_texts = [
        segment["text"]
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    text_bboxes = [
        segment["bbox"]
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "text"
    ]

    assert formula_texts == ["$ B $"]
    assert [0, 0, 170, 30] in text_bboxes
    assert all(segment.get("text") != "$ ^{②} $" for route in routes for segment in route["segments"])

    print("test_paddle_line_routing_cached_marker_formula_routes_are_rebuilt PASSED")


def test_paddle_line_routing_marker_formula_from_120169_does_not_eat_zero():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import line_routes_for_block, text_slice_routes_for_block
    from app.core.raw_ocr_artifact import layout_records_with_route_attachments
    from app.models import Page

    project_root = Path(__file__).resolve().parents[1]
    raw = json.loads((project_root / "file" / "244771纵校" / "120169.layout-api.json").read_text(encoding="utf-8"))
    page_info = raw["page"]
    page = Page(
        image_path=str(project_root / page_info["display_image_path"]),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    LayoutAnalyzer()._extract_api_blocks(page, raw["response"])

    parent = layout_records_with_route_attachments(page)[12]
    routes = line_routes_for_block(parent, page.width, page.height)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    text_slice_bboxes = [
        tuple(route["bbox"])
        for route in text_slice_routes_for_block(parent, page.width, page.height)
    ]

    assert [segment["text"] for segment in formula_segments] == ["$ Time_{t} $", "$ \\beta_{t} $"]
    assert (192, 2737, 619, 2793) not in text_slice_bboxes
    assert (619, 2737, 672, 2793) not in text_slice_bboxes
    assert (192, 2710, 2051, 2737) not in text_slice_bboxes
    assert (192, 2785, 1273, 2793) not in text_slice_bboxes
    assert (192, 2737, 1273, 2793) in text_slice_bboxes

    print("test_paddle_line_routing_marker_formula_from_120169_does_not_eat_zero PASSED")


def test_paddle_line_routing_ppocr_prefiltered_marker_keeps_later_formula_text():
    from app.core.paddle_line_routing import (
        PageOcrLineHint,
        attach_page_ocr_line_routes,
    )

    parent = {
        "block_label": "text",
        "block_bbox": [192, 2650, 2051, 2878],
        "block_content": (
            "其中， $ Time_{t} $ 为以样本数据第一年（2010年）为基期构建的改革时点变量，"
            "当年份为 t 时取 1，否则取 0。结果显示 $ ^{②} $，相比基期年， "
            "$ \\beta_{t} $ 在 2011~2015 年大致位于 0 附近"
        ),
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [434, 2658, 547, 2710]},
            {"block_label": "inline_formula", "block_bbox": [619, 2737, 672, 2785]},
            {"block_label": "inline_formula", "block_bbox": [1273, 2740, 1325, 2793]},
        ],
    }
    lines = [
        PageOcrLineHint(
            text="其中，Time,为以样本数据第一年(2010年)为基期构建的改革时点变量,当年份",
            bbox=(296, 2642, 2043, 2718),
        ),
        PageOcrLineHint(
            text="为t时取1,否则取0。结果显示②，相比基期年，β{在2011~2015年大致位于0附近",
            bbox=(198, 2715, 2046, 2796),
        ),
    ]

    attach_page_ocr_line_routes([parent], lines, 2320, 3416)

    formula_segments = [
        segment
        for route in parent["_layout_line_routes"]
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    text_slice_bboxes = [
        tuple(segment["bbox"])
        for route in parent["_layout_line_routes"]
        for segment in route["segments"]
        if segment["kind"] == "text"
    ]

    assert [segment["text"] for segment in formula_segments] == ["$ Time_{t} $", "$ \\beta_{t} $"]
    assert all(segment["text"] != "$ ^{②} $" for segment in formula_segments)
    assert (198, 2715, 1273, 2796) in text_slice_bboxes

    print("test_paddle_line_routing_ppocr_prefiltered_marker_keeps_later_formula_text PASSED")


def test_paddle_line_routing_complete_formula_geometry_ignores_ppocr_text():
    from app.core.paddle_line_routing import (
        PageOcrLineHint,
        attach_page_ocr_line_routes,
    )

    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 160, 40],
        "block_content": "甲 $ A $ 乙",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [45, 0, 75, 40]},
        ],
    }

    attach_page_ocr_line_routes(
        [parent],
        [PageOcrLineHint(text="甲 $ WRONG $ 乙", bbox=(0, 0, 160, 40))],
        200,
        80,
    )

    formula_segments = [
        segment
        for route in parent["_layout_line_routes"]
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]

    assert [segment["text"] for segment in formula_segments] == ["$ A $"]

    print("test_paddle_line_routing_complete_formula_geometry_ignores_ppocr_text PASSED")


def test_paddle_line_routing_merges_ppocr_fragments_as_bbox_only_physical_row():
    from app.core.paddle_line_routing import (
        PageOcrLineHint,
        attach_page_ocr_line_routes,
    )

    parent = {
        "block_label": "text",
        "block_bbox": [180, 555, 2060, 650],
        "block_content": "其中， $ GGF_{it}^{Post-short} $ ，均为虚拟变量， $ GGF_{it}^{Post-long} $ 在企业获得政府引导基金",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [445, 556, 884, 620]},
            {"block_label": "inline_formula", "block_bbox": [1242, 556, 1453, 620]},
        ],
    }
    ppocr_fragments = [
        PageOcrLineHint(text="其中，GGF", bbox=(295, 536, 792, 623)),
        PageOcrLineHint(text="Post-long", bbox=(766, 546, 901, 604)),
        PageOcrLineHint(text="均为虚拟变量，GGF", bbox=(875, 547, 1347, 626)),
        PageOcrLineHint(text="Post-short", bbox=(1326, 555, 1461, 606)),
        PageOcrLineHint(text="在企业获得政府引导基金", bbox=(1438, 556, 2052, 626)),
        PageOcrLineHint(text="it", bbox=(536, 587, 560, 614)),
        PageOcrLineHint(text="it", bbox=(1334, 594, 1358, 621)),
    ]

    attach_page_ocr_line_routes([parent], ppocr_fragments, 2320, 3416)

    routes = parent["_layout_line_routes"]
    assert len(routes) == 1
    assert routes[0]["bbox"] == [295, 555, 2052, 626]
    assert [segment["kind"] for segment in routes[0]["segments"]] == [
        "text",
        "formula",
        "text",
        "formula",
        "text",
    ]
    assert [segment["bbox"] for segment in routes[0]["segments"]] == [
        [295, 555, 445, 626],
        [445, 556, 884, 620],
        [884, 555, 1242, 626],
        [1242, 556, 1453, 620],
        [1453, 555, 2052, 626],
    ]
    assert [segment["text"] for segment in routes[0]["segments"] if segment["kind"] == "formula"] == [
        "$ GGF_{it}^{Post-short} $",
        "$ GGF_{it}^{Post-long} $",
    ]

    print("test_paddle_line_routing_merges_ppocr_fragments_as_bbox_only_physical_row PASSED")


def test_paddle_line_routing_restores_inter_formula_punctuation_gap():
    from app.core.paddle_line_routing import line_routes_for_block, text_slice_routes_for_block

    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 180, 50],
        "block_content": "甲 $ A $、$ B $ 乙",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [40, 0, 82, 40]},
            {"block_label": "inline_formula", "block_bbox": [78, 0, 120, 40]},
        ],
    }

    routes = line_routes_for_block(parent, 200, 80)
    segments = routes[0]["segments"]
    gap_segments = [
        segment for segment in segments
        if segment.get("kind") == "text" and segment.get("label") == "inter_formula_text_gap"
    ]

    assert len(gap_segments) == 1
    assert gap_segments[0]["text"] == "、"
    assert gap_segments[0]["bbox"][0] < 82
    assert gap_segments[0]["bbox"][2] > 78
    assert gap_segments[0]["bbox"][2] - gap_segments[0]["bbox"][0] >= 16
    assert any(route["bbox"] == gap_segments[0]["bbox"] for route in text_slice_routes_for_block(parent, 200, 80))

    print("test_paddle_line_routing_restores_inter_formula_punctuation_gap PASSED")


def test_paddle_line_routing_page_ocr_miss_invalidates_cached_routes():
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        PageOcrLineHint,
        attach_page_ocr_line_routes,
        text_slice_routes_for_block,
    )

    parent = {
        "block_label": "text",
        "block_bbox": [0, 0, 100, 100],
        "block_content": "甲 $ A $ 乙",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [30, 10, 50, 40]},
        ],
        LAYOUT_LINE_ROUTES_FIELD: [
            {
                "bbox": [0, 80, 100, 95],
                "segments": [{"kind": "text", "bbox": [0, 80, 100, 95]}],
            }
        ],
    }

    attach_page_ocr_line_routes(
        [parent],
        [PageOcrLineHint(text="unrelated", bbox=(200, 200, 260, 230))],
        300,
        300,
    )

    assert LAYOUT_LINE_ROUTES_FIELD not in parent
    text_slice_bboxes = [
        tuple(route["bbox"])
        for route in text_slice_routes_for_block(parent, 300, 300)
    ]

    assert (0, 80, 100, 95) not in text_slice_bboxes
    assert (0, 10, 30, 40) in text_slice_bboxes

    print("test_paddle_line_routing_page_ocr_miss_invalidates_cached_routes PASSED")


def test_paddle_line_routing_formula_number_is_skip_not_formula_carrier():
    from app.core.paddle_line_routing import build_layout_line_routes

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 230, 40],
        "block_content": "正文 $ A $ 公式编号",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [60, 0, 90, 30]},
            {"block_label": "formula_number", "block_bbox": [180, 0, 220, 30], "block_content": "(1)"},
        ],
    }

    routes = build_layout_line_routes(block, 240, 60)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    skip_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "skip"
    ]

    assert [(segment["label"], segment["text"]) for segment in formula_segments] == [
        ("inline_formula", "$ A $"),
    ]
    assert [(segment["label"], segment["text"]) for segment in skip_segments] == [
        ("formula_number", "(1)"),
    ]

    print("test_paddle_line_routing_formula_number_is_skip_not_formula_carrier PASSED")


def test_paddle_line_routing_formula_rows_use_single_horizontal_band():
    from app.core.paddle_line_routing import build_layout_line_routes

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 220, 80],
        "block_content": "前 $ A $ 中 $ B $ 后 $ C $ 末",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [40, 10, 70, 34]},
            {"block_label": "inline_formula", "block_bbox": [95, 14, 125, 42]},
            {"block_label": "inline_formula", "block_bbox": [160, 8, 190, 36]},
        ],
    }

    routes = build_layout_line_routes(block, 240, 100)
    formula_route = next(
        route for route in routes
        if sum(1 for segment in route["segments"] if segment["kind"] == "formula") == 3
    )

    assert [segment["text"] for segment in formula_route["segments"] if segment["kind"] == "formula"] == [
        "$ A $",
        "$ B $",
        "$ C $",
    ]
    assert [
        segment["bbox"]
        for segment in formula_route["segments"]
        if segment["kind"] == "formula"
    ] == [
        [40, 10, 70, 34],
        [95, 14, 125, 42],
        [160, 8, 190, 36],
    ]
    assert [segment["kind"] for segment in formula_route["segments"]] == [
        "text",
        "formula",
        "text",
        "formula",
        "text",
        "formula",
        "text",
    ]

    print("test_paddle_line_routing_formula_rows_use_single_horizontal_band PASSED")


def test_paddle_line_routing_has_layout_routes_is_pure():
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        has_layout_line_routes,
        line_routes_for_block,
    )

    block = {
        "block_label": "text",
        "block_bbox": [0, 0, 120, 30],
        "block_content": "before $ A $ after",
        "_route_subblocks": [
            {"block_label": "inline_formula", "block_bbox": [45, 0, 70, 30]},
        ],
    }

    assert LAYOUT_LINE_ROUTES_FIELD not in block
    assert has_layout_line_routes(block, 140, 50) is True
    assert LAYOUT_LINE_ROUTES_FIELD not in block

    routes = line_routes_for_block(block, 140, 50)
    assert routes
    assert LAYOUT_LINE_ROUTES_FIELD not in block

    stale_routes = [
        {
            "bbox": [90, 0, 120, 30],
            "segments": [{"kind": "text", "bbox": [90, 0, 120, 30], "text": ""}],
        }
    ]
    block[LAYOUT_LINE_ROUTES_FIELD] = stale_routes
    routes = line_routes_for_block(block, 140, 50)
    assert [route["bbox"] for route in routes] != [[90, 0, 120, 30]]
    assert block.get(LAYOUT_LINE_ROUTES_FIELD) is stale_routes

    print("test_paddle_line_routing_has_layout_routes_is_pure PASSED")


def test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        ROUTE_SUBBLOCKS_FIELD,
        line_routes_for_block,
    )
    from app.core.raw_ocr_artifact import layout_records_with_route_attachments
    from app.models import Page

    project_root = Path(__file__).resolve().parents[1]
    layout_json = (
        project_root
        / "tests"
        / "fixtures"
        / "layout"
        / "120166-layout-api-fixture.json"
    )
    raw = json.loads(layout_json.read_text(encoding="utf-8"))
    page_info = raw["page"]
    sample = project_root / page_info["display_image_path"]
    page = Page(
        image_path=str(sample),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )

    LayoutAnalyzer()._extract_api_blocks(page, raw["response"])

    skip_records = [
        record for record in _raw_layout_records(page)
        if record.get("block_label") in {"display_formula", "formula_number"}
    ]
    assert skip_records
    assert all(ROUTE_SUBBLOCKS_FIELD not in record for record in skip_records)
    assert all(LAYOUT_LINE_ROUTES_FIELD not in record for record in skip_records)

    routed_records = layout_records_with_route_attachments(page)
    text_record = next(
        record for record in routed_records
        if record.get("block_label") == "text" and "$ Y_{ct} $" in str(record.get("block_content") or "")
    )
    assert len(text_record[ROUTE_SUBBLOCKS_FIELD]) == 7

    formula_routes = [
        route for route in line_routes_for_block(text_record, page.width, page.height)
        if any(segment["kind"] == "formula" for segment in route["segments"])
    ]
    assert len(formula_routes) == 5
    three_formula_route = next(
        route for route in formula_routes
        if sum(1 for segment in route["segments"] if segment["kind"] == "formula") == 3
    )
    assert [segment["text"] for segment in three_formula_route["segments"] if segment["kind"] == "formula"] == [
        "",
        "",
        "",
    ]
    assert [segment["bbox"] for segment in three_formula_route["segments"] if segment["kind"] == "formula"] == [
        [623, 2287, 669, 2341],
        [729, 2295, 776, 2343],
        [1737, 2293, 1793, 2337],
    ]

    print("test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands PASSED")


def test_layout_fixture_page_ocr_routes_do_not_shift_after_missing_formula_box():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        attach_page_ocr_line_routes,
    )
    from app.core.raw_ocr_artifact import layout_records_with_route_attachments
    from app.models import BBox, Line, Page

    project_root = Path(__file__).resolve().parents[1]
    layout_json = (
        project_root
        / "tests"
        / "fixtures"
        / "layout"
        / "120166-layout-api-fixture.json"
    )
    ocr_lines_json = (
        project_root
        / "tests"
        / "fixtures"
        / "layout"
        / "120166-page-ocr-lines.json"
    )
    raw = json.loads(layout_json.read_text(encoding="utf-8"))
    page_info = raw["page"]
    sample = project_root / page_info["display_image_path"]
    page = Page(
        image_path=str(sample),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    LayoutAnalyzer()._extract_api_blocks(page, raw["response"])

    ocr_raw = json.loads(ocr_lines_json.read_text(encoding="utf-8"))
    page_ocr_lines = []
    for row in ocr_raw["lines"]:
        page_ocr_lines.append(
            Line(
                text=str(row["text"]),
                confidence=float(row["score"]),
                bbox=BBox.from_xyxy(*row["bbox"]),
            )
        )

    routed_records = layout_records_with_route_attachments(page)
    attach_page_ocr_line_routes(
        routed_records,
        page_ocr_lines,
        page.width,
        page.height,
    )
    text_record = next(
        record for record in routed_records
        if record.get("block_label") == "text" and "$ Y_{ct} $" in str(record.get("block_content") or "")
    )
    formula_texts = [
        segment["text"]
        for route in text_record[LAYOUT_LINE_ROUTES_FIELD]
        for segment in route["segments"]
        if segment["kind"] == "formula" and segment.get("text")
    ]

    assert formula_texts == [
        "$ Y_{ct} $",
        "$ Incentive_{c} \\times Post_{t} $",
        "$ Post_{t} $",
        "$ X_{ct} $",
        "$ \\delta_{c} $",
        "$ \\varphi_{t} $",
        "$ \\varepsilon_{ct} $",
    ]

    final_formula_route = next(
        route for route in text_record[LAYOUT_LINE_ROUTES_FIELD]
        if route["bbox"][1] == 2271
    )
    assert [segment["text"] for segment in final_formula_route["segments"] if segment["kind"] == "formula"] == [
        "$ \\delta_{c} $",
        "$ \\varphi_{t} $",
        "$ \\varepsilon_{ct} $",
    ]

    print("test_layout_fixture_page_ocr_routes_do_not_shift_after_missing_formula_box PASSED")


def test_paddle_artifact_index_marks_missing_inline_formula_for_crop_ocr():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_artifact_index import (
        BINDING_EMPTY_REVIEW,
        BINDING_GEOMETRY_HIT,
        PaddleArtifactIndex,
    )
    from app.models import BBox, BlockType, Page

    project_root = Path(__file__).resolve().parents[1]
    layout_json = (
        project_root
        / "tests"
        / "fixtures"
        / "layout"
        / "120166-layout-api-fixture.json"
    )
    raw = json.loads(layout_json.read_text(encoding="utf-8"))
    page_info = raw["page"]
    page = Page(
        image_path=str(project_root / page_info["display_image_path"]),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    LayoutAnalyzer()._extract_api_blocks(page, raw["response"])

    index = PaddleArtifactIndex.from_page(page)
    geometry = index.bind_manual_bbox(
        BBox.from_xyxy(1030, 1888, 1165, 1960),
        BlockType.EQUATION,
    )
    missing = index.bind_manual_bbox(
        BBox.from_xyxy(292, 1958, 620, 2038),
        BlockType.EQUATION,
    )

    assert geometry.status == BINDING_GEOMETRY_HIT
    assert geometry.text == ""
    assert geometry.ocr_policy != OcrPolicy.TEXT_OCR
    assert "manual_formula_needs_text" in geometry.review_flags
    assert missing.status == BINDING_EMPTY_REVIEW
    assert missing.text == ""
    assert "manual_formula_needs_text" in missing.review_flags

    print("test_paddle_artifact_index_marks_missing_inline_formula_for_crop_ocr PASSED")


def test_paddle_artifact_index_binds_parent_table_and_empty_formula_review():
    from app.core.paddle_artifact_index import (
        BINDING_EMPTY_REVIEW,
        BINDING_PARENT_TABLE_HIT,
        PaddleArtifactIndex,
    )
    from app.models import BBox, BlockType, Page

    page = Page(image_path="/tmp/table-page.png", width=300, height=220)
    _attach_raw_layout_records(page, [
        {
            "block_label": "table",
            "block_bbox": [40, 50, 260, 160],
            "block_content": "<table><tr><td>A</td></tr></table>",
        },
        {
            "block_label": "text",
            "block_bbox": [40, 170, 260, 205],
            "block_content": "plain text without formula",
        },
    ])

    index = PaddleArtifactIndex.from_page(page)
    table = index.bind_manual_bbox(BBox.from_xyxy(35, 45, 265, 165), BlockType.TABLE)
    empty_formula = index.bind_manual_bbox(BBox.from_xyxy(50, 174, 120, 198), BlockType.EQUATION)

    assert table.status == BINDING_PARENT_TABLE_HIT
    assert table.text == "<table><tr><td>A</td></tr></table>"
    assert table.source_label == "table"
    assert table.ocr_policy != OcrPolicy.TEXT_OCR
    assert empty_formula.status == BINDING_EMPTY_REVIEW
    assert empty_formula.text == ""
    assert "manual_formula_needs_text" in empty_formula.review_flags

    print("test_paddle_artifact_index_binds_parent_table_and_empty_formula_review PASSED")


def test_layout_edit_service_manual_formula_writes_typed_paddle_binding():
    from app.core.paddle_artifact_index import BINDING_EMPTY_REVIEW
    from app.models import BBox, Block, BlockSource, BlockType, Page
    from app.services.layout_edit_service import LayoutEditService

    page = Page(image_path="/tmp/page.png", width=300, height=120)
    _attach_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [10, 10, 260, 70],
            "block_content": "甲 $ A $ 乙 $ B $ 丙",
            "_route_subblocks": [
                {"block_label": "inline_formula", "block_bbox": [60, 12, 90, 40]},
            ],
        }
    ])
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox.from_xyxy(120, 12, 150, 42),
        source=BlockSource.MANUAL_DRAW,
    )

    LayoutEditService().bind_manual_block_to_paddle(page, block)

    binding = block.paddle_binding
    assert binding is not None
    assert binding.status == BINDING_EMPTY_REVIEW
    assert binding.text == ""
    assert block.origin is not None
    assert block.origin.source_label == "inline_formula"
    assert block.origin.raw_index == 0
    assert block.source_label == "inline_formula"
    assert block.ocr_policy != OcrPolicy.TEXT_OCR
    assert block.lines == []

    print("test_layout_edit_service_manual_formula_writes_typed_paddle_binding PASSED")


def test_inline_formula_crop_targets_use_structured_label_not_payload_labels():
    from app.engines.hanwang.micro_recblock import _inline_formula_crop_ocr_targets
    from app.models import BBox, Block, BlockType, Page

    structured_display = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(10, 10, 50, 20),
        source_label="display_formula",
    )
    structured_inline = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(70, 10, 50, 20),
        source_label="inline_formula",
    )
    page = Page(
        image_path="/tmp/inline-target-labels.png",
        width=200,
        height=100,
        blocks=[structured_display, structured_inline],
    )

    assert _inline_formula_crop_ocr_targets(page) == [structured_inline]

    print("test_inline_formula_crop_targets_use_structured_label_not_payload_labels PASSED")


def test_layout_analyzer_reads_formula_geometry_boxes_for_routes():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.core.raw_ocr_artifact import layout_route_attachments, layout_records_with_route_attachments
    from app.models import Page

    page = Page(image_path="/tmp/formula-geometry-key.png", width=160, height=80)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {"label": "inline_formula", "coordinate": [50, 10, 80, 32]},
                            ],
                        },
                        "parsing_res_list": [
                            {"block_label": "text", "block_bbox": [10, 0, 140, 50], "block_content": "甲 $ A $ 乙"},
                        ],
                    },
                },
            ],
        },
    }

    blocks, overlays = LayoutAnalyzer()._extract_api_blocks(page, data)
    assert ROUTE_SUBBLOCKS_FIELD not in _raw_layout_records(page)[0]
    subblocks = layout_route_attachments(page)[0]
    routed_records = layout_records_with_route_attachments(page)

    assert not hasattr(blocks[0], "raw_payload")
    assert [(item["block_label"], item["block_bbox"]) for item in subblocks] == [
        ("inline_formula", [50, 10, 80, 32]),
    ]
    assert routed_records[0][ROUTE_SUBBLOCKS_FIELD] == subblocks
    assert [label for label, _bbox in overlays] == ["text", "inline_formula"]

    print("test_layout_analyzer_reads_formula_geometry_boxes_for_routes PASSED")


def test_hanwang_inline_formula_empty_text_slices_keeps_empty_hanwang_result():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {"lines": []}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    micro_module.native_bridge.run_linecut_segimg = fake_segimg

    try:
        parent_text = "前文正文 $ x+y $ 后文正文"
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((40, 140, 3), dtype=np.uint8),
            [{
                "block_label": "text",
                "block_bbox": [0, 0, 120, 20],
                "block_content": parent_text,
                "_route_subblocks": [
                    {
                        "block_label": "inline_formula",
                        "block_bbox": [40, 0, 70, 20],
                        "raw_payload": {"block_label": "inline_formula"},
                    }
                ],
            }],
            include_chars=True,
        )

        assert len(rows) == 1
        assert rows[0].source == "hanwang"
        assert rows[0].text == ""
        assert rows[0].lines == []
        assert rows[0].fallback_reason == ""
        assert stats.n_blocks_fallback == 0
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg

    print("test_hanwang_inline_formula_empty_text_slices_keeps_empty_hanwang_result PASSED")


def test_hanwang_group_chunk_cannot_readmit_skipped_subregions():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [
                {"groups": [
                    {"bbox": {"left": 40, "top": 0, "right": 60, "bottom": 20}},
                    {"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}},
                ]}
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    recognized_recblocks = []

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        recognized_recblocks.extend(blocks)
        return {
            "lines": [
                {"groups": [{
                    "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    "chars": [{
                        "codes": [code("字")],
                        "scores": [5],
                        "bbox": {"left": x1, "top": y1, "right": min(x2, x1 + 10), "bottom": y2},
                    }],
                }]}
                for x1, y1, x2, y2 in blocks
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

    try:
        blocks = [{
            "block_label": "text",
            "block_bbox": [0, 0, 100, 20],
            "_route_subblocks": [
                {"block_label": "inline_formula", "block_bbox": [40, 0, 60, 20]},
            ],
        }]
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((40, 120, 3), dtype=np.uint8),
            blocks,
            include_chars=True,
        )

        assert recognized_recblocks
        assert len(rows) == len(blocks)
        assert [row.source for row in rows] == ["hanwang"]
        assert rows[0].block_bbox == (0, 0, 100, 20)
        assert stats.n_groups == 4
        assert stats.recog_probe_calls == 2
        assert stats.recog_batch_disabled is True
        assert all(
            not (char.bbox and 40 <= (char.bbox[0] + char.bbox[2]) / 2 <= 60)
            for line in rows[0].lines
            for char in line.chars
        )
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_group_chunk_cannot_readmit_skipped_subregions PASSED")


def test_hanwang_formula_style_footer_bypasses_hanwang():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    called_segimg = False

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        nonlocal called_segimg
        called_segimg = True
        return {"lines": []}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    micro_module.native_bridge.run_linecut_segimg = fake_segimg

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [
                {
                    "block_label": "footer",
                    "block_bbox": [10, 20, 90, 38],
                    "block_content": " $  \\frac{1}{2}  $",
                },
            ],
        )

        assert called_segimg is False
        assert len(rows) == 1
        assert rows[0].source == "ppvl"
        assert rows[0].block_label == "formula"
        assert rows[0].lines[0].text == "$  \\frac{1}{2}  $"
        assert rows[0].lines[0].chars == []
        assert stats.n_blocks_hanwang == 0
        assert stats.n_blocks_ppvl == 1
        assert stats.n_blocks_fallback == 0
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg

    print("test_hanwang_formula_style_footer_bypasses_hanwang PASSED")


def test_hanwang_footnote_labels_route_through_hanwang():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    captured_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        captured_recblocks.extend(recblocks_xyxy or [])
        return {
            "lines": [
                {"groups": [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]}
                for x1, y1, x2, y2 in (recblocks_xyxy or [])
            ]
        }

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        h, w = image_bgr.shape[:2]
        return {
            "lines": [
                {"groups": [{
                    "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                    "chars": [{
                        "codes": [code("注")],
                        "scores": [5],
                        "bbox": {"left": 0, "top": 0, "right": min(20, w), "bottom": h},
                    }],
                }]}
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module._BATCH_DISABLED_FOR_SESSION = True

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((100, 300, 3), dtype=np.uint8),
            [
                {
                    "block_label": "vision_footnote",
                    "block_bbox": [10, 10, 250, 40],
                    "block_content": "注",
                },
                {
                    "block_label": "footnote",
                    "block_bbox": [10, 50, 280, 90],
                    "block_content": "注",
                },
            ],
            include_chars=True,
        )

        assert captured_recblocks == [(10, 10, 250, 40), (10, 50, 280, 90)]
        assert [row.source for row in rows] == ["hanwang", "hanwang"]
        assert [row.block_label for row in rows] == ["footnote", "footnote"]
        assert [row.text for row in rows] == ["注", "注"]
        assert [row.lines[0].chars[0].text for row in rows] == ["注", "注"]
        assert stats.n_blocks_hanwang == 2
        assert stats.n_blocks_ppvl == 0
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled

    print("test_hanwang_footnote_labels_route_through_hanwang PASSED")


def test_hanwang_micro_recblock_keeps_caption_labels_on_hanwang_path():
    import app.engines.hanwang.micro_recblock as micro_module

    for label in ("figure_caption", "figure_title", "table_caption", "table_title", "table_note"):
        assert micro_module._is_skip_label(label) is False
        assert micro_module._is_text_label(label) is True

    print("test_hanwang_micro_recblock_keeps_caption_labels_on_hanwang_path PASSED")


def test_hanwang_micro_recblock_unknown_label_defaults_to_text_path_with_audit():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    seen_recblocks = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        seen_recblocks.extend(recblocks_xyxy or [])
        return {"lines": [{"groups": []}]}

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        raise AssertionError("no groups should mean no recog call")

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((80, 140, 3), dtype=np.uint8),
            [
                {
                    "block_label": "new_unknown_text_kind",
                    "block_bbox": [10, 20, 110, 60],
                    "block_content": "未知标签文字",
                }
            ],
            include_chars=True,
        )

        assert seen_recblocks == [(10, 20, 110, 60)]
        assert rows[0].source == "hanwang"
        assert rows[0].text == ""
        assert rows[0].ppvl_text == "未知标签文字"
        assert rows[0].fallback_reason == ""
        assert stats.n_blocks_hanwang == 1
        assert stats.n_blocks_ppvl == 0
        assert stats.n_unknown_paddle_labels == 1
        audit = rows[0].raw_block["_hanwang_bbox_audit"]
        assert audit["paddle_label"] == "new_unknown_text_kind"
        assert audit["paddle_label_unknown"] is True
        assert audit["paddle_label_unknown_action"] == "default_text_ocr"
        assert audit["route_text_slice_count"] == 1
        assert audit["hanwang_recog_group_count"] == 0
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_micro_recblock_unknown_label_defaults_to_text_path_with_audit PASSED")


def test_hanwang_micro_recblock_circuit_breaks_after_batch_failure():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [{
                "groups": [
                    {"bbox": {"left": 10 + i * 20, "top": 10, "right": 25 + i * 20, "bottom": 30}}
                    for i in range(5)
                ]
            }]
        }

    calls = {"batch": 0, "single": 0}

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        h, w = image_bgr.shape[:2]
        calls["single"] += 1
        return {"lines": [{"groups": [{
            "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            "chars": [{
                "codes": [code("甲")],
                "scores": [5],
                "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            }],
        }]}]}

    def fake_recog_batch(*_args, **_kwargs):
        calls["batch"] += 1
        raise RuntimeError("batch-list failed")

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_recog_batch = micro_module.native_bridge.run_linecut_recog_batch_list
    original_max_groups = micro_module.MAX_RECOG_BATCH_GROUPS
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    original_batch_reason = micro_module._BATCH_DISABLE_REASON
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module.native_bridge.run_linecut_recog_batch_list = fake_recog_batch
    micro_module.MAX_RECOG_BATCH_GROUPS = 2
    micro_module._BATCH_DISABLED_FOR_SESSION = False
    micro_module._BATCH_DISABLE_REASON = ""

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((80, 140, 3), dtype=np.uint8),
            [{"block_label": "text", "block_bbox": [0, 0, 140, 80], "block_content": "甲甲甲甲甲"}],
            include_chars=True,
        )

        assert rows[0].source == "hanwang"
        assert rows[0].text == "甲甲甲甲甲"
        assert calls == {"batch": 1, "single": 5}
        assert stats.recog_batch_chunks == 3
        assert stats.recog_batch_failures == 1
        assert stats.recog_batch_disabled is True
        assert stats.recog_probe_calls == 6
        assert micro_module._BATCH_DISABLED_FOR_SESSION is True
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module.native_bridge.run_linecut_recog_batch_list = original_recog_batch
        micro_module.MAX_RECOG_BATCH_GROUPS = original_max_groups
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled
        micro_module._BATCH_DISABLE_REASON = original_batch_reason

    print("test_hanwang_micro_recblock_circuit_breaks_after_batch_failure PASSED")


def test_hanwang_micro_recblock_batch_list_handles_wide_crops_without_collage_guard():
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    def code(ch):
        return int.from_bytes(ch.encode("gbk"), "little")

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [{
                "groups": [
                    {"bbox": {"left": 0, "top": 10, "right": 1850, "bottom": 40}},
                    {"bbox": {"left": 0, "top": 50, "right": 1850, "bottom": 80}},
                ]
            }]
        }

    calls = {"batch": 0, "single": 0}

    def fake_recog_batch(images_bgr, *, with_charrcg=True, timeout=0, **_kwargs):
        calls["batch"] += 1
        assert [tuple(image.shape[:2]) for image in images_bgr] == [(50, 1858), (50, 1858)]
        raws = []
        for image in images_bgr:
            h, w = image.shape[:2]
            raws.append({"lines": [{"groups": [{
                "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                "chars": [{
                    "codes": [code("乙")],
                    "scores": [5],
                    "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                }],
            }]}]})
        return raws

    def fake_recog_single(image_bgr, **_kwargs):
        calls["single"] += 1
        h, w = image_bgr.shape[:2]
        return {"lines": [{"groups": [{
            "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            "chars": [{
                "codes": [code("乙")],
                "scores": [5],
                "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            }],
        }]}]}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_recog_batch = micro_module.native_bridge.run_linecut_recog_batch_list
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    original_batch_reason = micro_module._BATCH_DISABLE_REASON
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog_single
    micro_module.native_bridge.run_linecut_recog_batch_list = fake_recog_batch
    micro_module._BATCH_DISABLED_FOR_SESSION = False
    micro_module._BATCH_DISABLE_REASON = ""

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((120, 2000, 3), dtype=np.uint8),
            [{"block_label": "text", "block_bbox": [0, 0, 2000, 120], "block_content": "乙乙"}],
            include_chars=True,
        )

        assert rows[0].source == "hanwang"
        assert calls == {"batch": 1, "single": 0}
        assert stats.recog_batch_chunks == 1
        assert stats.recog_batch_guarded_chunks == 0
        assert stats.recog_batch_failures == 0
        assert stats.recog_batch_disabled is False
        assert stats.recog_probe_calls == 1
        assert stats.recog_max_batch_crop_width == 1858
        assert stats.recog_max_batch_crop_height == 50
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module.native_bridge.run_linecut_recog_batch_list = original_recog_batch
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled
        micro_module._BATCH_DISABLE_REASON = original_batch_reason

    print("test_hanwang_micro_recblock_batch_list_handles_wide_crops_without_collage_guard PASSED")


def test_ocr_pipeline_runs_hanwang_micro_recblock_page_path():
    import tempfile
    import cv2
    import numpy as np
    from app.engines.hanwang.micro_recblock import (
        BlockResult, CharResult, HanwangMicroRecBlockEngine, LineResult, RunStats,
    )
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.models import BlockOrigin
    from app.services.ocr_pipeline import OcrPipeline

    calls = []

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        calls.append((ppvl_blocks, kwargs))
        kwargs["progress_callback"](1, 3, "Hanwang group 1/3")
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(10, 20, 110, 60),
                source="hanwang",
                text="汉王",
                ppvl_text="PPVL文本",
                group_count=1,
                raw_block={"block_label": "text", "block_content": "PPVL文本", "extra": {"role": "body"}},
                lines=[
                    LineResult(
                        text="汉王",
                        bbox=(12, 24, 90, 58),
                        confidence=0.93,
                        chars=[
                            CharResult(text="汉", confidence=0.95, bbox=(12, 24, 38, 58)),
                            CharResult(text="王", confidence=0.91, bbox=(42, 24, 68, 58)),
                        ],
                    )
                ],
            ),
            BlockResult(
                block_idx=1,
                block_label="display_formula",
                block_bbox=(20, 80, 180, 120),
                source="ppvl",
                text="$$x+y$$",
                ppvl_text="$$x+y$$",
                raw_block={"block_label": "display_formula", "block_content": "$$x+y$$", "formula_format": "latex"},
                lines=[LineResult(text="$$x+y$$", bbox=(20, 80, 180, 120), source="ppvl")],
            ),
            BlockResult(
                block_idx=2,
                block_label="reference",
                block_bbox=(20, 140, 180, 180),
                source="hanwang",
                text="",
                ppvl_text="参考文献",
                raw_block={"block_label": "reference", "block_content": "参考文献", "ref_level": 1},
                lines=[],
            ),
        ], RunStats(n_blocks_total=3, n_blocks_hanwang=2, n_blocks_ppvl=1, n_blocks_fallback=0)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        cv2.imwrite(img_path, np.ones((220, 240, 3), dtype=np.uint8) * 255)

    class FakePrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="预识别", bbox=BBox.from_xyxy(10, 20, 110, 60), confidence=0.9)]

    try:
        ppvl_records = [
            {"block_label": "text", "block_bbox": [10, 20, 110, 60], "block_content": "PPVL文本"},
            {"block_label": "display_formula", "block_bbox": [20, 80, 180, 120], "block_content": "$$x+y$$"},
            {"block_label": "reference", "block_bbox": [20, 140, 180, 180], "block_content": "参考文献"},
        ]
        page = Page(
            image_path=img_path,
            width=240,
            height=220,
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(10, 20, 110, 60),
                    source_label="text",
                    origin=BlockOrigin(source_label="text", raw_index=0),
                ),
                Block(
                    block_type=BlockType.EQUATION,
                    bbox=BBox.from_xyxy(20, 80, 180, 120),
                    source_label="display_formula",
                    origin=BlockOrigin(source_label="display_formula", raw_index=1),
                ),
                Block(
                    block_type=BlockType.REFERENCE,
                    bbox=BBox.from_xyxy(20, 140, 180, 180),
                    source_label="reference",
                    origin=BlockOrigin(source_label="reference", raw_index=2),
                ),
            ],
            raw_layout_artifact=_paddle_layout_artifact(ppvl_records),
        )
        progress_events = []
        result = OcrPipeline(
            engine=HanwangMicroRecBlockEngine(runner=fake_runner),
            hybrid_prepass_engine=FakePrepassEngine(),
        ).process_project(
            OcrProject(name="hybrid", pages=[page]),
            progress_callback=progress_events.append,
        )

        assert len(calls) == 1
        assert calls[0][0] is not _raw_layout_records(page)
        assert [block["block_label"] for block in calls[0][0]] == ["text", "display_formula", "reference"]
        assert [block["block_bbox"] for block in calls[0][0]] == [
            [10, 20, 110, 60],
            [20, 80, 180, 120],
            [20, 140, 180, 180],
        ]
        assert [block["block_content"] for block in calls[0][0]] == ["PPVL文本", "$$x+y$$", "参考文献"]
        assert calls[0][1]["include_chars"] is True
        assert len(calls[0][1]["page_ocr_lines"]) == 1
        assert calls[0][1]["page_ocr_lines"][0].text == "预识别"
        assert any("PP-OCRv5 page-line prepass complete: 1 lines" in event.message for event in progress_events)
        assert any(
            event.current_block == 1
            and event.total_blocks == 3
            and event.completed_pages == 0
            and event.message == "Hanwang group 1/3"
            for event in progress_events
        )
        out_page = result.pages[0]
        assert [block.block_type for block in out_page.blocks] == [
            BlockType.TEXT,
            BlockType.EQUATION,
            BlockType.REFERENCE,
        ]
        assert out_page.blocks[0].lines[0].text == "汉王"
        assert out_page.blocks[0].source_label == "text"
        assert not hasattr(out_page.blocks[0], "raw_payload")
        assert out_page.blocks[0].origin is not None
        assert out_page.blocks[0].origin.raw_index == 0
        assert out_page.blocks[0].lines[0].chars[0].bbox_source == "hanwang:micro_recblock"
        assert out_page.blocks[1].lines[0].text == "$$x+y$$"
        assert out_page.blocks[1].ocr_policy != OcrPolicy.TEXT_OCR
        assert not hasattr(out_page.blocks[1], "raw_payload")
        assert out_page.blocks[1].origin is not None
        assert out_page.blocks[1].origin.raw_index == 1
        assert "fallback_reason=" not in out_page.blocks[2].note
        assert out_page.blocks[2].lines == []
        assert not hasattr(out_page.blocks[2], "raw_payload")
        assert out_page.blocks[2].origin is not None
        assert out_page.blocks[2].origin.raw_index == 2
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_runs_hanwang_micro_recblock_page_path PASSED")


def test_ocr_pipeline_runs_hanwang_prepass_when_only_layout_routes_exist():
    import os
    import tempfile
    import cv2
    import numpy as np
    from app.engines.hanwang.micro_recblock import (
        BlockResult, CharResult, HanwangMicroRecBlockEngine, LineResult, RunStats,
    )
    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    calls = []

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        calls.append(kwargs)
        line_hints = kwargs["page_ocr_lines"]
        assert len(line_hints) == 1
        assert line_hints[0].text == "PP行"
        assert LAYOUT_LINE_ROUTES_FIELD not in ppvl_blocks[0]
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(10, 20, 110, 60),
                source="hanwang",
                text="重跑结果",
                ppvl_text="PPVL文本",
                raw_block=dict(ppvl_blocks[0]),
                lines=[
                    LineResult(
                        text="重跑结果",
                        bbox=(12, 24, 90, 58),
                        confidence=0.93,
                        chars=[CharResult(text="重", confidence=0.95, bbox=(12, 24, 38, 58))],
                    )
                ],
            )
        ], RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    class FakePrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="PP行", bbox=BBox.from_xyxy(10, 20, 110, 60), confidence=0.98)]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        cv2.imwrite(img_path, np.ones((90, 140, 3), dtype=np.uint8) * 255)

    try:
        page = Page(
            image_path=img_path,
            width=140,
            height=90,
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(10, 20, 110, 60),
                    lines=[Line(text="已有行框", bbox=BBox.from_xyxy(10, 20, 110, 60), confidence=0.9)],
                )
            ],
            raw_layout_artifact=_paddle_layout_artifact([
                {
                    "block_label": "text",
                    "block_bbox": [10, 20, 110, 60],
                    "block_content": "PPVL文本",
                    LAYOUT_LINE_ROUTES_FIELD: [
                        {
                            "bbox": [10, 20, 110, 60],
                            "segments": [{"kind": "text", "bbox": [10, 20, 110, 60], "text": ""}],
                        }
                    ],
                }
            ]),
        )
        progress_events = []
        result = OcrPipeline(
            engine=HanwangMicroRecBlockEngine(runner=fake_runner),
            hybrid_prepass_engine=FakePrepassEngine(),
        ).process_project(
            OcrProject(name="hybrid-skip-prepass", pages=[page]),
            progress_callback=progress_events.append,
        )

        assert len(calls) == 1
        assert len(calls[0]["page_ocr_lines"]) == 1
        assert calls[0]["page_ocr_lines"][0].text == "PP行"
        assert result.pages[0].blocks[0].lines[0].text == "重跑结果"
        assert any(
            "PP-OCRv5 page-line prepass complete: 1 lines" in event.message
            for event in progress_events
        )
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_runs_hanwang_prepass_when_only_layout_routes_exist PASSED")


def test_ocr_pipeline_does_not_reuse_hanwang_lines_as_ppocr_hints():
    import os
    import tempfile
    import cv2
    import numpy as np
    from app.engines.hanwang.micro_recblock import (
        BlockResult, HanwangMicroRecBlockEngine, LineResult, RunStats,
    )
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    calls = []

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        calls.append(kwargs)
        line_hints = kwargs["page_ocr_lines"]
        assert len(line_hints) == 1
        assert line_hints[0].text == "PP行"
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(10, 20, 110, 60),
                source="hanwang",
                text="重跑结果",
                ppvl_text="PPVL文本",
                raw_block=dict(ppvl_blocks[0]),
                lines=[LineResult(text="重跑结果", bbox=(10, 20, 110, 60), confidence=0.93)],
            )
        ], RunStats(n_blocks_total=1, n_blocks_hanwang=1)

    class PrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return [Line(text="PP行", bbox=BBox.from_xyxy(10, 20, 110, 60), confidence=0.91)]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        cv2.imwrite(img_path, np.ones((90, 140, 3), dtype=np.uint8) * 255)

    try:
        page = Page(
            image_path=img_path,
            width=140,
            height=90,
            blocks=[
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox.from_xyxy(10, 20, 110, 60),
                    lines=[Line(text="旧CharOCR行", bbox=BBox.from_xyxy(12, 24, 90, 58), confidence=0.9)],
                )
            ],
            raw_layout_artifact=_paddle_layout_artifact([
                {"block_label": "text", "block_bbox": [10, 20, 110, 60], "block_content": "PPVL文本"}
            ]),
        )
        progress_events = []
        result = OcrPipeline(
            engine=HanwangMicroRecBlockEngine(runner=fake_runner),
            hybrid_prepass_engine=PrepassEngine(),
        ).process_project(
            OcrProject(name="hybrid-no-stale-line-hints", pages=[page]),
            progress_callback=progress_events.append,
        )

        assert len(calls) == 1
        assert result.pages[0].blocks[0].lines[0].text == "重跑结果"
        assert any("PP-OCRv5 page-line prepass complete: 1 lines" in event.message for event in progress_events)
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_does_not_reuse_hanwang_lines_as_ppocr_hints PASSED")


def test_ocr_pipeline_parallelizes_hanwang_page_hybrid_with_prepass():
    import os
    import tempfile
    import threading
    import cv2
    import numpy as np
    from app.engines.hanwang.micro_recblock import (
        BlockResult, CharResult, HanwangMicroRecBlockEngine, LineResult, RunStats,
    )
    from app.models import BBox, Block, BlockType, OcrProject, Page
    from app.models import BlockOrigin
    from app.services.ocr_pipeline import OcrPipeline

    barrier = threading.Barrier(2, timeout=5)
    seen_pages: list[str] = []
    seen_lock = threading.Lock()

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        page_name = ppvl_blocks[0]["block_content"]
        with seen_lock:
            seen_pages.append(page_name)
        barrier.wait()
        kwargs["progress_callback"](1, 2, f"Hanwang micro-recblock SegImg done {page_name}")
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(0, 0, 80, 40),
                source="hanwang",
                text=f"完成{page_name}",
                ppvl_text=page_name,
                lines=[
                    LineResult(
                        text=f"完成{page_name}",
                        bbox=(0, 0, 80, 40),
                        chars=[CharResult(text="完", confidence=0.9, bbox=(0, 0, 20, 40))],
                    )
                ],
                raw_block=dict(ppvl_blocks[0]),
            )
        ], RunStats(n_blocks_total=1, n_blocks_hanwang=1, n_groups=2)

    class FakePrepassEngine:
        prefer_page_ocr = True
        bbox_space = "page"

        def recognize(self, image_bgr, context):
            return []

    paths = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
            cv2.imwrite(path, np.ones((80, 100, 3), dtype=np.uint8) * 255)
            paths.append(path)

        ppvl_records = [
            {"block_label": "text", "block_bbox": [0, 0, 80, 40], "block_content": "p1"},
            {"block_label": "text", "block_bbox": [0, 0, 80, 40], "block_content": "p2"},
        ]
        pages = [
            Page(
                image_path=paths[0],
                width=100,
                height=80,
                page_number=1,
                blocks=[
                    Block(
                        block_type=BlockType.TEXT,
                        bbox=BBox.from_xyxy(0, 0, 80, 40),
                        source_label="text",
                        origin=BlockOrigin(source_label="text", raw_index=0),
                    )
                ],
                raw_layout_artifact=_paddle_layout_artifact([ppvl_records[0]]),
            ),
            Page(
                image_path=paths[1],
                width=100,
                height=80,
                page_number=2,
                blocks=[
                    Block(
                        block_type=BlockType.TEXT,
                        bbox=BBox.from_xyxy(0, 0, 80, 40),
                        source_label="text",
                        origin=BlockOrigin(source_label="text", raw_index=0),
                    )
                ],
                raw_layout_artifact=_paddle_layout_artifact([ppvl_records[1]]),
            ),
        ]
        progress_events = []

        pipeline = OcrPipeline(
            engine=HanwangMicroRecBlockEngine(runner=fake_runner),
            page_concurrency=2,
        )
        pipeline._hybrid_page_ocr_prepass_engine = lambda: FakePrepassEngine()
        result = pipeline.process_project(
            OcrProject(name="parallel-hanwang", pages=pages),
            progress_callback=progress_events.append,
        )

        assert sorted(seen_pages) == ["p1", "p2"]
        assert [page.page_number for page in result.pages] == [1, 2]
        assert result.failed_blocks == []
        assert result.pages[0].blocks[0].lines[0].text == "完成p1"
        assert result.pages[1].blocks[0].lines[0].text == "完成p2"
        assert any("第 1/2 页：CharOCR SegImg done p1" == event.message for event in progress_events)
        assert any(event.completed_pages == 2 for event in progress_events)
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    print("test_ocr_pipeline_parallelizes_hanwang_page_hybrid_with_prepass PASSED")


def test_hanwang_page_blocks_from_layout_preserves_raw_source_label():
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    page = Page(
        image_path="/tmp/raw-label.png",
        width=200,
        height=100,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "paragraph_title",
                "block_bbox": [1, 2, 3, 4],
                "block_content": "raw text",
                "custom_attr": {"level": 2},
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(11, 22, 133, 88),
                note="active text",
                source_label="paragraph_title",
                origin=BlockOrigin(source_label="paragraph_title", raw_index=0),
            )
        ],
    )

    blocks = _page_blocks_from_layout(page)

    assert blocks[0]["block_label"] == "paragraph_title"
    assert blocks[0]["source_label"] == "paragraph_title"
    assert blocks[0]["block_bbox"] == [11, 22, 133, 88]
    assert blocks[0]["block_content"] == "raw text"
    assert blocks[0]["custom_attr"]["level"] == 2

    print("test_hanwang_page_blocks_from_layout_preserves_raw_source_label PASSED")


def test_hanwang_current_layout_blocks_for_ocr_uses_current_blocks_not_ppvl_source():
    from app.engines.hanwang.micro_recblock import _current_layout_blocks_for_ocr
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(
        image_path="/tmp/current-layout-source.png",
        width=200,
        height=120,
        raw_layout_artifact=_paddle_layout_artifact([
            {
                "block_label": "text",
                "block_bbox": [1, 2, 30, 40],
                "block_content": "stale paddle text",
            }
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(20, 30, 160, 90),
                source_label="text",
                lines=[Line(text="current layout text", confidence=0.9, bbox=BBox.from_xyxy(20, 30, 160, 90))],
            )
        ],
    )

    blocks = _current_layout_blocks_for_ocr(page)

    assert len(blocks) == 1
    assert blocks[0] is not _raw_layout_records(page)[0]
    assert blocks[0]["block_label"] == "text"
    assert blocks[0]["block_bbox"] == [20, 30, 160, 90]
    assert blocks[0]["block_content"] == "current layout text"

    print("test_hanwang_current_layout_blocks_for_ocr_uses_current_blocks_not_ppvl_source PASSED")


def test_hanwang_page_blocks_from_layout_does_not_promote_internal_merge_note_to_formula_text():
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockSource, BlockType, Page

    page = Page(
        image_path="/tmp/internal-note-formula.png",
        width=120,
        height=80,
        blocks=[
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(20, 10, 80, 30),
                source=BlockSource.USER_EDITED,
                note="manual_draw_merge_requires_ocr_rerun",
                source_label="inline_formula",
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(20, 40, 80, 60),
                source=BlockSource.USER_EDITED,
                note="active text",
                source_label="text",
            ),
        ],
    )

    blocks = _page_blocks_from_layout(page)

    assert blocks[0]["block_label"] == "inline_formula"
    assert blocks[0]["block_content"] == ""
    assert blocks[1]["block_content"] == "active text"

    print("test_hanwang_page_blocks_from_layout_does_not_promote_internal_merge_note_to_formula_text PASSED")


def test_hanwang_ppvl_skip_uses_current_layout_label_over_raw_payload_label():
    import numpy as np

    from app.engines.hanwang.micro_recblock import BlockResult, HanwangMicroRecBlockEngine, LineResult, RunStats
    from app.models import BBox, Block, BlockType, Page
    from app.models import BlockOrigin

    calls = []

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        calls.append(ppvl_blocks)
        assert ppvl_blocks[0]["block_label"] == "text"
        return [
            BlockResult(
                block_idx=0,
                block_label="text",
                block_bbox=(10, 10, 80, 40),
                source="hanwang",
                text="图",
                ppvl_text="",
                raw_block=dict(ppvl_blocks[0]),
                lines=[LineResult(text="图", bbox=(10, 10, 80, 40), source="hanwang")],
            )
        ], RunStats(n_blocks_total=1, n_blocks_ppvl=1)

    page = Page(
        image_path="/tmp/ppvl-skip-authority.png",
        width=100,
        height=100,
        raw_layout_artifact=_paddle_layout_artifact([
            {"block_label": "figure", "label": "text", "block_content": "图", "block_bbox": [10, 10, 80, 40]},
        ]),
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(10, 10, 80, 40),
                source_label="text",
                origin=BlockOrigin(source_label="figure", raw_index=0),
            )
        ],
    )

    HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((100, 100, 3), dtype=np.uint8),
        page,
    )

    assert len(calls) == 1
    assert page.blocks[0].block_type == BlockType.TEXT
    assert page.blocks[0].source_label == "text"
    assert page.blocks[0].ocr_policy == OcrPolicy.TEXT_OCR

    print("test_hanwang_ppvl_skip_uses_current_layout_label_over_raw_payload_label PASSED")


def test_ocr_pipeline_records_failed_page_when_block_ocr_fails():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class FailingEngine:
        bbox_space = "crop"

        def recognize(self, image_bgr, context):
            raise RuntimeError("hanwang boom")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((160, 240, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        block = Block(block_type=BlockType.TEXT, bbox=BBox(20, 30, 100, 40), order=0)
        page = Page(image_path=img_path, width=240, height=160, blocks=[block])
        result = OcrPipeline(engine=FailingEngine()).process_project(
            OcrProject(name="FailingBlockOCR", pages=[page])
        )

        assert result.failed_blocks == [(0, 0, "hanwang boom")]
        assert result.pages[0].error_message == "OCR 失败：块 0: hanwang boom"
        assert "OCR failed: hanwang boom" in result.pages[0].blocks[0].note
    finally:
        os.unlink(img_path)


def test_workflow_controller_auto_chains_ocr_after_layout():
    from app.controllers.workflow_controller import WorkflowController
    from app.models import BBox, Block, BlockType, OcrProject, Page

    controller = WorkflowController()
    page = Page(image_path="/tmp/auto-chain.png", width=300, height=200)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(20, 30, 100, 40))]
    controller._project = OcrProject(name="AutoChain", pages=[page])
    controller._auto_start_ocr_after_layout = True

    chained = []
    controller.start_ocr = lambda pages, notify_page_callback=None: chained.append((pages, notify_page_callback)) or True

    controller.on_layout_done([page])

    assert len(chained) == 1
    assert chained[0][0] == [page]

    print("test_workflow_controller_auto_chains_ocr_after_layout PASSED")


def test_workflow_controller_hanwang_layout_stays_on_block_ocr_path():
    import app.controllers.workflow_controller as workflow_module
    import app.core.layout_analyzer as layout_module
    from app.models import BBox, Block, BlockType, OcrProject, Page

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeLayoutWorker:
        def __init__(self, pages):
            self.page_done = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            for page in self._pages:
                page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, page.width, page.height), order=0)]
            self.all_done.emit(self._pages)
            self._running = False

    original_get_config = workflow_module.get_config
    original_create_engine = workflow_module.create_engine
    original_layout_worker = layout_module.LayoutWorker

    workflow_module.get_config = lambda: {"mode": "hanwang"}
    workflow_module.create_engine = lambda: object()
    layout_module.LayoutWorker = FakeLayoutWorker

    try:
        controller = workflow_module.WorkflowController()
        page = Page(image_path="/tmp/hanwang-layout.png", width=120, height=90)
        controller._project = OcrProject(name="HanwangFlow", pages=[page])
        started = []
        messages = []
        controller.start_ocr = lambda pages, notify_page_callback=None: started.append(pages) or True
        controller.status_message.connect(messages.append)

        ok = controller.start_layout_analysis([page])

        assert ok is True
        assert controller._proof_ocr_worker is None
        assert started == []
        assert controller._auto_start_ocr_after_layout is False
        assert any("版面分析中" in message for message in messages)
        assert not any("PP-OCRv5" in message for message in messages)
    finally:
        workflow_module.get_config = original_get_config
        workflow_module.create_engine = original_create_engine
        layout_module.LayoutWorker = original_layout_worker


def test_workflow_controller_hanwang_ocr_entry_redirects_to_first_pending_page():
    import app.controllers.workflow_controller as workflow_module
    from app.controllers.workflow_controller import STEP_LAYOUT
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page_done = Page(
            image_path="/tmp/p1.png",
            width=100,
            height=100,
            page_number=1,
            status=PageStatus.OCR_DONE,
        )
        page_done.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20), lines=[
            Line(text="已完成", bbox=BBox(0, 0, 50, 20), confidence=0.9)
        ])]
        page_pending = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page_done, page_pending])
        controller._current_page_number = 1
        started = []
        focused = []
        steps = []
        controller.start_ocr = lambda pages, notify_page_callback=None: started.append(pages) or True
        controller.focus_page.connect(focused.append)
        controller.step_requested.connect(steps.append)

        controller.handle_ocr_entry_requested("main_window", 1)

        assert started == []
        assert focused == [2]
        assert steps == [STEP_LAYOUT]
        assert controller.current_page_number == 2
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_ocr_entry_redirects_to_first_pending_page PASSED")


def test_workflow_controller_hanwang_main_entry_starts_all_actionable_pages():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page_done = Page(
            image_path="/tmp/p1.png",
            width=100,
            height=100,
            page_number=1,
            status=PageStatus.OCR_DONE,
        )
        page_done.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20), lines=[
            Line(text="已完成", bbox=BBox(0, 0, 50, 20), confidence=0.9)
        ])]
        page_ready_2 = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        page_ready_2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20))]
        page_ready_3 = Page(image_path="/tmp/p3.png", width=100, height=100, page_number=3)
        page_ready_3.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 60, 50, 20))]
        page_layout_pending = Page(image_path="/tmp/p4.png", width=100, height=100, page_number=4)
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(
            name="Gate",
            pages=[page_done, page_ready_2, page_ready_3, page_layout_pending],
        )
        controller._current_page_number = 1
        started = []
        controller.start_ocr = (
            lambda pages, notify_page_callback=None, target_page_numbers=None:
            started.append((pages, target_page_numbers)) or True
        )

        controller.handle_ocr_entry_requested("main_window", 1)

        assert started == [([page_ready_2, page_ready_3], {2, 3})]
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_main_entry_starts_all_actionable_pages PASSED")


def test_workflow_controller_hanwang_layout_submit_starts_ocr_when_ready():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, OcrProject, Page

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page = Page(image_path="/tmp/ready.png", width=100, height=100, page_number=1)
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20))]
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page])
        started = []
        controller.start_ocr = (
            lambda pages, notify_page_callback=None, target_page_numbers=None:
            started.append((pages, target_page_numbers)) or True
        )

        controller.handle_ocr_entry_requested("layout_submit", 1)

        assert started == [([page], {1})]
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_layout_submit_starts_ocr_when_ready PASSED")


def test_workflow_controller_hanwang_retries_ocr_error_page_from_main_entry():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, OcrProject, Page, PageStatus

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page = Page(image_path="/tmp/ocr-error.png", width=100, height=100, page_number=1)
        page.status = PageStatus.ERROR
        page.error_message = "OCR 失败：micro-recblock failed"
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20))]
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page])
        started = []
        controller.start_ocr = (
            lambda pages, notify_page_callback=None, target_page_numbers=None:
            started.append((pages, target_page_numbers)) or True
        )

        controller.handle_ocr_entry_requested("main_window", 1)

        assert started == [([page], {1})]
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_retries_ocr_error_page_from_main_entry PASSED")


def test_workflow_controller_hanwang_layout_submit_retries_ocr_error_page():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, OcrProject, Page, PageStatus

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page = Page(image_path="/tmp/ocr-error-submit.png", width=100, height=100, page_number=1)
        page.status = PageStatus.ERROR
        page.error_message = "OCR 失败：micro-recblock failed"
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20))]
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page])
        started = []
        controller.start_ocr = (
            lambda pages, notify_page_callback=None, target_page_numbers=None:
            started.append((pages, target_page_numbers)) or True
        )

        controller.handle_ocr_entry_requested("layout_submit", 1)

        assert started == [([page], {1})]
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_layout_submit_retries_ocr_error_page PASSED")


def test_workflow_controller_hanwang_layout_submit_processes_pending_pages_from_target():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeOcrWorker:
        created_pages = []
        result_pages = []

        def __init__(self, pipeline, pages):
            self.progress_state = DummySignal()
            self.progress_update = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self.finished = DummySignal()
            self._running = False
            FakeOcrWorker.created_pages.append(list(pages))

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            self.all_done.emit(FakeOcrWorker.result_pages)
            self._running = False
            self.finished.emit()

    original_get_config = workflow_module.get_config
    original_create_engine = workflow_module.create_engine
    original_worker = workflow_module.OcrPipelineWorker
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    workflow_module.create_engine = lambda: object()
    workflow_module.OcrPipelineWorker = FakeOcrWorker
    try:
        page1 = Page(image_path="/tmp/p1.png", width=100, height=100, page_number=1)
        page1.status = PageStatus.LAYOUT_DONE
        page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20))]
        page2 = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        page2.status = PageStatus.LAYOUT_DONE
        page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20))]
        processed_page1 = Page(image_path="/tmp/p1.png", width=100, height=100, page_number=1)
        processed_page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20), lines=[
            Line(text="第一页完成", bbox=BBox(0, 0, 50, 20), confidence=0.98)
        ])]
        processed_page2 = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        processed_page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20), lines=[
            Line(text="第二页完成", bbox=BBox(0, 30, 50, 20), confidence=0.99)
        ])]
        FakeOcrWorker.created_pages = []
        FakeOcrWorker.result_pages = [processed_page2, processed_page1]

        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page1, page2])

        controller.handle_ocr_entry_requested("layout_submit", 2)

        assert FakeOcrWorker.created_pages == [[page2, page1]]
        assert len(controller._project.pages) == 2
        assert controller._project.pages[0] is processed_page1
        assert controller._project.pages[0].status == PageStatus.OCR_DONE
        assert controller._project.pages[0].blocks[0].lines[0].text == "第一页完成"
        assert controller._project.pages[1] is processed_page2
        assert controller._project.pages[1].status == PageStatus.OCR_DONE
        assert controller._project.pages[1].blocks[0].lines[0].text == "第二页完成"
    finally:
        workflow_module.get_config = original_get_config
        workflow_module.create_engine = original_create_engine
        workflow_module.OcrPipelineWorker = original_worker

    print("test_workflow_controller_hanwang_layout_submit_processes_pending_pages_from_target PASSED")


def test_workflow_controller_hanwang_block_edit_invalidates_only_that_page():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
    from app.models.page_state import page_needs_ocr_rerun

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page1 = Page(image_path="/tmp/p1.png", width=100, height=100, page_number=1)
        page1.status = PageStatus.OCR_DONE
        page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20), lines=[
            Line(text="第一页", bbox=BBox(0, 0, 50, 20), confidence=0.9)
        ])]
        page2 = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        page2.status = PageStatus.OCR_DONE
        page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20), lines=[
            Line(text="第二页", bbox=BBox(0, 30, 50, 20), confidence=0.9)
        ])]
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page1, page2])

        controller.handle_block_contract_changed(1, "block_moved")

        assert page_ocr_line_count(page1) == 1
        assert page1.blocks[0].lines[0].text == "第一页"
        assert page1.status == PageStatus.LAYOUT_DONE
        assert page_needs_ocr_rerun(page1) is True
        assert page1.ocr_invalidated_reason == "block_moved"
        assert page_ocr_line_count(page2) == 1
        assert page2.status == PageStatus.OCR_DONE
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_block_edit_invalidates_only_that_page PASSED")


def test_pdf_import_cache_name_includes_source_path_hash(tmp_path):
    from app.services import ImportService

    first = tmp_path / "a" / "same.pdf"
    second = tmp_path / "b" / "same.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"%PDF-1.4\n")
    second.write_bytes(b"%PDF-1.4\n")

    first_name = ImportService._pdf_page_cache_name(first, 0)
    second_name = ImportService._pdf_page_cache_name(second, 0)

    assert first_name != second_name
    assert first_name.startswith("same_")
    assert first_name.endswith("_p0001.png")
    assert second_name.startswith("same_")
    assert second_name.endswith("_p0001.png")

    print("test_pdf_import_cache_name_includes_source_path_hash PASSED")


def test_import_cache_name_changes_when_same_path_content_changes(tmp_path):
    from app.services import ImportService

    src = tmp_path / "same.png"
    src.write_bytes(b"first-content")
    first_name = ImportService._pdf_page_cache_name(src, 0)

    src.write_bytes(b"second-content")
    second_name = ImportService._pdf_page_cache_name(src, 0)

    assert first_name != second_name

    print("test_import_cache_name_changes_when_same_path_content_changes PASSED")


def test_workflow_controller_hanwang_no_pending_reports_all_done_without_redirect():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    original_get_config = workflow_module.get_config
    workflow_module.get_config = lambda: {"mode": "hanwang"}
    try:
        page = Page(image_path="/tmp/done.png", width=100, height=100, page_number=1)
        page.status = PageStatus.OCR_DONE
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 20), lines=[
            Line(text="完成", bbox=BBox(0, 0, 50, 20), confidence=0.9)
        ])]
        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page])
        messages = []
        steps = []
        gate_events = []
        controller.status_message.connect(messages.append)
        controller.step_requested.connect(steps.append)
        controller.page_gate_state.connect(lambda *args: gate_events.append(args))

        controller.handle_ocr_entry_requested("main_window", 1)

        assert steps == []
        assert messages[-1] == "全部已完成 OCR"
        assert gate_events[-1][3:] == ("all_pages_done", "全部已完成 OCR")
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_no_pending_reports_all_done_without_redirect PASSED")


def test_workflow_controller_starts_parallel_proof_ocr_with_layout():
    import app.controllers.workflow_controller as workflow_module
    import app.core.layout_analyzer as layout_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeLayoutWorker:
        def __init__(self, pages):
            self.page_done = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            for page in self._pages:
                page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, page.width, page.height), order=0)]
            self.all_done.emit(self._pages)
            self._running = False

    class FakeProofWorker:
        def __init__(self, pipeline, pages, parent=None):
            self.progress_update = DummySignal()
            self.progress_state = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            self._pages[0].blocks[0].lines = [
                Line(text="税", confidence=0.96, bbox=BBox(20, 20, 20, 20))
            ]
            self.all_done.emit(self._pages)
            self._running = False

    class FakePageOcrEngine:
        prefer_page_ocr = True

    original_layout_worker = layout_module.LayoutWorker
    original_ocr_worker = workflow_module.OcrPipelineWorker
    original_create_engine = workflow_module.create_engine
    layout_module.LayoutWorker = FakeLayoutWorker
    workflow_module.OcrPipelineWorker = FakeProofWorker
    workflow_module.create_engine = lambda: FakePageOcrEngine()

    try:
        controller = workflow_module.WorkflowController()
        page = Page(image_path="/tmp/parallel-proof.png", width=100, height=80)
        controller._project = OcrProject(name="ParallelProof", pages=[page])
        finished = []
        controller.ocr_finished.connect(finished.append)

        ok = controller.start_layout_analysis([page])

        assert ok is True
        assert finished
        assert finished[0][0].blocks[0].lines[0].text == "税"
        assert finished[0][0].status == PageStatus.OCR_DONE
    finally:
        layout_module.LayoutWorker = original_layout_worker
        workflow_module.OcrPipelineWorker = original_ocr_worker
        workflow_module.create_engine = original_create_engine

    print("test_workflow_controller_starts_parallel_proof_ocr_with_layout PASSED")


def test_workflow_controller_parallel_proof_skips_missing_page_without_misalignment():
    import app.controllers.workflow_controller as workflow_module
    import app.core.layout_analyzer as layout_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeLayoutWorker:
        def __init__(self, pages):
            self.page_done = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            for page in self._pages:
                page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, page.width, page.height), order=0)]
            self.all_done.emit(self._pages)
            self._running = False

    class FakeProofWorker:
        def __init__(self, pipeline, pages, parent=None):
            self.progress_update = DummySignal()
            self.progress_state = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            self._pages[0].blocks[0].lines = [
                Line(text="第一页", confidence=0.96, bbox=BBox(10, 10, 20, 20))
            ]
            self._pages[2].blocks[0].lines = [
                Line(text="第三页", confidence=0.97, bbox=BBox(30, 30, 20, 20))
            ]
            self.all_done.emit([self._pages[0], self._pages[2]])
            self._running = False

    class FakePageOcrEngine:
        prefer_page_ocr = True

    original_layout_worker = layout_module.LayoutWorker
    original_ocr_worker = workflow_module.OcrPipelineWorker
    original_create_engine = workflow_module.create_engine
    layout_module.LayoutWorker = FakeLayoutWorker
    workflow_module.OcrPipelineWorker = FakeProofWorker
    workflow_module.create_engine = lambda: FakePageOcrEngine()

    try:
        controller = workflow_module.WorkflowController()
        pages = [
            Page(image_path="/tmp/parallel-proof-same.png", width=120, height=80, page_number=1),
            Page(image_path="/tmp/parallel-proof-same.png", width=120, height=80, page_number=1),
            Page(image_path="/tmp/parallel-proof-same.png", width=120, height=80, page_number=1),
        ]
        controller._project = OcrProject(name="ParallelProofSkip", pages=pages)
        finished = []
        controller.ocr_finished.connect(finished.append)

        ok = controller.start_layout_analysis(pages)

        assert ok is True
        assert finished
        out_pages = finished[0]
        assert out_pages[0].blocks[0].lines[0].text == "第一页"
        assert out_pages[1].blocks[0].lines == []
        assert out_pages[2].blocks[0].lines[0].text == "第三页"
    finally:
        layout_module.LayoutWorker = original_layout_worker
        workflow_module.OcrPipelineWorker = original_ocr_worker
        workflow_module.create_engine = original_create_engine

    print("test_workflow_controller_parallel_proof_skips_missing_page_without_misalignment PASSED")


def test_workflow_controller_keeps_qthreads_until_finished_after_error():
    from app.controllers.workflow_controller import WorkflowController

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeRunningWorker:
        def __init__(self):
            self.finished = DummySignal()
            self._running = True

        def isRunning(self):
            return self._running

    controller = WorkflowController()
    layout_worker = FakeRunningWorker()
    proof_worker = FakeRunningWorker()
    controller._layout_worker = layout_worker
    controller._proof_ocr_worker = proof_worker
    controller._connect_worker_cleanup("_layout_worker", layout_worker)
    controller._connect_worker_cleanup("_proof_ocr_worker", proof_worker)

    controller._on_worker_error("boom")

    assert controller._layout_worker is layout_worker
    assert controller._proof_ocr_worker is proof_worker
    assert controller._discard_parallel_proof_result is True

    layout_worker._running = False
    layout_worker.finished.emit()
    proof_worker._running = False
    proof_worker.finished.emit()

    assert controller._layout_worker is None
    assert controller._proof_ocr_worker is None

    print("test_workflow_controller_keeps_qthreads_until_finished_after_error PASSED")


def test_workflow_controller_falls_back_to_block_ocr_when_parallel_proof_failed():
    from app.controllers.workflow_controller import WorkflowController
    from app.models import BBox, Block, BlockType, OcrProject, Page

    controller = WorkflowController()
    page = Page(image_path="/tmp/layout-fallback.png", width=100, height=80)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 90, 60), order=0)]
    controller._project = OcrProject(name="FallbackOCR", pages=[page])
    controller._discard_parallel_proof_result = True
    callback = object()
    controller._queued_ocr_progress_callback = callback
    started = []
    controller.start_ocr = lambda pages, notify_page_callback=None: started.append((pages, notify_page_callback)) or True

    controller.on_layout_done([page])

    assert started == [([page], callback)]
    assert controller._discard_parallel_proof_result is False
    assert controller._pending_layout_pages is None
    assert controller._pending_proof_pages is None

    print("test_workflow_controller_falls_back_to_block_ocr_when_parallel_proof_failed PASSED")


def test_workflow_controller_emits_ocr_progress_and_navigation():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, OcrProject, Page
    from app.services.ocr_run_result import OcrProgress

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self._callbacks):
                callback(*args)

    class FakeWorker:
        def __init__(self, pipeline, pages, parent=None):
            self.progress_update = DummySignal()
            self.progress_state = DummySignal()
            self.all_done = DummySignal()
            self.error = DummySignal()
            self._pages = pages
            self._running = False

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            total = len(self._pages)
            self.progress_state.emit(OcrProgress(
                current_page=1,
                total_pages=total,
                current_block=1,
                total_blocks=1,
                completed_pages=1,
                message="OCR 识别中… 第 1/1 页，块 1/1",
            ))
            self.progress_update.emit(0, total)
            self.all_done.emit(self._pages)
            self._running = False

    original_worker = workflow_module.OcrPipelineWorker
    original_create_engine = workflow_module.create_engine
    workflow_module.OcrPipelineWorker = FakeWorker
    workflow_module.create_engine = lambda: object()

    try:
        controller = workflow_module.WorkflowController()
        page = Page(image_path="/tmp/controller-ocr.png", width=300, height=200)
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(20, 30, 100, 40))]
        controller._project = OcrProject(name="ControllerOCR", pages=[page])

        steps = []
        progress_payloads = []
        finished = []
        page_progress = []

        controller.step_requested.connect(steps.append)
        controller.ocr_progress.connect(progress_payloads.append)
        controller.ocr_finished.connect(finished.append)

        ok = controller.start_ocr([page], notify_page_callback=lambda idx, total: page_progress.append((idx, total)))

        assert ok is True
        assert steps[-1] == workflow_module.STEP_OCR
        assert len(progress_payloads) == 2
        assert progress_payloads[0].completed_pages == 0
        assert "准备中" in progress_payloads[0].message
        assert progress_payloads[1].completed_pages == 1
        assert page_progress == [(0, 1)]
        assert finished and finished[0] == [page]
    finally:
        workflow_module.OcrPipelineWorker = original_worker
        workflow_module.create_engine = original_create_engine

    print("test_workflow_controller_emits_ocr_progress_and_navigation PASSED")


def test_workflow_controller_ocr_done_does_not_force_hproof_step():
    from app.controllers.workflow_controller import STEP_HPROOF, WorkflowController
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    page = Page(image_path="/tmp/no-force-hproof.png", width=120, height=80)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 0, 100, 40),
            lines=[Line(text="完成", confidence=0.95, bbox=BBox(1, 2, 40, 16))],
        )
    ]
    controller = WorkflowController()
    controller._project = OcrProject(name="NoForceHProof", pages=[page])
    steps = []
    messages = []
    controller.step_requested.connect(steps.append)
    controller.status_message.connect(messages.append)

    controller.on_ocr_done([page])

    assert STEP_HPROOF not in steps
    assert controller.can_enter_step(STEP_HPROOF)
    assert any("可进入校对" in message for message in messages)

    print("test_workflow_controller_ocr_done_does_not_force_hproof_step PASSED")


def test_workflow_controller_ocr_done_keeps_error_status_even_with_prepass_lines():
    from app.controllers.workflow_controller import WorkflowController
    from app.core.workflow_state import page_gate_info
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

    page = Page(image_path="/tmp/partial-error.png", width=120, height=80)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 0, 100, 40),
            lines=[Line(text="预识别", confidence=0.95, bbox=BBox(1, 2, 40, 16))],
        )
    ]
    page.error_message = "OCR 失败：micro-recblock failed"
    controller = WorkflowController()
    controller._project = OcrProject(name="PartialError", pages=[page])

    controller.on_ocr_done([page])

    assert page_ocr_line_count(page) == 1
    assert page.status == PageStatus.ERROR
    gate = page_gate_info(page)
    assert gate.page_state == "ocr_error"
    assert gate.action_enabled is True

    print("test_workflow_controller_ocr_done_keeps_error_status_even_with_prepass_lines PASSED")


def test_main_window_ocr_finished_preserves_current_step():
    from app.controllers.workflow_controller import STEP_HPROOF, STEP_LAYOUT, STEP_OCR
    from app.services.ocr_run_result import OcrProgress
    from app.models import Page
    from app.ui.main_window import MainWindow

    _get_qapp()
    window = MainWindow()
    try:
        window._controller.set_current_step(STEP_OCR)
        synced = []
        window._controller.sync_proof_panels = lambda *args, **kwargs: synced.append(True)  # type: ignore[method-assign]
        assert window._stack.currentIndex() == STEP_LAYOUT
        assert window._stack.currentWidget() is window._layout_panel

        window._on_ocr_progress(OcrProgress(
            current_page=1,
            total_pages=1,
            current_block=2,
            total_blocks=4,
            completed_pages=0,
            message="Hanwang micro-recblock 已完成 group 2/4",
        ))
        assert not window._ocr_placeholder.isHidden()
        assert synced == []

        window._on_ocr_finished([Page(image_path="/tmp/ocr-finished.png", width=10, height=10)])

        assert synced == [True]
        assert window._controller.current_step == STEP_OCR
        assert window._stack.currentIndex() == STEP_LAYOUT
        assert window._stack.currentWidget() is window._layout_panel
        assert window._ocr_placeholder.isHidden()
        assert window._controller.current_step != STEP_HPROOF
    finally:
        window.close()

    print("test_main_window_ocr_finished_preserves_current_step PASSED")


def test_main_window_bottom_progress_shows_ocr_stage_and_page_count():
    from app.services.ocr_run_result import OcrProgress
    from app.ui.main_window import MainWindow

    _get_qapp()
    window = MainWindow()
    try:
        window._on_ocr_progress(OcrProgress(
            current_page=1,
            total_pages=3,
            current_block=2,
            total_blocks=10,
            completed_pages=0,
            message="Hanwang OCR 识别中 2/10",
        ))

        assert not window._ocr_placeholder.isHidden()
        assert window._ocr_placeholder._title.text() == "OCR"
        assert "第 1/3 页" in window._ocr_placeholder._detail.text()
        assert "字符识别" in window._ocr_placeholder._detail.text()
        assert window._ocr_placeholder._count.text() == "0/3 页"
        assert window._ocr_placeholder._bar.value() == 54
        assert window.statusBar().currentMessage() == ""

        window._on_ocr_progress(OcrProgress(
            current_page=1,
            total_pages=3,
            current_block=1,
            total_blocks=1,
            completed_pages=1,
            message="OCR 识别中… 第 1/3 页，CharOCR 已写回版面块",
        ))

        assert window._ocr_placeholder._bar.value() == 100
        assert window._ocr_placeholder._count.text() == "1/3 页"
    finally:
        window.close()


def test_main_window_bottom_progress_handles_layout_without_sidebar_progress():
    from app.ui.main_window import MainWindow

    _get_qapp()
    window = MainWindow()
    try:
        window._on_layout_progress(0, 2)

        assert not window._ocr_placeholder.isHidden()
        assert window._ocr_placeholder._title.text() == "版面分析"
        assert window._ocr_placeholder._detail.text() == "等待结果"
        assert window._ocr_placeholder._count.text() == "1/2 页"
        assert window._ocr_placeholder._bar.value() == 50
        assert window._layout_panel._progress_bar.isHidden()
        window._set_status_message("版面分析中…")
        assert window.statusBar().currentMessage() == ""
    finally:
        window.close()


def test_main_window_find_action_opens_layout_find_dialog():
    from app.controllers.workflow_controller import STEP_LAYOUT
    from app.models import OcrProject, Page
    from app.ui.main_window import MainWindow

    _get_qapp()
    window = MainWindow()
    try:
        page = Page(image_path="/tmp/find-page.png", width=100, height=100, page_number=1)
        window._controller._project = OcrProject(name="Find", pages=[page])
        window._layout_panel.set_pages([page])
        assert window._layout_panel._find_dialog.isHidden()

        window._show_layout_find()

        assert window._controller.current_step == STEP_LAYOUT
        assert window._stack.currentWidget() is window._layout_panel
        assert not window._layout_panel._find_dialog.isHidden()
    finally:
        window._controller._project = None
        window.close()

    print("test_main_window_find_action_opens_layout_find_dialog PASSED")


def test_workflow_controller_normalizes_loaded_project_geometry():
    import tempfile
    import cv2
    import numpy as np

    from app.controllers.workflow_controller import WorkflowController
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file, \
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        db_path = db_file.name
        img_path = img_file.name

    try:
        img = np.full((120, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (16, 18), (204, 30), (0, 0, 0), -1)
        cv2.rectangle(img, (18, 68), (196, 82), (0, 0, 0), -1)
        cv2.imwrite(img_path, img)

        page = Page(
            image_path=img_path,
            width=220,
            height=120,
            blocks=[Block(
                block_type=BlockType.TEXT,
                bbox=BBox(0, 0, 220, 120),
                lines=[Line(text="甲乙", confidence=0.95, bbox=BBox(10, 44, 160, 40))],
            )],
        )
        project = OcrProject(name="LoadedProof", pages=[page], db_path=db_path)

        with ProjectStore(db_path) as store:
            store.save_project(project)

        controller = WorkflowController()
        try:
            assert controller.open_project(db_path) is True
            loaded_line = controller.project.pages[0].blocks[0].lines[0]
            assert loaded_line.bbox == BBox(10, 44, 160, 40)
            assert len(loaded_line.chars) == 2
        finally:
            controller.close()
    finally:
        os.unlink(db_path)
        os.unlink(img_path)


def test_workflow_controller_auto_save_persists_quality_probe_sidecar():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core import quality_probe as qp
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    sidecar_path = qp.sidecar_path_for_project(db_path)
    store = None
    try:
        line = Line(text="已", confidence=0.9, bbox=BBox(1, 2, 30, 12))
        page = Page(
            image_path="/tmp/auto-save-qprobe.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="qprobe-auto-save", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)

        probe_store = qp.ProbeStore()
        probe = qp.Probe(
            key=qp.ProbeKey(page.page_number, 0, 0, 0),
            true_char="已",
            fake_char="己",
            observation="pending",
        )
        probe_store.add(probe)
        qp.set_active_store(probe_store)
        assert sidecar_path is not None
        assert qp.save_store_to_path(probe_store, sidecar_path)

        probe.observation = "corrected"
        controller = WorkflowController()
        controller._project = project
        controller._store = store
        controller.auto_save()

        loaded = qp.load_store_from_path(sidecar_path)
        assert loaded is not None
        assert next(iter(loaded.all())).observation == "corrected"
    finally:
        qp.reset_active_store()
        if store is not None:
            store.close()
        if sidecar_path:
            try:
                os.unlink(sidecar_path)
            except FileNotFoundError:
                pass
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_workflow_controller_auto_save_persists_quality_probe_sidecar PASSED")


def test_workflow_controller_auto_save_persists_flag_status_change():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core.proof_change import ProofChangeSet
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(text="疑点", confidence=0.9, bbox=BBox(1, 2, 30, 12))
        page = Page(
            image_path="/tmp/auto-save-flag.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="flag-auto-save", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)

        saved_line = project.pages[0].blocks[0].lines[0]
        set_line_proof_status(saved_line, ProofStatus.AUTO_FLAGGED)
        controller = WorkflowController()
        controller._project = project
        controller._store = store

        controller.auto_save(
            ProofChangeSet(status_changed=True).scoped_to_line(
                project.pages[0],
                project.pages[0].blocks[0],
                saved_line,
                write_chars=False,
            )
        )

        loaded = store.load_project(project.id)
        assert proof_status(loaded.pages[0].blocks[0].lines[0]) == ProofStatus.AUTO_FLAGGED
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_workflow_controller_auto_save_persists_flag_status_change PASSED")


def test_workflow_controller_auto_save_persists_line_chars_for_text_change():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core import quality_probe as qp
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService
    from app.services.proof_probe_text_service import save_displayed_edit_result

    qp.reset_active_store()
    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(
            text="甲",
            confidence=0.9,
            bbox=BBox(1, 2, 18, 20),
            chars=[
                Char(
                    char="甲",
                    confidence=0.9,
                    bbox=BBox(1, 2, 18, 20),
                    bbox_source="ocr",
                    bbox_granularity="char",
                    token_text="甲",
                )
            ],
        )
        page = Page(
            image_path="/tmp/auto-save-line-char.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="line-char-auto-save", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        page = project.pages[0]
        block = page.blocks[0]
        line = block.lines[0]

        change = save_displayed_edit_result(line, page, block, "乙").scoped_to_line(
            page,
            block,
            line,
            write_chars=True,
        )
        controller = WorkflowController()
        controller._project = project
        controller._store = store

        controller.auto_save(change)

        loaded = store.load_project(project.id)
        loaded_line = loaded.pages[0].blocks[0].lines[0]
        assert proof_display_text(loaded_line) == "乙"
        assert loaded_line.chars[0].char == "乙"
        assert loaded_line.chars[0].token_text == "乙"
        index = CharIndexService(include_non_cjk=True, include_fallback=True).build(loaded.pages)
        assert index.query("乙")
        assert index.query("甲") == []
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_workflow_controller_auto_save_persists_line_chars_for_text_change PASSED")


def test_workflow_controller_auto_save_persists_inline_formula_carrier_text_change():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core import quality_probe as qp
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService
    from app.services.proof_probe_text_service import save_displayed_edit_result

    qp.reset_active_store()
    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(
            text="甲$ A $乙",
            confidence=0.9,
            bbox=BBox(1, 2, 90, 20),
            chars=[
                Char("甲", 0.9, BBox(1, 2, 18, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
                Char(
                    "$ A $",
                    1.0,
                    BBox(24, 2, 42, 20),
                    bbox_source="paddle_inline_formula",
                    bbox_granularity="word",
                    token_text="$ A $",
                ),
                Char("乙", 0.9, BBox(70, 2, 18, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
            ],
        )
        page = Page(
            image_path="/tmp/auto-save-inline-formula-carrier.png",
            width=120,
            height=80,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 24), lines=[line])],
        )
        project = OcrProject(name="inline-formula-auto-save", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        page = project.pages[0]
        block = page.blocks[0]
        line = block.lines[0]

        change = save_displayed_edit_result(line, page, block, "丙$ B $乙").scoped_to_line(
            page,
            block,
            line,
            write_chars=True,
        )
        controller = WorkflowController()
        controller._project = project
        controller._store = store

        controller.auto_save(change)

        loaded = store.load_project(project.id)
        loaded_line = loaded.pages[0].blocks[0].lines[0]
        assert proof_display_text(loaded_line) == "丙$ B $乙"
        assert [(char.char, char.token_text) for char in loaded_line.chars] == [
            ("丙", "丙"),
            ("$ B $", "$ B $"),
            ("乙", "乙"),
        ]
        index = CharIndexService(include_non_cjk=True).build(loaded.pages)
        assert index.query("$ B $")
        assert index.query("$ A $") == []
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_workflow_controller_auto_save_persists_inline_formula_carrier_text_change PASSED")


def test_project_store_update_proof_lines_does_not_move_foreign_char_uid():
    import os
    import tempfile

    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line1 = Line(
            text="甲",
            confidence=0.9,
            bbox=BBox(1, 2, 20, 12),
            chars=[
                Char("甲", 0.9, BBox(1, 2, 10, 12), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            ],
        )
        line2 = Line(
            text="乙",
            confidence=0.9,
            bbox=BBox(1, 20, 20, 12),
            chars=[
                Char("乙", 0.9, BBox(1, 20, 10, 12), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
            ],
        )
        page = Page(
            image_path="/tmp/proof-char-uid-collision.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 40), lines=[line1, line2])],
        )
        project = OcrProject(name="proof-char-uid-collision", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        saved_line1, saved_line2 = project.pages[0].blocks[0].lines
        line1_original_uid = saved_line1.chars[0].uid
        line2_uid = saved_line2.chars[0].uid

        set_line_proof_text(saved_line1, "丙")
        saved_line1.chars[0].char = "丙"
        saved_line1.chars[0].token_text = "丙"
        saved_line1.chars[0].uid = line2_uid

        store.update_proof_lines([(saved_line1, True)])

        loaded = store.load_project(project.id)
        loaded_line1, loaded_line2 = loaded.pages[0].blocks[0].lines
        assert proof_display_text(loaded_line1) == "丙"
        assert [char.char for char in loaded_line1.chars] == ["丙"]
        assert loaded_line1.chars[0].uid == line1_original_uid
        assert proof_display_text(loaded_line2) == "乙"
        assert [char.char for char in loaded_line2.chars] == ["乙"]
        assert loaded_line2.chars[0].uid == line2_uid
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_project_store_update_proof_lines_does_not_move_foreign_char_uid PASSED")


def test_project_store_creates_and_loads_proof_line_state():
    import os
    import tempfile

    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(text="OCR文本", confidence=0.9, bbox=BBox(1, 2, 40, 12))
        set_line_proof_text(line, "人工文本")
        set_line_proof_status(line, ProofStatus.OK)
        page = Page(
            image_path="/tmp/proof-line-state.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="proof-line-state", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        saved_line = project.pages[0].blocks[0].lines[0]

        row = store.conn.execute(
            "SELECT final_text, final_text_set, proof_status FROM proof_line_state WHERE line_uid=?",
            (saved_line.uid,),
        ).fetchone()
        assert row is not None
        assert row["final_text"] == "人工文本"
        assert row["final_text_set"] == 1
        assert row["proof_status"] == ProofStatus.OK.value

        line_cols = {row["name"] for row in store.conn.execute("PRAGMA table_info(line)").fetchall()}
        assert {"final_text", "final_text_set", "proof_status"}.isdisjoint(line_cols)

        loaded = store.load_project(project.id)
        loaded_line = loaded.pages[0].blocks[0].lines[0]
        assert proof_display_text(loaded_line) == "人工文本"
        assert proof_status(loaded_line) == ProofStatus.OK
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_project_store_creates_and_loads_proof_line_state PASSED")


def test_project_store_update_proof_lines_writes_proof_line_state():
    import os
    import tempfile

    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(text="OCR文本", confidence=0.9, bbox=BBox(1, 2, 40, 12))
        page = Page(
            image_path="/tmp/proof-line-state-update.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="proof-line-state-update", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        saved_line = project.pages[0].blocks[0].lines[0]

        set_line_proof_text(saved_line, "")
        set_line_proof_status(saved_line, ProofStatus.MODIFIED)
        store.update_proof_lines([(saved_line, False)])

        row = store.conn.execute(
            "SELECT final_text, final_text_set, proof_status FROM proof_line_state WHERE line_uid=?",
            (saved_line.uid,),
        ).fetchone()
        assert row is not None
        assert row["final_text"] == ""
        assert row["final_text_set"] == 1
        assert row["proof_status"] == ProofStatus.MODIFIED.value

        line_cols = {row["name"] for row in store.conn.execute("PRAGMA table_info(line)").fetchall()}
        assert {"final_text", "final_text_set", "proof_status"}.isdisjoint(line_cols)

        loaded = store.load_project(project.id)
        loaded_line = loaded.pages[0].blocks[0].lines[0]
        assert proof_display_text(loaded_line) == ""
        assert proof_final_text_set(loaded_line) is True
        assert proof_status(loaded_line) == ProofStatus.MODIFIED
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_project_store_update_proof_lines_writes_proof_line_state PASSED")


def test_project_store_rejects_invalid_proof_line_state_status_on_load():
    import os
    import tempfile

    import pytest

    from app.core.project_store import ProjectDataError, ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line = Line(text="OCR文本", confidence=0.9, bbox=BBox(1, 2, 40, 12))
        page = Page(
            image_path="/tmp/invalid-proof-status.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        project = OcrProject(name="invalid-proof-status", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        saved_line = project.pages[0].blocks[0].lines[0]
        store.conn.execute(
            "UPDATE proof_line_state SET proof_status=? WHERE line_uid=?",
            ("not-a-status", saved_line.uid),
        )
        store.conn.commit()

        with pytest.raises(ProjectDataError, match="proof_line_state.proof_status invalid value"):
            store.load_project(project.id)
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_project_store_rejects_invalid_proof_line_state_status_on_load PASSED")


def test_project_store_save_project_prunes_stale_proof_line_state():
    import os
    import tempfile

    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line1 = Line(text="第一行", confidence=0.9, bbox=BBox(1, 2, 40, 12))
        line2 = Line(text="第二行", confidence=0.9, bbox=BBox(1, 20, 40, 12))
        page = Page(
            image_path="/tmp/proof-line-state-prune.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 40), lines=[line1, line2])],
        )
        project = OcrProject(name="proof-line-state-prune", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        block = project.pages[0].blocks[0]
        stale_uid = block.lines[1].uid

        assert store.conn.execute("SELECT COUNT(*) FROM proof_line_state").fetchone()[0] == 2
        block.lines = [block.lines[0]]
        store.save_project(project)

        rows = store.conn.execute("SELECT line_uid FROM proof_line_state").fetchall()
        assert [row["line_uid"] for row in rows] == [block.lines[0].uid]
        assert not store.conn.execute(
            "SELECT 1 FROM proof_line_state WHERE line_uid=?",
            (stale_uid,),
        ).fetchone()
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_project_store_save_project_prunes_stale_proof_line_state PASSED")


def test_proof_persistence_scoped_status_does_not_overwrite_other_db_lines():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core.proof_change import ProofChangeSet
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line1 = Line(text="第一行", confidence=0.9, bbox=BBox(1, 2, 30, 12))
        line2 = Line(text="第二行", confidence=0.9, bbox=BBox(1, 20, 30, 12))
        page = Page(
            image_path="/tmp/scoped-proof-save.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 40), lines=[line1, line2])],
        )
        project = OcrProject(name="scoped-proof-save", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        page = project.pages[0]
        block = page.blocks[0]
        saved_line1, stale_line2 = block.lines

        store.conn.execute(
            "UPDATE proof_line_state SET final_text=?, final_text_set=1, proof_status=? WHERE line_uid=?",
            ("数据库新值", ProofStatus.MODIFIED.value, stale_line2.uid),
        )
        store.conn.commit()

        set_line_proof_status(saved_line1, ProofStatus.AUTO_FLAGGED)
        controller = WorkflowController()
        controller._project = project
        controller._store = store
        controller.auto_save(
            ProofChangeSet(status_changed=True).scoped_to_line(
                page,
                block,
                saved_line1,
                write_chars=False,
            )
        )

        loaded = store.load_project(project.id)
        loaded_lines = loaded.pages[0].blocks[0].lines
        assert proof_status(loaded_lines[0]) == ProofStatus.AUTO_FLAGGED
        assert proof_display_text(loaded_lines[1]) == "数据库新值"
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_proof_persistence_scoped_status_does_not_overwrite_other_db_lines PASSED")


def test_proof_persistence_scoped_lines_commit_atomically():
    import os
    import tempfile

    import pytest

    from app.core.proof_change import ProofChangeSet
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.proof_persistence_service import ProofPersistenceService
    from app.services.proof_probe_text_service import save_displayed_edit_result

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    store = None
    try:
        line1 = Line(
            text="AAAA",
            confidence=0.9,
            bbox=BBox(1, 2, 40, 12),
            chars=[
                Char(char=ch, confidence=0.9, bbox=BBox(i * 10, 2, 8, 12), bbox_source="ocr", bbox_granularity="char", token_text=ch)
                for i, ch in enumerate("AAAA")
            ],
        )
        line2 = Line(
            text="BBBB",
            confidence=0.9,
            bbox=BBox(1, 20, 40, 12),
            chars=[
                Char(char=ch, confidence=0.9, bbox=BBox(i * 10, 20, 8, 12), bbox_source="ocr", bbox_granularity="char", token_text=ch)
                for i, ch in enumerate("BBBB")
            ],
        )
        page = Page(
            image_path="/tmp/scoped-proof-atomic.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 40), lines=[line1, line2])],
        )
        project = OcrProject(name="scoped-proof-atomic", pages=[page], db_path=db_path)
        store = ProjectStore(db_path)
        store.open()
        project = store.save_project(project)
        page = project.pages[0]
        block = page.blocks[0]
        line1, line2 = block.lines

        change1 = save_displayed_edit_result(line1, page, block, "CCCC").scoped_to_line(
            page,
            block,
            line1,
            write_chars=True,
        )
        change2 = save_displayed_edit_result(line2, page, block, "DDDD").scoped_to_line(
            page,
            block,
            line2,
            write_chars=True,
        )
        change = change1.merge(change2)
        line2.uid = ""

        with pytest.raises(RuntimeError):
            ProofPersistenceService(store, project).persist(change)

        loaded = store.load_project(project.id)
        loaded_lines = loaded.pages[0].blocks[0].lines
        assert [proof_display_text(line) for line in loaded_lines] == ["AAAA", "BBBB"]
        assert [[char.char for char in line.chars] for line in loaded_lines] == [
            list("AAAA"),
            list("BBBB"),
        ]
    finally:
        if store is not None:
            store.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_proof_persistence_scoped_lines_commit_atomically PASSED")


def test_workflow_controller_save_project_as_persists_quality_probe_sidecar():
    import os
    import tempfile

    from app.controllers.workflow_controller import WorkflowController
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as db_file:
        db_path = db_file.name
    os.unlink(db_path)
    sidecar_path = qp.sidecar_path_for_project(db_path)
    controller = WorkflowController()
    try:
        line = Line(text="已", confidence=0.9, bbox=BBox(1, 2, 30, 12))
        page = Page(
            image_path="/tmp/save-as-qprobe.png",
            width=100,
            height=100,
            page_number=1,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])],
        )
        controller._project = OcrProject(name="qprobe-save-as", pages=[page])

        probe_store = qp.ProbeStore()
        probe_store.add(qp.Probe(
            key=qp.ProbeKey(page.page_number, 0, 0, 0),
            true_char="已",
            fake_char="己",
            observation="corrected",
        ))
        qp.set_active_store(probe_store)

        assert controller.save_project_as(db_path) is True

        assert sidecar_path is not None
        loaded = qp.load_store_from_path(sidecar_path)
        assert loaded is not None
        assert next(iter(loaded.all())).observation == "corrected"
    finally:
        qp.reset_active_store()
        if controller.store is not None:
            controller.store.close()
        if sidecar_path:
            try:
                os.unlink(sidecar_path)
            except FileNotFoundError:
                pass
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    print("test_workflow_controller_save_project_as_persists_quality_probe_sidecar PASSED")


# =====================================================================
# ExportService 测试
# =====================================================================

def test_export_service():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus
    from app.services.export_service import (
        check_export_readiness, get_export_text,
    )

    bb = BBox(0, 0, 100, 20)

    # 测试 get_export_text
    line = Line(text="最终文本", confidence=0.9, bbox=bb)
    assert get_export_text(line) == "最终文本"
    set_line_proof_text(line, "人工最终真值")
    assert get_export_text(line) == "人工最终真值"

    # 测试空项目
    empty_project = OcrProject(name="empty", pages=[])
    warnings = check_export_readiness(empty_project)
    assert len(warnings) > 0

    # 测试有未校对行
    page = Page(image_path="/tmp/x.jpg", width=800, height=600)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[
        Line(text="未校对", confidence=0.6, bbox=bb),
    ])
    page.blocks = [block]
    project = OcrProject(name="test", pages=[page])
    warnings = check_export_readiness(project)
    has_unproofed = any("未校对" in w for w in warnings)
    assert has_unproofed

    print("test_export_service PASSED")


def test_proof_display_edit_writes_final_text():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.proof_probe_text_service import displayed_text, save_displayed_edit_result

    bb = BBox(0, 0, 100, 20)
    line = Line(text="OCR原文", confidence=0.9, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/x.jpg", width=800, height=600, blocks=[block])

    assert displayed_text(line, page, block) == "OCR原文"
    change = save_displayed_edit_result(line, page, block, "人工终稿")
    assert change.text_changed is True
    assert change.changed is True
    assert proof_final_text(line) == "人工终稿"
    assert line.text == "OCR原文"
    assert proof_display_text(line) == "人工终稿"

    print("test_proof_display_edit_writes_final_text PASSED")


# =====================================================================
# ImportService 测试
# =====================================================================

def test_import_service():
    import tempfile
    from PIL import Image
    from app.services import ImportService

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图片
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (200, 100), color="white")
        img.save(img_path)

        # 创建测试缓存目录
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)

        service = ImportService(cache_dir=cache_dir)
        result = service.import_paths([img_path])

        assert result.success_count == 1
        assert result.failed_count == 0
        assert result.pages[0].width == 200
        assert result.pages[0].height == 100
        assert result.pages[0].source_type == "image"
        assert result.pages[0].cache_image_path.endswith(".png")
        assert os.path.exists(result.pages[0].cache_image_path)

    print("test_import_service PASSED")


def test_import_service_sequential_page_numbers():
    import tempfile
    from PIL import Image
    from app.services import ImportService

    with tempfile.TemporaryDirectory() as tmpdir:
        first = os.path.join(tmpdir, "a.png")
        second = os.path.join(tmpdir, "b.png")
        Image.new("RGB", (120, 80), color="white").save(first)
        Image.new("RGB", (140, 90), color="white").save(second)

        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)

        service = ImportService(cache_dir=cache_dir)
        result = service.import_paths([first, second])

        assert result.success_count == 2
        assert [page.page_number for page in result.pages] == [1, 2]
        assert result.pages[0].source_path == first
        assert result.pages[1].source_path == second

    print("test_import_service_sequential_page_numbers PASSED")


def test_proof_state_bus():
    from app.core.proof_state import ProofUpdateRequest
    from app.core.proof_state_bus import TOPIC_LINE_PROOF_CHANGED, get_proof_state_bus

    bus = get_proof_state_bus()
    bus.clear()
    events = []

    unsubscribe = bus.subscribe(TOPIC_LINE_PROOF_CHANGED, events.append)
    request = ProofUpdateRequest(page_id=None, line_id=7, status="ok")
    bus.publish_line_update(request)

    assert events == [request]
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == 1

    unsubscribe()
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == 0

    bus.clear()
    print("test_proof_state_bus PASSED")


def test_proof_state_bus_typed_contracts():
    from app.core.proof_state import (
        TOPIC_LINE_PROOF_CHANGED,
        TOPIC_PROBE_OBSERVED,
        CandidateSet,
        ProbeObservation,
        ProofSelection,
        ProofUpdateRequest,
        proof_request_matches_line,
        proof_request_matches_page,
    )
    from app.core.proof_state_bus import get_proof_state_bus
    from app.models import BBox, Line, Page

    bus = get_proof_state_bus()
    bus.clear()
    line_events = []
    probe_events = []

    bus.subscribe(TOPIC_LINE_PROOF_CHANGED, line_events.append)
    bus.subscribe(TOPIC_PROBE_OBSERVED, probe_events.append)

    selection = ProofSelection(
        page_number=3,
        page_id=5,
        page_uid="page_uid_5",
        line_id=11,
        line_uid="line_uid_11",
        line_index=2,
        char_index=1,
        source="test",
    )
    request = ProofUpdateRequest(
        page_id=5,
        line_id=11,
        status="modified",
        page_uid="page_uid_5",
        line_uid="line_uid_11",
        origin=123,
        selection=selection,
        source="unit",
    )
    bus.publish_line_update(request)
    observation = ProbeObservation(
        page_number=3,
        block_index=1,
        line_index=2,
        char_index=4,
        true_char="真",
        fake_char="假",
    )
    bus.publish_probe_observed(observation)

    assert line_events == [request]
    page = Page(image_path="/tmp/page.png", width=100, height=100)
    page.id = 5
    page.uid = "page_uid_5"
    line = Line(text="甲", confidence=0.9, bbox=BBox(0, 0, 10, 10))
    line.id = 11
    line.uid = "line_uid_11"
    assert proof_request_matches_page(request, page)
    assert proof_request_matches_line(request, line)
    assert proof_request_matches_page(
        ProofUpdateRequest(page_id=-1, page_uid="page_uid_5", line_id=11, status="modified"),
        page,
    )
    assert not proof_request_matches_page(
        ProofUpdateRequest(page_id=5, page_uid="other_page", line_id=11, status="modified"),
        page,
    )
    assert proof_request_matches_line(
        ProofUpdateRequest(page_id=5, line_id=-1, line_uid="line_uid_11", status="modified"),
        line,
    )
    assert proof_request_matches_line(
        ProofUpdateRequest(page_id=5, line_id=11, line_uid="", status="modified"),
        line,
    )
    assert not proof_request_matches_line(
        ProofUpdateRequest(page_id=5, line_id=11, line_uid="other_line", status="modified"),
        line,
    )
    assert probe_events == [observation]

    candidates = CandidateSet.from_values(selection=selection, values=["甲", "乙"], source="unit")
    assert candidates.texts == ["甲", "乙"]
    assert candidates.options[0].rank == 0

    bus.clear()
    print("test_proof_state_bus_typed_contracts PASSED")


def test_workflow_controller_emits_typed_view_state():
    from app.controllers.workflow_controller import WorkflowController, STEP_LAYOUT
    from app.core.workflow_state import WorkflowViewState

    controller = WorkflowController()
    states = []
    controller.view_state_changed.connect(states.append)

    controller.set_layout_run_enabled(True)
    controller.set_current_page_number(7)
    controller.set_current_step(STEP_LAYOUT)

    assert isinstance(states[-1], WorkflowViewState)
    assert states[-1].layout_run_enabled is True
    assert states[-1].current_page_number == 7
    assert states[-1].current_step == STEP_LAYOUT

    print("test_workflow_controller_emits_typed_view_state PASSED")


def test_char_index_service():
    from app.core.proof_state import ProofSelection
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharEntry, CharIndexEntry, CharIndexService

    page = Page(image_path="/tmp/page.png", width=400, height=300)
    page.id = 42
    page.page_number = 7
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            order=3,
            bbox=BBox(0, 0, 100, 20),
            lines=[
                Line(
                    text="甲乙",
                    confidence=0.9,
                    bbox=BBox(10, 20, 100, 18),
                    chars=[
                        Char(char="甲", confidence=0.98, bbox=BBox(10, 20, 20, 18)),
                        Char(char="乙", confidence=0.97, bbox=BBox(30, 20, 18, 18)),
                    ],
                ),
                Line(text="乙丙", confidence=0.75, bbox=BBox(10, 50, 100, 18)),
            ],
        )
    ]
    project = OcrProject(name="char-index", pages=[page])

    service = CharIndexService(include_fallback=True).build_index(project)
    yi_entries = service.query("乙")

    assert len(yi_entries) == 2
    assert CharIndexEntry is CharEntry
    assert isinstance(yi_entries[0], CharEntry)
    assert yi_entries[0].page_idx == 0
    assert yi_entries[0].page_id == 42
    assert yi_entries[0].page_uid == page.uid
    assert yi_entries[0].page_path == page.display_image_path
    assert yi_entries[0].page_number == 7
    assert yi_entries[0].line is page.blocks[0].lines[0]
    assert yi_entries[0].block_order == 3
    assert yi_entries[0].bbox == BBox(30, 20, 18, 18)
    assert yi_entries[1].char_idx == 0
    assert yi_entries[1].bbox == BBox(10, 50, 50, 18)
    assert service.first_entry("乙") == yi_entries[0]
    assert service.unique_chars() == 3
    assert service.total_chars() == 4
    assert service.char_frequency() == [("乙", 2), ("丙", 1), ("甲", 1)]
    selection = ProofSelection.for_char_entry(yi_entries[0], source="unit")
    assert selection.page_id == 42
    assert selection.page_uid == page.uid
    assert selection.line_uid == yi_entries[0].line.uid

    print("test_char_index_service PASSED")


def test_char_index_service_synthesizes_vertical_char_boxes():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(text="天地玄黄", confidence=0.92, bbox=BBox(40, 10, 24, 160))
    page = Page(
        image_path="/tmp/page.png",
        width=200,
        height=240,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(30, 0, 60, 180), lines=[line])],
    )

    service = CharIndexService(include_fallback=True).build_index(OcrProject(name="vertical-index", pages=[page]))
    entries = service.query("玄")

    assert len(entries) == 1
    assert entries[0].bbox == BBox(40, 90, 24, 40)

    print("test_char_index_service_synthesizes_vertical_char_boxes PASSED")


def test_char_index_service_legacy_build_contract():
    import tempfile

    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.char_index_service import CharIndexService

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        image = np.full((160, 160, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (14, 36), (27, 74), (0, 0, 0), -1)
        cv2.rectangle(image, (40, 36), (70, 74), (0, 0, 0), -1)
        cv2.imwrite(page_path, image)

        line = Line(text="甲乙", confidence=0.9, bbox=BBox(12, 30, 80, 50))
        page = Page(
            image_path=page_path,
            width=160,
            height=160,
            page_number=3,
            blocks=[Block(block_type=BlockType.TEXT, order=1, bbox=BBox(10, 20, 90, 70), lines=[line])],
        )

        service = CharIndexService(include_fallback=True).build([page])
        entry = service.first_entry("甲")

        assert entry is not None
        assert entry.page_path == page.display_image_path
        assert entry.page_number == 3
        assert entry.line is line
        assert abs(entry.bbox.x - 14) <= 3
        assert abs(entry.bbox.w - 13) <= 4
    finally:
        os.unlink(page_path)

    print("test_char_index_service_legacy_build_contract PASSED")


def test_char_index_hides_fallback_units_by_default():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    token_bbox = BBox(30, 20, 42, 24)
    line = Line(
        text="估真实",
        confidence=0.93,
        bbox=BBox(10, 20, 80, 24),
        chars=[
            Char(char="估", confidence=0.8, bbox=BBox(10, 20, 18, 24), bbox_source="fallback", bbox_granularity="fallback", token_text="估"),
            Char(char="真", confidence=0.93, bbox=token_bbox, bbox_source="ocr", bbox_granularity="word", token_text="真实"),
            Char(char="实", confidence=0.93, bbox=token_bbox, bbox_source="ocr", bbox_granularity="word", token_text="真实"),
        ],
    )
    inferred_line = Line(text="推断", confidence=0.7, bbox=BBox(10, 60, 80, 24))
    page = Page(
        image_path="/tmp/p1.png",
        width=120,
        height=100,
        blocks=[
            Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 100, 90), lines=[line, inferred_line]),
        ],
    )

    svc = CharIndexService().build_index(OcrProject(name="hide-fallback", pages=[page]))

    assert svc.query("估") == []
    assert svc.query("推") == []
    assert svc.query("真实")[0].collection_kind == "token"
    assert svc.query("真") == []

    print("test_char_index_hides_fallback_units_by_default PASSED")


def test_char_index_indexes_digit_runs_as_single_digits():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="2026年",
        confidence=0.93,
        bbox=BBox(10, 20, 100, 24),
        chars=[
            Char(char="2", confidence=0.93, bbox=BBox(10, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="0", confidence=0.93, bbox=BBox(22, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="0"),
            Char(char="2", confidence=0.93, bbox=BBox(34, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="6", confidence=0.93, bbox=BBox(46, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="6"),
            Char(char="年", confidence=0.93, bbox=BBox(66, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="年"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=200,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 120, 40), lines=[line])],
    )

    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="digit-group", pages=[page]))
    assert svc.query("2026") == []
    two_entries = svc.query("2")
    assert len(two_entries) == 2
    assert [entry.bbox for entry in two_entries] == [
        BBox(10, 20, 12, 24),
        BBox(34, 20, 12, 24),
    ]
    assert all(entry.collection_kind == "char" for entry in two_entries)
    assert len(svc.query("0")) == 1
    assert len(svc.query("6")) == 1
    assert len(svc.query("年")) == 1

    print("test_char_index_indexes_digit_runs_as_single_digits PASSED")


def test_char_index_does_not_merge_plain_digit_with_circled_marker():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="附表1①。",
        confidence=0.93,
        bbox=BBox(10, 20, 120, 28),
        chars=[
            Char(char="附", confidence=0.93, bbox=BBox(10, 20, 18, 28), bbox_source="ocr", bbox_granularity="char", token_text="附"),
            Char(char="表", confidence=0.93, bbox=BBox(30, 20, 18, 28), bbox_source="ocr", bbox_granularity="char", token_text="表"),
            Char(char="1", confidence=0.93, bbox=BBox(52, 20, 10, 28), bbox_source="ocr", bbox_granularity="char", token_text="1"),
            Char(char="①", confidence=0.93, bbox=BBox(66, 20, 20, 28), bbox_source="ocr", bbox_granularity="char", token_text="①"),
            Char(char="。", confidence=0.93, bbox=BBox(90, 20, 8, 28), bbox_source="ocr", bbox_granularity="char", token_text="。"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=200,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 150, 50), lines=[line])],
    )

    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="marker-split", pages=[page]))

    assert svc.query("1①") == []
    assert svc.first_entry("1").bbox == BBox(52, 20, 10, 28)
    assert svc.first_entry("①").bbox == BBox(66, 20, 20, 28)

    print("test_char_index_does_not_merge_plain_digit_with_circled_marker PASSED")


def test_char_index_filters_non_cjk_from_default_vproof():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="税2026A+B，业",
        confidence=0.93,
        bbox=BBox(10, 20, 180, 24),
        chars=[
            Char(char="税", confidence=0.93, bbox=BBox(10, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="税"),
            Char(char="2", confidence=0.93, bbox=BBox(34, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="0", confidence=0.93, bbox=BBox(44, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="0"),
            Char(char="2", confidence=0.93, bbox=BBox(54, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="6", confidence=0.93, bbox=BBox(64, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="6"),
            Char(char="A", confidence=0.93, bbox=BBox(84, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="A"),
            Char(char="+", confidence=0.93, bbox=BBox(96, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="+"),
            Char(char="B", confidence=0.93, bbox=BBox(106, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="B"),
            Char(char="，", confidence=0.93, bbox=BBox(120, 20, 8, 24), bbox_source="ocr", bbox_granularity="char", token_text="，"),
            Char(char="业", confidence=0.93, bbox=BBox(136, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="业"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=200,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 190, 40), lines=[line])],
    )

    svc = CharIndexService().build_index(OcrProject(name="default-cjk-only", pages=[page]))

    assert len(svc.query("税")) == 1
    assert len(svc.query("业")) == 1
    assert svc.query("2026") == []
    assert svc.query("A+B") == []
    assert svc.query("，") == []

    print("test_char_index_filters_non_cjk_from_default_vproof PASSED")


def test_char_index_sorts_digit_characters_as_buckets():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="9 11 2026 税",
        confidence=0.9,
        bbox=BBox(10, 20, 180, 24),
        chars=[
            Char(char="9", confidence=0.9, bbox=BBox(10, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="9"),
            Char(char=" ", confidence=0.9, bbox=None),
            Char(char="1", confidence=0.9, bbox=BBox(28, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="1"),
            Char(char="1", confidence=0.9, bbox=BBox(38, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="1"),
            Char(char=" ", confidence=0.9, bbox=None),
            Char(char="2", confidence=0.9, bbox=BBox(56, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="0", confidence=0.9, bbox=BBox(66, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="0"),
            Char(char="2", confidence=0.9, bbox=BBox(76, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="6", confidence=0.9, bbox=BBox(86, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="6"),
            Char(char=" ", confidence=0.9, bbox=None),
            Char(char="税", confidence=0.9, bbox=BBox(104, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="税"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=200,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 180, 40), lines=[line])],
    )
    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="digit-sort", pages=[page]))
    sorted_keys = [key for key, _count in svc.sorted_chars()]
    digit_keys = [key for key in sorted_keys if key.isdigit()]
    assert digit_keys == ["0", "1", "2", "6", "9"]
    assert svc.query("11") == []
    assert svc.query("2026") == []

    print("test_char_index_sorts_digit_characters_as_buckets PASSED")


def test_char_index_exposes_latin_formula_like_runs_as_chars():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="税2026A+B",
        confidence=0.9,
        bbox=BBox(10, 20, 160, 24),
        chars=[
            Char(char="税", confidence=0.9, bbox=BBox(10, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="税"),
            Char(char="2", confidence=0.9, bbox=BBox(34, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="0", confidence=0.9, bbox=BBox(44, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="0"),
            Char(char="2", confidence=0.9, bbox=BBox(54, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="2"),
            Char(char="6", confidence=0.9, bbox=BBox(64, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="6"),
            Char(char="A", confidence=0.9, bbox=BBox(84, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="A"),
            Char(char="+", confidence=0.9, bbox=BBox(96, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="+"),
            Char(char="B", confidence=0.9, bbox=BBox(106, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="B"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=200,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 180, 40), lines=[line])],
    )
    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="formula-group", pages=[page]))

    assert svc.first_entry("A").bbox == BBox(84, 20, 12, 24)
    assert svc.first_entry("+").bbox == BBox(96, 20, 10, 24)
    assert svc.first_entry("B").bbox == BBox(106, 20, 12, 24)
    assert svc.query("A+B") == []

    sorted_keys = [key for key, _count in svc.sorted_chars()]
    assert "2026" not in sorted_keys

    print("test_char_index_exposes_latin_formula_like_runs_as_chars PASSED")


def test_char_index_keeps_formula_span_separate_from_word_level_inline_formula_carrier():
    from app.core.paddle_line_routing import ROUTE_INLINE_FORMULA_FLAG
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    carrier = "$ Incentive_{c} \\times Post_{t} $"
    line = Line(
        text=f"A+{carrier}税",
        confidence=0.9,
        bbox=BBox(10, 20, 220, 24),
        review_flags=[ROUTE_INLINE_FORMULA_FLAG],
        chars=[
            Char(char="A", confidence=0.9, bbox=BBox(10, 20, 12, 24), bbox_source="ocr", bbox_granularity="char", token_text="A"),
            Char(char="+", confidence=0.9, bbox=BBox(22, 20, 10, 24), bbox_source="ocr", bbox_granularity="char", token_text="+"),
            Char(
                char=carrier,
                confidence=0.9,
                bbox=BBox(40, 20, 140, 24),
                bbox_source="paddle_inline_formula",
                bbox_granularity="word",
                token_text=carrier,
            ),
            Char(char="税", confidence=0.9, bbox=BBox(188, 20, 18, 24), bbox_source="ocr", bbox_granularity="char", token_text="税"),
        ],
    )
    page = Page(
        image_path="p1.png",
        width=240,
        height=120,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 220, 40), lines=[line])],
    )

    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="formula-carrier-guard", pages=[page]))

    assert svc.first_entry("A").bbox == BBox(10, 20, 12, 24)
    assert svc.first_entry("+").bbox == BBox(22, 20, 10, 24)
    assert svc.query("A+") == []

    carrier_entry = svc.first_entry(carrier)
    assert carrier_entry is not None
    assert carrier_entry.bbox_source == "paddle_inline_formula"
    assert carrier_entry.bbox_granularity == "word"
    assert carrier_entry.char_idx == 2
    assert line.chars[2].char == carrier

    trailing_entry = svc.first_entry("税")
    assert trailing_entry is not None
    assert trailing_entry.char_idx == len("A+") + len(carrier)
    assert trailing_entry.bbox == BBox(188, 20, 18, 24)

    print("test_char_index_keeps_formula_span_separate_from_word_level_inline_formula_carrier PASSED")


def test_vproof_index_geometry_echo_renders_display_offset_overlay(tmp_path):
    import cv2
    import numpy as np

    from scripts.render_vproof_index_geometry_echo import render_vproof_index_geometry_echo

    result = render_vproof_index_geometry_echo(tmp_path)
    carrier = result["carrier"]
    tax_entry = result["tax_entry"]
    after_entry = result["after_entry"]
    flat_text = result["flat_text"]
    text_map = result["text_map"]
    overlay_path = result["overlay_path"]
    report_path = result["report_path"]

    assert tax_entry.char_idx == len("A+") + len(carrier)
    assert after_entry.char_idx == tax_entry.char_idx + 1
    assert flat_text[text_map[tax_entry.char_idx].start] == "税"
    assert flat_text[text_map[after_entry.char_idx].start] == "后"
    assert overlay_path.exists()
    assert report_path.exists()

    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    red_pixels = np.count_nonzero(
        (overlay[:, :, 2] > 180) & (overlay[:, :, 1] < 80) & (overlay[:, :, 0] < 80)
    )
    assert red_pixels > 30

    print("test_vproof_index_geometry_echo_renders_display_offset_overlay PASSED")


def test_char_index_suppresses_punctuation_topic_for_shared_token_bbox():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    shared_bbox = BBox(10, 20, 28, 24)
    line = Line(
        text="税。",
        confidence=0.91,
        bbox=shared_bbox,
        chars=[
            Char(char="税", confidence=0.91, bbox=shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税。"),
            Char(char="。", confidence=0.91, bbox=shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税。"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 40), lines=[line])],
    )
    svc = CharIndexService().build_index(OcrProject(name="punct-suppress", pages=[page]))

    assert len(svc.query("税")) == 1
    assert svc.query("。") == []
    assert svc.first_entry("税").bbox_granularity == "word"

    print("test_char_index_suppresses_punctuation_topic_for_shared_token_bbox PASSED")


def test_char_index_uses_token_collection_for_word_level_han_bbox():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    shared_bbox = BBox(20, 20, 40, 24)
    line = Line(
        text="税业",
        confidence=0.92,
        bbox=shared_bbox,
        chars=[
            Char(char="税", confidence=0.92, bbox=shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税业"),
            Char(char="业", confidence=0.92, bbox=shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税业"),
        ],
    )
    page = Page(
        image_path="/tmp/p1.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 100, 40), lines=[line])],
    )
    svc = CharIndexService().build_index(OcrProject(name="word-level-token", pages=[page]))

    assert len(svc.query("税业")) == 1
    assert svc.query("税") == []
    assert svc.query("业") == []
    assert svc.first_entry("税业").collection_kind == "token"

    print("test_char_index_uses_token_collection_for_word_level_han_bbox PASSED")


def test_char_index_skips_empty_narrow_ocr_bbox():
    import tempfile

    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 20), (26, 48), (0, 0, 0), -1)
        cv2.imwrite(page_path, image)

        line = Line(
            text="甲乙",
            confidence=0.92,
            bbox=BBox(10, 20, 60, 28),
            chars=[
                Char(char="甲", confidence=0.92, bbox=BBox(10, 20, 16, 28), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
                Char(char="乙", confidence=0.92, bbox=BBox(60, 22, 1, 1), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
            ],
        )
        page = Page(
            image_path=page_path,
            width=120,
            height=80,
            blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 40), lines=[line])],
        )
        svc = CharIndexService().build_index(OcrProject(name="bad-box-filter", pages=[page]))

        assert len(svc.query("甲")) == 1
        assert svc.query("乙") == []
    finally:
        os.unlink(page_path)

    print("test_char_index_skips_empty_narrow_ocr_bbox PASSED")


def test_char_index_reloads_page_image_between_builds_for_same_path():
    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        white = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.imwrite(page_path, white)

        line = Line(
            text="甲",
            confidence=0.92,
            bbox=BBox(10, 20, 30, 30),
        )
        page = Page(
            image_path=page_path,
            width=120,
            height=80,
            blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 50), lines=[line])],
        )
        project = OcrProject(name="char-index-image-cache", pages=[page])
        svc = CharIndexService(include_fallback=True)

        svc.build_index(project)
        assert svc.query("甲") == []

        ink = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.rectangle(ink, (10, 20), (40, 50), (0, 0, 0), -1)
        cv2.imwrite(page_path, ink)

        svc.build_index(project)
        assert svc.query("甲")
    finally:
        os.unlink(page_path)

    print("test_char_index_reloads_page_image_between_builds_for_same_path PASSED")


def test_char_index_skips_lines_with_unverified_geometry():
    import tempfile

    import cv2
    import numpy as np

    from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        image[10:70, 10:110] = 0
        cv2.imwrite(img_path, image)

    try:
        line = Line(
            text="有效OCR文本",
            confidence=0.9,
            bbox=BBox(0, 0, 120, 80),
            review_flags=[MISSING_LINE_BBOX_FLAG],
        )
        page = Page(
            image_path=img_path,
            width=120,
            height=80,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 80), lines=[line])],
        )
        svc = CharIndexService().build_index(OcrProject(name="unverified", pages=[page]))
        assert svc.query("有") == []
        assert line.text == "有效OCR文本"
    finally:
        os.unlink(img_path)

    print("test_char_index_skips_lines_with_unverified_geometry PASSED")


def test_char_index_deduplicates_overlapping_duplicate_lines():
    import tempfile

    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        image = np.full((100, 180, 3), 255, dtype=np.uint8)
        image[20:45, 20:100] = 0
        cv2.imwrite(img_path, image)

    try:
        line_a = Line(text="重复", confidence=0.9, bbox=BBox(20, 20, 80, 25))
        line_b = Line(text="重复", confidence=0.9, bbox=BBox(21, 20, 80, 25))
        page = Page(
            image_path=img_path,
            width=180,
            height=100,
            blocks=[
                Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 120, 50), lines=[line_a]),
                Block(block_type=BlockType.TEXT, bbox=BBox(10, 10, 120, 50), lines=[line_b]),
            ],
        )
        svc = CharIndexService(include_fallback=True).build_index(OcrProject(name="dedupe", pages=[page]))
        assert len(svc.query("重")) == 1
        assert len(svc.query("复")) == 1
    finally:
        os.unlink(img_path)

    print("test_char_index_deduplicates_overlapping_duplicate_lines PASSED")


def test_vproof_reference_context_deduplicates_overlapping_duplicate_lines():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.proof_reference_context import build_proof_reference_context

    line_a = Line(text="重复行", confidence=0.9, bbox=BBox(10, 10, 90, 20))
    line_b = Line(text="重复行", confidence=0.9, bbox=BBox(11, 10, 90, 20))
    line_c = Line(text="重复行", confidence=0.9, bbox=BBox(10, 60, 90, 20))
    page = Page(
        image_path="/tmp/p.png",
        width=200,
        height=120,
        blocks=[
            Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 40), lines=[line_a, line_b]),
            Block(block_type=BlockType.TEXT, bbox=BBox(0, 50, 120, 40), lines=[line_c]),
        ],
    )

    context = build_proof_reference_context(page)
    text = context.text
    mapping = context.slots
    assert text.count("重复行") == 2
    assert sum(1 for slot in mapping if slot.line is line_b) == 0
    assert sum(1 for slot in mapping if slot.line is line_c) == 3

    print("test_vproof_reference_context_deduplicates_overlapping_duplicate_lines PASSED")


def test_ui_import_smoke():
    import importlib
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    v_proof = importlib.import_module("app.ui.proof.v_proof")
    main_window = importlib.import_module("app.ui.main_window")

    assert hasattr(v_proof, "VProofPanel")
    assert hasattr(main_window, "MainWindow")

    print("test_ui_import_smoke PASSED")


def test_page_image_cache():
    import tempfile
    import cv2
    import numpy as np

    from app.core.coordinate_seam import BBOX_SPACE_CROP, CropCoordinateSeam
    from app.core.page_image_cache import PageImageCache
    from app.models import BBox

    with tempfile.TemporaryDirectory() as tmpdir:
        first = os.path.join(tmpdir, "p1.png")
        second = os.path.join(tmpdir, "p2.png")
        third = os.path.join(tmpdir, "p3.png")

        cv2.imwrite(first, np.full((20, 30, 3), 40, dtype=np.uint8))
        cv2.imwrite(second, np.full((20, 30, 3), 80, dtype=np.uint8))
        cv2.imwrite(third, np.full((20, 30, 3), 120, dtype=np.uint8))

        cache = PageImageCache(max_pages=2)
        img1 = cache.get_page_image(first)
        img1_again = cache.get_page_image(first)
        crop = cache.get_bbox_crop(first, BBox(5, 6, 10, 8))
        seam = CropCoordinateSeam.from_page_bbox(BBox(10, 4, 15, 10), page_w=30, page_h=20)
        seam_crop = cache.get_bbox_crop(
            first,
            BBox(3, 2, 4, 5),
            source_space=BBOX_SPACE_CROP,
            seam=seam,
        )

        assert img1.shape == (20, 30, 3)
        assert img1_again is img1
        assert crop.shape == (8, 10, 3)
        assert seam_crop.shape == (5, 4, 3)

        line_path = os.path.join(tmpdir, "line.png")
        line_img = np.full((120, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(line_img, (16, 18), (204, 30), (0, 0, 0), -1)
        cv2.rectangle(line_img, (18, 68), (196, 82), (0, 0, 0), -1)
        cv2.imwrite(line_path, line_img)
        loose_crop = cache.get_bbox_crop(line_path, BBox(10, 44, 200, 46))
        refined_crop = cache.get_line_crop(line_path, BBox(10, 44, 200, 46), pad_y=2)
        assert refined_crop.shape[0] < loose_crop.shape[0]
        assert refined_crop.shape[1] >= 170

        cache.get_page_image(second)
        cache.get_page_image(third)
        assert cache.cached_paths() == [second, third]

    print("test_page_image_cache PASSED")


def test_proof_stats_service():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, ProofStatus
    from app.services.proof_stats_service import ProofStatsService

    bb = BBox(0, 0, 100, 20)
    page = Page(image_path="/tmp/proof.png", width=400, height=300)
    confirmed = Line(text="确认", confidence=0.9, bbox=bb)
    set_line_proof_status(confirmed, ProofStatus.OK)
    modified = Line(text="修改", confidence=0.9, bbox=bb)
    set_line_proof_status(modified, ProofStatus.MODIFIED)
    flagged = Line(text="疑点", confidence=0.6, bbox=bb, review_flags=["low_confidence"])
    pending = Line(text="待处理", confidence=0.9, bbox=bb)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=bb,
            lines=[confirmed, modified, flagged, pending],
        )
    ]

    stats = ProofStatsService().summarize(OcrProject(name="proof-stats", pages=[page]))

    assert stats.total_lines == 4
    assert stats.confirmed_lines == 1
    assert stats.modified_lines == 1
    assert stats.flagged_lines == 1
    assert stats.pending_lines == 1
    assert stats.to_dict()["flagged_lines"] == 1

    print("test_proof_stats_service PASSED")


def test_api_model_profile_helpers():
    from app.core.api_profiles import (
        get_api_model_profile_options,
        get_api_model_profile_url,
        match_api_model_profile_from_url,
    )

    options = get_api_model_profile_options()
    assert [label for _, label in options] == [
        "PP-OCRv5",
        "PaddleOCR-VL-1.6",
    ]
    assert get_api_model_profile_url("pp-ocrv5").endswith("/ocr")
    assert get_api_model_profile_url("paddleocr-vl-1.6").endswith("/api/v2/ocr/jobs")
    assert get_api_model_profile_url("unknown-profile").endswith("/api/v2/ocr/jobs")
    assert match_api_model_profile_from_url("https://n6z9feddjca4l7b5.aistudio-app.com/ocr") == "pp-ocrv5"
    assert match_api_model_profile_from_url("https://example.com/custom-layout") is None

    print("test_api_model_profile_helpers PASSED")


def test_api_endpoint_role_resolution_keeps_layout_and_proof_separate():
    """Layout role 已全面切到 PaddleOCR-VL-1.6；OCR proof role 仍走 PP-OCRv5。

    官方 PP-OCRv5 / PaddleOCR-VL-1.6 根地址按 role 映射到固定链路；
    自托管根 URL 按 role 自动补 /api/v2/ocr/jobs 或 /ocr。旧
    /layout-parsing 只作为历史配置后缀剥离，不再作为请求端点。
    """
    from app.core.api_profiles import (
        get_api_model_profile_url,
        normalize_api_base_url,
        resolve_api_endpoint_for_role,
    )

    vl16_url = get_api_model_profile_url("paddleocr-vl-1.6")
    ocr_url = get_api_model_profile_url("pp-ocrv5")
    vl16_root = vl16_url.removesuffix("/api/v2/ocr/jobs")
    ocr_root = ocr_url.removesuffix("/ocr")
    legacy_structure_url = "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing"
    legacy_vl_url = "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing"
    legacy_vl15_root = "https://15j75bd0964dzbwe.aistudio-app.com"

    assert normalize_api_base_url(ocr_url) == ocr_root
    assert normalize_api_base_url(vl16_url) == vl16_root
    assert normalize_api_base_url("https://self-hosted.example.com/layout-parsing") == "https://self-hosted.example.com"
    assert normalize_api_base_url(legacy_structure_url) == vl16_root
    assert normalize_api_base_url(legacy_vl_url) == vl16_root
    assert normalize_api_base_url(legacy_vl15_root) == vl16_root

    # Layout role: 官方 PP-OCRv5 预设 -> VL-1.6
    assert resolve_api_endpoint_for_role(
        ocr_url,
        profile="pp-ocrv5",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="pp-ocrv5",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        legacy_structure_url,
        profile="",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        legacy_vl_url,
        profile="",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        legacy_vl15_root,
        profile="",
        role="layout",
    ) == vl16_url

    # OCR role: VL1.6 预设 -> PP-OCRv5；PP-OCRv5 自身保持
    assert resolve_api_endpoint_for_role(
        vl16_url,
        profile="paddleocr-vl-1.6",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        vl16_root,
        profile="",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        legacy_structure_url,
        profile="",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        ocr_url,
        profile="pp-ocrv5",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        ocr_url,
        profile="pp-ocrv5",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="",
        role="layout",
    ) == vl16_url

    # 自托管根 URL: 不做 host 跳转，仅按 VL1.6 jobs suffix 补全
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com",
        profile="",
        role="layout",
    ) == "https://self-hosted.example.com/api/v2/ocr/jobs"
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com/layout-parsing",
        profile="",
        role="layout",
    ) == "https://self-hosted.example.com/api/v2/ocr/jobs"
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com/layout-parsing",
        profile="",
        role="ocr",
    ) == "https://self-hosted.example.com/ocr"
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com/ocr",
        profile="",
        role="layout",
    ) == "https://self-hosted.example.com/api/v2/ocr/jobs"

    print("test_api_endpoint_role_resolution_keeps_layout_and_proof_separate PASSED")


def test_api_http_post_json_disables_environment_proxies():
    import requests

    from app.core.api_http import post_json_without_env_proxy

    captured = {}

    class DummyResponse:
        status_code = 200

    def fake_post(url, json, headers, timeout, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["proxies"] = kwargs.get("proxies")
        return DummyResponse()

    original_post = requests.post
    original_env = {key: os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
    requests.post = fake_post
    os.environ["HTTP_PROXY"] = "http://bad-proxy.invalid:9999"
    os.environ["HTTPS_PROXY"] = "http://bad-proxy.invalid:9999"
    os.environ["ALL_PROXY"] = "http://bad-proxy.invalid:9999"
    try:
        response = post_json_without_env_proxy(
            "https://example.com/ocr",
            json={"image": "abc"},
            headers={"Content-Type": "application/json"},
            timeout=12,
        )
        assert response.status_code == 200
        assert captured["url"] == "https://example.com/ocr"
        assert captured["json"] == {"image": "abc"}
        assert captured["timeout"] == 12
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
    finally:
        requests.post = original_post
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("test_api_http_post_json_disables_environment_proxies PASSED")


def test_app_config_tracks_api_model_profile():
    from app.core.app_config import AppConfig, get_config, update_config

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    defaults = get_config()
    assert defaults["mode"] == "local"
    assert defaults["api_model_profile"] == ""
    assert defaults["layout_concurrency"] == 8
    assert defaults["ocr_page_concurrency"] == 2
    assert defaults["paddle_api_network_mode"] == "auto"
    assert defaults["layout_debug_artifacts"] is False

    update_config(
        mode="api",
        api_model_profile="paddleocr-vl-1.6",
        api_url="https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        api_token="demo",
        api_timeout=12,
        api_layout_model_name="",
        layout_concurrency=3,
        ocr_page_concurrency=4,
        paddle_api_network_mode="env_proxy",
        layout_debug_artifacts="true",
    )
    current = get_config()
    assert current["api_model_profile"] == "paddleocr-vl-1.6"
    assert current["api_url"] == "https://paddleocr.aistudio-app.com"
    assert current["api_timeout"] == 12
    assert current["api_token"] == "demo"
    assert current["api_layout_model_name"] == ""
    assert current["layout_concurrency"] == 3
    assert current["ocr_page_concurrency"] == 4
    assert current["paddle_api_network_mode"] == "env_proxy"
    assert current["layout_debug_artifacts"] is True
    cfg.reset_to_defaults()

    print("test_app_config_tracks_api_model_profile PASSED")


def test_api_settings_dialog_syncs_model_and_url():
    from PySide6.QtWidgets import QApplication

    from app.core.app_config import AppConfig, update_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    app = QApplication.instance() or QApplication([])
    assert app is not None

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(
        mode="local",
        api_model_profile="",
        api_url="https://example.com/root",
        api_token="old",
        layout_concurrency=4,
        ocr_page_concurrency=3,
        paddle_api_network_mode="direct",
    )

    dialog = ApiSettingsDialog()
    assert dialog._radio_hanwang.isChecked()
    assert dialog._selected_mode() == "hanwang"
    assert dialog._mode_card.isHidden() is True
    assert dialog._api_model_row.isHidden()
    assert dialog._api_model_combo.currentIndex() == -1
    assert dialog._url_edit.text() == "https://example.com/root"
    assert "汉王混合链路" in dialog._summary_model.text()
    assert dialog._api_form_panel.isEnabled() is True
    assert not dialog._timeout_row.isHidden()
    assert dialog._timeout_spin.maximum() >= 600
    assert dialog._layout_concurrency_spin.value() == 4
    assert dialog._layout_concurrency_spin.maximum() == 10
    assert dialog._ocr_page_concurrency_spin.value() == 3
    assert dialog._ocr_page_concurrency_spin.maximum() == 20
    assert dialog._selected_network_mode() == "direct"

    cfg.reset_to_defaults()

    print("test_api_settings_dialog_syncs_model_and_url PASSED")


def _get_qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _reset_app_config_for_test(tmpdir: str) -> None:
    from PySide6.QtCore import QSettings
    from app.core.app_config import AppConfig

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmpdir)
    AppConfig._instance = None
    AppConfig.instance().reset_to_defaults()


def test_api_settings_dialog_keeps_model_preset_sync():
    from app.core.app_config import AppConfig
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()

        assert dialog._api_model_row.isHidden()
        assert dialog._model_note.isHidden()
        assert dialog._api_form_panel.isEnabled() is True
        assert dialog._btn_test.isEnabled() is True
        assert dialog._mode_card.isHidden() is True
        assert "汉王混合链路" in dialog._summary_model.text()

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_keeps_model_preset_sync PASSED")


def test_api_settings_dialog_reverse_matches_url_and_persists_profile():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._url_edit.setText("https://example.com/custom")
        dialog._token_edit.setText("secret")
        dialog._sync_model_from_url()
        assert dialog._api_model_combo.currentIndex() == -1
        assert "汉王混合链路" in dialog._summary_model.text()

        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["mode"] == "hanwang"
        assert cfg["api_model_profile"] == ""
        assert cfg["api_url"] == "https://example.com/custom"
        assert cfg["api_token"] == "secret"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_reverse_matches_url_and_persists_profile PASSED")


def test_api_settings_dialog_saves_base_url_from_endpoint_suffix():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._url_edit.setText("https://example.com/custom/layout-parsing")
        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["api_url"] == "https://example.com/custom"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_saves_base_url_from_endpoint_suffix PASSED")


def test_api_settings_dialog_migrates_legacy_official_layout_url():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._url_edit.setText("https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing")
        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["api_url"] == "https://paddleocr.aistudio-app.com"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_migrates_legacy_official_layout_url PASSED")


def test_api_settings_dialog_collapses_mode_to_hanwang_when_saving():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._radio_local.setChecked(True)
        dialog._url_edit.setText("https://example.com/custom")
        dialog._token_edit.setText("secret")

        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["mode"] == "hanwang"
        assert cfg["api_url"] == "https://example.com/custom"
        assert cfg["api_token"] == "secret"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_collapses_mode_to_hanwang_when_saving PASSED")


def test_api_settings_dialog_persists_hanwang_mode_with_api_runtime():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._radio_hanwang.setChecked(True)
        dialog._url_edit.setText("https://example.com/custom/ocr")
        dialog._token_edit.setText("kept-for-api")

        assert dialog._selected_mode() == "hanwang"
        assert dialog._api_form_panel.isEnabled() is True
        assert dialog._btn_test.isEnabled() is True
        assert "汉王混合链路" in dialog._summary_model.text()
        assert "PaddleOCR-VL-1.6" in dialog._summary_desc.text()
        assert "需要 API 地址与 Token" in dialog._api_mode_notice.text()

        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["mode"] == "hanwang"
        assert cfg["api_url"] == "https://example.com/custom"
        assert cfg["api_token"] == "kept-for-api"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_persists_hanwang_mode_with_api_runtime PASSED")


def test_fixed_api_chain_resolves_official_roots_to_vl16_and_ppocrv5():
    from app.core.api_profiles import get_api_model_profile_url, resolve_api_endpoint_for_role

    vl16_url = get_api_model_profile_url("paddleocr-vl-1.6")
    ppocr_url = get_api_model_profile_url("pp-ocrv5")
    vl16_root = vl16_url.removesuffix("/api/v2/ocr/jobs")
    ppocr_root = ppocr_url.removesuffix("/ocr")

    assert resolve_api_endpoint_for_role(vl16_root, role="layout") == vl16_url
    assert resolve_api_endpoint_for_role(vl16_root, role="ocr") == ppocr_url
    assert resolve_api_endpoint_for_role(ppocr_root, role="layout") == vl16_url
    assert resolve_api_endpoint_for_role(ppocr_root, role="ocr") == ppocr_url

    print("test_fixed_api_chain_resolves_official_roots_to_vl16_and_ppocrv5 PASSED")


def test_layout_analyzer_rescales_suspicious_blocks():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Block, BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=2400, height=3200)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(50, 40, 300, 80)),
        Block(block_type=BlockType.TEXT, bbox=BBox(60, 180, 320, 120)),
        Block(block_type=BlockType.TEXT, bbox=BBox(80, 420, 400, 120)),
    ]

    analyzer._rescale_blocks_if_suspicious(page)

    assert page.blocks[0].bbox.x > 100
    assert page.blocks[-1].bbox.y > 1500
    assert page.blocks[-1].bbox.y2 <= page.height

    print("test_layout_analyzer_rescales_suspicious_blocks PASSED")


def test_layout_analyzer_rescales_snapshot_view_bboxes_into_runtime_projection():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import (
        BBox, Block, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot,
        OcrPolicy, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test-snapshot-rescale.png", width=2400, height=3200)
    blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(900, 900, 30, 20)),
        Block(block_type=BlockType.TEXT, bbox=BBox(950, 950, 30, 20)),
        Block(block_type=BlockType.TEXT, bbox=BBox(1000, 1000, 30, 20)),
    ]
    page.blocks = blocks
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-layout-rescale",
        source_engine="test",
        source_run_id="run-layout-rescale",
        blocks=(
            LayoutBlockSnapshot(
                uid=blocks[0].uid,
                block_type=BlockType.TEXT,
                bbox=BBox(50, 40, 300, 80),
                order=0,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
            LayoutBlockSnapshot(
                uid=blocks[1].uid,
                block_type=BlockType.TEXT,
                bbox=BBox(60, 180, 320, 120),
                order=1,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
            LayoutBlockSnapshot(
                uid=blocks[2].uid,
                block_type=BlockType.TEXT,
                bbox=BBox(80, 420, 400, 120),
                order=2,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
        ),
    ))

    analyzer._rescale_blocks_if_suspicious(page)

    assert blocks[0].bbox.x == 150
    assert blocks[0].bbox.y > 200
    assert blocks[0].bbox.w > 800
    assert blocks[2].bbox.y > 2000

    print("test_layout_analyzer_rescales_snapshot_view_bboxes_into_runtime_projection PASSED")


def test_layout_analyzer_extracts_api_polygon_bbox():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    bbox = analyzer._extract_bbox_from_coordinate(
        [[10, 20], [210, 20], [210, 120], [10, 120]],
        page,
    )

    assert bbox == BBox(10, 20, 200, 100)

    print("test_layout_analyzer_extracts_api_polygon_bbox PASSED")


def test_layout_analyzer_extracts_api_blocks_from_v16_schema():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "paragraph_title",
                                    "coordinate": [10, 20, 210, 120],
                                    "score": 0.91,
                                },
                                {
                                    "label": "chart",
                                    "polygon_points": [[420, 60], [560, 60], [560, 180], [420, 180]],
                                    "score": 0.75,
                                },
                            ],
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "reference_content",
                                "block_bbox": [600, 80, 760, 150],
                                "block_content": "参考文献",
                            }
                        ],
                    }
                }
            ]
        }
    }

    blocks, overlays = analyzer._extract_api_blocks(page, data)

    assert [block.block_type for block in blocks] == [
        BlockType.REFERENCE,
    ]
    assert blocks[0].bbox.x == 600 and blocks[0].bbox.y == 80
    assert "参考文献" in blocks[0].note
    assert [label for label, _bbox in overlays] == [
        "reference_content",
        "paragraph_title",
        "chart",
    ]

    print("test_layout_analyzer_extracts_api_blocks_from_v16_schema PASSED")


def test_paddle_authority_prefers_block_label_over_conflicting_label_everywhere():
    import numpy as np

    from app.core.block_attributes import block_attributes
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.engines.hanwang.micro_recblock import run_micro_recblock
    from app.models import BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/authority-conflict.png", width=100, height=100)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {
                                "block_label": "equation",
                                "label": "text",
                                "block_bbox": [10, 10, 80, 30],
                                "block_content": "E=mc^2",
                            }
                        ]
                    }
                }
            ]
        }
    }

    blocks, _overlays = analyzer._extract_api_blocks(page, data)
    attrs = block_attributes(blocks[0])
    rows, stats = run_micro_recblock(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [{"block_label": "equation", "label": "text", "block_bbox": [10, 10, 80, 30], "block_content": "E=mc^2"}],
    )

    assert blocks[0].block_type == BlockType.EQUATION
    assert attrs.semantic_label == "equation"
    assert attrs.semantic_block_type == BlockType.EQUATION
    assert rows[0].source == "ppvl"
    assert rows[0].block_label == "equation"
    assert stats.n_blocks_hanwang == 0
    assert stats.n_blocks_ppvl == 1

    print("test_paddle_authority_prefers_block_label_over_conflicting_label_everywhere PASSED")


def test_layout_parsing_semantics_override_layout_det_when_both_exist():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/layout-semantic-mismatch.png", width=100, height=100)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {"label": "text", "coordinate": [5, 5, 90, 25]},
                            ],
                        },
                        "parsing_res_list": [
                            {"block_label": "equation", "block_bbox": [10, 10, 80, 30], "block_content": "x+y"},
                        ],
                    }
                }
            ]
        }
    }

    blocks, overlays = analyzer._extract_api_blocks(page, data)

    assert _raw_layout_records(page)[0]["block_label"] == "equation"
    assert [block.block_type for block in blocks] == [BlockType.EQUATION]
    assert blocks[0].source_label == "equation"
    assert len(overlays) == 2
    assert overlays[0][0] == "equation"
    assert overlays[1][0] == "text"

    print("test_layout_parsing_semantics_override_layout_det_when_both_exist PASSED")


def test_layout_analyzer_forwards_route_subblocks_from_layout_det_res():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import LAYOUT_LINE_ROUTES_FIELD, line_routes_for_block
    from app.core.raw_ocr_artifact import layout_route_attachments, layout_records_with_route_attachments
    from app.models import BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/route-subblocks.png", width=240, height=120)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {"label": "inline_formula", "coordinate": [60, 20, 90, 42]},
                                {"label": "table_region", "coordinate": [130, 10, 200, 70]},
                                {"label": "figure_caption", "coordinate": [10, 80, 90, 100]},
                            ],
                        },
                        "parsing_res_list": [
                            {"block_label": "text", "block_bbox": [10, 5, 220, 75], "block_content": "正文x+y表格"},
                        ],
                    }
                }
            ]
        }
    }

    blocks, overlays = analyzer._extract_api_blocks(page, data)
    assert "_route_subblocks" not in _raw_layout_records(page)[0]
    subblocks = layout_route_attachments(page)[0]
    routed_record = layout_records_with_route_attachments(page)[0]
    line_routes = line_routes_for_block(routed_record, page.width, page.height)

    assert blocks[0].block_type == BlockType.TEXT
    assert blocks[0].origin is not None
    assert blocks[0].origin.raw_index == 0
    assert not hasattr(blocks[0], "raw_payload")
    assert LAYOUT_LINE_ROUTES_FIELD not in _raw_layout_records(page)[0]
    assert [item["block_label"] for item in subblocks] == ["inline_formula", "table_region"]
    assert subblocks[0]["block_bbox"] == [60, 20, 90, 42]
    assert subblocks[0]["raw_payload"]["label"] == "inline_formula"
    assert any(segment["kind"] == "formula" for route in line_routes for segment in route["segments"])
    assert any(segment["kind"] == "skip" for route in line_routes for segment in route["segments"])
    assert [label for label, _bbox in overlays] == ["text", "inline_formula", "table_region", "figure_caption"]

    print("test_layout_analyzer_forwards_route_subblocks_from_layout_det_res PASSED")


def test_hanwang_layout_row_uses_page_artifact_origin_record():
    from app.engines.hanwang.micro_recblock import _layout_row_from_block
    from app.models import BBox, Block, BlockOrigin, BlockType, Page

    page = Page(image_path="/tmp/hanwang-layout-row.png", width=200, height=120)
    _attach_raw_layout_records(page, [
        {
            "block_label": "text",
            "block_bbox": [0, 0, 100, 40],
            "block_content": "artifact text",
        }
    ])
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox.from_xyxy(10, 12, 150, 52),
        source_label="text",
        origin=BlockOrigin(source_label="text", raw_index=0),
    )

    row = _layout_row_from_block(page, block)

    assert row["block_content"] == "artifact text"
    assert row["block_bbox"] == [10, 12, 150, 52]

    print("test_hanwang_layout_row_uses_page_artifact_origin_record PASSED")


def test_layout_analyzer_persists_raw_parsing_res_list():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    parsing_res_list = [
        {
            "block_label": "text",
            "block_bbox": [100, 200, 300, 260],
            "block_content": "PP-VL raw text",
            "custom_raw": {"keep": True},
        }
    ]
    data = {
        "result": {
            "layoutParsingResults": [
                {"prunedResult": {"parsing_res_list": parsing_res_list}}
            ]
        }
    }

    blocks, _ = analyzer._extract_api_blocks(page, data)

    assert len(blocks) == 1
    assert _raw_layout_records(page) == parsing_res_list
    assert _raw_layout_records(page)[0]["custom_raw"]["keep"] is True
    assert blocks[0].source_label == "text"
    assert not hasattr(blocks[0], "raw_payload")

    print("test_layout_analyzer_persists_raw_parsing_res_list PASSED")


def test_layout_analyzer_does_not_promote_ocr_results_to_layout_blocks():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    data = {
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "overall_ocr_res": {
                            "rec_texts": ["第一行", "第二行"],
                            "rec_scores": [0.95, 0.83],
                            "rec_polys": [
                                [10, 20, 210, 20, 210, 70, 10, 70],
                                [20, 100, 260, 100, 260, 150, 20, 150],
                            ],
                        }
                    }
                }
            ]
        }
    }

    blocks, overlays = analyzer._extract_api_blocks(page, data)

    assert blocks == []
    assert overlays == []
    assert _raw_layout_records(page) == []

    print("test_layout_analyzer_does_not_promote_ocr_results_to_layout_blocks PASSED")


# =====================================================================
# CharIndexService — 整条字索引链路：纵/横 bbox、去重、排序、稳定查询
# =====================================================================

def test_char_index_vertical_split():
    """纵排行 (h ≫ w) 的字符 bbox 应按高度等分，而非宽度。"""
    from app.services.char_index_service import _estimate_char_bbox
    from app.models import Line, BBox
    line = Line(text="永和九年", confidence=0.9, bbox=BBox(100, 100, 40, 160))
    boxes = [_estimate_char_bbox(line, i, 4) for i in range(4)]
    # 期望：x 不变，y 递增 40，w=bb.w，h=40
    assert boxes[0].x == 100 and boxes[0].y == 100
    assert boxes[1].y == 140
    assert boxes[2].y == 180
    assert boxes[3].y == 220
    for b in boxes:
        assert b.w == 40 and b.h == 40
    print("test_char_index_vertical_split PASSED")


def test_char_index_horizontal_split():
    """横排行 (w ≫ h) 的字符 bbox 应按宽度等分。"""
    from app.services.char_index_service import _estimate_char_bbox
    from app.models import Line, BBox
    line = Line(text="测试横排", confidence=0.9, bbox=BBox(100, 100, 200, 30))
    boxes = [_estimate_char_bbox(line, i, 4) for i in range(4)]
    assert boxes[0].x == 100 and boxes[1].x == 150
    assert boxes[2].x == 200 and boxes[3].x == 250
    for b in boxes:
        assert b.y == 100 and b.h == 30
    print("test_char_index_horizontal_split PASSED")


def test_char_index_dedup_on_rebuild():
    """二次 build 不应使 entries 累积。"""
    from app.services.char_index_service import CharIndexService
    from app.models import Line, BBox, Block, BlockType, Page
    page = Page(image_path="/tmp/p1.png", width=400, height=600, page_number=1)
    blk = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 400, 400), order=0)
    blk.lines = [
        Line(text="永和九年", confidence=0.9, bbox=BBox(50, 50, 40, 160)),
        Line(text="永字八法", confidence=0.9, bbox=BBox(100, 50, 40, 160)),
    ]
    page.blocks = [blk]
    svc = CharIndexService(include_fallback=True)
    svc.build([page])
    n1 = len(svc.query("永"))
    svc.build([page])
    n2 = len(svc.query("永"))
    assert n1 == n2 == 2, f"expected 2 entries each build, got {n1}/{n2}"
    print("test_char_index_dedup_on_rebuild PASSED")


def test_char_index_build_does_not_rewrite_line_bbox_or_chars(tmp_path):
    import cv2
    import numpy as np

    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.char_index_service import CharIndexService

    img = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 62), (160, 76), (0, 0, 0), -1)
    img_path = str(tmp_path / "page.png")
    cv2.imwrite(img_path, img)

    line = Line(text="天地", confidence=0.9, bbox=BBox(10, 44, 180, 46))
    page = Page(
        image_path=img_path,
        width=220,
        height=120,
        page_number=1,
        blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 200, 100), lines=[line])],
    )

    before_bbox = line.bbox
    before_chars = list(line.chars)

    svc = CharIndexService(include_fallback=True).build([page])

    assert line.bbox == before_bbox
    assert line.chars == before_chars
    assert svc.query("天")
    assert svc.query("地")

    print("test_char_index_build_does_not_rewrite_line_bbox_or_chars PASSED")


def test_char_index_uses_display_text_for_existing_positional_boxes():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="甲乙",
        confidence=0.9,
        bbox=BBox(10, 20, 60, 24),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(10, 20, 20, 24), bbox_source="hanwang:CharRcg", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(40, 20, 20, 24), bbox_source="hanwang:CharRcg", bbox_granularity="char"),
        ],
    )
    set_line_proof_text(line, "甲丙")
    page = Page(
        image_path="/tmp/display-text-index.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 100, 60), lines=[line])],
    )

    svc = CharIndexService(include_fallback=True).build_index(OcrProject(name="display-index", pages=[page]))

    assert svc.query("乙") == []
    entries = svc.query("丙")
    assert len(entries) == 1
    assert entries[0].char_idx == 1
    assert entries[0].bbox == BBox(40, 20, 20, 24)

    print("test_char_index_uses_display_text_for_existing_positional_boxes PASSED")


def test_char_index_skips_grossly_mismatched_geometry_instead_of_showing_wrong_crops():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="错错错",
        confidence=0.9,
        bbox=BBox(0, 10, 90, 30),
        chars=[
            Char(char="使", confidence=0.9, bbox=BBox(300, 10, 20, 30), bbox_source="fallback", bbox_granularity="fallback"),
            Char(char="接", confidence=0.9, bbox=BBox(330, 10, 20, 30), bbox_source="fallback", bbox_granularity="fallback"),
            Char(char="有", confidence=0.9, bbox=BBox(360, 10, 20, 30), bbox_source="fallback", bbox_granularity="fallback"),
        ],
    )
    set_line_proof_text(line, "甲二丙")
    page = Page(
        image_path="/tmp/gross-mismatch-index.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 100, 60), lines=[line])],
    )

    svc = CharIndexService(include_fallback=True).build_index(OcrProject(name="gross-mismatch", pages=[page]))

    assert svc.query("甲") == []
    assert svc.query("二") == []
    assert svc.query("丙") == []
    assert [char.char for char in line.chars] == ["使", "接", "有"]

    print("test_char_index_skips_grossly_mismatched_geometry_instead_of_showing_wrong_crops PASSED")


def test_char_index_skips_two_char_full_mismatch_geometry():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="使产",
        confidence=0.9,
        bbox=BBox(0, 10, 60, 30),
        chars=[
            Char(char="使", confidence=0.9, bbox=BBox(300, 10, 20, 30), bbox_source="hanwang:micro_recblock", bbox_granularity="char"),
            Char(char="产", confidence=0.9, bbox=BBox(330, 10, 20, 30), bbox_source="hanwang:micro_recblock", bbox_granularity="char"),
        ],
    )
    set_line_proof_text(line, "二三")
    page = Page(
        image_path="/tmp/two-char-full-mismatch-index.png",
        width=120,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 100, 60), lines=[line])],
    )

    svc = CharIndexService(include_fallback=True).build_index(OcrProject(name="short-mismatch", pages=[page]))

    assert svc.query("二") == []
    assert svc.query("三") == []
    assert [char.char for char in line.chars] == ["使", "产"]

    print("test_char_index_skips_two_char_full_mismatch_geometry PASSED")


def test_char_index_skips_existing_chars_when_display_length_changed():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    line = Line(
        text="旧旧旧",
        confidence=0.9,
        bbox=BBox(0, 10, 120, 30),
        chars=[
            Char(char="旧", confidence=0.9, bbox=BBox(300, 10, 20, 30), bbox_source="hanwang:micro_recblock", bbox_granularity="char"),
            Char(char="旧", confidence=0.9, bbox=BBox(330, 10, 20, 30), bbox_source="hanwang:micro_recblock", bbox_granularity="char"),
            Char(char="旧", confidence=0.9, bbox=BBox(360, 10, 20, 30), bbox_source="hanwang:micro_recblock", bbox_granularity="char"),
        ],
    )
    set_line_proof_text(line, "甲二丙丁")
    page = Page(
        image_path="/tmp/length-mismatch-index.png",
        width=160,
        height=80,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 150, 60), lines=[line])],
    )

    svc = CharIndexService(include_fallback=True).build_index(OcrProject(name="length-mismatch", pages=[page]))

    assert svc.query("二") == []
    assert [char.char for char in line.chars] == ["旧", "旧", "旧"]

    print("test_char_index_skips_existing_chars_when_display_length_changed PASSED")


def test_char_index_sort_categories():
    """sorted_chars 排序：拼音/字母 → 数字 → 标点 → 符号。"""
    from app.services.char_index_service import CharIndexService, _sort_key
    # 直接验证 _sort_key 的种类排序，与外部依赖 pypinyin 是否可用解耦
    samples = ["永", "A", "z", "1", "3", ",", "。", "=", "π"]
    sorted_samples = sorted(samples, key=_sort_key)
    # 字母/CJK 优先（kind=0），数字（kind=1），标点（kind=2），符号（kind=3）
    kinds = []
    for c in sorted_samples:
        kinds.append(_sort_key(c)[0])
    # 必须严格非递减
    assert kinds == sorted(kinds), f"kinds not sorted: {kinds}"
    # 数字必在标点之前
    assert sorted_samples.index("1") < sorted_samples.index(",")
    assert sorted_samples.index("3") < sorted_samples.index("。")
    # 标点必在符号之前
    assert sorted_samples.index(",") < sorted_samples.index("=")
    print("test_char_index_sort_categories PASSED")


def test_char_index_query_stable_order():
    """query 返回的 entries 按 (page_number, line.y, line.x, char_idx) 稳定排序。"""
    from app.services.char_index_service import CharIndexService
    from app.models import Line, BBox, Block, BlockType, Page
    p1 = Page(image_path="/tmp/p1.png", width=400, height=600, page_number=2)
    b1 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 400, 400), order=0)
    b1.lines = [
        Line(text="永明", confidence=0.9, bbox=BBox(100, 200, 40, 80)),
        Line(text="永和", confidence=0.9, bbox=BBox(100, 50, 40, 80)),  # y 更小
    ]
    p1.blocks = [b1]
    p0 = Page(image_path="/tmp/p0.png", width=400, height=600, page_number=1)
    b0 = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 400, 400), order=0)
    b0.lines = [Line(text="永远", confidence=0.9, bbox=BBox(50, 100, 40, 80))]
    p0.blocks = [b0]
    svc = CharIndexService(include_fallback=True)
    svc.build([p1, p0])
    entries = svc.query("永")
    assert len(entries) == 3
    # 排序键：(page_number, line.y, line.x, char_idx)
    # 期望顺序：page1(永远) → page2/y=50(永和) → page2/y=200(永明)
    assert entries[0].page_number == 1
    assert entries[1].page_number == 2 and entries[1].line.bbox.y == 50
    assert entries[2].page_number == 2 and entries[2].line.bbox.y == 200
    print("test_char_index_query_stable_order PASSED")


def test_char_index_skips_whitespace():
    """空白与空字符不进入索引。"""
    from app.services.char_index_service import CharIndexService
    from app.models import Line, BBox, Block, BlockType, Page
    page = Page(image_path="/tmp/p1.png", width=400, height=600, page_number=1)
    blk = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 400, 400), order=0)
    blk.lines = [Line(text="永 和\t九", confidence=0.9, bbox=BBox(50, 50, 40, 200))]
    page.blocks = [blk]
    svc = CharIndexService(include_fallback=True)
    svc.build([page])
    assert " " not in svc._index and "\t" not in svc._index
    assert svc.unique_chars() == 3  # 永和九
    print("test_char_index_skips_whitespace PASSED")


# =====================================================================
# 入口
# =====================================================================

# =====================================================================

def test_api_ocr_engine_resolves_ocr_endpoint_for_pp_ocrv5_profile():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine

    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"ocrResults": []}}

    def fake_post(url, json, headers, timeout, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        captured["proxies"] = kwargs.get("proxies")
        return DummyResponse()

    original_post = requests.post
    requests.post = fake_post
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(
        mode="api",
        api_model_profile="pp-ocrv5",
        api_url="https://example.com/root",
        api_timeout=12,
    )
    try:
        ApiOcrEngine().recognize(np.zeros((20, 30, 3), dtype=np.uint8), OcrContext())
        assert captured["url"] == "https://example.com/root/ocr"
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_resolves_ocr_endpoint_for_pp_ocrv5_profile PASSED")


def test_api_ocr_engine_parses_paddle_coordinate_variants():
    from app.engines.real_ocr_adapter import ApiOcrEngine

    engine = ApiOcrEngine()
    dict_bbox = engine._bbox_from_region({"coordinate": [10, 20, 50, 60]})
    assert dict_bbox.to_dict() == {"x": 10, "y": 20, "w": 40, "h": 40}

    poly_bbox = engine._bbox_from_region([10, 20, 50, 20, 50, 60, 10, 60])
    assert poly_bbox.to_dict() == {"x": 10, "y": 20, "w": 40, "h": 40}

    nested_poly_bbox = engine._bbox_from_region([[10, 20], [50, 20], [50, 60], [10, 60]])
    assert nested_poly_bbox.to_dict() == {"x": 10, "y": 20, "w": 40, "h": 40}

    print("test_api_ocr_engine_parses_paddle_coordinate_variants PASSED")


def test_api_request_builders_split_profile_params():
    from app.engines.real_ocr_adapter import ApiOcrEngine

    ocr_body = ApiOcrEngine()._build_request_body(
        "abc",
        profile="pp-ocrv5",
        endpoint_url="https://example.com/ocr",
    )
    assert ocr_body["file"] == "abc"
    assert "returnWordBox" not in ocr_body
    assert ocr_body["textDetLimitType"] == "max"
    assert ocr_body["useDocUnwarping"] is False

    vl_body = ApiOcrEngine()._build_request_body(
        "abc",
        profile="paddleocr-vl-1.6",
        endpoint_url="https://example.com/api/v2/ocr/jobs",
    )
    assert vl_body["file"] == "abc"
    assert vl_body["fileType"] == 1
    assert vl_body["useDocUnwarping"] is False
    assert vl_body["useDocOrientationClassify"] is False
    assert "returnWordBox" not in vl_body
    assert "textDetLimitType" not in vl_body

    print("test_api_request_builders_split_profile_params PASSED")


def test_layout_analyzer_uses_datainfo_canvas_scale():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    data = {
        "result": {
            "dataInfo": {"width": 500, "height": 1000},
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "text",
                                    "coordinate": [10, 20, 110, 70],
                                },
                            ],
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_bbox": [10, 20, 110, 70],
                                "block_content": "缩放测试",
                            },
                        ],
                    },
                },
            ],
        },
    }

    blocks, _overlays = analyzer._extract_api_blocks(page, data)

    assert len(blocks) == 1
    assert blocks[0].bbox == BBox(20, 40, 200, 100)
    assert _raw_layout_records(page)[0]["block_bbox"] == [20, 40, 220, 140]

    print("test_layout_analyzer_uses_datainfo_canvas_scale PASSED")


def test_layout_analyzer_ignores_conflicting_datainfo_when_bbox_is_page_space():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    data = {
        "result": {
            "dataInfo": {"width": 500, "height": 1000},
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "text",
                                    "coordinate": [800, 1500, 900, 1600],
                                },
                            ],
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_bbox": [800, 1500, 900, 1600],
                                "block_content": "原图坐标",
                            },
                        ],
                    },
                },
            ],
        },
    }

    blocks, _overlays = analyzer._extract_api_blocks(page, data)

    assert len(blocks) == 1
    assert blocks[0].bbox == BBox(800, 1500, 100, 100)

    print("test_layout_analyzer_ignores_conflicting_datainfo_when_bbox_is_page_space PASSED")


def test_layout_analyzer_ignores_conflicting_pruned_shape_when_bbox_is_page_space():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "input_img_shape": [1000, 500],
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "text",
                                    "coordinate": [800, 1500, 900, 1600],
                                },
                            ],
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_bbox": [800, 1500, 900, 1600],
                                "block_content": "原图坐标",
                            },
                        ],
                    },
                },
            ],
        },
    }

    blocks, _overlays = analyzer._extract_api_blocks(page, data)

    assert len(blocks) == 1
    assert blocks[0].bbox == BBox(800, 1500, 100, 100)

    print("test_layout_analyzer_ignores_conflicting_pruned_shape_when_bbox_is_page_space PASSED")


def test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected():
    import hashlib
    import json
    import tempfile

    import cv2
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page

    captured = {}

    class SubmitResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"jobId": "job-1"}}

    class PollResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "state": "done",
                    "resultUrl": {"jsonUrl": "https://result.example.com/page.jsonl"},
                },
            }

    class JsonlResponse:
        text = json.dumps(
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "prunedResult": {
                                "parsing_res_list": [
                                    {
                                        "block_label": "text",
                                        "block_bbox": [10, 20, 110, 50],
                                        "block_content": "OCR行",
                                    },
                                ],
                            },
                        },
                    ],
                },
            },
            ensure_ascii=False,
        )

        def raise_for_status(self):
            return None

    def fake_post(url, data, files, headers, timeout, **kwargs):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["proxies"] = kwargs.get("proxies")
        return SubmitResponse()

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured.setdefault("gets", []).append({
            "url": url,
            "headers": headers or {},
            "timeout": timeout,
            "proxies": kwargs.get("proxies"),
        })
        if url.endswith("/job-1"):
            return PollResponse()
        if url == "https://result.example.com/page.jsonl":
            return JsonlResponse()
        raise AssertionError(f"unexpected GET URL: {url}")

    original_post = requests.post
    original_get = requests.get
    requests.post = fake_post
    requests.get = fake_get
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(
        mode="api",
        api_model_profile="pp-ocrv5",
        api_url="https://example.com/root",
        api_token="demo",
        api_timeout=12,
        paddle_api_network_mode="direct",
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        cv2.imwrite(page_path, np.full((120, 200, 3), 255, dtype=np.uint8))
        original_bytes = Path(page_path).read_bytes()
        page = Page(image_path=page_path, width=200, height=120)
        LayoutAnalyzer()._api_analyze(page)
        assert captured["url"] == "https://example.com/root/api/v2/ocr/jobs"
        assert captured["timeout"] == 12
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
        assert captured["headers"]["Authorization"] == "bearer demo"
        assert captured["data"]["model"] == "PaddleOCR-VL-1.6"
        optional_payload = json.loads(captured["data"]["optionalPayload"])
        assert optional_payload["useDocOrientationClassify"] is False
        assert optional_payload["useDocUnwarping"] is False
        assert optional_payload["useChartRecognition"] is False
        file_obj = captured["files"]["file"]
        assert file_obj.name == Path(page_path).name
        assert file_obj.getvalue() == original_bytes
        assert captured["gets"][0]["url"] == "https://example.com/root/api/v2/ocr/jobs/job-1"
        assert captured["gets"][0]["headers"]["Authorization"] == "bearer demo"
        assert captured["gets"][0]["proxies"] == {"http": None, "https": None, "all": None}
        assert len(page.blocks) == 1
        assert page.blocks[0].block_type == BlockType.TEXT
        assert "OCR行" in page.blocks[0].note
        artifact = page.raw_layout_artifact
        assert artifact is not None
        assert artifact.engine == "paddleocr-vl"
        assert artifact.engine_version == "1.6"
        assert artifact.run_id == "job-1"
        artifact_path = Path(artifact.artifact_path)
        assert artifact_path.name == "job-1.json"
        assert artifact_path.parent.name == page.uid
        assert artifact_path.parent.parent.name == "paddle_artifacts"
        artifact_blob = artifact_path.read_bytes()
        assert artifact.artifact_hash == hashlib.sha256(artifact_blob).hexdigest()
        artifact_json = json.loads(artifact_blob.decode("utf-8"))
        assert artifact_json["page"]["uid"] == page.uid
        assert artifact_json["response"]["paddle_v16"]["jobId"] == "job-1"
        assert not Path(page_path).with_suffix(".layout-api.json").exists()
        assert not Path(page_path).with_suffix(".layout-api-raw.png").exists()
        assert not Path(page_path).with_suffix(".layout-app-overlay.png").exists()
    finally:
        requests.post = original_post
        requests.get = original_get
        cfg.reset_to_defaults()
        os.unlink(page_path)

    print("test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected PASSED")


def test_paddle_v16_submit_error_includes_response_body():
    import requests

    from app.core.paddle_v16_client import PaddleV16LayoutClient

    class BadResponse:
        status_code = 400
        text = '{"errorMsg":"invalid multipart"}'

        def raise_for_status(self):
            raise requests.HTTPError("400 Client Error")

    def fake_post(url, data, files, headers, timeout, **kwargs):
        return BadResponse()

    original_post = requests.post
    requests.post = fake_post
    try:
        client = PaddleV16LayoutClient(jobs_url="https://example.com/api/v2/ocr/jobs")
        try:
            client.submit_image_bytes(b"\x89PNG\r\n\x1a\nfake")
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")

        assert "HTTP 400" in message
        assert "invalid multipart" in message
    finally:
        requests.post = original_post

    print("test_paddle_v16_submit_error_includes_response_body PASSED")


def test_paddle_v16_client_supports_env_proxy_and_batch_id():
    import requests

    from app.core.paddle_v16_client import PaddleV16LayoutClient

    captured = {}

    class SubmitResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"jobId": "job-1"}}

    class BatchResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"batchId": "batch-1", "jobs": []}}

    def fake_post(url, data, files, headers, timeout, **kwargs):
        captured["post_url"] = url
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["post_proxies"] = kwargs.get("proxies")
        return SubmitResponse()

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["get_url"] = url
        captured["get_headers"] = headers or {}
        captured["get_timeout"] = timeout
        captured["get_proxies"] = kwargs.get("proxies")
        return BatchResponse()

    original_post = requests.post
    original_get = requests.get
    requests.post = fake_post
    requests.get = fake_get
    try:
        client = PaddleV16LayoutClient(
            jobs_url="https://example.com/api/v2/ocr/jobs",
            token="demo",
            network_mode="env_proxy",
        )
        job_id = client.submit_image_bytes(b"\x89PNG\r\n\x1a\nfake", batch_id="batch-1")
        assert job_id == "job-1"
        assert captured["data"]["batchId"] == "batch-1"
        assert captured["post_proxies"] is None
        assert captured["headers"]["Authorization"] == "bearer demo"
        assert client.telemetry["submit_network_mode"] == "env_proxy"

        body = client.get_batch_status("batch-1")
        assert body["data"]["batchId"] == "batch-1"
        assert captured["get_url"] == "https://example.com/api/v2/ocr/jobs/batch/batch-1"
        assert captured["get_proxies"] is None
    finally:
        requests.post = original_post
        requests.get = original_get

    print("test_paddle_v16_client_supports_env_proxy_and_batch_id PASSED")


def test_paddle_v16_client_can_cancel_between_polls():
    import requests

    from app.core.paddle_v16_client import PaddleV16LayoutClient, PaddleV16RequestCancelled

    cancelled = False
    calls = []

    class PollingResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"state": "running"}}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        calls.append((url, timeout))
        return PollingResponse()

    def fake_sleep(_seconds):
        nonlocal cancelled
        cancelled = True

    original_get = requests.get
    requests.get = fake_get
    try:
        client = PaddleV16LayoutClient(
            jobs_url="https://example.com/api/v2/ocr/jobs",
            request_timeout=1,
            poll_timeout=5,
            poll_interval_s=0.1,
            sleep=fake_sleep,
            cancel_callback=lambda: cancelled,
        )
        try:
            client.wait_for_result_json_url("job-1")
        except PaddleV16RequestCancelled:
            pass
        else:
            raise AssertionError("expected PaddleV16RequestCancelled")

        assert len(calls) == 1
    finally:
        requests.get = original_get

    print("test_paddle_v16_client_can_cancel_between_polls PASSED")


def test_paddle_v16_client_reports_submit_wait_download_parse_stages():
    import json
    import requests

    from app.core.paddle_v16_client import PaddleV16LayoutClient

    statuses = []

    class SubmitResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"jobId": "job-1"}}

    class PollDoneResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"state": "done", "resultUrl": {"jsonUrl": "https://example.com/result.jsonl"}}}

    class JsonlResponse:
        status_code = 200
        text = json.dumps({"result": {"layoutParsingResults": []}}, ensure_ascii=False)

        def raise_for_status(self):
            return None

    def fake_post(url, data, files, headers, timeout, **kwargs):
        return SubmitResponse()

    def fake_get(url, headers=None, timeout=None, **kwargs):
        if url.endswith("/job-1"):
            return PollDoneResponse()
        if url == "https://example.com/result.jsonl":
            return JsonlResponse()
        raise AssertionError(f"unexpected GET {url}")

    original_post = requests.post
    original_get = requests.get
    requests.post = fake_post
    requests.get = fake_get
    try:
        client = PaddleV16LayoutClient(
            jobs_url="https://example.com/api/v2/ocr/jobs",
            request_timeout=1,
            poll_timeout=5,
            status_callback=statuses.append,
        )
        data = client.analyze_image_bytes(b"\x89PNG\r\n\x1a\nfake")

        assert data["errorCode"] == 0
        assert statuses[:2] == ["提交请求", "等待服务端"]
        assert "下载结果" in statuses
        assert statuses[-1] == "解析结果"
    finally:
        requests.post = original_post
        requests.get = original_get

    print("test_paddle_v16_client_reports_submit_wait_download_parse_stages PASSED")


def test_paddle_v16_auto_network_mode_tries_direct_before_env_proxy():
    import os

    from app.core.paddle_v16_client import PaddleV16LayoutClient

    old_proxy = os.environ.get("HTTPS_PROXY")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8888"
    try:
        client = PaddleV16LayoutClient(network_mode="auto")
        assert client._network_attempts() == [False, True]
    finally:
        if old_proxy is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = old_proxy

    print("test_paddle_v16_auto_network_mode_tries_direct_before_env_proxy PASSED")


def test_layout_analyzer_routes_hanwang_mode_to_ppvl_layout():
    import app.core.app_config as config_module
    import app.core.layout_analyzer as layout_module
    from app.models import BBox, Block, BlockType, Page

    def fake_api_analyze(self, page):
        page.width = 300
        page.height = 200
        _attach_raw_layout_records(page, [
            {"block_label": "text", "block_bbox": [12, 18, 92, 58], "block_content": "PPVL"}
        ])
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(12, 18, 80, 40), order=0)]
        return page

    original_get_config = config_module.get_config
    original_api_analyze = layout_module.LayoutAnalyzer._api_analyze
    config_module.get_config = lambda: {"mode": "hanwang"}
    layout_module.LayoutAnalyzer._api_analyze = fake_api_analyze

    try:
        page = Page(image_path="/tmp/hanwang-hybrid-layout.png", width=0, height=0, page_number=1)
        result = layout_module.LayoutAnalyzer().analyze(page)

        assert len(result.blocks) == 1
        assert result.blocks[0].bbox == BBox(12, 18, 80, 40)
        assert result.width == 300
        assert result.height == 200
        assert _raw_layout_records(result)[0]["block_content"] == "PPVL"
    finally:
        config_module.get_config = original_get_config
        layout_module.LayoutAnalyzer._api_analyze = original_api_analyze


def test_hanwang_assets_env_accepts_bin_dir():
    from app.engines.hanwang.paths import get_hanwang_bin_dir, verify_hanwang_assets

    required = [
        "linecut_segimg_probe.exe",
        "linecut_recogimg_probe.exe",
        "docseg_probe.exe",
        "linecut.dll",
        "IntegratRcg.dll",
        "doc_seg.dll",
        "mp30.dll",
        "mp60.dll",
    ]

    old_env = os.environ.get("HANWANG_NATIVE_DIR")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "hanwang_native"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True)
        for name in required:
            (bin_dir / name).write_bytes(b"stub")
        os.environ["HANWANG_NATIVE_DIR"] = str(bin_dir)
        try:
            assert get_hanwang_bin_dir() == bin_dir
            verify_hanwang_assets()
        finally:
            if old_env is None:
                os.environ.pop("HANWANG_NATIVE_DIR", None)
            else:
                os.environ["HANWANG_NATIVE_DIR"] = old_env

    print("test_hanwang_assets_env_accepts_bin_dir PASSED")


def test_hanwang_native_bridge_writes_multi_recblocks():
    import numpy as np
    from app.engines.hanwang import native_bridge

    captured = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_dir = Path(tmpdir)
        (bin_dir / "linecut_recogimg_probe.exe").write_bytes(b"stub")

        def fake_get_bin_dir():
            return bin_dir

        def fake_save_temp_image(image_bgr, work_dir):
            path = work_dir / "_fake.png"
            path.write_bytes(b"png")
            return path

        def resolve_probe_arg(raw: str, cwd: Path) -> Path:
            text = str(raw)
            if len(text) >= 3 and text[1:3] == ":\\":
                drive = text[0].lower()
                return Path(f"/mnt/{drive}") / text[3:].replace("\\", "/")
            path = Path(text)
            if path.is_absolute():
                return path
            return cwd / path

        def fake_run_exe(exe, args, *, cwd, timeout):
            rb_path = resolve_probe_arg(args[2], Path(cwd))
            captured["rb_text"] = rb_path.read_text(encoding="utf-8")
            captured["args"] = list(args)
            resolve_probe_arg(args[1], Path(cwd)).write_text('{"lines":[]}', encoding="utf-8")
            return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

        original_get_bin_dir = native_bridge.get_hanwang_bin_dir
        original_save = native_bridge._save_temp_image
        original_run = native_bridge._run_exe
        native_bridge.get_hanwang_bin_dir = fake_get_bin_dir
        native_bridge._save_temp_image = fake_save_temp_image
        native_bridge._run_exe = fake_run_exe
        try:
            out = native_bridge.run_linecut_recog(
                np.zeros((20, 30, 3), dtype=np.uint8),
                recblocks_xyxy=[(1, 2, 3, 4), (5, 6, 7, 8)],
                with_charrcg=True,
            )
        finally:
            native_bridge.get_hanwang_bin_dir = original_get_bin_dir
            native_bridge._save_temp_image = original_save
            native_bridge._run_exe = original_run

    assert out == {"lines": []}
    assert captured["rb_text"] == "1\t2\t3\t4\n5\t6\t7\t8\n"
    assert captured["args"][6] == "via-seg"
    assert captured["args"][7] == "with-charrcg"

    print("test_hanwang_native_bridge_writes_multi_recblocks PASSED")


def test_layout_worker_continues_after_single_page_failure():
    from unittest.mock import patch

    from app.core.layout_analyzer import LayoutAnalyzer, LayoutWorker
    from app.models import BBox, Block, BlockType, Page

    pages = [
        Page(image_path="/tmp/layout-ok-1.png", width=100, height=100, page_number=1),
        Page(image_path="/tmp/layout-bad.png", width=100, height=100, page_number=2),
        Page(image_path="/tmp/layout-ok-2.png", width=100, height=100, page_number=3),
    ]
    done = []
    emitted_pages = []
    errors = []

    def fake_analyze(_self, page):
        if page.page_number == 2:
            raise RuntimeError("boom")
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(1, 2, 30, 40))]

    with patch.object(LayoutAnalyzer, "analyze", fake_analyze):
        worker = LayoutWorker(pages)
        worker.page_done.connect(lambda idx, total: done.append((idx, total)))
        worker.all_done.connect(lambda result: emitted_pages.append(result))
        worker.error.connect(lambda message: errors.append(message))
        worker.run()

    assert done == [(0, 3), (1, 3), (2, 3)]
    assert errors == []
    assert emitted_pages == [pages]
    assert len(pages[0].blocks) == 1
    assert pages[1].blocks == []
    assert pages[1].error_message.startswith("版面分析失败：boom")
    assert len(pages[2].blocks) == 1

    print("test_layout_worker_continues_after_single_page_failure PASSED")


def test_layout_worker_cancelled_does_not_emit_all_done():
    from app.core.layout_analyzer import LayoutWorker
    from app.models import Page

    pages = [
        Page(image_path="/tmp/layout-cancel.png", width=100, height=100, page_number=1),
    ]
    emitted_pages = []
    cancelled = []

    worker = LayoutWorker(pages)
    worker.all_done.connect(lambda result: emitted_pages.append(result))
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.cancel()
    worker.run()

    assert cancelled == [True]
    assert emitted_pages == []
    assert pages[0].blocks == []

    print("test_layout_worker_cancelled_does_not_emit_all_done PASSED")


def test_layout_worker_runs_api_pages_with_bounded_concurrency():
    import threading
    import time
    from unittest.mock import patch

    from app.core.app_config import AppConfig, update_config
    from app.core.layout_analyzer import LayoutAnalyzer, LayoutWorker
    from app.models import BBox, Block, BlockType, Page

    pages = [
        Page(image_path=f"/tmp/layout-api-{idx}.png", width=100, height=100, page_number=idx)
        for idx in range(1, 5)
    ]
    done = []
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_analyze(_self, page):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(1, 2, 30, 40))]
        finally:
            with lock:
                active -= 1

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(
        mode="api",
        api_url="https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        layout_concurrency=2,
    )
    try:
        with patch.object(LayoutAnalyzer, "analyze", fake_analyze):
            worker = LayoutWorker(pages)
            worker.page_done.connect(lambda idx, total: done.append((idx, total)))
            worker.run()
    finally:
        cfg.reset_to_defaults()

    assert done == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert max_active == 2
    assert all(len(page.blocks) == 1 for page in pages)

    print("test_layout_worker_runs_api_pages_with_bounded_concurrency PASSED")


def test_layout_worker_keeps_local_pages_serial_even_with_concurrency_config():
    import threading
    import time
    from unittest.mock import patch

    from app.core.app_config import AppConfig, update_config
    from app.core.layout_analyzer import LayoutAnalyzer, LayoutWorker
    from app.models import BBox, Block, BlockType, Page

    pages = [
        Page(image_path=f"/tmp/layout-local-{idx}.png", width=100, height=100, page_number=idx)
        for idx in range(1, 5)
    ]
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_analyze(_self, page):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(1, 2, 30, 40))]
        finally:
            with lock:
                active -= 1

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(mode="local", layout_concurrency=4)
    try:
        with patch.object(LayoutAnalyzer, "analyze", fake_analyze):
            worker = LayoutWorker(pages)
            worker.run()
    finally:
        cfg.reset_to_defaults()

    assert max_active == 1
    assert all(len(page.blocks) == 1 for page in pages)

    print("test_layout_worker_keeps_local_pages_serial_even_with_concurrency_config PASSED")


def test_workflow_controller_marks_partial_layout_failures_without_blocking_success_pages():
    from app.controllers.workflow_controller import WorkflowController
    from app.models import BBox, Block, BlockType, OcrProject, Page, PageStatus

    success_page = Page(image_path="/tmp/layout-success.png", width=100, height=100, page_number=1)
    success_page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(1, 2, 30, 40))]
    failed_page = Page(image_path="/tmp/layout-failed.png", width=100, height=100, page_number=2)
    failed_page.error_message = "版面分析失败：boom"
    pages = [success_page, failed_page]

    controller = WorkflowController()
    controller._project = OcrProject(name="partial-layout", pages=[])
    controller._auto_start_ocr_after_layout = True
    started = []
    messages = []
    controller.start_ocr = lambda result_pages, notify_page_callback=None: started.append(result_pages)
    controller.status_message.connect(messages.append)

    controller.on_layout_done(pages)

    assert success_page.status == PageStatus.LAYOUT_DONE
    assert failed_page.status == PageStatus.ERROR
    assert controller.project.pages == pages
    assert started == [pages]
    assert any("1/2 页成功" in message and "1 页失败" in message for message in messages)

    print("test_workflow_controller_marks_partial_layout_failures_without_blocking_success_pages PASSED")


def test_workflow_controller_enables_proof_steps_after_first_ocr_page():
    from app.controllers.workflow_controller import STEP_OCR, STEP_VPROOF, WorkflowController
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
    from app.services.ocr_run_result import OcrProgress

    page1 = Page(image_path="/tmp/ocr-page-1.png", width=100, height=100, page_number=1)
    page1.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 0, 80, 30),
            lines=[Line(text="第一页", confidence=0.9, bbox=BBox(1, 2, 60, 20))],
        )
    ]
    page2 = Page(image_path="/tmp/ocr-page-2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 30))]

    controller = WorkflowController()
    controller._project = OcrProject(name="partial-ocr", pages=[page1, page2])
    controller._max_step = STEP_OCR
    enabled = []
    controller.step_enabled_changed.connect(enabled.append)

    controller._on_ocr_progress(OcrProgress(total_pages=2, completed_pages=1))

    assert enabled[-1] == STEP_VPROOF
    assert controller.can_enter_step(STEP_VPROOF)
    assert page1.status == PageStatus.OCR_DONE

    print("test_workflow_controller_enables_proof_steps_after_first_ocr_page PASSED")


def test_proof_line_iterator_excludes_equation_lines():
    from app.core.proof_line_utils import iter_unique_page_text_lines
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(image_path="/tmp/proof-lines.png", width=100, height=100)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
            Line(text="正文", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
        ]),
        Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(0, 20, 80, 20), lines=[
            Line(text="图注", confidence=0.9, bbox=BBox(1, 21, 20, 10)),
        ]),
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 40, 80, 20), lines=[
            Line(text="E=mc2", confidence=0.9, bbox=BBox(1, 41, 30, 10)),
        ]),
        Block(block_type=BlockType.FIGURE, bbox=BBox(0, 60, 80, 20), lines=[
            Line(text="图片不校", confidence=0.9, bbox=BBox(1, 61, 30, 10)),
        ]),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_text_lines(page)]

    assert texts == ["正文", "图注"]

    print("test_proof_line_iterator_excludes_equation_lines PASSED")


def test_proof_line_iterator_trusts_layout_snapshot_type_over_runtime_projection():
    from app.core.proof_line_utils import iter_unique_page_hproof_lines, iter_unique_page_text_lines
    from app.models import (
        BBox, Block, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot,
        Line, OcrPolicy, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page

    runtime_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 20),
        lines=[Line(text="旧投影公式文本", confidence=0.9, bbox=BBox(1, 1, 30, 10))],
        order=0,
        source_label="text",
    )
    page = Page(image_path="/tmp/proof-lines-snapshot.png", width=100, height=100)
    page.blocks = [runtime_block]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-proof-lines",
        source_engine="test",
        source_run_id="run-proof-lines",
        blocks=(
            LayoutBlockSnapshot(
                uid=runtime_block.uid,
                block_type=BlockType.EQUATION,
                bbox=BBox(0, 0, 80, 20),
                order=0,
                source_label="formula",
                origin=BlockOrigin(source_label="formula"),
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
            ),
        ),
    ))

    assert list(iter_unique_page_text_lines(page)) == []
    assert list(iter_unique_page_hproof_lines(page)) == []

    print("test_proof_line_iterator_trusts_layout_snapshot_type_over_runtime_projection PASSED")


def test_char_index_service_does_not_repopulate_equation_chars():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.char_index_service import CharIndexService

    eq_line = Line(text="$  \\frac{1}{2}  $", confidence=0.9, bbox=BBox(1, 41, 60, 10), chars=[])
    page = Page(image_path="/tmp/equation-index.png", width=100, height=100)
    page.blocks = [
        Block(
            block_type=BlockType.EQUATION,
            bbox=BBox(0, 40, 80, 20),
            lines=[eq_line],
            order=0,
            source_label="formula",
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(0, 0, 80, 20),
            lines=[Line(text="正文", confidence=0.9, bbox=BBox(1, 1, 20, 10))],
            order=1,
        ),
    ]

    svc = CharIndexService(include_non_cjk=True).build_index(OcrProject(name="equation-skip", pages=[page]))

    assert eq_line.chars == []
    assert svc.query("$") == []

    print("test_char_index_service_does_not_repopulate_equation_chars PASSED")


def test_hproof_line_iterator_uses_shared_proof_text_elements():
    from app.core.proof_line_utils import iter_unique_page_hproof_lines
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(image_path="/tmp/hproof-lines.png", width=100, height=100)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
            Line(text="正文", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
        ]),
        Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(0, 20, 80, 20), lines=[
            Line(text="图注", confidence=0.9, bbox=BBox(1, 21, 20, 10)),
        ]),
        Block(block_type=BlockType.TABLE_CAPTION, bbox=BBox(0, 40, 80, 20), lines=[
            Line(text="表注", confidence=0.9, bbox=BBox(1, 41, 20, 10)),
        ]),
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 60, 80, 20), lines=[
            Line(text="E=mc2", confidence=0.9, bbox=BBox(1, 61, 30, 10)),
        ]),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)]

    assert texts == ["正文", "图注", "表注"]

    print("test_hproof_line_iterator_uses_shared_proof_text_elements PASSED")


def test_proof_line_iterators_exclude_route_table_lines():
    from app.core.proof_line_utils import iter_unique_page_hproof_lines, iter_unique_page_text_lines
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(image_path="/tmp/proof-route-table.png", width=100, height=100)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 40), lines=[
            Line(text="正文", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
            Line(
                text="表格子区",
                confidence=0.0,
                bbox=BBox(1, 20, 30, 10),
                review_flags=["hanwang_route_table"],
            ),
        ]),
    ]

    assert [line.text for _block, line, _idx in iter_unique_page_text_lines(page)] == ["正文"]
    assert [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)] == ["正文"]

    print("test_proof_line_iterators_exclude_route_table_lines PASSED")


def test_hproof_line_iterator_excludes_position_source_labels():
    from app.core.proof_line_utils import iter_unique_page_hproof_lines
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(image_path="/tmp/hproof-position.png", width=100, height=100)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 1, 20, 10),
            lines=[Line(text="12", confidence=0.9, bbox=BBox(1, 1, 20, 10))],
            note="score=0.99 | source_label=text",
            source_label="page_number",
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 20, 60, 12),
            lines=[Line(text="正文", confidence=0.9, bbox=BBox(1, 20, 60, 12))],
            note="source_label=page_number",
            source_label="text",
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 40, 80, 12),
            lines=[Line(text="脚注", confidence=0.9, bbox=BBox(1, 40, 80, 12))],
            source_label="footnote",
        ),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)]

    assert texts == ["正文", "脚注"]

    print("test_hproof_line_iterator_excludes_position_source_labels PASSED")


def test_block_attributes_use_origin_or_current_label_not_raw_payload():
    from app.core.block_attributes import block_attributes, block_display_label, is_position_only_block
    from app.models import BBox, Block, BlockOrigin, BlockType

    title_like = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 1, 20, 10),
        note="source_label=text",
        origin=BlockOrigin(source_label="paragraph_title"),
    )
    attrs = block_attributes(title_like)

    assert attrs.source_label == "paragraph_title"
    assert attrs.raw_label == ""
    assert attrs.semantic_label == "paragraph_title"
    assert attrs.semantic_block_type == BlockType.TITLE
    assert block_display_label(title_like) == "text · paragraph_title"

    position = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 1, 20, 10),
        note="source_label=text",
        source_label="page_number",
    )
    assert is_position_only_block(position)

    footnote = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 20, 80, 10),
        source_label="footnote",
    )
    assert not is_position_only_block(footnote)
    assert block_display_label(footnote) == "text · footnote"

    raw_only = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 40, 80, 10),
    )
    raw_attrs = block_attributes(raw_only)
    assert raw_attrs.semantic_label == "text"
    assert raw_attrs.semantic_block_type == BlockType.TEXT

    print("test_block_attributes_use_origin_or_current_label_not_raw_payload PASSED")


def test_hproof_page_filter_keeps_pages_separate():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    page1 = Page(image_path="/tmp/hproof-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
        Line(text="第一页", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
    ])]
    page2 = Page(image_path="/tmp/hproof-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
        Line(text="第二页", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
    ])]

    panel = HProofPanel()
    panel.load_pages([page1, page2])
    assert len(panel._pairs) == 2

    # Phase 25：页面过滤通过 set_current_page_number / 页面目录唯一入口
    panel.set_current_page_number(2)

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].page is page2
    assert panel._pairs[0]._line_in_page == 1
    panel.close()

    print("test_hproof_page_filter_keeps_pages_separate PASSED")


def test_hproof_page_filter_flushes_dirty_editor_before_switching_pages():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line1 = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page1 = Page(image_path="/tmp/hproof-filter-dirty-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    line2 = Line(text="BBBB", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page2 = Page(image_path="/tmp/hproof-filter-dirty-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]

    panel = HProofPanel()
    panel.load_pages([page1, page2])
    panel._pairs[0]._editor.setPlainText("CCCC")

    panel.set_current_page_number(2)

    assert proof_display_text(line1) == "CCCC"
    assert proof_final_text(line1) == "CCCC"
    assert panel._session.selected_page_number == 2
    assert len(panel._pairs) == 1
    assert panel._session.projections[0].page is page2

    panel.set_current_page_number(1)

    assert panel._session.selected_page_number == 1
    assert len(panel._pairs) == 1
    assert panel._session.projections[0].page is page1
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    panel.close()

    print("test_hproof_page_filter_flushes_dirty_editor_before_switching_pages PASSED")


def test_hproof_page_filter_blocks_switch_when_current_editor_has_conflict():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line1 = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page1 = Page(image_path="/tmp/hproof-filter-conflict-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    line2 = Line(text="BBBB", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page2 = Page(image_path="/tmp/hproof-filter-conflict-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]

    panel = HProofPanel()
    panel.load_pages([page1, page2])
    panel._pairs[0]._editor.setPlainText("CCCC")
    set_line_proof_text(line1, "DDDD")
    assert panel._pairs[0].refresh_text() == "conflict"

    panel.set_current_page_number(2)

    assert panel._session.selected_page_number is None
    assert len(panel._pairs) == 2
    assert panel._session.projections[0].page is page1
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    assert panel._pairs[0].has_external_conflict()
    assert proof_display_text(line1) == "DDDD"
    assert "冲突" in panel._pairs[0]._status_lbl.text()
    panel.close()

    print("test_hproof_page_filter_blocks_switch_when_current_editor_has_conflict PASSED")


def test_hproof_merge_pages_preserves_active_editor_text():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    page1 = Page(image_path="/tmp/hproof-merge-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
        Line(text="第一页", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
    ])]
    page2 = Page(image_path="/tmp/hproof-merge-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[
        Line(text="第二页", confidence=0.9, bbox=BBox(1, 1, 20, 10)),
    ])]

    panel = HProofPanel()
    panel.load_pages([page1])
    panel._pairs[0]._editor.setPlainText("未保存横校文本")

    panel.merge_pages([page1, page2])

    assert len(panel._pairs) == 2
    assert panel._session.current_projection_index == 0
    assert panel._pairs[0]._editor.toPlainText() == "未保存横校文本"
    panel.close()

    print("test_hproof_merge_pages_preserves_active_editor_text PASSED")


def test_hproof_merge_rebinds_replaced_lines_without_duplicates_or_orphans():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    old_line = Line(text="旧对象", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    old_page = Page(
        image_path="/tmp/hproof-rebind.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    old_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[old_line])
    old_page.blocks = [old_block]
    new_line = Line(text="旧对象", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    new_page = Page(
        image_path="/tmp/hproof-rebind.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    new_page.uid = old_page.uid
    new_line.uid = old_line.uid
    new_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[new_line])
    new_block.uid = old_block.uid
    new_page.blocks = [new_block]

    panel = HProofPanel()
    panel.load_pages([old_page])
    panel._pairs[0]._editor.setPlainText("用户未保存")

    panel.merge_pages([new_page])
    panel._save_current(silent=True)

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].line is new_line
    assert panel._pairs[0].line is new_line
    assert proof_final_text(new_line) == "用户未保存"
    assert proof_display_text(old_line) == "旧对象"
    panel.close()

    print("test_hproof_merge_rebinds_replaced_lines_without_duplicates_or_orphans PASSED")


def test_hproof_merge_rebind_marks_conflict_when_dirty_editor_meets_new_model_text():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    old_line = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    old_page = Page(
        image_path="/tmp/hproof-rebind-conflict.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    old_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[old_line])
    old_page.blocks = [old_block]
    new_line = Line(text="DDDD", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    new_page = Page(
        image_path="/tmp/hproof-rebind-conflict.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    new_page.uid = old_page.uid
    new_line.uid = old_line.uid
    new_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[new_line])
    new_block.uid = old_block.uid
    new_page.blocks = [new_block]

    panel = HProofPanel()
    panel.load_pages([old_page])
    panel._pairs[0]._editor.setPlainText("CCCC")

    panel.merge_pages([new_page])

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].line is new_line
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    assert panel._pairs[0].has_external_conflict()
    assert "冲突" in panel._pairs[0]._status_lbl.text()

    panel._save_current(silent=True)

    assert proof_display_text(new_line) == "DDDD"
    assert proof_display_text(new_line) == "DDDD"

    panel._pairs[0]._editor.setPlainText("DDDD")
    assert not panel._pairs[0].has_external_conflict()
    panel._save_current(silent=True)
    assert proof_display_text(new_line) == "DDDD"
    panel.close()

    print("test_hproof_merge_rebind_marks_conflict_when_dirty_editor_meets_new_model_text PASSED")


def test_hproof_merge_uses_stable_uid_when_geometry_changes():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    old_line = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    old_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[old_line])
    old_page = Page(
        image_path="/tmp/hproof-rebind-uid-geometry.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
        blocks=[old_block],
    )
    new_line = Line(text="DDDD", confidence=0.9, bbox=BBox(2, 1, 20, 10))
    new_line.uid = old_line.uid
    new_block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=5, lines=[new_line])
    new_block.uid = old_block.uid
    new_page = Page(
        image_path="/tmp/hproof-rebind-uid-geometry.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
        blocks=[new_block],
    )
    new_page.uid = old_page.uid

    panel = HProofPanel()
    panel.load_pages([old_page])
    panel._pairs[0]._editor.setPlainText("CCCC")

    panel.merge_pages([new_page])

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].line is new_line
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    assert panel._pairs[0].has_external_conflict()

    panel._save_current(silent=True)

    assert proof_display_text(new_line) == "DDDD"
    panel.close()

    print("test_hproof_merge_uses_stable_uid_when_geometry_changes PASSED")


def test_hproof_merge_pages_removes_orphan_rows_absent_from_new_pages():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    old_line1 = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    old_line2 = Line(text="BBBB", confidence=0.9, bbox=BBox(1, 31, 20, 10))
    old_page = Page(
        image_path="/tmp/hproof-orphan.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    old_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 50),
        order=0,
        lines=[old_line1, old_line2],
    )
    old_page.blocks = [old_block]
    new_line = Line(text="CCCC", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    new_page = Page(
        image_path="/tmp/hproof-orphan.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    new_page.uid = old_page.uid
    new_line.uid = old_line1.uid
    new_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 50),
        order=0,
        lines=[new_line],
    )
    new_block.uid = old_block.uid
    new_page.blocks = [new_block]

    panel = HProofPanel()
    panel.load_pages([old_page])
    assert len(panel._pairs) == 2

    panel.merge_pages([new_page])

    assert len(panel._pairs) == 1
    assert len(panel._session.projections) == 1
    assert panel._session.projections[0].line is new_line
    assert panel._pairs[0].line is new_line
    assert all(projection.line is not old_line2 for projection in panel._session.projections)

    panel._pairs[0]._editor.setPlainText("SAVE")
    panel._save_current(silent=True)
    assert proof_final_text(new_line) == "SAVE"
    assert proof_display_text(old_line2) == "BBBB"
    panel.close()

    print("test_hproof_merge_pages_removes_orphan_rows_absent_from_new_pages PASSED")


def test_hproof_orphan_merge_restore_dirty_text_marks_conflict_when_model_changed():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    old_line1 = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    old_line2 = Line(text="BBBB", confidence=0.9, bbox=BBox(1, 31, 20, 10))
    old_page = Page(
        image_path="/tmp/hproof-orphan-conflict.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    old_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 50),
        order=0,
        lines=[old_line1, old_line2],
    )
    old_page.blocks = [old_block]
    new_line = Line(text="DDDD", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    new_page = Page(
        image_path="/tmp/hproof-orphan-conflict.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    new_page.uid = old_page.uid
    new_line.uid = old_line1.uid
    new_block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 80, 50),
        order=0,
        lines=[new_line],
    )
    new_block.uid = old_block.uid
    new_page.blocks = [new_block]

    panel = HProofPanel()
    panel.load_pages([old_page])
    panel._pairs[0]._editor.setPlainText("CCCC")

    panel.merge_pages([new_page])

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].line is new_line
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    assert panel._pairs[0].has_external_conflict()
    assert "冲突" in panel._pairs[0]._status_lbl.text()

    panel._save_current(silent=True)

    assert proof_display_text(new_line) == "DDDD"
    assert proof_display_text(new_line) == "DDDD"

    panel._pairs[0]._editor.setPlainText("DDDD")
    assert not panel._pairs[0].has_external_conflict()
    panel._save_current(silent=True)
    assert proof_display_text(new_line) == "DDDD"
    panel.close()

    print("test_hproof_orphan_merge_restore_dirty_text_marks_conflict_when_model_changed PASSED")


def test_hproof_debug_filter_blocks_rebuild_when_current_editor_has_conflict():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    formula_line = Line(text="$$ x=1 $$", confidence=1.0, bbox=BBox(1, 30, 40, 10))
    page = Page(image_path="/tmp/hproof-debug-conflict.png", width=100, height=100, page_number=1)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line]),
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 30, 80, 20), lines=[formula_line]),
    ]

    panel = HProofPanel()
    panel.load_pages([page])
    panel._pairs[0]._editor.setPlainText("CCCC")
    set_line_proof_text(line, "DDDD")
    assert panel._pairs[0].refresh_text() == "conflict"

    panel._btn_debug_formula.setChecked(True)

    assert panel._session.show_formula_debug is False
    assert panel._btn_debug_formula.isChecked() is False
    assert len(panel._pairs) == 1
    assert panel._pairs[0]._editor.toPlainText() == "CCCC"
    assert panel._pairs[0].has_external_conflict()
    assert proof_display_text(line) == "DDDD"
    panel.close()

    print("test_hproof_debug_filter_blocks_rebuild_when_current_editor_has_conflict PASSED")


def test_hproof_save_all_emits_only_when_current_line_is_saved():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page = Page(image_path="/tmp/hproof-save-all.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]

    panel = HProofPanel()
    panel.load_pages([page])
    changes: list = []
    panel.proof_changed.connect(changes.append)

    panel._save_all()
    assert changes == []

    panel._pairs[0]._editor.setPlainText("CCCC")
    set_line_proof_text(line, "DDDD")
    assert panel._pairs[0].refresh_text() == "conflict"
    panel._save_all()
    assert changes == []
    assert proof_display_text(line) == "DDDD"

    panel._pairs[0]._editor.setPlainText("DDDD")
    panel._save_all()
    assert changes == []

    panel._pairs[0]._editor.setPlainText("EEEE")
    panel._save_all()
    assert len(changes) == 1
    assert changes[0].text_changed is True
    assert proof_display_text(line) == "EEEE"
    panel.close()

    print("test_hproof_save_all_emits_only_when_current_line_is_saved PASSED")


def test_hproof_save_all_persists_probe_only_correction():
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line = Line(text="已", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/hproof-probe-only.png", width=100, height=100, page_number=1, blocks=[block])
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, 0),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    try:
        panel = HProofPanel()
        panel.load_pages([page])
        changes: list = []
        panel.proof_changed.connect(changes.append)

        assert panel._pairs[0]._editor.toPlainText() == "己"
        panel._pairs[0]._editor.setPlainText("已")
        panel._save_all()

        assert probe.observation == "corrected"
        assert len(changes) == 1
        assert changes[0].probe_changed is True
        assert proof_display_text(line) == "已"
        assert panel._pairs[0]._editor.toPlainText() == "已"
        assert not panel._pairs[0].is_editor_dirty()
        panel.close()
    finally:
        qp.reset_active_store()

    print("test_hproof_save_all_persists_probe_only_correction PASSED")


def test_hproof_synthetic_debug_line_is_readonly_and_not_persisted():
    from app.models import BBox, Block, BlockOrigin, BlockType, Page
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    block = Block(
        block_type=BlockType.EQUATION,
        bbox=BBox(0, 0, 80, 20),
        lines=[],
        origin=BlockOrigin(source_label="display_formula", raw_index=0),
    )
    page = Page(
        image_path="/tmp/hproof-synthetic-debug.png",
        width=100,
        height=100,
        page_number=1,
        blocks=[block],
    )
    _attach_raw_layout_records(page, [
        {
            "block_label": "display_formula",
            "block_bbox": [0, 0, 80, 20],
            "block_content": "ORIGINAL",
        }
    ])

    panel = HProofPanel()
    panel.load_pages([page])
    panel._btn_debug_formula.setChecked(True)

    assert len(panel._pairs) == 1
    assert panel._session.projections[0].line_index == -1
    assert panel._pairs[0]._editor.isReadOnly()

    panel._pairs[0]._editor.setPlainText("EDITED")
    panel._save_current(silent=True)

    assert _raw_layout_records(page)[0]["block_content"] == "ORIGINAL"

    changes: list = []
    panel.proof_changed.connect(changes.append)
    panel._toggle_flag()
    panel._on_confirmed(0)
    assert changes == []
    assert proof_status(panel._session.projections[0].line).name == "UNCHECKED"

    panel._btn_debug_formula.setChecked(False)
    panel._btn_debug_formula.setChecked(True)

    assert panel._pairs[0]._editor.toPlainText() == "ORIGINAL"
    panel.close()

    print("test_hproof_synthetic_debug_line_is_readonly_and_not_persisted PASSED")


def test_hproof_toggle_flag_emits_proof_changed_for_persistent_line():
    from app.models import BBox, Block, BlockType, Line, Page, ProofStatus
    from app.ui.proof.h_proof import HProofPanel

    _get_qapp()
    line = Line(text="AAAA", confidence=0.9, bbox=BBox(1, 1, 40, 10))
    page = Page(image_path="/tmp/hproof-flag.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]

    panel = HProofPanel()
    panel.load_pages([page])
    changes = []
    panel.proof_changed.connect(changes.append)

    panel._toggle_flag()

    assert proof_status(line) == ProofStatus.AUTO_FLAGGED
    assert len(changes) == 1
    assert changes[0].status_changed is True
    assert changes[0].needs_persist is True
    panel.close()

    print("test_hproof_toggle_flag_emits_proof_changed_for_persistent_line PASSED")


def test_vproof_merge_pages_reloads_current_page_reference_text():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line1 = Line(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page1 = Page(image_path="/tmp/vproof-merge-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    line2 = Line(
        text="乙",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="乙", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page2 = Page(image_path="/tmp/vproof-merge-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]

    panel = VProofPanel()
    panel.load_pages([page1])
    panel._text_edit.setPlainText("未保存纵校文本")

    panel.merge_pages([page1, page2])

    assert panel._session.current_page_index() == 0
    assert panel._text_edit.toPlainText() == "甲\n"
    assert panel._char_svc.query("乙")
    panel.close()

    print("test_vproof_merge_pages_reloads_current_page_reference_text PASSED")


def test_vproof_merge_pages_keeps_current_page_by_uid_when_order_changes():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    p1_line = Line(
        text="P1",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="P", confidence=0.9, bbox=BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="1", confidence=0.9, bbox=BBox(10, 0, 10, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    p1 = Page(image_path="/tmp/vproof-order-p1.png", width=100, height=100, page_number=1)
    p1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[p1_line])]
    p2_line = Line(
        text="P2",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="P", confidence=0.9, bbox=BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="2", confidence=0.9, bbox=BBox(10, 0, 10, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    p2 = Page(image_path="/tmp/vproof-order-p2.png", width=100, height=100, page_number=2)
    p2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[p2_line])]

    new_p2 = Page(image_path="/tmp/vproof-order-p2.png", width=100, height=100, page_number=2)
    new_p2.uid = p2.uid
    new_p2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[p2_line])]
    new_p1 = Page(image_path="/tmp/vproof-order-p1.png", width=100, height=100, page_number=1)
    new_p1.uid = p1.uid
    new_p1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[p1_line])]

    panel = VProofPanel()
    panel.load_pages([p1, p2])
    assert panel._safe_load_page(1) is True
    assert panel._text_edit.toPlainText() == "P2\n"

    panel.merge_pages([new_p2, new_p1])

    assert panel._session.current_page_index() == 0
    assert panel._session.pages[0] is new_p2
    assert panel._text_edit.toPlainText() == "P2\n"
    panel.close()

    print("test_vproof_merge_pages_keeps_current_page_by_uid_when_order_changes PASSED")


def test_vproof_target_edit_persists_probe_only_correction():
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="已",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="已", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/vproof-probe-only.png", width=100, height=100, page_number=1, blocks=[block])
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, 0),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    try:
        panel = VProofPanel()
        panel.load_pages([page])
        changes: list = []
        panel.proof_changed.connect(changes.append)

        assert panel._text_edit.toPlainText() == "己\n"
        entries = panel._char_svc.query("已")
        assert entries
        entry = entries[0]
        panel._gallery_model.set_entries(entries)
        panel._sync_gallery_entry(panel._gallery_model.index(0, 0))

        assert panel._apply_replacement_to_selected("已", fallback_entry=entry) == 1
        assert probe.observation == "corrected"
        assert len(changes) == 1
        assert changes[0].probe_changed is True
        assert panel._text_edit.toPlainText() == "已\n"
        assert panel._session.loaded_text == "已\n"
        panel.close()
    finally:
        qp.reset_active_store()

    print("test_vproof_target_edit_persists_probe_only_correction PASSED")


def test_vproof_refresh_reference_context_persists_stale_probe_anchor_correction():
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="已",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="已", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/vproof-stale-probe.png", width=100, height=100, page_number=1, blocks=[block])
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, 0),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    try:
        set_line_proof_text(line, "巳")
        panel = VProofPanel()
        panel.load_pages([page])
        changes = []
        panel.proof_changed.connect(changes.append)

        assert panel._text_edit.toPlainText() == "巳\n"
        assert panel._refresh_reference_context() is True

        assert probe.observation == "corrected"
        assert len(changes) == 1
        assert changes[0].probe_changed is True
        assert changes[0].needs_persist is True
        assert proof_display_text(line) == "巳"
        panel.close()
    finally:
        qp.reset_active_store()

    print("test_vproof_refresh_reference_context_persists_stale_probe_anchor_correction PASSED")


def test_vproof_replacement_commits_text_and_probe_together():
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="已",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="已", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/vproof-precommit-probe.png", width=100, height=100, page_number=1, blocks=[block])
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, 0),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    try:
        panel = VProofPanel()
        panel.load_pages([page])
        changes = []
        panel.proof_changed.connect(changes.append)

        entries = panel._char_svc.query("已")
        assert entries
        entry = entries[0]
        panel._gallery_model.set_entries(entries)
        panel._sync_gallery_entry(panel._gallery_model.index(0, 0))

        assert panel._text_edit.toPlainText() == "己\n"
        assert panel._apply_replacement_to_selected("巳", fallback_entry=entry) == 1

        assert proof_display_text(line) == "巳"
        assert probe.observation == "corrected"
        assert len(changes) == 1
        assert changes[0].text_changed is True
        assert changes[0].probe_changed is True
        assert panel._text_edit.toPlainText() == "巳\n"
        panel.close()
    finally:
        qp.reset_active_store()

    print("test_vproof_replacement_commits_text_and_probe_together PASSED")


def test_vproof_undo_redo_uses_line_edit_actions():
    from app.core import quality_probe as qp
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="已",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="已", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])
    page = Page(image_path="/tmp/vproof-history-probe.png", width=100, height=100, page_number=1, blocks=[block])
    store = qp.ProbeStore()
    probe = qp.Probe(
        key=qp.ProbeKey(page.page_number, 0, 0, 0),
        true_char="已",
        fake_char="己",
        observation="pending",
    )
    store.add(probe)
    qp.set_active_store(store)
    try:
        panel = VProofPanel()
        panel.load_pages([page])
        changes = []
        panel.proof_changed.connect(changes.append)

        entries = panel._char_svc.query("已")
        assert entries
        entry = entries[0]
        assert panel._apply_replacement_to_selected("巳", fallback_entry=entry) == 1
        assert proof_display_text(line) == "巳"
        assert probe.observation == "corrected"
        assert len(panel._vproof_undo_stack) == 1

        assert panel._undo_vproof_edit() is True

        assert proof_display_text(line) == "已"
        assert probe.observation == "corrected"
        assert len(panel._vproof_redo_stack) == 1
        assert panel._text_edit.toPlainText() == "已\n"

        assert panel._redo_vproof_edit() is True

        assert proof_display_text(line) == "巳"
        assert panel._text_edit.toPlainText() == "巳\n"
        assert len(changes) == 3
        assert changes[0].text_changed is True
        assert changes[0].probe_changed is True
        assert changes[1].text_changed is True
        assert changes[2].text_changed is True
        panel.close()
    finally:
        qp.reset_active_store()

    print("test_vproof_undo_redo_uses_line_edit_actions PASSED")


def test_vproof_external_refresh_invalidates_undo_history_before_restore():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="AAAA",
        confidence=0.9,
        bbox=BBox(1, 1, 80, 10),
        chars=[
            Char(char="A", confidence=0.9, bbox=BBox(1 + i * 12, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")
            for i in range(4)
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 90, 20), lines=[line])
    page = Page(image_path="/tmp/vproof-external-undo.png", width=120, height=80, page_number=1, blocks=[block])

    panel = VProofPanel()
    panel.load_pages([page])
    entry = panel._char_svc.query("A")[0]
    assert panel._apply_replacement_to_selected("B", fallback_entry=entry) == 1
    assert panel._vproof_undo_stack

    set_line_proof_text(line, "CCCC")
    panel._session.queue_external_refresh(line_key=line.uid, page_keys=[])
    panel._do_external_refresh()

    assert proof_display_text(line) == "CCCC"
    assert panel._text_edit.toPlainText() == "CCCC\n"
    assert panel._vproof_undo_stack == []
    assert panel._vproof_redo_stack == []
    assert panel._undo_vproof_edit() is False
    assert proof_display_text(line) == "CCCC"
    panel.close()

    print("test_vproof_external_refresh_invalidates_undo_history_before_restore PASSED")


def test_vproof_save_refreshes_current_page_index_incrementally():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line1 = Line(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page1 = Page(image_path="/tmp/vproof-incremental-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    line2 = Line(
        text="乙",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="乙", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page2 = Page(image_path="/tmp/vproof-incremental-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]

    panel = VProofPanel()
    panel.load_pages([page1, page2])

    def fail_full_build(_pages):
        raise AssertionError("VProof save should refresh the current page, not rebuild every page")

    panel._char_svc.build = fail_full_build  # type: ignore[method-assign]
    entry = panel._char_svc.query("甲")[0]

    assert panel._apply_replacement_to_selected("丙", fallback_entry=entry) == 1
    assert not panel._char_svc.query("甲")
    assert panel._char_svc.query("丙")
    assert panel._char_svc.query("乙")
    panel.close()

    print("test_vproof_save_refreshes_current_page_index_incrementally PASSED")


def test_char_index_service_replace_pages_preserves_other_page_entries():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.services.char_index_service import CharIndexService

    line1 = Line(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page1 = Page(image_path="/tmp/char-index-replace-p1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    line2 = Line(
        text="乙",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="乙", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page2 = Page(image_path="/tmp/char-index-replace-p2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]

    service = CharIndexService(include_non_cjk=True, include_fallback=True).build([page1, page2])
    set_line_proof_text(line1, "丙")
    line1.chars = [
        Char(char="丙", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")
    ]

    service.replace_pages([page1])

    assert not service.query("甲")
    assert service.query("丙")
    assert service.query("乙")

    print("test_char_index_service_replace_pages_preserves_other_page_entries PASSED")


def test_vproof_indexes_latin_digits_and_punctuation():
    from PySide6.QtCore import Qt

    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="甲A，1。",
        confidence=0.9,
        bbox=BBox(1, 1, 90, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 12, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="A", confidence=0.9, bbox=BBox(18, 1, 12, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="，", confidence=0.9, bbox=BBox(34, 1, 8, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="1", confidence=0.9, bbox=BBox(48, 1, 10, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="。", confidence=0.9, bbox=BBox(62, 1, 8, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page = Page(image_path="/tmp/vproof-noncjk.png", width=120, height=80, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40), lines=[line])]

    panel = VProofPanel()
    panel.load_pages([page])

    assert panel._char_svc.query("甲")
    assert panel._char_svc.query("A")
    assert panel._char_svc.query("，")
    assert panel._char_svc.query("1")
    assert panel._char_svc.query("。")
    tokens = {
        panel._char_list.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(panel._char_list.count())
    }
    assert {"甲", "A", "，", "1", "。"}.issubset(tokens)

    item = next(
        panel._char_list.item(i)
        for i in range(panel._char_list.count())
        if panel._char_list.item(i).data(Qt.ItemDataRole.UserRole) == "A"
    )
    panel._on_char_clicked(item)
    assert panel._gallery_model.rowCount() == 1

    panel.reset()
    panel.load_pages([page])
    assert panel._char_svc.query("A")
    panel.close()

    print("test_vproof_indexes_latin_digits_and_punctuation PASSED")


def test_vproof_indexes_line_fallback_when_chars_are_missing():
    from PySide6.QtCore import Qt

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="甲A1。",
        confidence=0.8,
        bbox=BBox(1, 1, 80, 22),
        chars=[],
    )
    page = Page(image_path="/tmp/vproof-line-fallback.png", width=120, height=80, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40), lines=[line])]

    panel = VProofPanel()
    panel.load_pages([page])

    assert panel._char_svc.query("甲")
    assert panel._char_svc.query("A")
    assert panel._char_svc.query("1")
    assert panel._char_svc.query("。")
    tokens = {
        panel._char_list.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(panel._char_list.count())
    }
    assert {"甲", "A", "1", "。"}.issubset(tokens)
    panel.close()

    print("test_vproof_indexes_line_fallback_when_chars_are_missing PASSED")


def test_top_bar_hosts_workflow_steps_and_layout_run():
    from app.controllers.workflow_controller import STEP_HPROOF, STEP_LAYOUT
    from app.ui.widgets.top_bar import TopBar

    _get_qapp()
    nav = TopBar()

    assert not hasattr(nav, "_btn_prev")
    assert not hasattr(nav, "_btn_next")
    assert nav._btn_run_layout.text() == "运行版面分析"
    assert [btn.text() for btn in nav._step_buttons] == ["版面分析", "横校", "纵校"]
    nav.set_enabled_up_to(STEP_LAYOUT)
    assert nav._step_buttons[0].isEnabled()
    assert not nav._step_buttons[1].isEnabled()
    nav.set_enabled_up_to(STEP_HPROOF)
    assert nav._step_buttons[1].isEnabled()
    nav.set_active(STEP_LAYOUT)
    assert nav._step_buttons[0].isChecked()
    nav.set_layout_run_enabled(True)
    assert nav._btn_run_layout.isEnabled()
    nav.close()

    print("test_top_bar_hosts_workflow_steps_and_layout_run PASSED")


def test_vproof_gallery_uses_wrapping_white_grid():
    from PySide6.QtCore import Qt

    from app.ui.proof import v_proof
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    panel = VProofPanel()

    assert panel._gallery_view.isWrapping() is True
    assert panel._gallery_view.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert panel._gallery_view.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert "background:#ffffff" in panel._gallery_view.styleSheet()
    assert "background:#ffffff" in panel._char_list.styleSheet()
    # proof-bbox-boxedit round 9: thumb 从 27 抬到 34；round 12 再抬到 56
    # (CJK 真实可读)，sizeHint 上限同步放宽到 ≤ 80
    assert v_proof.GALLERY_THUMB >= 30
    assert v_proof.CHAR_LIST_THUMB == 18
    assert 170 <= panel._left_box.maximumWidth() <= 230
    assert panel._gallery_view.itemDelegate().sizeHint(None, panel._gallery_model.index(0, 0)).height() <= 80
    assert panel._gallery_box.parentWidget() is panel._proof_column
    assert panel._btn_save.parentWidget() is panel._gallery_box
    assert panel._conf_badge.isVisible() is False
    assert panel._ocr_text_box.parentWidget() is panel._proof_column
    assert panel._candidate_box.parentWidget() is panel._proof_column
    assert "border:0" in panel._candidate_box.styleSheet()
    panel.close()

    print("test_vproof_gallery_uses_wrapping_white_grid PASSED")


def test_vproof_text_highlight_targets_single_entry():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="甲乙",
        confidence=0.9,
        bbox=BBox(1, 1, 40, 10),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page = Page(image_path="/tmp/vproof-highlight.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.load_pages([page])
    entry = panel._char_svc.query("乙")[0]

    panel._highlight_char_in_text("乙", focus_entry=entry)

    selections = panel._text_edit.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.selectedText() == "乙"
    panel.close()

    print("test_vproof_text_highlight_targets_single_entry PASSED")


def test_vproof_highlight_can_repeat_without_losing_state():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="甲乙丙",
        confidence=0.9,
        bbox=BBox(1, 1, 60, 10),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
            Char(char="丙", confidence=0.9, bbox=BBox(40, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page = Page(image_path="/tmp/vproof-repeat.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.load_pages([page])

    for token in ("甲", "乙", "丙", "甲", "乙"):
        entry = panel._char_svc.query(token)[0]
        panel._highlight_char_in_text(token, focus_entry=entry)
        panel._highlight_char_in_viewer(entry)
        assert len(panel._text_edit.extraSelections()) == 1
        assert panel._text_edit.textCursor().selectedText() == token
        assert panel._viewer._highlight_item is not None

    panel.close()

    print("test_vproof_highlight_can_repeat_without_losing_state PASSED")


def test_vproof_highlight_survives_repeated_page_switches():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line1 = Line(
        text="甲",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="甲", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    line2 = Line(
        text="乙",
        confidence=0.9,
        bbox=BBox(20, 20, 20, 10),
        chars=[Char(char="乙", confidence=0.9, bbox=BBox(20, 20, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page1 = Page(image_path="/tmp/vproof-switch-1.png", width=100, height=100, page_number=1)
    page1.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    page2 = Page(image_path="/tmp/vproof-switch-2.png", width=100, height=100, page_number=2)
    page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line2])]
    panel = VProofPanel()
    panel.load_pages([page1, page2])

    for token in ("甲", "乙", "甲", "乙", "甲", "乙"):
        entry = panel._char_svc.query(token)[0]
        panel._highlight_char_in_viewer(entry)
        panel._highlight_char_in_text(token, focus_entry=entry)
        assert panel._viewer._highlight_item is not None
        assert len(panel._text_edit.extraSelections()) == 1
        assert panel._text_edit.textCursor().selectedText() == token

    panel.close()

    print("test_vproof_highlight_survives_repeated_page_switches PASSED")


def test_vproof_gallery_keyboard_selection_refreshes_linked_panels():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line = Line(
        text="田田",
        confidence=0.9,
        bbox=BBox(1, 1, 40, 10),
        chars=[
            Char(char="田", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
            Char(char="田", confidence=0.8, bbox=BBox(20, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page = Page(image_path="/tmp/vproof-keyboard.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.load_pages([page])
    entries = panel._char_svc.query("田")
    panel._selected_char = "田"
    panel._gallery_model.set_entries(entries)

    panel._gallery_view.setCurrentIndex(panel._gallery_model.index(0, 0))
    first_pos = panel._text_edit.textCursor().selectionStart()
    panel._gallery_view.setCurrentIndex(panel._gallery_model.index(1, 0))

    assert panel._current_candidate_entry is entries[1]
    assert panel._text_edit.textCursor().selectedText() == "田"
    assert panel._text_edit.textCursor().selectionStart() != first_pos
    assert panel._viewer._highlight_item is not None
    assert len(panel._candidate_buttons) >= 2
    assert "由" in [button.text() for button in panel._candidate_buttons]
    panel.close()

    print("test_vproof_gallery_keyboard_selection_refreshes_linked_panels PASSED")


def test_vproof_candidate_button_applies_to_ocr_text():
    import time

    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    app = _get_qapp()
    line = Line(
        text="田",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="田", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page = Page(image_path="/tmp/vproof-candidate-apply.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.load_pages([page])
    entry = panel._char_svc.query("田")[0]

    panel._selected_char = "田"
    panel._current_candidate_entry = entry
    panel._update_candidate_panel(entry)
    changes = []
    panel.proof_changed.connect(changes.append)
    button_by_text = {button.text(): button for button in panel._candidate_buttons}
    button_by_text["由"].click()

    assert panel._text_edit.toPlainText().startswith("由")
    assert "已应用" in panel._status_lbl.text()
    assert len(changes) == 1
    assert changes[0].text_changed is True
    assert len(changes[0].line_refs) == 1
    deadline = time.monotonic() + 0.6
    while time.monotonic() < deadline and proof_display_text(line) != "由":
        app.processEvents()
        time.sleep(0.01)
    assert proof_display_text(line) == "由"
    panel.close()

    print("test_vproof_candidate_button_applies_to_ocr_text PASSED")


def test_vproof_right_click_edit_bubble_batches_and_undoes():
    from PySide6.QtCore import QPoint
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    app = _get_qapp()
    line = Line(
        text="田田",
        confidence=0.9,
        bbox=BBox(1, 1, 40, 10),
        chars=[
            Char(char="田", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
            Char(char="田", confidence=0.9, bbox=BBox(20, 1, 10, 10), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page = Page(image_path="/tmp/vproof-bubble.png", width=100, height=100, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.resize(800, 600)
    panel.show()
    app.processEvents()
    panel.load_pages([page])
    entries = panel._char_svc.query("田")
    panel._selected_char = "田"
    panel._gallery_model.set_entries(entries)

    idx0 = panel._gallery_model.index(0, 0)
    idx1 = panel._gallery_model.index(1, 0)
    sel = panel._gallery_view.selectionModel()
    sel.select(idx0, sel.SelectionFlag.ClearAndSelect)
    sel.select(idx1, sel.SelectionFlag.Select)
    sel.setCurrentIndex(idx0, sel.SelectionFlag.Current)
    panel._sync_gallery_entry(idx0)
    changes = []
    panel.proof_changed.connect(changes.append)

    panel._show_edit_bubble_at(QPoint(20, 20))
    assert panel._edit_bubble.isVisible()
    assert panel._edit_bubble_input.placeholderText() == "替换 2 处"
    panel._edit_bubble_input.setText("由")
    panel._apply_edit_bubble()

    assert len(changes) == 1
    assert changes[0].text_changed is True
    assert len(changes[0].line_refs) == 1
    assert panel._text_edit.toPlainText().startswith("由由")
    assert len(panel._vproof_undo_stack) == 1
    assert len(panel._vproof_undo_stack) <= 5

    panel._undo_vproof_edit()
    assert panel._text_edit.toPlainText().startswith("田田")
    assert len(panel._vproof_redo_stack) == 1

    panel._redo_vproof_edit()
    assert panel._text_edit.toPlainText().startswith("由由")
    panel.close()

    print("test_vproof_right_click_edit_bubble_batches_and_undoes PASSED")


def test_vproof_undo_redo_restores_line_model_after_single_and_batch_edits():
    from PySide6.QtCore import QPoint
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    app = _get_qapp()
    text = "一四四四四"
    line = Line(
        text=text,
        confidence=0.9,
        bbox=BBox(1, 1, 100, 10),
        chars=[
            Char(
                char=ch,
                confidence=0.9,
                bbox=BBox(1 + i * 12, 1, 10, 10),
                bbox_source="ocr",
                bbox_granularity="char",
            )
            for i, ch in enumerate(text)
        ],
    )
    page = Page(image_path="/tmp/vproof-undo-redo-model.png", width=140, height=80, page_number=1)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 120, 20), lines=[line])]
    panel = VProofPanel()
    panel.resize(800, 600)
    panel.show()
    app.processEvents()
    panel.load_pages([page])

    def select_token(token: str, *, all_entries: bool = False) -> None:
        entries = panel._char_svc.query(token)
        panel._selected_char = token
        panel._gallery_model.set_entries(entries)
        panel._resize_gallery_for_entries(len(entries))
        idx0 = panel._gallery_model.index(0, 0)
        sel = panel._gallery_view.selectionModel()
        if all_entries:
            sel.select(
                panel._gallery_model.index(0, 0),
                sel.SelectionFlag.ClearAndSelect,
            )
            for row in range(1, panel._gallery_model.rowCount()):
                sel.select(panel._gallery_model.index(row, 0), sel.SelectionFlag.Select)
        else:
            sel.select(idx0, sel.SelectionFlag.ClearAndSelect)
        sel.setCurrentIndex(idx0, sel.SelectionFlag.Current)
        panel._sync_gallery_entry(idx0)

    select_token("一")
    panel._show_edit_bubble_at(QPoint(20, 20))
    panel._edit_bubble_input.setText("二")
    panel._apply_edit_bubble()
    panel._refresh_current_selection_context()
    assert proof_display_text(line) == "二四四四四"

    select_token("四", all_entries=True)
    panel._show_edit_bubble_at(QPoint(20, 20))
    panel._edit_bubble_input.setText("三")
    panel._apply_edit_bubble()
    panel._refresh_current_selection_context()
    assert proof_display_text(line) == "二三三三三"

    panel._undo_vproof_edit()
    assert proof_display_text(line) == "二四四四四"
    assert panel._text_edit.toPlainText().startswith("二四四四四")

    panel._undo_vproof_edit()
    assert proof_display_text(line) == "一四四四四"
    assert panel._text_edit.toPlainText().startswith("一四四四四")

    panel._redo_vproof_edit()
    assert proof_display_text(line) == "二四四四四"
    assert panel._text_edit.toPlainText().startswith("二四四四四")

    panel._redo_vproof_edit()
    assert proof_display_text(line) == "二三三三三"
    assert panel._text_edit.toPlainText().startswith("二三三三三")
    panel.close()

    print("test_vproof_undo_redo_restores_line_model_after_single_and_batch_edits PASSED")


def test_vproof_save_rejects_stale_text_editor_page_binding():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    line1 = Line(
        text="甲甲",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="甲", confidence=0.9, bbox=BBox(20, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    line2 = Line(
        text="乙乙",
        confidence=0.9,
        bbox=BBox(0, 30, 40, 20),
        chars=[
            Char(char="乙", confidence=0.9, bbox=BBox(0, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    page1 = Page(image_path="/tmp/vproof-stale-page-1.png", width=100, height=100, page_number=1)
    page2 = Page(image_path="/tmp/vproof-stale-page-2.png", width=100, height=100, page_number=2)
    page1.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 20), lines=[line1])]
    page2.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 30, 80, 20), lines=[line2])]

    panel = VProofPanel()
    panel.load_pages([page1, page2])
    assert panel._text_edit.toPlainText().startswith("甲甲")

    panel._text_edit.setPlainText("丙丙\n")
    panel._session.set_current_page_by_index(1)

    assert panel._refresh_reference_context() is False
    assert proof_display_text(line1) == "甲甲"
    assert proof_display_text(line2) == "乙乙"
    assert "已刷新" in panel._status_lbl.text()
    panel.close()

    print("test_vproof_save_rejects_stale_text_editor_page_binding PASSED")


def test_vproof_merge_pages_blocks_dirty_editor_when_current_page_object_replaced():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    old_line = Line(
        text="甲甲",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="甲", confidence=0.9, bbox=BBox(20, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    old_page = Page(
        image_path="/tmp/vproof-merge-stale-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-same-key",
    )
    old_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 20), lines=[old_line])]
    new_line = Line(
        text="乙乙",
        confidence=0.9,
        bbox=BBox(0, 30, 40, 20),
        chars=[
            Char(char="乙", confidence=0.9, bbox=BBox(0, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    new_page = Page(
        image_path="/tmp/vproof-merge-stale-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-same-key",
    )
    new_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 30, 80, 20), lines=[new_line])]

    panel = VProofPanel()
    panel.load_pages([old_page])
    panel._text_edit.setPlainText("丙丙\n")

    panel.merge_pages([new_page])

    assert panel._session.current_page_index() == 0
    assert panel._text_edit.toPlainText() == "乙乙\n"
    assert panel._refresh_reference_context() is False
    assert proof_display_text(old_line) == "甲甲"
    assert proof_display_text(new_line) == "乙乙"
    panel.close()

    print("test_vproof_merge_pages_blocks_dirty_editor_when_current_page_object_replaced PASSED")


def test_vproof_undo_redo_reject_stale_actions_after_current_page_replaced():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    old_line = Line(
        text="甲甲",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="甲", confidence=0.9, bbox=BBox(20, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    old_page = Page(
        image_path="/tmp/vproof-undo-stale-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-undo-stale",
    )
    old_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 20), lines=[old_line])]
    new_line = Line(
        text="乙乙",
        confidence=0.9,
        bbox=BBox(0, 30, 40, 20),
        chars=[
            Char(char="乙", confidence=0.9, bbox=BBox(0, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    new_page = Page(
        image_path="/tmp/vproof-undo-stale-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-undo-stale",
    )
    new_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 30, 80, 20), lines=[new_line])]

    panel = VProofPanel()
    panel.load_pages([old_page])
    entry = panel._char_svc.query("甲")[0]
    assert panel._apply_replacement_to_selected("丙", fallback_entry=entry) == 1
    assert proof_display_text(old_line) == "丙甲"
    old_action = panel._vproof_undo_stack[-1]
    panel.merge_pages([new_page])

    assert panel._vproof_undo_stack == []
    assert panel._vproof_redo_stack == []

    panel._vproof_undo_stack.append(old_action)
    panel._undo_vproof_edit()

    assert len(panel._vproof_undo_stack) == 1
    assert len(panel._vproof_redo_stack) == 0
    assert "撤销已取消" in panel._status_lbl.text()
    assert proof_display_text(new_line) == "乙乙"
    assert [char.char for char in new_line.chars] == ["乙", "乙"]
    assert [char.bbox.y for char in new_line.chars] == [30, 30]
    assert panel._char_svc.query("甲") == []
    assert panel._char_svc.query("乙")

    panel._vproof_undo_stack.clear()
    panel._vproof_redo_stack.append(old_action)
    panel._redo_vproof_edit()

    assert len(panel._vproof_redo_stack) == 1
    assert len(panel._vproof_undo_stack) == 0
    assert "重做已取消" in panel._status_lbl.text()
    assert proof_display_text(new_line) == "乙乙"
    assert [char.char for char in new_line.chars] == ["乙", "乙"]
    panel.close()

    print("test_vproof_undo_redo_reject_stale_actions_after_current_page_replaced PASSED")


def test_vproof_save_rejects_stale_text_map_even_when_page_key_matches():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
    old_line = Line(
        text="甲甲",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="甲", confidence=0.9, bbox=BBox(20, 0, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    old_page = Page(
        image_path="/tmp/vproof-stale-map-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-stale-map",
    )
    old_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 0, 80, 20), lines=[old_line])]
    new_line = Line(
        text="乙乙",
        confidence=0.9,
        bbox=BBox(0, 30, 40, 20),
        chars=[
            Char(char="乙", confidence=0.9, bbox=BBox(0, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 30, 18, 20), bbox_source="ocr", bbox_granularity="char"),
        ],
    )
    new_page = Page(
        image_path="/tmp/vproof-stale-map-same-key.png",
        width=100,
        height=100,
        page_number=1,
        uid="page-stale-map",
    )
    new_page.blocks = [Block(block_type=BlockType.TEXT, order=0, bbox=BBox(0, 30, 80, 20), lines=[new_line])]

    panel = VProofPanel()
    panel.load_pages([old_page])
    panel._text_edit.setPlainText("丙丙\n")
    panel._session.set_pages([new_page])
    panel._session.set_current_page_by_index(0)
    panel._session.current_page_key = panel._char_index_page_key(new_page)

    assert panel._refresh_reference_context() is False
    assert proof_display_text(old_line) == "甲甲"
    assert proof_display_text(new_line) == "乙乙"
    assert "已刷新" in panel._status_lbl.text()
    panel.close()

    print("test_vproof_save_rejects_stale_text_map_even_when_page_key_matches PASSED")


def test_hproof_visual_size_is_compact():
    from app.ui.proof import h_proof

    assert h_proof.IMAGE_ROW_H <= 38
    # 脚注/数字/标点的 Hanwang 字符框更窄，横校文本字号只能小幅放大。
    assert h_proof.TEXT_FONT_PX == 26
    assert h_proof.TEXT_FONT_PX <= 26
    assert h_proof.TEXT_FONT_WEIGHT_CSS >= 700
    assert h_proof.TEXT_LINE_HEIGHT_PX <= 34
    assert h_proof.TEXT_EDITOR_MAX_H <= 40
    assert h_proof.LINE_PAIR_H == 92
    assert (
        h_proof.LINE_PAIR_H
        >= h_proof.IMAGE_ROW_H + h_proof.TEXT_EDITOR_MAX_H + 2
    )
    assert "Noto Serif CJK SC" in h_proof.TEXT_FONT_FAMILY
    assert "SimSun" in h_proof.TEXT_FONT_FAMILY
    assert "Times New Roman" in h_proof.TEXT_FONT_FAMILY
    assert "SimHei" not in h_proof.TEXT_FONT_FAMILY
    assert h_proof.IMAGE_DIVIDER_COLOR != h_proof.TEXT_GUIDE_LINE_COLOR
    assert h_proof.FOCUS_BORDER_COLOR != "#2C2C2C"
    assert h_proof.STATUS_W <= 24
    assert h_proof.TEXT_SLOT_MIN_W >= 10.0
    assert h_proof.TEXT_SLOT_GUTTER_W >= 2.0

    print("test_hproof_visual_size_is_compact PASSED")


def test_image_viewer_char_boxes_update_char_bbox():
    from PySide6.QtGui import QImage

    from app.models import BBox, Char
    from app.ui.widgets.image_viewer import ImageViewer

    app = _get_qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(80, 60, QImage.Format.Format_RGB888))
    char = Char(char="字", confidence=0.9, bbox=BBox(10, 12, 20, 22))
    viewer.show_char_boxes([char])
    item, _ = viewer._char_items[0]

    assert item.pen().widthF() <= 1.0
    item.setPos(14, 16)
    app.processEvents()

    assert char.bbox == BBox(14, 16, 20, 22)
    viewer.close()

    print("test_image_viewer_char_boxes_update_char_bbox PASSED")


def test_image_viewer_space_pan_temporarily_disables_box_editing():
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QImage, QKeyEvent
    from PySide6.QtWidgets import QGraphicsItem, QGraphicsView

    from app.models import BBox, Block, BlockType
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(80, 60, QImage.Format.Format_RGB888))
    block = Block(block_type=BlockType.EQUATION, bbox=BBox(10, 12, 20, 22))
    viewer.show_blocks([block])
    item, _ = viewer._block_items[0]

    assert viewer.dragMode() == QGraphicsView.DragMode.NoDrag
    assert bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

    viewer.keyPressEvent(QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert viewer.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    assert not bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

    viewer.keyReleaseEvent(QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert viewer.dragMode() == QGraphicsView.DragMode.NoDrag
    assert bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
    viewer.close()

    print("test_image_viewer_space_pan_temporarily_disables_box_editing PASSED")


def test_image_viewer_shift_left_drag_creates_block_bbox():
    from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
    from PySide6.QtGui import QImage, QMouseEvent

    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.resize(300, 240)
    viewer.set_image_from_qimage(QImage(100, 80, QImage.Format.Format_RGB888))
    created = []
    viewer.block_created.connect(lambda bbox: created.append(bbox))

    start = QPointF(viewer.mapFromScene(QPointF(10, 10)))
    end = QPointF(viewer.mapFromScene(QPointF(40, 30)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    viewer.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove,
        end,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        end,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))

    assert len(created) == 1
    assert created[0].w >= 10
    assert created[0].h >= 10
    viewer.close()

    print("test_image_viewer_shift_left_drag_creates_block_bbox PASSED")


def test_image_viewer_draw_uses_snapper_only_for_created_bbox():
    from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
    from PySide6.QtGui import QImage, QMouseEvent

    from app.models import BBox
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.resize(300, 240)
    viewer.set_image_from_qimage(QImage(100, 80, QImage.Format.Format_RGB888))
    viewer.set_bbox_snapper(lambda _bbox: BBox(10, 10, 40, 20))
    created = []
    viewer.block_created.connect(lambda bbox: created.append(bbox))

    start = QPointF(viewer.mapFromScene(QPointF(9, 9)))
    end = QPointF(viewer.mapFromScene(QPointF(43, 27)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    viewer.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove,
        end,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    assert viewer._draw_item is not None
    assert viewer._draw_item.rect() != QRectF(10, 10, 40, 20)
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        end,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ShiftModifier,
    ))

    assert created == [BBox(10, 10, 40, 20)]
    viewer.close()

    print("test_image_viewer_draw_uses_snapper_only_for_created_bbox PASSED")


def test_image_viewer_ctrl_alt_wheel_scrolls_axes_without_zooming():
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QWheelEvent

    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.resize(160, 120)
    viewer.set_image_from_qimage(QImage(400, 300, QImage.Format.Format_RGB888))
    viewer.scale(4, 4)
    initial_scale = viewer.transform().m11()

    hbar = viewer.horizontalScrollBar()
    vbar = viewer.verticalScrollBar()
    hbar.setValue(min(80, hbar.maximum()))
    vbar.setValue(min(80, vbar.maximum()))
    h_before = hbar.value()
    v_before = vbar.value()

    viewer.wheelEvent(QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    ))
    assert hbar.value() != h_before
    assert vbar.value() == v_before
    assert viewer.transform().m11() == initial_scale

    v_before = vbar.value()
    viewer.wheelEvent(QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    ))
    assert vbar.value() != v_before
    assert viewer.transform().m11() == initial_scale
    viewer.close()

    print("test_image_viewer_ctrl_alt_wheel_scrolls_axes_without_zooming PASSED")


def test_image_viewer_right_drag_selects_blocks_without_creating_bbox():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent

    from app.models import BBox, Block, BlockType
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.resize(300, 240)
    viewer.set_image_from_qimage(QImage(120, 80, QImage.Format.Format_RGB888))
    first = Block(block_type=BlockType.EQUATION, bbox=BBox(10, 10, 20, 20))
    second = Block(block_type=BlockType.TABLE, bbox=BBox(45, 10, 20, 20))
    third = Block(block_type=BlockType.TEXT, bbox=BBox(78, 10, 20, 20))
    viewer.show_blocks([first, second, third])
    created = []
    viewer.block_created.connect(lambda bbox: created.append(bbox))

    start = QPointF(viewer.mapFromScene(QPointF(5, 5)))
    end = QPointF(viewer.mapFromScene(QPointF(105, 40)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove,
        end,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        end,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    ))

    assert created == []
    assert viewer.selected_blocks() == [first, second, third]
    viewer.close()

    print("test_image_viewer_right_drag_selects_blocks_without_creating_bbox PASSED")


def test_image_viewer_frame_selection_ignores_box_interior():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent

    from app.models import BBox, Block, BlockType
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    viewer = ImageViewer()
    viewer.resize(300, 240)
    viewer.set_image_from_qimage(QImage(160, 120, QImage.Format.Format_RGB888))
    block = Block(block_type=BlockType.TEXT, bbox=BBox(20, 20, 100, 70))
    viewer.show_blocks([block])

    inside = QPointF(viewer.mapFromScene(QPointF(70, 55)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        inside,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        inside,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert viewer.selected_blocks() == []

    start = QPointF(viewer.mapFromScene(QPointF(60, 45)))
    end = QPointF(viewer.mapFromScene(QPointF(80, 65)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove,
        end,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        end,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert viewer.selected_blocks() == []

    edge_start = QPointF(viewer.mapFromScene(QPointF(18, 18)))
    edge_end = QPointF(viewer.mapFromScene(QPointF(42, 28)))
    viewer.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress,
        edge_start,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove,
        edge_end,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    viewer.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        edge_end,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert viewer.selected_blocks() == [block]
    viewer.close()

    print("test_image_viewer_frame_selection_ignores_box_interior PASSED")


def test_ui_block_labels_use_structured_semantic_label():
    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockOrigin, BlockType, Line, Page
    from app.ui.recognize.ocr_panel import OcrPanel
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(5, 6, 30, 20),
        lines=[Line(text="标题", confidence=0.9, bbox=BBox(5, 6, 30, 10))],
        origin=BlockOrigin(source_label="paragraph_title"),
    )

    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(80, 60, QImage.Format.Format_RGB888))
    viewer.show_blocks([block])
    assert "paragraph_title" in viewer._block_items[0][0].toolTip()
    viewer.close()

    panel = OcrPanel()
    panel.on_recognition_complete([Page(image_path="/tmp/ui-label.png", width=80, height=60, blocks=[block])])
    page_item = panel._tree.topLevelItem(0)
    assert page_item.child(0).text(0) == "[text · paragraph_title]"
    panel.close()

    print("test_ui_block_labels_use_structured_semantic_label PASSED")


def test_ocr_panel_reads_block_order_from_layout_snapshot_view():
    from PySide6.QtCore import Qt

    from app.models import (
        BBox, Block, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot,
        Line, OcrPolicy, Page,
    )
    from app.models.layout_snapshot_store import set_layout_snapshot_for_page
    from app.ui.recognize.ocr_panel import OcrPanel

    _get_qapp()
    body = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(5, 6, 30, 20),
        lines=[Line(text="正文", confidence=0.9, bbox=BBox(5, 6, 30, 10))],
        order=0,
        source_label="text",
    )
    title = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(5, 30, 40, 20),
        lines=[Line(text="标题", confidence=0.9, bbox=BBox(5, 30, 40, 10))],
        order=1,
        source_label="text",
    )
    page = Page(image_path="/tmp/ocr-panel-snapshot-order.png", width=80, height=60)
    page.blocks = [body, title]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-ocr-panel",
        source_engine="test",
        source_run_id="run-ocr-panel",
        blocks=(
            LayoutBlockSnapshot(
                uid=title.uid,
                block_type=BlockType.TITLE,
                bbox=title.bbox,
                order=0,
                source_label="paragraph_title",
                origin=BlockOrigin(source_label="paragraph_title"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
            LayoutBlockSnapshot(
                uid=body.uid,
                block_type=BlockType.TEXT,
                bbox=body.bbox,
                order=1,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
        ),
    ))

    panel = OcrPanel()
    panel.on_recognition_complete([page])
    page_item = panel._tree.topLevelItem(0)

    assert page_item.child(0).data(0, Qt.ItemDataRole.UserRole) is title
    assert page_item.child(0).text(0) == "[title · paragraph_title]"
    assert page_item.child(1).data(0, Qt.ItemDataRole.UserRole) is body
    panel.close()

    print("test_ocr_panel_reads_block_order_from_layout_snapshot_view PASSED")


def test_hanwang_concurrency_evaluation_script_help():
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hanwang_page_concurrency.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "bounded page-level concurrency" in result.stdout
    assert "--workers" in result.stdout

    print("test_hanwang_concurrency_evaluation_script_help PASSED")


def test_hanwang_micro_recblock_benchmark_loads_v16_response_wrapper():
    from scripts.benchmark_hanwang_micro_recblock import _load_ppvl_blocks

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "page.layout-api.json"
        path.write_text(
            json.dumps({
                "page": {"width": 100, "height": 200},
                "response": {
                    "result": {
                        "layoutParsingResults": [
                            {
                                "prunedResult": {
                                    "parsing_res_list": [
                                        {
                                            "block_label": "text",
                                            "block_bbox": [1, 2, 30, 40],
                                            "block_content": "正文",
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        records = _load_ppvl_blocks(path, 100, 200)

    assert records == [{
        "block_label": "text",
        "block_bbox": [1, 2, 30, 40],
        "block_content": "正文",
    }]

    print("test_hanwang_micro_recblock_benchmark_loads_v16_response_wrapper PASSED")


def test_paddle_v16_jobs_benchmark_script_help():
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_paddle_v16_jobs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PaddleOCR-VL jobs API timing" in result.stdout
    assert "--poll-interval" in result.stdout
    assert "--file-field" in result.stdout
    assert "--optional-payload-json" in result.stdout
    assert "--batch-id" in result.stdout
    assert "--lean-output" in result.stdout
    assert "--use-env-proxy" in result.stdout
    assert "--separate-jobs" in result.stdout
    assert "--workers" in result.stdout

    print("test_paddle_v16_jobs_benchmark_script_help PASSED")


def test_ppocr_v5_v6_compare_script_help():
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_ppocr_v5_v6.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PP-OCRv5" in result.stdout
    assert "PP-OCRv6" in result.stdout
    assert "--poll-timeout" in result.stdout

    print("test_ppocr_v5_v6_compare_script_help PASSED")


if __name__ == "__main__":
    test_raw_block_payload_prefers_page_artifact_origin_record()
    test_models()
    test_workflow_state_keeps_project_and_page_ocr_state_separate()
    test_bbox_tools()
    test_block_type_mapping()
    test_block_state_helpers_use_typed_state_only()
    test_paddle_layout_schema_normalizes_record_fields()
    test_ocr_run_wraps_ir_lines_without_proof_model()
    test_block_origin_label_is_authoritative_for_attributes_and_dispatch()
    test_project_store()
    test_project_store_persists_raw_layout_artifact()
    test_project_store_persists_typed_paddle_binding()
    test_project_store_persists_block_ocr_invalidation()
    test_project_store_persists_inline_formula_origin()
    test_project_store_persists_ocr_audit()
    test_project_store_persists_table_text_layer_cells()
    test_project_store_persists_block_origin_separately_from_current_layout()
    test_project_store_persists_layout_edit_events()
    test_project_store_current_schema_omits_retired_block_payload_columns()
    test_project_store_migration_drops_empty_retired_block_payload_columns()
    test_project_store_migration_rejects_nonempty_retired_block_payload_columns()
    test_project_store_rejects_invalid_review_flags_json_on_load()
    test_model_validation_rejects_legacy_page_and_block_payload_attr()
    test_project_store_rejects_legacy_page_model_on_save()
    test_line_final_text_contract_and_project_store_roundtrip()
    test_project_store_preserves_empty_final_text_roundtrip()
    test_project_store_clean_on_resave()
    test_project_store_save_project_preserves_child_rowids()
    test_project_store_upsert_rejects_foreign_parent_rowids()
    test_project_store_cross_project_uid_collision_remints_without_stealing()
    test_project_store_cross_project_uid_pollution_preserves_valid_rowids()
    test_project_store_uid_recovers_same_parent_stale_rowid()
    test_project_store_duplicate_sibling_uids_are_reminted()
    test_project_store_cross_parent_moves_preserve_uids_regardless_of_save_order()
    test_project_store_persists_page_ocr_invalidation_reason()
    test_project_store_update_proof_lines_rolls_back_as_single_transaction()
    test_project_store_update_proof_lines_requires_stable_uid_match()
    test_project_store_new_db_records_current_schema_version()
    test_proof_auto_flag_service()
    test_export_txt()
    test_txt_dual_encoding_outputs_and_layout_contract()
    test_export_xml()
    test_export_html()
    test_export_markdown_structure()
    test_markdown_fallback_assets_are_cropped_regions()
    test_markdown_export_settings_filter_and_merge_layout_fragments()
    test_markdown_filter_ignores_raw_payload_labels()
    test_export_formats_share_structured_blocks()
    test_export_ir_rules_load_and_validate()
    test_project_to_export_ir_builder_maps_final_text_and_fallbacks()
    test_export_ir_keeps_render_paths_only_for_rendering_formats(Path(tempfile.mkdtemp()))
    test_export_ir_char_source_fallbacks_are_unique_across_lines()
    test_export_ir_preserves_structured_block_attributes()
    test_pdf_page_faithful_plans_use_image_and_char_layer()
    test_pdf_dual_textless_page_degrades_without_text_font()
    test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_region_fonts()
    test_pdf_dual_inline_formula_item_does_not_shift_text_baseline()
    test_pdf_dual_text_bbox_ratio_can_be_profile_tuned()
    test_pdf_dual_text_layer_splits_inline_formula_as_atomic_span()
    test_pdf_dual_skips_duplicate_inline_formula_equation_element()
    test_pdf_dual_dedup_ignores_raw_payload_inline_formula_label()
    test_pdf_dual_equation_text_collapses_identical_formula_repeat()
    test_pdf_dual_keeps_display_equation_even_if_it_overlaps_inline_formula_bbox()
    test_pdf_dual_table_text_layer_uses_atomic_rows()
    test_pdf_dual_html_table_text_layer_splits_cells_without_tags()
    test_pdf_dual_generated_table_rows_stay_inside_table_lines()
    test_pdf_dual_generated_html_table_cells_stay_inside_cells()
    test_pdf_dual_html_table_cells_prefer_image_text_clusters_over_equal_grid()
    test_table_text_layer_service_writes_hidden_cells_for_table_block()
    test_pdf_dual_table_cells_use_ocr_stage_payload_before_image_inference()
    test_pdf_dual_generated_formula_text_bbox_stays_inside_formula_block()
    test_pdf_dual_generated_pdf_deduplicates_inline_formula_equation_text()
    test_pdf_invisible_text_layer_resets_render_mode_on_font_size_error()
    test_ir_based_exporters_and_pdf_profiles()
    test_export_dialog_offers_markdown()
    test_export_default_styles_map_to_html_docx_and_pdf()
    test_rich_reflow_contract_shared_by_html_docx_and_rtf()
    test_export_worker_reports_completion_progress()
    test_export_filename_sanitizes_invalid_project_name()
    test_export_worker_sanitizes_project_name_for_all_formats()
    test_export_worker_keeps_formats_independent_when_one_fails()
    test_export_dialog_reports_partial_success_without_critical_error()
    test_export_dialog_surfaces_output_path_failure_from_real_worker()
    test_layout_panel_analysis_progress_lifecycle()
    test_layout_panel_workbench_height_is_not_forced_by_sidebar()
    test_layout_panel_splitter_keeps_sidebar_width_on_large_workbench()
    test_layout_panel_right_sidebar_uses_project_stats_without_selection_inspector()
    test_layout_panel_auto_text_blocks_are_editable_frames()
    test_layout_panel_pageup_pagedown_shortcuts_change_page()
    test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle()
    test_layout_panel_draw_merge_uses_large_box_and_removes_overlap()
    test_layout_panel_draw_inside_text_frame_does_not_merge_parent()
    test_layout_panel_formula_draw_touching_text_frame_stays_separate()
    test_layout_panel_text_draw_does_not_absorb_inline_formula_block()
    test_layout_panel_drawn_block_is_selected_and_type_editable()
    test_layout_panel_draw_snaps_to_image_ink_without_existing_blocks()
    test_layout_panel_ink_snap_reuses_cached_image_mask()
    test_layout_panel_delete_selected_removes_unlocked_box()
    test_layout_panel_readonly_char_boxes_do_not_block_formula_delete()
    test_layout_panel_hides_empty_and_invalidated_char_boxes()
    test_layout_panel_excludes_inline_formula_carriers_from_char_boxes()
    test_layout_panel_type_buttons_change_unlocked_block_type()
    test_layout_panel_type_buttons_are_grouped()
    test_layout_panel_subtype_buttons_write_paddle_source_label()
    test_layout_panel_search_results_can_batch_apply_heading_level()
    test_layout_panel_search_batch_apply_undo_restores_all_pages()
    test_layout_panel_find_dialog_preset_matches_chinese_heading_forms()
    test_layout_panel_undo_restores_block_edits()
    test_layout_panel_undo_preserves_view_transform()
    test_layout_panel_promotes_real_inline_formula_overlays_to_editable_blocks()
    test_layout_panel_moved_generated_inline_formula_keeps_manual_geometry()
    test_layout_panel_skips_superscript_marker_inline_formula_overlays_from_120169()
    test_workflow_controller_layout_progress_signal()
    test_workflow_controller_abstracts_internal_ocr_progress_messages()
    test_workflow_controller_clamps_ocr_page_concurrency_to_page_count()
    test_main_window_layout_error_is_status_only()
    test_layout_panel_status_label_elides_long_errors()
    test_main_window_centered_resize_expands_from_current_center()
    test_main_window_initial_import_window_is_screen_centered()
    test_main_window_maximize_state_is_not_forced_back_to_normal()
    test_main_window_file_menu_uses_close_project_action()
    test_main_window_close_project_prompts_save_and_resets_workspace()
    test_main_window_close_project_save_uses_save_as_for_transient_project()
    test_fake_ocr_engine()
    test_create_engine_hanwang_exposes_page_block_capability()
    test_confidence_normalization()
    test_api_ocr_engine_does_not_request_return_word_box()
    test_api_ocr_engine_does_not_promote_block_content_to_line()
    test_api_ocr_engine_reads_direct_pruned_ppocr_rows()
    test_api_ocr_engine_ignores_block_content_without_rec_rows()
    test_api_ocr_engine_preserves_rec_text_without_any_geometry()
    test_ocr_pipeline()
    test_ocr_pipeline_keeps_page_relative_boxes()
    test_ocr_pipeline_offsets_crop_relative_boxes()
    test_ocr_pipeline_prefers_engine_char_boxes_and_only_falls_back_for_missing_chars()
    test_ocr_pipeline_normalizes_proof_geometry()
    test_ocr_pipeline_process_block_normalizes_proof_geometry()
    test_ocr_pipeline_emits_nonblocking_warning_when_proof_fallback_triggers()
    test_ocr_pipeline_reports_real_page_progress()
    test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once()
    test_ocr_dispatch_policy_blocks_structural_and_paddle_skip_labels()
    test_page_ocr_refills_caption_blocks_and_preserves_equation_blocks()
    test_page_ocr_nested_equation_blocks_before_parent_text_assignment()
    test_hanwang_prepass_keeps_line_hint_overlapping_nested_formula_block()
    test_ocr_pipeline_skips_equation_block_ocr_when_policy_preserves_formula()
    test_ocr_pipeline_avoids_double_shift_for_page_space_boxes()
    test_ocr_pipeline_preserves_hanwang_crop_lines_and_chars()
    test_hanwang_micro_recblock_routes_and_fallbacks()
    test_hanwang_engine_uses_user_edited_layout_for_manual_formula_boxes()
    test_hanwang_layout_injects_manual_formula_binding_into_parent_route()
    test_hanwang_inline_formula_text_slices_keep_chars()
    test_hanwang_recog_filters_empty_decoded_char_boxes()
    test_hanwang_inline_formula_carrier_survives_model_and_proof_helpers()
    test_hanwang_pre_page_ocr_lines_split_before_recog()
    test_hanwang_route_assembly_recovers_tiny_punctuation_after_formula()
    test_hanwang_micro_recblock_sends_inter_formula_punctuation_gap_to_segimg()
    test_hanwang_micro_recblock_drops_stale_cached_layout_routes_without_page_hints()
    test_hanwang_bbox_audit_distinguishes_layout_route_and_recog_boxes()
    test_hanwang_micro_recblock_short_chinese_group_keeps_vertical_context()
    test_ocr_pipeline_hybrid_prepass_lines_feed_hanwang_splitter()
    test_paddle_line_routing_builds_layout_line_routes_from_reading_order()
    test_paddle_line_routing_display_formula_span_is_not_split_to_empty_pair()
    test_paddle_line_routing_line_hint_formula_recovery_does_not_shift_next_line()
    test_paddle_line_routing_missing_formula_box_does_not_shift_later_rows()
    test_paddle_line_routing_marker_formula_does_not_cut_text_slice()
    test_paddle_line_routing_cached_marker_formula_routes_are_rebuilt()
    test_paddle_line_routing_marker_formula_from_120169_does_not_eat_zero()
    test_paddle_line_routing_ppocr_prefiltered_marker_keeps_later_formula_text()
    test_paddle_line_routing_complete_formula_geometry_ignores_ppocr_text()
    test_paddle_line_routing_page_ocr_miss_invalidates_cached_routes()
    test_paddle_line_routing_restores_inter_formula_punctuation_gap()
    test_paddle_line_routing_formula_number_is_skip_not_formula_carrier()
    test_paddle_line_routing_formula_rows_use_single_horizontal_band()
    test_paddle_line_routing_has_layout_routes_is_pure()
    test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands()
    test_layout_fixture_page_ocr_routes_do_not_shift_after_missing_formula_box()
    test_paddle_artifact_index_marks_missing_inline_formula_for_crop_ocr()
    test_paddle_artifact_index_binds_parent_table_and_empty_formula_review()
    test_layout_panel_manual_formula_writes_typed_paddle_binding()
    test_layout_analyzer_reads_formula_geometry_boxes_for_routes()
    test_hanwang_inline_formula_empty_text_slices_keeps_empty_hanwang_result()
    test_hanwang_group_chunk_cannot_readmit_skipped_subregions()
    test_hanwang_formula_style_footer_bypasses_hanwang()
    test_hanwang_footnote_labels_route_through_hanwang()
    test_hanwang_micro_recblock_unknown_label_defaults_to_text_path_with_audit()
    test_hanwang_recog_group_failure_is_visible_in_audit_without_ppvl_fallback()
    test_hanwang_recog_group_failure_retries_with_top_trim_before_dropping_line()
    test_hanwang_micro_recblock_circuit_breaks_after_batch_failure()
    test_hanwang_micro_recblock_batch_list_handles_wide_crops_without_collage_guard()
    test_ocr_pipeline_runs_hanwang_micro_recblock_page_path()
    test_hanwang_page_blocks_from_layout_preserves_raw_source_label()
    test_hanwang_current_layout_blocks_for_ocr_uses_current_blocks_not_ppvl_source()
    test_hanwang_page_blocks_from_layout_does_not_promote_internal_merge_note_to_formula_text()
    test_hanwang_ppvl_skip_uses_layout_authority_label()
    test_ocr_pipeline_records_failed_page_when_block_ocr_fails()
    test_workflow_controller_auto_chains_ocr_after_layout()
    test_workflow_controller_hanwang_layout_stays_on_block_ocr_path()
    test_workflow_controller_hanwang_ocr_entry_redirects_to_first_pending_page()
    test_workflow_controller_hanwang_main_entry_starts_all_actionable_pages()
    test_workflow_controller_hanwang_layout_submit_starts_ocr_when_ready()
    test_workflow_controller_hanwang_retries_ocr_error_page_from_main_entry()
    test_workflow_controller_hanwang_layout_submit_retries_ocr_error_page()
    test_workflow_controller_hanwang_layout_submit_merges_only_target_page()
    test_workflow_controller_hanwang_block_edit_invalidates_only_that_page()
    test_workflow_controller_hanwang_no_pending_reports_all_done_without_redirect()
    test_workflow_controller_starts_parallel_proof_ocr_with_layout()
    test_workflow_controller_parallel_proof_skips_missing_page_without_misalignment()
    test_workflow_controller_keeps_qthreads_until_finished_after_error()
    test_workflow_controller_falls_back_to_block_ocr_when_parallel_proof_failed()
    test_workflow_controller_emits_ocr_progress_and_navigation()
    test_workflow_controller_ocr_done_does_not_force_hproof_step()
    test_workflow_controller_ocr_done_keeps_error_status_even_with_prepass_lines()
    test_main_window_ocr_finished_preserves_current_step()
    test_workflow_controller_normalizes_loaded_project_geometry()
    test_export_service()
    test_import_service()
    test_import_service_sequential_page_numbers()
    test_pdf_import_cache_name_includes_source_path_hash(Path(tempfile.mkdtemp()))
    test_import_cache_name_changes_when_same_path_content_changes(Path(tempfile.mkdtemp()))
    test_proof_state_bus()
    test_proof_state_bus_typed_contracts()
    test_workflow_controller_emits_typed_view_state()
    test_api_settings_dialog_keeps_model_preset_sync()
    test_api_settings_dialog_reverse_matches_url_and_persists_profile()
    test_api_settings_dialog_saves_base_url_from_endpoint_suffix()
    test_api_settings_dialog_migrates_legacy_official_layout_url()
    test_api_settings_dialog_collapses_mode_to_hanwang_when_saving()
    test_api_settings_dialog_persists_hanwang_mode_with_api_runtime()
    test_api_model_profile_helpers()
    test_api_endpoint_role_resolution_keeps_layout_and_proof_separate()
    test_api_http_post_json_disables_environment_proxies()
    test_fixed_api_chain_resolves_official_roots_to_vl16_and_ppocrv5()
    test_api_ocr_engine_resolves_ocr_endpoint_for_pp_ocrv5_profile()
    test_api_ocr_engine_parses_paddle_coordinate_variants()
    test_api_request_builders_split_profile_params()
    test_layout_analyzer_rescales_suspicious_blocks()
    test_layout_analyzer_extracts_api_polygon_bbox()
    test_layout_analyzer_extracts_api_blocks_from_v16_schema()
    test_paddle_authority_prefers_block_label_over_conflicting_label_everywhere()
    test_layout_parsing_semantics_override_layout_det_when_both_exist()
    test_layout_analyzer_forwards_route_subblocks_from_layout_det_res()
    test_hanwang_layout_row_uses_page_artifact_origin_record()
    test_layout_analyzer_persists_raw_parsing_res_list()
    test_layout_analyzer_does_not_promote_ocr_results_to_layout_blocks()
    test_layout_analyzer_uses_datainfo_canvas_scale()
    test_layout_analyzer_ignores_conflicting_datainfo_when_bbox_is_page_space()
    test_layout_analyzer_ignores_conflicting_pruned_shape_when_bbox_is_page_space()
    test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected()
    test_paddle_v16_submit_error_includes_response_body()
    test_paddle_v16_client_supports_env_proxy_and_batch_id()
    test_layout_analyzer_routes_hanwang_mode_to_ppvl_layout()
    test_hanwang_assets_env_accepts_bin_dir()
    test_hanwang_native_bridge_writes_multi_recblocks()
    test_layout_worker_runs_api_pages_with_bounded_concurrency()
    test_layout_worker_keeps_local_pages_serial_even_with_concurrency_config()
    test_layout_worker_continues_after_single_page_failure()
    test_workflow_controller_marks_partial_layout_failures_without_blocking_success_pages()
    test_workflow_controller_enables_proof_steps_after_first_ocr_page()
    test_proof_line_iterator_excludes_equation_lines()
    test_hproof_line_iterator_uses_shared_proof_text_elements()
    test_proof_line_iterators_exclude_route_table_lines()
    test_hproof_line_iterator_excludes_position_source_labels()
    test_block_attributes_use_origin_or_current_label_not_raw_payload()
    test_hproof_page_filter_keeps_pages_separate()
    test_hproof_page_filter_flushes_dirty_editor_before_switching_pages()
    test_hproof_page_filter_blocks_switch_when_current_editor_has_conflict()
    test_hproof_merge_pages_preserves_active_editor_text()
    test_hproof_merge_rebinds_replaced_lines_without_duplicates_or_orphans()
    test_hproof_merge_rebind_marks_conflict_when_dirty_editor_meets_new_model_text()
    test_hproof_merge_uses_stable_uid_when_geometry_changes()
    test_hproof_merge_pages_removes_orphan_rows_absent_from_new_pages()
    test_hproof_orphan_merge_restore_dirty_text_marks_conflict_when_model_changed()
    test_hproof_debug_filter_blocks_rebuild_when_current_editor_has_conflict()
    test_hproof_save_all_emits_only_when_current_line_is_saved()
    test_hproof_save_all_persists_probe_only_correction()
    test_hproof_synthetic_debug_line_is_readonly_and_not_persisted()
    test_vproof_merge_pages_reloads_current_page_reference_text()
    test_vproof_merge_pages_keeps_current_page_by_uid_when_order_changes()
    test_vproof_target_edit_persists_probe_only_correction()
    test_vproof_refresh_reference_context_persists_stale_probe_anchor_correction()
    test_vproof_replacement_commits_text_and_probe_together()
    test_vproof_undo_redo_uses_line_edit_actions()
    test_vproof_external_refresh_invalidates_undo_history_before_restore()
    test_vproof_save_refreshes_current_page_index_incrementally()
    test_char_index_service_replace_pages_preserves_other_page_entries()
    test_top_bar_hosts_workflow_steps_and_layout_run()
    test_vproof_gallery_uses_wrapping_white_grid()
    test_vproof_text_highlight_targets_single_entry()
    test_vproof_highlight_can_repeat_without_losing_state()
    test_vproof_highlight_survives_repeated_page_switches()
    test_vproof_gallery_keyboard_selection_refreshes_linked_panels()
    test_vproof_candidate_button_applies_to_ocr_text()
    test_vproof_right_click_edit_bubble_batches_and_undoes()
    test_vproof_undo_redo_restores_line_model_after_single_and_batch_edits()
    test_vproof_save_rejects_stale_text_editor_page_binding()
    test_vproof_merge_pages_blocks_dirty_editor_when_current_page_object_replaced()
    test_vproof_undo_redo_reject_stale_actions_after_current_page_replaced()
    test_vproof_save_rejects_stale_text_map_even_when_page_key_matches()
    test_hproof_visual_size_is_compact()
    test_image_viewer_char_boxes_update_char_bbox()
    test_image_viewer_space_pan_temporarily_disables_box_editing()
    test_image_viewer_shift_left_drag_creates_block_bbox()
    test_image_viewer_draw_uses_snapper_only_for_created_bbox()
    test_image_viewer_ctrl_alt_wheel_scrolls_axes_without_zooming()
    test_image_viewer_right_drag_selects_blocks_without_creating_bbox()
    test_image_viewer_frame_selection_ignores_box_interior()
    test_ui_block_labels_use_structured_semantic_label()
    test_hanwang_concurrency_evaluation_script_help()
    test_hanwang_micro_recblock_benchmark_loads_v16_response_wrapper()
    test_paddle_v16_jobs_benchmark_script_help()
    test_ppocr_v5_v6_compare_script_help()
    test_char_index_vertical_split()
    test_char_index_horizontal_split()
    test_char_index_hides_fallback_units_by_default()
    test_char_index_dedup_on_rebuild()
    test_char_index_skips_grossly_mismatched_geometry_instead_of_showing_wrong_crops()
    test_char_index_skips_two_char_full_mismatch_geometry()
    test_char_index_skips_existing_chars_when_display_length_changed()
    test_char_index_sort_categories()
    test_char_index_query_stable_order()
    test_char_index_skips_whitespace()
    test_char_index_indexes_digit_runs_as_single_digits()
    test_char_index_does_not_merge_plain_digit_with_circled_marker()
    test_char_index_filters_non_cjk_from_default_vproof()
    test_char_index_sorts_digit_characters_as_buckets()
    test_char_index_exposes_latin_formula_like_runs_as_chars()
    test_char_index_suppresses_punctuation_topic_for_shared_token_bbox()
    test_char_index_uses_token_collection_for_word_level_han_bbox()
    test_char_index_skips_empty_narrow_ocr_bbox()
    test_char_index_skips_lines_with_unverified_geometry()
    test_char_index_deduplicates_overlapping_duplicate_lines()
    test_vproof_reference_context_deduplicates_overlapping_duplicate_lines()
    print("\n✓ 所有测试通过")
