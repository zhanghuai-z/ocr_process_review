"""基础单元测试：无需 OCR 引擎，不启动 GUI。

覆盖：
- 模型基础行为（含新增字段）
- BBox 工具函数
- ProjectStore 保存/加载/迁移/脏数据清理
- ProofEngine 低置信标记
- TXT/XML/HTML 导出
- Fake OCR/Layout/LLM 引擎
- OcrPipeline
- ExportService
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 模型测试
# =====================================================================

def test_models():
    from app.models import (
        BBox, Block, BlockSource, BlockType, Char, Line,
        LlmReviewStatus, OcrProject, Page, PageStatus, ProofStatus,
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
    assert line.proof_status == ProofStatus.UNCHECKED
    assert line.llm_review_status == LlmReviewStatus.DISABLED
    assert line.review_flags == []
    line.update_text("修改文字")
    assert line.proof_status == ProofStatus.MODIFIED
    assert line.original_text == "测试文字"

    # Line with new fields
    line2 = Line(
        text="终稿", confidence=0.85, bbox=bb,
        ocr_text="OCR原文", llm_suggestion="LLM建议",
        llm_review_status=LlmReviewStatus.DONE,
        review_flags=["low_confidence"],
    )
    assert line2.ocr_text == "OCR原文"
    assert line2.llm_suggestion == "LLM建议"

    char = Char(char="测", confidence=0.9, bbox=bb)
    assert char.uid.startswith("char_")

    # Block
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line, line2])
    assert block.uid.startswith("block_")
    assert block.full_text == "修改文字\n终稿"
    assert block.source == BlockSource.AUTO_LAYOUT
    assert block.recognizable is True

    # Block new fields
    block2 = Block(
        block_type=BlockType.TABLE, bbox=bb,
        source=BlockSource.MANUAL_DRAW, is_locked=True,
        recognizable=False, note="测试备注",
    )
    assert block2.source == BlockSource.MANUAL_DRAW
    assert block2.is_locked is True
    assert block2.recognizable is False
    assert block2.note == "测试备注"

    # Page
    page = Page(image_path="/tmp/test.jpg", width=800, height=1200)
    assert page.uid.startswith("page_")
    page.blocks.append(block)
    formula_block = Block(block_type=BlockType.EQUATION, bbox=bb, recognizable=True)
    page.blocks.append(formula_block)
    assert page.is_analyzed
    assert page.status == PageStatus.IMPORTED
    assert page.source_type == "image"
    assert page.text_ocr_blocks == [block]

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
    assert project.total_lines == 2
    assert project.has_any_ocr_result is True
    assert project.all_pages_ocr_done is False
    page.status = PageStatus.OCR_DONE
    assert page.has_ocr_result is True
    assert page.is_ocr_done is True
    page.status = PageStatus.PROOFING
    assert page.is_ocr_done is True
    assert project.all_pages_ocr_done is True
    page.invalidate_ocr("block_moved")
    assert page.needs_ocr_rerun is True
    assert page.ocr_invalidated_reason == "block_moved"
    page.clear_ocr_invalidation()
    assert page.needs_ocr_rerun is False

    # Export summary
    summary = project.get_export_summary()
    assert summary["total_pages"] == 1
    assert summary["total_lines"] >= 0
    assert summary["unrecognized_blocks"] == 0
    assert project.has_unrecognized_blocks is False

    print("test_models PASSED")


def test_workflow_state_keeps_project_and_page_ocr_state_separate():
    from app.core.workflow_state import (
        STEP_OCR,
        STEP_VPROOF,
        compute_max_step,
        page_gate_info,
        pending_ocr_pages,
    )
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

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

    assert project.has_any_ocr_result is True
    assert project.all_pages_ocr_done is False
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
    assert lines_without_done_status.has_ocr_result is True
    assert lines_without_done_status.is_ocr_done is False
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

    pending_page.invalidate_ocr("block_moved")
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


def test_component_matcher_extracts_and_classifies_cjk_tokens():
    import cv2
    import numpy as np

    from app.core.component_matcher import (
        analyze_token_components, build_component_shape_filter,
        build_relative_component_shape_filter, count_cjk_tokens, extract_text_components,
    )
    from app.models import BBox

    img = np.full((80, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (12, 22), (32, 54), (0, 0, 0), -1)
    cv2.rectangle(img, (58, 22), (78, 54), (0, 0, 0), -1)
    cv2.rectangle(img, (104, 22), (124, 54), (0, 0, 0), -1)

    components = extract_text_components(
        img,
        BBox(0, 0, 150, 70),
        kernel_size=None,
        min_area=20,
    )
    assert [component.bbox for component in components] == [
        BBox(12, 22, 21, 33),
        BBox(58, 22, 21, 33),
        BBox(104, 22, 21, 33),
    ]

    single = analyze_token_components(img, "汉", BBox(8, 18, 30, 42), kernel_size=None)
    assert single.status == "single_cjk"
    assert single.component_count == 1

    multi = analyze_token_components(img, "天地", BBox(50, 18, 82, 42), kernel_size=None)
    assert multi.status == "split_candidate"
    assert multi.component_count == 2

    latin = analyze_token_components(img, "A1", BBox(50, 18, 82, 42), kernel_size=None)
    assert latin.status == "not_cjk"

    assert count_cjk_tokens(["天", "地玄", "2026", "A1"]) == (1, 1, 2)

    shape_filter = build_component_shape_filter(components)
    assert all(shape_filter.accepts(component) for component in components)
    assert not shape_filter.accepts(type(components[0])(BBox(2, 2, 3, 40), 30))

    relative_filter = build_relative_component_shape_filter(
        [(component, 40.0) for component in components]
    )
    assert all(relative_filter.accepts(component, 40.0) for component in components)
    assert not relative_filter.accepts(type(components[0])(BBox(2, 2, 3, 40), 30), 40.0)

    print("test_component_matcher_extracts_and_classifies_cjk_tokens PASSED")


def test_block_type_mapping():
    from app.models import BlockType

    assert BlockType.from_paddle("paragraph") == BlockType.TEXT
    assert BlockType.from_paddle("doc_title") == BlockType.TITLE
    assert BlockType.from_paddle("section_title") == BlockType.TITLE
    assert BlockType.from_paddle("image_caption") == BlockType.FIGURE_CAPTION
    assert BlockType.from_paddle("table_caption_text") == BlockType.TABLE_CAPTION
    assert BlockType.from_paddle("table_body") == BlockType.TABLE
    assert BlockType.from_paddle("graphic") == BlockType.FIGURE
    assert BlockType.from_paddle("isolated_formula") == BlockType.EQUATION
    assert BlockType.from_paddle("bibliography") == BlockType.REFERENCE
    assert BlockType.from_paddle("vision_footnote") == BlockType.TEXT

    print("test_block_type_mapping PASSED")


def test_block_payload_helpers_preserve_existing_entries():
    from app.core.block_payload import (
        OCR_INVALIDATION_KIND_KEY,
        OCR_TEXT_INVALIDATED_KEY,
        mark_ocr_text_invalidated,
        payload_bool,
        payload_get,
        set_payload_entries,
    )
    from app.models import BBox, Block, BlockType

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(0, 0, 10, 10),
        raw_payload={"vendor": {"keep": True}},
    )

    set_payload_entries(block, {"custom": 1})
    assert payload_get(block, "vendor") is None
    assert payload_get(block, "custom") == 1
    assert payload_bool(block, "custom") is True

    mark_ocr_text_invalidated(block, "block_moved")
    assert block.app_payload[OCR_TEXT_INVALIDATED_KEY] is True
    assert block.app_payload[OCR_INVALIDATION_KIND_KEY] == "block_moved"
    assert block.raw_payload["vendor"] == {"keep": True}

    print("test_block_payload_helpers_preserve_existing_entries PASSED")


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
        "block_score": "0.91",
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
    assert paddle_record_text({"markdown": " preview "}) == "preview"
    assert block_text({"markdown": "must not route"}) == ""

    print("test_paddle_layout_schema_normalizes_record_fields PASSED")


# =====================================================================
# ProjectStore 测试
# =====================================================================

def test_project_store():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
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


def test_project_store_persists_ppvl_parsing_res_list():
    from app.core.project_store import ProjectStore
    from app.models import BBox, Block, BlockType, OcrProject, Page

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        parsing_res_list = [
            {
                "block_label": "text",
                "block_bbox": [10, 20, 110, 60],
                "block_content": "天地玄黄",
            },
            {
                "block_label": "display_formula",
                "block_bbox": [20, 80, 180, 120],
                "block_content": "$$x+y$$",
            },
        ]
        project = OcrProject(
            name="ppvl-raw",
            pages=[
                Page(
                    image_path="/tmp/img.jpg",
                    width=800,
                    height=600,
                    ppvl_parsing_res_list=parsing_res_list,
                    blocks=[
                        Block(
                            block_type=BlockType.TEXT,
                            bbox=BBox(10, 20, 100, 40),
                            source_label="paragraph_title",
                            raw_payload={
                                "block_label": "paragraph_title",
                                "block_content": "属性保真",
                                "attributes": {"level": 2},
                            },
                        )
                    ],
                )
            ],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)

        assert loaded.pages[0].ppvl_parsing_res_list == parsing_res_list
        loaded_block = loaded.pages[0].blocks[0]
        assert loaded_block.source_label == "paragraph_title"
        assert loaded_block.raw_payload["attributes"]["level"] == 2

        print("test_project_store_persists_ppvl_parsing_res_list PASSED")
    finally:
        os.unlink(db_path)


def test_line_final_text_contract_and_project_store_roundtrip():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(text="OCR text", final_text="人工终稿", confidence=0.9, bbox=bb)
        assert line.text == "OCR text"
        assert line.final_text == "人工终稿"
        assert line.display_text == "人工终稿"
        line.text = "直接兼容写入"
        assert line.final_text == "人工终稿"
        line.ensure_text_contract()
        assert line.text == "直接兼容写入"
        assert line.final_text == "人工终稿"
        assert line.ocr_text == "OCR text"
        line.final_text = "最终真值"
        assert line.text == "直接兼容写入"
        line.final_text = ""
        assert line.display_text == "直接兼容写入"
        line.update_final_text("最终真值")

        line_without_text = Line(
            text="",
            ocr_text="OCR补全文本",
            confidence=0.8,
            bbox=bb,
            original_text="",
        )
        line_without_text.original_text = ""
        line_without_text.ensure_text_contract(fill_original=True)
        assert line_without_text.text == "OCR补全文本"
        assert line_without_text.final_text == "OCR补全文本"
        assert line_without_text.ocr_text == "OCR补全文本"
        assert line_without_text.original_text == "OCR补全文本"

        final_only = Line(text="", final_text="人工终稿", confidence=0.8, bbox=bb)
        final_only.ocr_text = ""
        final_only.original_text = ""
        final_only.ensure_text_contract(fill_original=True)
        assert final_only.final_text == "人工终稿"
        assert final_only.ocr_text == ""
        assert final_only.original_text == ""

        project = OcrProject(
            name="final_text",
            pages=[Page(image_path="/tmp/img.jpg", width=800, height=600,
                        blocks=[Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])])],
        )
        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)
            loaded_line = loaded.pages[0].blocks[0].lines[0]
            assert loaded_line.final_text == "最终真值"
            assert loaded_line.text == "直接兼容写入"

        print("test_line_final_text_contract_and_project_store_roundtrip PASSED")
    finally:
        os.unlink(db_path)


def test_project_store_clean_on_resave():
    """重新保存时旧 block 不残留。"""
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
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

            line1.update_text("第一行已校对")
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
        assert loaded_line.final_text == "第一行已校对"
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
            p2_line.update_text("乙已改")
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
        assert reloaded2.pages[0].blocks[0].lines[0].final_text == "乙已改"
        assert reloaded2.pages[0].blocks[0].lines[0].display_text == "乙已改"
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
            p2_line.update_text("乙已改")
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
        assert p1_loaded_line.display_text == "甲"
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
        assert p2_loaded_line.display_text == "乙已改"
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
            p2_line.update_text("乙已改")
            p2_char.char = "乙"

            store.save_project(project2)

            reloaded1 = store.load_project(project_id=project1.id)
            reloaded2 = store.load_project(project_id=project2.id)

        p1_loaded_page = reloaded1.pages[0]
        p1_loaded_block = p1_loaded_page.blocks[0]
        p1_loaded_line = p1_loaded_block.lines[0]
        p1_loaded_char = p1_loaded_line.chars[0]
        assert (p1_loaded_page.id, p1_loaded_block.id, p1_loaded_line.id, p1_loaded_char.id) == p1_ids
        assert p1_loaded_line.display_text == "甲"
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
        assert p2_loaded_line.display_text == "乙已改"
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
            line2.update_text("第二行已更新")
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
        assert loaded_line1.display_text == "第一行"
        assert loaded_line2.display_text == "第二行已更新"
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
        line2.final_text = "乙"
        line2.ocr_text = "乙"
        line2.chars[0].char = "乙"

        line_block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line1, line2])
        block_copy = copy.deepcopy(line_block)
        block_copy.order = 1
        block_copy.lines[0].text = "丙"
        block_copy.lines[0].final_text = "丙"
        block_copy.lines[0].ocr_text = "丙"
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
        assert [line.display_text for line in first_block.lines] == ["甲", "乙"]
        assert len({line.uid for line in first_block.lines}) == 2
        assert [line.display_text for line in second_block.lines] == ["丙", "乙"]
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
        page.invalidate_ocr("block_type_changed")
        project = OcrProject(name="invalidate", pages=[page])

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)

        assert loaded.pages[0].ocr_invalidated_reason == "block_type_changed"
        assert loaded.pages[0].needs_ocr_rerun is True
        assert loaded.pages[0].status == PageStatus.LAYOUT_DONE
    finally:
        os.unlink(db_path)

    print("test_project_store_persists_page_ocr_invalidation_reason PASSED")


def test_project_store_reconciles_legacy_ocr_status_from_lines():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        project = OcrProject(
            name="legacy status",
            pages=[
                Page(
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
            ],
        )

        with ProjectStore(db_path) as store:
            store.save_project(project)
            loaded = store.load_project(project_id=1)

        assert loaded.pages[0].has_ocr_result is True
        assert loaded.pages[0].status == PageStatus.OCR_DONE
        assert loaded.pages[0].is_ocr_done is True
    finally:
        os.unlink(db_path)

    print("test_project_store_reconciles_legacy_ocr_status_from_lines PASSED")


def test_project_store_update_lines_rolls_back_as_single_transaction():
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
            line1.update_text("第一行已改")
            line2.update_text("第二行已改")
            line2.id = -999999
            try:
                store.update_lines([line1, line2])
            except Exception:
                pass
            else:
                raise AssertionError("update_lines should fail on invalid line id")

            loaded = store.load_project(project_id=1)

        loaded_lines = loaded.pages[0].blocks[0].lines
        assert [line.final_text for line in loaded_lines] == ["第一行", "第二行"]
        assert [line.proof_status for line in loaded_lines] == [
            ProofStatus.UNCHECKED,
            ProofStatus.UNCHECKED,
        ]
    finally:
        os.unlink(db_path)

    print("test_project_store_update_lines_rolls_back_as_single_transaction PASSED")


def test_project_store_update_line_requires_stable_uid_match():
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
            line2.update_text("不应写入第一行")
            try:
                store.update_line(line2)
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_line should reject stale rowid with mismatched uid")

            loaded = store.load_project(project_id=project.id)
            loaded_lines = loaded.pages[0].blocks[0].lines
            assert loaded_lines[0].id == line1_id
            assert loaded_lines[0].uid == line1_uid
            assert loaded_lines[0].display_text == "第一行"
            assert loaded_lines[1].id == line2_id
            assert loaded_lines[1].uid == line2_uid
            assert loaded_lines[1].display_text == "第二行"

            line2.id = line1_id
            line2.uid = ""
            line2.update_text("仍不应写入第一行")
            try:
                store.update_line(line2)
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_line should reject missing uid even when rowid exists")

            loaded = store.load_project(project_id=project.id)
            loaded_lines = loaded.pages[0].blocks[0].lines
            assert loaded_lines[0].id == line1_id
            assert loaded_lines[0].uid == line1_uid
            assert loaded_lines[0].display_text == "第一行"
            assert loaded_lines[1].id == line2_id
            assert loaded_lines[1].uid == line2_uid
            assert loaded_lines[1].display_text == "第二行"

            line1.id = line1_id
            line1.uid = ""
            line1.update_text("第一行已改")
            try:
                store.update_line(line1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("update_line should reject missing uid on a valid rowid")

            line1.uid = line1_uid
            store.update_line(line1)
            loaded = store.load_project(project_id=project.id)

        assert loaded.pages[0].blocks[0].lines[0].id == line1_id
        assert loaded.pages[0].blocks[0].lines[0].uid == line1_uid
        assert loaded.pages[0].blocks[0].lines[0].display_text == "第一行已改"
    finally:
        os.unlink(db_path)

    print("test_project_store_update_line_requires_stable_uid_match PASSED")


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


def test_project_store_schema_migration():
    """从 v1 schema 迁移到当前版本。"""
    import sqlite3
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        # 创建 v1 风格的数据库
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS project (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS page (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL, width INTEGER NOT NULL,
                height INTEGER NOT NULL, page_number INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS block (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER NOT NULL REFERENCES page(id) ON DELETE CASCADE,
                block_type TEXT NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,
                w INTEGER NOT NULL, h INTEGER NOT NULL,
                block_order INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS line (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id INTEGER NOT NULL REFERENCES block(id) ON DELETE CASCADE,
                text TEXT NOT NULL DEFAULT '', original_text TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                proof_status TEXT NOT NULL DEFAULT 'unchecked',
                x INTEGER NOT NULL, y INTEGER NOT NULL,
                w INTEGER NOT NULL, h INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS char (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
                char TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.0,
                x INTEGER, y INTEGER, w INTEGER, h INTEGER
            );
            INSERT INTO project (id, name, created_at, updated_at) VALUES (1, 'legacy', 0, 0);
            INSERT INTO page (id, project_id, image_path, width, height, page_number)
                VALUES (1, 1, '/tmp/test.jpg', 800, 600, 1);
            INSERT INTO block (id, page_id, block_type, x, y, w, h, block_order)
                VALUES (1, 1, 'text', 0, 0, 100, 20, 0);
            INSERT INTO line (id, block_id, text, original_text, confidence, proof_status, x, y, w, h)
                VALUES (1, 1, 'legacy text', '', 0.9, 'unchecked', 0, 0, 100, 20);
        """)
        conn.commit()
        conn.close()

        # 用新 ProjectStore 打开（触发迁移）
        with ProjectStore(db_path) as store:
            loaded = store.load_project(project_id=1)
            assert loaded is not None
            assert loaded.name == "legacy"
            assert loaded.pages[0].blocks[0].lines[0].text == "legacy text"
            assert loaded.pages[0].blocks[0].lines[0].final_text == "legacy text"
            assert loaded.pages[0].uid.startswith("page_")
            assert loaded.pages[0].blocks[0].uid.startswith("block_")
            assert loaded.pages[0].blocks[0].lines[0].uid.startswith("line_")

        # 验证 schema 版本已更新
        conn2 = sqlite3.connect(db_path)
        ver = conn2.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert ver is not None
        assert int(ver[0]) >= 4
        final_text_col = conn2.execute("PRAGMA table_info(line)").fetchall()
        assert any(col[1] == "final_text" for col in final_text_col)
        page_cols = conn2.execute("PRAGMA table_info(page)").fetchall()
        assert any(col[1] == "ocr_invalidated_reason" for col in page_cols)
        operation_cols = conn2.execute("PRAGMA table_info(operation_log)").fetchall()
        assert any(col[1] == "object_uid" for col in operation_cols)
        for table in ("page", "block", "line", "char_"):
            cols = conn2.execute(f"PRAGMA table_info({table})").fetchall()
            assert any(col[1] == "uid" for col in cols)
        uid_counts = {
            table: conn2.execute(
                f"SELECT COUNT(*), COUNT(NULLIF(uid, '')) FROM {table}"
            ).fetchone()
            for table in ("page", "block", "line")
        }
        assert uid_counts == {
            "page": (1, 1),
            "block": (1, 1),
            "line": (1, 1),
        }
        indexes = {
            row[1]
            for table in ("page", "block", "line", "char_")
            for row in conn2.execute(f"PRAGMA index_list({table})").fetchall()
        }
        assert {"idx_page_uid", "idx_block_uid", "idx_line_uid", "idx_char_uid"} <= indexes
        conn2.close()

        print("test_project_store_schema_migration PASSED")
    finally:
        os.unlink(db_path)


# =====================================================================
# ProofEngine 测试
# =====================================================================

def test_proof_engine():
    from app.models import BBox, Block, BlockType, Line, Page, ProofStatus
    from app.core.proof_engine import ProofEngine

    bb = BBox(0, 0, 100, 20)
    lines = [
        Line(text="高置信", confidence=0.95, bbox=bb),
        Line(text="低置信", confidence=0.65, bbox=bb),
        Line(text="边界值", confidence=0.80, bbox=bb),
    ]
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=lines)
    page = Page(image_path="/tmp/x.jpg", width=800, height=600, blocks=[block])

    engine = ProofEngine(threshold=0.80)
    count = engine.auto_flag([page])
    assert count == 1
    assert lines[1].proof_status == ProofStatus.AUTO_FLAGGED
    print("test_proof_engine PASSED")


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
    edited = Line(text="OCR原文", confidence=0.9, bbox=bb, proof_status=ProofStatus.MODIFIED)
    edited.ocr_text = "OCR原文"
    edited.chars = [Char(char="O", confidence=0.9, bbox=bb)]
    edited.update_text("人工终审")
    table_line = Line(text="表格文字", confidence=0.8, bbox=bb)
    page = Page(
        image_path="/tmp/page.png",
        cache_image_path="/tmp/cache.png",
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
    assert data["pages"][0]["source_image"] == "/tmp/cache.png"

    print("test_project_to_export_ir_builder_maps_final_text_and_fallbacks PASSED")


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
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 2, 30, 40),
        order=0,
        lines=[Line(text="标题文本", confidence=0.95, bbox=BBox(1, 2, 30, 10))],
        source_label="text",
        raw_payload={"block_label": "paragraph_title", "block_bbox": [1, 2, 31, 42]},
    )
    page = Page(image_path="/tmp/attrs-page.png", width=100, height=100, blocks=[block])
    document = build_export_ir(OcrProject(name="AttrIR", pages=[page]), "json")
    element = document.to_dict()["pages"][0]["elements"][0]

    assert element["kind"] == "title"
    assert element["source"]["block_type"] == "text"
    assert element["source"]["source_label"] == "text"
    assert element["source"]["semantic_label"] == "paragraph_title"
    assert element["source"]["semantic_block_type"] == "title"
    assert element["source"]["raw_payload"]["block_label"] == "paragraph_title"
    assert element["layout_attributes"]["semantic_block_type"] == "title"

    print("test_export_ir_preserves_structured_block_attributes PASSED")


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


def test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_uniform_font():
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
            assert abs(first_rect.y0 - (20 * 72 / 300)) < 0.3
            assert abs(second_rect.y0 - (60 * 72 / 300)) < 0.3
            sizes = set()
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.add(round(float(span["size"]), 2))
            assert len(sizes) == 1
        finally:
            doc.close()

    print("test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_uniform_font PASSED")


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
        proof_status=ProofStatus.AUTO_FLAGGED,
        review_flags=["low_confidence"],
    )
    edited.ocr_text = "OCR旧文"
    edited.update_text("人工终文")
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
        assert "1/2" in panel._status_lbl.text()
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


def test_layout_panel_merges_selected_blocks_for_ocr_rerun():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(120, 80, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(image_path=str(image_path), width=120, height=80)
        page.blocks = [
            Block(block_type=BlockType.EQUATION, bbox=BBox(10, 10, 20, 10), lines=[
                Line(text="x", confidence=0.9, bbox=BBox(10, 10, 20, 10)),
            ], order=0),
            Block(block_type=BlockType.EQUATION, bbox=BBox(40, 10, 20, 10), lines=[
                Line(text="(1)", confidence=0.9, bbox=BBox(40, 10, 20, 10)),
            ], order=1),
        ]
        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            for item, _block in panel._viewer._block_items:
                item.setSelected(True)

            changed = []
            panel.geometry_changed.connect(lambda: changed.append(True))
            panel._merge_selected_blocks()

            assert len(page.blocks) == 1
            assert page.blocks[0].bbox == BBox(10, 10, 50, 10)
            assert page.blocks[0].lines == []
            assert page.blocks[0].source == BlockSource.USER_EDITED
            assert page.blocks[0].app_payload["ocr_text_invalidated"] is True
            assert changed
        finally:
            panel.close()

    print("test_layout_panel_merges_selected_blocks_for_ocr_rerun PASSED")


def test_layout_panel_defaults_auto_text_blocks_locked():
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

            assert text_block.is_locked is True
            assert formula_block.is_locked is False
            text_item = next(item for item, block in panel._viewer._block_items if block is text_block)
            formula_item = next(item for item, block in panel._viewer._block_items if block is formula_block)
            assert not bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert not bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            assert bool(formula_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert bool(formula_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

            panel._on_block_clicked(text_block)
            assert panel._selected_block is None
            assert "锁定" in panel._status_lbl.text()
            panel._unlock_page_blocks()
            assert text_block.is_locked is False
            assert text_block.app_payload["ui_lock_overridden"] is True
        finally:
            panel.close()

    print("test_layout_panel_defaults_auto_text_blocks_locked PASSED")


def test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Line, Page
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
            raw_payload={
                "block_label": "text",
            },
            app_payload={
                "_hanwang_bbox_audit": {
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
            },
        )
        page = Page(image_path=str(image_path), width=160, height=80, blocks=[text_block])
        skip_block = Block(
            block_type=BlockType.TABLE,
            bbox=BBox.from_xyxy(10, 50, 80, 70),
            source=BlockSource.AUTO_LAYOUT,
            app_payload={
                "_hanwang_bbox_audit": {
                    "schema": "hanwang_bbox_audit.v1",
                    "layout_block_bbox": [10, 50, 80, 70],
                    "effective_block_bbox": [10, 50, 80, 70],
                    "effective_block_bbox_source": "layout_block_bbox",
                    "route_text_slice_count": 0,
                    "hanwang_recog_group_count": 0,
                }
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

            panel._inspector.set_block(text_block)
            summary = panel._inspector._lbl_hanwang_audit.text()
            assert "route=2" in summary
            assert "recog=2" in summary
            assert "clipped=1" in summary
            panel._inspector.set_block(skip_block)
            assert "未进入 Hanwang text-slice 路由" in panel._inspector._lbl_hanwang_audit.text()

            text_block.app_payload["ocr_text_invalidated"] = True
            page.invalidate_ocr("block_moved")
            panel._refresh_current_page_layers()
            assert len(panel._viewer._readonly_overlay_items) == 0
            panel._inspector.set_block(text_block)
            assert "已失效，需要重新进入 OCR" in panel._inspector._lbl_hanwang_audit.text()
        finally:
            panel.close()

    print("test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle PASSED")


def test_layout_panel_draw_merge_uses_large_box_and_removes_overlap():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(140, 90, QImage.Format.Format_RGB888).save(str(image_path))
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
            for i in range(panel._new_type_combo.count()):
                if panel._new_type_combo.itemData(i) == BlockType.EQUATION:
                    panel._new_type_combo.setCurrentIndex(i)
                    break

            panel._on_block_created(BBox(10, 10, 90, 30))

            assert len(page.blocks) == 1
            assert page.blocks[0].bbox == BBox(10, 10, 90, 30)
            assert page.blocks[0].block_type == BlockType.EQUATION
            assert page.blocks[0].lines == []
            assert page.blocks[0].source == BlockSource.USER_EDITED
            assert page.blocks[0].app_payload["ocr_text_invalidated"] is True
        finally:
            panel.close()

    print("test_layout_panel_draw_merge_uses_large_box_and_removes_overlap PASSED")


def test_layout_panel_draw_ignores_locked_text_targets():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(140, 90, QImage.Format.Format_RGB888).save(str(image_path))
        text_block = Block(
            block_type=BlockType.TEXT,
            bbox=BBox(20, 20, 80, 20),
            lines=[Line(
                text="abc",
                confidence=0.9,
                bbox=BBox(20, 20, 80, 20),
                chars=[Char(char="a", confidence=0.9, bbox=BBox(20, 20, 18, 10))],
            )],
            source=BlockSource.AUTO_LAYOUT,
            order=0,
        )
        page = Page(image_path=str(image_path), width=140, height=90, blocks=[text_block])

        panel = LayoutPanel()
        try:
            panel.set_pages([page])
            app.processEvents()
            for i in range(panel._new_type_combo.count()):
                if panel._coerce_block_type(panel._new_type_combo.itemData(i), BlockType.UNKNOWN) == BlockType.EQUATION:
                    panel._new_type_combo.setCurrentIndex(i)
                    break

            panel._on_block_created(BBox(25, 22, 20, 12))

            assert text_block.is_locked is True
            assert len(page.blocks) == 2
            assert page.blocks[0] is text_block
            assert page.blocks[1].block_type == BlockType.EQUATION
            assert page.blocks[1].bbox == BBox(25, 22, 20, 12)
        finally:
            panel.close()

    print("test_layout_panel_draw_ignores_locked_text_targets PASSED")


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
            assert panel._type_combo.isEnabled()
            assert any(item.isSelected() and item_block is block for item, item_block in panel._viewer._block_items)

            for i in range(panel._type_combo.count()):
                if panel._coerce_block_type(panel._type_combo.itemData(i), BlockType.UNKNOWN) == BlockType.TABLE:
                    panel._type_combo.setCurrentIndex(i)
                    break

            assert block.block_type == BlockType.TABLE
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
            for i in range(panel._new_type_combo.count()):
                if panel._coerce_block_type(panel._new_type_combo.itemData(i), BlockType.UNKNOWN) == BlockType.EQUATION:
                    panel._new_type_combo.setCurrentIndex(i)
                    break

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

    from app.models import BBox, Block, BlockType, Page
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
            assert text_block.is_locked is True
            assert len(panel._viewer._char_items) == 1
            char_item, _ = panel._viewer._char_items[0]
            formula_item = next(item for item, block in panel._viewer._block_items if block is formula_block)
            assert not bool(char_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            assert not bool(char_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            assert formula_item.zValue() > char_item.zValue()

            formula_item.setSelected(True)
            panel._delete_selected()

            assert page.blocks == [text_block]
            assert text_block.is_locked is True
        finally:
            panel.close()

    print("test_layout_panel_readonly_char_boxes_do_not_block_formula_delete PASSED")


def test_layout_panel_hides_empty_and_invalidated_char_boxes():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Char, Line, Page
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

            block.app_payload["ocr_text_invalidated"] = True
            panel._refresh_current_page_layers()
            assert panel._viewer._char_items == []

            block.app_payload.clear()
            page.invalidate_ocr("block_moved")
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


def test_layout_panel_type_combo_changes_unlocked_block_type():
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
            for i in range(panel._type_combo.count()):
                if panel._coerce_block_type(panel._type_combo.itemData(i), BlockType.UNKNOWN) == BlockType.TABLE:
                    panel._type_combo.setCurrentIndex(i)
                    break

            assert formula_block.block_type == BlockType.TABLE
            assert formula_block.source == BlockSource.USER_EDITED
        finally:
            panel.close()

    print("test_layout_panel_type_combo_changes_unlocked_block_type PASSED")


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
        assert all(not block.is_locked for block in inline_blocks)
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
    finally:
        panel.close()

    print("test_layout_panel_promotes_real_inline_formula_overlays_to_editable_blocks PASSED")


def test_layout_panel_skips_superscript_marker_inline_formula_overlays_from_120169():
    import json
    from pathlib import Path

    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

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
    footnote = next(block for block in page.blocks if block.source_label == "vision_footnote")
    assert footnote.block_type == BlockType.TEXT

    panel = LayoutPanel()
    try:
        inline_bboxes = [
            bbox.to_xyxy()
            for _parent, _subblock, bbox in panel._iter_inline_formula_subblocks(page)
        ]
    finally:
        panel.close()

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
    assert statuses[-1] == "Paddle 版面分析中… 第 2/3 页"

    print("test_workflow_controller_layout_progress_signal PASSED")


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


def test_main_window_centered_resize_expands_from_current_center():
    from app.ui.main_window import MainWindow

    app = _get_qapp()
    window = MainWindow()
    try:
        window.setGeometry(200, 180, 300, 240)
        window.show()
        app.processEvents()
        before = window.frameGeometry().center()

        window._set_centered_window_size(500, 400)
        app.processEvents()

        after = window.frameGeometry().center()
        assert window.width() == 500
        assert window.height() == 400
        assert abs(after.x() - before.x()) <= 1
        assert abs(after.y() - before.y()) <= 1
    finally:
        window.close()

    print("test_main_window_centered_resize_expands_from_current_center PASSED")


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

    _get_qapp()
    window = MainWindow()
    try:
        file_menu_action = next(
            action
            for action in window.menuBar().actions()
            if action.menu() is not None and "文件" in action.text()
        )
        file_menu = file_menu_action.menu()
        actions = [action for action in file_menu.actions() if not action.isSeparator()]
        action_texts = [action.text() for action in actions]
        assert any("关闭项目" in text for text in action_texts)
        assert not any("退出" in text for text in action_texts)
        close_action = next(action for action in actions if "关闭项目" in action.text())
        assert close_action.shortcut().toString(QKeySequence.SequenceFormat.PortableText) == "Ctrl+W"
    finally:
        window.close()

    print("test_main_window_file_menu_uses_close_project_action PASSED")


def test_main_window_close_project_prompts_save_and_resets_workspace():
    from PySide6.QtWidgets import QMessageBox

    from app.models import OcrProject
    from app.ui.main_window import MainWindow

    _get_qapp()
    questions = []
    warnings = []
    saves = []
    original_question = QMessageBox.question
    original_warning = QMessageBox.warning
    QMessageBox.question = lambda *args, **kwargs: questions.append(args) or QMessageBox.StandardButton.Save
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

        window._close_project()

        assert questions
        assert saves == [True]
        assert warnings == []
        assert store.closed is True
        assert window._controller.project is None
        assert window._stack.currentWidget() is window._import_panel
        assert window._layout_panel._pages == []
        assert window.statusBar().currentMessage() == "项目已关闭"
    finally:
        QMessageBox.question = original_question
        QMessageBox.warning = original_warning
        window.close()

    print("test_main_window_close_project_prompts_save_and_resets_workspace PASSED")


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
        assert lines[0].proof_status == ProofStatus.AUTO_FLAGGED
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_preserves_rec_text_without_any_geometry PASSED")


def test_fake_layout_engine():
    from app.engines.fake_layout_engine import FakeLayoutEngine

    engine = FakeLayoutEngine()
    blocks = engine.analyze("/tmp/nonexistent.jpg")  # 应返回 fallback

    assert len(blocks) > 0
    assert blocks[0].bbox.w > 0

    print("test_fake_layout_engine PASSED")


# =====================================================================
# Fake LLM 引擎测试
# =====================================================================

def test_fake_llm_engine_disabled():
    """LLM 关闭时不调用 adapter。"""
    from app.models import Line, LlmReviewStatus
    # 当 llm_review_status 为 DISABLED 时，不触发审查
    line = Line(
        text="测试", confidence=0.9,
        bbox=__import__('app.models').models.BBox(0, 0, 10, 10),
        llm_review_status=LlmReviewStatus.DISABLED,
    )
    assert line.llm_review_status == LlmReviewStatus.DISABLED
    assert line.llm_suggestion == ""
    print("test_fake_llm_engine_disabled PASSED")


def test_fake_llm_engine():
    from app.engines.fake_llm_engine import (
        FakeLlmPreReviewEngine, LlmPreReviewLine, LlmPreReviewOptions,
    )

    engine = FakeLlmPreReviewEngine()
    lines = [
        LlmPreReviewLine(page_number=1, block_order=0, line_index=0,
                          text="正常文本", confidence=0.95),
        LlmPreReviewLine(page_number=1, block_order=0, line_index=1,
                          text="错別字", confidence=0.90),
        LlmPreReviewLine(page_number=1, block_order=0, line_index=2,
                          text="", confidence=0.0),
        LlmPreReviewLine(page_number=1, block_order=0, line_index=3,
                          text="低置信", confidence=0.60),
    ]

    suggestions = engine.review_lines(lines)

    assert len(suggestions) == 4

    # 正常文本：无修改
    assert suggestions[0].suggested_text == "正常文本"
    assert suggestions[0].flags == []

    # 错别字检测
    assert suggestions[1].suggested_text == "错别字"
    assert "ocr_typo" in suggestions[1].flags

    # 空文本
    assert "empty_text" in suggestions[2].flags

    # 低置信
    assert "low_confidence" in suggestions[3].flags

    print("test_fake_llm_engine PASSED")


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

        assert abs(line.bbox.y - 67) <= 3
        assert line.bbox.h <= 18
        assert len(line.chars) == 2
    finally:
        os.unlink(img_path)


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
        assert len(progress_events) == 2
        assert progress_events[0].completed_pages == 1
        assert progress_events[0].current_page == 1
        assert progress_events[1].completed_pages == 2
        assert "第 2/2 页" in progress_events[1].message

    print("test_ocr_pipeline_reports_real_page_progress PASSED")


def test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
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
    equation = Block(block_type=BlockType.EQUATION, bbox=bb, recognizable=True)
    table_binding = Block(
        block_type=BlockType.TEXT,
        bbox=bb,
        app_payload={"paddle_binding": {"source_label": "table", "block_type": "table"}},
    )
    disabled_text = Block(block_type=BlockType.TEXT, bbox=bb, recognizable=False)

    assert is_text_ocr_candidate(footnote) is True
    assert should_dispatch_to_text_ocr(footnote) is True
    assert should_dispatch_to_text_ocr(formula_label) is False
    assert should_dispatch_to_text_ocr(equation) is False
    assert should_dispatch_to_text_ocr(table_binding) is False
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
        equation = Block(block_type=BlockType.EQUATION, bbox=BBox(0, 0, 120, 60), lines=[make_line("旧公式", 5, 5)], order=0)
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
        recognizable=True,
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


def test_ocr_pipeline_skips_equation_block_ocr_even_when_recognizable():
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
            recognizable=True,
        )
        page = Page(image_path=img_path, width=160, height=80, blocks=[equation])
        result = OcrPipeline(engine=RaisingTextEngine()).process_project(
            OcrProject(name="EquationSkip", pages=[page])
        )

        assert [line.text for line in result.pages[0].blocks[0].lines] == ["E=mc^2"]
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_skips_equation_block_ocr_even_when_recognizable PASSED")


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

    recog_shapes = []
    recog_recblocks = []

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        recog_shapes.append(tuple(image_bgr.shape[:2]))
        recog_recblocks.append(recblocks_xyxy)
        assert recblock_xyxy is None
        h, w = image_bgr.shape[:2]
        assert (h, w) == (74, 136)
        assert recblocks_xyxy == [(0, 0, 96, 36), (0, 38, 136, 74)]
        return {"lines": [
            {"groups": [{
                "bbox": {"left": 0, "top": 0, "right": 96, "bottom": 36},
                "chars": [
                    {"codes": [code("天")], "scores": [5], "bbox": {"left": 0, "top": 0, "right": 23, "bottom": 36}},
                    {"codes": [code("地")], "scores": [6], "bbox": {"left": 28, "top": 0, "right": 51, "bottom": 36}},
                ],
            }]},
            {"groups": [{
                "bbox": {"left": 0, "top": 38, "right": 136, "bottom": 74},
                "chars": [
                    {"codes": [code("短")], "scores": [12], "bbox": {"left": 0, "top": 38, "right": 28, "bottom": 74}},
                ],
            }]},
        ]}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

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
        assert rows[0].lines[0].chars[0].bbox == (12, 22, 35, 58)
        assert rows[1].text == "$$x+y$$"
        assert rows[2].text == "短"
        assert rows[2].fallback_reason == ""
        assert stats.n_blocks_hanwang == 2
        assert stats.n_blocks_ppvl == 1
        assert stats.n_blocks_fallback == 0
        assert recog_shapes == [(74, 136)]
        assert recog_recblocks == [[(0, 0, 96, 36), (0, 38, 136, 74)]]
        assert stats.recog_full_page_pixels == 220 * 240 * 2
        assert stats.recog_crop_pixels == 36 * 96 + 36 * 136
        assert stats.recog_probe_calls == 1
        assert stats.recog_batch_chunks == 1
        assert stats.recog_batch_failures == 0
        assert stats.recog_batch_disabled is False
        assert stats.recog_max_collage_width == 136
        assert stats.recog_max_collage_height == 74
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

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
    page.ppvl_parsing_res_list = [
        {"block_label": "text", "block_bbox": [0, 0, 100, 20], "block_content": "stale paddle text"}
    ]
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
    assert page.blocks[0].recognizable is False
    assert len(page.blocks[0].lines) == 1
    assert page.blocks[0].lines[0].text == ""
    assert "manual_formula_needs_text" in page.blocks[0].lines[0].review_flags

    print("test_hanwang_engine_uses_user_edited_layout_for_manual_formula_boxes PASSED")


def test_hanwang_layout_injects_manual_formula_binding_into_parent_route():
    from app.core.paddle_artifact_index import BINDING_PARENT_FORMULA_INFERRED
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, line_routes_for_block, text_slice_routes_for_block
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockSource, BlockType, Line, Page

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
        ppvl_parsing_res_list=[dict(parent_record)],
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(0, 0, 200, 40),
                lines=[Line(text="bad active ocr", confidence=0.0, bbox=BBox.from_xyxy(0, 0, 200, 40))],
                raw_payload={
                    "block_label": "text",
                    "block_bbox": [0, 0, 200, 40],
                    "block_content": "甲 $ A $ 乙 $ B $ 丙",
                },
                app_payload={
                    ROUTE_SUBBLOCKS_FIELD: list(parent_record[ROUTE_SUBBLOCKS_FIELD]),
                    "_layout_line_routes": list(parent_record["_layout_line_routes"]),
                },
            ),
            Block(
                block_type=BlockType.EQUATION,
                bbox=BBox.from_xyxy(110, 0, 140, 30),
                source=BlockSource.MANUAL_DRAW,
                source_label="inline_formula",
                app_payload={
                    "paddle_binding": {
                        "status": BINDING_PARENT_FORMULA_INFERRED,
                        "block_type": "equation",
                        "source_label": "inline_formula",
                        "text": "$ B $",
                        "parent_index": 0,
                        "manual_bbox": [110, 0, 140, 30],
                    },
                },
            ),
        ],
    )

    blocks = _page_blocks_from_layout(page)
    assert len(blocks) == 1
    parent = blocks[0]
    assert parent["block_content"] == "甲 $ A $ 乙 $ B $ 丙"
    assert "_layout_line_routes" not in parent
    assert [sub["block_bbox"] for sub in parent[ROUTE_SUBBLOCKS_FIELD]] == [
        [40, 0, 70, 40],
        [110, 0, 140, 40],
    ]

    routes = line_routes_for_block(parent, 220, 60)
    formula_segments = [
        segment
        for route in routes
        for segment in route["segments"]
        if segment["kind"] == "formula"
    ]
    assert [segment["text"] for segment in formula_segments] == ["$ A $", "$ B $"]
    assert [route["bbox"] for route in text_slice_routes_for_block(parent, 220, 60)] == [
        [0, 0, 40, 40],
        [70, 0, 110, 40],
        [140, 0, 200, 40],
    ]

    print("test_hanwang_layout_injects_manual_formula_binding_into_parent_route PASSED")


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
        text_by_shape = {
            (70, 30): "甲甲",
            (100, 30): "乙乙。",
            (40, 30): "丙",
            (50, 30): "丁",
            (30, 30): "戊",
        }
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        lines = []
        for x1, y1, x2, y2 in blocks:
            text = text_by_shape.get((x2 - x1, y2 - y1), "")
            if not text:
                continue
            char_width = max(1, (x2 - x1) // max(1, len(text)))
            chars = []
            for idx, ch in enumerate(text):
                left = x1 + idx * char_width
                right = x2 if idx == len(text) - 1 else min(x2, left + char_width)
                chars.append({
                    "codes": [code(ch)],
                    "scores": [5],
                    "bbox": {"left": left, "top": y1, "right": right, "bottom": y2},
                })
            lines.append({"groups": [{
                "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                "chars": chars,
            }]})
        return {
            "lines": lines
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

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

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        text_by_shape = {
            (60, 20): "甲",
            (90, 20): "乙",
            (80, 20): "丙",
            (70, 20): "丁",
        }
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        return {
            "lines": [
                {"groups": [{
                    "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    "chars": [{
                        "codes": [code(text_by_shape[(x2 - x1, y2 - y1)])],
                        "scores": [5],
                        "bbox": {"left": x1, "top": y1, "right": min(x2, x1 + 20), "bottom": y2},
                    }],
                }]}
                for x1, y1, x2, y2 in blocks
                if (x2 - x1, y2 - y1) in text_by_shape
            ]
        }

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog

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
            ("$ A $", (60, 10, 90, 30), "paddle_inline_formula", "word"),
            ("$ B $", (80, 50, 110, 70), "paddle_inline_formula", "word"),
        ]
        assert stats.n_blocks_hanwang == 1
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_pre_page_ocr_lines_split_before_recog PASSED")


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
        text = {40: "甲", 50: "乙"}.get(w, "")
        if not text:
            return {"lines": []}
        return {
            "lines": [
                {
                    "groups": [
                        {
                            "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                            "chars": [
                                {
                                    "codes": [code(text)],
                                    "scores": [5],
                                    "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
                                }
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
        assert rows[0].recog_group_bboxes == [(0, 2, 40, 38), (70, 2, 120, 38)]
        assert rows[0].segimg_group_audits == [
            {
                "route_text_slice_bbox": [0, 0, 40, 40],
                "segimg_group_bbox": [0, 2, 42, 38],
                "recog_group_bbox": [0, 2, 40, 38],
                "clipped": True,
                "dropped": False,
            },
            {
                "route_text_slice_bbox": [70, 0, 120, 40],
                "segimg_group_bbox": [68, 2, 122, 38],
                "recog_group_bbox": [70, 2, 120, 38],
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
        assert audit["hanwang_recog_group_bboxes"] == [[0, 2, 40, 38], [70, 2, 120, 38]]
        assert audit["hanwang_segimg_group_clipped_count"] == 2
        assert audit["hanwang_segimg_group_dropped_count"] == 0
        assert audit["route_text_slice_count"] == 2
        assert audit["hanwang_recog_group_count"] == 2
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog

    print("test_hanwang_bbox_audit_distinguishes_layout_route_and_recog_boxes PASSED")


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


def test_ocr_pipeline_hybrid_prepass_lines_feed_hanwang_splitter():
    import os
    import tempfile

    import cv2
    import numpy as np
    import app.engines.hanwang.micro_recblock as micro_module

    from app.engines.hanwang.micro_recblock import HanwangMicroRecBlockEngine
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
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
        text_by_shape = {(60, 20): "甲", (90, 20): "乙"}
        blocks = recblocks_xyxy or [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
        return {
            "lines": [
                {"groups": [{
                    "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    "chars": [{
                        "codes": [code(text_by_shape[(x2 - x1, y2 - y1)])],
                        "scores": [5],
                        "bbox": {"left": x1, "top": y1, "right": min(x2, x1 + 20), "bottom": y2},
                    }],
                }]}
                for x1, y1, x2, y2 in blocks
                if (x2 - x1, y2 - y1) in text_by_shape
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
        page = Page(
            image_path=img_path,
            width=200,
            height=80,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 190, 50))],
            ppvl_parsing_res_list=[{
                "block_label": "text",
                "block_bbox": [0, 0, 190, 50],
                "block_content": "甲 $ A $ 乙",
                "_route_subblocks": [
                    {"block_label": "inline_formula", "block_bbox": [60, 0, 90, 40]},
                ],
            }],
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

    parent = page.ppvl_parsing_res_list[12]
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
    assert all(
        segment["bbox"][1] == formula_route["bbox"][1]
        and segment["bbox"][3] == formula_route["bbox"][3]
        for segment in formula_route["segments"]
    )
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
    assert block[LAYOUT_LINE_ROUTES_FIELD] == routes

    print("test_paddle_line_routing_has_layout_routes_is_pure PASSED")


def test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        ROUTE_SUBBLOCKS_FIELD,
    )
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
        record for record in page.ppvl_parsing_res_list
        if record.get("block_label") in {"display_formula", "formula_number"}
    ]
    assert skip_records
    assert all(ROUTE_SUBBLOCKS_FIELD not in record for record in skip_records)
    assert all(LAYOUT_LINE_ROUTES_FIELD not in record for record in skip_records)

    text_record = next(
        record for record in page.ppvl_parsing_res_list
        if record.get("block_label") == "text" and "$ Y_{ct} $" in str(record.get("block_content") or "")
    )
    assert len(text_record[ROUTE_SUBBLOCKS_FIELD]) == 7

    formula_routes = [
        route for route in text_record[LAYOUT_LINE_ROUTES_FIELD]
        if any(segment["kind"] == "formula" for segment in route["segments"])
    ]
    assert len(formula_routes) == 5
    three_formula_route = next(
        route for route in formula_routes
        if sum(1 for segment in route["segments"] if segment["kind"] == "formula") == 3
    )
    assert [segment["text"] for segment in three_formula_route["segments"] if segment["kind"] == "formula"] == [
        "$ X_{ct} $",
        "$ \\delta_{c} $",
        "$ \\varphi_{t} $",
    ]
    assert all(
        segment["bbox"][1] == three_formula_route["bbox"][1]
        and segment["bbox"][3] == three_formula_route["bbox"][3]
        for segment in three_formula_route["segments"]
    )

    print("test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands PASSED")


def test_layout_fixture_page_ocr_routes_do_not_shift_after_missing_formula_box():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import (
        LAYOUT_LINE_ROUTES_FIELD,
        attach_page_ocr_line_routes,
    )
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

    attach_page_ocr_line_routes(
        page.ppvl_parsing_res_list,
        page_ocr_lines,
        page.width,
        page.height,
    )
    text_record = next(
        record for record in page.ppvl_parsing_res_list
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


def test_paddle_artifact_index_binds_real_missing_inline_formula_from_parent_truth():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_artifact_index import (
        BINDING_GEOMETRY_HIT,
        BINDING_PARENT_FORMULA_INFERRED,
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
    assert geometry.text == "$ Incentive_{c} \\times Post_{t} $"
    assert geometry.recognizable is False
    assert missing.status == BINDING_PARENT_FORMULA_INFERRED
    assert missing.text == "$ Incentive_{c} $"
    assert "manual_formula_from_parent_text" in missing.review_flags

    print("test_paddle_artifact_index_binds_real_missing_inline_formula_from_parent_truth PASSED")


def test_paddle_artifact_index_binds_parent_table_and_empty_formula_review():
    from app.core.paddle_artifact_index import (
        BINDING_EMPTY_REVIEW,
        BINDING_PARENT_TABLE_HIT,
        PaddleArtifactIndex,
    )
    from app.models import BBox, BlockType, Page

    page = Page(image_path="/tmp/table-page.png", width=300, height=220)
    page.ppvl_parsing_res_list = [
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
    ]

    index = PaddleArtifactIndex.from_page(page)
    table = index.bind_manual_bbox(BBox.from_xyxy(35, 45, 265, 165), BlockType.TABLE)
    empty_formula = index.bind_manual_bbox(BBox.from_xyxy(50, 174, 120, 198), BlockType.EQUATION)

    assert table.status == BINDING_PARENT_TABLE_HIT
    assert table.text == "<table><tr><td>A</td></tr></table>"
    assert table.source_label == "table"
    assert table.recognizable is False
    assert empty_formula.status == BINDING_EMPTY_REVIEW
    assert empty_formula.text == ""
    assert "manual_formula_needs_text" in empty_formula.review_flags

    print("test_paddle_artifact_index_binds_parent_table_and_empty_formula_review PASSED")


def test_layout_panel_manual_formula_writes_paddle_binding_payload():
    from pathlib import Path
    import tempfile

    from PySide6.QtGui import QImage

    from app.core.paddle_artifact_index import BINDING_PARENT_FORMULA_INFERRED
    from app.models import BBox, Block, BlockSource, BlockType, Page
    from app.ui.recognize.layout_panel import LayoutPanel

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = _get_qapp()
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "page.png"
        QImage(300, 120, QImage.Format.Format_RGB888).save(str(image_path))
        page = Page(image_path=str(image_path), width=300, height=120)
        page.ppvl_parsing_res_list = [
            {
                "block_label": "text",
                "block_bbox": [10, 10, 260, 70],
                "block_content": "甲 $ A $ 乙 $ B $ 丙",
                "_route_subblocks": [
                    {"block_label": "inline_formula", "block_bbox": [60, 12, 90, 40]},
                ],
            }
        ]
        block = Block(
            block_type=BlockType.EQUATION,
            bbox=BBox.from_xyxy(120, 12, 150, 42),
            source=BlockSource.MANUAL_DRAW,
        )

        panel = LayoutPanel()
        try:
            panel._bind_manual_block_to_paddle(page, block)

            binding = block.app_payload["paddle_binding"]
            assert binding["status"] == BINDING_PARENT_FORMULA_INFERRED
            assert binding["text"] == "$ B $"
            assert block.source_label == "inline_formula"
            assert block.recognizable is False
            assert block.lines[0].text == "$ B $"
        finally:
            panel.close()
            app.processEvents()

    print("test_layout_panel_manual_formula_writes_paddle_binding_payload PASSED")


def test_layout_analyzer_reads_formula_geometry_records_for_routes():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD
    from app.models import Page

    page = Page(image_path="/tmp/formula-geometry-key.png", width=160, height=80)
    data = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "formula": [
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
    subblocks = page.ppvl_parsing_res_list[0][ROUTE_SUBBLOCKS_FIELD]

    assert blocks[0].app_payload[ROUTE_SUBBLOCKS_FIELD] == subblocks
    assert [(item["block_label"], item["block_bbox"]) for item in subblocks] == [
        ("inline_formula", [50, 10, 80, 32]),
    ]
    assert [label for label, _bbox in overlays] == ["text", "inline_formula"]

    print("test_layout_analyzer_reads_formula_geometry_records_for_routes PASSED")


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
        for x1, y1, x2, y2 in blocks:
            assert not (x1 == 40 and x2 == 60), "skipped formula region reached Recog"
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
        assert [row.block_label for row in rows] == ["vision_footnote", "footnote"]
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
        if recblocks_xyxy is not None:
            calls["batch"] += 1
            raise RuntimeError("AccessViolationException")
        calls["single"] += 1
        return {"lines": [{"groups": [{
            "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            "chars": [{
                "codes": [code("甲")],
                "scores": [5],
                "bbox": {"left": 0, "top": 0, "right": w, "bottom": h},
            }],
        }]}]}

    original_segimg = micro_module.native_bridge.run_linecut_segimg
    original_recog = micro_module.native_bridge.run_linecut_recog
    original_max_groups = micro_module.MAX_RECOG_BATCH_GROUPS
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    original_batch_reason = micro_module._BATCH_DISABLE_REASON
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
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
        micro_module.MAX_RECOG_BATCH_GROUPS = original_max_groups
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled
        micro_module._BATCH_DISABLE_REASON = original_batch_reason

    print("test_hanwang_micro_recblock_circuit_breaks_after_batch_failure PASSED")


def test_hanwang_micro_recblock_width_guard_skips_risky_batch():
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

    def fake_recog(
        image_bgr,
        *,
        recblock_xyxy=None,
        recblocks_xyxy=None,
        with_charrcg=True,
        timeout=0,
    ):
        h, w = image_bgr.shape[:2]
        if recblocks_xyxy is not None:
            calls["batch"] += 1
            raise AssertionError("wide collage should not use batch")
        calls["single"] += 1
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
    original_width = micro_module.MAX_RECOG_COLLAGE_WIDTH
    original_batch_disabled = micro_module._BATCH_DISABLED_FOR_SESSION
    original_batch_reason = micro_module._BATCH_DISABLE_REASON
    micro_module.native_bridge.run_linecut_segimg = fake_segimg
    micro_module.native_bridge.run_linecut_recog = fake_recog
    micro_module.MAX_RECOG_COLLAGE_WIDTH = 1600
    micro_module._BATCH_DISABLED_FOR_SESSION = False
    micro_module._BATCH_DISABLE_REASON = ""

    try:
        rows, stats = micro_module.run_micro_recblock(
            np.zeros((120, 2000, 3), dtype=np.uint8),
            [{"block_label": "text", "block_bbox": [0, 0, 2000, 120], "block_content": "乙乙"}],
            include_chars=True,
        )

        assert rows[0].source == "hanwang"
        assert calls == {"batch": 0, "single": 2}
        assert stats.recog_batch_chunks == 2
        assert stats.recog_batch_guarded_chunks == 2
        assert stats.recog_batch_failures == 0
        assert stats.recog_batch_disabled is False
        assert stats.recog_probe_calls == 2
        assert stats.recog_max_collage_width == 1850
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_segimg
        micro_module.native_bridge.run_linecut_recog = original_recog
        micro_module.MAX_RECOG_COLLAGE_WIDTH = original_width
        micro_module._BATCH_DISABLED_FOR_SESSION = original_batch_disabled
        micro_module._BATCH_DISABLE_REASON = original_batch_reason

    print("test_hanwang_micro_recblock_width_guard_skips_risky_batch PASSED")


def test_ocr_pipeline_runs_hanwang_micro_recblock_page_path():
    import tempfile
    import cv2
    import numpy as np
    from app.engines.hanwang.micro_recblock import (
        BlockResult, CharResult, HanwangMicroRecBlockEngine, LineResult, RunStats,
    )
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
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
        page = Page(
            image_path=img_path,
            width=240,
            height=220,
            blocks=[Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 30))],
            ppvl_parsing_res_list=[
                {"block_label": "text", "block_bbox": [10, 20, 110, 60], "block_content": "PPVL文本"},
                {"block_label": "display_formula", "block_bbox": [20, 80, 180, 120], "block_content": "$$x+y$$"},
                {"block_label": "reference", "block_bbox": [20, 140, 180, 180], "block_content": "参考文献"},
            ],
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
        assert calls[0][0] == page.ppvl_parsing_res_list
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
        assert out_page.blocks[0].raw_payload["extra"]["role"] == "body"
        assert out_page.blocks[0].lines[0].chars[0].bbox_source == "hanwang:micro_recblock"
        assert out_page.blocks[1].lines[0].text == "$$x+y$$"
        assert out_page.blocks[1].recognizable is False
        assert out_page.blocks[1].raw_payload["formula_format"] == "latex"
        assert "fallback_reason=" not in out_page.blocks[2].note
        assert out_page.blocks[2].lines == []
        assert out_page.blocks[2].raw_payload["ref_level"] == 1
    finally:
        os.unlink(img_path)

    print("test_ocr_pipeline_runs_hanwang_micro_recblock_page_path PASSED")


def test_hanwang_page_blocks_from_layout_preserves_raw_source_label():
    from app.engines.hanwang.micro_recblock import _page_blocks_from_layout
    from app.models import BBox, Block, BlockType, Page

    page = Page(
        image_path="/tmp/raw-label.png",
        width=200,
        height=100,
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(11, 22, 133, 88),
                note="active text",
                source_label="paragraph_title",
                raw_payload={
                    "block_label": "paragraph_title",
                    "block_bbox": [1, 2, 3, 4],
                    "block_content": "raw text",
                    "custom_attr": {"level": 2},
                },
            )
        ],
    )

    blocks = _page_blocks_from_layout(page)

    assert blocks[0]["block_label"] == "paragraph_title"
    assert blocks[0]["source_label"] == "paragraph_title"
    assert blocks[0]["block_bbox"] == [11, 22, 133, 88]
    assert blocks[0]["block_content"] == "active text"
    assert blocks[0]["custom_attr"]["level"] == 2

    print("test_hanwang_page_blocks_from_layout_preserves_raw_source_label PASSED")


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
                raw_payload={"block_label": "inline_formula"},
            ),
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(20, 40, 80, 60),
                source=BlockSource.USER_EDITED,
                note="active text",
                source_label="text",
                raw_payload={"block_label": "text"},
            ),
        ],
    )

    blocks = _page_blocks_from_layout(page)

    assert blocks[0]["block_label"] == "inline_formula"
    assert blocks[0]["block_content"] == ""
    assert blocks[1]["block_content"] == "active text"

    print("test_hanwang_page_blocks_from_layout_does_not_promote_internal_merge_note_to_formula_text PASSED")


def test_hanwang_ppvl_skip_uses_layout_authority_label():
    import numpy as np

    from app.engines.hanwang.micro_recblock import BlockResult, HanwangMicroRecBlockEngine, LineResult, RunStats
    from app.models import BBox, Block, BlockType, Page

    calls = []

    def fake_runner(image_bgr, ppvl_blocks, **kwargs):
        calls.append(ppvl_blocks)
        assert ppvl_blocks[0]["block_label"] == "figure"
        return [
            BlockResult(
                block_idx=0,
                block_label="figure",
                block_bbox=(10, 10, 80, 40),
                source="ppvl",
                text="图",
                ppvl_text="图",
                raw_block=dict(ppvl_blocks[0]),
                lines=[LineResult(text="图", bbox=(10, 10, 80, 40), source="ppvl")],
            )
        ], RunStats(n_blocks_total=1, n_blocks_ppvl=1)

    page = Page(
        image_path="/tmp/ppvl-skip-authority.png",
        width=100,
        height=100,
        ppvl_parsing_res_list=[],
        blocks=[
            Block(
                block_type=BlockType.TEXT,
                bbox=BBox.from_xyxy(10, 10, 80, 40),
                source_label="text",
                raw_payload={"block_label": "figure", "label": "text", "block_content": "图"},
            )
        ],
    )

    HanwangMicroRecBlockEngine(runner=fake_runner).recognize_page_blocks(
        np.zeros((100, 100, 3), dtype=np.uint8),
        page,
    )

    assert len(calls) == 1
    assert page.blocks[0].block_type == BlockType.FIGURE
    assert page.blocks[0].source_label == "figure"
    assert page.blocks[0].recognizable is False

    print("test_hanwang_ppvl_skip_uses_layout_authority_label PASSED")


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
        assert any("VL1.6 版面分析（汉王混合）中" in message for message in messages)
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
        page_pending.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20))]
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
        messages = []
        steps = []
        focused = []
        controller.status_message.connect(messages.append)
        controller.step_requested.connect(steps.append)
        controller.focus_page.connect(focused.append)

        controller.handle_ocr_entry_requested("main_window", 1)

        assert focused == [1]
        assert steps == [workflow_module.STEP_LAYOUT]
        assert messages[-1].startswith("当前页 OCR 失败，可重新进入 OCR")
        assert messages[-1] != "全部已完成 OCR"
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


def test_workflow_controller_hanwang_layout_submit_merges_only_target_page():
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
        processed_page2 = Page(image_path="/tmp/p2.png", width=100, height=100, page_number=2)
        processed_page2.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 30, 50, 20), lines=[
            Line(text="第二页完成", bbox=BBox(0, 30, 50, 20), confidence=0.99)
        ])]
        FakeOcrWorker.created_pages = []
        FakeOcrWorker.result_pages = [processed_page2]

        controller = workflow_module.WorkflowController()
        controller._project = OcrProject(name="Gate", pages=[page1, page2])

        controller.handle_ocr_entry_requested("layout_submit", 2)

        assert FakeOcrWorker.created_pages == [[page2]]
        assert len(controller._project.pages) == 2
        assert controller._project.pages[0] is page1
        assert controller._project.pages[0].status == PageStatus.LAYOUT_DONE
        assert controller._project.pages[1] is processed_page2
        assert controller._project.pages[1].status == PageStatus.OCR_DONE
        assert controller._project.pages[1].blocks[0].lines[0].text == "第二页完成"
    finally:
        workflow_module.get_config = original_get_config
        workflow_module.create_engine = original_create_engine
        workflow_module.OcrPipelineWorker = original_worker

    print("test_workflow_controller_hanwang_layout_submit_merges_only_target_page PASSED")


def test_workflow_controller_hanwang_block_edit_invalidates_only_that_page():
    import app.controllers.workflow_controller as workflow_module
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus

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

        assert page1.total_lines == 0
        assert page1.status == PageStatus.LAYOUT_DONE
        assert page1.needs_ocr_rerun is True
        assert page1.ocr_invalidated_reason == "block_moved"
        assert page2.total_lines == 1
        assert page2.status == PageStatus.OCR_DONE
    finally:
        workflow_module.get_config = original_get_config

    print("test_workflow_controller_hanwang_block_edit_invalidates_only_that_page PASSED")


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
        assert finished[0][0].status == workflow_module.PageStatus.OCR_DONE
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
    from app.services.ocr_pipeline import OcrProgress

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
        assert len(progress_payloads) == 1
        assert progress_payloads[0].completed_pages == 1
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
    assert any("校对已可进入" in message for message in messages)

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

    assert page.total_lines == 1
    assert page.status == PageStatus.ERROR
    gate = page_gate_info(page)
    assert gate.page_state == "ocr_error"
    assert gate.action_enabled is True

    print("test_workflow_controller_ocr_done_keeps_error_status_even_with_prepass_lines PASSED")


def test_main_window_ocr_finished_preserves_current_step():
    from app.controllers.workflow_controller import STEP_HPROOF, STEP_LAYOUT, STEP_OCR
    from app.services.ocr_pipeline import OcrProgress
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

        window._on_ocr_finished([Page(image_path="/tmp/ocr-finished.png", width=10, height=10)])

        assert synced == [True, True]
        assert window._controller.current_step == STEP_OCR
        assert window._stack.currentIndex() == STEP_LAYOUT
        assert window._stack.currentWidget() is window._layout_panel
        assert window._ocr_placeholder.isHidden()
        assert window._controller.current_step != STEP_HPROOF
    finally:
        window.close()

    print("test_main_window_ocr_finished_preserves_current_step PASSED")


def test_empty_llm_config_does_not_block_ocr_done():
    from app.controllers.workflow_controller import WorkflowController
    from app.core.app_config import AppConfig, update_config
    from app.engines.fake_llm_engine import FakeLlmPreReviewEngine
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page

    original_review = FakeLlmPreReviewEngine.review_lines

    def raising_review(self, lines, options=None):
        raise AssertionError("LLM pre-review must not be called from OCR completion")

    FakeLlmPreReviewEngine.review_lines = raising_review
    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
    update_config(llm_pre_review_enabled=True, llm_endpoint="", llm_api_key="")
    try:
        line = Line(text="人工终审文本", confidence=0.91, bbox=BBox(1, 2, 40, 16))
        page = Page(image_path="/tmp/llm-empty.png", width=120, height=80)
        page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40), lines=[line])]
        controller = WorkflowController()
        controller._project = OcrProject(name="LlmEmptyDoesNotBlock", pages=[page])

        controller.on_ocr_done([page])

        assert line.text == "人工终审文本"
        assert line.llm_suggestion == ""
    finally:
        FakeLlmPreReviewEngine.review_lines = original_review
        cfg.reset_to_defaults()

    print("test_empty_llm_config_does_not_block_ocr_done PASSED")


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
            assert abs(loaded_line.bbox.y - 67) <= 3
            assert loaded_line.bbox.h <= 18
            assert len(loaded_line.chars) == 2
        finally:
            controller.close()
    finally:
        os.unlink(db_path)
        os.unlink(img_path)


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
    line.update_final_text("人工最终真值")
    assert get_export_text(line) == "人工最终真值"

    # 测试空项目
    empty_project = OcrProject(name="empty", pages=[])
    warnings = check_export_readiness(empty_project)
    assert len(warnings) > 0

    # 测试有未校对行
    page = Page(image_path="/tmp/x.jpg", width=800, height=600)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[
        Line(text="未校对", confidence=0.6, bbox=bb,
             proof_status=ProofStatus.UNCHECKED),
    ])
    page.blocks = [block]
    project = OcrProject(name="test", pages=[page])
    warnings = check_export_readiness(project)
    has_unproofed = any("未校对" in w for w in warnings)
    assert has_unproofed

    print("test_export_service PASSED")


def test_proof_display_edit_writes_final_text():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.services.proof_probe_text_service import displayed_text, save_displayed_edit

    bb = BBox(0, 0, 100, 20)
    line = Line(text="OCR原文", confidence=0.9, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/x.jpg", width=800, height=600, blocks=[block])

    assert displayed_text(line, page, block) == "OCR原文"
    assert save_displayed_edit(line, page, block, "人工终稿") is True
    assert line.final_text == "人工终稿"
    assert line.text == "OCR原文"
    assert line.display_text == "人工终稿"

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
    from app.core.proof_state_bus import (
        TOPIC_LINE_PROOF_CHANGED, get_proof_state_bus,
    )

    bus = get_proof_state_bus()
    bus.clear()
    events = []

    unsubscribe = bus.subscribe(TOPIC_LINE_PROOF_CHANGED, events.append)
    bus.publish(TOPIC_LINE_PROOF_CHANGED, {"line_id": 7, "status": "ok"})

    assert events == [{"line_id": 7, "status": "ok"}]
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == 1

    unsubscribe()
    assert bus.subscriber_count(TOPIC_LINE_PROOF_CHANGED) == 0

    bus.clear()
    print("test_proof_state_bus PASSED")


def test_proof_state_bus_typed_contracts():
    from app.core import quality_probe as qp
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
    assert qp.TOPIC_PROBE_OBSERVED == TOPIC_PROBE_OBSERVED
    line_events = []
    legacy_line_events = []
    probe_events = []

    bus.subscribe(TOPIC_LINE_PROOF_CHANGED, line_events.append)
    bus.subscribe(TOPIC_LINE_PROOF_CHANGED, lambda **payload: legacy_line_events.append(payload))
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
    assert legacy_line_events[0]["line_id"] == 11
    assert legacy_line_events[0]["status"] == "modified"
    assert ProofUpdateRequest.from_legacy(request) == request
    assert ProofUpdateRequest.from_legacy(request.to_legacy_payload()).line_id == 11
    assert "line_uid" not in request.to_legacy_payload()
    assert ProofUpdateRequest.from_legacy({
        "page_id": 5,
        "page_uid": "page_uid_5",
        "line_id": 11,
        "line_uid": "line_uid_11",
        "status": "modified",
    }).line_uid == "line_uid_11"
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
    assert ProbeObservation.from_legacy(observation.to_legacy_payload()) == observation

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


def test_char_index_groups_digit_runs_as_tokens():
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
    digit_entries = svc.query("2026")
    assert len(digit_entries) == 1
    assert digit_entries[0].bbox == BBox(10, 20, 48, 24)
    assert digit_entries[0].collection_kind == "token"
    assert svc.query("2") == []
    assert len(svc.query("年")) == 1

    print("test_char_index_groups_digit_runs_as_tokens PASSED")


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


def test_char_index_sorts_digit_tokens_short_to_long():
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
    assert digit_keys == ["9", "11", "2026"]

    print("test_char_index_sorts_digit_tokens_short_to_long PASSED")


def test_char_index_groups_formula_runs_below_digits():
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

    assert svc.query("A") == []
    assert svc.query("+") == []
    assert svc.query("B") == []
    formula = svc.first_entry("A+B")
    assert formula is not None
    assert formula.collection_kind == "token"
    assert formula.bbox == BBox(84, 20, 34, 24)

    sorted_keys = [key for key, _count in svc.sorted_chars()]
    assert sorted_keys.index("2026") < sorted_keys.index("A+B")

    print("test_char_index_groups_formula_runs_below_digits PASSED")


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

    formula = svc.first_entry("A+")
    assert formula is not None
    assert formula.collection_kind == "token"
    assert formula.bbox == BBox(10, 20, 22, 24)

    carrier_entry = svc.first_entry(carrier)
    assert carrier_entry is not None
    assert carrier_entry.bbox_source == "paddle_inline_formula"
    assert carrier_entry.bbox_granularity == "word"
    assert line.chars[2].char == carrier

    print("test_char_index_keeps_formula_span_separate_from_word_level_inline_formula_carrier PASSED")


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


def test_vproof_text_map_deduplicates_overlapping_duplicate_lines():
    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.proof.v_proof import _build_text_map

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

    text, mapping = _build_text_map(page)
    assert text.count("重复行") == 2
    assert sum(1 for line, *_ in mapping if line is line_b) == 0
    assert sum(1 for line, *_ in mapping if line is line_c) == 3

    print("test_vproof_text_map_deduplicates_overlapping_duplicate_lines PASSED")


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
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=bb,
            lines=[
                Line(text="确认", confidence=0.9, bbox=bb, proof_status=ProofStatus.OK),
                Line(text="修改", confidence=0.9, bbox=bb, proof_status=ProofStatus.MODIFIED),
                Line(
                    text="疑点", confidence=0.6, bbox=bb,
                    proof_status=ProofStatus.UNCHECKED, review_flags=["low_confidence"],
                ),
                Line(text="待处理", confidence=0.9, bbox=bb, proof_status=ProofStatus.UNCHECKED),
            ],
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
    from app.ui.widgets.api_settings_dialog import (
        get_api_model_profile_options,
        get_api_model_profile_url,
        match_api_model_profile_from_url,
    )

    options = get_api_model_profile_options()
    assert [label for _, label in options] == [
        "PP-OCRv5",
        "PP-StructureV3",
        "PaddleOCR-VL",
        "PaddleOCR-VL-1.5",
        "PaddleOCR-VL-1.6",
    ]
    assert get_api_model_profile_url("pp-ocrv5").endswith("/ocr")
    assert get_api_model_profile_url("pp-structurev3").endswith("/layout-parsing")
    assert get_api_model_profile_url("paddleocr-vl-1.6").endswith("/api/v2/ocr/jobs")
    assert match_api_model_profile_from_url("https://n6z9feddjca4l7b5.aistudio-app.com/ocr") == "pp-ocrv5"
    assert match_api_model_profile_from_url("https://example.com/custom-layout") is None

    print("test_api_model_profile_helpers PASSED")


def test_api_endpoint_role_resolution_keeps_layout_and_proof_separate():
    """Layout role 已全面切到 PaddleOCR-VL-1.6；OCR proof role 仍走 PP-OCRv5。

    所有官方预设 (pp-ocrv5 / pp-structurev3 / paddleocr-vl) 在 role="layout"
    下都被 strong-redirect 到 paddleocr-vl-1.6 预设 URL。
    自托管根 URL 按 role 自动补 /api/v2/ocr/jobs 或 /ocr。
    """
    from app.core.api_profiles import (
        get_api_model_profile_url,
        normalize_api_base_url,
        resolve_api_endpoint_for_role,
    )

    vl16_url = get_api_model_profile_url("paddleocr-vl-1.6")
    ocr_url = get_api_model_profile_url("pp-ocrv5")
    structure_url = get_api_model_profile_url("pp-structurev3")
    structure_root = structure_url.removesuffix("/layout-parsing")
    vl16_root = vl16_url.removesuffix("/api/v2/ocr/jobs")
    ocr_root = ocr_url.removesuffix("/ocr")

    assert normalize_api_base_url(structure_url) == structure_root
    assert normalize_api_base_url(ocr_url) == ocr_root
    assert normalize_api_base_url(vl16_url) == vl16_root

    # Layout role: 任何官方 PP-* / 旧 VL 预设 -> VL-1.6
    assert resolve_api_endpoint_for_role(
        structure_url,
        profile="pp-structurev3",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        ocr_url,
        profile="pp-ocrv5",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        structure_root,
        profile="pp-structurev3",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="pp-ocrv5",
        role="layout",
    ) == vl16_url
    # 旧 paddleocr-vl 预设也归入 VL-1.6
    old_vl_url = get_api_model_profile_url("paddleocr-vl")
    assert resolve_api_endpoint_for_role(
        old_vl_url,
        profile="paddleocr-vl",
        role="layout",
    ) == vl16_url
    # VL-1.5 自身也升级到 VL-1.6
    vl15_url = get_api_model_profile_url("paddleocr-vl-1.5")
    assert resolve_api_endpoint_for_role(
        vl15_url,
        profile="paddleocr-vl-1.5",
        role="layout",
    ) == vl16_url

    # OCR role: 任何 layout 预设 -> PP-OCRv5；PP-OCRv5 自身保持
    assert resolve_api_endpoint_for_role(
        structure_url,
        profile="pp-structurev3",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        vl16_url,
        profile="paddleocr-vl-1.6",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        structure_root,
        profile="",
        role="layout",
    ) == vl16_url
    assert resolve_api_endpoint_for_role(
        vl16_root,
        profile="",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        structure_root,
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

    update_config(
        mode="api",
        api_model_profile="paddleocr-vl-1.6",
        api_url="https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        api_token="demo",
        api_timeout=12,
        api_layout_model_name="",
    )
    current = get_config()
    assert current["api_model_profile"] == "paddleocr-vl-1.6"
    assert current["api_url"] == "https://paddleocr.aistudio-app.com"
    assert current["api_timeout"] == 12
    assert current["api_token"] == "demo"
    assert current["api_layout_model_name"] == ""
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
    update_config(mode="local", api_model_profile="pp-structurev3", api_url="https://example.com/root", api_token="old")

    dialog = ApiSettingsDialog()
    assert dialog._radio_hanwang.isChecked()
    assert dialog._selected_mode() == "hanwang"
    assert dialog._mode_card.isHidden() is True
    assert dialog._api_model_row.isHidden()
    assert dialog._api_model_combo.currentIndex() == -1
    assert dialog._url_edit.text() == "https://example.com/root"
    assert "汉王混合链路" in dialog._summary_model.text()
    assert "汉王混合" in dialog._summary_mode.text()
    assert dialog._api_form_panel.isEnabled() is True
    assert not dialog._timeout_row.isHidden()
    assert dialog._timeout_spin.maximum() >= 600

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
        assert "汉王混合" in dialog._summary_mode.text()

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


def test_api_settings_dialog_llm_copy_is_suggestion_only_and_non_blocking():
    from app.core.app_config import AppConfig
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()

        note = dialog._llm_scope_note.text()
        assert "候选/预审建议层" in note
        assert "人工仍是终审" in note
        assert "不会阻断主流程" in note
        assert "颜色判定" in note

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_llm_copy_is_suggestion_only_and_non_blocking PASSED")


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


def test_api_settings_dialog_persists_llm_candidate_settings():
    from app.core.app_config import AppConfig
    from app.core.app_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._llm_url_edit.setText("https://llm.example.com/v1/chat/completions")
        dialog._llm_key_edit.setText("llm-secret")
        dialog._llm_rules_edit.setText("resources/llm_rules/default_rules.txt")

        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["llm_endpoint"] == "https://llm.example.com/v1/chat/completions"
        assert cfg["llm_api_key"] == "llm-secret"
        assert cfg["llm_rules_path"] == "resources/llm_rules/default_rules.txt"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_persists_llm_candidate_settings PASSED")


def test_llm_rules_loads_default_rules_file():
    from app.core.llm_rules import get_default_llm_rules_path, load_llm_rules

    rules = load_llm_rules()

    assert get_default_llm_rules_path().exists()
    assert "Do not overwrite final proof text automatically" in rules

    print("test_llm_rules_loads_default_rules_file PASSED")
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


def test_layout_analyzer_extracts_api_blocks_from_varied_schema():
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
                                    "category_name": "section_title",
                                    "polygon": [10, 20, 210, 20, 210, 120, 10, 120],
                                    "cls_score": 0.91,
                                },
                                {
                                    "type": "table_caption_text",
                                    "bbox": {"x": 240, "y": 40, "w": 160, "h": 60},
                                    "confidence": 0.88,
                                },
                                {
                                    "layout_label": "graphic",
                                    "points": [[420, 60], [560, 60], [560, 180], [420, 180]],
                                    "layout_score": "0.75",
                                },
                            ],
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "bibliography",
                                "block_bbox": [600, 80, 760, 150],
                                "block_score": 0.81,
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
    assert len(overlays) == 4

    print("test_layout_analyzer_extracts_api_blocks_from_varied_schema PASSED")


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

    assert page.ppvl_parsing_res_list[0]["block_label"] == "equation"
    assert [block.block_type for block in blocks] == [BlockType.EQUATION]
    assert blocks[0].source_label == "equation"
    assert len(overlays) == 2
    assert overlays[0][0] == "equation"
    assert overlays[1][0] == "text"

    print("test_layout_parsing_semantics_override_layout_det_when_both_exist PASSED")


def test_layout_analyzer_forwards_route_subblocks_from_layout_det_res():
    from app.core.layout_analyzer import LayoutAnalyzer
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
    subblocks = page.ppvl_parsing_res_list[0]["_route_subblocks"]
    line_routes = page.ppvl_parsing_res_list[0]["_layout_line_routes"]

    assert blocks[0].block_type == BlockType.TEXT
    assert blocks[0].app_payload["_route_subblocks"] == subblocks
    assert blocks[0].app_payload["_layout_line_routes"] == line_routes
    assert [item["block_label"] for item in subblocks] == ["inline_formula", "table_region"]
    assert subblocks[0]["block_bbox"] == [60, 20, 90, 42]
    assert subblocks[0]["raw_payload"]["label"] == "inline_formula"
    assert any(segment["kind"] == "formula" for route in line_routes for segment in route["segments"])
    assert any(segment["kind"] == "skip" for route in line_routes for segment in route["segments"])
    assert [label for label, _bbox in overlays] == ["text", "inline_formula", "table_region", "figure_caption"]

    print("test_layout_analyzer_forwards_route_subblocks_from_layout_det_res PASSED")


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
    assert page.ppvl_parsing_res_list == parsing_res_list
    assert page.ppvl_parsing_res_list[0]["custom_raw"]["keep"] is True
    assert blocks[0].source_label == "text"
    assert blocks[0].raw_payload["custom_raw"]["keep"] is True

    print("test_layout_analyzer_persists_raw_parsing_res_list PASSED")


def test_layout_analyzer_falls_back_to_ocr_results():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, BlockType, Page

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

    assert len(blocks) == 2
    assert all(block.block_type == BlockType.TEXT for block in blocks)
    assert blocks[0].bbox == BBox(10, 20, 200, 50)
    assert "第一行" in blocks[0].note
    assert "score=0.950" in blocks[0].note
    assert len(overlays) == 2

    print("test_layout_analyzer_falls_back_to_ocr_results PASSED")


def test_layout_analyzer_builds_api_payload():
    from app.core.layout_analyzer import LayoutAnalyzer

    analyzer = LayoutAnalyzer()
    payload = analyzer._build_api_payload("abc123", 1, "PP-DocLayout-L")
    assert payload["file"] == "abc123"
    assert payload["fileType"] == 1
    assert payload["model_name"] == "PP-DocLayout-L"

    payload_without_model = analyzer._build_api_payload("abc123", 1, "")
    assert "model_name" not in payload_without_model

    print("test_layout_analyzer_builds_api_payload PASSED")


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
    from app.core.layout_analyzer import LayoutAnalyzer
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
        profile="paddleocr-vl",
        endpoint_url="https://example.com/layout-parsing",
    )
    assert vl_body["file"] == "abc"
    assert vl_body["fileType"] == 1
    assert vl_body["useDocUnwarping"] is False
    assert vl_body["useDocOrientationClassify"] is False
    assert "returnWordBox" not in vl_body
    assert "textDetLimitType" not in vl_body

    layout_body = LayoutAnalyzer()._build_api_request_body(
        "abc",
        1,
        profile="pp-structurev3",
        endpoint_url="https://example.com/layout-parsing",
    )
    assert "returnWordBox" not in layout_body
    assert layout_body["textDetLimitType"] == "max"

    vl_layout_body = LayoutAnalyzer()._build_api_request_body(
        "abc",
        1,
        profile="paddleocr-vl-1.5",
        endpoint_url="https://example.com/layout-parsing",
    )
    assert "returnWordBox" not in vl_layout_body
    assert vl_layout_body["useDocUnwarping"] is False

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
                    },
                },
            ],
        },
    }

    blocks, _overlays = analyzer._extract_api_blocks(page, data)

    assert len(blocks) == 1
    assert blocks[0].bbox == BBox(20, 40, 200, 100)

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
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        cv2.imwrite(page_path, np.full((120, 200, 3), 255, dtype=np.uint8))
        page = Page(image_path=page_path, width=200, height=120)
        LayoutAnalyzer()._api_analyze(page)
        assert captured["url"] == "https://example.com/root/api/v2/ocr/jobs"
        assert captured["timeout"] == 180
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
        assert captured["headers"]["Authorization"] == "bearer demo"
        assert captured["data"]["model"] == "PaddleOCR-VL-1.6"
        optional_payload = json.loads(captured["data"]["optionalPayload"])
        assert optional_payload["useDocOrientationClassify"] is False
        assert optional_payload["useDocUnwarping"] is False
        assert optional_payload["useChartRecognition"] is False
        filename, image_bytes, mime = captured["files"]["file"]
        assert filename == "page.png"
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert mime == "image/png"
        assert captured["gets"][0]["url"] == "https://example.com/root/api/v2/ocr/jobs/job-1"
        assert captured["gets"][0]["headers"]["Authorization"] == "bearer demo"
        assert captured["gets"][0]["proxies"] == {"http": None, "https": None, "all": None}
        assert len(page.blocks) == 1
        assert page.blocks[0].block_type == BlockType.TEXT
        assert "OCR行" in page.blocks[0].note
    finally:
        requests.post = original_post
        requests.get = original_get
        cfg.reset_to_defaults()
        os.unlink(page_path)

    print("test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected PASSED")


def test_layout_analyzer_legacy_json_request_helper_preserves_png_payload():
    import base64
    import tempfile

    import cv2
    import numpy as np
    import requests

    from app.core.api_profiles import resolve_api_endpoint
    from app.core.app_config import AppConfig, update_config
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import Page

    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"layoutParsingResults": []}}

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
    update_config(mode="api", api_url="https://legacy.example.com", api_timeout=12)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        cv2.imwrite(page_path, np.full((120, 200, 3), 255, dtype=np.uint8))
        page = Page(image_path=page_path, width=200, height=120)
        analyzer = LayoutAnalyzer()
        legacy_url = resolve_api_endpoint(
            "https://legacy.example.com",
            default_suffix="/layout-parsing",
            profile="pp-structurev3",
        )
        assert legacy_url == "https://legacy.example.com/layout-parsing"
        # Directly exercise the legacy JSON request shape; main role resolution is VL1.6-first.
        import app.core.api_profiles as profiles

        original_fixed = profiles.FIXED_LAYOUT_PROFILE
        original_default = profiles.LAYOUT_DEFAULT_PROFILE
        profiles.FIXED_LAYOUT_PROFILE = "pp-structurev3"
        profiles.LAYOUT_DEFAULT_PROFILE = "pp-structurev3"
        try:
            analyzer._api_analyze(page)
        finally:
            profiles.FIXED_LAYOUT_PROFILE = original_fixed
            profiles.LAYOUT_DEFAULT_PROFILE = original_default
        assert captured["timeout"] == 180
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
        assert base64.b64decode(captured["json"]["file"]).startswith(b"\x89PNG\r\n\x1a\n")
        assert captured["json"]["useDocUnwarping"] is False
        assert "returnWordBox" not in captured["json"]
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()
        os.unlink(page_path)

    print("test_layout_analyzer_legacy_json_request_helper_preserves_png_payload PASSED")


def test_layout_analyzer_routes_hanwang_mode_to_ppvl_layout():
    import app.core.app_config as config_module
    import app.core.layout_analyzer as layout_module
    from app.models import BBox, Block, BlockType, Page

    def fake_api_analyze(self, page):
        page.width = 300
        page.height = 200
        page.ppvl_parsing_res_list = [
            {"block_label": "text", "block_bbox": [12, 18, 92, 58], "block_content": "PPVL"}
        ]
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
        assert result.ppvl_parsing_res_list[0]["block_content"] == "PPVL"
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

        def fake_run_exe(exe, args, *, cwd, timeout):
            rb_path = Path(cwd) / args[2]
            captured["rb_text"] = rb_path.read_text(encoding="utf-8")
            captured["args"] = list(args)
            (Path(cwd) / args[1]).write_text('{"lines":[]}', encoding="utf-8")
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
    from app.services.ocr_pipeline import OcrProgress

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
            raw_payload={"block_label": "footer", "block_content": "$  \\frac{1}{2}  $"},
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


def test_hproof_line_iterator_excludes_non_text_elements():
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
        Block(block_type=BlockType.EQUATION, bbox=BBox(0, 40, 80, 20), lines=[
            Line(text="E=mc2", confidence=0.9, bbox=BBox(1, 41, 30, 10)),
        ]),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)]

    assert texts == ["正文"]

    print("test_hproof_line_iterator_excludes_non_text_elements PASSED")


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
            raw_payload={"block_label": "page_number"},
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 20, 60, 12),
            lines=[Line(text="正文", confidence=0.9, bbox=BBox(1, 20, 60, 12))],
            note="source_label=page_number",
            source_label="text",
            raw_payload={"block_label": "text"},
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 40, 80, 12),
            lines=[Line(text="脚注", confidence=0.9, bbox=BBox(1, 40, 80, 12))],
            source_label="footnote",
            raw_payload={"block_label": "footnote"},
        ),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)]

    assert texts == ["正文", "脚注"]

    print("test_hproof_line_iterator_excludes_position_source_labels PASSED")


def test_block_attributes_reads_raw_payload_without_note():
    from app.core.block_attributes import block_attributes, block_display_label, is_position_only_block
    from app.models import BBox, Block, BlockType

    title_like = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 1, 20, 10),
        note="source_label=text",
        raw_payload={"block_label": "paragraph_title", "score": 0.99},
    )
    attrs = block_attributes(title_like)

    assert attrs.source_label == "paragraph_title"
    assert attrs.semantic_label == "paragraph_title"
    assert attrs.semantic_block_type == BlockType.TITLE
    assert block_display_label(title_like) == "text · paragraph_title"

    position = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 1, 20, 10),
        note="source_label=text",
        raw_payload={"block_label": "page_number"},
    )
    assert is_position_only_block(position)

    footnote = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(1, 20, 80, 10),
        raw_payload={"block_label": "footnote"},
    )
    assert not is_position_only_block(footnote)
    assert block_display_label(footnote) == "text · footnote"

    print("test_block_attributes_reads_raw_payload_without_note PASSED")


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
    assert panel._items[0][2] is page2
    assert panel._pairs[0]._line_in_page == 1
    panel.close()

    print("test_hproof_page_filter_keeps_pages_separate PASSED")


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
    assert panel._current_idx == 0
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
    old_page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[old_line])]
    new_line = Line(text="新对象", confidence=0.9, bbox=BBox(1, 1, 20, 10))
    new_page = Page(
        image_path="/tmp/hproof-rebind.png",
        width=100,
        height=100,
        page_number=1,
        source_path="/tmp/source.tif",
    )
    new_page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), order=0, lines=[new_line])]

    panel = HProofPanel()
    panel.load_pages([old_page])
    panel._pairs[0]._editor.setPlainText("用户未保存")

    panel.merge_pages([new_page])
    panel._save_current(silent=True)

    assert len(panel._pairs) == 1
    assert panel._items[0][1] is new_line
    assert panel._pairs[0].line is new_line
    assert new_line.final_text == "用户未保存"
    assert old_line.display_text == "旧对象"
    panel.close()

    print("test_hproof_merge_rebinds_replaced_lines_without_duplicates_or_orphans PASSED")


def test_vproof_merge_pages_preserves_current_page_text():
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

    assert panel._current_page_idx == 0
    assert panel._text_edit.toPlainText() == "未保存纵校文本"
    assert panel._char_svc.query("乙")
    panel.close()

    print("test_vproof_merge_pages_preserves_current_page_text PASSED")


def test_top_nav_moves_layout_run_button_and_removes_prev_next():
    from app.ui.main_window import TopNavBar

    _get_qapp()
    nav = TopNavBar()

    assert not hasattr(nav, "_btn_prev")
    assert not hasattr(nav, "_btn_next")
    assert nav._btn_run_layout.text() == "▶"
    nav.set_layout_run_enabled(True)
    assert nav._btn_run_layout.isEnabled()
    nav.close()

    print("test_top_nav_moves_layout_run_button_and_removes_prev_next PASSED")


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
    assert panel._left_box.maximumWidth() <= 170
    assert panel._gallery_view.itemDelegate().sizeHint(None, panel._gallery_model.index(0, 0)).height() <= 80
    assert panel._gallery_box.parentWidget() is panel._proof_column
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


def test_vproof_candidate_provider_interface_is_prepared():
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    class Provider:
        def __init__(self):
            self.requests = []

        def suggest_candidates(self, request):
            self.requests.append(request)
            return ["甲", "由"]

    _get_qapp()
    line = Line(
        text="田",
        confidence=0.9,
        bbox=BBox(1, 1, 20, 10),
        chars=[Char(char="田", confidence=0.9, bbox=BBox(1, 1, 10, 10), bbox_source="ocr", bbox_granularity="char")],
    )
    page = Page(image_path="/tmp/vproof-candidate.png", width=100, height=100, page_number=3)
    page.blocks = [Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 20), lines=[line])]
    panel = VProofPanel()
    panel.load_pages([page])
    provider = Provider()
    panel.set_candidate_provider(provider)

    entry = panel._char_svc.query("田")[0]
    panel._gallery_model.set_entries([entry])
    panel._on_gallery_clicked(panel._gallery_model.index(0, 0))

    assert provider.requests
    assert provider.requests[0].token == "田"
    assert provider.requests[0].page_number == 3
    assert provider.requests[0].bbox_source == "ocr"
    assert provider.requests[0].bbox_granularity == "char"
    # proof UI clarity（Task #3）：候选区不再展示 "候选：…｜bbox=ocr/char｜…"
    # 这种解释文案；只保留最多 5 个候选按钮，第一候选 = 当前最高可信来源。
    button_texts = [button.text() for button in panel._candidate_buttons]
    assert button_texts[:1] == ["田"]  # 当前字（最高可信）排首位
    assert "甲" in button_texts and "由" in button_texts  # provider 候选并入
    assert len(button_texts) <= 5
    # hint 不再承担长解释；有候选时应被隐藏（或为空）
    assert not panel._candidate_hint.isVisible() or panel._candidate_hint.text() == ""
    panel.close()

    print("test_vproof_candidate_provider_interface_is_prepared PASSED")


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
    from app.models import BBox, Block, BlockType, Char, Line, Page
    from app.ui.proof.v_proof import VProofPanel

    _get_qapp()
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
    button_by_text = {button.text(): button for button in panel._candidate_buttons}
    button_by_text["由"].click()

    assert panel._text_edit.toPlainText().startswith("由")
    assert "待保存" in panel._status_lbl.text()
    panel.close()

    print("test_vproof_candidate_button_applies_to_ocr_text PASSED")


def test_hproof_visual_size_is_compact():
    from app.ui.proof import h_proof

    assert h_proof.IMAGE_ROW_H <= 32
    # 脚注/数字/标点的 Hanwang 字符框更窄，横校文本字号不能再按正文 24px 硬挤。
    assert h_proof.TEXT_FONT_PX == 20
    assert h_proof.TEXT_LINE_HEIGHT_PX <= 28
    assert h_proof.TEXT_EDITOR_MAX_H <= 32
    assert h_proof.LINE_PAIR_H == 70
    assert (
        h_proof.LINE_PAIR_H
        >= h_proof.IMAGE_ROW_H + h_proof.TEXT_EDITOR_MAX_H + 2
    )
    assert "Noto Sans CJK SC" in h_proof.TEXT_FONT_FAMILY

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
    locked = Block(block_type=BlockType.TEXT, bbox=BBox(78, 10, 20, 20), is_locked=True)
    viewer.show_blocks([first, second, locked])
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
    assert viewer.selected_blocks() == [first, second]
    assert not viewer._block_items[2][0].isSelected()
    viewer.close()

    print("test_image_viewer_right_drag_selects_blocks_without_creating_bbox PASSED")


def test_ui_block_labels_use_structured_semantic_label():
    from PySide6.QtGui import QImage

    from app.models import BBox, Block, BlockType, Line, Page
    from app.ui.recognize.ocr_panel import OcrPanel
    from app.ui.widgets.image_viewer import ImageViewer

    _get_qapp()
    block = Block(
        block_type=BlockType.TEXT,
        bbox=BBox(5, 6, 30, 20),
        lines=[Line(text="标题", confidence=0.9, bbox=BBox(5, 6, 30, 10))],
        raw_payload={"block_label": "paragraph_title"},
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


if __name__ == "__main__":
    test_models()
    test_workflow_state_keeps_project_and_page_ocr_state_separate()
    test_bbox_tools()
    test_block_type_mapping()
    test_block_payload_helpers_preserve_existing_entries()
    test_paddle_layout_schema_normalizes_record_fields()
    test_project_store()
    test_project_store_persists_ppvl_parsing_res_list()
    test_line_final_text_alias_and_project_store_roundtrip()
    test_project_store_clean_on_resave()
    test_project_store_save_project_preserves_child_rowids()
    test_project_store_upsert_rejects_foreign_parent_rowids()
    test_project_store_cross_project_uid_collision_remints_without_stealing()
    test_project_store_cross_project_uid_pollution_preserves_valid_rowids()
    test_project_store_uid_recovers_same_parent_stale_rowid()
    test_project_store_duplicate_sibling_uids_are_reminted()
    test_project_store_cross_parent_moves_preserve_uids_regardless_of_save_order()
    test_project_store_persists_page_ocr_invalidation_reason()
    test_project_store_reconciles_legacy_ocr_status_from_lines()
    test_project_store_update_lines_rolls_back_as_single_transaction()
    test_project_store_update_line_requires_stable_uid_match()
    test_project_store_new_db_records_current_schema_version()
    test_project_store_schema_migration()
    test_proof_engine()
    test_export_txt()
    test_txt_dual_encoding_outputs_and_layout_contract()
    test_export_xml()
    test_export_html()
    test_export_markdown_structure()
    test_export_formats_share_structured_blocks()
    test_export_ir_rules_load_and_validate()
    test_project_to_export_ir_builder_maps_final_text_and_fallbacks()
    test_export_ir_char_source_fallbacks_are_unique_across_lines()
    test_export_ir_preserves_structured_block_attributes()
    test_pdf_page_faithful_plans_use_image_and_char_layer()
    test_pdf_dual_textless_page_degrades_without_text_font()
    test_pdf_dual_generated_pdf_searches_continuous_text_and_uses_uniform_font()
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
    test_layout_panel_merges_selected_blocks_for_ocr_rerun()
    test_layout_panel_defaults_auto_text_blocks_locked()
    test_layout_panel_has_no_hanwang_bbox_audit_overlay_toggle()
    test_layout_panel_draw_merge_uses_large_box_and_removes_overlap()
    test_layout_panel_draw_ignores_locked_text_targets()
    test_layout_panel_drawn_block_is_selected_and_type_editable()
    test_layout_panel_draw_snaps_to_image_ink_without_existing_blocks()
    test_layout_panel_ink_snap_reuses_cached_image_mask()
    test_layout_panel_delete_selected_removes_unlocked_box()
    test_layout_panel_readonly_char_boxes_do_not_block_formula_delete()
    test_layout_panel_hides_empty_and_invalidated_char_boxes()
    test_layout_panel_excludes_inline_formula_carriers_from_char_boxes()
    test_layout_panel_type_combo_changes_unlocked_block_type()
    test_layout_panel_undo_restores_block_edits()
    test_layout_panel_undo_preserves_view_transform()
    test_layout_panel_promotes_real_inline_formula_overlays_to_editable_blocks()
    test_layout_panel_skips_superscript_marker_inline_formula_overlays_from_120169()
    test_workflow_controller_layout_progress_signal()
    test_main_window_layout_error_is_status_only()
    test_main_window_centered_resize_expands_from_current_center()
    test_main_window_maximize_state_is_not_forced_back_to_normal()
    test_main_window_file_menu_uses_close_project_action()
    test_main_window_close_project_prompts_save_and_resets_workspace()
    test_fake_ocr_engine()
    test_create_engine_hanwang_exposes_page_block_capability()
    test_confidence_normalization()
    test_api_ocr_engine_does_not_request_return_word_box()
    test_api_ocr_engine_does_not_promote_block_content_to_line()
    test_api_ocr_engine_reads_direct_pruned_ppocr_rows()
    test_api_ocr_engine_ignores_block_content_without_rec_rows()
    test_api_ocr_engine_preserves_rec_text_without_any_geometry()
    test_fake_layout_engine()
    test_fake_llm_engine_disabled()
    test_fake_llm_engine()
    test_ocr_pipeline()
    test_ocr_pipeline_keeps_page_relative_boxes()
    test_ocr_pipeline_offsets_crop_relative_boxes()
    test_ocr_pipeline_prefers_engine_char_boxes_and_only_falls_back_for_missing_chars()
    test_ocr_pipeline_normalizes_proof_geometry()
    test_ocr_pipeline_reports_real_page_progress()
    test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once()
    test_ocr_dispatch_policy_blocks_structural_and_paddle_skip_labels()
    test_page_ocr_refills_caption_blocks_and_preserves_equation_blocks()
    test_page_ocr_nested_equation_blocks_before_parent_text_assignment()
    test_ocr_pipeline_skips_equation_block_ocr_even_when_recognizable()
    test_ocr_pipeline_avoids_double_shift_for_page_space_boxes()
    test_ocr_pipeline_preserves_hanwang_crop_lines_and_chars()
    test_hanwang_micro_recblock_routes_and_fallbacks()
    test_hanwang_engine_uses_user_edited_layout_for_manual_formula_boxes()
    test_hanwang_layout_injects_manual_formula_binding_into_parent_route()
    test_hanwang_inline_formula_text_slices_keep_chars()
    test_hanwang_recog_filters_empty_decoded_char_boxes()
    test_hanwang_inline_formula_carrier_survives_model_and_proof_helpers()
    test_hanwang_pre_page_ocr_lines_split_before_recog()
    test_hanwang_bbox_audit_distinguishes_layout_route_and_recog_boxes()
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
    test_paddle_line_routing_formula_number_is_skip_not_formula_carrier()
    test_paddle_line_routing_formula_rows_use_single_horizontal_band()
    test_paddle_line_routing_has_layout_routes_is_pure()
    test_layout_fixture_routes_skip_parents_and_collapse_formula_row_bands()
    test_layout_fixture_page_ocr_routes_do_not_shift_after_missing_formula_box()
    test_paddle_artifact_index_binds_real_missing_inline_formula_from_parent_truth()
    test_paddle_artifact_index_binds_parent_table_and_empty_formula_review()
    test_layout_panel_manual_formula_writes_paddle_binding_payload()
    test_layout_analyzer_reads_formula_geometry_records_for_routes()
    test_hanwang_inline_formula_empty_text_slices_keeps_empty_hanwang_result()
    test_hanwang_group_chunk_cannot_readmit_skipped_subregions()
    test_hanwang_formula_style_footer_bypasses_hanwang()
    test_hanwang_footnote_labels_route_through_hanwang()
    test_hanwang_micro_recblock_unknown_label_defaults_to_text_path_with_audit()
    test_hanwang_recog_group_failure_is_visible_in_audit_without_ppvl_fallback()
    test_hanwang_micro_recblock_circuit_breaks_after_batch_failure()
    test_hanwang_micro_recblock_width_guard_skips_risky_batch()
    test_ocr_pipeline_runs_hanwang_micro_recblock_page_path()
    test_hanwang_page_blocks_from_layout_preserves_raw_source_label()
    test_hanwang_page_blocks_from_layout_does_not_promote_internal_merge_note_to_formula_text()
    test_hanwang_ppvl_skip_uses_layout_authority_label()
    test_ocr_pipeline_records_failed_page_when_block_ocr_fails()
    test_workflow_controller_auto_chains_ocr_after_layout()
    test_workflow_controller_hanwang_layout_stays_on_block_ocr_path()
    test_workflow_controller_hanwang_ocr_entry_redirects_to_first_pending_page()
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
    test_empty_llm_config_does_not_block_ocr_done()
    test_workflow_controller_normalizes_loaded_project_geometry()
    test_export_service()
    test_import_service()
    test_import_service_sequential_page_numbers()
    test_proof_state_bus()
    test_proof_state_bus_typed_contracts()
    test_workflow_controller_emits_typed_view_state()
    test_api_settings_dialog_keeps_model_preset_sync()
    test_api_settings_dialog_reverse_matches_url_and_persists_profile()
    test_api_settings_dialog_saves_base_url_from_endpoint_suffix()
    test_api_settings_dialog_collapses_mode_to_hanwang_when_saving()
    test_api_settings_dialog_persists_hanwang_mode_with_api_runtime()
    test_api_settings_dialog_llm_copy_is_suggestion_only_and_non_blocking()
    test_api_settings_dialog_persists_llm_candidate_settings()
    test_llm_rules_loads_default_rules_file()
    test_api_model_profile_helpers()
    test_api_endpoint_role_resolution_keeps_layout_and_proof_separate()
    test_api_http_post_json_disables_environment_proxies()
    test_fixed_api_chain_resolves_official_roots_to_vl16_and_ppocrv5()
    test_api_ocr_engine_resolves_ocr_endpoint_for_pp_ocrv5_profile()
    test_api_ocr_engine_parses_paddle_coordinate_variants()
    test_api_request_builders_split_profile_params()
    test_layout_analyzer_rescales_suspicious_blocks()
    test_layout_analyzer_extracts_api_polygon_bbox()
    test_layout_analyzer_extracts_api_blocks_from_varied_schema()
    test_paddle_authority_prefers_block_label_over_conflicting_label_everywhere()
    test_layout_parsing_semantics_override_layout_det_when_both_exist()
    test_layout_analyzer_forwards_route_subblocks_from_layout_det_res()
    test_layout_analyzer_persists_raw_parsing_res_list()
    test_layout_analyzer_falls_back_to_ocr_results()
    test_layout_analyzer_uses_datainfo_canvas_scale()
    test_layout_analyzer_ignores_conflicting_datainfo_when_bbox_is_page_space()
    test_layout_analyzer_ignores_conflicting_pruned_shape_when_bbox_is_page_space()
    test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected()
    test_layout_analyzer_routes_hanwang_mode_to_ppvl_layout()
    test_hanwang_assets_env_accepts_bin_dir()
    test_hanwang_native_bridge_writes_multi_recblocks()
    test_layout_worker_continues_after_single_page_failure()
    test_workflow_controller_marks_partial_layout_failures_without_blocking_success_pages()
    test_workflow_controller_enables_proof_steps_after_first_ocr_page()
    test_proof_line_iterator_includes_caption_and_equation_lines()
    test_hproof_line_iterator_excludes_non_text_elements()
    test_proof_line_iterators_exclude_route_table_lines()
    test_hproof_line_iterator_excludes_position_source_labels()
    test_block_attributes_reads_raw_payload_without_note()
    test_hproof_page_filter_keeps_pages_separate()
    test_hproof_merge_pages_preserves_active_editor_text()
    test_hproof_merge_rebinds_replaced_lines_without_duplicates_or_orphans()
    test_vproof_merge_pages_preserves_current_page_text()
    test_top_nav_moves_layout_run_button_and_removes_prev_next()
    test_vproof_gallery_uses_wrapping_white_grid()
    test_vproof_text_highlight_targets_single_entry()
    test_vproof_highlight_can_repeat_without_losing_state()
    test_vproof_highlight_survives_repeated_page_switches()
    test_vproof_candidate_provider_interface_is_prepared()
    test_vproof_gallery_keyboard_selection_refreshes_linked_panels()
    test_vproof_candidate_button_applies_to_ocr_text()
    test_hproof_visual_size_is_compact()
    test_image_viewer_char_boxes_update_char_bbox()
    test_image_viewer_space_pan_temporarily_disables_box_editing()
    test_image_viewer_shift_left_drag_creates_block_bbox()
    test_image_viewer_draw_uses_snapper_only_for_created_bbox()
    test_image_viewer_right_drag_selects_blocks_without_creating_bbox()
    test_ui_block_labels_use_structured_semantic_label()
    test_hanwang_concurrency_evaluation_script_help()
    test_layout_analyzer_builds_api_payload()
    test_char_index_vertical_split()
    test_char_index_horizontal_split()
    test_char_index_hides_fallback_units_by_default()
    test_char_index_dedup_on_rebuild()
    test_char_index_sort_categories()
    test_char_index_query_stable_order()
    test_char_index_skips_whitespace()
    test_char_index_groups_digit_runs_as_tokens()
    test_char_index_filters_non_cjk_from_default_vproof()
    test_char_index_sorts_digit_tokens_short_to_long()
    test_char_index_groups_formula_runs_below_digits()
    test_char_index_suppresses_punctuation_topic_for_shared_token_bbox()
    test_char_index_uses_token_collection_for_word_level_han_bbox()
    test_char_index_skips_empty_narrow_ocr_bbox()
    test_char_index_skips_lines_with_unverified_geometry()
    test_char_index_deduplicates_overlapping_duplicate_lines()
    test_vproof_text_map_deduplicates_overlapping_duplicate_lines()
    print("\n✓ 所有测试通过")
