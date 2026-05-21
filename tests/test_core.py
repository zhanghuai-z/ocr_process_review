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
        BBox, Block, BlockSource, BlockType, Line,
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

    # Block
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line, line2])
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
    page.blocks.append(block)
    assert page.is_analyzed
    assert page.status == PageStatus.IMPORTED
    assert page.source_type == "image"

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
    assert project.ocr_completed is True

    # Export summary
    summary = project.get_export_summary()
    assert summary["total_pages"] == 1
    assert summary["total_lines"] >= 0

    print("test_models PASSED")


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

    print("test_block_type_mapping PASSED")


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

        # 验证 schema 版本已更新
        conn2 = sqlite3.connect(db_path)
        ver = conn2.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert ver is not None
        assert int(ver[0]) >= 3
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
    from app.export.txt import TxtExporter
    bb = BBox(0, 0, 100, 20)
    line = Line(text="导出测试行", confidence=0.9, bbox=bb)
    block = Block(block_type=BlockType.TEXT, bbox=bb, lines=[line])
    page = Page(image_path="/tmp/img.jpg", width=800, height=600, blocks=[block])
    project = OcrProject(name="TxtTest", pages=[page])
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out_path = f.name
    try:
        TxtExporter().export(project, out_path)
        assert "导出测试行" in open(out_path, encoding="utf-8").read()
        print("test_export_txt PASSED")
    finally:
        os.unlink(out_path)


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
        assert root.tag == "OcrDocument"
        ln = root.findall(".//Line")[0]
        assert ln.text == "XML测试"
        assert ln.get("x") == "10"
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

    bb = BBox(10, 20, 200, 30)
    page = Page(
        image_path="/tmp/img.jpg",
        width=800,
        height=600,
        page_number=1,
        blocks=[
            Block(block_type=BlockType.TITLE, bbox=bb, order=0, lines=[
                Line(text="章节标题", confidence=0.95, bbox=bb),
            ]),
            Block(block_type=BlockType.TEXT, bbox=BBox(10, 80, 200, 80), order=1, lines=[
                Line(text="正文第一行", confidence=0.90, bbox=BBox(10, 80, 200, 24)),
                Line(text="正文第二行", confidence=0.91, bbox=BBox(10, 110, 200, 24)),
            ]),
            Block(block_type=BlockType.EQUATION, bbox=BBox(10, 180, 200, 30), order=2, lines=[
                Line(text="E = mc^2", confidence=0.88, bbox=BBox(10, 180, 200, 30)),
            ]),
            Block(block_type=BlockType.FIGURE_CAPTION, bbox=BBox(10, 230, 200, 30), order=3, lines=[
                Line(text="图一 示例", confidence=0.93, bbox=BBox(10, 230, 200, 30)),
            ]),
        ],
    )
    project = OcrProject(name="MdTest", pages=[page])
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out_path = f.name
    try:
        MarkdownExporter().export(project, out_path)
        content = open(out_path, encoding="utf-8").read()
        assert content.startswith("# MdTest")
        assert "## 第 1 页" in content
        assert '<!-- block type="title"' in content
        assert "### 章节标题" in content
        assert "正文第一行" in content
        assert '<!-- line bbox="10,80,200,24"' in content
        assert "$$\nE = mc^2\n$$" in content
        assert "*图注：图一 示例*" in content
        print("test_export_markdown_structure PASSED")
    finally:
        os.unlink(out_path)


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
            content = open(out_path, encoding="utf-8").read()
            assert "表注文字" in content
            assert content.index("表注文字") < content.index("正文")
        finally:
            os.unlink(out_path)

    print("test_export_formats_share_structured_blocks PASSED")


def test_export_dialog_offers_markdown():
    from app.models import OcrProject
    from app.ui.export.export_dialog import ExportDialog

    _get_qapp()
    dialog = ExportDialog(OcrProject(name="DialogExport"))
    try:
        assert "md" in dialog._checkboxes
        assert dialog._checkboxes["md"].isChecked()
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
        assert os.path.exists(os.path.join(tmpdir, "WorkerExport.txt"))
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
        assert out_path.name == "卷_一_测试.txt"

    print("test_export_filename_sanitizes_invalid_project_name PASSED")


def test_export_worker_sanitizes_project_name_for_all_formats():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.export_service import sanitize_export_filename
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
    formats = ["txt", "md", "rtf", "pdf", "xml", "html", "docx"]
    with tempfile.TemporaryDirectory() as tmpdir:
        worker = ExportWorker(project, formats, tmpdir)
        worker.completed.connect(completed.append)
        worker.run()
        base = sanitize_export_filename(project.name)
        for fmt in formats:
            assert os.path.exists(os.path.join(tmpdir, f"{base}.{fmt}"))

    assert len(completed) == 1
    assert completed[0].all_ok is True
    assert {result.fmt for result in completed[0].successes} == set(formats)

    print("test_export_worker_sanitizes_project_name_for_all_formats PASSED")


def test_export_worker_keeps_formats_independent_when_one_fails():
    import app.export as export_module
    from app.export.base import ExporterBase
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.export_service import sanitize_export_filename
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
            assert open(os.path.join(tmpdir, f"{base}.txt"), encoding="utf-8").read() == "txt"
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


def test_workflow_controller_layout_progress_signal():
    from app.controllers.workflow_controller import WorkflowController

    controller = WorkflowController()
    events = []
    statuses = []
    controller.layout_progress.connect(lambda current, total: events.append((current, total)))
    controller.status_message.connect(statuses.append)

    controller._on_layout_progress(1, 3)

    assert events == [(1, 3)]
    assert statuses[-1] == "版面分析中… 第 2/3 页"

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


def test_api_ocr_engine_requests_return_word_box():
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
        assert captured["json"]["returnWordBox"] is True
        assert captured["json"]["useDocOrientationClassify"] is False
        assert captured["json"]["useDocUnwarping"] is False
        assert captured["json"]["useTextlineOrientation"] is False
        assert captured["json"]["textDetLimitSideLen"] == 1536
        assert captured["json"]["textDetBoxThresh"] == 0.6
        assert captured["json"]["textDetUnclipRatio"] == 2.0
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_requests_return_word_box PASSED")


def test_api_ocr_engine_parses_char_level_word_boxes():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox

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
                                    "rec_texts": ["天地"],
                                    "rec_scores": [0.93],
                                    "rec_boxes": [[10, 20, 50, 80]],
                                },
                                "text_word": [["天", "地"]],
                                "text_word_region": [[
                                    [[10, 20], [28, 20], [28, 80], [10, 80]],
                                    [[30, 20], [48, 20], [48, 80], [30, 80]],
                                ]],
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
        lines = engine.recognize(np.zeros((120, 80, 3), dtype=np.uint8), OcrContext())
        assert len(lines) == 1
        assert lines[0].bbox == BBox(10, 20, 40, 60)
        assert len(lines[0].chars) == 2
        assert lines[0].chars[0].bbox == BBox(10, 20, 18, 60)
        assert lines[0].chars[0].bbox_source == "ocr"
        assert lines[0].chars[0].bbox_granularity == "char"
        assert lines[0].chars[1].token_text == "地"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_parses_char_level_word_boxes PASSED")


def test_api_ocr_engine_parses_pruned_direct_camelcase_word_boxes():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "rec_texts": ["天地"],
                                "rec_scores": [0.93],
                                "rec_boxes": [[10, 20, 70, 50]],
                                "textWord": [["天", "地"]],
                                "textWordRegion": [[
                                    [[10, 20], [40, 20], [40, 50], [10, 50]],
                                    [[41, 20], [70, 20], [70, 50], [41, 50]],
                                ]],
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
        line = engine.recognize(np.zeros((80, 100, 3), dtype=np.uint8), OcrContext())[0]
        assert line.bbox == BBox(10, 20, 60, 30)
        assert len(line.chars) == 2
        assert all(ch.bbox_source == "ocr" for ch in line.chars)
        assert line.chars[0].bbox == BBox(10, 20, 30, 30)
        assert line.chars[1].token_text == "地"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_parses_pruned_direct_camelcase_word_boxes PASSED")


def test_api_ocr_engine_parses_text_word_boxes_alias():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "overall_ocr_res": {
                                    "rec_texts": ["源码"],
                                    "rec_scores": [0.94],
                                    "rec_boxes": [[10, 20, 70, 50]],
                                },
                                "text_word": [["源", "码"]],
                                "text_word_boxes": [[
                                    [10, 20, 40, 50],
                                    [41, 20, 70, 50],
                                ]],
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
        line = engine.recognize(np.zeros((80, 100, 3), dtype=np.uint8), OcrContext())[0]
        assert line.text == "源码"
        assert line.chars[0].bbox == BBox(10, 20, 30, 30)
        assert line.chars[1].bbox == BBox(41, 20, 29, 30)
        assert all(ch.bbox_source == "ocr" for ch in line.chars)
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_parses_text_word_boxes_alias PASSED")


def test_wordbox_anchor_allows_cjk_left_overflow_without_right_expansion():
    import numpy as np

    from app.core.wordbox_anchor import refine_wordbox_anchors
    from app.models import BBox

    image = np.full((100, 140, 3), 255, dtype=np.uint8)
    image[30:58, 26:45] = 0
    source = BBox(30, 20, 50, 50)

    anchors = refine_wordbox_anchors(
        image,
        BBox(20, 20, 90, 50),
        [("税", source)],
    )

    assert len(anchors) == 1
    assert anchors[0].kind == "cjk"
    assert anchors[0].crop_bbox.x < source.x
    assert anchors[0].crop_bbox.x2 <= source.x2
    assert anchors[0].kept_cc_count >= 1

    print("test_wordbox_anchor_allows_cjk_left_overflow_without_right_expansion PASSED")


def test_api_ocr_engine_refines_cjk_word_box_with_anchor():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "overall_ocr_res": {
                                    "rec_texts": ["税"],
                                    "rec_scores": [0.96],
                                    "rec_boxes": [[20, 20, 110, 70]],
                                },
                                "text_word": [["税"]],
                                "text_word_boxes": [[[30, 20, 80, 70]]],
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
        image = np.full((100, 140, 3), 255, dtype=np.uint8)
        image[30:58, 26:45] = 0
        line = ApiOcrEngine().recognize(image, OcrContext())[0]
        assert line.chars[0].bbox.x < 30
        assert line.chars[0].bbox.x2 <= 80
        assert line.chars[0].bbox_source == "ocr"
        assert line.chars[0].bbox_granularity == "char"
        assert line.chars[0].bbox != BBox(30, 20, 50, 50)
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_refines_cjk_word_box_with_anchor PASSED")


def test_api_ocr_engine_marks_word_level_boxes_without_fake_char_precision():
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
                                    "rec_texts": ["南京市长江大桥"],
                                    "rec_scores": [0.97],
                                    "rec_polys": [[
                                        [5, 10], [105, 10], [105, 40], [5, 40],
                                    ]],
                                },
                                "text_word": [["南京市", "长江大桥"]],
                                "text_word_region": [[
                                    [[5, 10], [42, 10], [42, 40], [5, 40]],
                                    [[48, 10], [105, 10], [105, 40], [48, 40]],
                                ]],
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
        line = engine.recognize(np.zeros((80, 160, 3), dtype=np.uint8), OcrContext())[0]
        assert len(line.chars) == len(line.text)
        assert all(ch.bbox_source == "ocr" for ch in line.chars)
        assert all(ch.bbox_granularity == "word" for ch in line.chars[:3])
        assert all(ch.bbox_granularity == "word" for ch in line.chars[3:])
        assert line.chars[0].bbox == line.chars[2].bbox
        assert line.chars[3].bbox == line.chars[-1].bbox
        assert line.chars[0].token_text == "南京市"
        assert line.chars[-1].token_text == "长江大桥"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_marks_word_level_boxes_without_fake_char_precision PASSED")


def test_api_ocr_engine_filters_empty_narrow_word_boxes():
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
                                    "rec_texts": ["甲乙"],
                                    "rec_scores": [0.95],
                                    "rec_boxes": [[10, 10, 70, 40]],
                                },
                                "text_word": [["甲", "乙"]],
                                "text_word_region": [[
                                    [[10, 10], [28, 10], [28, 40], [10, 40]],
                                    [[60, 12], [61, 12], [61, 13], [60, 13]],
                                ]],
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
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        image[10:40, 10:28] = 0
        line = engine.recognize(image, OcrContext())[0]
        assert line.chars[0].bbox_source == "ocr"
        assert line.chars[0].bbox is not None
        assert line.chars[1].bbox is None
        assert line.chars[1].bbox_source == "fallback"
        assert line.chars[1].bbox_granularity == "fallback"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_filters_empty_narrow_word_boxes PASSED")


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


def test_api_ocr_engine_aligns_token_rows_by_bbox_not_index():
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
                                    "rec_texts": ["因为", "2016"],
                                    "rec_scores": [0.96, 0.94],
                                    "rec_boxes": [
                                        [10, 10, 60, 42],
                                        [10, 60, 90, 92],
                                    ],
                                },
                                "text_word": [
                                    ["2", "0", "1", "6"],
                                    ["因", "为"],
                                ],
                                "text_word_region": [
                                    [
                                        [[10, 60], [26, 60], [26, 92], [10, 92]],
                                        [[28, 60], [44, 60], [44, 92], [28, 92]],
                                        [[46, 60], [62, 60], [62, 92], [46, 92]],
                                        [[64, 60], [80, 60], [80, 92], [64, 92]],
                                    ],
                                    [
                                        [[10, 10], [30, 10], [30, 42], [10, 42]],
                                        [[34, 10], [54, 10], [54, 42], [34, 42]],
                                    ],
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
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        image[10:42, 10:30] = 0
        image[10:42, 34:54] = 0
        image[60:92, 10:80] = 0
        lines = engine.recognize(image, OcrContext())
        assert [line.text for line in lines] == ["因为", "2016"]
        assert lines[0].chars[0].token_text == "因"
        assert lines[0].chars[1].token_text == "为"
        assert lines[1].chars[0].token_text == "2"
        assert lines[1].chars[-1].token_text == "6"
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_aligns_token_rows_by_bbox_not_index PASSED")


def test_api_ocr_engine_uses_matching_token_row_when_rec_bbox_missing():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.core.char_bbox_utils import MISSING_LINE_BBOX_FLAG
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import ApiOcrEngine
    from app.models import BBox

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
                                    "rec_texts": ["因为", "错配"],
                                    "rec_scores": [0.96, 0.94],
                                    "rec_boxes": [],
                                },
                                "text_word": [
                                    ["因", "为"],
                                    ["不", "同"],
                                ],
                                "text_word_region": [
                                    [
                                        [[10, 10], [30, 10], [30, 42], [10, 42]],
                                        [[34, 10], [54, 10], [54, 42], [34, 42]],
                                    ],
                                    [
                                        [[10, 60], [30, 60], [30, 92], [10, 92]],
                                        [[34, 60], [54, 60], [54, 92], [34, 92]],
                                    ],
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
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        image[10:42, 10:30] = 0
        image[10:42, 34:54] = 0
        lines = engine.recognize(image, OcrContext())
        assert len(lines) == 2
        assert lines[0].text == "因为"
        assert lines[0].bbox == BBox(10, 10, 44, 32)
        assert lines[0].chars[0].token_text == "因"
        assert lines[0].chars[1].token_text == "为"
        assert lines[1].text == "错配"
        assert lines[1].bbox == BBox(0, 0, 120, 120)
        assert MISSING_LINE_BBOX_FLAG in lines[1].review_flags
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_uses_matching_token_row_when_rec_bbox_missing PASSED")


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


def test_api_ocr_engine_preserves_token_text_when_rec_rows_missing():
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.core.ocr_ir import OCR_IR_TOKEN_TEXT_FALLBACK_FLAG
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
                                    "rec_texts": [],
                                    "rec_scores": [],
                                    "rec_boxes": [],
                                },
                                "text_word": [["漏", "字"]],
                                "text_word_region": [[
                                    [[10, 20], [28, 20], [28, 44], [10, 44]],
                                    [[32, 20], [50, 20], [50, 44], [32, 44]],
                                ]],
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
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        image[20:44, 10:28] = 0
        image[20:44, 32:50] = 0
        lines = engine.recognize(image, OcrContext())
        assert len(lines) == 1
        assert lines[0].text == "漏字"
        assert lines[0].bbox == BBox(10, 20, 40, 24)
        assert OCR_IR_TOKEN_TEXT_FALLBACK_FLAG in lines[0].review_flags
        assert lines[0].proof_status == ProofStatus.AUTO_FLAGGED
        assert [char.token_text for char in lines[0].chars] == ["漏", "字"]
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()

    print("test_api_ocr_engine_preserves_token_text_when_rec_rows_missing PASSED")


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


def test_ocr_pipeline_prefers_ocr_boxes_and_only_falls_back_for_missing_chars():
    import tempfile
    import cv2
    import numpy as np
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class PartialWordBoxEngine:
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
        result = OcrPipeline(engine=PartialWordBoxEngine()).process_project(
            OcrProject(name="PartialWordBox", pages=[page])
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


def test_page_ocr_refills_caption_and_equation_blocks():
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
        equation = Block(block_type=BlockType.EQUATION, bbox=BBox(0, 0, 120, 60), lines=[make_line("旧", 5, 5)], order=0)
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

        assert [line.text for line in blocks[0].lines] == ["式"]
        assert [line.text for line in blocks[1].lines] == ["图"]
        assert [line.text for line in blocks[2].lines] == ["表"]
        assert "旧" not in [line.text for block in blocks for line in block.lines]
    finally:
        os.unlink(img_path)

    print("test_page_ocr_refills_caption_and_equation_blocks PASSED")


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


def test_main_window_ocr_finished_preserves_current_step():
    from app.controllers.workflow_controller import STEP_HPROOF, STEP_OCR
    from app.models import Page
    from app.ui.main_window import MainWindow

    _get_qapp()
    window = MainWindow()
    try:
        window._controller.set_current_step(STEP_OCR)
        synced = []
        window._controller.sync_proof_panels = lambda *args, **kwargs: synced.append(True)  # type: ignore[method-assign]

        window._on_ocr_finished([Page(image_path="/tmp/ocr-finished.png", width=10, height=10)])

        assert synced == [True]
        assert window._controller.current_step == STEP_OCR
        assert window._stack.currentIndex() == STEP_OCR
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


def test_char_index_service():
    from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
    from app.services.char_index_service import CharEntry, CharIndexEntry, CharIndexService

    page = Page(image_path="/tmp/page.png", width=400, height=300)
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


def test_char_index_suppresses_punctuation_topic_for_shared_word_box():
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

    print("test_char_index_suppresses_punctuation_topic_for_shared_word_box PASSED")


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
    ]
    assert get_api_model_profile_url("pp-ocrv5").endswith("/ocr")
    assert get_api_model_profile_url("pp-structurev3").endswith("/layout-parsing")
    assert match_api_model_profile_from_url("https://n6z9feddjca4l7b5.aistudio-app.com/ocr") == "pp-ocrv5"
    assert match_api_model_profile_from_url("https://example.com/custom-layout") is None

    print("test_api_model_profile_helpers PASSED")


def test_api_endpoint_role_resolution_keeps_layout_and_proof_separate():
    """Layout role 已全面切到 PaddleOCR-VL-1.5；OCR proof role 仍走 PP-OCRv5。

    所有官方预设 (pp-ocrv5 / pp-structurev3 / paddleocr-vl) 在 role="layout"
    下都被 strong-redirect 到 paddleocr-vl-1.5 预设 URL。
    自托管根 URL 仍只做 /ocr <-> /layout-parsing 后缀切换。
    """
    from app.core.api_profiles import (
        get_api_model_profile_url,
        normalize_api_base_url,
        resolve_api_endpoint_for_role,
    )

    vl15_url = get_api_model_profile_url("paddleocr-vl-1.5")
    ocr_url = get_api_model_profile_url("pp-ocrv5")
    structure_url = get_api_model_profile_url("pp-structurev3")
    structure_root = structure_url.removesuffix("/layout-parsing")
    vl15_root = vl15_url.removesuffix("/layout-parsing")
    ocr_root = ocr_url.removesuffix("/ocr")
    vl_root = vl15_url.removesuffix("/layout-parsing")

    assert normalize_api_base_url(structure_url) == structure_root
    assert normalize_api_base_url(ocr_url) == ocr_root

    # Layout role: 任何官方 PP-* 预设 -> VL-1.5
    assert resolve_api_endpoint_for_role(
        structure_url,
        profile="pp-structurev3",
        role="layout",
    ) == vl15_url
    assert resolve_api_endpoint_for_role(
        ocr_url,
        profile="pp-ocrv5",
        role="layout",
    ) == vl15_url
    assert resolve_api_endpoint_for_role(
        structure_root,
        profile="pp-structurev3",
        role="layout",
    ) == vl15_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="pp-ocrv5",
        role="layout",
    ) == vl15_url
    # 旧 paddleocr-vl 预设也归入 VL-1.5（统一升级到 1.5）
    old_vl_url = get_api_model_profile_url("paddleocr-vl")
    assert resolve_api_endpoint_for_role(
        old_vl_url,
        profile="paddleocr-vl",
        role="layout",
    ) == vl15_url
    # VL-1.5 自身保持
    assert resolve_api_endpoint_for_role(
        vl15_url,
        profile="paddleocr-vl-1.5",
        role="layout",
    ) == vl15_url

    # OCR role: 任何 layout 预设 -> PP-OCRv5；PP-OCRv5 自身保持
    assert resolve_api_endpoint_for_role(
        structure_url,
        profile="pp-structurev3",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        vl15_url,
        profile="paddleocr-vl-1.5",
        role="ocr",
    ) == ocr_url
    assert resolve_api_endpoint_for_role(
        structure_root,
        profile="",
        role="layout",
    ) == vl15_url
    assert resolve_api_endpoint_for_role(
        vl15_root,
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
    ) == vl15_url
    assert resolve_api_endpoint_for_role(
        ocr_root,
        profile="",
        role="layout",
    ) == vl15_url

    # 自托管根 URL: 不做 host 跳转，仅按后缀切换
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com",
        profile="",
        role="layout",
    ) == "https://self-hosted.example.com/layout-parsing"
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com/layout-parsing",
        profile="",
        role="ocr",
    ) == "https://self-hosted.example.com/ocr"
    assert resolve_api_endpoint_for_role(
        "https://self-hosted.example.com/ocr",
        profile="",
        role="layout",
    ) == "https://self-hosted.example.com/layout-parsing"

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
    assert defaults["mode"] == "api"
    assert defaults["api_model_profile"] == ""

    update_config(
        mode="api",
        api_model_profile="paddleocr-vl-1.5",
        api_url="https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
        api_token="demo",
        api_timeout=12,
        api_layout_model_name="",
    )
    current = get_config()
    assert current["api_model_profile"] == "paddleocr-vl-1.5"
    assert current["api_url"] == "https://15j75bd0964dzbwe.aistudio-app.com"
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
    assert dialog._radio_api.isChecked()
    assert dialog._api_model_row.isHidden()
    assert dialog._api_model_combo.currentIndex() == -1
    assert dialog._url_edit.text() == "https://example.com/root"
    assert "固定双模型" in dialog._summary_model.text()

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
        assert dialog._api_form_panel.isEnabled()
        assert "API 双模型链" in dialog._summary_mode.text()

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_keeps_model_preset_sync PASSED")


def test_api_settings_dialog_reverse_matches_url_and_persists_profile():
    from app.core.app_config import AppConfig
    from app.core.ocr_config import get_config
    from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._url_edit.setText("https://example.com/custom")
        dialog._token_edit.setText("secret")
        dialog._sync_model_from_url()
        assert dialog._api_model_combo.currentIndex() == -1
        assert "固定双模型" in dialog._summary_model.text()

        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["mode"] == "api"
        assert cfg["api_model_profile"] == ""
        assert cfg["api_url"] == "https://example.com/custom"
        assert cfg["api_token"] == "secret"

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_reverse_matches_url_and_persists_profile PASSED")


def test_api_settings_dialog_saves_base_url_from_endpoint_suffix():
    from app.core.app_config import AppConfig
    from app.core.ocr_config import get_config
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


def test_fixed_api_chain_resolves_official_roots_to_vl15_and_ppocrv5():
    from app.core.api_profiles import get_api_model_profile_url, resolve_api_endpoint_for_role

    vl15_url = get_api_model_profile_url("paddleocr-vl-1.5")
    ppocr_url = get_api_model_profile_url("pp-ocrv5")
    vl15_root = vl15_url.removesuffix("/layout-parsing")
    ppocr_root = ppocr_url.removesuffix("/ocr")

    assert resolve_api_endpoint_for_role(vl15_root, role="layout") == vl15_url
    assert resolve_api_endpoint_for_role(vl15_root, role="ocr") == ppocr_url
    assert resolve_api_endpoint_for_role(ppocr_root, role="layout") == vl15_url
    assert resolve_api_endpoint_for_role(ppocr_root, role="ocr") == ppocr_url

    print("test_fixed_api_chain_resolves_official_roots_to_vl15_and_ppocrv5 PASSED")


def test_api_settings_dialog_persists_llm_candidate_settings():
    from app.core.app_config import AppConfig
    from app.core.ocr_config import get_config
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
        BlockType.TITLE,
        BlockType.TABLE_CAPTION,
        BlockType.FIGURE,
        BlockType.REFERENCE,
    ]
    assert blocks[0].bbox.x == 10 and blocks[0].bbox.y == 20
    assert blocks[1].bbox.w == 160 and blocks[1].bbox.h == 60
    assert "score=0.910" in blocks[0].note
    assert "参考文献" in blocks[3].note
    assert len(overlays) == 4

    print("test_layout_analyzer_extracts_api_blocks_from_varied_schema PASSED")


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


def test_inspector_structure_ocr_falls_back_when_ppstructure_pipeline_missing():
    import sys
    import types

    from tools.ocr_inspector.ui.panels.run_ocr import _run_structure_ocr

    class FakeResult:
        def json(self):
            return {
                "overall_ocr_res": {
                    "rec_texts": ["兼容回退"],
                    "rec_boxes": [[10, 20, 80, 40]],
                }
            }

    class FakePPStructureV3:
        def __init__(self, **kwargs):
            raise RuntimeError(
                "The pipeline (PP-StructureV3) does not exist! Please use a pipeline name or a config file path!"
            )

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path, **kwargs):
            return [FakeResult()]

    fake_module = types.SimpleNamespace(
        PPStructureV3=FakePPStructureV3,
        PaddleOCR=FakePaddleOCR,
    )

    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        raw = _run_structure_ocr(
            "/tmp/sample.png",
            {
                "structure": {"layout_threshold": 0.5},
                "ocr_init": {"lang": "ch", "ocr_version": None},
                "ocr_pred": {"return_word_box": True},
            },
        )
    finally:
        if original is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = original

    assert raw["overall_ocr_res"]["rec_texts"] == ["兼容回退"]
    assert raw["_inspector_meta"]["fallback"] == "paddleocr_layout_compat"
    assert "PP-StructureV3" in raw["_inspector_meta"]["reason"]

    print("test_inspector_structure_ocr_falls_back_when_ppstructure_pipeline_missing PASSED")


def test_inspector_flattens_api_layout_parsing_result():
    from tools.ocr_inspector.ui.panels.run_ocr import _flatten_api_result

    raw = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {"block_label": "paragraph", "block_bbox": [1, 2, 30, 40]},
                        ],
                        "layout_det_res": {
                            "boxes": [
                                {"label": "paragraph", "coordinate": [1, 2, 30, 40], "score": 0.9},
                            ],
                        },
                        "overall_ocr_res": {
                            "rec_texts": ["第一行"],
                            "rec_boxes": [[1, 2, 30, 20]],
                            "text_word": [["第一行"]],
                            "text_word_region": [[[1, 2, 30, 20]]],
                        },
                    }
                }
            ]
        }
    }

    flattened = _flatten_api_result(raw)

    assert flattened["parsing_res_list"][0]["block_label"] == "paragraph"
    assert flattened["layout_det_res"]["boxes"][0]["label"] == "paragraph"
    assert flattened["overall_ocr_res"]["rec_texts"] == ["第一行"]
    assert flattened["overall_ocr_res"]["text_word"][0] == ["第一行"]

    print("test_inspector_flattens_api_layout_parsing_result PASSED")


def test_inspector_flattens_api_pruned_word_boxes_for_adapter_chars():
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter
    from tools.ocr_inspector.ui.panels.run_ocr import _flatten_api_result

    raw = {
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "overall_ocr_res": {
                            "rec_texts": ["测试"],
                            "rec_scores": [0.96],
                            "rec_boxes": [[10, 20, 70, 50]],
                        },
                        "text_word": [["测", "试"]],
                        "text_word_region": [
                            [
                                [[10, 20], [35, 20], [35, 50], [10, 50]],
                                [[36, 20], [70, 20], [70, 50], [36, 50]],
                            ]
                        ],
                    }
                }
            ]
        }
    }

    flattened = _flatten_api_result(raw)
    doc = PaddleAdapter().parse(flattened)
    line = doc.pages[0].all_lines[0]

    assert flattened["text_word"][0] == ["测", "试"]
    assert len(line.chars) == 2
    assert all(ch.bbox_source == "ocr" for ch in line.chars)
    assert all(ch.bbox_granularity == "char" for ch in line.chars)

    print("test_inspector_flattens_api_pruned_word_boxes_for_adapter_chars PASSED")


def test_inspector_local_flatteners_preserve_word_box_rows():
    from tools.ocr_inspector.ui.panels.run_ocr import (
        _flatten_paddle_result,
        _flatten_structure_result,
    )

    class FakeResult:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    payload = {
        "parsing_res_list": [
            {"block_label": "text", "block_bbox": [1, 2, 80, 40], "block_content": "本地"},
        ],
        "overall_ocr_res": {
            "rec_texts": ["本地"],
            "rec_scores": [0.92],
            "rec_boxes": [[1, 2, 80, 40]],
        },
        "text_word": [["本", "地"]],
        "text_word_region": [
            [
                [[1, 2], [30, 2], [30, 40], [1, 40]],
                [[31, 2], [80, 2], [80, 40], [31, 40]],
            ]
        ],
    }

    paddle_flattened = _flatten_paddle_result([FakeResult(payload)])
    structure_flattened = _flatten_structure_result([FakeResult(payload)])
    direct_flattened = _flatten_paddle_result([FakeResult({
        "rec_texts": ["直出"],
        "rec_boxes": [[2, 3, 40, 20]],
        "textWord": [["直", "出"]],
        "textWordRegion": [
            [
                [[2, 3], [20, 3], [20, 20], [2, 20]],
                [[21, 3], [40, 3], [40, 20], [21, 20]],
            ]
        ],
    })])
    boxes_flattened = _flatten_paddle_result([FakeResult({
        "rec_texts": ["源码"],
        "rec_boxes": [[3, 4, 50, 24]],
        "text_word": [["源", "码"]],
        "text_word_region": [],
        "text_word_boxes": [
            [
                [3, 4, 25, 24],
                [26, 4, 50, 24],
            ]
        ],
    })])

    assert paddle_flattened["text_word"][0] == ["本", "地"]
    assert paddle_flattened["overall_ocr_res"]["rec_texts"] == ["本地"]
    assert structure_flattened["text_word_region"][0][0][0] == [1, 2]
    assert structure_flattened["parsing_res_list"][0]["block_label"] == "text"
    assert direct_flattened["overall_ocr_res"]["rec_texts"] == ["直出"]
    assert direct_flattened["text_word"][0] == ["直", "出"]
    assert boxes_flattened["text_word_region"][0] == [[3, 4, 25, 24], [26, 4, 50, 24]]

    print("test_inspector_local_flatteners_preserve_word_box_rows PASSED")


def test_inspector_builds_api_request_body_from_shared_config():
    from tools.ocr_inspector.ui.panels.run_ocr import _build_api_request_body

    body = _build_api_request_body(
        "abc123",
        {
            "ocr_init": {
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            },
            "ocr_pred": {
                "return_word_box": True,
                "text_det_thresh": 0.25,
                "text_det_box_thresh": 0.55,
                "text_det_unclip_ratio": 1.4,
                "text_det_limit_side_len": 960,
                "text_det_limit_type": "max",
                "text_rec_score_thresh": 0.2,
            },
        },
        {"api_layout_model_name": "PP-StructureV3"},
    )

    assert body["file"] == "abc123"
    assert body["fileType"] == 1
    assert body["model_name"] == "PP-StructureV3"
    assert body["returnWordBox"] is True
    assert body["useDocOrientationClassify"] is True
    assert body["useDocUnwarping"] is False
    assert body["useTextlineOrientation"] is True
    assert body["textDetThresh"] == 0.25
    assert body["textDetBoxThresh"] == 0.55
    assert body["textDetUnclipRatio"] == 1.4
    assert body["textDetLimitSideLen"] == 960
    assert body["textDetLimitType"] == "max"
    assert body["textRecScoreThresh"] == 0.2

    print("test_inspector_builds_api_request_body_from_shared_config PASSED")


def test_inspector_runtime_meta_records_actual_request_and_response_fields():
    from tools.ocr_inspector.core import build_paddle_document
    from tools.ocr_inspector.ui.panels.run_ocr import _attach_inspector_runtime_meta, _summarize_relevant_response_fields

    raw = {
        "overall_ocr_res": {
            "rec_texts": ["测"],
            "rec_boxes": [[10, 20, 40, 50]],
        }
    }

    _attach_inspector_runtime_meta(
        raw,
        source="api",
        pipeline="aistudio",
        image_path="/tmp/test.jpg",
        request_summary={"returnWordBox": True, "textDetUnclipRatio": 2.0},
        response_raw={"result": {"ocrResults": [{"prunedResult": raw}]}},
        api_url="https://example.test/ocr",
        api_model_profile="pp-ocrv5",
    )
    result = build_paddle_document(raw, image_path="/tmp/test.jpg")
    log_text = "\n".join(str(entry) for entry in result.document.parse_log)
    codes = {diag.code for diag in result.diagnostics}

    assert raw["_inspector_meta"]["request_summary"]["returnWordBox"] is True
    assert raw["_inspector_meta"]["response_field_summary"]["ocrResults"] == 1
    assert raw["_inspector_meta"]["flattened_field_summary"]["rec_texts"] == 1
    assert "text_word_boxes" not in _summarize_relevant_response_fields(raw)
    assert "returnWordBox=True" in log_text
    assert "response-fields" in log_text
    assert "server_missing_text_word_region" in codes

    print("test_inspector_runtime_meta_records_actual_request_and_response_fields PASSED")


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
# OCR Inspector — crop module + char bbox source tests
# =====================================================================

def test_crop_bbox_ascii_path():
    """crop_bbox should work on ASCII paths and return a PIL Image."""
    import tempfile, os, shutil
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("test_crop_bbox_ascii_path SKIPPED (cv2 not installed)")
        return
    from tools.ocr_inspector.crop import crop_bbox
    from tools.ocr_inspector.models.ir import BBox
    tmpdir = tempfile.mkdtemp()
    img_path = os.path.join(tmpdir, "test.png")
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[20:60, 50:150] = [0, 128, 255]
    cv2.imwrite(img_path, arr)
    bbox = BBox(x=50, y=20, w=100, h=40)
    crop = crop_bbox(img_path, bbox)
    assert crop.size == (100, 40), crop.size
    crop_pad = crop_bbox(img_path, bbox, padding=5)
    assert crop_pad.size == (110, 50), crop_pad.size
    shutil.rmtree(tmpdir)
    print("test_crop_bbox_ascii_path PASSED")


def test_crop_bbox_chinese_path():
    """crop_bbox must handle Chinese directory names."""
    import tempfile, os, shutil
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("test_crop_bbox_chinese_path SKIPPED (cv2 not installed)")
        return
    from tools.ocr_inspector.crop import crop_bbox
    from tools.ocr_inspector.models.ir import BBox
    tmpdir = tempfile.mkdtemp()
    cn_dir = os.path.join(tmpdir, "纵校测试")
    os.makedirs(cn_dir, exist_ok=True)
    img_path = os.path.join(cn_dir, "120167.png")
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(img_path, arr)
    bbox = BBox(x=10, y=10, w=80, h=40)
    crop = crop_bbox(img_path, bbox)
    assert crop.size == (80, 40), crop.size
    shutil.rmtree(tmpdir)
    print("test_crop_bbox_chinese_path PASSED")


def test_find_text_matches_lines():
    """find_text_matches finds line-level matches."""
    from tools.ocr_inspector.crop import find_text_matches
    from tools.ocr_inspector.models.ir import BBox, DocumentNode, LineNode, PageNode
    doc = DocumentNode.make(source_path="test.json")
    page = PageNode.make(page_number=1, image_path="/tmp/test.jpg")
    doc.pages.append(page)
    bbox = BBox(x=10, y=20, w=200, h=30)
    line = LineNode.make(text="测试文字", confidence=0.95, bbox=bbox)
    page.orphan_lines.append(line)
    matches = find_text_matches(doc, "测试", search_chars=False, search_lines=True)
    assert len(matches) == 1 and matches[0].kind == "line" and matches[0].bbox == bbox
    assert len(find_text_matches(doc, "", search_lines=True)) == 0
    print("test_find_text_matches_lines PASSED")


def test_find_text_matches_ocr_chars():
    """find_text_matches finds OCR-true chars, skips fallback chars."""
    from tools.ocr_inspector.crop import find_text_matches
    from tools.ocr_inspector.models.ir import BBox, CharNode, DocumentNode, LineNode, PageNode
    doc = DocumentNode.make(source_path="test.json")
    page = PageNode.make(page_number=1, image_path="/tmp/test.jpg")
    doc.pages.append(page)
    line_bbox = BBox(x=10, y=20, w=200, h=30)
    line = LineNode.make(text="测试", confidence=0.95, bbox=line_bbox)
    c_real = CharNode.make(char="测", bbox=BBox(10, 20, 20, 30),
        confidence=0.95, bbox_source="ocr", bbox_granularity="char", token_text="测")
    c_fb = CharNode.make(char="试", bbox=line_bbox,
        confidence=0.95, bbox_source="fallback", bbox_granularity="line", token_text="试")
    line.chars = [c_real, c_fb]
    page.orphan_lines.append(line)
    m = find_text_matches(doc, "测", search_chars=True, search_lines=False)
    assert len(m) == 1 and m[0].kind == "char"
    m2 = find_text_matches(doc, "试", search_chars=True, search_lines=False)
    assert len(m2) == 0, f"fallback char should not match, got {len(m2)}"
    print("test_find_text_matches_ocr_chars PASSED")


def test_planb_core_seam_diagnoses_missing_word_region():
    from tools.ocr_inspector.core import build_paddle_document, raw_contains_word_regions

    raw = {"overall_ocr_res": {
        "rec_texts": ["测试"],
        "rec_boxes": [[10, 20, 80, 50]],
        "rec_scores": [0.95],
    }}

    result = build_paddle_document(raw, image_path="/tmp/test.jpg")
    doc = result.document
    line = doc.pages[0].all_lines[0]
    codes = {diag.code for diag in result.diagnostics}

    assert raw_contains_word_regions(raw) is False
    assert "server_missing_text_word_region" in codes
    assert all(ch.bbox_source == "unavailable" and ch.bbox is None for ch in line.chars)

    print("test_planb_core_seam_diagnoses_missing_word_region PASSED")


def test_planb_core_seam_preserves_word_region_and_search_identity():
    from tools.ocr_inspector.core import build_paddle_document, query_text_search_index, raw_contains_word_regions

    raw = {
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "overall_ocr_res": {
                            "rec_texts": ["南京市"],
                            "rec_boxes": [[10, 20, 100, 50]],
                            "rec_scores": [0.96],
                        },
                        "text_word": [["南京市"]],
                        "text_word_region": [[[10, 20, 100, 20, 100, 50, 10, 50]]],
                    }
                }
            ]
        }
    }

    result = build_paddle_document(raw, image_path="/tmp/test.jpg")
    doc = result.document
    line = doc.pages[0].all_lines[0]
    matches = query_text_search_index(doc, "南京", include_lines=False)
    codes = {diag.code for diag in result.diagnostics}

    assert raw_contains_word_regions(raw) is True
    assert "word_regions_preserved" in codes
    assert all(ch.bbox_source == "ocr" for ch in line.chars)
    assert all(ch.bbox_granularity == "word" for ch in line.chars)
    assert len(matches) == 1
    assert matches[0].identity.startswith("p0:l0:token0:")
    assert matches[0].bbox_source == "ocr"
    assert matches[0].bbox_granularity == "word"

    print("test_planb_core_seam_preserves_word_region_and_search_identity PASSED")


def test_planb_core_seam_reads_paddle_json_text_word_boxes_alias():
    from tools.ocr_inspector.core import build_paddle_document, query_text_search_index, raw_contains_word_regions

    raw = {
        "overall_ocr_res": {
            "rec_texts": ["源码"],
            "rec_boxes": [[10, 20, 80, 50]],
            "rec_scores": [0.97],
        },
        "text_word": [["源", "码"]],
        "text_word_region": [],
        "text_word_boxes": [[[10, 20, 40, 50], [41, 20, 80, 50]]],
    }

    result = build_paddle_document(raw, image_path="/tmp/test.jpg")
    doc = result.document
    line = doc.pages[0].all_lines[0]
    matches = query_text_search_index(doc, "源", include_lines=False)
    codes = {diag.code for diag in result.diagnostics}

    assert raw_contains_word_regions(raw) is True
    assert "word_regions_preserved" in codes
    assert len(line.chars) == 2
    assert all(ch.bbox_source == "ocr" for ch in line.chars)
    assert all(ch.bbox_granularity == "char" for ch in line.chars)
    assert line.chars[0].bbox is not None and line.chars[0].bbox.x == 10
    assert len(matches) == 1
    assert matches[0].identity.startswith("p0:l0:token0:")

    print("test_planb_core_seam_reads_paddle_json_text_word_boxes_alias PASSED")


def test_planb_core_seam_flags_invalid_word_region():
    from tools.ocr_inspector.core import build_paddle_document, raw_contains_word_regions

    raw = {"overall_ocr_res": {
        "rec_texts": ["测"],
        "rec_boxes": [[10, 20, 80, 50]],
        "rec_scores": [0.95],
    }, "text_word": [["测"]], "text_word_region": [[["bad-region"]]]}

    result = build_paddle_document(raw, image_path="/tmp/test.jpg")
    codes = {diag.code for diag in result.diagnostics}
    char = result.document.pages[0].all_lines[0].chars[0]

    assert raw_contains_word_regions(raw) is True
    assert "invalid_region_format" in codes
    assert "word_regions_not_consumed" in codes
    assert char.bbox_source == "unavailable"

    print("test_planb_core_seam_flags_invalid_word_region PASSED")


def test_crop_panel_search_selects_canvas_node():
    """Selecting a text-search result must update AppState so canvas/tree can highlight it."""
    from tools.ocr_inspector.models.ir import BBox, DocumentNode, LineNode, PageNode
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.panels.crop_panel import CropPanel

    _get_qapp()

    doc = DocumentNode.make(source_path="test.json")
    page = PageNode.make(page_number=1, image_path="")
    doc.pages.append(page)
    line = LineNode.make(text="测试文字", confidence=0.95, bbox=BBox(x=10, y=20, w=200, h=30))
    page.orphan_lines.append(line)

    state = AppState()
    state.set_document(doc)
    panel = CropPanel(state)
    panel.set_query("测试")

    assert state.active_document is doc
    assert state.selected_node is line
    panel.close()
    print("test_crop_panel_search_selects_canvas_node PASSED")


def test_paddle_adapter_char_bbox_source():
    """PaddleAdapter sets bbox_source=ocr for text_word_region chars."""
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter
    raw = {"overall_ocr_res": {
        "rec_texts": ["测试"], "rec_boxes": [[10, 20, 210, 50]], "rec_scores": [0.95],
        "rec_polys": [],
        "text_word": [["测", "试"]],
        "text_word_region": [
            [[[10,20],[30,20],[30,50],[10,50]], [[31,20],[60,20],[60,50],[31,50]]]
        ],
    }}
    doc = PaddleAdapter().parse(raw, image_path="/tmp/test.jpg")
    line = doc.pages[0].all_lines[0]
    assert len(line.chars) == 2
    assert all(c.bbox_source == "ocr" for c in line.chars)
    assert line.chars[0].bbox != line.chars[1].bbox
    print("test_paddle_adapter_char_bbox_source PASSED")


def test_paddle_adapter_reads_direct_and_camelcase_word_rows():
    """PaddleAdapter accepts flattened top-level and camelCase word-box fields."""
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter

    raw = {
        "rec_texts": ["天地"],
        "rec_boxes": [[10, 20, 70, 50]],
        "rec_scores": [0.95],
        "textWord": [["天", "地"]],
        "textWordRegion": [
            [
                [[10, 20], [40, 20], [40, 50], [10, 50]],
                [[41, 20], [70, 20], [70, 50], [41, 50]],
            ]
        ],
    }

    doc = PaddleAdapter().parse(raw, image_path="/tmp/test.jpg")
    line = doc.pages[0].all_lines[0]

    assert line.text == "天地"
    assert len(line.chars) == 2
    assert all(ch.bbox_source == "ocr" for ch in line.chars)
    assert line.chars[0].bbox != line.chars[1].bbox

    print("test_paddle_adapter_reads_direct_and_camelcase_word_rows PASSED")


def test_paddle_adapter_char_fallback_when_no_word_region():
    """PaddleAdapter keeps char bbox unavailable when text_word_region is absent."""
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter
    raw = {"overall_ocr_res": {
        "rec_texts": ["测试"], "rec_boxes": [[10, 20, 210, 50]], "rec_scores": [0.95],
    }}
    doc = PaddleAdapter().parse(raw, image_path="/tmp/test.jpg")
    line = doc.pages[0].all_lines[0]
    assert len(line.chars) == 2
    for ch in line.chars:
        assert ch.bbox_source == "unavailable"
        assert ch.bbox_granularity == "unavailable"
        assert ch.bbox is None
    print("test_paddle_adapter_char_fallback_when_no_word_region PASSED")



# =====================================================================
# Canvas node_item_map + char selection tests (no Qt required)
# =====================================================================

def test_canvas_node_item_map_char_fallback():
    """All fallback chars sharing a bbox must be mapped in _node_item_map."""
    from tools.ocr_inspector.models.ir import BBox, CharNode, LineNode, PageNode, DocumentNode
    # Simulate what _draw_chars does:  seen_fallback maps bbox_key→item
    # and ALL char nodes get mapped to that item.
    page = PageNode.make(page_number=1, image_path="")
    line_bbox = BBox(x=10, y=20, w=200, h=30)
    line = LineNode.make(text="AB", confidence=0.9, bbox=line_bbox)
    c1 = CharNode.make(char="A", bbox=line_bbox, confidence=0.9, bbox_source="fallback", bbox_granularity="line", token_text="A")
    c2 = CharNode.make(char="B", bbox=line_bbox, confidence=0.9, bbox_source="fallback", bbox_granularity="line", token_text="B")
    line.chars = [c1, c2]
    page.orphan_lines.append(line)

    # Replicate the dedup logic from _draw_chars
    seen_fallback: dict = {}
    node_item_map: dict = {}
    _sentinel = object()  # stand-in for a canvas item

    for char in page.all_chars:
        if char.bbox:
            key = (char.bbox.x, char.bbox.y, char.bbox.w, char.bbox.h)
            bs = getattr(char, "bbox_source", "") or "fallback"
            is_ocr = (bs == "ocr")
            if not is_ocr:
                if key not in seen_fallback:
                    item = object()  # stand-in
                    seen_fallback[key] = item
                node_item_map[id(char)] = seen_fallback[key]

    # Both chars should be in node_item_map
    assert id(c1) in node_item_map, "c1 not mapped"
    assert id(c2) in node_item_map, "c2 not mapped"
    # Both point to the same display item
    assert node_item_map[id(c1)] is node_item_map[id(c2)], "c1 and c2 should share item"
    print("test_canvas_node_item_map_char_fallback PASSED")


def test_canvas_node_item_map_char_ocr():
    """OCR-true chars with distinct token_text get separate items."""
    from tools.ocr_inspector.models.ir import BBox, CharNode, LineNode, PageNode
    page = PageNode.make(page_number=1, image_path="")
    bbox_a = BBox(x=10, y=20, w=20, h=30)
    bbox_b = BBox(x=30, y=20, w=20, h=30)
    line = LineNode.make(text="AB", confidence=0.9, bbox=BBox(10, 20, 40, 30))
    c1 = CharNode.make(char="A", bbox=bbox_a, confidence=0.9, bbox_source="ocr", bbox_granularity="char", token_text="A")
    c2 = CharNode.make(char="B", bbox=bbox_b, confidence=0.9, bbox_source="ocr", bbox_granularity="char", token_text="B")
    line.chars = [c1, c2]
    page.orphan_lines.append(line)

    # Replicate _draw_chars OCR dedup logic
    seen_ocr: dict = {}
    node_item_map: dict = {}
    for char in page.all_chars:
        if char.bbox:
            b = char.bbox
            coord_key = (b.x, b.y, b.w, b.h)
            bs = getattr(char, "bbox_source", "") or "fallback"
            if bs == "ocr":
                tok = getattr(char, "token_text", char.char) or char.char
                key = (coord_key, tok)
                if key not in seen_ocr:
                    seen_ocr[key] = object()  # stand-in item
                node_item_map[id(char)] = seen_ocr[key]

    assert id(c1) in node_item_map, "c1 not mapped"
    assert id(c2) in node_item_map, "c2 not mapped"
    # Different bbox → different items
    assert node_item_map[id(c1)] is not node_item_map[id(c2)], "distinct bboxes should get distinct items"
    print("test_canvas_node_item_map_char_ocr PASSED")


def test_canvas_char_fallback_source_colour():
    """SOURCE_COLOURS must contain char_fallback key."""
    from tools.ocr_inspector.ui.canvas import SOURCE_COLOURS
    assert "char_fallback" in SOURCE_COLOURS, f"char_fallback missing from SOURCE_COLOURS: {list(SOURCE_COLOURS)}"
    assert "text_word_region" in SOURCE_COLOURS
    print("test_canvas_char_fallback_source_colour PASSED")


def test_canvas_char_overlay_default_visible():
    """OCR char/token bbox layer should be visible without requiring the user to discover F3 first."""
    from tools.ocr_inspector.models.ir import BBox, CharNode, LineNode, PageNode
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.canvas import OcrCanvas

    _get_qapp()

    state = AppState()
    assert state.overlay_flags["chars"] is True
    page = PageNode.make(page_number=1, image_path="")
    line = LineNode.make(text="测", confidence=0.95, bbox=BBox(10, 20, 40, 30))
    char = CharNode.make(
        char="测",
        bbox=BBox(10, 20, 20, 30),
        confidence=0.95,
        bbox_source="ocr",
        bbox_granularity="char",
        token_text="测",
    )
    line.chars = [char]
    page.orphan_lines.append(line)

    canvas = OcrCanvas(state)
    canvas.load_page(page)
    item = canvas._node_item_map.get(id(char))

    assert item is not None
    assert item.isVisible()
    assert item._source_field == "text_word_region"
    canvas.close()
    print("test_canvas_char_overlay_default_visible PASSED")


def test_canvas_shows_unavailable_note_without_word_region():
    """When no text_word_region exists, canvas must show a visible unavailable note instead of faking char boxes."""
    from PySide6.QtWidgets import QGraphicsTextItem

    from tools.ocr_inspector.models.ir import BBox, CharNode, LineNode, PageNode
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.canvas import OcrCanvas

    _get_qapp()

    state = AppState()
    page = PageNode.make(page_number=1, image_path="")
    line = LineNode.make(text="测", confidence=0.95, bbox=BBox(10, 20, 40, 30))
    line.chars = [
        CharNode.make(
            char="测",
            bbox=None,
            confidence=0.95,
            bbox_source="unavailable",
            bbox_granularity="unavailable",
            token_text="测",
        )
    ]
    page.orphan_lines.append(line)

    canvas = OcrCanvas(state)
    canvas.load_page(page)
    notes = [
        item.toPlainText()
        for item in canvas._scene.items()
        if isinstance(item, QGraphicsTextItem)
    ]

    assert any("text_word_region" in note and "unavailable" in note for note in notes), notes
    assert id(line.chars[0]) not in canvas._node_item_map
    canvas.close()
    print("test_canvas_shows_unavailable_note_without_word_region PASSED")


def test_canvas_warning_reports_server_missing_after_return_word_box_sent():
    from PySide6.QtWidgets import QGraphicsTextItem

    from tools.ocr_inspector.adapters.paddle import PaddleAdapter
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.canvas import OcrCanvas
    from tools.ocr_inspector.ui.panels.run_ocr import _attach_inspector_runtime_meta

    _get_qapp()

    raw = {
        "overall_ocr_res": {
            "rec_texts": ["测"],
            "rec_boxes": [[10, 20, 40, 50]],
        }
    }
    _attach_inspector_runtime_meta(
        raw,
        source="api",
        pipeline="aistudio",
        image_path="/tmp/test.jpg",
        request_summary={"returnWordBox": True},
        response_raw={"result": {"ocrResults": [{"prunedResult": raw}]}},
        api_url="https://example.test/ocr",
        api_model_profile="pp-ocrv5",
    )
    doc = PaddleAdapter().parse(raw, image_path="")
    state = AppState()
    state.set_document(doc)

    canvas = OcrCanvas(state)
    canvas.load_page(doc.pages[0])
    notes = [
        item.toPlainText()
        for item in canvas._scene.items()
        if isinstance(item, QGraphicsTextItem)
    ]

    assert any("returnWordBox=true" in note and "响应没有 text_word_region/text_word_boxes" in note for note in notes), notes
    canvas.close()
    print("test_canvas_warning_reports_server_missing_after_return_word_box_sent PASSED")


def test_inspector_tree_syncs_external_char_selection():
    """Canvas-selected char nodes must already exist in the left tree and become current."""
    from tools.ocr_inspector.models.ir import BBox, CharNode, DocumentNode, LineNode, PageNode
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.panels import JsonTreePanel

    _get_qapp()

    doc = DocumentNode.make(source_path="test.json")
    page = PageNode.make(page_number=1, image_path="")
    doc.pages.append(page)
    line = LineNode.make(text="测", confidence=0.95, bbox=BBox(10, 20, 40, 30))
    char = CharNode.make(
        char="测",
        bbox=BBox(10, 20, 20, 30),
        confidence=0.95,
        bbox_source="ocr",
        bbox_granularity="char",
        token_text="测",
    )
    line.chars = [char]
    page.orphan_lines.append(line)

    state = AppState()
    tree = JsonTreePanel(state)
    state.set_document(doc)
    state.set_selection(char)

    assert tree.currentItem() is not None
    assert tree.currentItem().data(0, 0x0100) is char  # Qt.UserRole
    tree.close()
    print("test_inspector_tree_syncs_external_char_selection PASSED")


def test_params_ref_matrix_marks_vl_word_box_unsupported():
    from app.core.api_profiles import PADDLE_COORD_STABILITY_FLAGS, PADDLE_OCR_WORD_BOX_PARAMS
    from tools.ocr_inspector.ui.panels.params_ref import _PARAMS, build_paddle_param_matrix

    ocr_rows, ocr_payload = build_paddle_param_matrix("pp-ocrv5")
    ocr_map = {row.name: row for row in ocr_rows}
    assert ocr_map["returnWordBox"].sent is True
    assert ocr_map["returnWordBox"].current_value is PADDLE_OCR_WORD_BOX_PARAMS["returnWordBox"]
    assert ocr_map["returnWordBox"].default_value is PADDLE_OCR_WORD_BOX_PARAMS["returnWordBox"]
    assert ocr_payload["returnWordBox"] is True
    assert ocr_payload["textDetLimitSideLen"] == 1536
    assert ocr_map["textDetLimitSideLen"].current_value == PADDLE_OCR_WORD_BOX_PARAMS["textDetLimitSideLen"]
    assert ocr_map["textDetLimitSideLen"].default_value == PADDLE_OCR_WORD_BOX_PARAMS["textDetLimitSideLen"]
    assert ocr_payload["textDetUnclipRatio"] == 2.0
    assert ocr_map["textDetUnclipRatio"].default_value == PADDLE_OCR_WORD_BOX_PARAMS["textDetUnclipRatio"]
    assert ocr_map["useDocOrientationClassify"].current_value is PADDLE_COORD_STABILITY_FLAGS["useDocOrientationClassify"]
    assert ocr_map["useDocOrientationClassify"].default_value is PADDLE_COORD_STABILITY_FLAGS["useDocOrientationClassify"]
    assert ocr_map["useTextlineOrientation"].current_value is PADDLE_COORD_STABILITY_FLAGS["useTextlineOrientation"]
    assert ocr_map["useTextlineOrientation"].default_value is PADDLE_COORD_STABILITY_FLAGS["useTextlineOrientation"]

    vl_rows, vl_payload = build_paddle_param_matrix("paddleocr-vl")
    vl_map = {row.name: row for row in vl_rows}
    assert vl_map["returnWordBox"].sent is False
    assert vl_map["returnWordBox"].current_value == "unsupported"
    assert vl_map["returnWordBox"].default_value is PADDLE_OCR_WORD_BOX_PARAMS["returnWordBox"]
    assert "VL" in vl_map["returnWordBox"].reason
    assert "returnWordBox" not in vl_payload
    assert vl_payload["useDocUnwarping"] is False

    static_defaults = {
        entry[0]: entry[2]
        for entry in _PARAMS
        if entry[0] != "__cat__"
    }
    assert static_defaults["returnWordBox"] == "True"
    assert static_defaults["useDocOrientationClassify"] == "False"
    assert static_defaults["useTextlineOrientation"] == "False"
    assert static_defaults["textDetUnclipRatio"] == "2.0"
    assert static_defaults["textDetLimitSideLen"] == "1536"

    print("test_params_ref_matrix_marks_vl_word_box_unsupported PASSED")



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
    assert ocr_body["returnWordBox"] is True
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
    assert layout_body["returnWordBox"] is True

    vl_layout_body = LayoutAnalyzer()._build_api_request_body(
        "abc",
        1,
        profile="paddleocr-vl-1.5",
        endpoint_url="https://example.com/layout-parsing",
    )
    assert "returnWordBox" not in vl_layout_body
    assert vl_layout_body["useDocUnwarping"] is False

    print("test_api_request_builders_split_profile_params PASSED")


def test_ocr_inspector_run_panel_profile_request_params():
    from tools.ocr_inspector.state import AppState
    from tools.ocr_inspector.ui.panels.run_ocr import RunOcrPanel, _build_api_request_body

    _get_qapp()

    panel = RunOcrPanel(AppState())
    assert panel._return_word_box.isChecked() is True
    assert panel._det_unclip_ratio.value() == 2.0
    assert panel._det_limit_side_len.value() == 1536
    panel.close()

    params = {
        "ocr_init": {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        "ocr_pred": {
            "return_word_box": True,
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 2.0,
            "text_det_limit_side_len": 1536,
            "text_det_limit_type": "max",
            "text_rec_score_thresh": 0.0,
        },
    }

    ocr_body = _build_api_request_body(
        "abc",
        params,
        {
            "api_model_profile": "pp-ocrv5",
            "_resolved_api_url": "https://example.com/ocr",
        },
    )
    assert ocr_body["returnWordBox"] is True
    assert ocr_body["textDetLimitSideLen"] == 1536

    vl_body = _build_api_request_body(
        "abc",
        params,
        {
            "api_model_profile": "paddleocr-vl",
            "_resolved_api_url": "https://example.com/layout-parsing",
        },
    )
    assert vl_body["file"] == "abc"
    assert vl_body["fileType"] == 1
    assert vl_body["useDocUnwarping"] is False
    assert "returnWordBox" not in vl_body

    print("test_ocr_inspector_run_panel_profile_request_params PASSED")


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
    import tempfile

    import cv2
    import numpy as np
    import requests

    from app.core.app_config import AppConfig, update_config
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BlockType, Page

    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "overall_ocr_res": {
                                    "rec_texts": ["OCR行"],
                                    "rec_scores": [0.91],
                                    "rec_boxes": [[10, 20, 110, 50]],
                                },
                            },
                        },
                    ],
                },
            }

    def fake_post(url, json, headers, timeout, **kwargs):
        captured["url"] = url
        captured["json"] = json
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
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        cv2.imwrite(page_path, np.full((120, 200, 3), 255, dtype=np.uint8))
        page = Page(image_path=page_path, width=200, height=120)
        LayoutAnalyzer()._api_analyze(page)
        assert captured["url"] == "https://example.com/root/layout-parsing"
        assert captured["proxies"] == {"http": None, "https": None, "all": None}
        assert captured["json"]["useDocUnwarping"] is False
        assert "returnWordBox" not in captured["json"]
        assert "textDetLimitSideLen" not in captured["json"]
        assert len(page.blocks) == 1
        assert page.blocks[0].block_type == BlockType.TEXT
        assert "OCR行" in page.blocks[0].note
    finally:
        requests.post = original_post
        cfg.reset_to_defaults()
        os.unlink(page_path)

    print("test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected PASSED")


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
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
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

    print("test_workflow_controller_enables_proof_steps_after_first_ocr_page PASSED")


def test_proof_line_iterator_includes_caption_and_equation_lines():
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

    assert texts == ["正文", "图注", "E=mc2"]

    print("test_proof_line_iterator_includes_caption_and_equation_lines PASSED")


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


def test_hproof_line_iterator_excludes_position_source_labels():
    from app.core.proof_line_utils import iter_unique_page_hproof_lines
    from app.models import BBox, Block, BlockType, Line, Page

    page = Page(image_path="/tmp/hproof-position.png", width=100, height=100)
    page.blocks = [
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 1, 20, 10),
            lines=[Line(text="12", confidence=0.9, bbox=BBox(1, 1, 20, 10))],
            note="score=0.99 | source_label=page_number",
        ),
        Block(
            block_type=BlockType.TEXT,
            bbox=BBox(1, 20, 60, 12),
            lines=[Line(text="正文", confidence=0.9, bbox=BBox(1, 20, 60, 12))],
            note="source_label=text",
        ),
    ]

    texts = [line.text for _block, line, _idx in iter_unique_page_hproof_lines(page)]

    assert texts == ["正文"]

    print("test_hproof_line_iterator_excludes_position_source_labels PASSED")


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
    assert new_line.text == "用户未保存"
    assert old_line.text == "旧对象"
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
    # proof-bbox-boxedit round 9: thumb 从 27 抬到 34 修"压字"，sizeHint 上限同步 ≤ 50
    assert v_proof.GALLERY_THUMB >= 30
    assert v_proof.CHAR_LIST_THUMB == 18
    assert panel._left_box.maximumWidth() <= 170
    assert panel._gallery_view.itemDelegate().sizeHint(None, panel._gallery_model.index(0, 0)).height() <= 50
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
    # proof-layout-collections 第 3 任务：字号贴近行图字号。
    assert h_proof.TEXT_FONT_PX == 24
    assert h_proof.TEXT_EDITOR_MAX_H <= 42
    # proof-layout-collections 第 1 任务：第三行文本去掉，pair 高度
    # 回到「图 + editor」的紧凑值（image_row 32 + editor 36 + spacing/padding ≈ 76）。
    assert h_proof.LINE_PAIR_H == 76
    assert (
        h_proof.LINE_PAIR_H
        >= h_proof.IMAGE_ROW_H + h_proof.TEXT_EDITOR_MAX_H
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


def test_ocr_inspector_paddle_adapter_keeps_line_only_chars_unavailable():
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter

    raw = {
        "api_model_profile": "paddleocr-vl",
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "text",
                                    "coordinate": [8, 18, 72, 55],
                                    "score": 0.87,
                                }
                            ]
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "paragraph",
                                "block_bbox": [10, 20, 70, 50],
                                "block_content": "天地",
                            }
                        ],
                        "overall_ocr_res": {
                            "rec_texts": ["天地"],
                            "rec_scores": [0.91],
                            "rec_boxes": [[10, 20, 70, 50]],
                        }
                    }
                }
            ]
        },
    }

    doc = PaddleAdapter().parse(raw)
    page = doc.pages[0]
    assert page.blocks[0].source_field == "parsing_res_list"
    assert page.layout_det_blocks[0].source_field == "layout_det_res"
    assert page.layout_det_blocks[0].bbox.area > 0
    line = page.blocks[0].lines[0]
    assert line.bbox is not None
    assert line.chars[0].bbox is None
    assert line.chars[0].bbox_source == "unavailable"
    assert line.chars[0].bbox_granularity == "unavailable"
    assert any("max_text_bbox_granularity=line" in msg for msg in doc.parse_log)

    print("test_ocr_inspector_paddle_adapter_keeps_line_only_chars_unavailable PASSED")


def test_ocr_inspector_paddle_adapter_marks_word_not_fake_char():
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter

    raw = {
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "overall_ocr_res": {
                            "rec_texts": ["南京市"],
                            "rec_scores": [0.95],
                            "rec_polys": [[10, 20, 90, 20, 90, 50, 10, 50]],
                        },
                        "text_word": [["南京市"]],
                        "text_word_region": [[[10, 20, 90, 20, 90, 50, 10, 50]]],
                    }
                }
            ]
        }
    }

    doc = PaddleAdapter().parse(raw)
    line = doc.pages[0].orphan_lines[0]
    assert line.bbox.to_dict() == {"x": 10, "y": 20, "w": 80, "h": 30}
    assert len(line.chars) == 3
    assert all(ch.bbox_source == "ocr" for ch in line.chars)
    assert all(ch.bbox_granularity == "word" for ch in line.chars)
    assert all(ch.collection_kind == "token" for ch in line.chars)
    assert line.chars[0].bbox == line.chars[-1].bbox

    print("test_ocr_inspector_paddle_adapter_marks_word_not_fake_char PASSED")


if __name__ == "__main__":
    test_models()
    test_bbox_tools()
    test_block_type_mapping()
    test_project_store()
    test_project_store_clean_on_resave()
    test_project_store_schema_migration()
    test_proof_engine()
    test_export_txt()
    test_export_xml()
    test_export_html()
    test_export_markdown_structure()
    test_export_formats_share_structured_blocks()
    test_export_dialog_offers_markdown()
    test_export_default_styles_map_to_html_docx_and_pdf()
    test_export_worker_reports_completion_progress()
    test_export_filename_sanitizes_invalid_project_name()
    test_export_worker_sanitizes_project_name_for_all_formats()
    test_export_worker_keeps_formats_independent_when_one_fails()
    test_export_dialog_reports_partial_success_without_critical_error()
    test_export_dialog_surfaces_output_path_failure_from_real_worker()
    test_layout_panel_analysis_progress_lifecycle()
    test_workflow_controller_layout_progress_signal()
    test_main_window_layout_error_is_status_only()
    test_fake_ocr_engine()
    test_confidence_normalization()
    test_api_ocr_engine_requests_return_word_box()
    test_api_ocr_engine_parses_char_level_word_boxes()
    test_api_ocr_engine_parses_pruned_direct_camelcase_word_boxes()
    test_api_ocr_engine_parses_text_word_boxes_alias()
    test_wordbox_anchor_allows_cjk_left_overflow_without_right_expansion()
    test_api_ocr_engine_refines_cjk_word_box_with_anchor()
    test_api_ocr_engine_marks_word_level_boxes_without_fake_char_precision()
    test_api_ocr_engine_filters_empty_narrow_word_boxes()
    test_api_ocr_engine_does_not_promote_block_content_to_line()
    test_api_ocr_engine_ignores_block_content_without_rec_rows()
    test_api_ocr_engine_aligns_token_rows_by_bbox_not_index()
    test_api_ocr_engine_uses_matching_token_row_when_rec_bbox_missing()
    test_api_ocr_engine_preserves_rec_text_without_any_geometry()
    test_api_ocr_engine_preserves_token_text_when_rec_rows_missing()
    test_fake_layout_engine()
    test_fake_llm_engine_disabled()
    test_fake_llm_engine()
    test_ocr_pipeline()
    test_ocr_pipeline_keeps_page_relative_boxes()
    test_ocr_pipeline_offsets_crop_relative_boxes()
    test_ocr_pipeline_prefers_ocr_boxes_and_only_falls_back_for_missing_chars()
    test_ocr_pipeline_normalizes_proof_geometry()
    test_ocr_pipeline_reports_real_page_progress()
    test_ocr_pipeline_assigns_page_ocr_lines_to_structure_blocks_once()
    test_page_ocr_refills_caption_and_equation_blocks()
    test_ocr_pipeline_avoids_double_shift_for_page_space_boxes()
    test_workflow_controller_auto_chains_ocr_after_layout()
    test_workflow_controller_starts_parallel_proof_ocr_with_layout()
    test_workflow_controller_parallel_proof_skips_missing_page_without_misalignment()
    test_workflow_controller_keeps_qthreads_until_finished_after_error()
    test_workflow_controller_falls_back_to_block_ocr_when_parallel_proof_failed()
    test_workflow_controller_emits_ocr_progress_and_navigation()
    test_workflow_controller_ocr_done_does_not_force_hproof_step()
    test_main_window_ocr_finished_preserves_current_step()
    test_empty_llm_config_does_not_block_ocr_done()
    test_workflow_controller_normalizes_loaded_project_geometry()
    test_export_service()
    test_import_service()
    test_import_service_sequential_page_numbers()
    test_api_settings_dialog_keeps_model_preset_sync()
    test_api_settings_dialog_reverse_matches_url_and_persists_profile()
    test_api_settings_dialog_saves_base_url_from_endpoint_suffix()
    test_api_settings_dialog_llm_copy_is_suggestion_only_and_non_blocking()
    test_api_settings_dialog_persists_llm_candidate_settings()
    test_llm_rules_loads_default_rules_file()
    test_api_model_profile_helpers()
    test_api_endpoint_role_resolution_keeps_layout_and_proof_separate()
    test_api_http_post_json_disables_environment_proxies()
    test_fixed_api_chain_resolves_official_roots_to_vl15_and_ppocrv5()
    test_api_ocr_engine_resolves_ocr_endpoint_for_pp_ocrv5_profile()
    test_api_ocr_engine_parses_paddle_coordinate_variants()
    test_api_request_builders_split_profile_params()
    test_layout_analyzer_rescales_suspicious_blocks()
    test_layout_analyzer_extracts_api_polygon_bbox()
    test_layout_analyzer_extracts_api_blocks_from_varied_schema()
    test_layout_analyzer_falls_back_to_ocr_results()
    test_layout_analyzer_uses_datainfo_canvas_scale()
    test_layout_analyzer_ignores_conflicting_datainfo_when_bbox_is_page_space()
    test_layout_analyzer_ignores_conflicting_pruned_shape_when_bbox_is_page_space()
    test_layout_analyzer_resolves_layout_role_even_when_pp_ocrv5_profile_selected()
    test_layout_worker_continues_after_single_page_failure()
    test_workflow_controller_marks_partial_layout_failures_without_blocking_success_pages()
    test_workflow_controller_enables_proof_steps_after_first_ocr_page()
    test_proof_line_iterator_includes_caption_and_equation_lines()
    test_hproof_line_iterator_excludes_non_text_elements()
    test_hproof_line_iterator_excludes_position_source_labels()
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
    test_layout_analyzer_builds_api_payload()
    test_inspector_structure_ocr_falls_back_when_ppstructure_pipeline_missing()
    test_inspector_flattens_api_layout_parsing_result()
    test_inspector_flattens_api_pruned_word_boxes_for_adapter_chars()
    test_inspector_local_flatteners_preserve_word_box_rows()
    test_inspector_builds_api_request_body_from_shared_config()
    test_inspector_runtime_meta_records_actual_request_and_response_fields()
    test_ocr_inspector_run_panel_profile_request_params()
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
    test_char_index_suppresses_punctuation_topic_for_shared_word_box()
    test_char_index_uses_token_collection_for_word_level_han_bbox()
    test_char_index_skips_empty_narrow_ocr_bbox()
    test_char_index_skips_lines_with_unverified_geometry()
    test_char_index_deduplicates_overlapping_duplicate_lines()
    test_vproof_text_map_deduplicates_overlapping_duplicate_lines()
    test_crop_bbox_ascii_path()
    test_crop_bbox_chinese_path()
    test_find_text_matches_lines()
    test_find_text_matches_ocr_chars()
    test_planb_core_seam_diagnoses_missing_word_region()
    test_planb_core_seam_preserves_word_region_and_search_identity()
    test_planb_core_seam_reads_paddle_json_text_word_boxes_alias()
    test_planb_core_seam_flags_invalid_word_region()
    test_crop_panel_search_selects_canvas_node()
    test_paddle_adapter_char_bbox_source()
    test_paddle_adapter_reads_direct_and_camelcase_word_rows()
    test_paddle_adapter_char_fallback_when_no_word_region()
    test_ocr_inspector_paddle_adapter_keeps_line_only_chars_unavailable()
    test_ocr_inspector_paddle_adapter_marks_word_not_fake_char()
    test_canvas_node_item_map_char_fallback()
    test_canvas_node_item_map_char_ocr()
    test_canvas_char_fallback_source_colour()
    test_canvas_char_overlay_default_visible()
    test_canvas_shows_unavailable_note_without_word_region()
    test_canvas_warning_reports_server_missing_after_return_word_box_sent()
    test_inspector_tree_syncs_external_char_selection()
    test_params_ref_matrix_marks_vl_word_box_unsupported()
    print("\n✓ 所有测试通过")
