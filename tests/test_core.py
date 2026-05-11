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
    assert shifted.chars[0].bbox == BBox(10, 20, 30, 30)
    assert shifted.chars[0].bbox_source == "fallback"
    assert shifted.chars[1].bbox == BBox(40, 20, 30, 30)
    assert shifted.chars[1].bbox_source == "ocr"

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

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
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
        assert captured["url"] == "https://example.com/layout-parsing"
        assert captured["json"]["returnWordBox"] is True
        assert captured["json"]["useDocOrientationClassify"] is False
        assert captured["json"]["useDocUnwarping"] is False
        assert captured["json"]["useTextlineOrientation"] is False
        assert captured["json"]["textDetLimitSideLen"] == 1536
        assert captured["json"]["textDetBoxThresh"] == 0.6
        assert captured["json"]["textDetUnclipRatio"] == 1.3
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

    service = CharIndexService().build_index(project)
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

    service = CharIndexService().build_index(OcrProject(name="vertical-index", pages=[page]))
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

        service = CharIndexService().build([page])
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

    svc = CharIndexService().build_index(OcrProject(name="digit-group", pages=[page]))
    digit_entries = svc.query("2026")
    assert len(digit_entries) == 1
    assert digit_entries[0].bbox == BBox(10, 20, 48, 24)
    assert digit_entries[0].collection_kind == "token"
    assert svc.query("2") == []
    assert len(svc.query("年")) == 1

    print("test_char_index_groups_digit_runs_as_tokens PASSED")


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
    svc = CharIndexService().build_index(OcrProject(name="digit-sort", pages=[page]))
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
    svc = CharIndexService().build_index(OcrProject(name="formula-group", pages=[page]))

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
        svc = CharIndexService().build_index(OcrProject(name="dedupe", pages=[page]))
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


def test_app_config_tracks_api_model_profile():
    from app.core.app_config import AppConfig, get_config, update_config

    cfg = AppConfig.instance()
    cfg.reset_to_defaults()
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
    assert current["api_url"] == "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing"
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
    update_config(
        mode="api",
        api_model_profile="pp-structurev3",
        api_url="https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
        api_token="",
        api_timeout=30,
        api_layout_model_name="",
    )

    dialog = ApiSettingsDialog()
    assert dialog._api_model_combo.currentData() == "pp-structurev3"
    assert dialog._url_edit.text() == "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing"

    index = dialog._api_model_combo.findData("pp-ocrv5")
    dialog._api_model_combo.setCurrentIndex(index)
    assert dialog._url_edit.text() == "https://n6z9feddjca4l7b5.aistudio-app.com/ocr"

    dialog._url_edit.setText("https://example.com/custom-layout")
    dialog._sync_model_from_url()
    assert dialog._api_model_combo.currentIndex() == -1

    dialog._url_edit.setText("https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing")
    dialog._sync_model_from_url()
    assert dialog._api_model_combo.currentData() == "paddleocr-vl"

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
    from app.core.ocr_config import get_config
    from app.ui.widgets.api_settings_dialog import (
        ApiSettingsDialog,
        get_api_model_profile_url,
    )

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()

        idx = dialog._api_model_combo.findData("paddleocr-vl")
        dialog._api_model_combo.setCurrentIndex(idx)

        assert dialog._url_edit.text() == get_api_model_profile_url("paddleocr-vl")
        assert "PaddleOCR-VL" in dialog._summary_model.text()
        assert "官方预设" in dialog._model_note.text()

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_keeps_model_preset_sync PASSED")


def test_api_settings_dialog_reverse_matches_url_and_persists_profile():
    from app.core.app_config import AppConfig
    from app.core.ocr_config import get_config
    from app.ui.widgets.api_settings_dialog import (
        ApiSettingsDialog,
        get_api_model_profile_url,
    )

    _get_qapp()

    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_app_config_for_test(tmpdir)
        dialog = ApiSettingsDialog()
        dialog._radio_api.setChecked(True)
        dialog._url_edit.setText(get_api_model_profile_url("pp-ocrv5"))
        dialog._sync_model_from_url()

        assert dialog._api_model_combo.currentData() == "pp-ocrv5"
        assert "/ocr" in dialog._summary_endpoint_kind.text()

        dialog._url_edit.setText("https://example.com/custom")
        dialog._sync_model_from_url()
        assert dialog._api_model_combo.currentIndex() == -1
        assert "自定义" in dialog._summary_model.text()

        dialog._url_edit.setText(get_api_model_profile_url("pp-ocrv5"))
        dialog._sync_model_from_url()
        dialog._save_and_accept()

        cfg = get_config()
        assert cfg["mode"] == "api"
        assert cfg["api_model_profile"] == "pp-ocrv5"
        assert cfg["api_url"] == get_api_model_profile_url("pp-ocrv5")

        dialog.close()
        AppConfig.instance().reset_to_defaults()
        AppConfig._instance = None

    print("test_api_settings_dialog_reverse_matches_url_and_persists_profile PASSED")


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
    svc = CharIndexService()
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
    svc = CharIndexService()
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
    svc = CharIndexService()
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


def test_paddle_adapter_char_fallback_when_no_word_region():
    """PaddleAdapter sets bbox_source=fallback when text_word_region absent."""
    from tools.ocr_inspector.adapters.paddle import PaddleAdapter
    raw = {"overall_ocr_res": {
        "rec_texts": ["测试"], "rec_boxes": [[10, 20, 210, 50]], "rec_scores": [0.95],
    }}
    doc = PaddleAdapter().parse(raw, image_path="/tmp/test.jpg")
    line = doc.pages[0].all_lines[0]
    assert len(line.chars) == 2
    for ch in line.chars:
        assert ch.bbox_source == "fallback" and ch.bbox == line.bbox
    print("test_paddle_adapter_char_fallback_when_no_word_region PASSED")



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
    test_fake_ocr_engine()
    test_confidence_normalization()
    test_api_ocr_engine_requests_return_word_box()
    test_api_ocr_engine_parses_char_level_word_boxes()
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
    test_ocr_pipeline_avoids_double_shift_for_page_space_boxes()
    test_workflow_controller_auto_chains_ocr_after_layout()
    test_workflow_controller_emits_ocr_progress_and_navigation()
    test_workflow_controller_normalizes_loaded_project_geometry()
    test_export_service()
    test_import_service()
    test_import_service_sequential_page_numbers()
    test_api_settings_dialog_keeps_model_preset_sync()
    test_api_settings_dialog_reverse_matches_url_and_persists_profile()
    test_layout_analyzer_rescales_suspicious_blocks()
    test_layout_analyzer_extracts_api_polygon_bbox()
    test_layout_analyzer_extracts_api_blocks_from_varied_schema()
    test_layout_analyzer_falls_back_to_ocr_results()
    test_layout_analyzer_builds_api_payload()
    test_inspector_structure_ocr_falls_back_when_ppstructure_pipeline_missing()
    test_inspector_flattens_api_layout_parsing_result()
    test_inspector_builds_api_request_body_from_shared_config()
    test_char_index_vertical_split()
    test_char_index_horizontal_split()
    test_char_index_dedup_on_rebuild()
    test_char_index_sort_categories()
    test_char_index_query_stable_order()
    test_char_index_skips_whitespace()
    test_char_index_groups_digit_runs_as_tokens()
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
    test_paddle_adapter_char_bbox_source()
    test_paddle_adapter_char_fallback_when_no_word_region()
    print("\n✓ 所有测试通过")
