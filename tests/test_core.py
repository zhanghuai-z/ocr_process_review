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
    from app.core.bbox_utils import (
        bbox_from_quad, bbox_from_xyxy, is_crop_relative_bbox, project_line_bbox,
        sanitize_xyxy_bbox, scale_bbox, scale_bbox_to_page,
    )
    from app.models import BBox

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
    )
    assert page_bbox == BBox(110, 55, 40, 10)

    scaled = scale_bbox(BBox(10, 20, 30, 40), 2.0, 1.5)
    assert scaled == BBox(20, 30, 60, 60)

    quad = bbox_from_quad([[10, 20], [50, 20], [50, 60], [10, 60]])
    assert quad == BBox(10, 20, 40, 40)

    scaled_to_page = scale_bbox_to_page(
        BBox(100, 50, 200, 100),
        source_w=800,
        source_h=600,
        page_w=1600,
        page_h=1200,
    )
    assert scaled_to_page == BBox(200, 100, 400, 200)

    print("test_bbox_tools PASSED")


def test_block_type_mapping():
    from app.models import BlockType

    assert BlockType.from_paddle("paragraph") == BlockType.TEXT
    assert BlockType.from_paddle("doc_title") == BlockType.TITLE
    assert BlockType.from_paddle("page_number") == BlockType.TEXT
    assert BlockType.from_paddle("sidebar_text") == BlockType.TEXT
    assert BlockType.from_paddle("image_caption") == BlockType.FIGURE_CAPTION
    assert BlockType.from_paddle("figure_title") == BlockType.FIGURE_CAPTION
    assert BlockType.from_paddle("table_body") == BlockType.TABLE
    assert BlockType.from_paddle("table_title") == BlockType.TABLE_CAPTION
    assert BlockType.from_paddle("bibliography") == BlockType.REFERENCE
    assert BlockType.from_paddle("formula_number") == BlockType.EQUATION

    print("test_block_type_mapping PASSED")


# =====================================================================
# ProjectStore 测试
# =====================================================================

def test_project_store():
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.core.project_store import ProjectStore

    with tempfile.NamedTemporaryFile(suffix=".ocrproj", delete=False) as f:
        db_path = f.name

    try:
        bb = BBox(0, 0, 100, 20)
        line = Line(text="Hello OCR", confidence=0.92, bbox=bb)
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
    """从 v1 schema 迁移到 v2。"""
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
        assert int(ver[0]) >= 2
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


def test_fake_layout_engine():
    from app.engines.fake_layout_engine import FakeLayoutEngine

    engine = FakeLayoutEngine()
    blocks = engine.analyze("/tmp/nonexistent.jpg")  # 应返回 fallback

    assert len(blocks) > 0
    assert blocks[0].bbox.w > 0

    print("test_fake_layout_engine PASSED")


def test_local_ocr_engine_parses_predict_result():
    import types
    import numpy as np
    from app.engines import OcrContext
    from app.engines.real_ocr_adapter import LocalOcrEngine

    class FakeResult:
        def __init__(self, payload):
            self.json = {"res": payload}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_bgr):
            return [FakeResult({
                "rec_texts": ["第一行", "第二行"],
                "rec_scores": [0.93, 88],
                "rec_boxes": [
                    [10, 15, 110, 35],
                    [[20, 45], [120, 45], [120, 70], [20, 70]],
                ],
            })]

    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    try:
        engine = LocalOcrEngine()
        lines = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8), OcrContext())
        assert engine._engine.kwargs["ocr_version"] == "PP-OCRv5"
        assert len(lines) == 2
        assert lines[0].text == "第一行"
        assert lines[0].bbox.to_xyxy() == (10, 15, 110, 35)
        assert lines[1].confidence == 0.88
        assert lines[1].bbox.to_xyxy() == (20, 45, 120, 70)
    finally:
        if original is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = original

    print("test_local_ocr_engine_parses_predict_result PASSED")


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
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class CropCoordEngine:
        def recognize(self, image_bgr, context):
            return [Line(text="局部坐标", confidence=0.92, bbox=BBox(20, 10, 80, 18))]

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
    finally:
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


def test_layout_analyzer_does_not_rescale_sparse_top_blocks():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Block, BlockType, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=2400, height=3200)
    page.blocks = [
        Block(block_type=BlockType.TEXT, bbox=BBox(50, 40, 300, 80)),
        Block(block_type=BlockType.TEXT, bbox=BBox(60, 180, 320, 120)),
    ]
    original = [block.bbox for block in page.blocks]

    analyzer._rescale_blocks_if_suspicious(page)

    assert [block.bbox for block in page.blocks] == original

    print("test_layout_analyzer_does_not_rescale_sparse_top_blocks PASSED")


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


def test_layout_analyzer_extracts_relative_polygon_bbox():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=1000, height=2000)
    bbox = analyzer._extract_bbox_from_coordinate(
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2]],
        page,
    )

    assert bbox == BBox(100, 200, 200, 200)

    print("test_layout_analyzer_extracts_relative_polygon_bbox PASSED")


def test_layout_analyzer_scales_bbox_from_response_size():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=2400, height=3200)

    bbox = analyzer._extract_bbox_from_coordinate(
        [80, 60, 400, 120],
        page,
        coordinate_space=(800, 608),
    )

    assert bbox == BBox(240, 316, 960, 316)

    print("test_layout_analyzer_scales_bbox_from_response_size PASSED")


def test_layout_analyzer_resolves_coordinate_space():
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import Page

    analyzer = LayoutAnalyzer()
    page = Page(image_path="/tmp/test.png", width=2400, height=3200)
    data = {
        "result": {
            "image_size": [800, 608],
            "layoutParsingResults": [],
        }
    }

    size = analyzer._resolve_coordinate_space(page, data=data)
    assert size == (800, 608)

    print("test_layout_analyzer_resolves_coordinate_space PASSED")


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


def test_layout_analyzer_reads_local_ppstructurev3_result():
    import tempfile
    import cv2
    import numpy as np
    from app.core.layout_analyzer import LayoutAnalyzer
    from app.models import BBox, BlockType, Page

    class FakeResult:
        def __init__(self, payload):
            self.json = {"res": payload}

    class FakeEngine:
        def predict(self, image_path):
            return [FakeResult({
                "prunedResult": {
                    "layout_det_res": {
                        "boxes": [
                            {
                                "label": "table_title",
                                "coordinate": [0.1, 0.2, 0.5, 0.4],
                            }
                        ]
                    }
                }
            })]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

    try:
        page = Page(image_path=img_path, width=0, height=0)
        analyzer = LayoutAnalyzer()
        analyzer._engine = FakeEngine()
        result = analyzer.analyze(page)

        assert len(result.blocks) == 1
        assert result.blocks[0].block_type == BlockType.TABLE_CAPTION
        assert result.blocks[0].bbox == BBox(20, 20, 80, 20)
    finally:
        os.unlink(img_path)
        for suffix in (".layout-local.json", ".layout-local-raw.png", ".layout-local-app-overlay.png"):
            debug_path = Path(img_path).with_suffix(suffix)
            if debug_path.exists():
                debug_path.unlink()

    print("test_layout_analyzer_reads_local_ppstructurev3_result PASSED")


# =====================================================================
# 入口
# =====================================================================

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
    test_fake_layout_engine()
    test_local_ocr_engine_parses_predict_result()
    test_fake_llm_engine_disabled()
    test_fake_llm_engine()
    test_ocr_pipeline()
    test_ocr_pipeline_keeps_page_relative_boxes()
    test_ocr_pipeline_offsets_crop_relative_boxes()
    test_export_service()
    test_import_service()
    test_import_service_sequential_page_numbers()
    test_layout_analyzer_rescales_suspicious_blocks()
    test_layout_analyzer_does_not_rescale_sparse_top_blocks()
    test_layout_analyzer_extracts_api_polygon_bbox()
    test_layout_analyzer_extracts_relative_polygon_bbox()
    test_layout_analyzer_scales_bbox_from_response_size()
    test_layout_analyzer_resolves_coordinate_space()
    test_layout_analyzer_builds_api_payload()
    test_layout_analyzer_reads_local_ppstructurev3_result()
    print("\n✓ 所有测试通过")
