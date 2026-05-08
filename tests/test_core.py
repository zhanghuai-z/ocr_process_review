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
        bbox_from_xyxy, is_crop_relative_bbox, project_line_bbox, sanitize_xyxy_bbox, scale_bbox,
    )
    from app.core.coordinate_seam import (
        BBOX_SPACE_CROP, BBOX_SPACE_PAGE, CropCoordinateSeam,
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
        source_space=BBOX_SPACE_CROP,
    )
    assert page_bbox == BBox(110, 55, 40, 10)

    seam = CropCoordinateSeam.from_page_bbox(BBox(100, 50, 120, 80), page_w=400, page_h=300)
    assert seam.to_page_bbox(BBox(12, 8, 40, 12), source_space=BBOX_SPACE_CROP) == BBox(112, 58, 40, 12)
    assert seam.to_page_bbox(BBox(120, 70, 40, 12), source_space=BBOX_SPACE_PAGE) == BBox(120, 70, 40, 12)
    assert seam.to_crop_bbox(BBox(120, 70, 40, 12)) == BBox(20, 20, 40, 12)

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
    from app.models import BBox, Block, BlockType, Line, OcrProject, Page
    from app.services.ocr_pipeline import OcrPipeline

    class CropCoordEngine:
        bbox_space = "crop"

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
    from app.services.char_index_service import CharIndexService

    page = Page(image_path="/tmp/page.png", width=400, height=300)
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
    assert yi_entries[0].page_idx == 0
    assert yi_entries[0].block_order == 3
    assert yi_entries[0].bbox == BBox(30, 20, 18, 18)
    assert yi_entries[1].char_idx == 0
    assert yi_entries[1].bbox == BBox(10, 50, 100, 18)
    assert service.char_frequency() == [("乙", 2), ("丙", 1), ("甲", 1)]

    print("test_char_index_service PASSED")


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
    test_fake_llm_engine_disabled()
    test_fake_llm_engine()
    test_ocr_pipeline()
    test_ocr_pipeline_keeps_page_relative_boxes()
    test_ocr_pipeline_offsets_crop_relative_boxes()
    test_ocr_pipeline_avoids_double_shift_for_page_space_boxes()
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
    print("\n✓ 所有测试通过")
