"""End-to-end smoke test: 跑 HanwangLayoutEngine + HanwangOcrEngine on 120168.tif，
不依赖任何 UI / 主流程，纯验证翻译层 + 引擎适配器是否打通。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from app.engines import OcrContext
from app.engines.hanwang import verify_hanwang_assets, get_hanwang_bin_dir
from app.engines.hanwang_layout_engine import HanwangLayoutEngine
from app.engines.hanwang_ocr_engine import HanwangOcrEngine
from app.engines.real_ocr_adapter import create_engine
from app.core.layout_analyzer import LayoutAnalyzer
from app.models import OcrProject, Page
from app.services.ocr_pipeline import OcrPipeline


def main(img_path: str) -> int:
    verify_hanwang_assets()
    print(f"[ok] bin_dir = {get_hanwang_bin_dir()}")

    page = cv2.imread(img_path)
    if page is None:
        print(f"[err] cannot read {img_path}")
        return 1
    print(f"[ok] page shape = {page.shape}")

    layout = HanwangLayoutEngine()
    blocks = layout.analyze(img_path)
    print(f"[ok] layout: {len(blocks)} blocks")
    for i, b in enumerate(blocks[:5]):
        print(f"  block {i}: type={b.block_type} bbox={b.bbox} note={b.note}")

    if not blocks:
        return 0

    ocr = HanwangOcrEngine()
    total_lines = total_chars = 0
    for i, b in enumerate(blocks):
        bb = b.bbox
        crop = page[bb.y: bb.y + bb.h, bb.x: bb.x + bb.w]
        if crop.size == 0:
            continue
        lines = ocr.recognize(crop, OcrContext(page_number=1, block_id=f"b{i}"))
        total_lines += len(lines)
        for ln in lines:
            total_chars += len(ln.chars)
        if i < 3:
            preview = " | ".join(ln.text for ln in lines[:3])
            print(f"  block {i}: {len(lines)} lines, preview={preview[:80]!r}")
    print(f"[ok] total lines={total_lines} chars={total_chars}")

    os.environ["OCR_OCR_MODE"] = "hanwang"
    shared_page = Page(
        image_path=img_path,
        cache_image_path=img_path,
        width=page.shape[1],
        height=page.shape[0],
        page_number=1,
    )
    shared_page = LayoutAnalyzer().analyze(shared_page)
    project = OcrProject(name="hanwang-smoke", pages=[shared_page])
    shared_result = OcrPipeline(engine=create_engine("hanwang")).process_project(project)
    shared_page = shared_result.pages[0]
    shared_blocks = len(shared_page.blocks)
    shared_lines = sum(len(block.lines) for block in shared_page.blocks)
    shared_chars = sum(len(line.chars) for block in shared_page.blocks for line in block.lines)
    print(f"[ok] shared pipeline: {shared_blocks} blocks / {shared_lines} lines / {shared_chars} chars")
    if shared_page.error_message:
        print(f"[err] shared pipeline page error: {shared_page.error_message}")
        return 2
    if shared_blocks == 0 or shared_lines == 0:
        print("[err] shared pipeline produced no layout/OCR output")
        return 2
    return 0


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[1] / "file" / "244771纵校" / "120168.tif"
    )
    sys.exit(main(img))
