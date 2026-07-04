from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from app.models import BBox, Block, BlockType, Line, Page
from app.models.enums import OcrPolicy
from app.models.ocr_observation import block_ocr_lines
from app.services.ocr_dispatch_plan import build_text_ocr_dispatch_plan
from app.services.ocr_pipeline import OcrPipeline
from app.services.proof_crop_service import ProofCropService


def _block(
    block_type: BlockType,
    bbox: BBox,
    *,
    policy: OcrPolicy,
    order: int,
) -> Block:
    return Block(
        block_type=block_type,
        bbox=bbox,
        ocr_policy=policy,
        order=order,
    )


def test_dispatch_plan_selects_text_ocr_blocks_and_preserves_page_order():
    text = _block(BlockType.TEXT, BBox(0, 0, 100, 40), policy=OcrPolicy.TEXT_OCR, order=10)
    formula = _block(
        BlockType.EQUATION,
        BBox(20, 5, 30, 20),
        policy=OcrPolicy.PRESERVE_AS_FORMULA,
        order=11,
    )
    skipped_title = _block(BlockType.TITLE, BBox(0, 45, 100, 30), policy=OcrPolicy.SKIP, order=12)
    reference = _block(BlockType.REFERENCE, BBox(0, 80, 100, 30), policy=OcrPolicy.TEXT_OCR, order=13)
    page = Page(
        image_path="page.png",
        width=200,
        height=120,
        blocks=[text, formula, skipped_title, reference],
    )

    plan = build_text_ocr_dispatch_plan(page)

    assert [target.index for target in plan.text_blocks] == [0, 3]
    assert [target.block for target in plan.text_blocks] == [text, reference]
    assert [target.index for target in plan.blocked_blocks] == [1, 2]
    assert [target.reason for target in plan.text_blocks] == ["policy:text_ocr", "policy:text_ocr"]
    assert plan.total_text_blocks == 2
    assert plan.has_text_work is True


def test_page_ocr_assignment_uses_dispatch_plan_blockers_before_text_container():
    text = _block(BlockType.TEXT, BBox(0, 0, 200, 200), policy=OcrPolicy.TEXT_OCR, order=1)
    formula = _block(
        BlockType.EQUATION,
        BBox(50, 50, 60, 30),
        policy=OcrPolicy.PRESERVE_AS_FORMULA,
        order=2,
    )
    page = Page(image_path="page.png", width=220, height=220, blocks=[text, formula])
    formula_line = Line(text="式", confidence=0.9, bbox=BBox(55, 55, 20, 12))
    normal_line = Line(text="正文", confidence=0.9, bbox=BBox(10, 10, 40, 12))

    OcrPipeline().assign_page_ocr_lines_to_blocks(page, [formula_line, normal_line])

    assert block_ocr_lines(text) == [normal_line]
    assert block_ocr_lines(formula) == []


def test_proof_crop_service_uses_dispatch_plan_not_block_type_text_blocks():
    text = _block(BlockType.TEXT, BBox(0, 0, 60, 30), policy=OcrPolicy.TEXT_OCR, order=1)
    text.lines = [Line(text="甲", confidence=0.9, bbox=BBox(5, 5, 20, 20))]
    skipped_title = _block(BlockType.TITLE, BBox(0, 40, 60, 30), policy=OcrPolicy.SKIP, order=2)
    skipped_title.lines = [Line(text="乙", confidence=0.9, bbox=BBox(5, 45, 20, 20))]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    try:
        cv2.imwrite(img_path, np.ones((90, 80, 3), dtype=np.uint8) * 255)
        page = Page(
            image_path=img_path,
            width=80,
            height=90,
            blocks=[text, skipped_title],
        )

        stats = ProofCropService().normalize_pages([page])

        assert stats.lines == 1
        assert [char.char for char in text.lines[0].chars] == ["甲"]
        assert skipped_title.lines[0].chars == []
    finally:
        try:
            os.unlink(img_path)
        except FileNotFoundError:
            pass
