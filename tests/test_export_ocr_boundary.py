from __future__ import annotations

from app.core.proof_line_facts import proof_line_facts
from app.export.ir_builder import _is_line_corrected
from app.models import BBox, Line


def test_export_correction_uses_proof_ocr_text_not_mutated_original_text() -> None:
    line = Line(
        text="OCR 文本",
        confidence=0.91,
        bbox=BBox(0, 0, 120, 20),
    )
    line.original_text = "被污染的布局文本"
    assert _is_line_corrected(proof_line_facts(line)) is False


def test_export_correction_uses_proof_display_text_vs_ocr_text() -> None:
    line = Line(
        text="OCR 文本",
        confidence=0.91,
        bbox=BBox(0, 0, 120, 20),
    )
    line.set_proof_text("人工改正文本")
    assert _is_line_corrected(proof_line_facts(line)) is True
