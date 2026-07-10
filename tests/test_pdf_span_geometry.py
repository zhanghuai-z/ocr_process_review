from __future__ import annotations

from app.export.pdf import (
    PDF_TEXT_ASCENDER_RATIO,
    PdfPagePlan,
    PdfTextItem,
    PdfTextSpan,
    _span_baseline_top_y,
    _span_font_size,
)


def test_non_atomic_span_uses_full_vertical_observation_envelope():
    short = PdfTextItem("x", x=10, y=100, w=8, h=10, bbox_granularity="char")
    tall = PdfTextItem("I", x=20, y=95, w=6, h=20, bbox_granularity="char")
    span = PdfTextSpan(
        text="xI",
        x=10,
        y=95,
        w=16,
        h=20,
        items=(short, tall),
    )
    plan = PdfPagePlan(
        page_number=1,
        image_path="/tmp/page.png",
        page_width_px=100,
        page_height_px=200,
        width_pt=100,
        height_pt=200,
        text_items=[short, tall],
        text_spans=[span],
    )

    font_size = _span_font_size(plan, span, default_font_size=5)

    assert font_size == 16
    expected_top = plan.height_pt - (115 - font_size * PDF_TEXT_ASCENDER_RATIO)
    assert _span_baseline_top_y(plan, span, font_size) == expected_top
