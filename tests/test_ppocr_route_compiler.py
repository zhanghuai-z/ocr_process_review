from __future__ import annotations

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import (
    PpOcrV6LineHint,
    PpOcrV6PrepassArtifact,
    PpOcrV6WordBox,
)
from app.models.charocr_routing import (
    ROUTE_SEGMENT_TEXT_OTHER,
    ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT,
    TextSliceRoute,
)
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.models import BBox, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot, OcrPolicy


def _block(
    uid: str,
    block_type: BlockType,
    xyxy: tuple[int, int, int, int],
    *,
    policy: OcrPolicy,
    order: int,
    label: str = "",
) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        uid=uid,
        block_type=block_type,
        bbox=BBox.from_xyxy(*xyxy),
        order=order,
        source_label=label or block_type.value,
        origin=BlockOrigin(source_label=label or block_type.value, original_bbox=BBox.from_xyxy(*xyxy), original_kind=block_type),
        ocr_policy=policy,
    )


def _snapshot(*blocks: LayoutBlockSnapshot) -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid="page-1",
        artifact_uid="layout-run-1",
        source_engine="paddleocr-vl-1.6",
        source_run_id="layout-job-1",
        blocks=blocks,
    )


def _prepass(*lines: PpOcrV6LineHint) -> PpOcrV6PrepassArtifact:
    return PpOcrV6PrepassArtifact(page_uid="page-1", run_id="ppocr-job-1", lines=lines)


def test_compiler_excludes_table_figure_and_formula_from_charocr_routes():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 300, 100), policy=OcrPolicy.TEXT_OCR, order=0),
        _block("table-1", BlockType.TABLE, (100, 0, 180, 40), policy=OcrPolicy.PRESERVE_AS_TABLE, order=1),
        _block("formula-1", BlockType.EQUATION, (200, 0, 220, 40), policy=OcrPolicy.PRESERVE_AS_FORMULA, order=2, label="inline_formula"),
        _block("figure-1", BlockType.FIGURE, (0, 50, 80, 90), policy=OcrPolicy.SKIP, order=3),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=0, text="城市", bbox=(0, 0, 300, 40), words=()),
        PpOcrV6LineHint(index=1, text="image text", bbox=(0, 50, 80, 90), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=300, page_height=100)

    assert plan.is_dispatchable is True
    assert plan.for_block("table-1") is None
    assert plan.for_block("figure-1") is None
    route = plan.for_block("text-1").lines[0]
    assert [(segment.kind, segment.bbox) for segment in route.segments] == [
        ("text_other", (0, 0, 100, 40)),
        ("skip", (100, 0, 180, 40)),
        ("text_other", (180, 0, 200, 40)),
        ("formula", (200, 0, 220, 40)),
        ("text_other", (220, 0, 300, 40)),
    ]
    assert len(plan.for_block("text-1").lines) == 1


def test_compiler_routes_vertical_text_from_layout_without_ppocr_line():
    snapshot = _snapshot(
        _block(
            "vertical-1",
            BlockType.TEXT,
            (20, 30, 80, 520),
            policy=OcrPolicy.TEXT_OCR,
            order=0,
            label="vertical_text",
        ),
    )

    plan = compile_page_routing_plan(
        snapshot,
        _prepass(),
        page_width=300,
        page_height=600,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("vertical-1").lines[0]
    assert route.source == ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT
    assert route.bbox == (20, 30, 80, 520)
    assert [(segment.kind, segment.bbox) for segment in route.segments] == [
        ("text_other", (20, 30, 80, 520)),
    ]


def test_compiler_does_not_duplicate_ppocr_fragments_inside_vertical_text():
    snapshot = _snapshot(
        _block(
            "vertical-1",
            BlockType.TEXT,
            (20, 30, 80, 520),
            policy=OcrPolicy.TEXT_OCR,
            order=0,
            label="vertical_text",
        ),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=4, text="宁", bbox=(25, 40, 75, 100), words=()),
    )

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=300,
        page_height=600,
    )

    assert plan.is_dispatchable is True
    routes = plan.for_block("vertical-1").lines
    assert len(routes) == 1
    assert routes[0].source == ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT


def test_compiler_keeps_formula_content_geometry_when_line_mask_is_clipped():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 240, 100), policy=OcrPolicy.TEXT_OCR, order=0),
        _block("formula-1", BlockType.EQUATION, (90, 10, 150, 80), policy=OcrPolicy.PRESERVE_AS_FORMULA, order=1, label="inline_formula"),
    )
    plan = compile_page_routing_plan(
        snapshot,
        _prepass(PpOcrV6LineHint(index=0, text="甲乙", bbox=(0, 30, 220, 60), words=())),
        page_width=240,
        page_height=100,
    )

    formula = next(segment for segment in plan.for_block("text-1").lines[0].segments if segment.kind == "formula")
    assert formula.bbox == (90, 30, 150, 60)
    assert formula.content_bbox == (90, 10, 150, 80)


def test_compiler_blocks_page_when_formula_masks_overlap_in_one_text_line():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 240, 80), policy=OcrPolicy.TEXT_OCR, order=0),
        _block("formula-1", BlockType.EQUATION, (40, 0, 100, 50), policy=OcrPolicy.PRESERVE_AS_FORMULA, order=1, label="inline_formula"),
        _block("formula-2", BlockType.EQUATION, (96, 0, 150, 50), policy=OcrPolicy.PRESERVE_AS_FORMULA, order=2, label="inline_formula"),
    )
    plan = compile_page_routing_plan(
        snapshot,
        _prepass(PpOcrV6LineHint(index=0, text="甲乙", bbox=(0, 0, 220, 40), words=())),
        page_width=240,
        page_height=80,
    )

    assert plan.is_dispatchable is False
    assert [(issue.code, issue.bbox) for issue in plan.validation_issues] == [
        ("overlapping_formula_masks", (96, 0, 100, 40)),
    ]


def test_compiler_stops_only_current_page_when_a_text_line_has_no_layout_owner():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 80, 40), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=7, text="orphan", bbox=(120, 0, 220, 40), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=300, page_height=100)

    assert plan.is_dispatchable is False
    assert [(issue.code, issue.line_index, issue.bbox) for issue in plan.validation_issues] == [
        ("unmatched_prepass_line", 7, (120, 0, 220, 40)),
    ]


def test_compiler_accepts_detector_seam_inside_structural_block():
    snapshot = _snapshot(
        _block("header-1", BlockType.FIGURE, (10, 10, 90, 50), policy=OcrPolicy.SKIP, order=0),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=3, text="header", bbox=(5, 5, 95, 55), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=100, page_height=60)

    assert plan.is_dispatchable is True
    assert plan.validation_issues == ()


def test_compiler_assigns_formula_number_row_by_center_when_detector_box_is_larger():
    snapshot = _snapshot(
        _block(
            "formula-number-1",
            BlockType.EQUATION,
            (20, 20, 80, 60),
            policy=OcrPolicy.PRESERVE_AS_FORMULA,
            order=0,
            label="formula_number",
        ),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=3, text="(1)", bbox=(5, 5, 95, 75), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=100, page_height=80)

    assert plan.is_dispatchable is True
    assert plan.validation_issues == ()


def test_compiler_structural_owner_beats_incidental_text_overlap():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 180, 22), policy=OcrPolicy.TEXT_OCR, order=0),
        _block(
            "formula-1",
            BlockType.EQUATION,
            (30, 20, 150, 80),
            policy=OcrPolicy.PRESERVE_AS_FORMULA,
            order=1,
            label="display_formula",
        ),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=4, text="x = 1", bbox=(25, 10, 155, 85), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=180, page_height=90)

    assert plan.is_dispatchable is True
    assert plan.for_block("text-1").lines == ()


def test_text_other_is_an_explicit_text_route_kind():
    route = TextSliceRoute(
        line_index=0,
        segment_index=0,
        bbox=(0, 0, 10, 10),
        carved=False,
        kind=ROUTE_SEGMENT_TEXT_OTHER,
    )

    assert route.kind == "text_other"


def test_compiler_partitions_mixed_line_from_word_box_proposals_and_ink():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 160, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    image = np.full((50, 160, 3), 255, dtype=np.uint8)
    image[10:30, 10:28] = 0
    image[10:30, 43:48] = 0
    image[10:30, 52:57] = 0
    image[10:30, 61:66] = 0
    image[24:30, 83:87] = 0
    image[10:30, 102:120] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="甲ABC,乙",
        bbox=(0, 0, 140, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (8, 8, 30, 32)),
            PpOcrV6WordBox(0, 1, "ABC", (40, 8, 75, 32)),
            PpOcrV6WordBox(0, 2, ",", (80, 8, 90, 32)),
            PpOcrV6WordBox(0, 3, "乙", (100, 8, 122, 32)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=160,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    assert [segment.kind for segment in plan.for_block("text-1").lines[0].segments] == [
        "text_other", "text_latin", "text_other",
    ]


def test_compiler_rejects_pure_latin_row_without_word_masks():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 180, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="vol. 26, no. 2, 1998.",
        bbox=(10, 5, 170, 40),
        words=(),
    ))

    plan = compile_page_routing_plan(snapshot, prepass, page_width=180, page_height=50)

    assert plan.is_dispatchable is False
    assert [(issue.code, issue.line_index, issue.bbox) for issue in plan.validation_issues] == [
        ("latin_line_missing_word_boxes", 0, (10, 5, 170, 40)),
    ]


def test_compiler_partitions_quotes_out_of_pure_latin_row():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 100, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    image = np.full((50, 100, 3), 255, dtype=np.uint8)
    image[8:30, 10:14] = 0
    image[8:30, 30:40] = 0
    image[8:30, 50:54] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text='“A”',
        bbox=(0, 0, 70, 40),
        words=(
            PpOcrV6WordBox(0, 0, "“", (8, 8, 16, 32)),
            PpOcrV6WordBox(0, 1, "A", (28, 8, 42, 32)),
            PpOcrV6WordBox(0, 2, "”", (48, 8, 56, 32)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    assert [(segment.kind, segment.bbox) for segment in plan.for_block("text-1").lines[0].segments] == [
        ("text_symbol", (0, 0, 30, 40)),
        ("text_latin", (30, 8, 40, 30)),
        ("text_symbol", (40, 0, 70, 40)),
    ]


def test_compiler_drops_blank_text_sliver_left_by_formula_carving():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 180, 50), policy=OcrPolicy.TEXT_OCR, order=0),
        _block(
            "formula-1",
            BlockType.EQUATION,
            (80, 5, 140, 45),
            policy=OcrPolicy.PRESERVE_AS_FORMULA,
            order=1,
            label="inline_formula",
        ),
    )
    image = np.full((50, 180, 3), 255, dtype=np.uint8)
    image[10:30, 20:40] = 0
    image[10:30, 85:125] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="甲D_t",
        bbox=(0, 0, 160, 50),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (18, 8, 42, 32)),
            PpOcrV6WordBox(0, 1, "D", (74, 8, 84, 32)),
            PpOcrV6WordBox(0, 2, "_", (84, 8, 92, 32)),
            PpOcrV6WordBox(0, 3, "t", (110, 8, 120, 32)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=180,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("text-1").lines[0]
    assert [(segment.kind, segment.bbox) for segment in route.segments] == [
        ("text_other", (0, 0, 80, 50)),
        ("formula", (80, 5, 140, 45)),
    ]


def test_compiler_assigns_boundary_glyph_to_only_one_latin_token():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 140, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    image = np.full((50, 140, 3), 255, dtype=np.uint8)
    image[10:30, 10:30] = 0
    image[10:30, 43:50] = 0
    image[10:30, 52:80] = 0
    image[8:32, 100:124] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="one two甲",
        bbox=(0, 0, 130, 40),
        words=(
            PpOcrV6WordBox(0, 0, "one", (8, 8, 40, 32)),
            PpOcrV6WordBox(0, 1, " ", (40, 8, 43, 32)),
            PpOcrV6WordBox(0, 2, "two", (46, 8, 85, 32)),
            PpOcrV6WordBox(0, 3, "甲", (98, 8, 126, 32)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=140,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    latin = [segment for segment in plan.for_block("text-1").lines[0].segments if segment.kind == "text_latin"]
    assert len(latin) == 1
    assert latin[0].text == "onetwo"


def test_compiler_reclaims_displaced_narrow_latin_glyph_from_punctuation_seam():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 120, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[8:30, 42:48] = 0
    image[10:30, 70:90] = 0
    image[8:32, 100:118] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text=". I word甲",
        bbox=(0, 0, 110, 40),
        words=(
            PpOcrV6WordBox(0, 0, ". ", (20, 8, 43, 32)),
            PpOcrV6WordBox(0, 1, "I", (49, 8, 55, 32)),
            PpOcrV6WordBox(0, 2, " ", (56, 8, 65, 32)),
            PpOcrV6WordBox(0, 3, "word", (68, 8, 95, 32)),
            PpOcrV6WordBox(0, 4, "甲", (98, 8, 120, 32)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=120,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    latin = [segment for segment in plan.for_block("text-1").lines[0].segments if segment.kind == "text_latin"]
    assert latin[0].bbox[0] == 42
    assert latin[0].text.startswith("I")


def test_compiler_blocks_latin_containing_line_without_ppocr_word_boxes():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 160, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(index=4, text="甲ABC乙", bbox=(0, 0, 140, 40), words=()))

    plan = compile_page_routing_plan(snapshot, prepass, page_width=160, page_height=50)

    assert plan.is_dispatchable is False
    assert [(issue.code, issue.line_index) for issue in plan.validation_issues] == [
        ("latin_line_missing_word_boxes", 4),
    ]
