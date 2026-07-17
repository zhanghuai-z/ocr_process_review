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
from app.core.ppocr_route_compiler import compile_page_routing_plan as _compile_page_routing_plan
from app.models import BBox, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot, OcrPolicy
from tests.charocr_routing_observation_fixture import routing_observation_bundle


def compile_page_routing_plan(snapshot, prepass, **kwargs):
    block_vl_observations = kwargs.pop("block_vl_observations", ())
    return _compile_page_routing_plan(
        routing_observation_bundle(
            snapshot,
            prepass,
            block_vl_observations=block_vl_observations,
        ),
        **kwargs,
    )


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
        ("text_other", (0, 0, 300, 40)),
        ("skip", (100, 0, 180, 40)),
        ("formula", (200, 0, 220, 40)),
    ]
    assert len(plan.for_block("text-1").lines) == 1


def test_compiler_requires_exact_block_local_vl_pp_text_alignment():
    from app.core.layout_scope import layout_snapshot_fingerprint
    from app.models.ocr_routing_observation import (
        BlockVlObservation,
        BlockVlObservationStatus,
        BlockVlTextRegion,
    )

    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 120, 40), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(0, "甲 乙", (0, 0, 120, 40), ()))

    def observation(text):
        return BlockVlObservation(
            page_uid="page-1",
            block_uid="text-1",
            block_bbox=(0, 0, 120, 40),
            image_hash="image-hash-test",
            layout_fingerprint=layout_snapshot_fingerprint(snapshot),
            status=BlockVlObservationStatus.OBSERVED,
            regions=(BlockVlTextRegion(0, "text", text, (0, 0, 120, 40)),),
        )

    exact = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=120,
        page_height=40,
        block_vl_observations=(observation("甲乙"),),
    )
    mismatch = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=120,
        page_height=40,
        block_vl_observations=(observation("甲丙"),),
    )

    assert exact.is_dispatchable is True
    assert exact.alignments[0].status.value == "exact"
    assert mismatch.is_dispatchable is False
    assert [issue.code for issue in mismatch.validation_issues] == [
        "ambiguous_block_text_alignment",
    ]


def test_text_block_owns_row_without_clipping_ppocr_geometry():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (10, 10, 90, 30), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(
        PpOcrV6LineHint(index=0, text="城市", bbox=(0, 0, 100, 40), words=()),
    )

    plan = compile_page_routing_plan(snapshot, prepass, page_width=100, page_height=40)

    assert plan.is_dispatchable is True
    route = plan.for_block("text-1").lines[0]
    assert route.bbox == (0, 0, 100, 40)
    assert route.segments[0].bbox == (0, 0, 100, 40)


def test_compiler_routes_directory_leader_as_non_text_decoration():
    image = np.full((80, 600, 3), 255, dtype=np.uint8)
    image[20:65, 20:100] = 0
    for index in range(20):
        x = 140 + index * 18
        image[42:47, x:x + 5] = 0
    image[18:66, 520:530] = 0
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 600, 80), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="目录……(1)",
        bbox=(10, 10, 590, 70),
        words=(
            PpOcrV6WordBox(0, 0, "目录", (20, 10, 100, 70)),
            PpOcrV6WordBox(0, 1, "……(", (120, 10, 540, 70)),
            PpOcrV6WordBox(0, 2, "1", (545, 10, 560, 70)),
            PpOcrV6WordBox(0, 3, ")", (565, 10, 580, 70)),
        ),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=600,
        page_height=80,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("text-1").lines[0]
    decorations = [segment for segment in route.segments if segment.kind == "decoration"]
    assert [(segment.label, segment.bbox) for segment in decorations] == [
        ("dot_leader", (140, 42, 487, 47)),
    ]
    assert all(slice_.segment_index != route.segments.index(decorations[0]) for slice_ in plan.for_block("text-1").text_slices)


def test_compiler_preserves_rotated_ppocr_axis_and_geometry_beyond_layout_edge():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (20, 40, 80, 180), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="人口数",
        bbox=(10, 20, 90, 200),
        words=(),
        text_axis="vertical",
        orientation_angle=0,
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=220,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("text-1").lines[0]
    assert route.bbox == (10, 20, 90, 200)
    assert route.text_axis == "vertical"
    assert route.orientation_angle == 0


def test_page_boundary_clipping_preserves_rotated_line_orientation_facts():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 20, 90, 200), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="人口数",
        bbox=(-10, 20, 90, 200),
        words=(),
        text_axis="vertical",
        orientation_angle=0,
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=220,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("text-1").lines[0]
    assert route.bbox == (0, 20, 90, 200)
    assert route.text_axis == "vertical"
    assert route.orientation_angle == 0


def test_compiler_blocks_only_text_page_with_unoriented_rotated_ppocr_row():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (20, 20, 80, 200), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="人口数",
        bbox=(20, 20, 80, 200),
        words=(),
        text_axis="vertical",
        orientation_angle=-1,
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=220,
    )

    assert plan.is_dispatchable is False
    assert [issue.code for issue in plan.validation_issues] == [
        "missing_rotated_line_orientation"
    ]


def test_unoriented_rotated_table_row_does_not_block_text_routing_page():
    snapshot = _snapshot(
        _block("table-1", BlockType.TABLE, (20, 20, 80, 200), policy=OcrPolicy.PRESERVE_AS_TABLE, order=0),
    )
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="人口数",
        bbox=(20, 20, 80, 200),
        words=(),
        text_axis="vertical",
        orientation_angle=-1,
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=220,
    )

    assert plan.is_dispatchable is True
    assert plan.for_block("table-1") is None


def test_structural_edge_contact_does_not_steal_text_token_ownership():
    snapshot = _snapshot(
        _block("caption", BlockType.FIGURE_CAPTION, (0, 10, 100, 30), policy=OcrPolicy.TEXT_OCR, order=0),
        _block("table", BlockType.TABLE, (0, 30, 100, 60), policy=OcrPolicy.PRESERVE_AS_TABLE, order=1),
    )
    prepass = _prepass(
        PpOcrV6LineHint(
            index=0,
            text="表3",
            bbox=(0, 0, 100, 40),
            words=(
                PpOcrV6WordBox(0, 0, "表", (10, 0, 55, 40)),
                PpOcrV6WordBox(0, 1, "3", (60, 0, 75, 40)),
            ),
        ),
    )
    image = np.full((60, 100, 3), 255, dtype=np.uint8)
    image[12:28, 12:50] = 0
    image[12:28, 61:74] = 0

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=100,
        page_height=60,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("caption").lines[0]
    assert route.bbox == (0, 12, 100, 28)
    assert any(segment.kind == "text_latin" for segment in route.segments)
    assert all(segment.kind != "skip" for segment in route.segments)


def test_formula_mask_owns_edge_token_when_no_text_ink_remains():
    snapshot = _snapshot(
        _block("text", BlockType.TEXT, (0, 0, 120, 40), policy=OcrPolicy.TEXT_OCR, order=0),
        _block("formula", BlockType.EQUATION, (52, 0, 80, 40), policy=OcrPolicy.PRESERVE_AS_FORMULA, order=1),
    )
    prepass = _prepass(
        PpOcrV6LineHint(
            index=0,
            text="甲D_t乙",
            bbox=(0, 0, 120, 40),
            words=(
                PpOcrV6WordBox(0, 0, "甲", (5, 0, 30, 40)),
                PpOcrV6WordBox(0, 1, "D", (45, 0, 55, 40)),
                PpOcrV6WordBox(0, 2, "_", (55, 0, 60, 40)),
                PpOcrV6WordBox(0, 3, "t", (60, 0, 70, 40)),
                PpOcrV6WordBox(0, 4, "乙", (85, 0, 110, 40)),
            ),
        ),
    )
    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    image[10:30, 8:28] = 0
    image[10:30, 52:68] = 0
    image[10:30, 88:108] = 0

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=120,
        page_height=40,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("text").lines[0]
    assert all(segment.kind != "text_latin" for segment in route.segments)
    formula = next(segment for segment in route.segments if segment.kind == "formula")
    assert formula.bbox == (52, 10, 80, 30)
    assert formula.content_bbox == (52, 0, 80, 40)


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


def test_compiler_keeps_full_text_row_when_formula_only_touches_its_edge():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 240, 140), policy=OcrPolicy.TEXT_OCR, order=0),
        _block(
            "formula-1",
            BlockType.EQUATION,
            (20, 10, 90, 48),
            policy=OcrPolicy.PRESERVE_AS_FORMULA,
            order=1,
            label="inline_formula",
        ),
    )
    plan = compile_page_routing_plan(
        snapshot,
        _prepass(PpOcrV6LineHint(index=1, text="生产率", bbox=(0, 40, 220, 100), words=())),
        page_width=240,
        page_height=140,
    )

    route = plan.for_block("text-1").lines[0]
    assert [(segment.kind, segment.bbox) for segment in route.segments] == [
        ("text_other", (0, 40, 220, 100)),
        ("formula", (20, 40, 90, 48)),
    ]


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


def test_compiler_keeps_isolated_numeric_footnote_marker_on_linecut():
    from app.core.layout_scope import layout_snapshot_fingerprint
    from app.models.ocr_routing_observation import (
        BlockVlObservation,
        BlockVlObservationStatus,
        BlockVlTextRegion,
    )

    snapshot = _snapshot(
        _block(
            "footnote-1",
            BlockType.TEXT,
            (0, 0, 140, 50),
            policy=OcrPolicy.TEXT_OCR,
            order=0,
            label="footnote",
        ),
    )
    image = np.full((50, 140, 3), 255, dtype=np.uint8)
    image[10:30, 8:22] = 0
    for x in (45, 55, 65, 75, 85):
        image[10:30, x:x + 4] = 0
    prepass = _prepass(
        PpOcrV6LineHint(
            index=0,
            text="Barry",
            bbox=(38, 5, 110, 38),
            words=(PpOcrV6WordBox(0, 0, "Barry", (42, 8, 100, 34)),),
        ),
        PpOcrV6LineHint(
            index=1,
            text="8",
            bbox=(4, 5, 28, 38),
            words=(PpOcrV6WordBox(1, 0, "8", (8, 8, 22, 34)),),
        ),
    )

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=140,
        page_height=50,
        page_image_bgr=image,
        block_vl_observations=(BlockVlObservation(
            page_uid="page-1",
            block_uid="footnote-1",
            block_bbox=(0, 0, 140, 50),
            image_hash="image-hash-test",
            layout_fingerprint=layout_snapshot_fingerprint(snapshot),
            status=BlockVlObservationStatus.OBSERVED,
            regions=(BlockVlTextRegion(0, "footnote", "⑧ Barry!", (0, 0, 140, 50)),),
        ),),
    )

    assert plan.is_dispatchable is True
    assert plan.alignments[0].status.value == "matched"
    route = plan.for_block("footnote-1").lines[0]
    assert [segment.kind for segment in route.segments] == ["text_other", "text_latin"]
    assert [
        token.text
        for segment in route.segments
        for token in segment.ppocr_latin_tokens
    ] == ["Barry"]
    assert [(item.code, item.bbox) for item in plan.diagnostics] == [
        ("vl_marker_owned_by_linecut", (4, 10, 38, 30)),
    ]


def test_compiler_keeps_vl_marker_ink_on_linecut_when_ppocr_omits_marker():
    from app.core.layout_scope import layout_snapshot_fingerprint
    from app.models.ocr_routing_observation import (
        BlockVlObservation,
        BlockVlObservationStatus,
        BlockVlTextRegion,
    )

    snapshot = _snapshot(
        _block(
            "footnote-1",
            BlockType.TEXT,
            (0, 0, 140, 50),
            policy=OcrPolicy.TEXT_OCR,
            order=0,
            label="footnote",
        ),
    )
    image = np.full((50, 140, 3), 255, dtype=np.uint8)
    image[10:30, 8:22] = 0
    for x in (45, 55, 65, 75, 85):
        image[10:30, x:x + 4] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="Barry",
        bbox=(38, 5, 110, 38),
        words=(PpOcrV6WordBox(0, 0, "Barry", (42, 8, 100, 34)),),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=140,
        page_height=50,
        page_image_bgr=image,
        block_vl_observations=(BlockVlObservation(
            page_uid="page-1",
            block_uid="footnote-1",
            block_bbox=(0, 0, 140, 50),
            image_hash="image-hash-test",
            layout_fingerprint=layout_snapshot_fingerprint(snapshot),
            status=BlockVlObservationStatus.OBSERVED,
            regions=(BlockVlTextRegion(0, "footnote", "⑫ Barry", (0, 0, 140, 50)),),
        ),),
    )

    assert plan.is_dispatchable is True
    route = plan.for_block("footnote-1").lines[0]
    assert [segment.kind for segment in route.segments] == ["text_other", "text_latin"]
    assert route.segments[0].bbox[2] <= route.segments[1].bbox[0]
    assert route.segments[1].text == "Barry"


def test_compiler_blocks_vl_marker_alignment_below_sixty_percent():
    from app.core.layout_scope import layout_snapshot_fingerprint
    from app.models.ocr_routing_observation import (
        BlockVlObservation,
        BlockVlObservationStatus,
        BlockVlTextRegion,
    )

    snapshot = _snapshot(
        _block("footnote-1", BlockType.TEXT, (0, 0, 140, 50), policy=OcrPolicy.TEXT_OCR, order=0, label="footnote"),
    )
    image = np.full((50, 140, 3), 255, dtype=np.uint8)
    for x in (45, 55, 65, 75, 85):
        image[10:30, x:x + 4] = 0
    prepass = _prepass(PpOcrV6LineHint(
        index=0,
        text="Barry",
        bbox=(38, 5, 110, 38),
        words=(PpOcrV6WordBox(0, 0, "Barry", (42, 8, 100, 34)),),
    ))

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=140,
        page_height=50,
        page_image_bgr=image,
        block_vl_observations=(BlockVlObservation(
            page_uid="page-1",
            block_uid="footnote-1",
            block_bbox=(0, 0, 140, 50),
            image_hash="image-hash-test",
            layout_fingerprint=layout_snapshot_fingerprint(snapshot),
            status=BlockVlObservationStatus.OBSERVED,
            regions=(BlockVlTextRegion(0, "footnote", "⑫ unrelated", (0, 0, 140, 50)),),
        ),),
    )

    assert plan.is_dispatchable is False
    assert [issue.code for issue in plan.validation_issues] == [
        "ambiguous_block_text_alignment",
    ]


def test_compiler_does_not_reclassify_numeric_prefix_outside_footnote():
    snapshot = _snapshot(
        _block("text-1", BlockType.TEXT, (0, 0, 140, 50), policy=OcrPolicy.TEXT_OCR, order=0),
    )
    image = np.full((50, 140, 3), 255, dtype=np.uint8)
    for x in (8, 14, 20):
        image[12:28, x:x + 2] = 0
    for x in (45, 55, 65, 75, 85):
        image[12:28, x:x + 3] = 0
    prepass = _prepass(
        PpOcrV6LineHint(
            index=0,
            text="Barry",
            bbox=(38, 5, 110, 38),
            words=(PpOcrV6WordBox(0, 0, "Barry", (42, 8, 100, 34)),),
        ),
        PpOcrV6LineHint(
            index=1,
            text="8",
            bbox=(4, 5, 28, 38),
            words=(PpOcrV6WordBox(1, 0, "8", (8, 8, 22, 34)),),
        ),
    )

    plan = compile_page_routing_plan(
        snapshot,
        prepass,
        page_width=140,
        page_height=50,
        page_image_bgr=image,
    )

    assert plan.is_dispatchable is True
    assert [
        (segment.kind, segment.text)
        for segment in plan.for_block("text-1").lines[0].segments
    ] == [
        ("text_latin", "8"),
        ("text_latin", "Barry"),
    ]
    assert plan.diagnostics == ()


def test_compiler_blocks_pure_latin_row_without_word_masks():
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
    assert [issue.code for issue in plan.validation_issues] == [
        "latin_line_missing_word_boxes",
    ]


def test_compiler_routes_quoted_pure_latin_row_by_typed_ink_ownership():
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
    assert [
        (segment.kind, segment.bbox, segment.text)
        for segment in plan.for_block("text-1").lines[0].segments
    ] == [
        ("text_other", (0, 8, 30, 30), ""),
        ("text_latin", (30, 8, 40, 30), "A"),
        ("text_other", (40, 8, 70, 30), ""),
    ]


def test_compiler_preserves_full_text_route_and_adds_exact_formula_mask():
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
    prepass = _prepass(PpOcrV6LineHint(index=0, text="甲", bbox=(0, 0, 160, 50), words=()))

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
        ("text_other", (0, 0, 160, 50)),
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
    assert [segment.text for segment in latin] == ["one", "two"]


def test_compiler_keeps_ambiguous_latin_glyph_on_linecut_with_diagnostic():
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
    assert plan.validation_issues == ()
    assert [(item.code, item.line_index, item.bbox) for item in plan.diagnostics] == [
        ("missing_latin_token_ink", 0, (49, 8, 55, 32)),
    ]
    line = plan.for_block("text-1").lines[0]
    assert [segment.kind for segment in line.segments] == ["text_other", "text_latin", "text_other"]


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
