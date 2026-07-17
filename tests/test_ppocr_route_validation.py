from __future__ import annotations

from dataclasses import replace

from app.core.ppocr_route_validation import validate_compiled_page_routing_plan
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)


def test_compiled_plan_validation_reports_page_local_geometry_issues():
    plan = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routingrun-test",
        layout_fingerprint="layout-fingerprint-test",
        prepass_run_id="run-1",
        blocks=(
            BlockRoutingPlan(
                block_uid="text-1",
                plan=RoutingPlan(
                    lines=(
                        RoutingLine(
                            index=4,
                            bbox=(20, 20, 20, 40),
                            segments=(
                                RoutingSegment(kind="text_other", bbox=(0, 0, 0, 10)),
                            ),
                        ),
                    ),
                    text_slices=(),
                    has_layout_routes=False,
                ),
            ),
        ),
    )

    issues = validate_compiled_page_routing_plan(
        plan,
        page_width=100,
        page_height=100,
    )

    assert {issue.code for issue in issues} >= {
        "compiled_line_empty_bbox",
        "compiled_segment_empty_bbox",
        "compiled_has_layout_routes_mismatch",
        "compiled_text_segment_missing_slice",
    }
    assert replace(plan, validation_issues=issues).is_dispatchable is False


def test_compiled_plan_validation_is_bidirectional_for_text_slices_and_segments():
    plan = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routingrun-test",
        layout_fingerprint="layout-fingerprint-test",
        prepass_run_id="run-1",
        blocks=(
            BlockRoutingPlan(
                block_uid="text-1",
                plan=RoutingPlan(
                    lines=(
                        RoutingLine(
                            index=3,
                            bbox=(0, 0, 40, 20),
                            segments=(
                                RoutingSegment(kind="text_other", bbox=(10, 0, 45, 20)),
                            ),
                        ),
                    ),
                    text_slices=(
                        TextSliceRoute(3, 0, (10, 0, 44, 20), True, kind="text_latin"),
                        TextSliceRoute(3, 0, (10, 0, 44, 20), True, kind="text_latin"),
                    ),
                    has_layout_routes=True,
                ),
            ),
        ),
    )

    issues = validate_compiled_page_routing_plan(
        plan,
        page_width=100,
        page_height=100,
    )

    assert {issue.code for issue in issues} >= {
        "compiled_segment_outside_line",
        "compiled_duplicate_segment_identity",
        "compiled_text_slice_mismatch",
    }
    assert replace(plan, validation_issues=issues).is_dispatchable is False


def test_compiled_plan_validation_rejects_out_of_page_and_duplicate_line_identity():
    line = RoutingLine(
        index=4,
        bbox=(0, 0, 101, 20),
        segments=(RoutingSegment(kind="text_other", bbox=(0, 0, 101, 20)),),
    )
    plan = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routingrun-test",
        layout_fingerprint="layout-fingerprint-test",
        prepass_run_id="run-1",
        blocks=(
            BlockRoutingPlan(
                block_uid="text-1",
                plan=RoutingPlan(
                    lines=(line, line),
                    text_slices=(
                        TextSliceRoute(4, 0, (0, 0, 101, 20), False),
                    ),
                    has_layout_routes=True,
                ),
            ),
        ),
    )

    issues = validate_compiled_page_routing_plan(
        plan,
        page_width=100,
        page_height=100,
    )

    assert {issue.code for issue in issues} >= {
        "compiled_line_out_of_page",
        "compiled_segment_out_of_page",
        "compiled_text_slice_out_of_page",
        "compiled_duplicate_line_index",
        "compiled_duplicate_segment_identity",
    }
    assert replace(plan, validation_issues=issues).is_dispatchable is False


def test_formula_content_bbox_may_extend_beyond_line_but_stays_page_bounded():
    plan = PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="routingrun-test",
        layout_fingerprint="layout-fingerprint-test",
        prepass_run_id="run-1",
        blocks=(
            BlockRoutingPlan(
                block_uid="text-1",
                plan=RoutingPlan(
                    lines=(
                        RoutingLine(
                            index=0,
                            bbox=(20, 20, 40, 40),
                            segments=(
                                RoutingSegment(
                                    kind="formula",
                                    bbox=(25, 25, 35, 35),
                                    content_bbox=(25, 10, 35, 50),
                                ),
                            ),
                        ),
                    ),
                    text_slices=(),
                    has_layout_routes=True,
                ),
            ),
        ),
    )

    assert validate_compiled_page_routing_plan(
        plan,
        page_width=100,
        page_height=100,
    ) == ()
