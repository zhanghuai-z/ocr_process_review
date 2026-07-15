from __future__ import annotations

from app.core.ppocr_route_validation import validate_compiled_page_routing_plan
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
)


def test_compiled_plan_validation_reports_page_local_geometry_issues():
    plan = PageRoutingPlan(
        page_uid="page-1",
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
                    has_layout_routes=True,
                ),
            ),
        ),
    )

    assert [issue.code for issue in validate_compiled_page_routing_plan(plan)] == [
        "compiled_line_empty_bbox",
        "compiled_segment_empty_bbox",
    ]
