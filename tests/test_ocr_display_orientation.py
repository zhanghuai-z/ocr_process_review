from app.core.ocr_display_orientation import (
    display_rotation_from_metadata,
    observe_page_display_orientation,
)
from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
)


def _line(index, bbox, *, axis="horizontal", angle=-1):
    return RoutingLine(
        index=index,
        bbox=bbox,
        segments=(RoutingSegment(kind="text_other", bbox=bbox),),
        text_axis=axis,
        orientation_angle=angle,
    )


def _plan(lines) -> PageRoutingPlan:
    return PageRoutingPlan(
        page_uid="page-1",
        routing_run_uid="route-1",
        layout_fingerprint="layout-1",
        prepass_run_id="prepass-1",
        blocks=(BlockRoutingPlan(
            block_uid="block-1",
            plan=RoutingPlan(lines=tuple(lines), text_slices=(), has_layout_routes=True),
        ),),
    )


def test_sideways_table_page_uses_oriented_text_consensus() -> None:
    plan = _plan((
            _line(0, (365, 540, 411, 738), axis="vertical", angle=180),
            _line(1, (348, 858, 410, 2856), axis="vertical", angle=180),
            _line(2, (411, 342, 485, 377)),
            _line(3, (977, 333, 1385, 377)),
    ))

    observation = observe_page_display_orientation(plan)

    assert observation.quarters_clockwise == 1
    assert observation.weights[1] > observation.weights[0]
    assert display_rotation_from_metadata(observation.metadata()) == 1


def test_weak_vertical_margin_does_not_rotate_a_horizontal_page() -> None:
    plan = _plan((
            _line(0, (20, 20, 980, 70)),
            _line(1, (20, 100, 980, 150)),
            _line(2, (10, 200, 45, 340), axis="vertical", angle=180),
    ))

    assert observe_page_display_orientation(plan).quarters_clockwise == 0


def test_invalid_or_missing_rotation_metadata_is_zero() -> None:
    assert display_rotation_from_metadata(()) == 0
    assert display_rotation_from_metadata(
        (("display_rotation_quarters_clockwise", "9"),)
    ) == 0
