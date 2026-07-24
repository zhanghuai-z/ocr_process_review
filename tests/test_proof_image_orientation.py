from app.ui.image_orientation import (
    rotate_bbox,
    rotated_size,
    source_point_from_display,
)


def test_clockwise_bbox_and_point_roundtrip() -> None:
    bbox = (10, 20, 30, 60)

    displayed = rotate_bbox(bbox, 100, 200, 1)

    assert displayed == (140, 10, 180, 30)
    assert rotated_size(100, 200, 1) == (200, 100)
    assert source_point_from_display(160, 20, 100, 200, 1) == (20, 40)


def test_other_quarter_turn_bbox_transforms() -> None:
    bbox = (10, 20, 30, 60)

    assert rotate_bbox(bbox, 100, 200, 2) == (70, 140, 90, 180)
    assert rotate_bbox(bbox, 100, 200, 3) == (20, 70, 60, 90)
