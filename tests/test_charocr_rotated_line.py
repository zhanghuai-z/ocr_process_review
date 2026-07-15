from __future__ import annotations

import numpy as np

from app.engines.hanwang.micro_recblock import (
    CharResult,
    LineResult,
    RunStats,
    _LineCutMaskedLineRoute,
    _TextRoute,
    _line_results_from_oriented_crop,
    _orient_crop_for_native,
    _recognize_oriented_linecut_route,
)


def test_counterclockwise_native_rotation_maps_bboxes_back_to_source():
    crop = np.zeros((80, 30, 3), dtype=np.uint8)
    oriented = _orient_crop_for_native(
        crop,
        text_axis="vertical",
        orientation_angle=0,
    )

    assert oriented.image.shape[:2] == (30, 80)
    assert oriented.rotation_quarters_clockwise == 3
    assert oriented.bbox_to_source((20, 5, 50, 20)) == (10, 20, 25, 50)


def test_clockwise_native_rotation_maps_line_and_char_geometry_back_to_source():
    crop = np.zeros((80, 30, 3), dtype=np.uint8)
    oriented = _orient_crop_for_native(
        crop,
        text_axis="vertical",
        orientation_angle=180,
    )
    native = [LineResult(
        text="甲",
        bbox=(20, 5, 50, 20),
        chars=[CharResult(text="甲", bbox=(22, 7, 48, 18))],
    )]

    restored = _line_results_from_oriented_crop(native, oriented)

    assert oriented.image.shape[:2] == (30, 80)
    assert oriented.rotation_quarters_clockwise == 1
    assert restored[0].bbox == (5, 30, 20, 60)
    assert restored[0].chars[0].bbox == (7, 32, 18, 58)


def test_horizontal_line_without_orientation_change_keeps_native_geometry():
    crop = np.zeros((30, 80, 3), dtype=np.uint8)
    oriented = _orient_crop_for_native(
        crop,
        text_axis="horizontal",
        orientation_angle=0,
    )
    native = [LineResult(text="甲", bbox=(5, 4, 25, 24))]

    assert oriented.image is crop
    assert _line_results_from_oriented_crop(native, oriented) is native


def test_horizontal_180_large_line_is_normalized_before_native_recognition(monkeypatch):
    image = np.full((80, 200, 3), 255, dtype=np.uint8)
    segment = _TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(0, 0, 200, 80),
        kind="text_other",
    )
    route = _LineCutMaskedLineRoute(
        block_idx=0,
        line_idx=0,
        bbox=(0, 0, 200, 80),
        linecut_segments=(segment,),
        excluded_segments=(),
        text_axis="horizontal",
        orientation_angle=180,
    )

    def fake_recog(crop, **_kwargs):
        assert crop.shape[:2] == (56, 140)
        return {
            "lines": [{
                "groups": [{
                    "bbox": {"left": 0, "top": 0, "right": 140, "bottom": 56},
                    "chars": [{
                        "codes": [int.from_bytes("甲".encode("gbk"), "little")],
                        "scores": [8],
                        "bbox": {"left": 7, "top": 7, "right": 35, "bottom": 49},
                    }],
                }],
            }],
        }

    monkeypatch.setattr(
        "app.engines.hanwang.micro_recblock.native_bridge.run_linecut_recog",
        fake_recog,
    )

    routed = _recognize_oriented_linecut_route(
        image,
        route,
        RunStats(),
        timeout=30,
        include_chars=True,
    )

    line = routed[(0, 0, 0)][0]
    assert line.text == "甲"
    assert line.chars[0].bbox == (150, 10, 190, 70)
