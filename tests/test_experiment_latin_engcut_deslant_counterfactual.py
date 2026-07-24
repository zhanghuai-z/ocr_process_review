from __future__ import annotations

import numpy as np

from app.engines.hanwang.engcut_payload import EngcutChar
from scripts.experiment_latin_engcut_deslant_counterfactual import (
    _counter_shear,
    _engcut_metrics,
    _selection_reasons,
    _token_isolated_canvas,
)


def test_token_canvas_copies_only_route_pixels_and_adds_equal_shift_guard():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:15, 10:20] = 80

    canvas, local_bbox = _token_isolated_canvas(image, (10, 5, 20, 15), 0.2)

    x1, y1, x2, y2 = local_bbox
    assert canvas.shape == (14, 22, 3)
    assert np.all(canvas[y1:y2, x1:x2] == 80)
    outside = canvas.copy()
    outside[y1:y2, x1:x2] = 255
    assert np.all(outside == 255)


def test_counter_shear_moves_upper_ink_left_and_keeps_baseline_fixed():
    canvas = np.full((14, 22, 3), 255, dtype=np.uint8)
    local_bbox = (6, 2, 16, 12)
    canvas[2, 12] = 0
    canvas[12, 12] = 0

    corrected = _counter_shear(canvas, local_bbox, 0.2)

    assert corrected[2, 10, 0] < 255
    assert corrected[12, 12, 0] == 0


def test_engcut_metrics_counts_overlap_within_native_group_only():
    chars = [
        EngcutChar("A", (1, 1, 7, 9), line_index=0, group_index=0),
        EngcutChar("B", (6, 1, 12, 9), line_index=0, group_index=0),
        EngcutChar("C", (2, 1, 8, 9), line_index=0, group_index=1),
    ]

    metrics = _engcut_metrics(chars)

    assert metrics["text"] == "AB C"
    assert metrics["char_count"] == 3
    assert metrics["overlap_pair_count"] == 1
    assert metrics["nonmonotonic_center_count"] == 0


def test_selection_requires_label_fallback_or_measured_right_slant():
    base = {
        "known_label": "unlabelled",
        "current_word_fallback": False,
        "structural_conflict": False,
        "slant": {"measurable": True, "slope": 0.15, "score_improvement": 0.06},
    }

    assert _selection_reasons(base) == ["right_slant_clean_control"]
    assert _selection_reasons({**base, "structural_conflict": True}) == [
        "right_slant_structural"
    ]
    assert _selection_reasons({
        **base,
        "slant": {"measurable": True, "slope": -0.2, "score_improvement": 0.2},
    }) == []
