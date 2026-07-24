from __future__ import annotations

import numpy as np

from scripts.experiment_cjk_slot_bbox_cleanup import (
    _best_vertical_seam,
    _slot_x_bounds,
    _tight_foreground_bbox,
)


def _atom(index: int, text: str, bbox: tuple[int, int, int, int]) -> dict:
    return {
        "index": index,
        "text": text,
        "bbox": list(bbox),
        "source": "hanwang:micro_recblock",
        "granularity": "char",
    }


def test_vertical_seam_chooses_empty_projection_gap() -> None:
    foreground = np.zeros((30, 80), dtype=bool)
    foreground[5:25, 10:32] = True
    foreground[5:25, 45:68] = True

    seam, risky = _best_vertical_seam(foreground, (0, 30), 21.0, 56.0)

    assert 33 <= seam <= 44
    assert risky is False


def test_slot_bounds_use_projection_gap_between_cjk_atoms() -> None:
    foreground = np.zeros((30, 80), dtype=bool)
    foreground[5:25, 10:32] = True
    foreground[5:25, 45:68] = True
    atoms = [_atom(0, "民", (10, 5, 34, 25)), _atom(1, "族", (29, 5, 68, 25))]

    bounds = _slot_x_bounds(
        atoms,
        1,
        page_width=80,
        foreground=foreground,
        band=(0, 30),
    )

    assert bounds is not None
    left, right, risky = bounds
    assert 33 <= left <= 44
    assert right == 68
    assert risky is False


def test_tight_foreground_bbox_keeps_detached_parts_inside_slot() -> None:
    foreground = np.zeros((40, 60), dtype=bool)
    foreground[12:34, 22:42] = True
    foreground[5:9, 30:34] = True
    foreground[20:24, 8:11] = True

    bbox, area, component_count = _tight_foreground_bbox(
        foreground,
        (20, 3, 45, 36),
    )

    assert bbox == (22, 5, 42, 34)
    assert area == 456
    assert component_count == 2
