from __future__ import annotations

import cv2
import numpy as np

from app.geometry.char_reconciler import reconcile_char_geometry
from app.models.char_geometry import NativeGeometryProposal


def _canvas(width: int = 180, height: int = 80) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_merges_two_native_proposals_claiming_one_visual_glyph() -> None:
    image = _canvas()
    cv2.rectangle(image, (30, 15), (66, 62), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (30, 17, 55, 60), 0.19),
        NativeGeometryProposal(1, (48, 15, 67, 61), 0.19),
        NativeGeometryProposal(2, (90, 15, 128, 63), 0.95),
    )

    result = reconcile_char_geometry(image, proposals)

    assert len(result.atoms) == 2
    assert result.atoms[0].proposal_indices == (0, 1)
    assert result.atoms[0].bbox == (30, 15, 67, 63)
    assert result.atoms[0].granularity == "token"
    assert result.atoms[0].reason == "foreground_conflict_reconciled"
    assert result.atoms[1].proposal_indices == (2,)


def test_fused_dash_uses_main_component_and_drops_isolated_noise() -> None:
    image = _canvas()
    cv2.rectangle(image, (20, 50), (136, 54), (0, 0, 0), -1)
    cv2.rectangle(image, (32, 16), (34, 18), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (20, 15, 76, 55), 0.19),
        NativeGeometryProposal(1, (68, 49, 137, 56), 0.19),
    )

    result = reconcile_char_geometry(image, proposals)

    assert len(result.atoms) == 1
    assert result.atoms[0].proposal_indices == (0, 1)
    assert result.atoms[0].bbox == (20, 50, 137, 55)


def test_underfilled_cjk_slot_absorbs_adjacent_fragment() -> None:
    image = _canvas(width=220)
    # Three visual strokes occupy one normal CJK slot.
    cv2.rectangle(image, (80, 30), (99, 55), (0, 0, 0), -1)
    cv2.rectangle(image, (96, 16), (110, 64), (0, 0, 0), -1)
    cv2.rectangle(image, (115, 30), (130, 53), (0, 0, 0), -1)
    # Stable neighboring character establishes the line's slot width.
    cv2.rectangle(image, (20, 16), (69, 64), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (20, 16, 70, 65), 0.95),
        NativeGeometryProposal(1, (80, 16, 112, 64), 0.19),
        NativeGeometryProposal(2, (96, 16, 111, 64), 0.19),
        NativeGeometryProposal(3, (115, 29, 123, 54), 0.19),
    )

    result = reconcile_char_geometry(image, proposals)

    assert len(result.atoms) == 2
    assert result.atoms[0].proposal_indices == (0,)
    assert result.atoms[1].proposal_indices == (1, 2, 3)
    assert result.atoms[1].bbox == (80, 16, 131, 65)
    assert result.atoms[1].granularity == "token"


def test_does_not_merge_neighboring_boxes_without_shared_foreground() -> None:
    image = _canvas()
    cv2.rectangle(image, (20, 18), (55, 61), (0, 0, 0), -1)
    cv2.rectangle(image, (62, 18), (97, 61), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (20, 18, 56, 62), 0.19),
        NativeGeometryProposal(1, (62, 18, 98, 62), 0.19),
    )

    result = reconcile_char_geometry(image, proposals)

    assert [atom.proposal_indices for atom in result.atoms] == [(0,), (1,)]


def test_does_not_merge_touching_glyphs_with_distinct_dominant_components() -> None:
    image = _canvas()
    cv2.rectangle(image, (20, 18), (48, 61), (0, 0, 0), -1)
    cv2.rectangle(image, (52, 18), (90, 61), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (20, 18, 56, 62), 0.19),
        NativeGeometryProposal(1, (44, 18, 91, 62), 0.19),
    )

    result = reconcile_char_geometry(image, proposals)

    assert [atom.proposal_indices for atom in result.atoms] == [(0,), (1,)]


def test_does_not_merge_one_component_spanning_two_normal_slots() -> None:
    image = _canvas(width=240)
    cv2.rectangle(image, (20, 16), (69, 64), (0, 0, 0), -1)
    cv2.rectangle(image, (76, 16), (175, 64), (0, 0, 0), -1)
    proposals = (
        NativeGeometryProposal(0, (20, 16, 70, 65), 0.19),
        NativeGeometryProposal(1, (76, 16, 130, 65), 0.19),
        NativeGeometryProposal(2, (123, 16, 176, 65), 0.19),
    )

    result = reconcile_char_geometry(image, proposals)

    assert [atom.proposal_indices for atom in result.atoms] == [(0,), (1,), (2,)]
