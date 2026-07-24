from __future__ import annotations

import numpy as np

from app.geometry.text_decoration import (
    DOT_LEADER,
    DecorationLine,
    DecorationToken,
    TextDecoration,
    detect_text_decorations,
    detect_page_text_decorations,
)


def _leader_image(dot_count: int, *, spacing: int = 18) -> np.ndarray:
    image = np.full((80, 600, 3), 255, dtype=np.uint8)
    for index in range(dot_count):
        x = 140 + index * spacing
        image[42:47, x:x + 5] = 0
    image[18:66, 520:530] = 0
    return image


def test_detects_long_dot_leader_without_absorbing_trailing_parenthesis():
    image = _leader_image(20)

    result = detect_text_decorations(
        image,
        (10, 10, 590, 70),
        (DecorationToken("……(", (120, 10, 540, 70)),),
    )

    assert result == (TextDecoration(DOT_LEADER, (140, 42, 487, 47)),)


def test_extends_proven_leader_to_regular_mark_clipped_by_vendor_token() -> None:
    image = _leader_image(20)
    image[42:47, 100:105] = 0

    result = detect_text_decorations(
        image,
        (10, 10, 590, 70),
        (DecorationToken("……(", (158, 10, 540, 70)),),
    )

    assert result == (TextDecoration(DOT_LEADER, (140, 42, 487, 47)),)


def test_does_not_classify_ordinary_ellipsis_as_a_dot_leader():
    image = _leader_image(6)

    assert detect_text_decorations(
        image,
        (10, 10, 590, 70),
        (DecorationToken("……", (120, 10, 260, 70)),),
    ) == ()


def test_requires_vendor_punctuation_geometry_not_plain_text_guessing():
    image = _leader_image(20)

    assert detect_text_decorations(
        image,
        (10, 10, 590, 70),
        (DecorationToken("chapter", (120, 10, 540, 70)),),
    ) == ()


def test_punctuation_token_does_not_authorize_scanning_distant_text_components():
    image = _leader_image(20)

    assert detect_text_decorations(
        image,
        (10, 10, 590, 70),
        (DecorationToken(",", (40, 10, 55, 70)),),
    ) == ()


def test_detects_one_leader_across_split_co_baseline_vendor_lines():
    image = _leader_image(20)

    result = detect_page_text_decorations(
        image,
        (
            DecorationLine(
                (10, 10, 250, 70),
                (DecorationToken("……", (120, 10, 250, 70)),),
            ),
            DecorationLine(
                (450, 10, 590, 70),
                (DecorationToken("……(", (450, 10, 540, 70)),),
            ),
        ),
    )

    assert result == (TextDecoration(DOT_LEADER, (140, 42, 487, 47)),)


def test_detects_unboxed_leader_in_gap_between_co_baseline_lines():
    image = _leader_image(20)

    result = detect_page_text_decorations(
        image,
        (
            DecorationLine(
                (10, 10, 130, 70),
                (DecorationToken("chapter", (10, 10, 120, 70)),),
            ),
            DecorationLine(
                (500, 10, 590, 70),
                (DecorationToken("(12)", (500, 10, 590, 70)),),
            ),
        ),
    )

    assert result == (TextDecoration(DOT_LEADER, (140, 42, 487, 47)),)
