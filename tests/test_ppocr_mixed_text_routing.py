from __future__ import annotations

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.core.charocr_text_partition import partition_charocr_text_region


def _image() -> np.ndarray:
    return np.full((50, 180, 3), 255, dtype=np.uint8)


def _ink(image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    image[y1:y2, x1:x2] = 0


def test_mixed_partition_routes_latin_mask_and_keeps_punctuation_with_other_route():
    image = _image()
    _ink(image, (10, 10, 28, 30))
    _ink(image, (43, 10, 48, 30))
    _ink(image, (52, 10, 57, 30))
    _ink(image, (61, 10, 66, 30))
    _ink(image, (83, 24, 87, 30))
    _ink(image, (102, 10, 120, 30))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲ABC,乙",
        bbox=(0, 0, 140, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (8, 8, 30, 32)),
            PpOcrV6WordBox(0, 1, "ABC", (40, 8, 75, 32)),
            PpOcrV6WordBox(0, 2, ",", (80, 8, 90, 32)),
            PpOcrV6WordBox(0, 3, "乙", (100, 8, 122, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 140, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_other", (10, 10, 28, 30), "甲"),
        ("text_latin", (43, 10, 66, 30), "ABC"),
        ("text_other", (83, 10, 120, 30), ",乙"),
    ]


def test_punctuation_capacity_cannot_claim_leading_i_after_its_comma_is_owned():
    image = _image()
    _ink(image, (2, 10, 18, 30))       # 甲
    _ink(image, (22, 24, 26, 30))      # comma
    _ink(image, (30, 10, 34, 30))      # i stem outside PP word box
    _ink(image, (31, 4, 33, 8))        # i dot outside PP word box
    _ink(image, (38, 10, 43, 30))      # n
    _ink(image, (72, 10, 88, 30))      # 乙
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲,in乙",
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (0, 8, 20, 32)),
            PpOcrV6WordBox(0, 1, ",", (20, 8, 29, 32)),
            PpOcrV6WordBox(0, 2, "in", (36, 8, 50, 32)),
            PpOcrV6WordBox(0, 3, "乙", (70, 8, 90, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 100, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((30, 4, 43, 30), "in")]


def test_shifted_punctuation_proposal_does_not_block_when_cjk_owns_its_ink():
    image = _image()
    _ink(image, (10, 10, 28, 30))       # 甲
    _ink(image, (28, 24, 32, 30))       # actual comma, left of its proposal
    _ink(image, (46, 10, 51, 30))       # A
    _ink(image, (55, 10, 60, 30))       # B
    _ink(image, (64, 10, 69, 30))       # C
    _ink(image, (92, 10, 110, 30))      # 乙
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲,ABC乙",
        bbox=(0, 0, 130, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (8, 8, 30, 32)),
            PpOcrV6WordBox(0, 1, ",", (32, 8, 38, 32)),
            PpOcrV6WordBox(0, 2, "ABC", (43, 8, 75, 32)),
            PpOcrV6WordBox(0, 3, "乙", (90, 8, 112, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 130, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_other", (10, 10, 32, 30), "甲"),
        ("text_latin", (46, 10, 69, 30), "ABC"),
        ("text_other", (92, 10, 110, 30), "乙"),
    ]
