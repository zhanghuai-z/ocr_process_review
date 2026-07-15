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
        ("text_other", (0, 0, 43, 40), ""),
        ("text_latin", (43, 10, 66, 30), "ABC"),
        ("text_other", (66, 0, 140, 40), ""),
    ]


def test_ambiguous_leading_i_stays_in_linecut_instead_of_crossing_comma_boundary():
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
    assert [(segment.bbox, segment.text) for segment in latin] == [((38, 10, 43, 30), "in")]
    assert any(
        segment.kind == "text_other" and segment.bbox[0] <= 30 < segment.bbox[2]
        for segment in result.segments
    )


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
        ("text_other", (0, 0, 46, 40), ""),
        ("text_latin", (46, 10, 69, 30), "ABC"),
        ("text_other", (69, 0, 130, 40), ""),
    ]


def test_punctuation_fragment_before_latin_mask_cannot_pull_cjk_into_engcut():
    image = _image()
    _ink(image, (10, 10, 28, 30))      # 甲
    _ink(image, (38, 20, 43, 28))      # detached quote fragment
    _ink(image, (58, 10, 76, 30))      # 乙
    _ink(image, (82, 10, 100, 30))     # 丙
    _ink(image, (132, 10, 137, 30))    # A
    _ink(image, (141, 10, 146, 30))    # B
    _ink(image, (150, 10, 155, 30))    # C
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='甲”乙丙ABC',
        bbox=(0, 0, 170, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (8, 8, 30, 32)),
            PpOcrV6WordBox(0, 1, "”", (44, 8, 50, 32)),
            PpOcrV6WordBox(0, 2, "乙", (54, 8, 78, 32)),
            PpOcrV6WordBox(0, 3, "丙", (80, 8, 104, 32)),
            PpOcrV6WordBox(0, 4, "ABC", (130, 8, 160, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 170, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_other", (0, 0, 132, 40), ""),
        ("text_latin", (132, 10, 155, 30), "ABC"),
    ]


def test_pure_latin_row_keeps_punctuation_in_full_engcut_context():
    image = _image()
    _ink(image, (20, 10, 25, 30))      # A
    _ink(image, (37, 20, 42, 28))      # detached quote fragment
    _ink(image, (64, 10, 69, 30))      # B
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='A”B',
        bbox=(0, 0, 90, 40),
        words=(
            PpOcrV6WordBox(0, 0, "A", (18, 8, 28, 32)),
            PpOcrV6WordBox(0, 1, "”", (43, 8, 49, 32)),
            PpOcrV6WordBox(0, 2, "B", (62, 8, 72, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 90, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 90, 40), "A”B"),
    ]


def test_shifted_two_part_quote_is_owned_wholly_by_symbol_route():
    image = _image()
    _ink(image, (20, 10, 25, 30))      # final Latin body
    _ink(image, (34, 4, 40, 12))       # left half of closing quote
    _ink(image, (45, 4, 51, 12))       # right half inside quote proposal
    _ink(image, (70, 10, 88, 30))      # following CJK
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='A”甲',
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "A", (18, 8, 30, 32)),
            PpOcrV6WordBox(0, 1, "”", (43, 2, 53, 16)),
            PpOcrV6WordBox(0, 2, "甲", (68, 8, 90, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 100, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (20, 10, 25, 30), "A"),
        ("text_other", (25, 0, 100, 40), ""),
    ]


def test_multi_glyph_symbol_reclaims_glyph_left_of_its_raw_box_from_latin_mask():
    image = _image()
    _ink(image, (10, 10, 30, 30))      # Latin body
    _ink(image, (34, 8, 39, 31))       # right parenthesis, left of symbol proposal
    _ink(image, (45, 22, 50, 29))      # enumeration comma inside symbol proposal
    _ink(image, (62, 10, 82, 30))      # following CJK
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="A）、甲",
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "A", (8, 8, 33, 32)),
            PpOcrV6WordBox(0, 1, "）、", (40, 8, 52, 32)),
            PpOcrV6WordBox(0, 2, "甲", (60, 8, 84, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 100, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (10, 10, 30, 30), "A"),
        ("text_other", (30, 0, 100, 40), ""),
    ]


def test_pure_latin_row_does_not_depend_on_quote_component_ownership():
    image = _image()
    _ink(image, (20, 10, 25, 30))      # A body
    _ink(image, (39, 17, 44, 29))      # quote fragment, inside the measured seed only
    _ink(image, (64, 10, 69, 30))      # B body
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='AA”B',
        bbox=(0, 0, 90, 40),
        words=(
            PpOcrV6WordBox(0, 0, "AA", (18, 8, 34, 32)),
            PpOcrV6WordBox(0, 1, "”", (45, 8, 51, 32)),
            PpOcrV6WordBox(0, 2, "B", (62, 8, 72, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 90, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 90, 40), "AA”B"),
    ]


def test_pure_latin_row_preserves_spaces_in_full_engcut_context():
    image = _image()
    _ink(image, (20, 10, 25, 30))      # A
    _ink(image, (50, 10, 55, 30))      # B
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="A B",
        bbox=(0, 0, 80, 40),
        words=(
            PpOcrV6WordBox(0, 0, "A", (18, 8, 28, 32)),
            PpOcrV6WordBox(0, 1, " ", (34, 8, 38, 32)),
            PpOcrV6WordBox(0, 2, "B", (48, 8, 58, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 80, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 80, 40), "A B"),
    ]


def test_light_text_on_dark_background_uses_the_same_mixed_route_contract():
    image = np.full((50, 120, 3), 70, dtype=np.uint8)
    image[10:30, 5:23] = 240       # 甲
    image[10:30, 36:43] = 240      # 2
    image[10:30, 46:53] = 240      # 0
    image[10:30, 56:63] = 240      # 2
    image[10:30, 66:73] = 240      # 4
    image[10:30, 90:108] = 240     # 乙
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲2024乙",
        bbox=(0, 0, 115, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (4, 8, 24, 32)),
            PpOcrV6WordBox(0, 1, "2024", (34, 8, 76, 32)),
            PpOcrV6WordBox(0, 2, "乙", (88, 8, 110, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 115, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((36, 10, 73, 30), "2024")]


def test_narrow_digit_wordbox_owns_its_nearest_complete_component():
    image = _image()
    _ink(image, (10, 8, 30, 32))
    _ink(image, (36, 10, 58, 30))
    _ink(image, (64, 8, 84, 32))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="第4期",
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "第", (8, 6, 42, 34)),
            PpOcrV6WordBox(0, 1, "4", (55, 6, 62, 34)),
            PpOcrV6WordBox(0, 2, "期", (62, 6, 88, 34)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, prepass_line.bbox)

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((36, 10, 58, 30), "4")]


def test_latin_leading_glyph_follows_material_overlap_not_neighbor_token_center():
    image = _image()
    _ink(image, (8, 8, 30, 32))        # preceding CJK token
    _ink(image, (40, 10, 70, 30))      # leading A: one-pixel CJK overlap
    _ink(image, (74, 10, 82, 30))      # remaining Latin token ink
    _ink(image, (94, 8, 116, 32))      # following CJK token
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="据Ab等",
        bbox=(0, 0, 125, 40),
        words=(
            PpOcrV6WordBox(0, 0, "据", (8, 6, 41, 34)),
            PpOcrV6WordBox(0, 1, "Ab", (40, 6, 86, 34)),
            PpOcrV6WordBox(0, 2, "等", (92, 6, 118, 34)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, prepass_line.bbox)

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((40, 10, 82, 30), "Ab")]


def test_horizontal_table_rule_cannot_widen_or_overlap_latin_masks():
    image = _image()
    _ink(image, (4, 8, 20, 28))
    _ink(image, (30, 8, 35, 28))
    _ink(image, (45, 8, 61, 28))
    _ink(image, (70, 8, 75, 28))
    _ink(image, (82, 8, 98, 28))
    _ink(image, (0, 34, 110, 36))       # table border in the same detector row
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲A乙B丙",
        bbox=(0, 0, 110, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (2, 6, 22, 30)),
            PpOcrV6WordBox(0, 1, "A", (28, 6, 38, 30)),
            PpOcrV6WordBox(0, 2, "乙", (43, 6, 63, 30)),
            PpOcrV6WordBox(0, 3, "B", (68, 6, 78, 30)),
            PpOcrV6WordBox(0, 4, "丙", (80, 6, 100, 30)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 110, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [
        ((30, 8, 35, 28), "A"),
        ((70, 8, 75, 28), "B"),
    ]


def test_latin_token_recovers_only_its_fragment_from_fused_punctuation_component():
    image = _image()
    _ink(image, (4, 10, 20, 30))
    _ink(image, (30, 10, 50, 30))       # fused left parenthesis + h
    _ink(image, (56, 10, 61, 30))
    _ink(image, (72, 10, 88, 30))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲(h)乙",
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (2, 8, 22, 32)),
            PpOcrV6WordBox(0, 1, "(", (28, 8, 42, 32)),
            PpOcrV6WordBox(0, 2, "h", (42, 8, 50, 32)),
            PpOcrV6WordBox(0, 3, ")", (54, 8, 63, 32)),
            PpOcrV6WordBox(0, 4, "乙", (70, 8, 90, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 100, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((42, 10, 50, 30), "h")]


def test_latin_token_recovers_trailing_glyph_fragment_fused_to_symbol():
    image = _image()
    _ink(image, (4, 10, 20, 30))       # preceding CJK
    _ink(image, (30, 10, 38, 30))      # P
    _ink(image, (40, 10, 48, 30))      # E
    _ink(image, (52, 10, 72, 30))      # fused slash + V
    _ink(image, (74, 10, 82, 30))      # C
    _ink(image, (94, 10, 110, 30))     # following CJK
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲PE/VC乙",
        bbox=(0, 0, 120, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (2, 8, 22, 32)),
            PpOcrV6WordBox(0, 1, "PE", (28, 8, 50, 32)),
            PpOcrV6WordBox(0, 2, "/", (50, 8, 62, 32)),
            PpOcrV6WordBox(0, 3, "VC", (62, 8, 84, 32)),
            PpOcrV6WordBox(0, 4, "乙", (92, 8, 112, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, prepass_line.bbox)

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [
        ((30, 10, 48, 30), "PE"),
        ((62, 10, 82, 30), "VC"),
    ]


def test_shifted_comma_reclaims_its_mark_without_taking_following_latin_body():
    image = _image()
    _ink(image, (4, 10, 20, 30))
    _ink(image, (27, 24, 31, 30))       # comma left of its proposal
    _ink(image, (36, 10, 52, 30))       # C body closer to comma than word center
    _ink(image, (56, 10, 66, 30))
    _ink(image, (70, 10, 80, 30))
    _ink(image, (84, 10, 94, 30))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲,China",
        bbox=(0, 0, 110, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (2, 8, 22, 32)),
            PpOcrV6WordBox(0, 1, ",", (31, 8, 39, 32)),
            PpOcrV6WordBox(0, 2, "China", (40, 8, 100, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 110, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((36, 10, 94, 30), "China")]


def test_shifted_multi_part_symbol_reclaims_component_from_adjacent_latin_token():
    image = _image()
    _ink(image, (4, 10, 20, 30))
    _ink(image, (30, 10, 35, 30))       # digit body
    _ink(image, (40, 9, 46, 15))        # upper percent dot, inside digit proposal
    _ink(image, (43, 17, 48, 22))       # percent slash, owned by symbol proposal
    _ink(image, (46, 24, 52, 30))       # lower percent dot; projection chain
    _ink(image, (68, 10, 84, 30))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="甲1%乙",
        bbox=(0, 0, 100, 40),
        words=(
            PpOcrV6WordBox(0, 0, "甲", (2, 8, 22, 32)),
            PpOcrV6WordBox(0, 1, "1", (28, 8, 43, 32)),
            PpOcrV6WordBox(0, 2, "%", (45, 8, 54, 32)),
            PpOcrV6WordBox(0, 3, "乙", (66, 8, 86, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 100, 40))

    assert result.issues == ()
    latin = [segment for segment in result.segments if segment.kind == "text_latin"]
    assert [(segment.bbox, segment.text) for segment in latin] == [((30, 10, 35, 30), "1")]


def test_pure_latin_row_keeps_superscript_in_full_engcut_context():
    image = _image()
    _ink(image, (20, 10, 42, 30))       # fused x + superscript 2
    _ink(image, (44, 10, 50, 30))       # following digit 2
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="x²2",
        bbox=(0, 0, 70, 40),
        words=(
            PpOcrV6WordBox(0, 0, "x", (20, 8, 29, 32)),
            PpOcrV6WordBox(0, 1, "²", (32, 8, 42, 32)),
            PpOcrV6WordBox(0, 2, "2", (44, 8, 52, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 70, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 70, 40), "x²2"),
    ]


def test_pure_latin_row_ignores_displaced_quote_word_boxes():
    image = _image()
    _ink(image, (20, 10, 25, 30))       # first Latin glyph
    _ink(image, (30, 10, 35, 30))       # final Latin glyph
    _ink(image, (45, 10, 49, 20))       # displaced left half of closing quote
    _ink(image, (56, 10, 60, 20))       # quote anchor inside its PP proposal
    _ink(image, (70, 10, 74, 20))       # following opening quote
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='AB” “',
        bbox=(0, 0, 90, 40),
        words=(
            PpOcrV6WordBox(0, 0, "AB", (18, 8, 40, 32)),
            PpOcrV6WordBox(0, 1, '” “', (52, 8, 80, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 90, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 90, 40), "AB” “"),
    ]


def test_pure_latin_row_does_not_split_symbol_only_gap():
    image = _image()
    _ink(image, (10, 10, 15, 30))
    _ink(image, (30, 10, 34, 20))
    _ink(image, (40, 10, 44, 20))
    _ink(image, (64, 10, 68, 20))
    _ink(image, (74, 10, 78, 20))
    _ink(image, (92, 10, 97, 30))
    prepass_line = PpOcrV6LineHint(
        index=0,
        text='A” “B',
        bbox=(0, 0, 110, 40),
        words=(
            PpOcrV6WordBox(0, 0, "A", (8, 8, 18, 32)),
            PpOcrV6WordBox(0, 1, '” “', (36, 8, 72, 32)),
            PpOcrV6WordBox(0, 2, "B", (90, 8, 100, 32)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, (0, 0, 110, 40))

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 110, 40), "A” “B"),
    ]


def test_pure_latin_row_keeps_shifted_question_mark_in_engcut_context():
    image = np.full((60, 220, 3), 255, dtype=np.uint8)
    for bbox in (
        (20, 16, 24, 48),
        (28, 14, 36, 37),
        (46, 17, 51, 39),
        (61, 10, 75, 36),
        (80, 9, 89, 43),
        (102, 13, 107, 31),
        (103, 42, 107, 48),
    ):
        _ink(image, bbox)
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="China? ",
        bbox=(0, 0, 200, 55),
        words=(
            PpOcrV6WordBox(0, 0, "China", (15, 2, 91, 52)),
            PpOcrV6WordBox(0, 1, "? ", (108, 2, 116, 52)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, prepass_line.bbox)

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 200, 55), "China? "),
    ]


def test_pure_latin_row_does_not_partition_around_question_mark():
    image = np.full((121, 1200, 3), 255, dtype=np.uint8)
    for bbox in (
        (683, 23, 730, 89),    # preceding n
        (735, 50, 754, 88),    # preceding i body
        (740, 26, 750, 36),    # preceding i dot
        (855, 50, 893, 89),    # China final a
        (902, 32, 934, 79),    # question body
        (912, 90, 922, 100),   # question dot
        (984, 25, 1009, 88),   # following I
        (1015, 50, 1043, 89),  # following s
    ):
        _ink(image, bbox)
    prepass_line = PpOcrV6LineHint(
        index=0,
        text="China? Is",
        bbox=(0, 0, 1200, 121),
        words=(
            PpOcrV6WordBox(0, 0, "China", (705, 0, 886, 121)),
            PpOcrV6WordBox(0, 1, "? ", (906, 0, 966, 121)),
            PpOcrV6WordBox(0, 2, "Is", (984, 0, 1080, 121)),
        ),
    )

    result = partition_charocr_text_region(image, prepass_line, prepass_line.bbox)

    assert result.issues == ()
    assert [(segment.kind, segment.bbox, segment.text) for segment in result.segments] == [
        ("text_latin", (0, 0, 1200, 121), "China? Is"),
    ]
