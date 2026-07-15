from __future__ import annotations

import numpy as np
import pytest

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.geometry.foreground import analyze_foreground_components
from app.core.ppocr_line_geometry import (
    PhysicalLineNormalization,
    PhysicalTextRow,
    TextBlockLineGroup,
    normalize_physical_text_rows,
)


def _line_for_source_index(result, source_index: int):
    return result.line_for_source_index(source_index)


def _line(
    index: int,
    text: str,
    bbox: tuple[int, int, int, int],
) -> PpOcrV6LineHint:
    return PpOcrV6LineHint(
        index=index,
        text=text,
        bbox=bbox,
        words=(PpOcrV6WordBox(index, 0, text, bbox),),
    )


def test_co_baseline_fragments_in_one_layout_block_become_one_row():
    prefix = _line(4, "12", (5, 10, 15, 30))
    body = _line(3, "Austin", (25, 8, 90, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body, prefix)),),
        None,
    )

    assert _line_for_source_index(result, 3) == PpOcrV6LineHint(
        index=3,
        text="12Austin",
        bbox=(5, 8, 90, 32),
        words=(
            PpOcrV6WordBox(3, 0, "12", (5, 10, 15, 30)),
            PpOcrV6WordBox(3, 1, "Austin", (25, 8, 90, 32)),
        ),
    )
    assert _line_for_source_index(result, 4) is None


def test_typed_physical_normalization_exposes_merged_members_without_null_sentinel():
    prefix = _line(4, "12", (5, 10, 15, 30))
    body = _line(3, "Austin", (25, 8, 90, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body, prefix)),),
        None,
    )

    assert len(result.rows) == 1
    assert result.rows[0].source_indices == (4, 3)
    assert result.line_for_source_index(3) is not None
    assert result.line_for_source_index(4) is None
    assert result.row_for_source_index(4) is result.rows[0]


def test_physical_source_index_lookup_is_immutable_and_o_one():
    source_indices = [3, 4]
    row = PhysicalTextRow(_line(3, "Austin", (5, 8, 90, 32)), source_indices)
    result = PhysicalLineNormalization([row])

    source_indices.append(9)

    assert result.rows[0].source_indices == (3, 4)
    assert result.row_for_source_index(4) is row
    with pytest.raises(TypeError):
        result._rows_by_source_index[9] = row


def test_foreground_analysis_is_page_coordinate_evidence():
    image = np.full((30, 50, 3), 255, dtype=np.uint8)
    image[5:10, 7:12] = 0
    image[15:22, 30:38] = 0

    result = analyze_foreground_components(image, (5, 3, 45, 26))

    assert result.region_bbox == (5, 3, 45, 26)
    assert [component.bbox for component in result.components] == [
        (7, 5, 12, 10),
        (30, 15, 38, 22),
    ]


def test_overlapping_co_baseline_fragments_in_one_layout_block_become_one_row():
    left = _line(3, "第三节", (5, 8, 45, 32))
    right = _line(4, "民居", (40, 8, 95, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (left, right)),),
        None,
    )

    assert _line_for_source_index(result, 3) is not None
    assert _line_for_source_index(result, 3).text == "第三节民居"
    assert _line_for_source_index(result, 3).bbox == (5, 8, 95, 32)
    assert _line_for_source_index(result, 4) is None


def test_unclaimed_prefix_ink_extends_row_inside_its_layout_block():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[4:36, 6:15] = 0
    image[10:30, 30:90] = 0
    body = _line(3, "Austin", (25, 8, 95, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3) is not None
    assert _line_for_source_index(result, 3).bbox == (6, 4, 95, 36)
    assert _line_for_source_index(result, 3).text == "Austin"


def test_unclaimed_suffix_ink_extends_row_without_clipping_last_glyph():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[10:30, 30:90] = 0
    image[8:34, 96:104] = 0
    body = _line(3, "Austin", (25, 8, 95, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 110, 40), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 8, 104, 34)


def test_rotated_row_recovers_reading_axis_ends_and_cross_axis_ink():
    image = np.full((130, 70, 3), 255, dtype=np.uint8)
    image[20:100, 24:44] = 0
    image[104:116, 20:48] = 0
    body = PpOcrV6LineHint(
        index=3,
        text="单位人",
        bbox=(25, 25, 45, 103),
        words=(PpOcrV6WordBox(3, 0, "单位人", (25, 25, 45, 103)),),
        text_axis="vertical",
        orientation_angle=180,
    )

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (10, 10, 60, 120), (body,)),),
        image,
    )

    line = _line_for_source_index(result, 3)
    assert line.bbox == (20, 20, 48, 116)
    assert line.text_axis == "vertical"
    assert line.orientation_angle == 180


def test_blank_layout_margin_does_not_expand_row():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[10:30, 30:90] = 0
    body = _line(3, "Austin", (25, 8, 95, 32))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 10, 95, 30)


def test_row_height_follows_word_owned_ink_including_detached_dot():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[12:30, 30:38] = 0
    image[5:9, 32:36] = 0
    body = _line(3, "i", (25, 0, 45, 40))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 5, 45, 30)


def test_row_height_recovers_complete_owned_glyph_beyond_ppocr_box():
    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    image[8:34, 30:38] = 0
    body = _line(3, "I", (25, 12, 45, 30))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 50), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 8, 45, 34)


def test_row_height_may_recover_owned_glyph_beyond_layout_block():
    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    image[8:34, 30:38] = 0
    body = _line(3, "I", (25, 12, 45, 30))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (20, 10, 50, 31), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 8, 45, 34)


def test_row_height_recovers_connected_component_closure():
    image = np.full((70, 120, 3), 255, dtype=np.uint8)
    image[12:38, 30:38] = 0
    image[32:42, 40:48] = 0
    body = _line(3, "ab", (25, 12, 55, 30))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (20, 8, 60, 45), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 12, 55, 42)


def test_row_height_never_expands_from_page_spanning_component():
    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    image[0:55, 34:37] = 0
    body = _line(3, "I", (25, 15, 45, 35))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 60), (body,)),),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 15, 45, 35)


def test_co_baseline_fragments_in_different_layout_blocks_do_not_merge():
    left = _line(3, "left", (5, 8, 45, 32))
    right = _line(4, "right", (55, 8, 95, 32))

    result = normalize_physical_text_rows(
        (
            TextBlockLineGroup("block-1", (0, 0, 50, 40), (left,)),
            TextBlockLineGroup("block-2", (50, 0, 100, 40), (right,)),
        ),
        None,
    )

    assert [(row.line.index, row.line) for row in result.rows] == [(3, left), (4, right)]


def test_vertically_touching_neighbor_rows_do_not_merge():
    upper = _line(3, "upper", (5, 5, 45, 20))
    lower = _line(4, "lower", (55, 18, 95, 35))

    result = normalize_physical_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (upper, lower)),),
        None,
    )

    assert [(row.line.index, row.line) for row in result.rows] == [(3, upper), (4, lower)]


def test_structural_ink_is_not_recovered_as_unclaimed_prefix():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[10:30, 30:55] = 0
    image[2:38, 5:20] = 0
    body = _line(3, "text", (25, 8, 95, 32))

    result = normalize_physical_text_rows(
        (
            TextBlockLineGroup(
                "block-1",
                (0, 0, 100, 40),
                (body,),
                excluded_bboxes=((0, 0, 22, 40),),
            ),
        ),
        image,
    )

    assert _line_for_source_index(result, 3).bbox == (25, 10, 95, 30)
