from __future__ import annotations

import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.core.ppocr_line_geometry import TextBlockLineGroup, derive_complete_text_rows


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

    result = derive_complete_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body, prefix)),),
        None,
    )

    assert result[3] == PpOcrV6LineHint(
        index=3,
        text="12Austin",
        bbox=(5, 8, 90, 32),
        words=(
            PpOcrV6WordBox(3, 0, "12", (5, 10, 15, 30)),
            PpOcrV6WordBox(3, 1, "Austin", (25, 8, 90, 32)),
        ),
    )
    assert result[4] is None


def test_unclaimed_prefix_ink_extends_row_inside_its_layout_block():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[12:28, 6:15] = 0
    image[10:30, 30:90] = 0
    body = _line(3, "Austin", (25, 8, 95, 32))

    result = derive_complete_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body,)),),
        image,
    )

    assert result[3] is not None
    assert result[3].bbox == (6, 8, 95, 32)
    assert result[3].text == "Austin"


def test_blank_layout_margin_does_not_expand_row():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[10:30, 30:90] = 0
    body = _line(3, "Austin", (25, 8, 95, 32))

    result = derive_complete_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (body,)),),
        image,
    )

    assert result[3] == body


def test_co_baseline_fragments_in_different_layout_blocks_do_not_merge():
    left = _line(3, "left", (5, 8, 45, 32))
    right = _line(4, "right", (55, 8, 95, 32))

    result = derive_complete_text_rows(
        (
            TextBlockLineGroup("block-1", (0, 0, 50, 40), (left,)),
            TextBlockLineGroup("block-2", (50, 0, 100, 40), (right,)),
        ),
        None,
    )

    assert result == {3: left, 4: right}


def test_vertically_touching_neighbor_rows_do_not_merge():
    upper = _line(3, "upper", (5, 5, 45, 20))
    lower = _line(4, "lower", (55, 18, 95, 35))

    result = derive_complete_text_rows(
        (TextBlockLineGroup("block-1", (0, 0, 100, 40), (upper, lower)),),
        None,
    )

    assert result == {3: upper, 4: lower}


def test_structural_ink_is_not_recovered_as_unclaimed_prefix():
    image = np.full((50, 120, 3), 255, dtype=np.uint8)
    image[10:30, 30:55] = 0
    image[2:38, 5:20] = 0
    body = _line(3, "text", (25, 8, 95, 32))

    result = derive_complete_text_rows(
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

    assert result[3].bbox == body.bbox
