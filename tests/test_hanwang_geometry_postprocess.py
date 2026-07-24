from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from app.engines.hanwang import micro_recblock as micro_module
from app.engines.hanwang.engcut_payload import EngcutChar
from app.engines.hanwang.geometry_postprocess import (
    LineAtomGeometry,
    conservative_cjk_bbox_cleanup,
    is_latin_right_slant_fallback,
    measure_latin_right_slant,
)
from app.models.charocr_routing import (
    PpOcrLatinTokenObservation,
    PpOcrSymbolObservation,
    RoutingLine,
    RoutingSegment,
)


def _right_slanted_strokes() -> np.ndarray:
    image = np.full((60, 160, 3), 255, np.uint8)
    for base in (20, 45, 70, 95, 120):
        for y in range(10, 50):
            x = round(base + 0.20 * (49 - y))
            cv2.rectangle(image, (x, y), (x + 3, y), (0, 0, 0), -1)
    return image


def test_latin_right_slant_measurement_selects_synthetic_italic_ink():
    measurement = measure_latin_right_slant(
        _right_slanted_strokes(),
        (10, 8, 145, 52),
    )

    assert measurement.measurable
    assert measurement.slope >= 0.12
    assert measurement.score_improvement >= 0.05
    assert is_latin_right_slant_fallback(measurement)


def test_conservative_cjk_cleanup_tightens_only_inside_clear_seams():
    image = np.full((32, 52, 3), 255, np.uint8)
    image[8:23, 8:17] = 0
    image[7:24, 29:38] = 0
    atoms = (
        LineAtomGeometry(0, "中", (5, 5, 25, 26), "linecut", "char"),
        LineAtomGeometry(1, "文", (25, 5, 45, 26), "linecut", "char"),
    )

    proposals = conservative_cjk_bbox_cleanup(
        image,
        (4, 4, 46, 27),
        atoms,
        linecut_source="linecut",
    )

    assert proposals == {0: (8, 8, 17, 23), 1: (29, 7, 38, 24)}


def test_cjk_production_adapter_infers_native_char_granularity_before_cleanup():
    image = np.full((32, 52, 3), 255, np.uint8)
    image[8:23, 8:17] = 0
    image[7:24, 29:38] = 0
    line = micro_module._NativeLineResult(
        text="中文",
        bbox=(4, 4, 46, 27),
        chars=[
            micro_module._NativeAtomResult(text="中", bbox=(5, 5, 25, 26)),
            micro_module._NativeAtomResult(text="文", bbox=(25, 5, 45, 26)),
        ],
    )
    route = RoutingLine(
        index=0,
        bbox=line.bbox,
        segments=(RoutingSegment(kind="text_other", bbox=line.bbox),),
    )
    stats = micro_module.RunStats()

    result = micro_module._postprocess_cjk_line_geometry(
        image, route, [line], stats
    )[0]

    assert [atom.bbox for atom in result.chars] == [(8, 8, 17, 23), (29, 7, 38, 24)]
    assert all(
        atom.source == micro_module.LINECUT_CJK_CLEANUP_SOURCE
        for atom in result.chars
    )
    assert [atom.external_candidates[-1].bbox for atom in result.chars] == [
        (5, 5, 25, 26),
        (25, 5, 45, 26),
    ]
    assert result.review_flags == [micro_module.LINECUT_CJK_CLEANUP_FLAG]
    assert stats.linecut_cjk_bbox_cleanups == 2


def test_italic_postcheck_uses_existing_unique_pp_word_fallback_contract():
    token = PpOcrLatinTokenObservation(text="Finance", bbox=(10, 10, 80, 30))
    chars = [
        EngcutChar(text=char, bbox=(10 + index * 8, 10, 18 + index * 8, 30))
        for index, char in enumerate("Flnance")
    ]

    text, atoms, disagreement = micro_module._engcut_route_line_text_and_chars(
        chars,
        ppocr_tokens=(token,),
        foreground_word_bbox=(9, 9, 82, 31),
        italic_fallback_tokens=frozenset((token,)),
    )

    assert text == "Finance"
    assert len(atoms) == 1
    assert atoms[0].bbox == (9, 9, 82, 31)
    assert atoms[0].source == micro_module.PPOCR_LATIN_ITALIC_WORD_FALLBACK_SOURCE
    assert atoms[0].external_candidates[0].text == "Flnance"
    assert disagreement


def test_italic_postcheck_distinguishes_recognition_noise_from_owned_symbol():
    token = PpOcrLatinTokenObservation(text="Unbalanced", bbox=(10, 8, 145, 52))
    segment = micro_module._TextRoute(
        block_idx=0,
        line_idx=0,
        segment_idx=0,
        bbox=(10, 8, 145, 52),
        kind="text_latin",
        ppocr_latin_tokens=(token,),
    )
    chars = [EngcutChar(text=char, bbox=(20, 10, 24, 50)) for char in "Unbalan,ced"]
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=0,
        line_idx=0,
        bbox=(8, 6, 150, 54),
        segments=(segment,),
    )

    assert micro_module._italic_word_fallback_tokens(
        _right_slanted_strokes(), route, segment, chars
    ) == frozenset((token,))

    conflicted = replace(
        route,
        ppocr_symbol_observations=(PpOcrSymbolObservation(
            text=",",
            bbox=(138, 40, 142, 48),
            proposal_bbox=(136, 8, 148, 52),
        ),),
    )
    assert not micro_module._italic_word_fallback_tokens(
        _right_slanted_strokes(), conflicted, segment, chars
    )
