import numpy as np

import app.engines.hanwang.micro_recblock as micro_module


def _line(text: str) -> micro_module.LineResult:
    return micro_module.LineResult(
        text=text,
        bbox=(0, 0, 160, 40),
        chars=[
            micro_module.CharResult(
                text=char,
                bbox=(idx * 14, 0, idx * 14 + 10, 24),
            )
            for idx, char in enumerate(text)
        ],
    )


def _eng20_payload(text: str, *, overlap_of: bool = False) -> dict:
    chars = []
    for idx, char in enumerate(text):
        left = idx * 10
        right = left + 8
        if overlap_of and char == "f":
            left = 5
            right = 18
        chars.append({
            "codes": [ord(char)],
            "bbox": {
                "left": left,
                "top": 2,
                "right": right,
                "bottom": 22,
            },
        })
    return {"lines": [{"groups": [{"chars": chars}]}]}


def test_latin_engcut_overlapped_exact_word_becomes_word_granularity():
    line = _line("of")
    stats = micro_module.RunStats()

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = lambda image_bgr, *, timeout=0: _eng20_payload("of", overlap_of=True)
    try:
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((50, 180, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert line.text == "of"
    assert len(line.chars) == 1
    assert line.chars[0].text == "of"
    assert line.chars[0].source == "hanwang:EngCut:latin_word_fallback"
    assert line.chars[0].bbox_granularity == "word"
    assert line.chars[0].bbox == (0, 2, 18, 22)
    assert stats.latin_engcut_exact_tokens == 0
    assert stats.latin_engcut_word_tokens == 1
    assert stats.latin_engcut_review_tokens == 0


def test_latin_engcut_repeated_token_updates_matching_occurrence():
    line = _line("of of")
    stats = micro_module.RunStats()

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = lambda image_bgr, *, timeout=0: _eng20_payload("of of")
    try:
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((50, 180, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert line.text == "of of"
    assert [char.text for char in line.chars] == list("of of")
    assert line.chars[0].bbox == (0, 2, 8, 22)
    assert line.chars[1].bbox == (10, 2, 18, 22)
    assert line.chars[3].bbox == (30, 2, 38, 22)
    assert line.chars[4].bbox == (40, 2, 48, 22)
    assert [char.source for char in line.chars] == [
        "hanwang:EngCut:latin_exact",
        "hanwang:EngCut:latin_exact",
        "hanwang:micro_recblock",
        "hanwang:EngCut:latin_exact",
        "hanwang:EngCut:latin_exact",
    ]
    assert stats.latin_engcut_exact_tokens == 2


def test_latin_engcut_internal_noise_word_becomes_word_granularity_from_block_text():
    line = _line("Second")
    stats = micro_module.RunStats()

    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = lambda image_bgr, *, timeout=0: _eng20_payload("Secon.d")
    try:
        micro_module._enhance_lines_with_latin_engcut(
            np.zeros((50, 180, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
            block_text="Second",
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert line.text == "Second"
    assert len(line.chars) == 1
    assert line.chars[0].text == "Second"
    assert line.chars[0].source == "hanwang:EngCut:latin_word_fallback"
    assert line.chars[0].bbox_granularity == "word"
    assert line.chars[0].bbox == (0, 2, 68, 22)
    assert stats.latin_engcut_exact_tokens == 0
    assert stats.latin_engcut_word_tokens == 1
    assert stats.latin_engcut_review_tokens == 0
