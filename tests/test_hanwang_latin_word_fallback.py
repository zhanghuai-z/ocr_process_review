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


def _eng20_grouped_payload(groups: list[str]) -> dict:
    payload_groups = []
    cursor = 0
    for text in groups:
        chars = []
        for idx, char in enumerate(text):
            left = cursor + idx * 10
            chars.append({
                "codes": [ord(char)],
                "bbox": {
                    "left": left,
                    "top": 2,
                    "right": left + 8,
                    "bottom": 22,
                },
            })
        cursor += len(text) * 10 + 12
        payload_groups.append({"chars": chars})
    return {"lines": [{"groups": payload_groups}]}


def _linecut_payload(text: str) -> dict:
    chars = []
    for idx, char in enumerate(text):
        chars.append({
            "codes": [ord(char)],
            "scores": [20],
            "bbox": {
                "left": idx * 8 + 2,
                "top": 3,
                "right": idx * 8 + 12,
                "bottom": 27,
            },
        })
    return {"lines": [{"groups": [{"bbox": {"left": 0, "top": 0, "right": 50, "bottom": 30}, "chars": chars}]}]}


def _percent_fragment_line() -> micro_module.LineResult:
    chars = [
        micro_module.CharResult("约", confidence=0.90, bbox=(0, 0, 18, 30)),
        micro_module.CharResult("2", confidence=0.80, bbox=(22, 0, 34, 28)),
        micro_module.CharResult("0", confidence=0.80, bbox=(36, 0, 48, 28)),
        micro_module.CharResult("0", confidence=0.19, bbox=(50, 0, 72, 30), candidates=["0", "叼"]),
        micro_module.CharResult("/", confidence=0.19, bbox=(56, 0, 80, 31), candidates=["/", "驼"]),
        micro_module.CharResult("0", confidence=0.19, bbox=(66, 3, 86, 30), candidates=["0", "勿"]),
    ]
    return micro_module.LineResult(
        text="约200/0",
        bbox=(0, 0, 100, 40),
        confidence=0.5,
        chars=chars,
    )


def _overlapped_digit_line(text: str = "01") -> micro_module.LineResult:
    chars = [
        micro_module.CharResult(text[0], confidence=0.19, bbox=(0, 0, 20, 30), candidates=[text[0]]),
        micro_module.CharResult(text[1], confidence=0.19, bbox=(8, 0, 28, 30), candidates=[text[1]]),
    ]
    return micro_module.LineResult(
        text=text,
        bbox=(0, 0, 40, 36),
        confidence=0.19,
        chars=chars,
    )


def test_overlap_merge_recrop_replaces_percent_fragment():
    line = _percent_fragment_line()
    stats = micro_module.RunStats()

    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_recog = lambda image_bgr, **_kwargs: _linecut_payload("%")
    try:
        micro_module._refine_overlap_fragments_with_recrop(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_linecut_recog = original_recog

    assert line.text == "约20%"
    assert [char.text for char in line.chars] == ["约", "2", "0", "%"]
    assert line.chars[-1].source == "hanwang:overlap_merge_recrop"
    assert line.chars[-1].bbox_granularity == "char"
    assert stats.overlap_merge_clusters == 1
    assert stats.overlap_merge_probe_calls == 1
    assert stats.overlap_merge_replacements == 1


def test_overlap_merge_does_not_force_percent_when_recrop_keeps_fragment():
    line = _percent_fragment_line()
    stats = micro_module.RunStats()

    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_recog = lambda image_bgr, **_kwargs: _linecut_payload("0/0")
    try:
        micro_module._refine_overlap_fragments_with_recrop(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_linecut_recog = original_recog

    assert line.text == "约200/0"
    assert [char.text for char in line.chars] == ["约", "2", "0", "0", "/", "0"]
    assert stats.overlap_merge_clusters == 1
    assert stats.overlap_merge_probe_calls == 1
    assert stats.overlap_merge_replacements == 0


def test_overlap_merge_does_not_rewrite_low_conf_digit_string_as_percent():
    line = _overlapped_digit_line("01")
    stats = micro_module.RunStats()

    original_recog = micro_module.native_bridge.run_linecut_recog
    micro_module.native_bridge.run_linecut_recog = lambda image_bgr, **_kwargs: _linecut_payload("%")
    try:
        micro_module._refine_overlap_fragments_with_recrop(
            np.zeros((60, 120, 3), dtype=np.uint8),
            [line],
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_linecut_recog = original_recog

    assert line.text == "01"
    assert [char.text for char in line.chars] == ["0", "1"]
    assert stats.overlap_merge_clusters == 1
    assert stats.overlap_merge_probe_calls == 1
    assert stats.overlap_merge_replacements == 0


def test_text_latin_route_uses_engcut_without_linecut():
    from app.core.paddle_line_routing import PageOcrLineHint

    image = np.full((80, 240, 3), 255, dtype=np.uint8)
    blocks = [{
        "block_label": "text",
        "block_bbox": [0, 0, 220, 50],
        "block_content": "Urban Crisis",
    }]
    linecut_called = False

    def fake_linecut(*_args, **_kwargs):
        nonlocal linecut_called
        linecut_called = True
        raise AssertionError("LineCut should not receive text_latin routes")

    original_linecut = micro_module.native_bridge.run_linecut_segimg
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_linecut_segimg = fake_linecut
    micro_module.native_bridge.run_eng20_recogline = (
        lambda image_bgr, *, timeout=0: _eng20_grouped_payload(["Urban", "Crisis"])
    )
    try:
        rows, stats = micro_module.run_micro_recblock(
            image,
            blocks,
            page_ocr_lines=[PageOcrLineHint(text="Urban Crisis", bbox=(0, 0, 220, 40))],
        )
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_linecut
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert linecut_called is False
    assert stats.latin_engcut_probe_calls == 1
    assert len(rows) == 1
    assert rows[0].lines[0].text == "Urban Crisis"
    assert rows[0].lines[0].source == "hanwang:EngCut:latin_route"
    assert [char.text for char in rows[0].lines[0].chars] == list("Urban Crisis")


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
