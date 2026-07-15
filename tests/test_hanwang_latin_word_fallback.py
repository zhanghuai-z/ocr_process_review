import json

import numpy as np
import pytest

import app.engines.hanwang.micro_recblock as micro_module
from app.models.charocr_routing import (
    PpOcrLatinTokenObservation,
    RoutingLine,
    RoutingSegment,
)
from tests.charocr_native_route_fixture import run_micro_recblock_with_explicit_routes


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


def _eng20_grouped_payload_at(groups: list[tuple[int, str]]) -> dict:
    payload_groups = []
    for left, text in groups:
        chars = []
        for index, char in enumerate(text):
            char_left = left + index * 10
            chars.append({
                "codes": [ord(char)],
                "bbox": {
                    "left": char_left,
                    "top": 2,
                    "right": char_left + 8,
                    "bottom": 22,
                },
            })
        payload_groups.append({"chars": chars})
    return {"lines": [{"groups": payload_groups}]}


def test_geometry_reconciler_replaces_conflicting_native_boxes_with_token_atom():
    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    image[10:50, 20:60] = 0
    line = micro_module.LineResult(
        text="JJ甲",
        bbox=(0, 0, 120, 60),
        chars=[
            micro_module.CharResult("J", confidence=0.19, bbox=(20, 12, 50, 48)),
            micro_module.CharResult("J", confidence=0.19, bbox=(42, 10, 61, 50)),
            micro_module.CharResult("甲", confidence=0.95, bbox=(75, 10, 110, 52)),
        ],
    )
    stats = micro_module.RunStats()

    micro_module._reconcile_native_char_geometry(image, [line], stats)

    assert line.text == "JJ甲"
    assert [char.text for char in line.chars] == ["JJ", "甲"]
    assert line.chars[0].bbox == (20, 10, 60, 50)
    assert line.chars[0].source == "hanwang:geometry_reconciled"
    assert line.chars[0].bbox_granularity == "word"
    assert stats.geometry_conflict_groups == 1
    assert stats.geometry_token_atoms == 1


def test_native_character_without_bbox_is_not_given_the_line_bbox():
    raw = {
        "lines": [{
            "groups": [{
                "bbox": {"left": 0, "top": 0, "right": 80, "bottom": 30},
                "chars": [{"codes": [ord("A")], "scores": [20]}],
            }],
        }],
    }

    lines = micro_module._line_results_from_recog(
        raw,
        fallback_bbox=(0, 0, 80, 30),
        include_chars=True,
    )

    assert lines[0].chars[0].bbox is None
    assert lines[0].review_flags == ["hanwang_missing_char_geometry"]


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
        rows, stats = run_micro_recblock_with_explicit_routes(
            image,
            blocks,
            page_ocr_lines=[PageOcrLineHint(text="Urban Crisis", bbox=(0, 0, 220, 40))],
        )
    finally:
        micro_module.native_bridge.run_linecut_segimg = original_linecut
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert linecut_called is False
    assert stats.engcut_route_calls == 1
    assert len(rows) == 1
    assert rows[0].lines[0].text == "Urban Crisis"
    assert rows[0].lines[0].source == "hanwang:EngCut:latin_route"
    assert [char.text for char in rows[0].lines[0].chars] == list("Urban Crisis")


def test_engcut_overlapping_native_group_degrades_to_one_word_observation():
    chars = [
        micro_module.EngcutChar("o", bbox=(10, 5, 24, 30), group_index=0, char_index=0),
        micro_module.EngcutChar("f", bbox=(22, 4, 36, 30), group_index=0, char_index=1),
        micro_module.EngcutChar("A", bbox=(50, 4, 64, 30), group_index=1, char_index=0),
    ]

    text, results = micro_module._engcut_route_line_text_and_chars(chars)

    assert text == "of A"
    assert [(char.text, char.bbox, char.bbox_granularity) for char in results] == [
        ("of", (10, 4, 36, 30), "word"),
        (" ", None, "space"),
        ("A", (50, 4, 64, 30), "char"),
    ]
    assert results[0].source.endswith(":overlap_word")


def test_engcut_overlapping_group_uses_uniquely_bound_ppocr_word_observation():
    chars = [
        micro_module.EngcutChar("o", bbox=(10, 5, 24, 30), group_index=0, char_index=0),
        micro_module.EngcutChar("f", bbox=(22, 4, 36, 30), group_index=0, char_index=1),
    ]

    text, results = micro_module._engcut_route_line_text_and_chars(
        chars,
        ppocr_tokens=(PpOcrLatinTokenObservation("of", (14, 4, 32, 31)),),
    )

    assert text == "of"
    assert [(char.text, char.bbox, char.source, char.bbox_granularity) for char in results] == [
        ("of", (10, 4, 36, 31), "ppocrv6:latin_token_geometry_fallback", "word"),
    ]


def test_engcut_text_disagreement_uses_uniquely_bound_ppocr_word_observation():
    chars = [
        micro_module.EngcutChar(char, bbox=(10 + index * 10, 4, 19 + index * 10, 30), group_index=0, char_index=index)
        for index, char in enumerate("Unbalan,ced")
    ]

    text, results = micro_module._engcut_route_line_text_and_chars(
        chars,
        ppocr_tokens=(PpOcrLatinTokenObservation("Unbalanced", (10, 3, 130, 31)),),
    )

    assert text == "Unbalanced"
    assert len(results) == 1
    assert results[0].text == "Unbalanced"
    assert results[0].candidates == ["Unbalan,ced", "Unbalanced"]
    assert results[0].source == "ppocrv6:latin_token_geometry_fallback"


def test_engcut_cannot_reuse_one_ppocr_token_for_multiple_native_groups():
    chars = [
        micro_module.EngcutChar("A", bbox=(10, 4, 18, 30), group_index=0, char_index=0),
        micro_module.EngcutChar("B", bbox=(24, 4, 32, 30), group_index=1, char_index=0),
    ]

    text, results = micro_module._engcut_route_line_text_and_chars(
        chars,
        ppocr_tokens=(PpOcrLatinTokenObservation("AB", (8, 3, 34, 31)),),
    )

    assert text == "A B"
    assert all(char.source != "ppocrv6:latin_token_geometry_fallback" for char in results)


def test_masked_latin_line_keeps_only_latin_pixels_and_rebinds_groups():
    image = np.full((40, 180, 3), 255, dtype=np.uint8)
    image[8:30, 8:25] = 0
    image[8:30, 33:35] = 0
    image[8:30, 42:68] = 0
    image[8:30, 112:138] = 0
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=2,
        line_idx=4,
        bbox=(0, 0, 160, 36),
        segments=(
            micro_module._TextRoute(2, 4, 1, (35, 0, 80, 36), carved=True, kind="text_latin"),
            micro_module._TextRoute(2, 4, 3, (105, 0, 150, 36), carved=True, kind="text_latin"),
        ),
    )
    stats = micro_module.RunStats()
    captured: list[np.ndarray] = []
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = lambda crop, *, timeout=0: (
        captured.append(crop.copy()) or _eng20_grouped_payload_at([(42, "AB"), (112, "CD")])
    )
    try:
        results = micro_module._recognize_engcut_masked_line(
            image,
            route,
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert stats.engcut_route_calls == 1
    assert len(captured) == 1
    assert np.all(captured[0][10, 12] == 255)
    assert np.all(captured[0][10, 34] == 255)
    assert np.all(captured[0][10, 48] == 0)
    assert np.all(captured[0][10, 118] == 0)
    assert {key: result.text for key, result in results.items()} == {
        (2, 4, 1): "AB",
        (2, 4, 3): "CD",
    }
    assert all(result.bbox_source == "text_latin_masked_line_engcut" for result in results.values())


def test_masked_latin_line_audits_ppocr_word_geometry_fallback():
    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=0,
        line_idx=0,
        bbox=(0, 0, 100, 36),
        segments=(
            micro_module._TextRoute(
                0,
                0,
                0,
                (20, 0, 80, 36),
                kind="text_latin",
                ppocr_latin_tokens=(PpOcrLatinTokenObservation("Word", (24, 3, 68, 31)),),
            ),
        ),
    )
    stats = micro_module.RunStats()
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = (
        lambda _crop, *, timeout=0: _eng20_grouped_payload_at([(25, "W0rd")])
    )
    try:
        result = micro_module._recognize_engcut_masked_line(
            image,
            route,
            stats,
            timeout=1.0,
        )[(0, 0, 0)]
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert result.text == "Word"
    assert result.source.endswith("+ppocrv6_token")
    assert result.review_flags == ["latin_token_geometry_fallback"]
    assert stats.latin_token_geometry_fallbacks == 1


def test_engcut_masked_routes_exclude_linecut_owned_punctuation():
    line = RoutingLine(
        index=0,
        bbox=(0, 0, 100, 36),
        segments=(
            RoutingSegment(kind="text_latin", bbox=(30, 0, 54, 36), text="AB"),
            RoutingSegment(kind="text_other", bbox=(54, 0, 70, 36)),
        ),
    )

    routes = micro_module._engcut_masked_line_routes_from_lines(0, (line,))

    assert len(routes) == 1
    assert [(segment.kind, segment.bbox) for segment in routes[0].segments] == [
        ("text_latin", (30, 0, 54, 36)),
    ]


def test_masked_latin_line_uses_pp_text_only_when_one_segment_has_zero_native_output():
    image = np.full((40, 180, 3), 255, dtype=np.uint8)
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=2,
        line_idx=4,
        bbox=(0, 0, 160, 36),
        segments=(
            micro_module._TextRoute(
                2, 4, 1, (35, 0, 80, 36), carved=True, kind="text_latin",
                ppocr_latin_fallback_text="AB",
            ),
            micro_module._TextRoute(
                2, 4, 3, (105, 0, 150, 36), carved=True, kind="text_latin",
                ppocr_latin_fallback_text="s",
            ),
        ),
    )
    stats = micro_module.RunStats()
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = (
        lambda _crop, *, timeout=0: _eng20_grouped_payload_at([(42, "AB")])
    )
    try:
        results = micro_module._recognize_engcut_masked_line(
            image,
            route,
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert results[(2, 4, 1)].source == micro_module.LATIN_ENGCUT_ROUTE_SOURCE
    fallback = results[(2, 4, 3)]
    assert fallback.text == "s"
    assert fallback.source == micro_module.LATIN_EMPTY_NATIVE_FALLBACK_SOURCE
    assert fallback.bbox == (105, 0, 150, 36)
    assert fallback.chars[0].bbox_granularity == "word"
    assert fallback.review_flags == [micro_module.LATIN_EMPTY_NATIVE_FALLBACK_FLAG]
    assert stats.latin_empty_native_fallbacks == 1


def test_masked_latin_line_rejects_unbound_native_group():
    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=0,
        line_idx=0,
        bbox=(0, 0, 100, 36),
        segments=(
            micro_module._TextRoute(0, 0, 0, (20, 0, 50, 36), kind="text_latin"),
        ),
    )
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = (
        lambda _crop, *, timeout=0: _eng20_grouped_payload_at([(70, "X")])
    )
    try:
        with pytest.raises(RuntimeError, match="cannot be uniquely rebound"):
            micro_module._recognize_engcut_masked_line(
                image,
                route,
                micro_module.RunStats(),
                timeout=1.0,
            )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20


def test_masked_engcut_group_assigns_each_character_to_its_latin_route():
    segments = (
        micro_module._TextRoute(0, 0, 0, (20, 0, 50, 36), kind="text_latin"),
        micro_module._TextRoute(0, 0, 2, (50, 0, 80, 36), kind="text_latin"),
    )
    left = micro_module.EngcutChar("A", bbox=(25, 4, 33, 24))
    right = micro_module.EngcutChar("B", bbox=(60, 4, 68, 24))

    assert micro_module._engcut_segment_for_char(left, segments) is segments[0]
    assert micro_module._engcut_segment_for_char(right, segments) is segments[1]


def test_masked_latin_lines_use_bounded_parallel_native_calls():
    import threading
    import time

    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    routes = [
        micro_module._EngCutMaskedLineRoute(
            block_idx=0,
            line_idx=line_idx,
            bbox=(0, 0, 100, 36),
            segments=(
                micro_module._TextRoute(
                    0,
                    line_idx,
                    0,
                    (20, 0, 50, 36),
                    kind="text_latin",
                ),
            ),
        )
        for line_idx in range(4)
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_eng20(_crop, *, timeout=0):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return _eng20_grouped_payload_at([(25, "A")])

    stats = micro_module.RunStats()
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = fake_eng20
    try:
        results = micro_module._recognize_engcut_masked_lines(
            image,
            routes,
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert max_active > 1
    assert max_active <= micro_module._MAX_ENGCUT_LINES_PER_PAGE
    assert stats.engcut_route_calls == 4
    assert [route.line_idx for route, _route_results in results] == [0, 1, 2, 3]


def test_masked_latin_parallel_runner_aggregates_token_fallback_stats():
    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    routes = [
        micro_module._EngCutMaskedLineRoute(
            block_idx=0,
            line_idx=line_idx,
            bbox=(0, 0, 100, 36),
            segments=(
                micro_module._TextRoute(
                    0,
                    line_idx,
                    0,
                    (20, 0, 80, 36),
                    kind="text_latin",
                    ppocr_latin_tokens=(
                        PpOcrLatinTokenObservation("Word", (20, 2, 70, 30)),
                    ),
                ),
            ),
        )
        for line_idx in range(2)
    ]
    stats = micro_module.RunStats()
    original_eng20 = micro_module.native_bridge.run_eng20_recogline
    micro_module.native_bridge.run_eng20_recogline = (
        lambda _crop, *, timeout=0: _eng20_grouped_payload_at([(22, "W0rd")])
    )
    try:
        results = micro_module._recognize_engcut_masked_lines(
            image,
            routes,
            stats,
            timeout=1.0,
        )
    finally:
        micro_module.native_bridge.run_eng20_recogline = original_eng20

    assert len(results) == 2
    assert stats.latin_token_geometry_fallbacks == 2


def test_masked_latin_line_hook_writes_actual_engcut_input(tmp_path, monkeypatch):
    monkeypatch.setenv("HANWANG_MICRO_RECBLOCK_HOOK_DIR", str(tmp_path))
    route = micro_module._EngCutMaskedLineRoute(
        block_idx=1,
        line_idx=2,
        bbox=(10, 20, 70, 50),
        segments=(
            micro_module._TextRoute(1, 2, 1, (25, 20, 50, 50), kind="text_latin"),
        ),
    )
    crop = np.full((30, 60, 3), 255, dtype=np.uint8)
    micro_module._write_masked_engcut_line_hook(crop, route, offset_x=10, offset_y=20)

    output_dir = tmp_path / "engcut_masked_inputs"
    images = list(output_dir.glob("*.png"))
    metadata = list(output_dir.glob("*.json"))
    assert len(images) == 1
    assert len(metadata) == 1
    payload = json.loads(metadata[0].read_text(encoding="utf-8"))
    assert payload["crop_bbox"] == [10, 20, 70, 50]
    assert payload["engcut_segments"] == [{"route_key": [1, 2, 1], "bbox": [25, 20, 50, 50]}]
