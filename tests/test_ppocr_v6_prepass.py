from __future__ import annotations

import json

import pytest

from app.adapters.paddle import (
    PPOCR_V6_MODEL,
    PpOcrV6LineHint,
    PpOcrV6PrepassClient,
    PpOcrV6RoutingRequestProfile,
    PpOcrV6WordBox,
    build_ppocr_v6_routing_request_profile,
    normalize_ppocr_v6_prepass_result,
    parse_ppocr_v6_prepass_jsonl,
)


def _result(*, words: list[str] | None = None, word_boxes: list[list[int]] | None = None) -> dict:
    return {
        "prunedResult": {
            "rec_texts": ["Urban 2026"],
            "rec_boxes": [[10, 20, 210, 60]],
            "text_word": [words if words is not None else ["Urban", "2026"]],
            "text_word_boxes": [word_boxes if word_boxes is not None else [[10, 20, 100, 60], [120, 20, 210, 60]]],
        }
    }


def test_ppocr_v6_routing_profile_keeps_features_without_geometry_overrides():
    options = build_ppocr_v6_routing_request_profile().optional_payload()

    assert options == {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
        "returnWordBox": True,
        "textDetBoxThresh": 0.4,
        "textRecScoreThresh": 0.0,
    }


def test_ppocr_v6_request_profile_owns_model_and_optional_payload():
    profile = build_ppocr_v6_routing_request_profile()

    assert isinstance(profile, PpOcrV6RoutingRequestProfile)
    assert profile.model == PPOCR_V6_MODEL
    assert profile.optional_payload()["returnWordBox"] is True


def test_parse_ppocr_v6_prepass_keeps_lines_and_word_boxes_outside_proof_model():
    artifact = normalize_ppocr_v6_prepass_result(
        _result(),
        page_uid="page-1",
        run_id="job-1",
        width=300,
        height=100,
    )

    assert artifact.page_uid == "page-1"
    assert artifact.run_id == "job-1"
    assert [(line.index, line.text, line.bbox) for line in artifact.lines] == [
        (0, "Urban 2026", (10, 20, 210, 60)),
    ]
    assert [(word.text, word.bbox) for word in artifact.lines[0].words] == [
        ("Urban", (10, 20, 100, 60)),
        ("2026", (120, 20, 210, 60)),
    ]


def test_external_observation_normalizer_is_the_parse_result_boundary():
    parsed = normalize_ppocr_v6_prepass_result(
        _result(),
        page_uid="page-1",
        run_id="job-1",
        width=300,
        height=100,
    )

    assert isinstance(parsed.lines, tuple)
    assert parsed.lines[0].words[0].line_index == parsed.lines[0].index


def test_ppocr_observation_bboxes_copy_external_lists():
    line_bbox = [10, 20, 210, 60]
    word_bbox = [10, 20, 100, 60]
    word = PpOcrV6WordBox(0, 0, "Urban", word_bbox)
    line = PpOcrV6LineHint(0, "Urban", line_bbox, [word])

    line_bbox[0] = 99
    word_bbox[0] = 99

    assert line.bbox == (10, 20, 210, 60)
    assert line.words == (word,)
    assert word.bbox == (10, 20, 100, 60)


def test_parse_ppocr_v6_prepass_rejects_word_count_mismatch():
    with pytest.raises(ValueError, match="mismatched text_word/text_word_boxes"):
        normalize_ppocr_v6_prepass_result(
            _result(words=["Urban", "2026"], word_boxes=[[10, 20, 100, 60]]),
            page_uid="page-1",
        )


def test_parse_ppocr_v6_prepass_rejects_word_text_stream_mismatch():
    with pytest.raises(ValueError, match="word tokens do not reproduce"):
        normalize_ppocr_v6_prepass_result(
            _result(words=["Urban", "2025"]),
            page_uid="page-1",
        )


def test_parse_ppocr_v6_prepass_jsonl_rejects_multi_page_result_for_one_page_input():
    jsonl = "\n".join(
        json.dumps({"result": {"ocrResults": [_result()]}}, ensure_ascii=False)
        for _ in range(2)
    )

    with pytest.raises(ValueError, match="expected one page result"):
        parse_ppocr_v6_prepass_jsonl(jsonl, page_uid="page-1")


def test_ppocr_v6_client_submits_explicit_model_and_word_box_payload():
    class FakeTransport:
        def __init__(self):
            self.kwargs = None

        def submit_image_bytes(self, image_bytes, **kwargs):
            self.kwargs = {"image_bytes": image_bytes, **kwargs}
            return "job-1"

        def wait_for_result_json_url(self, job_id):
            assert job_id == "job-1"
            return "https://example.test/result.jsonl", {}

        def download_jsonl(self, json_url):
            assert json_url == "https://example.test/result.jsonl"
            return json.dumps({"result": {"ocrResults": [_result()]}}, ensure_ascii=False)

    transport = FakeTransport()
    artifact = PpOcrV6PrepassClient(transport).analyze_page_bytes(
        b"png-bytes",
        page_uid="page-1",
        batch_id="batch-1",
        filename="page-1.png",
    )

    assert artifact.run_id == "job-1"
    assert transport.kwargs["model"] == PPOCR_V6_MODEL
    assert transport.kwargs["optional_payload"]["returnWordBox"] is True
    assert transport.kwargs["batch_id"] == "batch-1"
