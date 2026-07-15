from __future__ import annotations

import json

import pytest

from app.adapters.paddle.ppocr_v6_prepass import (
    PPOCR_V6_MODEL,
    PpOcrV6PrepassClient,
    build_ppocr_v6_prepass_options,
    parse_ppocr_v6_prepass_jsonl,
    parse_ppocr_v6_prepass_result,
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


def test_ppocr_v6_prepass_options_keep_features_without_geometry_overrides():
    options = build_ppocr_v6_prepass_options()

    assert options == {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
        "returnWordBox": True,
        "textDetBoxThresh": 0.4,
        "textRecScoreThresh": 0.0,
    }


def test_parse_ppocr_v6_prepass_keeps_lines_and_word_boxes_outside_proof_model():
    artifact = parse_ppocr_v6_prepass_result(
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


def test_parse_ppocr_v6_prepass_rejects_word_count_mismatch():
    with pytest.raises(ValueError, match="mismatched text_word/text_word_boxes"):
        parse_ppocr_v6_prepass_result(
            _result(words=["Urban", "2026"], word_boxes=[[10, 20, 100, 60]]),
            page_uid="page-1",
        )


def test_parse_ppocr_v6_prepass_rejects_word_text_stream_mismatch():
    with pytest.raises(ValueError, match="word tokens do not reproduce"):
        parse_ppocr_v6_prepass_result(
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
