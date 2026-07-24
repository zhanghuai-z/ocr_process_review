from scripts.render_120180_italic_word_fallback_candidates import (
    _is_candidate,
    _proposed_token,
)


def test_candidate_requires_source_right_slant_and_structural_conflict():
    record = {
        "source_name": "120180.tif",
        "structural_conflict": True,
        "slant": {"measurable": True, "slope": 0.25, "score_improvement": 0.1},
    }
    assert _is_candidate(record, "120180.tif") is True
    assert _is_candidate({**record, "structural_conflict": False}, "120180.tif") is False
    assert _is_candidate(record, "other.tif") is False


def test_proposed_token_uses_pp_text_and_route_bbox_as_diagnostic_word():
    token = {"text": "Finance", "route_bbox": [10, 20, 80, 40]}
    atom = _proposed_token(token)["current_result"]["atoms"][0]
    assert atom["text"] == "Finance"
    assert atom["bbox"] == [10, 20, 80, 40]
    assert atom["granularity"] == "word"
