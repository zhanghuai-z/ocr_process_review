from scripts.render_120180_italic_word_fallback_candidates import (
    _classification,
    _is_candidate,
    _latin_letter_count,
    _passes_slant,
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


def test_classification_prioritizes_ownership_conflict_and_accepts_gate_hit():
    token = {
        "symbol_conflicts": [{"text": ":"}],
        "current_result": {"word_fallback": True},
    }
    assert _classification(token, None, "120180.tif")[0] == "OWNERSHIP CONFLICT"

    clean = {"symbol_conflicts": [], "current_result": {"word_fallback": False}}
    slant = {
        "source_name": "120180.tif",
        "structural_conflict": True,
        "slant": {"measurable": True, "slope": 0.25, "score_improvement": 0.1},
    }
    assert _classification(clean, slant, "120180.tif")[0] == "GATE HIT"
    assert _passes_slant({**slant, "structural_conflict": False}, "120180.tif") is True
    assert _classification(
        clean, {**slant, "structural_conflict": False}, "120180.tif"
    )[0] == "SLOPE RECALL"
    assert _latin_letter_count("A1中b") == 2
