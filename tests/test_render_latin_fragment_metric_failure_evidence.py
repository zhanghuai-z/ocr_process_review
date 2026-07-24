from scripts.render_latin_fragment_metric_failure_evidence import (
    _known_cohorts,
    _production_word_quality,
    _strict_intersects,
    _v3_candidates,
)


def _record(label: str, *, area: float = 0.0, fallback: bool = False):
    return {
        "known_label": label,
        "current_word_fallback": fallback,
        "source_name": "page.tif",
        "fragment": {
            "measurable": True,
            "fragment_char_ratio": 0.2 if area else 0.0,
            "fragment_area_ratio": area,
        },
    }


def test_known_cohorts_separate_targets_and_controls():
    target = _record("target_bad_geometry")
    control = _record("control_usable_geometry")
    targets, controls = _known_cohorts([control, target, _record("unlabelled")])
    assert targets == [target]
    assert controls == [control]


def test_v3_candidates_exclude_zero_fragment_targets_and_rank_fallback_first():
    missed_target = _record("target_bad_geometry")
    fallback = _record("unlabelled", area=0.02, fallback=True)
    other = _record("unlabelled", area=0.03)

    selected = _v3_candidates([missed_target, other, fallback])

    assert selected == [fallback, other]


def test_production_word_quality_reports_route_and_other_atom_overlap():
    record = {
        "source_name": "page.tif",
        "text": "pj",
        "route_bbox": [10, 10, 30, 20],
        "owned_components": [{"bbox": [11, 11, 29, 19]}],
    }
    token = {
        "current_result": {
            "atoms": [
                {"text": "pj", "bbox": [10, 10, 30, 20], "granularity": "word"},
                {"text": "o", "bbox": [28, 10, 34, 20], "granularity": "char"},
            ]
        }
    }

    quality = _production_word_quality(record, token)

    assert quality["word_equals_route"] is True
    assert quality["other_atom_overlap_count"] == 1
    assert quality["owned_component_outside_count"] == 0
    assert _strict_intersects([10, 10, 30, 20], [30, 10, 40, 20]) is False
