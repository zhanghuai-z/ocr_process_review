from app.core.latin_span_recovery import (
    LATIN_ENGCUT_EXACT_STATUS,
    LATIN_ENGCUT_NOT_FOUND_STATUS,
    bind_latin_tokens_to_engcut_chars,
    engcut_chars_from_payload,
    latin_token_spans,
)


def _payload_for_text(text: str) -> dict:
    return {
        "lines": [
            {
                "groups": [
                    {
                        "chars": [
                            {
                                "codes": [ord(ch)],
                                "bbox": {
                                    "left": idx * 10,
                                    "top": 0,
                                    "right": idx * 10 + 8,
                                    "bottom": 12,
                                },
                            }
                            for idx, ch in enumerate(text)
                        ]
                    }
                ]
            }
        ]
    }


def test_latin_token_spans_skip_formula_text():
    tokens = latin_token_spans("模型 $ Incentive_c $ uses PE/VC and Share")

    assert [token.text for token in tokens] == ["uses", "PE/VC", "and", "Share"]


def test_engcut_payload_binds_latin_token_exact_with_page_noise():
    chars = engcut_chars_from_payload(_payload_for_text("~PE/VC~"))
    bindings = bind_latin_tokens_to_engcut_chars("甲PE/VC乙", chars)

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.status == LATIN_ENGCUT_EXACT_STATUS
    assert binding.token.text == "PE/VC"
    assert binding.engcut_start == 1
    assert binding.engcut_end == 6
    assert binding.bbox == (10, 0, 58, 12)
    assert binding.char_bboxes == (
        (10, 0, 18, 12),
        (20, 0, 28, 12),
        (30, 0, 38, 12),
        (40, 0, 48, 12),
        (50, 0, 58, 12),
    )


def test_engcut_binding_does_not_fuzzy_accept_variant_text():
    chars = engcut_chars_from_payload(_payload_for_text("~PEfVC~"))
    bindings = bind_latin_tokens_to_engcut_chars("甲PE/VC乙", chars)

    assert [binding.status for binding in bindings] == [LATIN_ENGCUT_NOT_FOUND_STATUS]


def test_engcut_binding_uses_formula_tokens_only_as_source_order_blockers():
    chars = engcut_chars_from_payload(_payload_for_text("~Incentive~Incentive~"))
    bindings = bind_latin_tokens_to_engcut_chars("$ Incentive_c $ Incentive", chars)

    assert len(bindings) == 1
    assert bindings[0].status == LATIN_ENGCUT_EXACT_STATUS
    assert bindings[0].engcut_start == 11
