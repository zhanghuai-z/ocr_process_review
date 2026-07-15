from collections import Counter

from app.services.char_index_service import CharIndexService


def test_sorted_chars_accepts_multichar_letter_tokens() -> None:
    service = CharIndexService(include_non_cjk=True)
    service._freq = Counter({"甲": 1, "Δ10 Δ10": 1, "PE/VC": 1})

    assert dict(service.sorted_chars()) == {
        "PE/VC": 1,
        "Δ10 Δ10": 1,
        "甲": 1,
    }
