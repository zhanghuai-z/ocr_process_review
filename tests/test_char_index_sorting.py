from app.core.char_index import GEOMETRY_AVAILABLE, CharIndex, CharIndexEntry
from app.services.char_index_service import CharIndexService


def test_sorted_chars_accepts_multichar_letter_tokens() -> None:
    entries = tuple(
        CharIndexEntry(
            proof_uid="proof-1",
            text_unit_uid="unit-1",
            char_index=index,
            text=text,
            page_uid="page-1",
            page_number=1,
            image_path="page.png",
            scope_uid="page-1",
            batch_uid="batch-1",
            line_uid="line-1",
            atom_uid=f"atom-{index}",
            atom_index=index,
            bbox=(index, 0, index + 1, 1),
            geometry_status=GEOMETRY_AVAILABLE,
        )
        for index, text in enumerate(("甲", "Δ10 Δ10", "PE/VC"))
    )
    index = CharIndex(
        proof_uid="proof-1",
        state_revision=1,
        state_fingerprint="fingerprint",
        batch_uid="batch-1",
        entries=entries,
    )

    assert dict(CharIndexService.char_frequency(index)) == {
        "PE/VC": 1,
        "Δ10 Δ10": 1,
        "甲": 1,
    }
