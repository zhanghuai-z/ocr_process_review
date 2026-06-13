from app.models import BBox, Line
from app.services.char_index_service import CharEntry
from app.ui.proof.v_proof import _gallery_crop_pad_for_entry


def _entry(
    char: str,
    *,
    bbox_granularity: str = "char",
    collection_kind: str = "char",
) -> CharEntry:
    return CharEntry(
        char=char,
        page_path="/tmp/page.png",
        page_number=1,
        line=Line(text=char, confidence=0.9, bbox=BBox(0, 0, 20, 30)),
        char_idx=0,
        bbox=BBox(10, 10, 12, 24),
        bbox_source="hanwang:EngCut:latin_exact",
        bbox_granularity=bbox_granularity,
        token_text=char,
        collection_kind=collection_kind,
    )


def test_vproof_gallery_uses_tight_safety_margin_for_ascii_letters_and_digits():
    assert _gallery_crop_pad_for_entry(_entry("A")) == 1
    assert _gallery_crop_pad_for_entry(_entry("9")) == 1


def test_vproof_gallery_keeps_small_context_for_ascii_punctuation():
    assert _gallery_crop_pad_for_entry(_entry(".")) == 2
    assert _gallery_crop_pad_for_entry(_entry("/")) == 2


def test_vproof_gallery_keeps_adaptive_padding_for_cjk_and_tokens():
    assert _gallery_crop_pad_for_entry(_entry("甲")) is None
    assert _gallery_crop_pad_for_entry(_entry("①")) is None
    assert _gallery_crop_pad_for_entry(_entry("A", bbox_granularity="word")) is None
    assert _gallery_crop_pad_for_entry(_entry("A", collection_kind="token")) is None
