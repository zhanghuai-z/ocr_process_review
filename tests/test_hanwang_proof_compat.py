"""Hanwang proof compatibility tests.

Verifies that HProof / VProof / char-index / proof-crop flows handle
Hanwang OCR output correctly, including char_fallback single-char lines.

Hanwang-specific data shapes (from micro_recblock / translator):
  - Char.bbox_source = "hanwang:CharRcg"
  - Char.bbox_granularity = "char"
  - char_fallback lines: bbox_source = "hanwang:CharRcg:char_fallback", 1 char per line
  - Line bboxes come from chars-union + margin (crop-local → page via runner)
  - No "word" granularity; every char is individually boxed
"""
from __future__ import annotations

from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status

from typing import List, Optional
import numpy as np
import pytest

from app.models import BBox, Block, Char, Line, Page
from app.models.enums import BlockType, ProofStatus
from app.models.ocr_observation import replace_block_ocr_line_observations
from app.core.char_bbox_utils import (
    ensure_line_char_bboxes,
    _is_trusted_char_bbox_source,
    MISSING_LINE_BBOX_FLAG,
)
from app.services.char_index_service import CharIndexService


# ── Fixtures / builders ──────────────────────────────────────────


def _hw_char(glyph: str, x: int, y: int, w: int = 30, h: int = 30, *,
             source: str = "hanwang:CharRcg") -> Char:
    """Build a Char as micro_recblock/translator produces it."""
    return Char(
        char=glyph,
        confidence=0.92,
        bbox=BBox(x, y, w, h),
        bbox_source=source,
        bbox_granularity="char",
        token_text="",
    )


def _hw_fallback_char(glyph: str, x: int, y: int, w: int = 30, h: int = 30) -> Char:
    """Build a char as produced by the Hanwang char fallback path."""
    return _hw_char(glyph, x, y, w, h, source="hanwang:CharRcg:char_fallback")


def _hw_line(text: str, chars: List[Char], x: int = 0, y: int = 0,
             line_w: int = 300, line_h: int = 35) -> Line:
    return Line(
        text=text,
        confidence=0.90,
        bbox=BBox(x, y, line_w, line_h),
        chars=chars,
    )


def _page_with_lines(lines: List[Line], img_shape=(400, 300, 3)) -> Page:
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 300, 400))
    replace_block_ocr_line_observations(block.uid, lines)
    page = Page(
        image_path="",
        width=img_shape[1],
        height=img_shape[0],
        blocks=[block],
        page_number=1,
    )
    page.cache_image_path = ""
    return page


def _dummy_image(h: int = 400, w: int = 300) -> np.ndarray:
    """White image with scattered dark pixels so is_meaningful_text_bbox passes.

    is_meaningful_text_bbox uses Otsu thresholding + ink-pixel checks.  A
    fully-white image has zero foreground pixels, causing every bbox to be
    rejected.  Placing dark dots every ~10px gives the Otsu threshold enough
    contrast to detect ink without requiring a real scanned image.
    """
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Sprinkle dark pixels to ensure some ink is detectable in any crop region
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            img[y, x] = (20, 20, 20)
    return img


def _gbk_code(glyph: str) -> int:
    return int.from_bytes(glyph.encode("gbk"), "little")


def _raw_hanwang_line(text: str = "国家", *, source_x: int = 10) -> dict:
    chars = []
    for idx, glyph in enumerate(text):
        x = source_x + idx * 24
        chars.append({
            "codes": [_gbk_code(glyph)],
            "scores": [8 + idx],
            "bbox": {"left": x, "top": 5, "right": x + 20, "bottom": 30, "width": 21, "height": 26},
        })
    return {
        "lines": [{
            "groups": [{
                "bbox": {"left": source_x, "top": 5, "right": source_x + len(text) * 24, "bottom": 30},
                "chars": chars,
            }]
        }]
    }


# ════════════════════════════════════════════════════════════════
# 0. Hanwang translator emits OCR_IR first
# ════════════════════════════════════════════════════════════════


def test_hanwang_translate_linecut_ir_preserves_char_semantics():
    from app.engines.hanwang.translator import translate_linecut_ir

    ir_lines = translate_linecut_ir(_raw_hanwang_line("国家"))

    assert len(ir_lines) == 1
    ir_line = ir_lines[0]
    assert ir_line.text == "国家"
    assert ir_line.source_text == "国家"
    assert len(ir_line.tokens) == 2
    assert ir_line.tokens[0].bbox_source == "hanwang:CharRcg"
    assert ir_line.tokens[0].bbox_granularity == "char"
    assert ir_line.tokens[0].kind == "text"


def test_hanwang_translate_linecut_run_wraps_ir_observation():
    from app.engines.hanwang.translator import translate_linecut_run

    run = translate_linecut_run(
        _raw_hanwang_line("国家"),
        page_uid="page_1",
        block_uid="block_1",
        input_layout_revision=3,
    )

    assert run.uid.startswith("ocrrun_")
    assert run.engine == "hanwang"
    assert run.page_uid == "page_1"
    assert run.block_uid == "block_1"
    assert run.input_layout_revision == 3
    assert [line.text for line in run.lines] == ["国家"]
    assert not hasattr(run.lines[0], "proof_state")


def test_hanwang_ir_to_line_conversion_keeps_final_text_and_fallback_source():
    from app.core.ocr_proof_projection import project_ocr_line_to_proof_line
    from app.engines.hanwang.translator import translate_linecut_ir

    ir_line = translate_linecut_ir(
        _raw_hanwang_line("已"),
        bbox_source="hanwang:CharRcg:char_fallback",
        review_flags=["hanwang_char_fallback"],
    )[0]
    line = project_ocr_line_to_proof_line(ir_line)

    assert line.text == "已"
    assert proof_display_text(line) == "已"
    assert line.ocr_text == "已"
    assert line.review_flags == ["hanwang_char_fallback"]
    assert len(line.chars) == 1
    assert line.chars[0].bbox_source == "hanwang:CharRcg:char_fallback"
    assert line.chars[0].bbox_granularity == "char"


# ════════════════════════════════════════════════════════════════
# 1. _is_trusted_char_bbox_source
# ════════════════════════════════════════════════════════════════


def test_trusted_source_ocr():
    assert _is_trusted_char_bbox_source("ocr") is True


def test_trusted_source_hanwang_charrcg():
    assert _is_trusted_char_bbox_source("hanwang:CharRcg") is True


def test_trusted_source_hanwang_charrcg_char_fallback():
    assert _is_trusted_char_bbox_source("hanwang:CharRcg:char_fallback") is True


def test_trusted_source_hanwang_prefix_case_insensitive():
    assert _is_trusted_char_bbox_source("Hanwang:CharRcg") is True


def test_not_trusted_empty():
    assert _is_trusted_char_bbox_source("") is False


def test_not_trusted_fallback():
    assert _is_trusted_char_bbox_source("fallback") is False


def test_not_trusted_other():
    assert _is_trusted_char_bbox_source("some_other_engine") is False


# ════════════════════════════════════════════════════════════════
# 2. CharIndexService._is_fallback_unit — Hanwang not treated as fallback
# ════════════════════════════════════════════════════════════════


class _FallbackTester:
    """Expose the private _is_fallback_unit via a minimal CharIndexService."""

    def __init__(self) -> None:
        self._svc = CharIndexService()

    def is_fallback(self, source: str, granularity: str) -> bool:
        return self._svc._is_fallback_unit(source, granularity)


@pytest.fixture()
def fallback_tester():
    return _FallbackTester()


def test_hanwang_charrcg_not_fallback(fallback_tester):
    assert fallback_tester.is_fallback("hanwang:CharRcg", "char") is False


def test_hanwang_fallback_char_not_fallback(fallback_tester):
    assert fallback_tester.is_fallback("hanwang:CharRcg:char_fallback", "char") is False


def test_ocr_char_not_fallback(fallback_tester):
    assert fallback_tester.is_fallback("ocr", "char") is False


def test_ocr_word_is_fallback(fallback_tester):
    # "line" granularity is still treated as fallback even for known engines
    assert fallback_tester.is_fallback("ocr", "line") is True


def test_empty_source_is_fallback(fallback_tester):
    assert fallback_tester.is_fallback("", "char") is True


def test_unknown_source_is_fallback(fallback_tester):
    assert fallback_tester.is_fallback("some_engine", "char") is True


def test_hanwang_line_granularity_is_fallback(fallback_tester):
    # Even for Hanwang, "line" granularity is still fallback
    assert fallback_tester.is_fallback("hanwang:CharRcg", "line") is True


# ════════════════════════════════════════════════════════════════
# 3. CharIndexService.build — Hanwang chars enter the index
# ════════════════════════════════════════════════════════════════


def _build_svc_for_pages(pages, *, include_fallback=False, include_non_cjk=False):
    svc = CharIndexService(include_fallback=include_fallback, include_non_cjk=include_non_cjk)
    # Patch build to skip the cv2.imread call; supply None as page_image
    # by overriding _index_line to use a blank image.
    original_build = svc.build

    def _patched_build(pg_list):
        from collections import Counter
        from typing import Set, Tuple
        svc._index = {}
        svc._freq = Counter()
        seen: Set[Tuple[int, int, str]] = set()
        for page_idx, page in enumerate(pg_list):
            page_image = _dummy_image()
            from app.core.proof_line_utils import iter_unique_page_text_lines
            for block, line, line_idx in iter_unique_page_text_lines(page):
                svc._index_line(
                    page_idx=page_idx,
                    page=page,
                    page_image=page_image,
                    block_order=block.order,
                    line_idx=line_idx,
                    line=line,
                    seen=seen,
                )
        for entries in svc._index.values():
            entries.sort(key=lambda e: e._entry_sort_key)
        return svc

    svc.build = _patched_build
    _patched_build([])  # reset only — real call below
    return svc, _patched_build


def test_char_index_includes_hanwang_chars():
    """Standard Hanwang line chars must appear in the char index."""
    chars = [_hw_char("国", 0, 0), _hw_char("家", 30, 0), _hw_char("建", 60, 0)]
    line = _hw_line("国家建", chars, line_w=90)
    page = _page_with_lines([line])
    svc, build = _build_svc_for_pages(page)
    build([page])
    assert svc.unique_chars() >= 1
    entries = svc.query("国")
    assert len(entries) == 1
    e = entries[0]
    assert e.bbox_source == "hanwang:CharRcg"
    assert e.bbox_granularity == "char"


def test_char_index_includes_char_fallback_lines():
    """char_fallback single-char lines must also enter the index."""
    # Simulate char_fallback: 1 char per line, bbox_source has :char_fallback suffix
    fallback_line = _hw_line(
        "已",
        [_hw_fallback_char("已", 50, 100, 28, 28)],
        x=50, y=100, line_w=28, line_h=28,
    )
    page = _page_with_lines([fallback_line])
    svc, build = _build_svc_for_pages(page)
    build([page])
    entries = svc.query("已")
    assert len(entries) == 1
    assert "char_fallback" in entries[0].bbox_source


def test_char_index_multi_line_hanwang():
    """Multiple Hanwang lines → index collects all unique chars."""
    lines = [
        _hw_line("中文", [_hw_char("中", i*30, 0) for i, _ in enumerate("中文")], y=0),
        _hw_line("文字", [_hw_char("文", i*30, 40) for i, _ in enumerate("文字")], y=40),
    ]
    page = _page_with_lines(lines)
    svc, build = _build_svc_for_pages(page)
    build([page])
    # "文" appears in both lines → 2 entries
    assert len(svc.query("文")) == 2
    # "中" appears once
    assert len(svc.query("中")) == 1
    # "字" appears once
    assert len(svc.query("字")) == 1


# ════════════════════════════════════════════════════════════════
# 4. ensure_line_char_bboxes — Hanwang char bboxes preserved
# ════════════════════════════════════════════════════════════════


def test_ensure_preserves_hanwang_char_bboxes():
    """Hanwang CharRcg bboxes must not be replaced by split-line estimates."""
    # Build a line whose char bboxes are slightly off-centre (as CharRcg might
    # return) but definitely still pointing to the correct glyph slot.
    chars = [
        _hw_char("过", x=2, y=2, w=28, h=28),
        _hw_char("去", x=32, y=2, w=28, h=28),
        _hw_char("了", x=62, y=2, w=28, h=28),
    ]
    line = _hw_line("过去了", chars, line_w=90, line_h=32)
    image = _dummy_image()
    result = ensure_line_char_bboxes(line, page_image=image)
    assert len(result) == 3
    # Bboxes should be preserved (not replaced by split-line uniform slices)
    assert result[0].bbox_source == "hanwang:CharRcg"
    assert result[1].bbox_source == "hanwang:CharRcg"
    assert result[2].bbox_source == "hanwang:CharRcg"
    # Granularity also preserved
    for ch in result:
        assert ch.bbox_granularity == "char"


def test_ensure_backfills_missing_hanwang_granularity_as_char():
    """Old projects may have Hanwang char bboxes with an empty granularity field."""
    chars = [
        Char(
            char="甲",
            confidence=0.92,
            bbox=BBox(2, 2, 28, 28),
            bbox_source="hanwang:micro_recblock",
            bbox_granularity="",
        ),
        Char(
            char="A",
            confidence=0.91,
            bbox=BBox(34, 2, 18, 28),
            bbox_source="hanwang:micro_recblock",
            bbox_granularity="",
        ),
    ]
    line = Line(
        text="甲A",
        confidence=0.9,
        bbox=BBox(0, 0, 60, 32),
        chars=chars,
    )

    result = ensure_line_char_bboxes(line, page_image=_dummy_image(h=50, w=80))

    assert [ch.bbox_source for ch in result] == ["hanwang:micro_recblock", "hanwang:micro_recblock"]
    assert [ch.bbox_granularity for ch in result] == ["char", "char"]


def test_ensure_preserves_char_fallback_single_char_line():
    """Single-char char_fallback lines must not lose their bbox."""
    ch = _hw_fallback_char("己", x=50, y=80, w=30, h=30)
    line = Line(
        text="己", confidence=0.88,
        bbox=BBox(50, 80, 30, 30),
        chars=[ch],
    )
    image = _dummy_image()
    result = ensure_line_char_bboxes(line, page_image=image)
    assert len(result) == 1
    r = result[0]
    assert r.bbox is not None
    assert r.bbox.area > 0
    assert r.bbox_source == "hanwang:CharRcg:char_fallback"
    assert r.bbox_granularity == "char"


def test_ensure_does_not_repair_hanwang_even_when_neighbour_overlap_might_be_higher():
    """Regression: even if a Hanwang bbox happens to overlap slightly with a
    neighbour slot (e.g. shared strokes), it must NOT be replaced."""
    # Make chars where the bboxes slightly overlap each other (real-world glyphs
    # can share ink regions between adjacent characters).
    chars = [
        _hw_char("一", x=0, y=0, w=35, h=20),   # slightly wider than slot
        _hw_char("二", x=30, y=0, w=35, h=20),   # overlaps with slot 0
        _hw_char("三", x=60, y=0, w=35, h=20),   # overlaps with slot 1
    ]
    line = _hw_line("一二三", chars, line_w=90, line_h=20)
    image = _dummy_image(h=50, w=100)
    result = ensure_line_char_bboxes(line, page_image=image)
    # Sources must remain "hanwang:CharRcg" — neighbour-repair must not fire
    for ch in result:
        assert ch.bbox_source == "hanwang:CharRcg", (
            f"Expected bbox_source 'hanwang:CharRcg', got {ch.bbox_source!r}"
        )


# ════════════════════════════════════════════════════════════════
# 5. ProofCropService normalises Hanwang pages without crashing
# ════════════════════════════════════════════════════════════════


def test_proof_crop_service_handles_hanwang_block(tmp_path):
    """ProofCropService.normalize_pages must not crash on Hanwang output."""
    from app.services.proof_crop_service import ProofCropService

    # Write a real image file so cv2.imread succeeds
    img = _dummy_image(h=200, w=300)
    import cv2
    img_path = str(tmp_path / "page.png")
    cv2.imwrite(img_path, img)

    chars = [_hw_char("研", 10, 10), _hw_char("究", 40, 10), _hw_char("者", 70, 10)]
    line = _hw_line("研究者", chars, x=10, y=10, line_w=90, line_h=30)
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 300, 200), lines=[line])
    page = Page(image_path=img_path, width=300, height=200, blocks=[block], page_number=1)
    page.cache_image_path = img_path

    svc = ProofCropService()
    stats = svc.normalize_pages([page])
    # Should process 1 line without exception
    assert stats.lines == 1


def test_proof_crop_service_handles_char_fallback_line(tmp_path):
    """Single-char char_fallback lines must survive ProofCropService normalisation."""
    from app.services.proof_crop_service import ProofCropService
    import cv2

    img = _dummy_image(h=200, w=300)
    img_path = str(tmp_path / "page_fb.png")
    cv2.imwrite(img_path, img)

    # Simulate several char_fallback single-char lines
    fallback_lines = [
        _hw_line(ch, [_hw_fallback_char(ch, x=10 + i*40, y=50, w=30, h=30)],
                 x=10 + i*40, y=50, line_w=30, line_h=30)
        for i, ch in enumerate("国民")
    ]
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 300, 200),
                  lines=fallback_lines)
    page = Page(image_path=img_path, width=300, height=200, blocks=[block],
                page_number=1)
    page.cache_image_path = img_path

    svc = ProofCropService()
    stats = svc.normalize_pages([page])
    assert stats.lines == 2


# ════════════════════════════════════════════════════════════════
# 6. h_proof helpers are Hanwang-safe
# ════════════════════════════════════════════════════════════════


def test_chars_are_single_codepoint_with_hanwang_chars():
    """_chars_are_single_codepoint should be True for Hanwang char-granularity output."""
    from app.ui.proof.h_proof import _chars_are_single_codepoint
    chars = [_hw_char("中", 0, 0), _hw_char("文", 30, 0)]
    assert _chars_are_single_codepoint(chars) is True


def test_chars_are_single_codepoint_with_char_fallback_chars():
    """char_fallback chars are single chars — slot model must hold."""
    from app.ui.proof.h_proof import _chars_are_single_codepoint
    chars = [_hw_fallback_char("己", 0, 0)]
    assert _chars_are_single_codepoint(chars) is True


def test_canonicalize_text_to_slots_hanwang_exact_match():
    """When Hanwang chars == text length, canonicalize must lock slots."""
    from app.ui.proof.h_proof import _canonicalize_text_to_slots
    chars = [_hw_char("A", 0, 0), _hw_char("B", 30, 0)]
    canonical, locked = _canonicalize_text_to_slots("AB", chars)
    assert canonical == "AB"
    assert locked is True


def test_canonicalize_text_to_slots_hanwang_short_line():
    """Short OCR text for a Hanwang line → padded to match char count."""
    from app.ui.proof.h_proof import _canonicalize_text_to_slots
    chars = [_hw_char("国", 0, 0), _hw_char("家", 30, 0), _hw_char("力", 60, 0)]
    # OCR text returned 2 chars but line has 3 chars
    canonical, locked = _canonicalize_text_to_slots("国家", chars)
    assert len(canonical) == 3
    assert locked is True
    assert canonical[2] == " "


# ════════════════════════════════════════════════════════════════
# 7. proof_image_service — verified_char_crop with Hanwang bboxes
# ════════════════════════════════════════════════════════════════


class _StubCache:
    def __init__(self, img):
        self._img = img
        self.crop_calls: list = []

    def get_page_image(self, path):
        return self._img

    def get_char_crop(self, path, bbox, size, *, pad):
        self.crop_calls.append((bbox, size, pad))
        return "PIXMAP"


def test_verified_char_crop_hanwang_inbounds():
    """Hanwang char bboxes within image bounds must be passed unchanged."""
    from app.services.proof_image_service import verified_char_crop
    img = _dummy_image(h=200, w=300)
    cache = _StubCache(img)
    bbox = BBox(10, 10, 30, 30)
    result = verified_char_crop(cache, "p.png", bbox, size=56)
    assert result == "PIXMAP"
    assert len(cache.crop_calls) == 1
    called_bbox, _, _ = cache.crop_calls[0]
    assert (called_bbox.x, called_bbox.y, called_bbox.w, called_bbox.h) == (10, 10, 30, 30)


def test_verified_char_crop_hanwang_out_of_bounds_clamped():
    """Hanwang bboxes that exceed image bounds are clamped before calling cache."""
    from app.services.proof_image_service import verified_char_crop
    img = _dummy_image(h=100, w=100)
    cache = _StubCache(img)
    bbox = BBox(90, 90, 40, 40)   # x2=130, y2=130 → out of bounds
    verified_char_crop(cache, "p.png", bbox, size=56)
    called_bbox, _, _ = cache.crop_calls[0]
    assert called_bbox.x + called_bbox.w <= 100
    assert called_bbox.y + called_bbox.h <= 100


# ════════════════════════════════════════════════════════════════
# 8. VProofPanel CharIndexService wiring (smoke, no Qt required)
# ════════════════════════════════════════════════════════════════


def test_char_index_svc_default_params_hanwang_chars_indexed():
    """CharIndexService() with default params must index Hanwang CJK chars."""
    svc = CharIndexService()   # include_fallback=False, include_non_cjk=False
    # Not fallback: hanwang:CharRcg / char → should be indexed
    assert not svc._is_fallback_unit("hanwang:CharRcg", "char")
    assert not svc._is_fallback_unit("hanwang:CharRcg:char_fallback", "char")
    # Still fallback: empty source, line granularity, unknown engine
    assert svc._is_fallback_unit("", "char")
    assert svc._is_fallback_unit("hanwang:CharRcg", "line")
    assert svc._is_fallback_unit("some_engine", "char")


def test_char_index_indexes_old_hanwang_chars_with_empty_granularity():
    """Old Hanwang chars are indexed without mutating persisted char metadata."""
    line = Line(
        text="甲A，",
        confidence=0.9,
        bbox=BBox(0, 0, 90, 32),
        chars=[
            Char(char="甲", confidence=0.92, bbox=BBox(2, 2, 28, 28), bbox_source="hanwang:micro_recblock"),
            Char(char="A", confidence=0.91, bbox=BBox(34, 2, 18, 28), bbox_source="hanwang:micro_recblock"),
            Char(char="，", confidence=0.89, bbox=BBox(58, 22, 8, 8), bbox_source="hanwang:micro_recblock"),
        ],
    )
    page = Page(image_path="/tmp/old-hanwang-granularity.png", width=100, height=50)
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 90, 32), order=0)
    replace_block_ocr_line_observations(block.uid, [line])
    page.blocks = [block]

    svc = CharIndexService(include_non_cjk=True).build([page])

    assert svc.query("甲")
    assert svc.query("A")
    assert svc.query("，")
    assert {ch.bbox_granularity for ch in line.chars} == {""}
