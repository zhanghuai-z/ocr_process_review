from app.core.proof_line_facts import proof_display_text, proof_final_text, proof_final_text_set, proof_status
from app.core.proof_occurrence import line_signature
from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
from app.services.char_index_service import CharIndexService
from app.services.proof_edit_service import (
    ProofEditService,
    ProofEditStatus,
    ProofSpanReplacement,
)


def _page_with_line(text: str = "甲乙"):
    chars = [
        Char(
            char=ch,
            confidence=0.9,
            bbox=BBox(1 + idx * 12, 1, 10, 10),
            bbox_source="ocr",
            bbox_granularity="char",
        )
        for idx, ch in enumerate(text)
    ]
    line = Line(text=text, confidence=0.9, bbox=BBox(0, 0, 80, 20), chars=chars)
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 40), lines=[line])
    page = Page(
        image_path="/tmp/proof-edit-service.png",
        width=100,
        height=60,
        page_number=1,
        blocks=[block],
    )
    return page, block, line


def test_proof_edit_service_replaces_span_and_scopes_line_change():
    page, block, line = _page_with_line()
    result = ProofEditService.replace_spans(
        page,
        block,
        line,
        [ProofSpanReplacement(1, 2, "丙")],
        expected_signature=line_signature(line),
    )

    assert result.status == ProofEditStatus.SAVED
    assert proof_display_text(line) == "甲丙"
    assert [char.char for char in line.chars] == ["甲", "丙"]
    assert result.change.text_changed is True
    assert result.change.line_refs
    assert result.change.line_refs[0].line_uid == line.uid
    assert result.change.line_refs[0].write_chars is True


def test_proof_edit_service_rejects_stale_signature():
    page, block, line = _page_with_line()
    stale_signature = line_signature(line)
    line.set_proof_text("甲丁")

    result = ProofEditService.replace_spans(
        page,
        block,
        line,
        [ProofSpanReplacement(1, 2, "丙")],
        expected_signature=stale_signature,
    )

    assert result.status == ProofEditStatus.CONFLICT
    assert result.change.conflict is True
    assert proof_display_text(line) == "甲丁"


def test_char_index_entries_carry_proof_identity():
    page, block, line = _page_with_line()
    service = CharIndexService(include_non_cjk=True).build_index(
        OcrProject(name="proof-identity", pages=[page])
    )

    entry = service.query("甲")[0]

    assert entry.page_uid == page.uid
    assert entry.block_uid == block.uid
    assert entry.line_uid == line.uid
    assert entry.line_signature == line_signature(line)
