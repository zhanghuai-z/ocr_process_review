from app.core.proof_line_facts import proof_display_text
from app.core.proof_atom import ProofAtomKind, build_line_proof_atoms, has_non_char_atoms
from app.core.proof_char_text import chars_display_text
from app.core.proof_occurrence import resolve_entry_owner
from app.core.proof_projection import build_proof_line_projection
from app.models import (
    BBox, Block, BlockOrigin, BlockType, Char, LayoutBlockSnapshot, LayoutSnapshot,
    Line, OcrPolicy, OcrProject, Page,
)
from app.models.layout_snapshot_store import set_layout_snapshot_for_page
from app.services.char_index_service import CharIndexService
from app.services.proof_probe_text_service import save_displayed_edit_result


def _line(text: str, chars: list[Char]) -> Line:
    return Line(text=text, confidence=0.9, bbox=BBox(0, 0, 200, 20), chars=chars)


def test_build_line_proof_atoms_keeps_stable_single_char_atoms():
    line = _line(
        "甲，2",
        [
            Char("甲", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            Char("，", 0.9, BBox(12, 0, 6, 20), bbox_source="ocr", bbox_granularity="char", token_text="，"),
            Char("2", 0.9, BBox(20, 0, 8, 20), bbox_source="ocr", bbox_granularity="char", token_text="2"),
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert [atom.kind for atom in atoms] == [
        ProofAtomKind.CHAR,
        ProofAtomKind.PUNCT,
        ProofAtomKind.NUMBER,
    ]
    assert [atom.text for atom in atoms] == ["甲", "，", "2"]
    assert not has_non_char_atoms(atoms)


def test_build_line_proof_atoms_keeps_low_confidence_number_punctuation_slots():
    line = _line(
        "1．生产率",
        [
            Char("1", 0.19, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="1"),
            Char("．", 0.19, BBox(12, 15, 5, 5), bbox_source="ocr", bbox_granularity="char", token_text="．"),
            Char("生", 0.9, BBox(22, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="生"),
            Char("产", 0.9, BBox(34, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="产"),
            Char("率", 0.9, BBox(46, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="率"),
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert [atom.kind for atom in atoms] == [
        ProofAtomKind.NUMBER,
        ProofAtomKind.PUNCT,
        ProofAtomKind.CHAR,
        ProofAtomKind.CHAR,
        ProofAtomKind.CHAR,
    ]
    assert [atom.text for atom in atoms] == ["1", "．", "生", "产", "率"]
    assert atoms[0].reliable is False
    assert atoms[0].reason == "low_confidence_single_char"
    assert atoms[1].reliable is False
    assert atoms[1].reason == "low_confidence_single_char"


def test_proof_line_projection_is_the_single_line_view_contract():
    line = _line(
        "甲",
        [
            Char(
                "甲",
                0.9,
                BBox(0, 0, 10, 20),
                bbox_source="ocr",
                bbox_granularity="char",
                token_text="甲",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line])
    page = Page(image_path="/tmp/proof-line-projection.png", width=40, height=40, page_number=1)
    page.blocks = [block]

    projection = build_proof_line_projection(page, block, line, 0, display_text="乙")
    view = projection.view_model(source="test")

    assert projection.block is block
    assert projection.line is line
    assert projection.page is page
    assert projection.line_index == 0
    assert projection.display_text == "乙"
    assert projection.uid
    assert projection.line_signature
    assert [atom.text for atom in projection.atoms] == ["甲"]
    assert view.text == "乙"
    assert view.selection.line_uid == line.uid
    assert view.selection.page_uid == page.uid


def test_resolve_entry_owner_prefers_stable_ids_and_keeps_runtime_fallback():
    line = _line("甲", [Char("甲", 0.9, BBox(0, 0, 10, 20))])
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line])
    block.order = 7
    page = Page(image_path="/tmp/proof-entry-owner.png", width=40, height=40, page_number=3)
    page.blocks = [block]

    class Entry:
        pass

    entry = Entry()
    entry.page_uid = page.uid
    entry.page_id = None
    entry.page_path = ""
    entry.page_number = 0
    entry.block_uid = block.uid
    entry.block_order = -1
    entry.line = line
    assert resolve_entry_owner([page], entry) == (page, block)

    entry.page_uid = ""
    entry.block_uid = ""
    entry.page_path = page.display_image_path
    entry.page_number = page.page_number
    entry.block_order = block.order
    assert resolve_entry_owner([page], entry) == (page, block)


def test_resolve_entry_owner_uses_layout_snapshot_order_for_runtime_fallback():
    line = _line("甲", [Char("甲", 0.9, BBox(0, 0, 10, 20))])
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line], order=9)
    page = Page(image_path="/tmp/proof-entry-owner-snapshot.png", width=40, height=40, page_number=3)
    page.blocks = [block]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-entry-owner",
        source_engine="test",
        source_run_id="run-entry-owner",
        blocks=(
            LayoutBlockSnapshot(
                uid=block.uid,
                block_type=BlockType.TEXT,
                bbox=BBox(0, 0, 20, 20),
                order=2,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
        ),
    ))

    class Entry:
        pass

    entry = Entry()
    entry.page_uid = page.uid
    entry.page_id = None
    entry.page_path = ""
    entry.page_number = 0
    entry.block_uid = ""
    entry.block_order = 2
    entry.line = line

    assert resolve_entry_owner([page], entry) == (page, block)


def test_char_index_build_does_not_mutate_source_char_granularity():
    line = _line(
        "甲",
        [
            Char(
                "甲",
                0.9,
                BBox(0, 0, 10, 20),
                bbox_source="ocr",
                bbox_granularity="",
                token_text="甲",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line])
    page = Page(image_path="/tmp/proof-char-index-no-mutate.png", width=40, height=40, page_number=1)
    page.blocks = [block]

    index = CharIndexService().build_index(OcrProject(name="char-index-no-mutate", pages=[page]))

    assert index.query("甲")
    assert line.chars[0].bbox_granularity == ""


def test_char_index_entries_use_layout_snapshot_order():
    line = _line(
        "甲",
        [
            Char(
                "甲",
                0.9,
                BBox(0, 0, 10, 20),
                bbox_source="ocr",
                bbox_granularity="char",
                token_text="甲",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line], order=9)
    page = Page(image_path="/tmp/char-index-snapshot-order.png", width=40, height=40, page_number=1)
    page.blocks = [block]
    set_layout_snapshot_for_page(page, LayoutSnapshot(
        page_uid=page.uid,
        artifact_uid="artifact-char-index-order",
        source_engine="test",
        source_run_id="run-char-index-order",
        blocks=(
            LayoutBlockSnapshot(
                uid=block.uid,
                block_type=BlockType.TEXT,
                bbox=block.bbox,
                order=2,
                source_label="text",
                origin=BlockOrigin(source_label="text"),
                ocr_policy=OcrPolicy.TEXT_OCR,
            ),
        ),
    ))

    index = CharIndexService().build_index(OcrProject(name="char-index-snapshot-order", pages=[page]))
    entry = index.query("甲")[0]

    assert entry.block_order == 2
    assert entry.block_uid == block.uid


def test_single_char_token_text_syncs_after_proof_edit():
    line = _line(
        "甲",
        [
            Char(
                "甲",
                0.9,
                BBox(0, 0, 10, 20),
                bbox_source="ocr",
                bbox_granularity="char",
                token_text="甲",
            ),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 20, 20), lines=[line])
    page = Page(image_path="/tmp/proof-atom-token-sync.png", width=40, height=40, page_number=1)
    page.blocks = [block]

    change = save_displayed_edit_result(line, page, block, "乙")
    atoms = build_line_proof_atoms(block, line)

    assert change.text_changed
    assert proof_display_text(line) == "乙"
    assert line.chars[0].char == "乙"
    assert line.chars[0].token_text == "乙"
    assert [atom.text for atom in atoms] == ["乙"]


def test_char_granularity_word_token_text_is_metadata_not_display_carrier():
    text = "Guariglia"
    line = _line(
        text,
        [
            Char(
                ch,
                0.9,
                BBox(idx * 10, 0, 8, 20),
                bbox_source="hanwang:EngCut:latin_exact",
                bbox_granularity="char",
                token_text=text,
            )
            for idx, ch in enumerate(text)
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-engcut-exact-token-metadata.png", width=120, height=40, page_number=1)
    page.blocks = [block]

    atoms = build_line_proof_atoms(block, line)

    assert chars_display_text(line.chars) == text
    assert [atom.text for atom in atoms] == list(text)
    assert all(atom.reliable for atom in atoms)

    edited = "GuarigIia"
    change = save_displayed_edit_result(line, page, block, edited)
    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService(include_non_cjk=True).build_index(
        OcrProject(name="engcut-exact-token-metadata", pages=[page])
    )

    assert change.text_changed
    assert proof_display_text(line) == edited
    assert [char.char for char in line.chars] == list(edited)
    assert [char.token_text for char in line.chars] == [text] * len(text)
    assert [atom.text for atom in atoms] == list(edited)
    assert atoms[6].text == "I"
    assert index.query("I")


def test_word_token_text_downgrades_after_equal_length_proof_edit():
    shared_bbox = BBox(0, 0, 40, 20)
    line = _line(
        "税业",
        [
            Char("税", 0.9, shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税业"),
            Char("业", 0.9, shared_bbox, bbox_source="ocr", bbox_granularity="word", token_text="税业"),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-word-token-sync.png", width=80, height=40, page_number=1)
    page.blocks = [block]

    change = save_displayed_edit_result(line, page, block, "利业")
    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService().build_index(OcrProject(name="word-token-sync", pages=[page]))

    assert change.text_changed
    assert proof_display_text(line) == "利业"
    assert [char.char for char in line.chars] == ["利", "业"]
    assert [char.token_text for char in line.chars] == ["利", "业"]
    assert [atom.text for atom in atoms] == ["利业"]
    assert index.query("利")
    assert all(entry.token_text != "税业" for entry in index.query("利"))
    assert index.query("税业") == []


def test_single_word_carrier_syncs_after_proof_edit():
    line = _line(
        "PE/VC",
        [
            Char(
                "PE/VC",
                0.88,
                BBox(0, 0, 48, 20),
                bbox_source="engcut",
                bbox_granularity="word",
                token_text="PE/VC",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 60, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-single-word-carrier.png", width=80, height=40, page_number=1)
    page.blocks = [block]

    change = save_displayed_edit_result(line, page, block, "PE/VD")
    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService(include_fallback=True, include_non_cjk=True).build_index(
        OcrProject(name="single-word-carrier", pages=[page])
    )

    assert change.text_changed
    assert proof_display_text(line) == "PE/VD"
    assert line.chars[0].char == "PE/VD"
    assert line.chars[0].token_text == "PE/VD"
    assert [atom.text for atom in atoms] == ["PE/VD"]
    assert index.query("PE/VD")
    assert index.query("PE/VC") == []


def test_single_cjk_word_carrier_syncs_and_indexes_after_proof_edit():
    line = _line(
        "真实",
        [
            Char(
                "真实",
                0.9,
                BBox(0, 0, 40, 20),
                bbox_source="ocr",
                bbox_granularity="word",
                token_text="真实",
            )
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 50, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-single-cjk-carrier.png", width=80, height=40, page_number=1)
    page.blocks = [block]

    save_displayed_edit_result(line, page, block, "真值")
    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService().build_index(OcrProject(name="single-cjk-carrier", pages=[page]))

    assert proof_display_text(line) == "真值"
    assert line.chars[0].char == "真值"
    assert line.chars[0].token_text == "真值"
    assert [atom.text for atom in atoms] == ["真值"]
    assert index.query("真值")
    assert index.query("真实") == []


def test_inline_formula_carrier_syncs_by_display_span_after_proof_edit():
    formula_bbox = BBox(12, 0, 42, 20)
    line = _line(
        "甲$ A $乙",
        [
            Char("甲", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            Char(
                "$ A $",
                1.0,
                formula_bbox,
                bbox_source="paddle_inline_formula",
                bbox_granularity="word",
                token_text="$ A $",
            ),
            Char("乙", 0.9, BBox(58, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-inline-formula-sync.png", width=100, height=40, page_number=1)
    page.blocks = [block]

    change = save_displayed_edit_result(line, page, block, "丙$ B $乙")
    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService(include_non_cjk=True).build_index(
        OcrProject(name="inline-formula-carrier-sync", pages=[page])
    )

    assert change.text_changed
    assert proof_display_text(line) == "丙$ B $乙"
    assert [(char.char, char.token_text) for char in line.chars] == [
        ("丙", "丙"),
        ("$ B $", "$ B $"),
        ("乙", "乙"),
    ]
    assert [atom.kind for atom in atoms] == [
        ProofAtomKind.CHAR,
        ProofAtomKind.FORMULA,
        ProofAtomKind.CHAR,
    ]
    assert atoms[1].text == "$ B $"
    assert atoms[1].bbox == formula_bbox
    assert index.first_entry("$ B $").bbox == formula_bbox
    assert index.query("$ A $") == []


def test_proof_atoms_degrade_when_chars_no_longer_match_display_text():
    line = _line(
        "甲乙",
        [
            Char("甲", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            Char("乙", 0.9, BBox(12, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 40, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-structural-edit.png", width=80, height=40, page_number=1)
    page.blocks = [block]

    save_displayed_edit_result(line, page, block, "甲乙丙")
    atoms = build_line_proof_atoms(block, line)

    assert proof_display_text(line) == "甲乙丙"
    assert [char.char for char in line.chars] == ["甲", "乙"]
    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.TOKEN
    assert atoms[0].text == "甲乙丙"
    assert atoms[0].reliable is False
    assert atoms[0].reason == "chars_do_not_match_display_text"


def test_formula_carrier_mismatch_downgrades_atom_and_skips_index():
    line = _line(
        "丙$ B $乙",
        [
            Char("甲", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            Char(
                "$ A $",
                1.0,
                BBox(12, 0, 42, 20),
                bbox_source="paddle_inline_formula",
                bbox_granularity="word",
                token_text="$ A $",
            ),
            Char("乙", 0.9, BBox(58, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
        ],
    )
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 80, 24), lines=[line])
    page = Page(image_path="/tmp/proof-atom-inline-formula-mismatch.png", width=100, height=40, page_number=1)
    page.blocks = [block]

    atoms = build_line_proof_atoms(block, line)
    index = CharIndexService(include_non_cjk=True).build_index(
        OcrProject(name="inline-formula-carrier-mismatch", pages=[page])
    )

    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.TOKEN
    assert atoms[0].text == "丙$ B $乙"
    assert atoms[0].reliable is False
    assert atoms[0].reason == "chars_do_not_match_display_text"
    assert index.query("$ A $") == []
    assert index.query("$ B $") == []


def test_build_line_proof_atoms_uses_word_atom_for_multichar_token():
    line = _line(
        "PE/VC",
        [
            Char(
                "PE/VC",
                0.88,
                BBox(0, 0, 48, 20),
                bbox_source="engcut",
                bbox_granularity="word",
                token_text="PE/VC",
            )
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.WORD
    assert atoms[0].text == "PE/VC"
    assert atoms[0].char_indices == (0,)
    assert has_non_char_atoms(atoms)


def test_build_line_proof_atoms_uses_formula_atom_from_paddle_source():
    formula = "$ E=mc^2 $"
    line = _line(
        f"含{formula}",
        [
            Char("含", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="含"),
            Char(
                formula,
                1.0,
                BBox(12, 0, 80, 20),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text="$ E=mc^2 $",
            ),
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert [atom.kind for atom in atoms] == [ProofAtomKind.CHAR, ProofAtomKind.FORMULA]
    assert atoms[1].bbox == BBox(12, 0, 80, 20)


def test_build_line_proof_atoms_downgrades_formula_placeholder_mismatch():
    line = _line(
        "正常文本",
        [
            Char("甲", 0.9, BBox(0, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="甲"),
            Char(
                "$ A $",
                1.0,
                BBox(12, 0, 42, 20),
                bbox_source="paddle_inline_formula",
                bbox_granularity="formula",
                token_text="$ A $",
            ),
            Char("乙", 0.9, BBox(58, 0, 10, 20), bbox_source="ocr", bbox_granularity="char", token_text="乙"),
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.TOKEN
    assert atoms[0].text == "正常文本"
    assert atoms[0].reliable is False
    assert atoms[0].reason == "chars_do_not_match_display_text"


def test_build_line_proof_atoms_groups_overlapping_unstable_chars():
    line = _line(
        "of",
        [
            Char("o", 0.9, BBox(0, 0, 12, 20), bbox_source="engcut", bbox_granularity="char", token_text="o"),
            Char("f", 0.9, BBox(6, 0, 12, 20), bbox_source="engcut", bbox_granularity="char", token_text="f"),
        ],
    )

    atoms = build_line_proof_atoms(None, line)

    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.WORD
    assert atoms[0].text == "of"
    assert atoms[0].reliable is False
    assert atoms[0].bbox == BBox(0, 0, 18, 20)


def test_build_line_proof_atoms_promotes_table_block_to_table_atom():
    line = Line(text="表格OCR", confidence=0.8, bbox=BBox(10, 20, 100, 40))
    block = Block(block_type=BlockType.TABLE, bbox=BBox(0, 0, 120, 60), lines=[line])

    atoms = build_line_proof_atoms(block, line)

    assert len(atoms) == 1
    assert atoms[0].kind == ProofAtomKind.TABLE
    assert atoms[0].text == "表格OCR"
    assert atoms[0].bbox == line.bbox
