from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from PySide6.QtGui import QColor, QImage
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from app.application.proof_workspace import (
    ProofAtomView,
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspaceView,
)
from app.application.contracts import ProofEditCommand


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _workspace(tmp_path: Path) -> ProofWorkspaceView:
    image_path = tmp_path / "page.png"
    image = QImage(100, 60, QImage.Format.Format_RGB32)
    image.fill(QColor("black"))
    for y in range(10, 30):
        for x in range(10, 30):
            image.setPixelColor(x, y, QColor("red"))
        for x in range(30, 50):
            image.setPixelColor(x, y, QColor("green"))
    assert image.save(str(image_path))

    atom_a = ProofAtomView(
        proof_uid="proof-1",
        batch_uid="batch-1",
        page_uid="page-1",
        region_uid="region-1",
        line_uid="line-1",
        atom_uid="atom-a",
        atom_index=0,
        text="a",
        confidence=0.96,
        bbox=(10, 10, 30, 30),
        source="test",
        granularity="char",
        token_text="a",
        render_kind="text",
        char_span=(0, 1),
        geometry_available=True,
    )
    atom_b = ProofAtomView(
        proof_uid="proof-1",
        batch_uid="batch-1",
        page_uid="page-1",
        region_uid="region-1",
        line_uid="line-1",
        atom_uid="atom-b",
        atom_index=1,
        text="b",
        confidence=0.70,
        bbox=(30, 10, 50, 30),
        source="test",
        granularity="char",
        token_text="b",
        render_kind="text",
        char_span=(1, 2),
        geometry_available=True,
    )
    unit = ProofTextUnitView(
        proof_uid="proof-1",
        text_unit_uid="unit-1",
        order=0,
        text="ab",
        status="unchecked",
        revision=1,
        fingerprint="unit-fingerprint-1",
    )
    line = ProofLineView(
        proof_uid="proof-1",
        batch_uid="batch-1",
        page_uid="page-1",
        line_uid="line-1",
        region_uid="region-1",
        line_uids=("line-1",),
        region_uids=("region-1",),
        text_unit_uid="unit-1",
        order=0,
        ocr_text="ab",
        proof_text="ab",
        status="unchecked",
        confidence=0.83,
        bbox=(10, 10, 50, 30),
        render_kind="text",
        atoms=(atom_a, atom_b),
        state_revision=3,
        state_fingerprint="state-fingerprint-1",
        text_unit_revision=1,
        text_unit_fingerprint="unit-fingerprint-1",
    )
    page = ProofPageView(
        project_uid="project-1",
        page_uid="page-1",
        page_number=1,
        source_page_index=0,
        image_path=str(image_path),
        source_path="source.pdf",
        cache_image_path=str(image_path),
        thumbnail_path=str(image_path),
        width=100,
        height=60,
        status="imported",
        error="",
        image_hash="image-hash-1",
        image_revision=1,
        page_fingerprint="page-fingerprint-1",
    )
    state = ProofStateView(
        proof_uid="proof-1",
        page_uid="page-1",
        scope_uid="page-1",
        anchor_uid="anchor-1",
        anchor_revision=1,
        layout_fingerprint="layout-fingerprint-1",
        source_fingerprint="source-fingerprint-1",
        active_pointer_uid="pointer-1",
        active_pointer_revision=1,
        active_batch_uid="batch-1",
        revision=3,
        fingerprint="state-fingerprint-1",
        rebind_required=False,
        text_units=(unit,),
        lines=(line,),
    )
    return ProofWorkspaceView(
        project_uid="project-1",
        pages=(page,),
        proof_states=(state,),
        lines=(line,),
    )


def test_vproof_uses_character_crops_and_highlights_one_ocr_occurrence(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel, _page_pixmap

    workspace = _workspace(tmp_path)
    panel = VProofPanel(workspace)
    panel.show()
    qapp.processEvents()
    page = workspace.pages[0]
    entry_a = next(entry for entry in panel._entries if entry.text == "a")
    entry_b = next(entry for entry in panel._entries if entry.text == "b")

    assert panel._selected_entry == entry_a
    assert panel._ocr_context.toPlainText() == "ab"
    assert panel._ocr_context.extraSelections()[0].cursor.selectedText() == "a"

    crop_a = _page_pixmap(page, entry_a.bbox, QSize(200, 200))
    assert crop_a.size() == QSize(200, 200)
    assert crop_a.toImage().pixelColor(100, 100).red() > 180
    crop_b = _page_pixmap(page, entry_b.bbox, QSize(200, 200))
    assert crop_b.size() == QSize(200, 200)
    assert crop_b.toImage().pixelColor(100, 100).green() > 100
    assert _page_pixmap(page, None, QSize(200, 200)).isNull()
    icon_a = panel._entry_icon(entry_a).pixmap(QSize(56, 56)).toImage()
    icon_b = panel._entry_icon(entry_b).pixmap(QSize(56, 56)).toImage()
    assert icon_a.pixelColor(28, 28).red() > 180
    assert icon_b.pixelColor(28, 28).green() > 100

    panel._set_gallery((entry_a,))
    qapp.processEvents()
    assert panel._selected_entry == entry_a
    gallery_rect = panel._gallery.visualItemRect(panel._gallery.item(0))
    assert gallery_rect.width() >= 56
    assert gallery_rect.height() >= 56
    assert panel._ocr_context.objectName() == "vproofOcrContext"
    assert panel._ocr_context.isReadOnly()
    assert not hasattr(panel, "_proof_context")
    assert len(panel.findChildren(QPlainTextEdit)) == 1
    assert panel._ocr_context.toPlainText() == "ab"
    selections = panel._ocr_context.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.selectedText() == "a"

    panel._set_gallery((entry_b,))
    qapp.processEvents()
    assert panel._selected_entry == entry_b
    selections = panel._ocr_context.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.selectedText() == "b"

    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)
    assert panel._gallery_direct_overwrite("x") is True
    assert len(commands) == 1
    assert commands[0].op == "replace_many"
    assert commands[0].proof_uid == "proof-1"
    assert commands[0].replacements == (("unit-1", "ax"),)
    panel.close()


def test_vproof_page_context_aggregates_regions_without_changing_char_identity(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel

    workspace = _workspace(tmp_path)
    state = workspace.proof_states[0]
    first_line = state.lines[0]
    second_unit = replace(
        state.text_units[0],
        text_unit_uid="unit-2",
        order=1,
        text="cd",
        fingerprint="unit-fingerprint-2",
    )
    second_line = replace(
        first_line,
        line_uid="line-2",
        region_uid="region-2",
        line_uids=("line-2",),
        region_uids=("region-2",),
        text_unit_uid="unit-2",
        order=1,
        ocr_text="cd",
        proof_text="cd",
        atoms=(),
        text_unit_fingerprint=second_unit.fingerprint,
    )
    next_state = replace(
        state,
        text_units=(state.text_units[0], second_unit),
        lines=(first_line, second_line),
    )
    panel = VProofPanel(
        replace(
            workspace,
            proof_states=(next_state,),
            lines=(first_line, second_line),
        )
    )

    assert panel._ocr_context.toPlainText() == "ab\n\ncd"
    assert panel._selected_entry is not None
    assert panel._selected_entry.text_unit_uid == "unit-1"
    assert panel._ocr_context.extraSelections()[0].cursor.selectedText() == "a"
    panel.close()
