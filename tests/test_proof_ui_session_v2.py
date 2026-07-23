from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QListWidgetItem

from app.application.proof_workspace import build_proof_workspace_view
from app.application.contracts import ProofBatchEditCommand, ProofEditCommand
from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.layout_snapshot import LayoutSnapshot
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.paddle_artifact import PaddleArtifact
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.models.project_session import PageRecord, ProjectSession
from app.services.proof_session_service import ProofSessionService


ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    ROOT / "app/ui/proof/h_proof.py",
    ROOT / "app/ui/proof/v_proof.py",
    ROOT / "app/ui/proof/confidence_view.py",
    ROOT / "app/ui/proof/char_verdict.py",
    ROOT / "app/ui/proof/quality_stats_dialog.py",
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _session(tmp_path: Path) -> tuple[ProjectSession, ProofSessionService]:
    project_uid = "ui-proof-project"
    image_path = tmp_path / "page.png"
    image = QImage(120, 80, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path))
    session = ProjectSession(project_uid)
    session.page_repository.put(
        PageRecord(
            project_uid=project_uid,
            uid="page-1",
            image_path=str(image_path),
            source_path="source.pdf",
            cache_image_path=str(image_path),
            thumbnail_path=str(image_path),
            width=120,
            height=80,
            page_number=1,
            source_page_index=0,
            status="imported",
            error="",
            image_hash="page-hash",
            image_revision=1,
        ),
        expected_revision=0,
    )
    layout = LayoutSnapshot(
            page_uid="page-1",
            revision=1,
            artifact_uid="layout-1",
            source_engine="test",
            source_run_id="layout-run-1",
            blocks=(),
    )
    session.paddle_artifact_repository.append(
        PaddleArtifact(
            project_uid=project_uid,
            uid="layout-1",
            page_uid="page-1",
            source_engine="test",
            source_run_id="layout-run-1",
            image_hash="page-hash",
            payload_json="{}",
        )
    )
    session.layout_repository.put(layout, expected_revision=0)
    page = session.page_repository.get("page-1")
    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=project_uid,
            uid="run-1",
            engine="test",
            layout_fingerprint=layout_snapshot_fingerprint(layout),
            input_fingerprint="input-1",
            metadata=(("page_fingerprint", page.fingerprint),),
        )
    )
    region = ocr.append_region(
        OcrRegion(
            project_uid=project_uid,
            uid="region-1",
            run_uid=run.uid,
            page_uid="page-1",
            bbox=(0, 0, 100, 60),
            kind="text",
        )
    )
    line = ocr.append_line(
        OcrLine(
            project_uid=project_uid,
            uid="line-1",
            run_uid=run.uid,
            region_uid=region.uid,
            page_uid="page-1",
            text="ab",
            bbox=(10, 10, 50, 30),
            confidence=0.9,
            atom_uids=("atom-1", "atom-2"),
        )
    )
    atoms = (
        ocr.append_atom(
            OcrAtom(
                project_uid=project_uid,
                uid="atom-1",
                run_uid=run.uid,
                region_uid=region.uid,
                line_uid=line.uid,
                index=0,
                text="a",
                bbox=(10, 10, 30, 30),
                confidence=0.96,
                source="test",
                granularity="char",
                token_text="a",
            )
        ),
        ocr.append_atom(
            OcrAtom(
                project_uid=project_uid,
                uid="atom-2",
                run_uid=run.uid,
                region_uid=region.uid,
                line_uid=line.uid,
                index=1,
                text="b",
                bbox=(30, 10, 50, 30),
                confidence=0.7,
                source="test",
                granularity="char",
                token_text="b",
            )
        ),
    )
    batch = ocr.append_batch(
        OcrBatch(
            project_uid=project_uid,
            uid="batch-1",
            run_uid=run.uid,
            scope_uid="page-1",
            input_fingerprint=run.input_fingerprint,
            layout_fingerprint=run.layout_fingerprint,
            region_uids=(region.uid,),
            line_uids=(line.uid,),
            atom_uids=tuple(atom.uid for atom in atoms),
        )
    )
    ocr.switch_active_pointer(
        OcrActivePointer(
            project_uid=project_uid,
            uid="pointer-1",
            scope_uid="page-1",
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=1,
        ),
        expected_revision=0,
        expected_fingerprint=None,
    )
    anchor = ProofAnchorSnapshot(
        project_uid=project_uid,
        uid="anchor-1",
        scope_uid="page-1",
        layout_fingerprint=batch.layout_fingerprint,
        source_fingerprint=batch.fingerprint,
        anchor_revision=1,
    )
    segment = ProofAlignmentSegment(
        project_uid=project_uid,
        uid="segment-1",
        anchor_uid=anchor.uid,
        text_unit_uid="unit-1",
        source_line_uids=("line-1",),
        source_start=0,
        source_end=2,
        proof_start=0,
        proof_end=2,
    )
    alignment = ProofAlignmentSlice(
        project_uid=project_uid,
        uid="slice-1",
        segment_uid=segment.uid,
        source_start=0,
        source_end=2,
        proof_start=0,
        proof_end=2,
        source_text="ab",
        proof_text="ab",
    )
    state = ProofState(
        project_uid=project_uid,
        uid="proof-1",
        anchor_snapshot=anchor,
        text_units=(
            ProofTextUnit(
                project_uid=project_uid,
                uid="unit-1",
                order=0,
                text="ab",
            ),
        ),
        alignment_segments=(segment,),
        alignment_slices=(alignment,),
    )
    service = ProofSessionService(session)
    service.create_state(state)
    return session, service


def test_owned_proof_ui_accepts_only_workspace_and_has_no_legacy_dependencies() -> None:
    forbidden_modules = {
        "app.core.proof_projection",
        "app.core.proof_state_bus",
        "app.services.proof_edit_service",
        "app.services.proof_hproof_session",
        "app.services.proof_occurrence_session",
        "app.services.proof_probe_text_service",
    }
    forbidden_names = {
        "OcrProject",
        "ProofStateBus",
        "ProjectSession",
        "ProofSessionService",
        "set_session",
    }
    for path in OWNED:
        source = path.read_text(encoding="utf-8")
        assert "set_session" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden_modules
                assert not {alias.name for alias in node.names} & forbidden_names
                assert not module.startswith("app.models")
                assert not module.startswith("app.services")
                assert not module.startswith("app.core")
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names


def test_hproof_emits_replace_many_with_workspace_cas_without_writing_source(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    panel = HProofPanel(workspace)
    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)
    assert panel.objectName() == "proofRoot"
    assert panel._splitter.objectName() == "hproofSplitter"
    assert panel._page_directory.objectName() == "pageDirectoryList"
    assert panel._status_bar.objectName() == "proofStatusBar"
    assert panel._scroll.objectName() == "proofScroll"
    assert panel._rows_root.objectName() == "proofLineList"
    assert len(panel._rows) == 1
    row_widget = panel._row_widgets[("proof-1", "unit-1")]
    assert row_widget.property("active") is True
    assert row_widget._focus_depth == "active"

    editor = row_widget.editor
    cursor = editor.textCursor()
    cursor.setPosition(1)
    editor.setTextCursor(cursor)
    assert row_widget._selected_char_index == 1
    row_widget.set_focus_depth("near")
    assert editor.isHidden() is True
    row_widget.set_focus_depth("active")
    assert editor.isHidden() is False
    editor.setPlainText("ax")
    assert panel.save() is True
    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, ProofEditCommand)
    assert command.op == "replace_many"
    assert command.proof_uid == "proof-1"
    assert command.expected_revision == workspace.proof_states[0].revision
    assert command.expected_fingerprint == workspace.proof_states[0].fingerprint
    assert command.replacements == (("unit-1", "ax"),)
    assert session.proof_repository.get_state("proof-1").text_units[0].text == "ab"
    assert session.ocr_observation_repository.get_line("line-1").text == "ab"
    assert not hasattr(panel, "set_session")
    panel.close()


def test_hproof_status_and_history_are_commands(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    panel = HProofPanel(workspace)
    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)
    row = panel._rows[0]
    widget = panel._row_widgets[row.key]
    panel._confirm_row(row, widget)
    panel._apply_history(row, -1)
    panel._apply_history(row, 1)
    assert [command.op for command in commands] == ["set_status", "undo", "redo"]
    assert commands[0].status == "checked"
    assert commands[0].expected_unit_revision == row.unit.revision
    panel.close()


def test_hproof_restores_atom_geometry_for_image_text_lookup(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from PySide6.QtCore import QPoint
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    row = panel._rows[0]
    row_widget = panel._row_widgets[row.key]

    assert [entry.atom_uid for entry in row.entries] == ["atom-1", "atom-2"]
    assert [entry.bbox for entry in row.entries] == [
        (10, 10, 30, 30),
        (30, 10, 50, 30),
    ]
    assert row_widget._line_bbox == (10, 10, 50, 30)

    row_widget._image.resize(200, 58)
    row_widget._refresh_image()
    row_widget._on_image_clicked(QPoint(16, 29))

    assert row_widget.editor.textCursor().selectedText() == "a"
    panel.close()


def test_hproof_projects_atom_geometry_into_editor_typography(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from PySide6.QtGui import QFont, QTextCursor
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    row = panel._rows[0]
    row_widget = panel._row_widgets[row.key]

    row_widget._image.resize(200, 58)
    row_widget._refresh_image()

    first = QTextCursor(row_widget.editor.document())
    first.setPosition(0)
    first.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
    first_format = first.charFormat()

    assert row_widget.editor.font().pixelSize() >= 17
    assert first_format.fontLetterSpacingType() == QFont.SpacingType.AbsoluteSpacing
    assert first_format.fontLetterSpacing() != 0.0

    row_widget.editor.setPlainText("abc")
    reset = QTextCursor(row_widget.editor.document())
    reset.setPosition(0)
    reset.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
    assert reset.charFormat().fontLetterSpacing() == 0.0
    panel.close()


def test_hproof_interaction_cells_cover_wide_glyph_without_moving_atom_centers(
    qapp: QApplication,
) -> None:
    from PySide6.QtGui import QFont, QTextCursor
    from app.ui.proof.h_proof import _SlotLineEditor

    editor = _SlotLineEditor()
    editor.resize(120, 40)
    font = QFont(editor.font())
    font.setPixelSize(28)
    editor.setFont(font)
    editor.setPlainText(")一(")
    editor.set_slot_geometry([20.0, 50.0, 80.0], [14.0, 4.0, 14.0])
    editor.set_active_visual(True)
    cursor = QTextCursor(editor.document())
    cursor.setPosition(1)
    cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    cells = editor._interaction_cells()

    assert all(cell is not None for cell in cells)
    left_cell, middle_cell, right_cell = cells
    assert left_cell is not None
    assert middle_cell is not None
    assert right_cell is not None
    assert left_cell.right() + 1 == middle_cell.left()
    assert middle_cell.right() + 1 == right_cell.left()
    assert (middle_cell.left(), middle_cell.right()) == (35, 64)
    assert middle_cell.width() > 4
    assert editor.slot_geometry() == ([20.0, 50.0, 80.0], [14.0, 4.0, 14.0])
    assert editor._slot_index_for_x(37.0) == 1
    assert editor._slot_index_for_x(63.0) == 1
    assert editor._cursor_rect() == middle_cell

    editor.show()
    qapp.processEvents()
    rendered = QPixmap(editor.size())
    rendered.fill(Qt.GlobalColor.transparent)
    editor.render(rendered)
    assert rendered.toImage().pixelColor(37, 10) == QColor("#cfe2ff")
    editor.close()


def test_hproof_shrinks_overflowing_glyph_to_atom_without_clipping(
    qapp: QApplication,
) -> None:
    from PySide6.QtGui import QFont, QFontMetrics
    from app.ui.proof.h_proof import _fit_glyph_font, _SlotLineEditor

    base_font = QFont()
    base_font.setPixelSize(28)
    fitted_font = _fit_glyph_font("一", base_font, 4.0)
    assert fitted_font.pixelSize() < base_font.pixelSize()
    assert QFontMetrics(fitted_font).tightBoundingRect("一").width() <= 4
    assert _fit_glyph_font("i", base_font, 20.0).pixelSize() == 28
    assert _fit_glyph_font("j", base_font, 2.0).pixelSize() == 28

    def render(text: str) -> QImage:
        editor = _SlotLineEditor()
        editor.resize(100, 40)
        font = QFont(base_font)
        editor.setFont(font)
        editor.setPlainText(text)
        editor.set_slot_geometry([50.0], [4.0])
        editor.show()
        qapp.processEvents()
        rendered = QImage(editor.size(), QImage.Format.Format_ARGB32)
        rendered.fill(Qt.GlobalColor.transparent)
        editor.render(rendered)
        editor.close()
        return rendered

    glyph = render("一")
    blank = render(" ")
    differing_pixels = [
        (x, y)
        for y in range(glyph.height())
        for x in range(glyph.width())
        if glyph.pixelColor(x, y) != blank.pixelColor(x, y)
    ]

    assert differing_pixels
    assert all(48 <= x < 52 for x, _y in differing_pixels)


def test_hproof_double_click_edit_expands_one_slot_without_changing_fast_overwrite(
    qapp: QApplication,
) -> None:
    from PySide6.QtGui import QTextCursor
    from app.ui.proof.h_proof import _SlotLineEditor

    editor = _SlotLineEditor()
    editor.resize(180, 40)
    editor.setPlainText("PENC")
    editor.set_slot_geometry(
        [20.0, 50.0, 80.0, 110.0],
        [20.0, 20.0, 12.0, 20.0],
    )
    editor.show()
    qapp.processEvents()

    QTest.mouseDClick(editor, Qt.MouseButton.LeftButton, pos=QPoint(80, 20))
    qapp.processEvents()
    assert editor._expanded_edit.isVisible()
    assert editor._expanded_edit.text() == "N"
    editor._expanded_edit.setText("/V")
    QTest.keyClick(editor._expanded_edit, Qt.Key.Key_Return)
    assert editor.toPlainText() == "PE/VC"
    assert editor.textCursor().selectedText() == "/V"

    editor.setPlainText("PENC")
    cursor = QTextCursor(editor.document())
    cursor.setPosition(2)
    cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    QTest.keyClicks(editor, "/V")
    assert editor.toPlainText() == "PE/V"

    editor.setPlainText("PENC")
    editor._open_expanded_edit(2)
    editor._expanded_edit.setText("/V")
    QTest.keyClick(editor._expanded_edit, Qt.Key.Key_Escape)
    assert editor.toPlainText() == "PENC"
    assert editor._expanded_edit.isHidden()

    editor._open_expanded_edit(2)
    editor._expanded_edit.setText("")
    QTest.keyClick(editor._expanded_edit, Qt.Key.Key_Return)
    assert editor.toPlainText() == "PENC"
    assert editor._expanded_edit.isVisible()

    other = QLineEdit()
    other.show()
    other.setFocus()
    qapp.processEvents()
    assert editor._expanded_edit.isHidden()
    assert editor.toPlainText() == "PENC"
    other.close()
    editor.close()


def _formula_workspace(workspace, unit, line, state, *, text, line_kind, atoms):
    next_unit = replace(unit, text=text)
    next_line = replace(
        line,
        ocr_text=text,
        proof_text=text,
        bbox=(10, 10, 10 + len(text) * 20, 30),
        render_kind=line_kind,
        atoms=atoms,
    )
    next_state = replace(state, text_units=(next_unit,), lines=(next_line,))
    return replace(workspace, proof_states=(next_state,), lines=(next_line,))


def _inline_atom(line, uid, index, text, span, kind, bbox):
    return replace(
        line.atoms[0],
        atom_uid=uid,
        atom_index=index,
        text=text,
        render_kind=kind,
        char_span=span,
        geometry_available=span is not None,
        bbox=bbox,
    )


def test_hproof_plain_text_row_has_no_formula_overlay(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget.row.kind == "text"
    assert widget._formula_render_area.isHidden()
    widget._image.resize(200, 58)
    widget._refresh_image()
    assert widget.editor.atom_visual_overlays() == []
    panel.close()


def test_hproof_inline_formula_atoms_render_over_exact_spans(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    atoms = (
        _inline_atom(line, "atom-a", 0, "a", (0, 1), "text", (10, 10, 30, 30)),
        _inline_atom(line, "atom-x", 1, "X", (1, 2), "formula", (30, 10, 50, 30)),
        _inline_atom(line, "atom-b", 2, "b", (2, 3), "text", (50, 10, 70, 30)),
    )
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text="aXb", line_kind="text", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    # 含 inline formula 的文本行保持 text 行
    assert widget.row.kind == "text"
    widget._image.resize(400, 58)
    widget._refresh_image()
    overlays = widget.editor.atom_visual_overlays()
    assert len(overlays) == 1
    assert (overlays[0].start, overlays[0].end) == (1, 2)
    assert overlays[0].kind == "formula"
    panel.close()


def test_hproof_multiple_inline_formulas_render_independently(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    atoms = (
        _inline_atom(line, "atom-a", 0, "a", (0, 1), "text", (10, 10, 30, 30)),
        _inline_atom(line, "atom-x", 1, "X", (1, 2), "formula", (30, 10, 50, 30)),
        _inline_atom(line, "atom-b", 2, "b", (2, 3), "text", (50, 10, 70, 30)),
        _inline_atom(line, "atom-y", 3, "Y", (3, 4), "formula", (70, 10, 90, 30)),
        _inline_atom(line, "atom-c", 4, "c", (4, 5), "text", (90, 10, 110, 30)),
    )
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text="aXbYc", line_kind="text", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget.row.kind == "text"
    widget._image.resize(600, 58)
    widget._refresh_image()
    overlays = widget.editor.atom_visual_overlays()
    assert [(item.start, item.end) for item in overlays] == [(1, 2), (3, 4)]
    panel.close()


def test_hproof_display_formula_uses_three_row_presentation(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import (
        FORMULA_IMAGE_ROW_H,
        FORMULA_RENDER_AREA_H,
        FORMULA_RENDER_TARGET_H,
        NEAR_LINE_PAIR_MAX_H,
        NEAR_LINE_PAIR_MIN_H,
        HProofPanel,
        _FormulaLineSourceEdit,
    )

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    text = r"$E=mc^2$"
    atoms = (_inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30)),)
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget.row.kind == "formula"
    # 默认双行：crop + 渲染。源码第三栏只由右键显式打开。
    assert isinstance(widget.editor, _FormulaLineSourceEdit)
    assert widget.editor.objectName() == "formulaSourceEdit"
    assert widget.editor.isHidden()
    assert widget._formula_source_panel is None
    assert not widget._formula_render_area.isHidden()
    assert FORMULA_IMAGE_ROW_H == 96
    assert FORMULA_RENDER_TARGET_H == 72
    assert FORMULA_IMAGE_ROW_H > FORMULA_RENDER_TARGET_H
    assert widget._image.height() == FORMULA_IMAGE_ROW_H
    assert widget._formula_render_area.height() == FORMULA_RENDER_AREA_H
    has_render = (
        not widget._formula_render_label.pixmap().isNull()
        or bool(widget._formula_render_label.text())
    )
    assert has_render
    widget._open_standalone_formula_editor(QPoint(20, 20))
    assert widget._formula_source_panel is not None
    assert not widget._formula_source_panel.isHidden()
    assert widget.editor.isHidden()
    widget.set_focus_depth("near")
    assert widget._formula_render_area.isHidden()
    assert widget.minimumHeight() == NEAR_LINE_PAIR_MIN_H
    assert widget.maximumHeight() == NEAR_LINE_PAIR_MAX_H
    panel.close()


def test_hproof_formula_source_panel_forwards_row_shortcuts(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    unit = state.text_units[0]
    line = state.lines[0]
    text = r"$E=mc^2$"
    atoms = (
        _inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30)),
    )
    panel = HProofPanel(
        _formula_workspace(
            workspace,
            unit,
            line,
            state,
            text=text,
            line_kind="formula",
            atoms=atoms,
        )
    )
    widget = panel._row_widgets[("proof-1", "unit-1")]
    widget._open_standalone_formula_editor(QPoint(20, 20))
    source_panel = widget._formula_source_panel
    assert source_panel is not None
    edits: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(edits.append)
    source_panel._source_edit.setPlainText(r"$E=mc^3$")
    QTest.keyClick(
        source_panel._source_edit,
        Qt.Key.Key_S,
        Qt.KeyboardModifier.ControlModifier,
    )
    qapp.processEvents()
    assert edits[-1].op == "replace_many"
    assert edits[-1].replacements == (("unit-1", r"$E=mc^3$"),)

    navigation: list[int] = []
    widget.navigate_requested.connect(navigation.append)
    QTest.keyClick(source_panel._source_edit, Qt.Key.Key_Tab)
    assert navigation == [1]
    assert source_panel.isHidden()
    panel.close()


def test_formula_number_link_is_soft_and_hproof_groups_its_visuals(qapp, tmp_path, monkeypatch) -> None:
    import app.ui.proof.h_proof as h_proof
    from app.application.proof_workspace import FormulaNumberLinkView

    monkeypatch.setattr(
        h_proof,
        "_render_formula_visual",
        lambda text, target_height: h_proof._FormulaVisual(text=f"render:{text}"),
    )
    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    unit = state.text_units[0]
    line = state.lines[0]
    formula_text = r"$E=mc^2$"
    formula_unit = replace(unit, text=formula_text)
    formula_line = replace(
        line,
        ocr_text=formula_text,
        proof_text=formula_text,
        bbox=(10, 10, 90, 30),
        render_kind="formula",
        region_kinds=("display_formula",),
        atoms=(),
    )
    number_unit = replace(unit, text_unit_uid="unit-number", order=1, text="(2)")
    number_line = replace(
        line,
        line_uid="line-number",
        region_uid="region-number",
        line_uids=("line-number",),
        region_uids=("region-number",),
        text_unit_uid="unit-number",
        order=1,
        ocr_text="(2)",
        proof_text="(2)",
        bbox=(100, 10, 120, 30),
        render_kind="text",
        region_kinds=("formula_number",),
        atoms=(),
        text_unit_revision=number_unit.revision,
        text_unit_fingerprint=number_unit.fingerprint,
    )
    next_state = replace(
        state,
        text_units=(formula_unit, number_unit),
        lines=(formula_line, number_line),
    )
    link = FormulaNumberLinkView(
        proof_uid=state.proof_uid,
        page_uid=state.page_uid,
        formula_text_unit_uid=formula_unit.text_unit_uid,
        number_text_unit_uid=number_unit.text_unit_uid,
    )
    linked_workspace = replace(
        workspace,
        proof_states=(next_state,),
        lines=(formula_line, number_line),
        formula_number_links=(link,),
    )

    panel = h_proof.HProofPanel(linked_workspace)
    assert [row.unit.text_unit_uid for row in panel._rows] == ["unit-1"]
    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget.row.formula_number_line == number_line
    assert widget._line_bbox == (10, 10, 120, 30)
    assert widget._formula_render_label.text() == r"render:E=mc^2 \qquad (2)"
    assert widget.editor.toPlainText() == formula_text
    panel.close()


def test_formula_preview_replaces_source_tag_with_linked_number() -> None:
    from app.ui.proof.h_proof import _formula_preview_source

    assert _formula_preview_source(r"$$x^2 \tag{old}$$", "（7）") == (
        r"x^2 \qquad (7)"
    )


def test_hproof_page_directory_can_collapse_and_restore(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    panel.show()
    qapp.processEvents()

    panel._toggle_page_directory()
    assert panel._left_pane.maximumWidth() == 56
    assert panel._page_directory.isHidden()

    panel._toggle_page_directory()
    assert panel._left_pane.minimumWidth() == 210
    assert panel._left_pane.maximumWidth() == 270
    assert not panel._page_directory.isHidden()
    panel.close()


def test_hproof_discards_formula_result_after_source_changes(
    qapp,
    tmp_path,
    monkeypatch,
) -> None:
    import app.ui.proof.h_proof as h_proof

    session, _service = _session(tmp_path)
    panel = h_proof.HProofPanel(build_proof_workspace_view(session))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    stale_hash = h_proof.formula_source_hash("old formula")
    widget._formula_request_targets[stale_hash].add(34)
    monkeypatch.setattr(
        h_proof,
        "materialize_formula_preview",
        lambda *_args, **_kwargs: pytest.fail("stale payload was materialized"),
    )

    widget._on_formula_preview_completed(
        h_proof.FormulaPreviewResult(source_hash=stale_hash, payload=object())
    )

    assert stale_hash not in widget._formula_request_targets
    panel.close()


def test_hproof_whole_row_surfaces_activate_and_focus_editor(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    panel.show()
    qapp.processEvents()
    widget = panel._row_widgets[("proof-1", "unit-1")]
    activations: list[bool] = []
    widget.activated.connect(lambda: activations.append(True))

    for surface in (widget._title, widget._status, widget._active_bar):
        widget.editor.clearFocus()
        activations.clear()
        QTest.mouseClick(surface, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert activations
        assert widget.editor.hasFocus()

    widget.editor.clearFocus()
    activations.clear()
    widget._displayed_pixmap_size.setWidth(1)
    widget._displayed_pixmap_size.setHeight(1)
    widget._on_image_clicked(QPoint(widget._image.width() - 1, 0))
    assert activations
    assert widget.editor.hasFocus()
    panel.close()


def test_hproof_resolves_image_hit_before_focus_rescales_row(
    qapp,
    tmp_path,
    monkeypatch,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    target = widget.row.entries[1]
    order: list[str] = []

    def resolve(_point):
        order.append("resolve")
        return target

    def activate():
        order.append("activate")
        widget._displayed_pixmap_size.setWidth(999)

    monkeypatch.setattr(widget, "_entry_at_image_point", resolve)
    monkeypatch.setattr(widget, "_activate_from_pointer", activate)

    widget._on_image_clicked(QPoint(1, 1))

    assert order == ["resolve", "activate"]
    assert widget.editor.textCursor().selectionStart() == target.char_index
    panel.close()


def test_hproof_text_image_scale_stays_stable_across_focus_depths(
    qapp,
    tmp_path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel, IMAGE_ROW_H

    session, _service = _session(tmp_path)
    panel = HProofPanel(build_proof_workspace_view(session))
    widget = panel._row_widgets[("proof-1", "unit-1")]

    widget.set_focus_depth("far")
    far_size = QSize(widget._displayed_pixmap_size)
    widget.set_focus_depth("near")
    near_size = QSize(widget._displayed_pixmap_size)
    widget.set_focus_depth("active")
    active_size = QSize(widget._displayed_pixmap_size)

    assert widget._image.height() == IMAGE_ROW_H
    assert far_size == near_size == active_size
    panel.close()


def test_hproof_recreates_editor_when_row_kind_changes(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel, _FormulaLineSourceEdit

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    unit = state.text_units[0]
    line = state.lines[0]
    initial = _formula_workspace(
        workspace,
        unit,
        line,
        state,
        text="ab",
        line_kind="text",
        atoms=line.atoms,
    )
    panel = HProofPanel(initial)
    old_widget = panel._row_widgets[("proof-1", "unit-1")]
    assert not isinstance(old_widget.editor, _FormulaLineSourceEdit)

    formula_text = r"$E=mc^2$"
    atom = _inline_atom(
        line,
        "atom-f",
        0,
        formula_text,
        (0, len(formula_text)),
        "formula",
        (10, 10, 150, 30),
    )
    panel.set_workspace(
        _formula_workspace(
            workspace,
            unit,
            line,
            state,
            text=formula_text,
            line_kind="formula",
            atoms=(atom,),
        )
    )

    new_widget = panel._row_widgets[("proof-1", "unit-1")]
    assert new_widget is not old_widget
    assert isinstance(new_widget.editor, _FormulaLineSourceEdit)
    panel.close()


def test_hproof_dirty_formula_rebuild_refreshes_render(qapp, tmp_path, monkeypatch) -> None:
    import app.ui.proof.h_proof as h_proof

    monkeypatch.setattr(
        h_proof,
        "_render_formula_visual",
        lambda text, target_height: h_proof._FormulaVisual(text=f"render:{text}"),
    )
    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    unit = state.text_units[0]
    line = state.lines[0]
    text = r"$E=mc^2$"
    atom = _inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30))
    panel = h_proof.HProofPanel(
        _formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=(atom,))
    )
    key = ("proof-1", "unit-1")
    panel._dirty_text[key] = "NEW_FORMULA"
    panel._render_mode = "uninitialized"
    panel._render_rows()

    widget = panel._row_widgets[key]
    assert widget.editor.toPlainText() == "NEW_FORMULA"
    assert widget._formula_render_label.text() == "render:NEW_FORMULA"
    panel.close()


def test_hproof_long_formula_render_uses_horizontal_scroll(qapp, tmp_path, monkeypatch) -> None:
    import app.ui.proof.h_proof as h_proof

    pixmap = QPixmap(1200, 40)
    pixmap.fill(QColor("black"))
    monkeypatch.setattr(
        h_proof,
        "_render_formula_visual",
        lambda text, target_height: h_proof._FormulaVisual(
            text=None,
            pixmap=pixmap,
            logical_size=pixmap.size(),
        ),
    )
    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    unit = state.text_units[0]
    line = state.lines[0]
    text = r"$E=mc^2$"
    atom = _inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30))
    panel = h_proof.HProofPanel(
        _formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=(atom,))
    )
    panel.resize(800, 600)
    panel.show()
    qapp.processEvents()

    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget._formula_render_label.width() >= pixmap.width()
    assert widget._formula_render_area.horizontalScrollBar().maximum() > 0
    panel.close()


def test_hproof_formula_source_edit_commits_once(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    text = r"$E=mc^2$"
    atoms = (_inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30)),)
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=atoms))
    commands = []
    panel.proof_edit_requested.connect(commands.append)
    widget = panel._row_widgets[("proof-1", "unit-1")]
    widget.editor.setPlainText(r"$E=mc^3$")
    assert panel.save() is True
    assert len(commands) == 1
    assert commands[0].op == "replace_many"
    assert commands[0].replacements == (("unit-1", r"$E=mc^3$"),)
    # 权威文本仍是会话存储，UI 只发命令不落地
    assert session.proof_repository.get_state("proof-1").text_units[0].text == "ab"
    panel.close()


def test_hproof_formula_geometry_unavailable_shows_source(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel, _FormulaLineSourceEdit

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    text = r"$E=mc^2$"
    atoms = (_inline_atom(line, "atom-f", 0, text, None, "formula", (10, 10, 150, 30)),)
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    # 几何不可用：不伪造覆盖层，源码编辑保持可用
    assert isinstance(widget.editor, _FormulaLineSourceEdit)
    assert not widget.editor.isReadOnly()
    panel.close()


def test_hproof_inline_overlay_hidden_for_missing_span(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    atoms = (
        _inline_atom(line, "atom-a", 0, "a", (0, 1), "text", (10, 10, 30, 30)),
        _inline_atom(line, "atom-x", 1, "X", None, "formula", (30, 10, 50, 30)),
    )
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text="aXb", line_kind="text", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    widget._image.resize(400, 58)
    widget._refresh_image()
    assert widget.editor.atom_visual_overlays() == []
    panel.close()


def test_hproof_formula_edit_then_history_and_external_conflict(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    text = r"$E=mc^2$"
    atoms = (_inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30)),)
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=atoms))
    commands = []
    panel.proof_edit_requested.connect(commands.append)
    row = panel._rows[0]
    widget = panel._row_widgets[row.key]

    widget.editor.setPlainText(r"$E=mc^3$")
    panel._apply_history(row, -1)
    panel._apply_history(row, 1)
    assert [command.op for command in commands] == ["undo", "redo"]

    # 外部变更 + 本地 dirty → 冲突而不是覆盖
    current = service.get_state("proof-1")
    service.replace_text(
        "proof-1", "unit-1", r"$E=mc^4$",
        expected_revision=current.revision,
        expected_fingerprint=current.fingerprint,
    )
    panel.set_workspace(build_proof_workspace_view(session))
    widget2 = panel._row_widgets[row.key]
    assert row.key in panel._conflict_keys
    assert widget2.editor.toPlainText() == r"$E=mc^3$"
    before = len(commands)
    panel._commit_row(row, widget2)
    assert len(commands) == before  # 冲突行拒写
    panel.close()


def test_hproof_table_and_plain_rows_keep_current_behavior(qapp, tmp_path) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]

    table_atoms = (_inline_atom(line, "atom-t", 0, "a|b", (0, 3), "table", (10, 10, 70, 30)),)
    table_panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text="a|b", line_kind="table", atoms=table_atoms))
    table_widget = table_panel._row_widgets[("proof-1", "unit-1")]
    assert table_widget.row.kind == "table"
    assert not table_widget._line_crop.isNull()
    assert table_widget._formula_render_area.isHidden()
    table_panel.close()

    plain_panel = HProofPanel(build_proof_workspace_view(session))
    plain_widget = plain_panel._row_widgets[("proof-1", "unit-1")]
    assert plain_widget.row.kind == "text"
    assert plain_widget._formula_render_area.isHidden()
    plain_panel.close()


def test_hproof_page_directory_uses_proof_page_thumbnail_cards(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    panel = HProofPanel(workspace)

    assert panel._page_directory.count() == 1
    item = panel._page_directory.item(0)
    card = panel._page_directory.itemWidget(item)
    assert card is panel._page_cards["page-1"]
    assert card.page is workspace.pages[0]
    assert card._page_number.text() == "第 1 页"
    assert card._filename.text() == "source.pdf"
    assert not card._source_pixmap.isNull()
    assert card.property("selected") is True
    panel.close()


def test_char_views_use_explicit_char_span_without_sequential_guessing(qapp, tmp_path) -> None:
    """Review regression: a missing span must not shift later char mappings."""

    from app.ui.proof.confidence_view import build_char_views

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    line = state.lines[0]
    first_atom = replace(line.atoms[0], char_span=None, geometry_available=False)
    second_atom = replace(line.atoms[1], char_span=(1, 2), geometry_available=True)
    next_line = replace(line, atoms=(first_atom, second_atom))
    entries = build_char_views(next_line, workspace.pages[0])

    assert entries[0].atom_uid is None
    assert entries[0].available is False
    assert entries[1].atom_uid == "atom-2"
    assert entries[1].bbox == (30, 10, 50, 30)


def test_vproof_gallery_crop_padding_depends_only_on_character_geometry(
    qapp,
    tmp_path,
) -> None:
    from app.ui.proof.confidence_view import build_char_views
    from app.ui.proof.v_proof import _gallery_crop_pad

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    line = workspace.proof_states[0].lines[0]
    punctuation_atom = replace(
        line.atoms[0],
        text="，",
        token_text="，",
        bbox=(18, 24, 24, 29),
        char_span=(0, 1),
        geometry_available=True,
    )
    punctuation_line = replace(
        line,
        proof_text="，b",
        bbox=(10, 8, 50, 32),
        atoms=(punctuation_atom, line.atoms[1]),
    )
    entry = build_char_views(punctuation_line, workspace.pages[0])[0]

    assert entry.bbox == (18, 24, 24, 29)
    assert _gallery_crop_pad(entry) == 2


def test_char_views_keep_one_word_bbox_as_one_range_occurrence(
    qapp,
    tmp_path,
) -> None:
    """A word carrier owns one exact word crop, never guessed char crops."""

    from app.ui.proof.confidence_view import build_char_views

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    line = workspace.proof_states[0].lines[0]
    word_atom = replace(
        line.atoms[0],
        text="ab",
        token_text="ab",
        granularity="word",
        char_span=(0, 2),
        geometry_available=True,
    )
    entries = build_char_views(
        replace(line, atoms=(word_atom,)),
        workspace.pages[0],
    )

    assert len(entries) == 1
    assert entries[0].text == "ab"
    assert entries[0].ocr_char == "ab"
    assert entries[0].atom_uid == word_atom.atom_uid
    assert (entries[0].char_index, entries[0].char_end) == (0, 2)
    assert entries[0].bbox == word_atom.bbox
    assert entries[0].available is True


def test_hproof_word_carrier_draws_and_selects_one_word_occurrence(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    line = state.lines[0]
    word_atom = replace(
        line.atoms[0],
        text="ab",
        token_text="ab",
        granularity="word",
        char_span=(0, 2),
        geometry_available=True,
    )
    word_line = replace(line, atoms=(word_atom,))
    word_state = replace(state, lines=(word_line,))
    panel = HProofPanel(replace(
        workspace,
        proof_states=(word_state,),
        lines=(word_line,),
    ))
    panel.resize(900, 500)
    panel.show()
    qapp.processEvents()
    row = panel._row_widgets[(state.proof_uid, line.text_unit_uid)]
    row._refresh_editor_geometry()
    overlays = row.editor.atom_visual_overlays()

    assert row._proof_char_spans() == {0: (10.0, 30.0), 1: (10.0, 30.0)}
    assert len(overlays) == 1
    assert (overlays[0].kind, overlays[0].text) == ("word", "ab")
    assert (overlays[0].start, overlays[0].end) == (0, 2)

    click_x = round((overlays[0].left + overlays[0].right) / 2.0)
    QTest.mouseClick(
        row.editor,
        Qt.MouseButton.LeftButton,
        pos=QPoint(click_x, row.editor.height() // 2),
    )
    assert row.editor.textCursor().selectedText() == "ab"
    QTest.mouseDClick(
        row.editor,
        Qt.MouseButton.LeftButton,
        pos=QPoint(click_x, row.editor.height() // 2),
    )
    assert row.editor._expanded_edit.text() == "ab"
    panel.close()


def test_vproof_word_carrier_renders_one_word_crop_and_replaces_its_range(
    qapp,
    tmp_path,
) -> None:
    from app.ui.proof.confidence_view import build_char_views
    from app.ui.proof.v_proof import VProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    line = workspace.proof_states[0].lines[0]
    word_atom = replace(
        line.atoms[0],
        text="ab",
        token_text="ab",
        granularity="word",
        char_span=(0, 2),
        geometry_available=True,
    )
    word_line = replace(line, atoms=(word_atom,))
    state = workspace.proof_states[0]
    word_state = replace(state, lines=(word_line,))
    word_workspace = replace(
        workspace,
        proof_states=(word_state,),
        lines=(word_line,),
    )
    entry = build_char_views(word_line, workspace.pages[0])[0]
    panel = VProofPanel(word_workspace)
    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)

    assert entry.text == "ab"
    assert entry.bbox == word_atom.bbox
    assert panel._entries == (entry,)
    assert not panel._entry_icon(entry).isNull()
    assert panel._apply_replacement_to_selected("word") == 1
    assert commands[0].replacements == (("unit-1", "word"),)
    panel.close()


def test_hproof_does_not_invent_atom_mapping_without_explicit_span(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state = workspace.proof_states[0]
    line = state.lines[0]
    unit = state.text_units[0]
    first_atom = replace(line.atoms[0], char_span=(0, 1), geometry_available=True)
    second_atom = replace(line.atoms[1], char_span=None, geometry_available=False)
    next_line = replace(
        line,
        ocr_text="a",
        proof_text="a",
        bbox=None,
        atoms=(first_atom, second_atom),
    )
    next_state = replace(
        state,
        text_units=(replace(unit, text="a"),),
        lines=(next_line,),
    )
    panel = HProofPanel(replace(workspace, proof_states=(next_state,), lines=(next_line,)))
    row = panel._rows[0]

    assert row.entries[0].atom_uid == "atom-1"
    assert row.entries[0].bbox == (10, 10, 30, 30)
    assert row.atom_placements[1].char_indices == ()
    panel.close()


def test_vproof_index_edit_uses_stable_entry_ids_and_emits_command(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    panel = VProofPanel(workspace)
    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)
    assert panel.objectName() == "proofRoot"
    assert panel._content_splitter.objectName() == "proofContentSplitter"
    assert panel._char_list.objectName() == "charIndexList"
    assert panel._char_list.parentWidget().objectName() == "proofLeftPane"
    assert panel._gallery.parentWidget().objectName() == "proofCard"
    assert panel._image.parentWidget().objectName() == "proofRightPane"
    assert panel._char_list.count() == 2
    assert panel._selected_entry is not None
    selected_key = (
        panel._selected_entry.proof_uid,
        panel._selected_entry.text_unit_uid,
        panel._selected_entry.char_index,
        panel._selected_entry.atom_uid,
    )
    assert panel._gallery_direct_overwrite("x") is True
    assert len(commands) == 1
    assert commands[0].op == "replace_many"
    assert commands[0].replacements == (("unit-1", "xb"),)
    assert selected_key[0:2] == ("proof-1", "unit-1")
    assert session.proof_repository.get_state("proof-1").text_units[0].text == "ab"
    assert session.ocr_observation_repository.get_atom("atom-1").text == "a"
    panel.close()


def test_vproof_adjacent_toggle_preserves_current_occurrence(qapp, tmp_path) -> None:
    from app.ui.proof.v_proof import VProofPanel

    session, _service = _session(tmp_path)
    panel = VProofPanel(build_proof_workspace_view(session))
    panel._gallery.clear()
    for index in range(3):
        panel._gallery.addItem(f"item-{index}")
    panel._gallery.setCurrentRow(1)

    panel._toggle_adjacent_gallery(1)

    assert panel._gallery.currentRow() == 1
    assert panel._gallery.item(2).isSelected()
    panel._toggle_adjacent_gallery(1)
    assert not panel._gallery.item(2).isSelected()
    panel.close()


def test_vproof_cross_state_selection_emits_one_atomic_batch_command(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel

    session, _service = _session(tmp_path)
    panel = VProofPanel(build_proof_workspace_view(session))
    first_entry = panel._selected_entry
    assert first_entry is not None
    first_state = panel._states[first_entry.proof_uid]
    first_unit = panel._units[(first_entry.proof_uid, first_entry.text_unit_uid)]
    second_unit = replace(
        first_unit,
        proof_uid="proof-2",
        text_unit_uid="unit-2",
        text=first_unit.text,
    )
    second_state = replace(
        first_state,
        proof_uid="proof-2",
        text_units=(second_unit,),
        lines=(),
    )
    second_entry = replace(
        first_entry,
        proof_uid="proof-2",
        text_unit_uid="unit-2",
    )
    panel._states[second_state.proof_uid] = second_state
    panel._units[(second_state.proof_uid, second_unit.text_unit_uid)] = second_unit
    second_item = QListWidgetItem("")
    second_item.setData(Qt.ItemDataRole.UserRole, second_entry)
    panel._gallery.addItem(second_item)
    panel._gallery.item(0).setSelected(True)
    second_item.setSelected(True)

    commands: list[object] = []
    panel.proof_edit_requested.connect(commands.append)
    assert panel._apply_replacement_to_selected("x") == 2
    assert len(commands) == 1
    assert isinstance(commands[0], ProofBatchEditCommand)
    assert tuple(item.proof_uid for item in commands[0].commands) == (
        "proof-1",
        "proof-2",
    )
    panel.close()


def test_vproof_edit_bubble_and_history_emit_only_commands(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel

    session, _service = _session(tmp_path)
    panel = VProofPanel(build_proof_workspace_view(session))
    commands: list[ProofEditCommand] = []
    panel.proof_edit_requested.connect(commands.append)
    panel.show()
    qapp.processEvents()
    item = panel._gallery.currentItem()
    assert item is not None
    position = panel._gallery.visualItemRect(item).center()
    panel._show_edit_bubble_at(position)
    assert panel._edit_bubble.isVisible() is True
    panel._edit_bubble_input.setText("z")
    panel._apply_edit_bubble()
    assert panel._edit_bubble.isVisible() is False
    panel._apply_history(-1)
    panel._apply_history(1)
    assert [command.op for command in commands] == ["replace_many", "undo", "redo"]
    assert session.proof_repository.get_state("proof-1").text_units[0].text == "ab"
    panel.close()


def test_quality_stats_reads_only_workspace_snapshot(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.quality_stats_dialog import QualityStatsDialog

    session, service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    dialog = QualityStatsDialog(workspace)
    assert dialog.stats.character_count == 2
    assert dialog.stats.checked_character_count == 0
    assert dialog.detail_table().rowCount() == 1

    current = service.get_state("proof-1")
    service.set_status(
        "proof-1",
        "unit-1",
        "checked",
        expected_revision=current.revision,
        expected_fingerprint=current.fingerprint,
    )
    assert dialog.stats.checked_character_count == 0
    dialog.set_workspace(build_proof_workspace_view(session))
    assert dialog.stats.checked_character_count == 2
    assert dialog.stats.checked_ratio == 1.0
    dialog.close()
