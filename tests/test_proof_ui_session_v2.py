from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QListWidgetItem

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
    from app.ui.proof.h_proof import HProofPanel, _FormulaLineSourceEdit

    session, _service = _session(tmp_path)
    workspace = build_proof_workspace_view(session)
    state, unit, line = workspace.proof_states[0], workspace.proof_states[0].text_units[0], workspace.proof_states[0].lines[0]
    text = r"$E=mc^2$"
    atoms = (_inline_atom(line, "atom-f", 0, text, (0, len(text)), "formula", (10, 10, 150, 30)),)
    panel = HProofPanel(_formula_workspace(workspace, unit, line, state, text=text, line_kind="formula", atoms=atoms))
    widget = panel._row_widgets[("proof-1", "unit-1")]
    assert widget.row.kind == "formula"
    # 三行：crop（图像）、渲染区、源码编辑器
    assert isinstance(widget.editor, _FormulaLineSourceEdit)
    assert widget.editor.objectName() == "formulaSourceEdit"
    assert not widget._formula_render_area.isHidden()
    has_render = (
        not widget._formula_render_label.pixmap().isNull()
        or bool(widget._formula_render_label.text())
    )
    assert has_render
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


def test_char_views_do_not_reuse_one_word_bbox_as_multiple_character_crops(
    qapp,
    tmp_path,
) -> None:
    """A word carrier owns text, but cannot provide exact per-char geometry."""

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

    assert [entry.text for entry in entries] == ["a", "b"]
    assert [entry.ocr_char for entry in entries] == ["a", "b"]
    assert all(entry.atom_uid == word_atom.atom_uid for entry in entries)
    assert all(entry.bbox is None for entry in entries)
    assert all(entry.available is False for entry in entries)


def test_vproof_word_carrier_does_not_render_a_fake_character_crop(
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
    entry = build_char_views(
        replace(line, atoms=(word_atom,)),
        workspace.pages[0],
    )[0]
    panel = VProofPanel(workspace)

    assert entry.bbox is None
    assert panel._entry_icon(entry).isNull()
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
