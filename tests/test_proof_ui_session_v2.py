from __future__ import annotations

import ast
from pathlib import Path

import pytest

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from app.application.proof_workspace import build_proof_workspace_view
from app.application.contracts import ProofEditCommand
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
