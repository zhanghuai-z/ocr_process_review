from __future__ import annotations

import ast
from pathlib import Path

import pytest

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

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
    ROOT / "app/ui/proof/confidence_utils.py",
    ROOT / "app/ui/proof/char_verdict.py",
    ROOT / "app/ui/widgets/quality_stats_dialog.py",
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
    session.layout_repository.put(
        LayoutSnapshot(
            page_uid="page-1",
            revision=1,
            artifact_uid="layout-1",
            source_engine="test",
            source_run_id="layout-run-1",
            blocks=(),
        ),
        expected_revision=0,
    )
    ocr = session.ocr_observation_repository
    run = ocr.append_run(
        OcrRun(
            project_uid=project_uid,
            uid="run-1",
            engine="test",
            layout_fingerprint="layout-1",
            input_fingerprint="input-1",
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


def test_owned_proof_ui_has_no_legacy_model_or_bus_dependencies() -> None:
    forbidden_modules = {
        "app.core.proof_projection",
        "app.core.proof_state_bus",
        "app.services.proof_edit_service",
        "app.services.proof_hproof_session",
        "app.services.proof_occurrence_session",
        "app.services.proof_probe_text_service",
    }
    forbidden_names = {"OcrProject", "ProofStateBus"}
    for path in OWNED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden_modules
                assert not {alias.name for alias in node.names} & forbidden_names
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names


def test_hproof_commits_text_with_service_cas_and_preserves_ocr(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, service = _session(tmp_path)
    panel = HProofPanel(session=session, proof_service=service)
    assert panel.objectName() == "proofRoot"
    assert panel._toolbar.objectName() == "proofToolbar"
    assert panel._scroll.objectName() == "proofScroll"
    assert panel._rows_root.objectName() == "proofLineList"
    assert len(panel._rows) == 1
    editor = panel._row_widgets[("proof-1", "unit-1")].editor
    editor.setPlainText("ax")
    assert panel.save() is True
    assert service.get_state("proof-1").text_units[0].text == "ax"
    assert session.ocr_observation_repository.get_line("line-1").text == "ab"
    panel.close()


def test_hproof_does_not_overwrite_an_external_cas_update(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.h_proof import HProofPanel

    session, service = _session(tmp_path)
    panel = HProofPanel(session=session, proof_service=service)
    editor = panel._row_widgets[("proof-1", "unit-1")].editor
    current = service.get_state("proof-1")
    service.replace_text(
        "proof-1",
        "unit-1",
        "external",
        expected_revision=current.revision,
        expected_fingerprint=current.fingerprint,
    )
    editor.setPlainText("local")
    assert panel.save() is False
    assert service.get_state("proof-1").text_units[0].text == "external"
    panel.close()


def test_vproof_index_edit_uses_stable_entry_ids_and_service_cas(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.proof.v_proof import VProofPanel

    session, service = _session(tmp_path)
    panel = VProofPanel(session=session, proof_service=service)
    assert panel.objectName() == "proofRoot"
    assert panel._toolbar.objectName() == "proofToolbar"
    assert panel._char_list.objectName() == "charIndexList"
    assert panel._char_list.parentWidget().objectName() == "proofLeftPane"
    assert panel._gallery.parentWidget().objectName() == "proofRightPane"
    assert panel._char_list.count() == 2
    assert panel._selected_entry is not None
    selected_key = (
        panel._selected_entry.proof_uid,
        panel._selected_entry.text_unit_uid,
        panel._selected_entry.char_index,
        panel._selected_entry.atom_uid,
    )
    assert panel._gallery_direct_overwrite("x") is True
    assert service.get_state("proof-1").text_units[0].text == "xb"
    assert selected_key[0:2] == ("proof-1", "unit-1")
    assert session.ocr_observation_repository.get_atom("atom-1").text == "a"
    panel.close()


def test_quality_stats_reads_proof_states_only(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    from app.ui.widgets.quality_stats_dialog import QualityStatsDialog

    session, service = _session(tmp_path)
    dialog = QualityStatsDialog(session_provider=lambda: session)
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
    dialog._on_manual_refresh()
    assert dialog.stats.checked_character_count == 2
    assert dialog.stats.checked_ratio == 1.0
    dialog.close()
