from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from app.core.project_store import ProjectDataError, ProjectStore
from app.models import (
    BlockAlignmentAuditSummary,
    BlockVlObservationSummary,
    OcrProject,
    OcrRoutingRunAudit,
    Page,
    RouteAuditSummary,
    RoutingAuditMessage,
)
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection


def _save_project(store: ProjectStore, *, name: str = "audit-project") -> OcrProject:
    page = Page(image_path="/tmp/routing-audit.png", width=1200, height=1800)
    sync_page_layout_snapshot_from_projection(page, source_engine="test")
    project = OcrProject(name=name, pages=[page])
    return store.save_project(project)


def _audit(project: OcrProject, *, run_uid: str, created_at: float) -> OcrRoutingRunAudit:
    page = project.pages[0]
    assert project.id is not None
    return OcrRoutingRunAudit(
        run_uid=run_uid,
        project_id=project.id,
        page_uid=page.uid,
        image_hash="sha256:image",
        layout_fingerprint="sha256:layout",
        pp_run_id="pp-run-7",
        block_vl_observations=(
            BlockVlObservationSummary(
                block_uid="block_A",
                status="exact",
                observation_uid="vlobs_A",
                text_length=18,
                record_count=1,
                raw_refs=("rawocr_A#record=0", "debug/vl-A.json"),
            ),
        ),
        alignment_statuses=(
            BlockAlignmentAuditSummary(
                block_uid="block_A",
                status="exact",
                matched_span_count=2,
            ),
        ),
        route_summary=RouteAuditSummary(
            block_count=1,
            line_count=2,
            segment_count=3,
            is_dispatchable=False,
            validation_issues=(
                RoutingAuditMessage(
                    code="ambiguous_prefix",
                    message="prefix boundary crosses one component",
                    block_uid="block_A",
                    line_index=1,
                    bbox=(10, 20, 30, 40),
                ),
            ),
            diagnostics=(
                RoutingAuditMessage(code="vl_empty_retry", message="VL retry returned empty"),
            ),
        ),
        created_at=created_at,
    )


def test_ocr_routing_run_audit_is_immutable_and_serializable(tmp_path):
    with ProjectStore(str(tmp_path / "audit.ocrproj")) as store:
        project = _save_project(store)
        audit = _audit(project, run_uid="ocrrun_A", created_at=10.0)

    restored = OcrRoutingRunAudit.from_dict(audit.to_dict())

    assert restored == audit
    assert restored.block_vl_observations[0].raw_refs == (
        "rawocr_A#record=0",
        "debug/vl-A.json",
    )
    with pytest.raises(FrozenInstanceError):
        audit.page_uid = "page_rewritten"  # type: ignore[misc]


def test_project_store_appends_and_retains_multiple_routing_runs(tmp_path):
    db_path = tmp_path / "audit.ocrproj"
    with ProjectStore(str(db_path)) as store:
        project = _save_project(store)
        first = _audit(project, run_uid="ocrrun_first", created_at=10.0)
        second = _audit(project, run_uid="ocrrun_second", created_at=20.0)

        store.save_ocr_routing_run_audit(first)
        store.save_ocr_routing_run_audit(second)
        store.save_project(project)

        loaded = store.load_ocr_routing_run_audit(project.id, first.run_uid)
        page_runs = store.list_ocr_routing_run_audits(
            project.id,
            page_uid=project.pages[0].uid,
        )

    assert loaded == first
    assert page_runs == [first, second]
    assert not hasattr(project.pages[0], "ocr_routing_run_audits")


def test_project_store_rejects_duplicate_run_uid_and_foreign_page(tmp_path):
    with ProjectStore(str(tmp_path / "audit.ocrproj")) as store:
        project = _save_project(store)
        audit = _audit(project, run_uid="ocrrun_duplicate", created_at=10.0)
        store.save_ocr_routing_run_audit(audit)

        with pytest.raises(ProjectDataError, match="run_uid already exists"):
            store.save_ocr_routing_run_audit(audit)

        foreign = OcrRoutingRunAudit.from_dict({
            **audit.to_dict(),
            "run_uid": "ocrrun_foreign",
            "page_uid": "page_not_in_project",
        })
        with pytest.raises(ProjectDataError, match="does not belong"):
            store.save_ocr_routing_run_audit(foreign)

    assert [item.run_uid for item in page_runs_from_file(tmp_path / "audit.ocrproj", project.id)] == [
        "ocrrun_duplicate"
    ]


def page_runs_from_file(path, project_id: int) -> list[OcrRoutingRunAudit]:
    with ProjectStore(str(path)) as store:
        return store.list_ocr_routing_run_audits(project_id)


def test_project_store_rejects_malformed_routing_audit_payload(tmp_path):
    db_path = tmp_path / "audit.ocrproj"
    with ProjectStore(str(db_path)) as store:
        project = _save_project(store)
        audit = _audit(project, run_uid="ocrrun_corrupt", created_at=10.0)
        store.save_ocr_routing_run_audit(audit)
        store.conn.execute(
            "UPDATE ocr_routing_run_audit SET alignment_statuses_json=? WHERE run_uid=?",
            ("{}", audit.run_uid),
        )
        store.conn.commit()

        with pytest.raises(ProjectDataError, match="alignment_statuses_json must be list"):
            store.load_ocr_routing_run_audit(project.id, audit.run_uid)


def test_schema_v25_has_independent_routing_audit_table(tmp_path):
    db_path = tmp_path / "audit.ocrproj"
    with ProjectStore(str(db_path)):
        pass

    with sqlite3.connect(db_path) as conn:
        schema_version = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(ocr_routing_run_audit)")
        }

    assert schema_version == "25"
    assert "ocr_routing_run_audit" in tables
    assert {
        "run_uid",
        "project_id",
        "page_uid",
        "image_hash",
        "layout_fingerprint",
        "pp_run_id",
        "block_vl_observations_json",
        "alignment_statuses_json",
        "route_summary_json",
        "created_at",
    } <= columns
