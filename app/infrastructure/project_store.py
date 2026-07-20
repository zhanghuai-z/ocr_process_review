"""Strict SQLite persistence for the ProjectSession format generation."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, TypeVar

from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrCandidate,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import (
    BindingRecord,
    LayoutRepository,
    PageRecord,
    ProjectRecord,
    ProjectSession,
    TableTextRecord,
)
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)


FORMAT_GENERATION = "ocr_process.project_session.v2"


class ProjectStoreError(RuntimeError):
    """Base class for the new project persistence boundary."""


class InvalidProjectFormatError(ProjectStoreError):
    """The file is not exactly this project format generation."""


class ProjectUidMismatchError(ProjectStoreError):
    """The file and requested session belong to different projects."""


class ProjectStoreDataError(ProjectStoreError):
    """Persisted rows violate the new record contracts."""


class ProjectStoreConflictError(ProjectStoreError):
    """A save would replace newer or conflicting persisted state."""


_SCHEMA_STATEMENTS = (
    f"""
    CREATE TABLE project_format (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        generation TEXT NOT NULL CHECK (generation = '{FORMAT_GENERATION}')
    )
    """,
    """
    CREATE TABLE project (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        project_uid TEXT NOT NULL UNIQUE,
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE page (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE layout_snapshot (
        page_uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE paddle_artifact (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_run (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_region (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_line (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_atom (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_candidate (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_batch (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE ocr_active_pointer (
        scope_uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE proof_state (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE binding (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE table_text (
        uid TEXT PRIMARY KEY,
        project_uid TEXT NOT NULL REFERENCES project(project_uid),
        revision INTEGER NOT NULL CHECK (revision > 0),
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
)

_REQUIRED_TABLES = frozenset(
    {
        "project_format",
        "project",
        "page",
        "layout_snapshot",
        "paddle_artifact",
        "ocr_run",
        "ocr_region",
        "ocr_line",
        "ocr_atom",
        "ocr_candidate",
        "ocr_batch",
        "ocr_active_pointer",
        "proof_state",
        "binding",
        "table_text",
    }
)


@dataclass(frozen=True)
class _SessionRecords:
    project: ProjectRecord
    pages: tuple[PageRecord, ...]
    layouts: tuple[LayoutSnapshot, ...]
    paddle_artifacts: tuple[PaddleArtifact, ...]
    ocr_runs: tuple[OcrRun, ...]
    ocr_regions: tuple[OcrRegion, ...]
    ocr_lines: tuple[OcrLine, ...]
    ocr_atoms: tuple[OcrAtom, ...]
    ocr_candidates: tuple[OcrCandidate, ...]
    ocr_batches: tuple[OcrBatch, ...]
    ocr_active_pointers: tuple[OcrActivePointer, ...]
    proof_states: tuple[ProofState, ...]
    bindings: tuple[BindingRecord, ...]
    table_texts: tuple[TableTextRecord, ...]

    def hydration_kwargs(self) -> dict[str, tuple[Any, ...]]:
        return {
            "pages": self.pages,
            "layouts": self.layouts,
            "paddle_artifacts": self.paddle_artifacts,
            "ocr_runs": self.ocr_runs,
            "ocr_regions": self.ocr_regions,
            "ocr_lines": self.ocr_lines,
            "ocr_atoms": self.ocr_atoms,
            "ocr_candidates": self.ocr_candidates,
            "ocr_batches": self.ocr_batches,
            "ocr_active_pointers": self.ocr_active_pointers,
            "proof_states": self.proof_states,
            "bindings": self.bindings,
            "table_texts": self.table_texts,
        }


def _snapshot_session(session: ProjectSession) -> _SessionRecords:
    if not isinstance(session, ProjectSession):
        raise TypeError("save_session requires ProjectSession")
    ocr = session.ocr_observation_repository
    proof = session.proof_repository
    records = _SessionRecords(
        project=session.project_record,
        pages=session.page_repository.all(),
        layouts=session.layout_repository.all(),
        paddle_artifacts=session.paddle_artifact_repository.all(),
        ocr_runs=ocr.all_runs(),
        ocr_regions=ocr.all_regions(),
        ocr_lines=ocr.all_lines(),
        ocr_atoms=ocr.all_atoms(),
        ocr_candidates=ocr.all_candidates(),
        ocr_batches=ocr.all_batches(),
        ocr_active_pointers=ocr.all_active_pointers(),
        proof_states=proof.all_states(),
        bindings=session.binding_repository.all(),
        table_texts=session.table_text_repository.all(),
    )
    ProjectSession.from_records(records.project, **records.hydration_kwargs())
    return records


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if item.name != "fingerprint"
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot persist value of type {type(value).__name__}")


def _payload(record: object) -> str:
    return json.dumps(
        _json_value(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_object(raw: object, context: str) -> dict[str, object]:
    if not isinstance(raw, str):
        raise ProjectStoreDataError(f"{context} payload must be text")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProjectStoreDataError(f"{context} payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProjectStoreDataError(f"{context} payload must be an object")
    return value


T = TypeVar("T")


def _construct(record_type: Callable[..., T], payload: dict[str, object], context: str) -> T:
    try:
        return record_type(**payload)
    except (TypeError, ValueError) as exc:
        raise ProjectStoreDataError(f"invalid {context} payload") from exc


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProjectStoreDataError(f"{context} must be an object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ProjectStoreDataError(f"{context} must be an array")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ProjectStoreDataError(f"{context} must be text")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectStoreDataError(f"{context} must be an integer")
    return value


def _optional_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context)


def _optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectStoreDataError(f"{context} must be a number or null")
    return float(value)


def _take(payload: dict[str, object], key: str, context: str) -> object:
    try:
        return payload.pop(key)
    except KeyError as exc:
        raise ProjectStoreDataError(f"{context} is missing {key!r}") from exc


def _reject_extra(payload: dict[str, object], context: str) -> None:
    if payload:
        raise ProjectStoreDataError(
            f"{context} contains unknown fields: {sorted(payload)}"
        )


def _decode_bbox(value: object, context: str) -> BBox:
    payload = dict(_object(value, context))
    bbox = BBox(
        x=_integer(_take(payload, "x", context), f"{context}.x"),
        y=_integer(_take(payload, "y", context), f"{context}.y"),
        w=_integer(_take(payload, "w", context), f"{context}.w"),
        h=_integer(_take(payload, "h", context), f"{context}.h"),
    )
    _reject_extra(payload, context)
    return bbox


def _decode_origin(value: object, context: str) -> BlockOrigin:
    payload = dict(_object(value, context))
    original_bbox_raw = _take(payload, "original_bbox", context)
    original_kind_raw = _take(payload, "original_kind", context)
    try:
        original_kind = (
            None if original_kind_raw is None else BlockType(_text(original_kind_raw, context))
        )
    except ValueError as exc:
        raise ProjectStoreDataError(f"{context}.original_kind is invalid") from exc
    origin = BlockOrigin(
        created_by=_text(_take(payload, "created_by", context), f"{context}.created_by"),
        source_engine=_text(_take(payload, "source_engine", context), f"{context}.source_engine"),
        source_run_id=_text(_take(payload, "source_run_id", context), f"{context}.source_run_id"),
        vendor_label=_text(_take(payload, "vendor_label", context), f"{context}.vendor_label"),
        source_confidence=_optional_number(
            _take(payload, "source_confidence", context), f"{context}.source_confidence"
        ),
        original_bbox=(
            None
            if original_bbox_raw is None
            else _decode_bbox(original_bbox_raw, f"{context}.original_bbox")
        ),
        original_kind=original_kind,
        raw_artifact_uid=_text(
            _take(payload, "raw_artifact_uid", context), f"{context}.raw_artifact_uid"
        ),
        raw_json_path=_text(
            _take(payload, "raw_json_path", context), f"{context}.raw_json_path"
        ),
        raw_index=_optional_integer(
            _take(payload, "raw_index", context), f"{context}.raw_index"
        ),
    )
    _reject_extra(payload, context)
    return origin


def _decode_layout_block(value: object, context: str) -> LayoutBlockSnapshot:
    payload = dict(_object(value, context))
    try:
        block = LayoutBlockSnapshot(
            block_type=BlockType(_text(_take(payload, "block_type", context), context)),
            bbox=_decode_bbox(_take(payload, "bbox", context), f"{context}.bbox"),
            order=_integer(_take(payload, "order", context), f"{context}.order"),
            source_label=_text(
                _take(payload, "source_label", context), f"{context}.source_label"
            ),
            origin=_decode_origin(_take(payload, "origin", context), f"{context}.origin"),
            ocr_policy=OcrPolicy(_text(_take(payload, "ocr_policy", context), context)),
            authorship=BlockSource(_text(_take(payload, "authorship", context), context)),
            uid=_text(_take(payload, "uid", context), f"{context}.uid"),
        )
    except ValueError as exc:
        raise ProjectStoreDataError(f"invalid {context} enum value") from exc
    _reject_extra(payload, context)
    return block


def _decode_layout(payload: dict[str, object]) -> LayoutSnapshot:
    data = dict(payload)
    blocks_raw = _list(_take(data, "blocks", "layout"), "layout.blocks")
    snapshot = LayoutSnapshot(
        page_uid=_text(_take(data, "page_uid", "layout"), "layout.page_uid"),
        revision=_integer(_take(data, "revision", "layout"), "layout.revision"),
        artifact_uid=_text(_take(data, "artifact_uid", "layout"), "layout.artifact_uid"),
        source_engine=_text(
            _take(data, "source_engine", "layout"), "layout.source_engine"
        ),
        source_run_id=_text(
            _take(data, "source_run_id", "layout"), "layout.source_run_id"
        ),
        blocks=tuple(
            _decode_layout_block(item, f"layout.blocks[{index}]")
            for index, item in enumerate(blocks_raw)
        ),
    )
    _reject_extra(data, "layout")
    return snapshot


def _decode_proof_state(payload: dict[str, object]) -> ProofState:
    data = dict(payload)
    anchor = _construct(
        ProofAnchorSnapshot,
        _object(_take(data, "anchor_snapshot", "proof state"), "proof state anchor"),
        "proof state anchor",
    )
    text_units = tuple(
        _construct(ProofTextUnit, _object(item, "proof text unit"), "proof text unit")
        for item in _list(_take(data, "text_units", "proof state"), "proof text units")
    )
    segments = tuple(
        _construct(
            ProofAlignmentSegment,
            _object(item, "proof state segment"),
            "proof state segment",
        )
        for item in _list(
            _take(data, "alignment_segments", "proof state"),
            "proof state segments",
        )
    )
    slices = tuple(
        _construct(
            ProofAlignmentSlice,
            _object(item, "proof state slice"),
            "proof state slice",
        )
        for item in _list(
            _take(data, "alignment_slices", "proof state"),
            "proof state slices",
        )
    )
    data["anchor_snapshot"] = anchor
    data["text_units"] = text_units
    data["alignment_segments"] = segments
    data["alignment_slices"] = slices
    return _construct(ProofState, data, "proof state")


def _read_only_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise InvalidProjectFormatError(f"project file does not exist: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise InvalidProjectFormatError("project file cannot be opened read-only") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _validate_generation(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute(
            "SELECT generation FROM project_format WHERE singleton = 1"
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    except sqlite3.Error as exc:
        raise InvalidProjectFormatError(
            "file is not a project-session v2 database"
        ) from exc
    if len(rows) != 1 or rows[0]["generation"] != FORMAT_GENERATION:
        raise InvalidProjectFormatError("project format generation does not match")
    missing = _REQUIRED_TABLES - tables
    if missing:
        raise InvalidProjectFormatError(
            f"project-session v2 schema is incomplete: {sorted(missing)}"
        )
    try:
        foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    except sqlite3.Error as exc:
        raise ProjectStoreDataError("project file relationships cannot be validated") from exc
    if foreign_key_violations:
        raise ProjectStoreDataError("project file contains cross-project relationships")


def _existing_project_uid(connection: sqlite3.Connection) -> str | None:
    rows = connection.execute(
        "SELECT project_uid FROM project WHERE singleton = 1"
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1 or not isinstance(rows[0]["project_uid"], str):
        raise ProjectStoreDataError("project file must contain exactly one project")
    return rows[0]["project_uid"]


def _validate_existing_path(path: Path, expected_project_uid: str | None = None) -> None:
    connection = _read_only_connection(path)
    try:
        _validate_generation(connection)
        project_uid = _existing_project_uid(connection)
        if expected_project_uid is not None and project_uid not in {
            None,
            expected_project_uid,
        }:
            raise ProjectUidMismatchError(
                f"project file belongs to {project_uid!r}, not {expected_project_uid!r}"
            )
    except sqlite3.Error as exc:
        raise ProjectStoreDataError("failed to inspect project file") from exc
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    existing = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view', 'trigger')"
    ).fetchall()
    if existing:
        raise InvalidProjectFormatError("new project path became non-empty before creation")
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        "INSERT INTO project_format (singleton, generation) VALUES (1, ?)",
        (FORMAT_GENERATION,),
    )


def _write_project(connection: sqlite3.Connection, record: ProjectRecord) -> None:
    existing = connection.execute(
        "SELECT project_uid FROM project WHERE singleton = 1"
    ).fetchone()
    payload = _payload(record)
    if existing is None:
        connection.execute(
            """
            INSERT INTO project (singleton, project_uid, fingerprint, payload)
            VALUES (1, ?, ?, ?)
            """,
            (record.project_uid, record.fingerprint, payload),
        )
        return
    if existing["project_uid"] != record.project_uid:
        raise ProjectUidMismatchError(
            f"project file belongs to {existing['project_uid']!r}, not {record.project_uid!r}"
        )
    connection.execute(
        "UPDATE project SET fingerprint = ?, payload = ? WHERE singleton = 1",
        (record.fingerprint, payload),
    )


def _upsert_current(
    connection: sqlite3.Connection,
    *,
    table: str,
    key_name: str,
    key: str,
    project_uid: str,
    revision: int,
    fingerprint: str,
    payload: str,
) -> None:
    existing = connection.execute(
        f"SELECT project_uid, revision, fingerprint, payload FROM {table} WHERE {key_name} = ?",
        (key,),
    ).fetchone()
    if existing is None:
        connection.execute(
            f"""
            INSERT INTO {table} ({key_name}, project_uid, revision, fingerprint, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, project_uid, revision, fingerprint, payload),
        )
        return
    if existing["project_uid"] != project_uid:
        raise ProjectUidMismatchError(f"{table} record {key!r} belongs to another project")
    persisted_revision = _integer(existing["revision"], f"{table} revision")
    if revision < persisted_revision:
        raise ProjectStoreConflictError(
            f"{table} record {key!r} is newer in the project file"
        )
    if revision == persisted_revision:
        if existing["fingerprint"] != fingerprint or existing["payload"] != payload:
            raise ProjectStoreConflictError(
                f"{table} record {key!r} conflicts at revision {revision}"
            )
        return
    connection.execute(
        f"""
        UPDATE {table}
        SET project_uid = ?, revision = ?, fingerprint = ?, payload = ?
        WHERE {key_name} = ?
        """,
        (project_uid, revision, fingerprint, payload, key),
    )


def _append_fact(
    connection: sqlite3.Connection,
    *,
    table: str,
    record: PaddleArtifact | OcrRun | OcrRegion | OcrLine | OcrAtom | OcrCandidate | OcrBatch,
) -> None:
    payload = _payload(record)
    existing = connection.execute(
        f"SELECT project_uid, fingerprint, payload FROM {table} WHERE uid = ?",
        (record.uid,),
    ).fetchone()
    if existing is None:
        connection.execute(
            f"""
            INSERT INTO {table} (uid, project_uid, fingerprint, payload)
            VALUES (?, ?, ?, ?)
            """,
            (record.uid, record.project_uid, record.fingerprint, payload),
        )
        return
    if (
        existing["project_uid"] != record.project_uid
        or existing["fingerprint"] != record.fingerprint
        or existing["payload"] != payload
    ):
        raise ProjectStoreConflictError(
            f"append-only {table} UID {record.uid!r} has conflicting content"
        )


def _upsert_active_pointer(
    connection: sqlite3.Connection,
    pointer: OcrActivePointer,
) -> None:
    existing = connection.execute(
        "SELECT payload FROM ocr_active_pointer WHERE scope_uid = ?",
        (pointer.scope_uid,),
    ).fetchone()
    if existing is not None:
        persisted = _construct(
            OcrActivePointer,
            _payload_object(existing["payload"], "OCR active pointer"),
            "OCR active pointer",
        )
        if persisted.uid != pointer.uid:
            raise ProjectStoreConflictError(
                f"active pointer UID cannot change for scope {pointer.scope_uid!r}"
            )
    _upsert_current(
        connection,
        table="ocr_active_pointer",
        key_name="scope_uid",
        key=pointer.scope_uid,
        project_uid=pointer.project_uid,
        revision=pointer.revision,
        fingerprint=pointer.fingerprint,
        payload=_payload(pointer),
    )


def _write_records(connection: sqlite3.Connection, records: _SessionRecords) -> None:
    _write_project(connection, records.project)
    project_uid = records.project.project_uid
    for page in records.pages:
        _upsert_current(
            connection,
            table="page",
            key_name="uid",
            key=page.uid,
            project_uid=project_uid,
            revision=page.image_revision,
            fingerprint=page.fingerprint,
            payload=_payload(page),
        )
    for layout in records.layouts:
        _upsert_current(
            connection,
            table="layout_snapshot",
            key_name="page_uid",
            key=layout.page_uid,
            project_uid=project_uid,
            revision=layout.revision,
            fingerprint=LayoutRepository.fingerprint_for(layout),
            payload=_payload(layout),
        )
    for artifact in records.paddle_artifacts:
        _append_fact(connection, table="paddle_artifact", record=artifact)
    for table, values in (
        ("ocr_run", records.ocr_runs),
        ("ocr_region", records.ocr_regions),
        ("ocr_line", records.ocr_lines),
        ("ocr_atom", records.ocr_atoms),
        ("ocr_batch", records.ocr_batches),
        ("ocr_candidate", records.ocr_candidates),
    ):
        for record in values:
            _append_fact(connection, table=table, record=record)
    for pointer in records.ocr_active_pointers:
        _upsert_active_pointer(connection, pointer)
    for table, values in (
        ("proof_state", records.proof_states),
        ("binding", records.bindings),
        ("table_text", records.table_texts),
    ):
        for record in values:
            _upsert_current(
                connection,
                table=table,
                key_name="uid",
                key=record.uid,
                project_uid=project_uid,
                revision=record.revision,
                fingerprint=record.fingerprint,
                payload=_payload(record),
            )


def _verify_row_record(
    row: sqlite3.Row,
    record: object,
    *,
    context: str,
    key_name: str = "uid",
    record_key: str | None = None,
    revision: int | None = None,
    fingerprint: str | None = None,
) -> None:
    expected_key = record_key if record_key is not None else getattr(record, "uid")
    if row[key_name] != expected_key:
        raise ProjectStoreDataError(f"{context} row key does not match its payload")
    if row["project_uid"] != getattr(record, "project_uid", row["project_uid"]):
        raise ProjectStoreDataError(f"{context} project UID does not match its payload")
    if revision is not None and row["revision"] != revision:
        raise ProjectStoreDataError(f"{context} revision does not match its payload")
    expected_fingerprint = fingerprint or getattr(record, "fingerprint")
    if row["fingerprint"] != expected_fingerprint:
        raise ProjectStoreDataError(f"{context} fingerprint does not match its payload")


def _load_simple_records(
    connection: sqlite3.Connection,
    *,
    table: str,
    record_type: Callable[..., T],
    revision_getter: Callable[[T], int] | None = None,
) -> tuple[T, ...]:
    revision_column = ", revision" if revision_getter is not None else ""
    rows = connection.execute(
        f"SELECT uid, project_uid, fingerprint, payload{revision_column} FROM {table} ORDER BY uid"
    ).fetchall()
    result: list[T] = []
    for row in rows:
        record = _construct(
            record_type,
            _payload_object(row["payload"], table),
            table,
        )
        revision = revision_getter(record) if revision_getter is not None else None
        _verify_row_record(row, record, context=table, revision=revision)
        result.append(record)
    return tuple(result)


def _load_layouts(connection: sqlite3.Connection) -> tuple[LayoutSnapshot, ...]:
    rows = connection.execute(
        """
        SELECT page_uid, project_uid, revision, fingerprint, payload
        FROM layout_snapshot ORDER BY page_uid
        """
    ).fetchall()
    result: list[LayoutSnapshot] = []
    for row in rows:
        layout = _decode_layout(_payload_object(row["payload"], "layout snapshot"))
        _verify_row_record(
            row,
            layout,
            context="layout snapshot",
            key_name="page_uid",
            record_key=layout.page_uid,
            revision=layout.revision,
            fingerprint=LayoutRepository.fingerprint_for(layout),
        )
        result.append(layout)
    return tuple(result)


def _load_active_pointers(connection: sqlite3.Connection) -> tuple[OcrActivePointer, ...]:
    rows = connection.execute(
        """
        SELECT scope_uid, project_uid, revision, fingerprint, payload
        FROM ocr_active_pointer ORDER BY scope_uid
        """
    ).fetchall()
    result: list[OcrActivePointer] = []
    for row in rows:
        pointer = _construct(
            OcrActivePointer,
            _payload_object(row["payload"], "OCR active pointer"),
            "OCR active pointer",
        )
        _verify_row_record(
            row,
            pointer,
            context="OCR active pointer",
            key_name="scope_uid",
            record_key=pointer.scope_uid,
            revision=pointer.revision,
        )
        result.append(pointer)
    return tuple(result)


def _load_proof_states(connection: sqlite3.Connection) -> tuple[ProofState, ...]:
    rows = connection.execute(
        "SELECT uid, project_uid, revision, fingerprint, payload FROM proof_state ORDER BY uid"
    ).fetchall()
    result: list[ProofState] = []
    for row in rows:
        state = _decode_proof_state(_payload_object(row["payload"], "proof state"))
        _verify_row_record(
            row,
            state,
            context="proof state",
            revision=state.revision,
        )
        result.append(state)
    return tuple(result)


def _load_records(connection: sqlite3.Connection) -> _SessionRecords:
    project_rows = connection.execute(
        "SELECT project_uid, fingerprint, payload FROM project WHERE singleton = 1"
    ).fetchall()
    if len(project_rows) != 1:
        raise ProjectStoreDataError("project file must contain exactly one project")
    project_row = project_rows[0]
    project = _construct(
        ProjectRecord,
        _payload_object(project_row["payload"], "project"),
        "project",
    )
    if (
        project_row["project_uid"] != project.project_uid
        or project_row["fingerprint"] != project.fingerprint
    ):
        raise ProjectStoreDataError("project row does not match its payload")
    return _SessionRecords(
        project=project,
        pages=_load_simple_records(
            connection,
            table="page",
            record_type=PageRecord,
            revision_getter=lambda record: record.image_revision,
        ),
        layouts=_load_layouts(connection),
        paddle_artifacts=_load_simple_records(
            connection,
            table="paddle_artifact",
            record_type=PaddleArtifact,
        ),
        ocr_runs=_load_simple_records(connection, table="ocr_run", record_type=OcrRun),
        ocr_regions=_load_simple_records(
            connection, table="ocr_region", record_type=OcrRegion
        ),
        ocr_lines=_load_simple_records(connection, table="ocr_line", record_type=OcrLine),
        ocr_atoms=_load_simple_records(connection, table="ocr_atom", record_type=OcrAtom),
        ocr_candidates=_load_simple_records(
            connection, table="ocr_candidate", record_type=OcrCandidate
        ),
        ocr_batches=_load_simple_records(
            connection, table="ocr_batch", record_type=OcrBatch
        ),
        ocr_active_pointers=_load_active_pointers(connection),
        proof_states=_load_proof_states(connection),
        bindings=_load_simple_records(
            connection,
            table="binding",
            record_type=BindingRecord,
            revision_getter=lambda record: record.revision,
        ),
        table_texts=_load_simple_records(
            connection,
            table="table_text",
            record_type=TableTextRecord,
            revision_getter=lambda record: record.revision,
        ),
    )


def save_session(path: str | Path, session: ProjectSession) -> None:
    """Atomically persist one new-model project session to one SQLite file."""
    target = Path(path)
    records = _snapshot_session(session)
    existed = target.exists()
    if existed:
        _validate_existing_path(target, records.project.project_uid)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(target)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        if existed:
            _validate_generation(connection)
        else:
            _create_schema(connection)
        _write_records(connection, records)
        connection.commit()
    except ProjectStoreError:
        if connection is not None:
            connection.rollback()
        raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise ProjectStoreDataError("failed to save project session") from exc
    finally:
        if connection is not None:
            connection.close()
    session.save_path = str(target)


def load_session(
    path: str | Path,
    *,
    expected_project_uid: str | None = None,
) -> ProjectSession:
    """Load one exact-generation file into fresh ProjectSession repositories."""
    target = Path(path)
    connection = _read_only_connection(target)
    try:
        _validate_generation(connection)
        records = _load_records(connection)
        if (
            expected_project_uid is not None
            and records.project.project_uid != expected_project_uid
        ):
            raise ProjectUidMismatchError(
                f"project file belongs to {records.project.project_uid!r}, "
                f"not {expected_project_uid!r}"
            )
        try:
            return ProjectSession.from_records(
                records.project,
                **records.hydration_kwargs(),
                save_path=str(target),
            )
        except (TypeError, ValueError) as exc:
            raise ProjectStoreDataError("persisted records cannot hydrate a session") from exc
    except ProjectStoreError:
        raise
    except sqlite3.Error as exc:
        raise ProjectStoreDataError("failed to load project session") from exc
    finally:
        connection.close()


class ProjectStore:
    """Path-bound facade for the strict project-session persistence format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save_session(self, session: ProjectSession) -> None:
        save_session(self.path, session)

    def load_session(self, *, expected_project_uid: str | None = None) -> ProjectSession:
        return load_session(self.path, expected_project_uid=expected_project_uid)


__all__ = [
    "FORMAT_GENERATION",
    "InvalidProjectFormatError",
    "ProjectStore",
    "ProjectStoreConflictError",
    "ProjectStoreDataError",
    "ProjectStoreError",
    "ProjectUidMismatchError",
    "load_session",
    "save_session",
]
