"""SQLite 项目文件读写。

项目文件扩展名：.ocrproj（本质是 SQLite 数据库）。
图片数据只存路径，不存 Blob，避免数据库过大。

Schema 版本管理：
- meta 表记录 schema_version
- 启动时检测版本并执行增量迁移
"""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional

from app.models import (
    BBox, Block, BlockOrigin, BlockSource, BlockType, Char, LayoutEditEvent, Line,
    OcrPolicy, OcrProject, PaddleBinding, Page, PageStatus, ProofLineState, ProofStatus,
    RawOcrArtifact,
)
from app.models.entity_id import ensure_entity_uid, new_entity_uid
from app.models.layout_projection import page_layout_blocks, replace_page_layout_blocks
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.layout_snapshot_store import layout_snapshot_for_page, set_layout_snapshot_for_page
from app.models.layout_block_state import (
    set_layout_block_bbox,
    set_layout_block_note,
    set_layout_block_ocr_policy,
    set_layout_block_order,
    set_layout_block_source_label,
    set_layout_block_type,
)
from app.models.ocr_character_observation import line_ocr_chars_by_uid, replace_line_ocr_char_observations
from app.models.ocr_observation import (
    block_ocr_line_observations_by_uid,
    iter_project_ocr_line_observation_occurrences,
    line_ocr_bbox,
    replace_block_ocr_line_observations,
)
from app.models.ocr_text_observation import line_ocr_review_flags
from app.models.page_state import reconcile_page_ocr_done_from_result

from app.core.logging import get_logger, APP_VERSION, SCHEMA_VERSION
from app.core.model_validation import (
    ModelValidationError,
    validate_block_model,
    validate_page_model,
)
from app.core.line_text_contract import line_text_contract
from app.core.proof_line_mutation import apply_line_proof_state

logger = get_logger(__name__)


class ProjectDataError(RuntimeError):
    """Current schema project data violates model boundaries."""


def _enum_from_db(enum_cls: Any, raw_value: object, *, field: str) -> Any:
    value = str(raw_value or "").strip()
    if not value:
        raise ProjectDataError(f"{field} is empty")
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ProjectDataError(f"{field} invalid value: {value!r}") from exc


def _int_from_db(raw_value: object, *, field: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ProjectDataError(f"{field} must be int")
    return raw_value


def _bbox_from_db(
    row: sqlite3.Row,
    *,
    fields: tuple[str, str, str, str],
    field: str,
    optional: bool = False,
) -> BBox | None:
    values = [row[name] for name in fields]
    if optional and all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ProjectDataError(f"{field} bbox is partial")
    x_name, y_name, w_name, h_name = fields
    return BBox(
        _int_from_db(row[x_name], field=f"{field}.{x_name}"),
        _int_from_db(row[y_name], field=f"{field}.{y_name}"),
        _int_from_db(row[w_name], field=f"{field}.{w_name}"),
        _int_from_db(row[h_name], field=f"{field}.{h_name}"),
    )


# --------------------------------------------------------------------- schema v3
DDL_V3 = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS project (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS page (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT    NOT NULL DEFAULT '',
    project_id      INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    image_path      TEXT    NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    page_number     INTEGER NOT NULL DEFAULT 1,
    source_path     TEXT    NOT NULL DEFAULT '',
    source_type     TEXT    NOT NULL DEFAULT 'image',
    source_page_index INTEGER NOT NULL DEFAULT 1,
    cache_image_path TEXT   NOT NULL DEFAULT '',
    thumbnail_path  TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'imported',
    error_message   TEXT    NOT NULL DEFAULT '',
    ocr_invalidated_reason TEXT NOT NULL DEFAULT '',
    raw_layout_artifact_uid TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS raw_ocr_artifact (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT    NOT NULL DEFAULT '',
    project_id      INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    page_uid        TEXT    NOT NULL DEFAULT '',
    engine          TEXT    NOT NULL,
    engine_version  TEXT    NOT NULL DEFAULT '',
    run_id          TEXT    NOT NULL DEFAULT '',
    artifact_path   TEXT    NOT NULL DEFAULT '',
    artifact_hash   TEXT    NOT NULL DEFAULT '',
    records_json    TEXT    NOT NULL DEFAULT '[]',
    route_attachments_json TEXT NOT NULL DEFAULT '{}',
    created_at      REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(project_id, uid)
);

CREATE INDEX IF NOT EXISTS idx_raw_ocr_artifact_project_page
    ON raw_ocr_artifact(project_id, page_uid);

CREATE TABLE IF NOT EXISTS block (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT    NOT NULL DEFAULT '',
    page_id         INTEGER NOT NULL REFERENCES page(id) ON DELETE CASCADE,
    block_type      TEXT    NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL,
    block_order     INTEGER NOT NULL DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT 'auto_layout',
    ocr_policy      TEXT    NOT NULL DEFAULT 'text_ocr',
    note            TEXT    NOT NULL DEFAULT '',
    source_label    TEXT    NOT NULL DEFAULT '',
    paddle_binding_json TEXT NOT NULL DEFAULT '{}',
    ocr_invalidated_reason TEXT NOT NULL DEFAULT '',
    ocr_audit_json TEXT NOT NULL DEFAULT '{}',
    table_text_layer_cells_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS block_origin (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    block_uid           TEXT    NOT NULL,
    created_by          TEXT    NOT NULL DEFAULT 'auto_layout',
    source_engine       TEXT    NOT NULL DEFAULT '',
    source_run_id       TEXT    NOT NULL DEFAULT '',
    source_label        TEXT    NOT NULL DEFAULT '',
    source_confidence   REAL,
    original_x          INTEGER,
    original_y          INTEGER,
    original_w          INTEGER,
    original_h          INTEGER,
    original_kind       TEXT    NOT NULL DEFAULT '',
    raw_artifact_uid    TEXT    NOT NULL DEFAULT '',
    raw_json_path       TEXT    NOT NULL DEFAULT '',
    raw_index           INTEGER,
    UNIQUE(project_id, block_uid)
);

CREATE INDEX IF NOT EXISTS idx_block_origin_project
    ON block_origin(project_id);

CREATE TABLE IF NOT EXISTS layout_edit_event (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT    NOT NULL DEFAULT '',
    project_id      INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    page_uid        TEXT    NOT NULL DEFAULT '',
    target_uid      TEXT    NOT NULL DEFAULT '',
    op              TEXT    NOT NULL,
    before_json     TEXT    NOT NULL DEFAULT '{}',
    after_json      TEXT    NOT NULL DEFAULT '{}',
    actor           TEXT    NOT NULL DEFAULT 'user',
    created_at      REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(project_id, uid)
);

CREATE INDEX IF NOT EXISTS idx_layout_edit_event_project_page
    ON layout_edit_event(project_id, page_uid);

CREATE TABLE IF NOT EXISTS layout_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    page_uid        TEXT    NOT NULL DEFAULT '',
    artifact_uid    TEXT    NOT NULL DEFAULT '',
    source_engine   TEXT    NOT NULL DEFAULT '',
    source_run_id   TEXT    NOT NULL DEFAULT '',
    blocks_json     TEXT    NOT NULL DEFAULT '[]',
    updated_at      REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(project_id, page_uid)
);

CREATE INDEX IF NOT EXISTS idx_layout_snapshot_project_page
    ON layout_snapshot(project_id, page_uid);

CREATE TABLE IF NOT EXISTS line (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    uid               TEXT    NOT NULL DEFAULT '',
    block_id          INTEGER NOT NULL REFERENCES block(id) ON DELETE CASCADE,
    text              TEXT    NOT NULL DEFAULT '',
    confidence        REAL    NOT NULL DEFAULT 0.0,
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL,
    ocr_text          TEXT    NOT NULL DEFAULT '',
    review_flags_json TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS char_ (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uid         TEXT    NOT NULL DEFAULT '',
    line_id     INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    char        TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    bbox_source TEXT    NOT NULL DEFAULT '',
    bbox_granularity TEXT NOT NULL DEFAULT '',
    token_text  TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS proof_line_state (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    line_uid          TEXT    NOT NULL,
    final_text        TEXT    NOT NULL DEFAULT '',
    final_text_set    INTEGER NOT NULL DEFAULT 0,
    proof_status      TEXT    NOT NULL DEFAULT 'unchecked',
    alignment_state   TEXT    NOT NULL DEFAULT 'aligned',
    updated_at        REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(project_id, line_uid)
);

CREATE INDEX IF NOT EXISTS idx_proof_line_state_project
    ON proof_line_state(project_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operation_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL,
    page_id         INTEGER,
    object_type     TEXT NOT NULL,
    object_id       INTEGER,
    object_uid      TEXT NOT NULL DEFAULT '',
    action          TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
"""

# ------------------------------------------------------------------- migration

MIGRATIONS: dict[int, list[str]] = {
    # v1 -> v2: add new columns and tables
    2: [
        # page table additions are handled by CREATE IF NOT EXISTS,
        # but we need ALTER for existing v1 dbs
        "ALTER TABLE page ADD COLUMN source_path TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE page ADD COLUMN source_type TEXT NOT NULL DEFAULT 'image';",
        "ALTER TABLE page ADD COLUMN source_page_index INTEGER NOT NULL DEFAULT 1;",
        "ALTER TABLE page ADD COLUMN cache_image_path TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE page ADD COLUMN thumbnail_path TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE page ADD COLUMN status TEXT NOT NULL DEFAULT 'imported';",
        "ALTER TABLE page ADD COLUMN error_message TEXT NOT NULL DEFAULT '';",
        # block table additions
        "ALTER TABLE block ADD COLUMN source TEXT NOT NULL DEFAULT 'auto_layout';",
        "ALTER TABLE block ADD COLUMN note TEXT NOT NULL DEFAULT '';",
        # line table additions
        "ALTER TABLE line ADD COLUMN ocr_text TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE line ADD COLUMN review_flags_json TEXT NOT NULL DEFAULT '[]';",
        # new tables
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        "CREATE TABLE IF NOT EXISTS operation_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_id INTEGER NOT NULL, page_id INTEGER, "
        "object_type TEXT NOT NULL, object_id INTEGER, "
        "action TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', "
        "created_at REAL NOT NULL);",
    ],
    3: [
        "ALTER TABLE char_ ADD COLUMN bbox_source TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE char_ ADD COLUMN bbox_granularity TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE char_ ADD COLUMN token_text TEXT NOT NULL DEFAULT '';",
    ],
    4: [],
    5: [],
    6: [
        "ALTER TABLE block ADD COLUMN source_label TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE block ADD COLUMN raw_payload_json TEXT NOT NULL DEFAULT '{}';",
    ],
    7: [
        "ALTER TABLE page ADD COLUMN ocr_invalidated_reason TEXT NOT NULL DEFAULT '';",
    ],
    8: [
        "ALTER TABLE block ADD COLUMN app_payload_json TEXT NOT NULL DEFAULT '{}';",
    ],
    9: [
        "ALTER TABLE page ADD COLUMN uid TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE block ADD COLUMN uid TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE line ADD COLUMN uid TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE char_ ADD COLUMN uid TEXT NOT NULL DEFAULT '';",
    ],
    10: [
        "ALTER TABLE operation_log ADD COLUMN object_uid TEXT NOT NULL DEFAULT '';",
    ],
    11: [],
    12: [
        "CREATE TABLE IF NOT EXISTS proof_line_state ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE, "
        "line_uid TEXT NOT NULL, "
        "final_text TEXT NOT NULL DEFAULT '', "
        "final_text_set INTEGER NOT NULL DEFAULT 0, "
        "proof_status TEXT NOT NULL DEFAULT 'unchecked', "
        "alignment_state TEXT NOT NULL DEFAULT 'aligned', "
        "updated_at REAL NOT NULL DEFAULT 0.0, "
        "UNIQUE(project_id, line_uid));",
        "CREATE INDEX IF NOT EXISTS idx_proof_line_state_project "
        "ON proof_line_state(project_id);",
    ],
    13: [],
    14: [
        "ALTER TABLE page ADD COLUMN raw_layout_artifact_uid TEXT NOT NULL DEFAULT '';",
        "CREATE TABLE IF NOT EXISTS raw_ocr_artifact ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "uid TEXT NOT NULL DEFAULT '', "
        "project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE, "
        "page_uid TEXT NOT NULL DEFAULT '', "
        "engine TEXT NOT NULL, "
        "engine_version TEXT NOT NULL DEFAULT '', "
        "run_id TEXT NOT NULL DEFAULT '', "
        "artifact_path TEXT NOT NULL DEFAULT '', "
        "artifact_hash TEXT NOT NULL DEFAULT '', "
        "records_json TEXT NOT NULL DEFAULT '[]', "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "UNIQUE(project_id, uid));",
        "CREATE INDEX IF NOT EXISTS idx_raw_ocr_artifact_project_page "
        "ON raw_ocr_artifact(project_id, page_uid);",
    ],
    15: [
        "ALTER TABLE block ADD COLUMN ocr_policy TEXT NOT NULL DEFAULT 'text_ocr';",
    ],
    16: [
        "CREATE TABLE IF NOT EXISTS block_origin ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE, "
        "block_uid TEXT NOT NULL, "
        "created_by TEXT NOT NULL DEFAULT 'auto_layout', "
        "source_engine TEXT NOT NULL DEFAULT '', "
        "source_run_id TEXT NOT NULL DEFAULT '', "
        "source_label TEXT NOT NULL DEFAULT '', "
        "source_confidence REAL, "
        "original_x INTEGER, "
        "original_y INTEGER, "
        "original_w INTEGER, "
        "original_h INTEGER, "
        "original_kind TEXT NOT NULL DEFAULT '', "
        "raw_artifact_uid TEXT NOT NULL DEFAULT '', "
        "raw_json_path TEXT NOT NULL DEFAULT '', "
        "raw_index INTEGER, "
        "UNIQUE(project_id, block_uid));",
        "CREATE INDEX IF NOT EXISTS idx_block_origin_project "
        "ON block_origin(project_id);",
    ],
    17: [
        "CREATE TABLE IF NOT EXISTS layout_edit_event ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "uid TEXT NOT NULL DEFAULT '', "
        "project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE, "
        "page_uid TEXT NOT NULL DEFAULT '', "
        "target_uid TEXT NOT NULL DEFAULT '', "
        "op TEXT NOT NULL, "
        "before_json TEXT NOT NULL DEFAULT '{}', "
        "after_json TEXT NOT NULL DEFAULT '{}', "
        "actor TEXT NOT NULL DEFAULT 'user', "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "UNIQUE(project_id, uid));",
        "CREATE INDEX IF NOT EXISTS idx_layout_edit_event_project_page "
        "ON layout_edit_event(project_id, page_uid);",
    ],
    18: [
        "ALTER TABLE block ADD COLUMN paddle_binding_json TEXT NOT NULL DEFAULT '{}';",
    ],
    19: [
        "ALTER TABLE block ADD COLUMN ocr_invalidated_reason TEXT NOT NULL DEFAULT '';",
    ],
    20: [
        "ALTER TABLE block ADD COLUMN ocr_audit_json TEXT NOT NULL DEFAULT '{}';",
    ],
    21: [
        "ALTER TABLE block ADD COLUMN table_text_layer_cells_json TEXT NOT NULL DEFAULT '[]';",
    ],
    22: [
        "ALTER TABLE raw_ocr_artifact ADD COLUMN route_attachments_json TEXT NOT NULL DEFAULT '{}';",
    ],
    23: [],
    24: [
        "CREATE TABLE IF NOT EXISTS layout_snapshot ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_id INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE, "
        "page_uid TEXT NOT NULL DEFAULT '', "
        "artifact_uid TEXT NOT NULL DEFAULT '', "
        "source_engine TEXT NOT NULL DEFAULT '', "
        "source_run_id TEXT NOT NULL DEFAULT '', "
        "blocks_json TEXT NOT NULL DEFAULT '[]', "
        "updated_at REAL NOT NULL DEFAULT 0.0, "
        "UNIQUE(project_id, page_uid));",
        "CREATE INDEX IF NOT EXISTS idx_layout_snapshot_project_page "
        "ON layout_snapshot(project_id, page_uid);",
    ],
}


# ------------------------------------------------------------------- helpers

def _row_to_blocks(row: sqlite3.Row) -> dict:
    return dict(row)


def _review_flags_to_json(flags: list[str]) -> str:
    return json.dumps(flags, ensure_ascii=False)


def _json_to_review_flags(s: str, *, field: str = "line.review_flags_json") -> list[str]:
    if not s:
        raise ProjectDataError(f"{field} is empty")
    try:
        value = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProjectDataError(f"{field} invalid json") from exc
    if not isinstance(value, list):
        raise ProjectDataError(f"{field} must be list")
    result: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ProjectDataError(f"{field}[{item_index}] must be str")
        result.append(item)
    return result


def _dict_list(values: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ProjectDataError(f"{field} must be list")
    result: list[dict[str, Any]] = []
    for item_index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ProjectDataError(f"{field}[{item_index}] must be dict")
        result.append(dict(value))
    return result


def _json_to_list(s: str, *, field: str) -> list:
    if not s:
        raise ProjectDataError(f"{field} is empty")
    try:
        value = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProjectDataError(f"{field} invalid json") from exc
    if not isinstance(value, list):
        raise ProjectDataError(f"{field} must be list")
    return value


def _json_to_dict(s: str, *, field: str) -> dict:
    if not s:
        raise ProjectDataError(f"{field} is empty")
    try:
        value = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProjectDataError(f"{field} invalid json") from exc
    if not isinstance(value, dict):
        raise ProjectDataError(f"{field} must be dict")
    return value


def _str_from_json(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ProjectDataError(f"{field} must be str")
    return value


def _non_empty_str_from_json(value: object, *, field: str) -> str:
    text = _str_from_json(value, field=field).strip()
    if not text:
        raise ProjectDataError(f"{field} is empty")
    return text


def _int_from_json(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectDataError(f"{field} must be int")
    return value


def _optional_float_from_json(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectDataError(f"{field} must be number or null")
    return float(value)


def _optional_int_from_json(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _int_from_json(value, field=field)


def _dict_from_json_value(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectDataError(f"{field} must be dict")
    return dict(value)


def _bbox_from_json(value: object, *, field: str, optional: bool = False) -> BBox | None:
    if value is None and optional:
        return None
    payload = _dict_from_json_value(value, field=field)
    return BBox(
        _int_from_json(payload.get("x"), field=f"{field}.x"),
        _int_from_json(payload.get("y"), field=f"{field}.y"),
        _int_from_json(payload.get("w"), field=f"{field}.w"),
        _int_from_json(payload.get("h"), field=f"{field}.h"),
    )


def _origin_to_json(origin: BlockOrigin) -> dict[str, Any]:
    return origin.to_dict()


def _origin_from_json(value: object, *, field: str) -> BlockOrigin:
    payload = _dict_from_json_value(value, field=field)
    raw_kind = _str_from_json(payload.get("original_kind", ""), field=f"{field}.original_kind")
    original_kind = (
        _enum_from_db(BlockType, raw_kind, field=f"{field}.original_kind")
        if raw_kind
        else None
    )
    return BlockOrigin(
        created_by=_str_from_json(payload.get("created_by", ""), field=f"{field}.created_by"),
        source_engine=_str_from_json(payload.get("source_engine", ""), field=f"{field}.source_engine"),
        source_run_id=_str_from_json(payload.get("source_run_id", ""), field=f"{field}.source_run_id"),
        source_label=_str_from_json(payload.get("source_label", ""), field=f"{field}.source_label"),
        source_confidence=_optional_float_from_json(
            payload.get("source_confidence"),
            field=f"{field}.source_confidence",
        ),
        original_bbox=_bbox_from_json(
            payload.get("original_bbox"),
            field=f"{field}.original_bbox",
            optional=True,
        ),
        original_kind=original_kind,
        raw_artifact_uid=_str_from_json(
            payload.get("raw_artifact_uid", ""),
            field=f"{field}.raw_artifact_uid",
        ),
        raw_json_path=_str_from_json(payload.get("raw_json_path", ""), field=f"{field}.raw_json_path"),
        raw_index=_optional_int_from_json(payload.get("raw_index"), field=f"{field}.raw_index"),
    )


def _layout_block_snapshot_to_json(block: LayoutBlockSnapshot) -> dict[str, Any]:
    return {
        "uid": block.uid,
        "block_type": block.block_type.value,
        "bbox": block.bbox.to_dict(),
        "order": block.order,
        "source_label": block.source_label,
        "origin": _origin_to_json(block.origin),
        "ocr_policy": block.ocr_policy.value,
        "note": block.note,
    }


def _layout_block_snapshot_from_json(value: object, *, index: int) -> LayoutBlockSnapshot:
    field = f"layout_snapshot.blocks_json[{index}]"
    payload = _dict_from_json_value(value, field=field)
    return LayoutBlockSnapshot(
        uid=_non_empty_str_from_json(payload.get("uid", ""), field=f"{field}.uid"),
        block_type=_enum_from_db(
            BlockType,
            payload.get("block_type", ""),
            field=f"{field}.block_type",
        ),
        bbox=_bbox_from_json(payload.get("bbox"), field=f"{field}.bbox"),
        order=_int_from_json(payload.get("order"), field=f"{field}.order"),
        source_label=_str_from_json(payload.get("source_label", ""), field=f"{field}.source_label"),
        origin=_origin_from_json(payload.get("origin"), field=f"{field}.origin"),
        ocr_policy=_enum_from_db(
            OcrPolicy,
            payload.get("ocr_policy", ""),
            field=f"{field}.ocr_policy",
        ),
        note=_str_from_json(payload.get("note", ""), field=f"{field}.note"),
    )


def _layout_snapshot_blocks_to_json(snapshot: LayoutSnapshot) -> list[dict[str, Any]]:
    return [_layout_block_snapshot_to_json(block) for block in snapshot.blocks]


def _route_attachments_to_json_dict(
    attachments: dict[int, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for index, values in dict(attachments or {}).items():
        try:
            normalized_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ProjectDataError("raw_ocr_artifact.route_attachments key must be int-like") from exc
        result[str(normalized_index)] = _route_attachment_dicts(
            values,
            field=f"raw_ocr_artifact.route_attachments[{index!r}]",
        )
    return result


def _route_attachment_dicts(values: object, *, field: str) -> list[dict[str, Any]]:
    return _dict_list(values, field=field)


def _json_to_route_attachments(s: str, *, field: str) -> dict[int, list[dict[str, Any]]]:
    raw = _json_to_dict(s, field=field)
    result: dict[int, list[dict[str, Any]]] = {}
    for key, values in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ProjectDataError(f"{field} key must be int-like") from exc
        result[index] = _route_attachment_dicts(values, field=f"{field}[{key!r}]")
    return result


def _origin_from_current_block(block: Block) -> BlockOrigin:
    """Create a minimal origin for blocks that were built without provenance.

    This is a storage boundary fallback, not legacy payload migration. It only
    records the current block fields available on the active layout model.
    """
    return BlockOrigin(
        created_by=getattr(block.source, "value", str(block.source or BlockSource.AUTO_LAYOUT.value)),
        source_label=str(block.source_label or ""),
        original_bbox=block.bbox,
        original_kind=block.block_type,
    )


def _proof_alignment_state(line: Line, display_text: str | None = None) -> str:
    chars = line_ocr_chars_by_uid(line.uid)
    if not chars:
        return "line_only"
    try:
        from app.core.proof_char_text import chars_display_text
        text = display_text if display_text is not None else line_text_contract(line).text
        return "aligned" if chars_display_text(chars) == text else "degraded"
    except Exception:
        return "degraded"


def _raise_project_data_error(exc: ModelValidationError) -> None:
    raise ProjectDataError(str(exc)) from exc


_ENTITY_UID_TABLES = (
    ("page", "page", "idx_page_uid"),
    ("block", "block", "idx_block_uid"),
    ("line", "line", "idx_line_uid"),
    ("char_", "char", "idx_char_uid"),
)


def _new_unique_uid(kind: str, seen: set[str]) -> str:
    while True:
        uid = new_entity_uid(kind)
        if uid not in seen:
            return uid


# ---------------------------------------------------------------- store class

class ProjectStore:
    """负责 OcrProject 的持久化。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ open/close

    def open(self) -> None:
        db_file = Path(self.db_path)
        is_new_database = not db_file.exists() or db_file.stat().st_size == 0
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(DDL_V3)
        self._conn.commit()
        self._ensure_meta(schema_version=SCHEMA_VERSION if is_new_database else 1)
        self._migrate()
        self._ensure_entity_uids()
        self._ensure_proof_line_states()

    def _ensure_meta(self, *, schema_version: int) -> None:
        """确保 meta 表和版本记录存在。"""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            app_version = APP_VERSION if schema_version >= SCHEMA_VERSION else "0.1.0"
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(schema_version),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('app_version', ?)",
                (app_version,),
            )
            self.conn.commit()

    def _migrate(self) -> None:
        """顺序执行迁移到当前版本。"""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        current_version = int(row["value"]) if row else 0

        target_version = SCHEMA_VERSION
        for ver in range(current_version + 1, target_version + 1):
            if ver in MIGRATIONS:
                stmts = MIGRATIONS[ver]
                logger.info("Running schema migration v%d -> v%d", ver - 1, ver)
                try:
                    if ver in (23, 24):
                        self._migrate_v23_drop_retired_block_payload_columns()
                    for stmt in stmts:
                        try:
                            self.conn.execute(stmt)
                        except sqlite3.OperationalError as e:
                            # "duplicate column name" is safe to ignore
                            # when the column already exists (table recreated)
                            if "duplicate column" in str(e).lower():
                                logger.debug("Skipping (already applied): %s", e)
                            else:
                                raise
                    self.conn.execute(
                        "UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(ver),),
                    )
                    self.conn.execute(
                        "UPDATE meta SET value=? WHERE key='app_version'",
                        (APP_VERSION,),
                    )
                    self.conn.commit()
                    logger.info("Migration to v%d completed", ver)
                except Exception as e:
                    self.conn.rollback()
                    logger.error("Migration to v%d failed: %s", ver, e)
                    raise

    def _migrate_v23_drop_retired_block_payload_columns(self) -> None:
        """Remove retired block payload columns from the current schema.

        These columns previously carried mixed vendor/app state. Current code
        stores typed fields instead; non-empty legacy payload means the project
        needs an explicit repair path, not silent migration.
        """
        columns = self._table_columns("block")
        retired_columns = ("raw_payload_json", "app_payload_json")
        present = [col for col in retired_columns if col in columns]
        if not present:
            return
        for col in present:
            rows = self.conn.execute(f"SELECT id, {col} FROM block").fetchall()
            for row in rows:
                raw = str(row[col] or "").strip()
                if raw in {"", "{}"}:
                    continue
                raise ProjectDataError(
                    f"block.{col} contains retired payload; "
                    "open a clean project or run an explicit repair tool"
                )
        for col in present:
            self.conn.execute(f"ALTER TABLE block DROP COLUMN {col}")

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self.conn.execute(f"PRAGMA table_info({table})")}

    def _ensure_entity_uids(self) -> None:
        """Backfill stable business IDs and enforce per-entity uniqueness."""
        for table, kind, index_name in _ENTITY_UID_TABLES:
            rows = self.conn.execute(
                f"SELECT id, uid FROM {table} ORDER BY id"
            ).fetchall()
            seen: set[str] = set()
            for row in rows:
                uid = str(row["uid"] or "").strip()
                if not uid or uid in seen:
                    uid = _new_unique_uid(kind, seen)
                    self.conn.execute(
                        f"UPDATE {table} SET uid=? WHERE id=?",
                        (uid, row["id"]),
                    )
                seen.add(uid)
            self.conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}(uid)"
            )
        self.conn.commit()

    def _ensure_proof_line_states(self) -> None:
        """Ensure every persisted line has one proof truth row."""
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO proof_line_state ("
            "project_id, line_uid, final_text, final_text_set, proof_status, "
            "alignment_state, updated_at) "
            "SELECT p.project_id, l.uid, '', 0, 'unchecked', 'aligned', ? "
            "FROM line l "
            "JOIN block b ON b.id = l.block_id "
            "JOIN page p ON p.id = b.page_id "
            "WHERE l.uid <> ''",
            (now,),
        )
        self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self.flush_to_main_file()
            self._conn.close()
            self._conn = None

    def flush_to_main_file(self) -> None:
        """Checkpoint WAL pages so the .ocrproj file is externally readable."""
        if not self._conn:
            return
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            logger.warning("Project WAL checkpoint failed: %s", exc)

    def __enter__(self) -> "ProjectStore":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ProjectStore is not open. Call open() first.")
        return self._conn

    def _row_by_id_and_parent(
        self,
        cur: sqlite3.Cursor,
        table: str,
        object_id: int,
        parent_col: str,
        parent_id: int,
    ) -> sqlite3.Row | None:
        return cur.execute(
            f"SELECT id, uid FROM {table} WHERE id=? AND {parent_col}=?",
            (object_id, parent_id),
        ).fetchone()

    def _row_by_uid_and_parent(
        self,
        cur: sqlite3.Cursor,
        table: str,
        uid: str,
        parent_col: str,
        parent_id: int,
    ) -> sqlite3.Row | None:
        return cur.execute(
            f"SELECT id, uid FROM {table} WHERE uid=? AND {parent_col}=?",
            (uid, parent_id),
        ).fetchone()

    def _row_by_uid_in_project(
        self,
        cur: sqlite3.Cursor,
        table: str,
        uid: str,
        project_id: int,
    ) -> sqlite3.Row | None:
        if table == "page":
            return cur.execute(
                "SELECT id, uid FROM page WHERE uid=? AND project_id=?",
                (uid, project_id),
            ).fetchone()
        if table == "block":
            return cur.execute(
                "SELECT b.id, b.uid FROM block b "
                "JOIN page p ON p.id = b.page_id "
                "WHERE b.uid=? AND p.project_id=?",
                (uid, project_id),
            ).fetchone()
        if table == "line":
            return cur.execute(
                "SELECT l.id, l.uid FROM line l "
                "JOIN block b ON b.id = l.block_id "
                "JOIN page p ON p.id = b.page_id "
                "WHERE l.uid=? AND p.project_id=?",
                (uid, project_id),
            ).fetchone()
        if table == "char_":
            return cur.execute(
                "SELECT c.id, c.uid FROM char_ c "
                "JOIN line l ON l.id = c.line_id "
                "JOIN block b ON b.id = l.block_id "
                "JOIN page p ON p.id = b.page_id "
                "WHERE c.uid=? AND p.project_id=?",
                (uid, project_id),
            ).fetchone()
        raise ValueError(f"Unsupported uid lookup table: {table}")

    def _uid_exists(self, cur: sqlite3.Cursor, table: str, uid: str) -> bool:
        return cur.execute(
            f"SELECT 1 FROM {table} WHERE uid=? LIMIT 1",
            (uid,),
        ).fetchone() is not None

    def _fresh_db_uid(
        self,
        cur: sqlite3.Cursor,
        table: str,
        kind: str,
        reserved: set[str] | None = None,
    ) -> str:
        reserved = reserved or set()
        while True:
            uid = new_entity_uid(kind)
            if uid not in reserved and not self._uid_exists(cur, table, uid):
                return uid

    def _ensure_unique_child_uid(
        self,
        cur: sqlite3.Cursor,
        obj: object,
        *,
        kind: str,
        table: str,
        seen_uids: set[str],
    ) -> None:
        uid = ensure_entity_uid(getattr(obj, "uid", ""), kind)
        if uid in seen_uids:
            setattr(obj, "id", None)
            uid = self._fresh_db_uid(cur, table, kind, seen_uids)
        setattr(obj, "uid", uid)
        seen_uids.add(uid)

    def _prepare_entity_identity(
        self,
        cur: sqlite3.Cursor,
        obj: object,
        *,
        kind: str,
        table: str,
        parent_col: str,
        parent_id: int,
        project_id: int,
    ) -> None:
        raw_uid = str(getattr(obj, "uid", "") or "").strip()
        uid_was_missing = not raw_uid
        uid = ensure_entity_uid(raw_uid, kind)
        setattr(obj, "uid", uid)

        object_id = getattr(obj, "id", None)
        id_row = None
        if object_id is not None:
            id_row = self._row_by_id_and_parent(cur, table, object_id, parent_col, parent_id)

        uid_row = self._row_by_uid_and_parent(cur, table, uid, parent_col, parent_id)

        if id_row is not None and uid_row is not None:
            if id_row["id"] != uid_row["id"]:
                logger.warning(
                    "Stable uid recovered stale rowid: table=%s parent=%s:%s rowid=%s uid=%s",
                    table,
                    parent_col,
                    parent_id,
                    object_id,
                    uid,
                )
            setattr(obj, "id", uid_row["id"])
            setattr(obj, "uid", uid_row["uid"])
            return

        if uid_row is not None:
            setattr(obj, "id", uid_row["id"])
            setattr(obj, "uid", uid_row["uid"])
            return

        project_uid_row = self._row_by_uid_in_project(cur, table, uid, project_id)
        if project_uid_row is not None:
            if id_row is not None and id_row["id"] != project_uid_row["id"]:
                logger.warning(
                    "Stable uid moved across parent and ignored stale rowid: "
                    "table=%s parent=%s:%s rowid=%s uid=%s",
                    table,
                    parent_col,
                    parent_id,
                    object_id,
                    uid,
                )
            setattr(obj, "id", project_uid_row["id"])
            setattr(obj, "uid", project_uid_row["uid"])
            return

        if id_row is not None:
            db_uid = str(id_row["uid"] or "").strip()
            if uid_was_missing or db_uid == uid:
                setattr(obj, "id", id_row["id"])
                setattr(obj, "uid", db_uid or uid)
                return
            logger.warning(
                "Stable uid restored current row from conflicting uid: "
                "table=%s parent=%s:%s rowid=%s db_uid=%s incoming_uid=%s",
                table,
                parent_col,
                parent_id,
                object_id,
                db_uid,
                uid,
            )
            setattr(obj, "id", id_row["id"])
            setattr(obj, "uid", db_uid or uid)
            return

        setattr(obj, "id", None)
        if self._uid_exists(cur, table, uid):
            setattr(obj, "uid", self._fresh_db_uid(cur, table, kind))

    # ------------------------------------------------------------------ save (transactional upsert)

    def save_project(self, project: OcrProject) -> OcrProject:
        """保存或更新整个项目（事务化全量写入）。"""
        now = time.time()
        project.updated_at = now

        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")

            cur = conn.cursor()
            if project.id is None:
                cur.execute(
                    "INSERT INTO project (name, created_at, updated_at) VALUES (?, ?, ?)",
                    (project.name, project.created_at, project.updated_at),
                )
                project.id = cur.lastrowid
            else:
                cur.execute(
                    "UPDATE project SET name=?, updated_at=? WHERE id=?",
                    (project.name, project.updated_at, project.id),
                )

            # 获取当前所有 page id/uid，用于清理旧数据
            old_page_rows = conn.execute(
                "SELECT id, uid FROM page WHERE project_id=?", (project.id,)
            ).fetchall()
            old_page_ids = [r["id"] for r in old_page_rows]
            old_page_uid_by_id = {r["id"]: str(r["uid"] or "") for r in old_page_rows}

            save_seen_uids: dict[str, set[str]] = {
                "page": set(),
                "block": set(),
                "line": set(),
                "char": set(),
            }
            for page in project.pages:
                self._ensure_unique_child_uid(
                    cur,
                    page,
                    kind="page",
                    table="page",
                    seen_uids=save_seen_uids["page"],
                )
                self._save_page(cur, page, project.id, save_seen_uids=save_seen_uids)

            # 删除已移除的 page（级联删除 block/line/char）
            saved_page_ids = {p.id for p in project.pages if p.id is not None}
            for old_id in old_page_ids:
                if old_id not in saved_page_ids:
                    old_uid = old_page_uid_by_id.get(old_id, "")
                    if old_uid:
                        cur.execute(
                            "DELETE FROM layout_snapshot WHERE project_id=? AND page_uid=?",
                            (project.id, old_uid),
                        )
                    cur.execute("DELETE FROM page WHERE id=?", (old_id,))

            self._sync_project_proof_line_states_no_commit(cur, project)

            conn.commit()
            self.flush_to_main_file()
            project.db_path = self.db_path
            return project

        except Exception as e:
            conn.rollback()
            logger.error("Save project failed: %s", e)
            raise

    def _save_page(
        self,
        cur: sqlite3.Cursor,
        page: Page,
        project_id: int,
        *,
        save_seen_uids: dict[str, set[str]],
    ) -> None:
        try:
            validate_page_model(page)
        except ModelValidationError as exc:
            _raise_project_data_error(exc)
        self._prepare_entity_identity(
            cur,
            page,
            kind="page",
            table="page",
            parent_col="project_id",
            parent_id=project_id,
            project_id=project_id,
        )
        raw_layout_artifact_uid = self._save_raw_layout_artifact(cur, page, project_id)
        if page.id is None:
            cur.execute(
                "INSERT INTO page (uid, project_id, image_path, width, height, "
                "page_number, source_path, source_type, source_page_index, "
                "cache_image_path, thumbnail_path, status, error_message, "
                "ocr_invalidated_reason, raw_layout_artifact_uid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (page.uid, project_id, page.image_path, page.width, page.height,
                 page.page_number, page.source_path, page.source_type,
                 page.source_page_index, page.cache_image_path,
                 page.thumbnail_path, page.status.value, page.error_message,
                 page.ocr_invalidated_reason,
                 raw_layout_artifact_uid),
            )
            page.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE page SET image_path=?, width=?, height=?, page_number=?, "
                "source_path=?, source_type=?, source_page_index=?, "
                "cache_image_path=?, thumbnail_path=?, status=?, error_message=?, "
                "ocr_invalidated_reason=?, "
                "raw_layout_artifact_uid=?, project_id=? "
                "WHERE id=? AND uid=?",
                (page.image_path, page.width, page.height, page.page_number,
                 page.source_path, page.source_type, page.source_page_index,
                 page.cache_image_path, page.thumbnail_path, page.status.value,
                 page.error_message,
                 page.ocr_invalidated_reason,
                 raw_layout_artifact_uid,
                 project_id, page.id, page.uid),
            )
            if cur.rowcount != 1:
                page.id = None
                self._save_page(cur, page, project_id, save_seen_uids=save_seen_uids)
                return

        self._save_layout_edit_events(cur, page, project_id)

        old_block_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM block WHERE page_id=?", (page.id,)
            ).fetchall()
        }
        old_block_uid_by_id = {
            r["id"]: str(r["uid"] or "")
            for r in cur.execute(
                "SELECT id, uid FROM block WHERE page_id=?", (page.id,)
            ).fetchall()
        }
        saved_block_ids: set[int] = set()
        for block in self._sync_layout_projection_from_snapshot(page):
            self._ensure_unique_child_uid(
                cur,
                block,
                kind="block",
                table="block",
                seen_uids=save_seen_uids["block"],
            )
            self._save_block(cur, block, page.id, project_id, save_seen_uids=save_seen_uids)
            if block.id is not None:
                saved_block_ids.add(block.id)

        for old_id in old_block_ids - saved_block_ids:
            old_uid = old_block_uid_by_id.get(old_id, "")
            if old_uid:
                cur.execute(
                    "DELETE FROM block_origin WHERE project_id=? AND block_uid=?",
                    (project_id, old_uid),
                )
            cur.execute("DELETE FROM block WHERE id=?", (old_id,))
        self._save_layout_snapshot(cur, page, project_id)

    @staticmethod
    def _sync_layout_projection_from_snapshot(
        page: Page,
    ) -> list[Block]:
        snapshot = layout_snapshot_for_page(page)
        if snapshot is None:
            raise ProjectDataError(
                f"page {page.uid or page.page_number!r} has no layout snapshot in normal save path"
            )
        runtime_blocks = page_layout_blocks(page)
        runtime_uid_counts: dict[str, int] = {}
        runtime_by_uid: dict[str, Block] = {}
        for block in runtime_blocks:
            uid = str(block.uid or "")
            if not uid:
                continue
            runtime_uid_counts[uid] = runtime_uid_counts.get(uid, 0) + 1
            runtime_by_uid.setdefault(uid, block)

        blocks: list[Block] = []
        for index, snapshot_block in enumerate(snapshot.blocks):
            block: Block | None = None
            if index < len(runtime_blocks) and runtime_blocks[index].uid == snapshot_block.uid:
                block = runtime_blocks[index]
            elif runtime_uid_counts.get(snapshot_block.uid) == 1:
                block = runtime_by_uid[snapshot_block.uid]
            if block is None:
                block = Block(
                    block_type=snapshot_block.block_type,
                    bbox=snapshot_block.bbox,
                    order=snapshot_block.order,
                    note=snapshot_block.note,
                    source_label=snapshot_block.source_label,
                    origin=snapshot_block.origin,
                    ocr_policy=snapshot_block.ocr_policy,
                    uid=snapshot_block.uid,
                )
            set_layout_block_type(block, snapshot_block.block_type)
            set_layout_block_bbox(block, snapshot_block.bbox)
            set_layout_block_order(block, snapshot_block.order)
            set_layout_block_source_label(block, snapshot_block.source_label)
            block.origin = snapshot_block.origin
            set_layout_block_ocr_policy(block, snapshot_block.ocr_policy)
            set_layout_block_note(block, snapshot_block.note)
            blocks.append(block)
        replace_page_layout_blocks(page, blocks)
        return blocks

    def _save_layout_edit_events(
        self,
        cur: sqlite3.Cursor,
        page: Page,
        project_id: int,
    ) -> None:
        cur.execute(
            "DELETE FROM layout_edit_event WHERE project_id=? AND page_uid=?",
            (project_id, page.uid),
        )
        seen_uids: set[str] = set()
        for event in page.layout_edit_events:
            event.page_uid = page.uid
            event.uid = ensure_entity_uid(event.uid, "layoutedit")
            if event.uid in seen_uids:
                event.uid = self._fresh_db_uid(cur, "layout_edit_event", "layoutedit", seen_uids)
            seen_uids.add(event.uid)
            cur.execute(
                "INSERT INTO layout_edit_event ("
                "uid, project_id, page_uid, target_uid, op, before_json, "
                "after_json, actor, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.uid,
                    project_id,
                    event.page_uid,
                    event.target_uid,
                    event.op,
                    json.dumps(event.before, ensure_ascii=False),
                    json.dumps(event.after, ensure_ascii=False),
                    event.actor,
                    event.created_at,
                ),
            )
            event.id = cur.lastrowid

    def _save_layout_snapshot(
        self,
        cur: sqlite3.Cursor,
        page: Page,
        project_id: int,
    ) -> None:
        snapshot = layout_snapshot_for_page(page)
        if snapshot is None:
            raise ProjectDataError(
                f"page {page.uid or page.page_number!r} has no layout snapshot in normal save path"
            )
        cur.execute(
            "INSERT INTO layout_snapshot ("
            "project_id, page_uid, artifact_uid, source_engine, source_run_id, "
            "blocks_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, page_uid) DO UPDATE SET "
            "artifact_uid=excluded.artifact_uid, "
            "source_engine=excluded.source_engine, "
            "source_run_id=excluded.source_run_id, "
            "blocks_json=excluded.blocks_json, "
            "updated_at=excluded.updated_at",
            (
                project_id,
                page.uid,
                snapshot.artifact_uid,
                snapshot.source_engine,
                snapshot.source_run_id,
                json.dumps(_layout_snapshot_blocks_to_json(snapshot), ensure_ascii=False),
                time.time(),
            ),
        )

    def _save_raw_layout_artifact(
        self,
        cur: sqlite3.Cursor,
        page: Page,
        project_id: int,
    ) -> str:
        artifact = page.raw_layout_artifact
        if artifact is None:
            return ""
        artifact.page_uid = page.uid
        if not artifact.uid:
            artifact.uid = new_entity_uid("rawocr")
        payload = (
            artifact.uid,
            project_id,
            artifact.page_uid,
            artifact.engine,
            artifact.engine_version,
            artifact.run_id,
            artifact.artifact_path,
            artifact.artifact_hash,
            json.dumps(_dict_list(artifact.records, field="raw_ocr_artifact.records"), ensure_ascii=False),
            json.dumps(_route_attachments_to_json_dict(artifact.route_attachments), ensure_ascii=False),
            artifact.created_at,
        )
        if artifact.id is None:
            row = cur.execute(
                "SELECT id FROM raw_ocr_artifact WHERE project_id=? AND uid=?",
                (project_id, artifact.uid),
            ).fetchone()
            if row is not None:
                artifact.id = row["id"]
        if artifact.id is None:
            cur.execute(
                "INSERT INTO raw_ocr_artifact ("
                "uid, project_id, page_uid, engine, engine_version, run_id, "
                "artifact_path, artifact_hash, records_json, route_attachments_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
            artifact.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE raw_ocr_artifact SET page_uid=?, engine=?, engine_version=?, "
                "run_id=?, artifact_path=?, artifact_hash=?, records_json=?, "
                "route_attachments_json=?, created_at=? "
                "WHERE id=? AND project_id=? AND uid=?",
                (
                    artifact.page_uid,
                    artifact.engine,
                    artifact.engine_version,
                    artifact.run_id,
                    artifact.artifact_path,
                    artifact.artifact_hash,
                    json.dumps(_dict_list(artifact.records, field="raw_ocr_artifact.records"), ensure_ascii=False),
                    json.dumps(_route_attachments_to_json_dict(artifact.route_attachments), ensure_ascii=False),
                    artifact.created_at,
                    artifact.id,
                    project_id,
                    artifact.uid,
                ),
            )
            if cur.rowcount != 1:
                artifact.id = None
                return self._save_raw_layout_artifact(cur, page, project_id)
        return artifact.uid

    def _save_block(
        self,
        cur: sqlite3.Cursor,
        block: Block,
        page_id: int,
        project_id: int,
        *,
        save_seen_uids: dict[str, set[str]],
    ) -> None:
        self._prepare_entity_identity(
            cur,
            block,
            kind="block",
            table="block",
            parent_col="page_id",
            parent_id=page_id,
            project_id=project_id,
        )
        bb = block.bbox
        try:
            validate_block_model(block)
        except ModelValidationError as exc:
            _raise_project_data_error(exc)
        values = (
            page_id, block.block_type.value, bb.x, bb.y, bb.w, bb.h, block.order,
            block.source.value, block.ocr_policy.value,
            block.note, block.source_label,
            json.dumps(block.paddle_binding.to_dict() if block.paddle_binding else {}, ensure_ascii=False),
            block.ocr_invalidated_reason,
            json.dumps(block.ocr_audit, ensure_ascii=False),
            json.dumps(
                _dict_list(block.table_text_layer_cells, field="block.table_text_layer_cells"),
                ensure_ascii=False,
            ),
        )
        if block.id is None:
            cur.execute(
                "INSERT INTO block (uid, page_id, block_type, x, y, w, h, block_order, "
                "source, ocr_policy, note, source_label, "
                "paddle_binding_json, ocr_invalidated_reason, "
                "ocr_audit_json, table_text_layer_cells_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (block.uid, *values),
            )
            block.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE block SET page_id=?, block_type=?, x=?, y=?, w=?, h=?, "
                "block_order=?, source=?, ocr_policy=?, note=?, "
                "source_label=?, paddle_binding_json=?, "
                "ocr_invalidated_reason=?, ocr_audit_json=?, "
                "table_text_layer_cells_json=? "
                "WHERE id=? AND uid=?",
                (*values, block.id, block.uid),
            )
            if cur.rowcount != 1:
                block.id = None
                self._save_block(cur, block, page_id, project_id, save_seen_uids=save_seen_uids)
                return

        self._save_block_origin(cur, block, project_id)

        old_line_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM line WHERE block_id=?", (block.id,)
            ).fetchall()
        }
        saved_line_ids: set[int] = set()
        for line in block_ocr_line_observations_by_uid(block.uid):
            self._ensure_unique_child_uid(
                cur,
                line,
                kind="line",
                table="line",
                seen_uids=save_seen_uids["line"],
            )
            self._save_line(cur, line, block.id, project_id, save_seen_uids=save_seen_uids)
            if line.id is not None:
                saved_line_ids.add(line.id)

        for old_id in old_line_ids - saved_line_ids:
            cur.execute("DELETE FROM line WHERE id=?", (old_id,))

    def _save_block_origin(
        self,
        cur: sqlite3.Cursor,
        block: Block,
        project_id: int,
    ) -> None:
        origin = block.origin or _origin_from_current_block(block)
        block.origin = origin
        bb = origin.original_bbox
        x, y, w, h = (bb.x, bb.y, bb.w, bb.h) if bb else (None, None, None, None)
        cur.execute(
            "INSERT INTO block_origin ("
            "project_id, block_uid, created_by, source_engine, source_run_id, "
            "source_label, source_confidence, original_x, original_y, original_w, "
            "original_h, original_kind, raw_artifact_uid, raw_json_path, raw_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, block_uid) DO UPDATE SET "
            "created_by=excluded.created_by, "
            "source_engine=excluded.source_engine, "
            "source_run_id=excluded.source_run_id, "
            "source_label=excluded.source_label, "
            "source_confidence=excluded.source_confidence, "
            "original_x=excluded.original_x, "
            "original_y=excluded.original_y, "
            "original_w=excluded.original_w, "
            "original_h=excluded.original_h, "
            "original_kind=excluded.original_kind, "
            "raw_artifact_uid=excluded.raw_artifact_uid, "
            "raw_json_path=excluded.raw_json_path, "
            "raw_index=excluded.raw_index",
            (
                project_id,
                block.uid,
                origin.created_by,
                origin.source_engine,
                origin.source_run_id,
                origin.source_label,
                origin.source_confidence,
                x,
                y,
                w,
                h,
                origin.original_kind.value if origin.original_kind else "",
                origin.raw_artifact_uid,
                origin.raw_json_path,
                origin.raw_index,
            ),
        )

    def _sync_project_proof_line_states_no_commit(
        self,
        cur: sqlite3.Cursor,
        project: OcrProject,
    ) -> None:
        """Persist in-memory proof state into the proof truth table."""
        if project.id is None:
            return
        current_line_uids: set[str] = set()
        for occurrence in iter_project_ocr_line_observation_occurrences(project):
            line = occurrence.line
            if not str(line.uid or "").strip():
                continue
            current_line_uids.add(line.uid)
            self._upsert_proof_line_state_no_commit(cur, project.id, line)
        if not current_line_uids:
            cur.execute("DELETE FROM proof_line_state WHERE project_id=?", (project.id,))
            return
        placeholders = ",".join("?" for _ in current_line_uids)
        cur.execute(
            "DELETE FROM proof_line_state "
            f"WHERE project_id=? AND line_uid NOT IN ({placeholders})",
            (project.id, *sorted(current_line_uids)),
        )

    def _upsert_proof_line_state_no_commit(
        self,
        cur: sqlite3.Cursor,
        project_id: int,
        line: Line,
    ) -> None:
        if not str(line.uid or "").strip():
            return
        contract = line_text_contract(line)
        state = contract.proof_state
        display_text = state.final_text if state.final_text_set else (state.final_text or contract.text)
        now = time.time()
        cur.execute(
            "INSERT INTO proof_line_state ("
            "project_id, line_uid, final_text, final_text_set, proof_status, "
            "alignment_state, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, line_uid) DO UPDATE SET "
            "final_text=excluded.final_text, "
            "final_text_set=excluded.final_text_set, "
            "proof_status=excluded.proof_status, "
            "alignment_state=excluded.alignment_state, "
            "updated_at=excluded.updated_at",
            (
                project_id,
                line.uid,
                state.final_text,
                int(state.final_text_set),
                state.proof_status.value,
                _proof_alignment_state(line, display_text),
                now,
            ),
        )

    def _save_line(
        self,
        cur: sqlite3.Cursor,
        line: Line,
        block_id: int,
        project_id: int,
        *,
        save_seen_uids: dict[str, set[str]],
    ) -> None:
        self._prepare_entity_identity(
            cur,
            line,
            kind="line",
            table="line",
            parent_col="block_id",
            parent_id=block_id,
            project_id=project_id,
        )
        bb = line_ocr_bbox(line)
        contract = line_text_contract(line)
        values = (
            block_id, contract.text, contract.confidence,
            bb.x, bb.y, bb.w, bb.h,
            contract.ocr_text,
            _review_flags_to_json(list(line_ocr_review_flags(line))),
        )
        if line.id is None:
            cur.execute(
                "INSERT INTO line (uid, block_id, text, confidence, "
                "x, y, w, h, ocr_text, review_flags_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (line.uid, *values),
            )
            line.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE line SET block_id=?, text=?, "
                "confidence=?, x=?, y=?, w=?, h=?, ocr_text=?, review_flags_json=? "
                "WHERE id=? AND uid=?",
                (*values, line.id, line.uid),
            )
            if cur.rowcount != 1:
                line.id = None
                self._save_line(cur, line, block_id, project_id, save_seen_uids=save_seen_uids)
                return

        old_char_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM char_ WHERE line_id=?", (line.id,)
            ).fetchall()
        }
        saved_char_ids: set[int] = set()
        for char in line_ocr_chars_by_uid(line.uid):
            self._ensure_unique_child_uid(
                cur,
                char,
                kind="char",
                table="char_",
                seen_uids=save_seen_uids["char"],
            )
            self._save_char(
                cur,
                char,
                line.id,
                project_id,
            )
            if char.id is not None:
                saved_char_ids.add(char.id)

        for old_id in old_char_ids - saved_char_ids:
            cur.execute("DELETE FROM char_ WHERE id=?", (old_id,))

    def _save_char(
        self,
        cur: sqlite3.Cursor,
        char: Char,
        line_id: int,
        project_id: int,
        *,
        allow_project_move: bool = True,
    ) -> None:
        if allow_project_move:
            self._prepare_entity_identity(
                cur,
                char,
                kind="char",
                table="char_",
                parent_col="line_id",
                parent_id=line_id,
                project_id=project_id,
            )
        else:
            self._prepare_char_identity_for_line_sync(cur, char, line_id)
        bb = char.bbox
        x, y, w, h = (bb.x, bb.y, bb.w, bb.h) if bb else (None, None, None, None)
        values = (
            line_id,
            char.char,
            char.confidence,
            x,
            y,
            w,
            h,
            char.bbox_source,
            char.bbox_granularity,
            char.token_text,
        )
        if char.id is None:
            cur.execute(
                "INSERT INTO char_ (uid, line_id, char, confidence, x, y, w, h, "
                "bbox_source, bbox_granularity, token_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (char.uid, *values),
            )
            char.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE char_ SET line_id=?, char=?, confidence=?, x=?, y=?, w=?, h=?, "
                "bbox_source=?, bbox_granularity=?, token_text=? WHERE id=? AND uid=?",
                (*values, char.id, char.uid),
            )
            if cur.rowcount != 1:
                char.id = None
                self._save_char(
                    cur,
                    char,
                    line_id,
                    project_id,
                    allow_project_move=allow_project_move,
                )

    def _prepare_char_identity_for_line_sync(
        self,
        cur: sqlite3.Cursor,
        char: Char,
        line_id: int,
    ) -> None:
        """Prepare char identity for proof-line updates without cross-line moves.

        Full project saves support moving a Char between lines. Incremental
        proof persistence is different: it only syncs one line's current text
        facts and must not adopt a matching UID row from another line.
        """
        raw_uid = str(getattr(char, "uid", "") or "").strip()
        uid_was_missing = not raw_uid
        uid = ensure_entity_uid(raw_uid, "char")
        char.uid = uid

        id_row = None
        if char.id is not None:
            id_row = self._row_by_id_and_parent(cur, "char_", char.id, "line_id", line_id)
        uid_row = self._row_by_uid_and_parent(cur, "char_", uid, "line_id", line_id)

        if id_row is not None and uid_row is not None:
            char.id = uid_row["id"]
            char.uid = uid_row["uid"]
            return
        if uid_row is not None:
            char.id = uid_row["id"]
            char.uid = uid_row["uid"]
            return
        if id_row is not None:
            db_uid = str(id_row["uid"] or "").strip()
            if uid_was_missing or db_uid == uid or self._uid_exists(cur, "char_", uid):
                char.id = id_row["id"]
                char.uid = db_uid or uid
                return
            char.id = id_row["id"]
            char.uid = db_uid or uid
            return

        char.id = None
        if self._uid_exists(cur, "char_", uid):
            char.uid = self._fresh_db_uid(cur, "char_", "char")

    # ------------------------------------------------------------------ update proof lines

    def update_proof_lines(self, line_updates: list[tuple[Line, bool]]) -> None:
        """Persist proof line updates as one transaction.

        ``line_updates`` is a list of ``(line, write_chars)``.  The proof
        change, not the individual line, is the transaction boundary.
        """
        if not line_updates:
            return
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            cur = self.conn.cursor()
            for line, write_chars in line_updates:
                project_id = self._project_id_for_line(cur, line)
                self._update_line_no_commit(line)
                self._upsert_proof_line_state_no_commit(cur, project_id, line)
                if write_chars:
                    self._sync_line_chars_no_commit(cur, line, project_id)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _update_line_no_commit(self, line: Line) -> None:
        if line.id is None:
            raise RuntimeError("Line update requires a rowid and stable uid")
        if not str(line.uid or "").strip():
            raise RuntimeError(f"Line update requires stable uid: id={line.id!r}")
        line.uid = ensure_entity_uid(line.uid, "line")
        contract = line_text_contract(line)
        cur = self.conn.execute(
            "UPDATE line SET text=?, ocr_text=?, "
            "review_flags_json=? WHERE id=? AND uid=?",
            (
                contract.text,
                contract.ocr_text,
                _review_flags_to_json(list(line_ocr_review_flags(line))),
                line.id,
                line.uid,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"Line update failed or matched multiple rows: id={line.id!r} uid={line.uid!r}"
            )

    def _project_id_for_line(self, cur: sqlite3.Cursor, line: Line) -> int:
        if line.id is None or not str(line.uid or "").strip():
            raise RuntimeError("Line project lookup requires rowid and stable uid")
        row = cur.execute(
            "SELECT p.project_id FROM line l "
            "JOIN block b ON b.id = l.block_id "
            "JOIN page p ON p.id = b.page_id "
            "WHERE l.id=? AND l.uid=?",
            (line.id, line.uid),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"Line project lookup failed: id={line.id!r} uid={line.uid!r}"
            )
        return int(row["project_id"])

    def _sync_line_chars_no_commit(
        self,
        cur: sqlite3.Cursor,
        line: Line,
        project_id: int,
    ) -> None:
        if line.id is None:
            raise RuntimeError("Char sync requires persisted line id")
        old_char_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM char_ WHERE line_id=?", (line.id,)
            ).fetchall()
        }
        saved_char_ids: set[int] = set()
        seen_uids: set[str] = set()
        for char in line_ocr_chars_by_uid(line.uid):
            self._ensure_unique_child_uid(
                cur,
                char,
                kind="char",
                table="char_",
                seen_uids=seen_uids,
            )
            self._save_char(
                cur,
                char,
                line.id,
                project_id,
                allow_project_move=False,
            )
            if char.id is not None:
                saved_char_ids.add(char.id)
        for old_id in old_char_ids - saved_char_ids:
            cur.execute("DELETE FROM char_ WHERE id=?", (old_id,))

    # ------------------------------------------------------------------ load

    def load_project(self, project_id: int = 1) -> Optional[OcrProject]:
        """从数据库加载项目（含所有页面/块/行/字符）。"""
        row = self.conn.execute(
            "SELECT * FROM project WHERE id=?", (project_id,)
        ).fetchone()
        if row is None:
            return None

        project = OcrProject(
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            id=row["id"],
            db_path=self.db_path,
        )

        pages = self.conn.execute(
            "SELECT * FROM page WHERE project_id=? ORDER BY page_number",
            (project.id,),
        ).fetchall()
        for pr in pages:
            raw_layout_artifact = self._load_raw_layout_artifact(
                project.id,
                str(pr["raw_layout_artifact_uid"] or ""),
            )
            page = Page(
                image_path=pr["image_path"],
                width=pr["width"],
                height=pr["height"],
                page_number=pr["page_number"],
                id=pr["id"],
                uid=pr["uid"],
                source_path=pr["source_path"],
                source_type=pr["source_type"],
                source_page_index=pr["source_page_index"],
                cache_image_path=pr["cache_image_path"],
                thumbnail_path=pr["thumbnail_path"],
                status=_enum_from_db(PageStatus, pr["status"], field="page.status"),
                error_message=pr["error_message"],
                ocr_invalidated_reason=pr["ocr_invalidated_reason"],
                raw_layout_artifact=raw_layout_artifact,
                layout_edit_events=self._load_layout_edit_events(project.id, str(pr["uid"] or "")),
            )
            replace_page_layout_blocks(page, self._load_blocks(page.id, project.id))
            snapshot = self._load_layout_snapshot(project.id, page.uid)
            if snapshot is not None:
                set_layout_snapshot_for_page(page, snapshot)
                self._sync_layout_projection_from_snapshot(page)
            else:
                raise ProjectDataError(
                    f"page {page.uid or page.page_number!r} has no persisted layout snapshot"
                )
            reconcile_page_ocr_done_from_result(page)
            project.pages.append(page)

        self._apply_proof_line_states(project)
        return project

    def _load_raw_layout_artifact(
        self,
        project_id: int,
        artifact_uid: str,
    ) -> RawOcrArtifact | None:
        if not artifact_uid:
            return None
        row = self.conn.execute(
            "SELECT * FROM raw_ocr_artifact WHERE project_id=? AND uid=?",
            (project_id, artifact_uid),
        ).fetchone()
        if row is None:
            return None
        return RawOcrArtifact(
            id=row["id"],
            uid=row["uid"],
            page_uid=row["page_uid"],
            engine=row["engine"],
            engine_version=row["engine_version"],
            run_id=row["run_id"],
            artifact_path=row["artifact_path"],
            artifact_hash=row["artifact_hash"],
            records=_dict_list(
                _json_to_list(row["records_json"], field="raw_ocr_artifact.records_json"),
                field="raw_ocr_artifact.records_json",
            ),
            route_attachments=_json_to_route_attachments(
                row["route_attachments_json"] if "route_attachments_json" in row.keys() else "{}",
                field="raw_ocr_artifact.route_attachments_json",
            ),
            created_at=row["created_at"],
        )

    def _load_layout_snapshot(
        self,
        project_id: int,
        page_uid: str,
    ) -> LayoutSnapshot | None:
        if not page_uid:
            return None
        row = self.conn.execute(
            "SELECT * FROM layout_snapshot WHERE project_id=? AND page_uid=?",
            (project_id, page_uid),
        ).fetchone()
        if row is None:
            return None
        raw_blocks = _json_to_list(row["blocks_json"], field="layout_snapshot.blocks_json")
        blocks = tuple(
            _layout_block_snapshot_from_json(block, index=index)
            for index, block in enumerate(raw_blocks)
        )
        return LayoutSnapshot(
            page_uid=page_uid,
            artifact_uid=row["artifact_uid"],
            source_engine=row["source_engine"],
            source_run_id=row["source_run_id"],
            blocks=blocks,
        )

    def _load_layout_edit_events(
        self,
        project_id: int,
        page_uid: str,
    ) -> list[LayoutEditEvent]:
        if not page_uid:
            return []
        rows = self.conn.execute(
            "SELECT * FROM layout_edit_event WHERE project_id=? AND page_uid=? "
            "ORDER BY created_at, id",
            (project_id, page_uid),
        ).fetchall()
        events: list[LayoutEditEvent] = []
        for row in rows:
            events.append(
                LayoutEditEvent(
                    id=row["id"],
                    uid=row["uid"],
                    page_uid=row["page_uid"],
                    target_uid=row["target_uid"],
                    op=row["op"],
                    before=_json_to_dict(row["before_json"], field="layout_edit_event.before_json"),
                    after=_json_to_dict(row["after_json"], field="layout_edit_event.after_json"),
                    actor=row["actor"],
                    created_at=row["created_at"],
                )
            )
        return events

    def _apply_proof_line_states(self, project: OcrProject) -> None:
        """Apply persisted proof truth to runtime line state."""
        if project.id is None:
            return
        rows = self.conn.execute(
            "SELECT line_uid, final_text, final_text_set, proof_status "
            "FROM proof_line_state WHERE project_id=?",
            (project.id,),
        ).fetchall()
        states = {str(row["line_uid"] or ""): row for row in rows}
        if not states:
            return
        for occurrence in iter_project_ocr_line_observation_occurrences(project):
            line = occurrence.line
            row = states.get(line.uid)
            if row is None:
                continue
            proof_status = _enum_from_db(
                ProofStatus,
                row["proof_status"],
                field="proof_line_state.proof_status",
            )
            apply_line_proof_state(
                line,
                ProofLineState(
                    line_uid=line.uid,
                    final_text=row["final_text"],
                    final_text_set=bool(row["final_text_set"]),
                    proof_status=proof_status,
                )
            )

    def _load_blocks(self, page_id: int, project_id: int) -> List[Block]:
        rows = self.conn.execute(
            "SELECT * FROM block WHERE page_id=? ORDER BY block_order", (page_id,)
        ).fetchall()
        blocks = []
        for r in rows:
            paddle_binding_payload = (
                _json_to_dict(r["paddle_binding_json"], field="block.paddle_binding_json")
                if "paddle_binding_json" in r.keys()
                else {}
            )
            ocr_audit_payload = (
                _json_to_dict(r["ocr_audit_json"], field="block.ocr_audit_json")
                if "ocr_audit_json" in r.keys()
                else {}
            )
            table_text_layer_cells = (
                _dict_list(
                    _json_to_list(
                        r["table_text_layer_cells_json"],
                        field="block.table_text_layer_cells_json",
                    ),
                    field="block.table_text_layer_cells_json",
                )
                if "table_text_layer_cells_json" in r.keys()
                else []
            )
            ocr_invalidated_reason = (
                str(r["ocr_invalidated_reason"] or "")
                if "ocr_invalidated_reason" in r.keys()
                else ""
            )
            try:
                paddle_binding = PaddleBinding.from_dict(paddle_binding_payload)
            except ValueError as exc:
                raise ProjectDataError(f"block.paddle_binding_json {exc}") from exc
            origin = self._load_block_origin(project_id, str(r["uid"] or ""))
            block = Block(
                block_type=_enum_from_db(BlockType, r["block_type"], field="block.block_type"),
                bbox=_bbox_from_db(r, fields=("x", "y", "w", "h"), field="block"),
                order=r["block_order"],
                id=r["id"],
                uid=r["uid"],
                source=_enum_from_db(BlockSource, r["source"], field="block.source"),
                ocr_policy=_enum_from_db(OcrPolicy, r["ocr_policy"], field="block.ocr_policy"),
                note=r["note"],
                source_label=r["source_label"],
                origin=origin,
                paddle_binding=paddle_binding,
                ocr_invalidated_reason=ocr_invalidated_reason,
                ocr_audit=ocr_audit_payload,
                table_text_layer_cells=table_text_layer_cells,
            )
            replace_block_ocr_line_observations(block.uid, self._load_lines(block.id))
            blocks.append(block)
        return blocks

    def _load_block_origin(self, project_id: int, block_uid: str) -> BlockOrigin | None:
        if not block_uid:
            return None
        row = self.conn.execute(
            "SELECT * FROM block_origin WHERE project_id=? AND block_uid=?",
            (project_id, block_uid),
        ).fetchone()
        if row is None:
            return None
        bbox = _bbox_from_db(
            row,
            fields=("original_x", "original_y", "original_w", "original_h"),
            field="block_origin.original_bbox",
            optional=True,
        )
        kind = None
        raw_kind = str(row["original_kind"] or "")
        if raw_kind:
            kind = _enum_from_db(
                BlockType,
                raw_kind,
                field="block_origin.original_kind",
            )
        return BlockOrigin(
            created_by=row["created_by"],
            source_engine=row["source_engine"],
            source_run_id=row["source_run_id"],
            source_label=row["source_label"],
            source_confidence=row["source_confidence"],
            original_bbox=bbox,
            original_kind=kind,
            raw_artifact_uid=row["raw_artifact_uid"],
            raw_json_path=row["raw_json_path"],
            raw_index=row["raw_index"],
        )

    def _load_lines(self, block_id: int) -> List[Line]:
        rows = self.conn.execute(
            "SELECT * FROM line WHERE block_id=?", (block_id,)
        ).fetchall()
        lines = []
        for r in rows:
            line = Line(
                text=r["text"],
                confidence=r["confidence"],
                bbox=_bbox_from_db(r, fields=("x", "y", "w", "h"), field="line"),
                id=r["id"],
                uid=r["uid"],
                ocr_text=r["ocr_text"] or r["text"],
                review_flags=_json_to_review_flags(
                    r["review_flags_json"],
                    field="line.review_flags_json",
                ),
            )
            replace_line_ocr_char_observations(line.uid, self._load_chars(line.id))
            lines.append(line)
        return lines

    def _load_chars(self, line_id: int) -> List[Char]:
        rows = self.conn.execute(
            "SELECT * FROM char_ WHERE line_id=?", (line_id,)
        ).fetchall()
        chars = []
        for r in rows:
            bbox = _bbox_from_db(r, fields=("x", "y", "w", "h"), field="char", optional=True)
            chars.append(Char(
                char=r["char"],
                confidence=r["confidence"],
                bbox=bbox,
                id=r["id"],
                uid=r["uid"],
                bbox_source=r["bbox_source"],
                bbox_granularity=r["bbox_granularity"],
                token_text=r["token_text"],
            ))
        return chars

    # ------------------------------------------------------------------ list projects

    def list_projects(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM project ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ settings

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ operation log

    def log_operation(
        self,
        project_id: int,
        action: str,
        object_type: str = "",
        object_id: int | None = None,
        page_id: int | None = None,
        payload: dict | None = None,
        *,
        object_uid: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO operation_log (project_id, page_id, object_type, object_id, "
            "object_uid, action, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, page_id, object_type, object_id,
             object_uid, action, json.dumps(payload or {}, ensure_ascii=False), time.time()),
        )
        self.conn.commit()
