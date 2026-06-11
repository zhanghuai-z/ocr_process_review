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
    BBox, Block, BlockSource, BlockType, Char, Line,
    LlmReviewStatus, OcrProject, Page, PageStatus, ProofStatus,
)

from app.core.logging import get_logger, APP_VERSION, SCHEMA_VERSION

logger = get_logger(__name__)

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
    ppvl_parsing_res_list_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS block (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES page(id) ON DELETE CASCADE,
    block_type      TEXT    NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL,
    block_order     INTEGER NOT NULL DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT 'auto_layout',
    is_locked       INTEGER NOT NULL DEFAULT 0,
    recognizable    INTEGER NOT NULL DEFAULT 1,
    note            TEXT    NOT NULL DEFAULT '',
    source_label    TEXT    NOT NULL DEFAULT '',
    raw_payload_json TEXT   NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS line (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id          INTEGER NOT NULL REFERENCES block(id) ON DELETE CASCADE,
    text              TEXT    NOT NULL DEFAULT '',
    final_text        TEXT    NOT NULL DEFAULT '',
    original_text     TEXT    NOT NULL DEFAULT '',
    confidence        REAL    NOT NULL DEFAULT 0.0,
    proof_status      TEXT    NOT NULL DEFAULT 'unchecked',
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL,
    ocr_text          TEXT    NOT NULL DEFAULT '',
    llm_suggestion    TEXT    NOT NULL DEFAULT '',
    llm_reason        TEXT    NOT NULL DEFAULT '',
    llm_review_status TEXT    NOT NULL DEFAULT 'disabled',
    review_flags_json TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS char_ (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id     INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    char        TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    bbox_source TEXT    NOT NULL DEFAULT '',
    bbox_granularity TEXT NOT NULL DEFAULT '',
    token_text  TEXT    NOT NULL DEFAULT ''
);

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
        "ALTER TABLE block ADD COLUMN is_locked INTEGER NOT NULL DEFAULT 0;",
        "ALTER TABLE block ADD COLUMN recognizable INTEGER NOT NULL DEFAULT 1;",
        "ALTER TABLE block ADD COLUMN note TEXT NOT NULL DEFAULT '';",
        # line table additions
        "ALTER TABLE line ADD COLUMN ocr_text TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE line ADD COLUMN llm_suggestion TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE line ADD COLUMN llm_reason TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE line ADD COLUMN llm_review_status TEXT NOT NULL DEFAULT 'disabled';",
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
    4: [
        "ALTER TABLE line ADD COLUMN final_text TEXT NOT NULL DEFAULT '';",
        "UPDATE line SET final_text = text WHERE final_text = '';",
    ],
    5: [
        "ALTER TABLE page ADD COLUMN ppvl_parsing_res_list_json TEXT NOT NULL DEFAULT '[]';",
    ],
    6: [
        "ALTER TABLE block ADD COLUMN source_label TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE block ADD COLUMN raw_payload_json TEXT NOT NULL DEFAULT '{}';",
    ],
    7: [
        "ALTER TABLE page ADD COLUMN ocr_invalidated_reason TEXT NOT NULL DEFAULT '';",
    ],
}


# ------------------------------------------------------------------- helpers

def _row_to_blocks(row: sqlite3.Row) -> dict:
    return dict(row)


def _review_flags_to_json(flags: list[str]) -> str:
    return json.dumps(flags, ensure_ascii=False)


def _json_to_review_flags(s: str) -> list[str]:
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def _json_to_list(s: str) -> list:
    if not s:
        return []
    try:
        value = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _json_to_dict(s: str) -> dict:
    if not s:
        return {}
    try:
        value = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


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

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

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

            # 获取当前所有 page id，用于清理旧数据
            old_page_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM page WHERE project_id=?", (project.id,)
                ).fetchall()
            ]

            for page in project.pages:
                self._save_page(cur, page, project.id)

            # 删除已移除的 page（级联删除 block/line/char）
            saved_page_ids = {p.id for p in project.pages if p.id is not None}
            for old_id in old_page_ids:
                if old_id not in saved_page_ids:
                    cur.execute("DELETE FROM page WHERE id=?", (old_id,))

            conn.commit()
            project.db_path = self.db_path
            return project

        except Exception as e:
            conn.rollback()
            logger.error("Save project failed: %s", e)
            raise

    def _save_page(self, cur: sqlite3.Cursor, page: Page, project_id: int) -> None:
        if page.id is None:
            cur.execute(
                "INSERT INTO page (project_id, image_path, width, height, "
                "page_number, source_path, source_type, source_page_index, "
                "cache_image_path, thumbnail_path, status, error_message, "
                "ocr_invalidated_reason, ppvl_parsing_res_list_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, page.image_path, page.width, page.height,
                 page.page_number, page.source_path, page.source_type,
                 page.source_page_index, page.cache_image_path,
                 page.thumbnail_path, page.status.value, page.error_message,
                 page.ocr_invalidated_reason,
                 json.dumps(page.ppvl_parsing_res_list, ensure_ascii=False)),
            )
            page.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE page SET image_path=?, width=?, height=?, page_number=?, "
                "source_path=?, source_type=?, source_page_index=?, "
                "cache_image_path=?, thumbnail_path=?, status=?, error_message=?, "
                "ocr_invalidated_reason=?, "
                "ppvl_parsing_res_list_json=? "
                "WHERE id=? AND project_id=?",
                (page.image_path, page.width, page.height, page.page_number,
                 page.source_path, page.source_type, page.source_page_index,
                 page.cache_image_path, page.thumbnail_path, page.status.value,
                 page.error_message,
                 page.ocr_invalidated_reason,
                 json.dumps(page.ppvl_parsing_res_list, ensure_ascii=False),
                 page.id, project_id),
            )
            if cur.rowcount != 1:
                page.id = None
                self._save_page(cur, page, project_id)
                return

        old_block_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM block WHERE page_id=?", (page.id,)
            ).fetchall()
        }
        saved_block_ids: set[int] = set()
        for block in page.blocks:
            self._save_block(cur, block, page.id)
            if block.id is not None:
                saved_block_ids.add(block.id)

        for old_id in old_block_ids - saved_block_ids:
            cur.execute("DELETE FROM block WHERE id=?", (old_id,))

    def _save_block(self, cur: sqlite3.Cursor, block: Block, page_id: int) -> None:
        bb = block.bbox
        values = (
            page_id, block.block_type.value, bb.x, bb.y, bb.w, bb.h, block.order,
            block.source.value, int(block.is_locked), int(block.recognizable),
            block.note, block.source_label,
            json.dumps(block.raw_payload, ensure_ascii=False),
        )
        if block.id is None:
            cur.execute(
                "INSERT INTO block (page_id, block_type, x, y, w, h, block_order, "
                "source, is_locked, recognizable, note, source_label, raw_payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            block.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE block SET page_id=?, block_type=?, x=?, y=?, w=?, h=?, "
                "block_order=?, source=?, is_locked=?, recognizable=?, note=?, "
                "source_label=?, raw_payload_json=? WHERE id=? AND page_id=?",
                (*values, block.id, page_id),
            )
            if cur.rowcount != 1:
                block.id = None
                self._save_block(cur, block, page_id)
                return

        old_line_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM line WHERE block_id=?", (block.id,)
            ).fetchall()
        }
        saved_line_ids: set[int] = set()
        for line in block.lines:
            self._save_line(cur, line, block.id)
            if line.id is not None:
                saved_line_ids.add(line.id)

        for old_id in old_line_ids - saved_line_ids:
            cur.execute("DELETE FROM line WHERE id=?", (old_id,))

    def _save_line(self, cur: sqlite3.Cursor, line: Line, block_id: int) -> None:
        bb = line.bbox
        final_text = line.final_text or line.text
        line.final_text = final_text
        line.text = final_text
        values = (
            block_id, line.text, line.final_text, line.original_text, line.confidence,
            line.proof_status.value, bb.x, bb.y, bb.w, bb.h,
            line.ocr_text, line.llm_suggestion, line.llm_reason,
            line.llm_review_status.value,
            _review_flags_to_json(line.review_flags),
        )
        if line.id is None:
            cur.execute(
                "INSERT INTO line (block_id, text, final_text, original_text, confidence, proof_status, "
                "x, y, w, h, ocr_text, llm_suggestion, llm_reason, llm_review_status, "
                "review_flags_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            line.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE line SET block_id=?, text=?, final_text=?, original_text=?, "
                "confidence=?, proof_status=?, x=?, y=?, w=?, h=?, ocr_text=?, "
                "llm_suggestion=?, llm_reason=?, llm_review_status=?, review_flags_json=? "
                "WHERE id=? AND block_id=?",
                (*values, line.id, block_id),
            )
            if cur.rowcount != 1:
                line.id = None
                self._save_line(cur, line, block_id)
                return

        old_char_ids = {
            r["id"] for r in cur.execute(
                "SELECT id FROM char_ WHERE line_id=?", (line.id,)
            ).fetchall()
        }
        saved_char_ids: set[int] = set()
        for char in line.chars:
            self._save_char(cur, char, line.id)
            if char.id is not None:
                saved_char_ids.add(char.id)

        for old_id in old_char_ids - saved_char_ids:
            cur.execute("DELETE FROM char_ WHERE id=?", (old_id,))

    def _save_char(self, cur: sqlite3.Cursor, char: Char, line_id: int) -> None:
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
                "INSERT INTO char_ (line_id, char, confidence, x, y, w, h, "
                "bbox_source, bbox_granularity, token_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            char.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE char_ SET line_id=?, char=?, confidence=?, x=?, y=?, w=?, h=?, "
                "bbox_source=?, bbox_granularity=?, token_text=? WHERE id=? AND line_id=?",
                (*values, char.id, line_id),
            )
            if cur.rowcount != 1:
                char.id = None
                self._save_char(cur, char, line_id)

    # ------------------------------------------------------------------ update single line

    def update_line(self, line: Line) -> None:
        """只更新单行文字（校对时使用）。"""
        self._update_line_no_commit(line)
        self.conn.commit()

    def _update_line_no_commit(self, line: Line) -> None:
        bb = line.bbox
        final_text = line.final_text or line.text
        line.final_text = final_text
        line.text = final_text
        cur = self.conn.execute(
            "UPDATE line SET text=?, final_text=?, original_text=?, proof_status=?, "
            "ocr_text=?, llm_suggestion=?, llm_reason=?, llm_review_status=?, "
            "review_flags_json=? WHERE id=?",
            (line.text, line.final_text, line.original_text, line.proof_status.value,
             line.ocr_text, line.llm_suggestion, line.llm_reason,
             line.llm_review_status.value,
             _review_flags_to_json(line.review_flags), line.id),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"Line update failed or matched multiple rows: id={line.id!r}")

    # ------------------------------------------------------------------ batch update lines

    def update_lines(self, lines: list[Line]) -> None:
        """批量更新行（事务）。"""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            for line in lines:
                self._update_line_no_commit(line)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

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
            page = Page(
                image_path=pr["image_path"],
                width=pr["width"],
                height=pr["height"],
                page_number=pr["page_number"],
                id=pr["id"],
                source_path=pr["source_path"],
                source_type=pr["source_type"],
                source_page_index=pr["source_page_index"],
                cache_image_path=pr["cache_image_path"],
                thumbnail_path=pr["thumbnail_path"],
                status=PageStatus(pr["status"]),
                error_message=pr["error_message"],
                ocr_invalidated_reason=pr["ocr_invalidated_reason"],
                ppvl_parsing_res_list=_json_to_list(pr["ppvl_parsing_res_list_json"]),
            )
            page.blocks = self._load_blocks(page.id)
            project.pages.append(page)

        return project

    def _load_blocks(self, page_id: int) -> List[Block]:
        rows = self.conn.execute(
            "SELECT * FROM block WHERE page_id=? ORDER BY block_order", (page_id,)
        ).fetchall()
        blocks = []
        for r in rows:
            block = Block(
                block_type=BlockType(r["block_type"]),
                bbox=BBox(r["x"], r["y"], r["w"], r["h"]),
                order=r["block_order"],
                id=r["id"],
                source=BlockSource(r["source"]),
                is_locked=bool(r["is_locked"]),
                recognizable=bool(r["recognizable"]),
                note=r["note"],
                source_label=r["source_label"],
                raw_payload=_json_to_dict(r["raw_payload_json"]),
            )
            block.lines = self._load_lines(block.id)
            blocks.append(block)
        return blocks

    def _load_lines(self, block_id: int) -> List[Line]:
        rows = self.conn.execute(
            "SELECT * FROM line WHERE block_id=?", (block_id,)
        ).fetchall()
        lines = []
        for r in rows:
            line = Line(
                text=r["final_text"] or r["text"],
                final_text=r["final_text"] or r["text"],
                original_text=r["original_text"],
                confidence=r["confidence"],
                bbox=BBox(r["x"], r["y"], r["w"], r["h"]),
                proof_status=ProofStatus(r["proof_status"]),
                id=r["id"],
                ocr_text=r["ocr_text"],
                llm_suggestion=r["llm_suggestion"],
                llm_reason=r["llm_reason"],
                llm_review_status=LlmReviewStatus(r["llm_review_status"]),
                review_flags=_json_to_review_flags(r["review_flags_json"]),
            )
            line.chars = self._load_chars(line.id)
            lines.append(line)
        return lines

    def _load_chars(self, line_id: int) -> List[Char]:
        rows = self.conn.execute(
            "SELECT * FROM char_ WHERE line_id=?", (line_id,)
        ).fetchall()
        chars = []
        for r in rows:
            bbox = BBox(r["x"], r["y"], r["w"], r["h"]) if r["x"] is not None else None
            chars.append(Char(
                char=r["char"],
                confidence=r["confidence"],
                bbox=bbox,
                id=r["id"],
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
    ) -> None:
        self.conn.execute(
            "INSERT INTO operation_log (project_id, page_id, object_type, object_id, "
            "action, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, page_id, object_type, object_id,
             action, json.dumps(payload or {}, ensure_ascii=False), time.time()),
        )
        self.conn.commit()
