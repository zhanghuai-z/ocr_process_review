"""SQLite 项目文件读写。

项目文件扩展名：.ocrproj（本质是 SQLite 数据库）。
图片数据只存路径，不存 Blob，避免数据库过大。
"""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from app.models import (
    BBox, Block, BlockType, Char, Line, OcrProject, Page, ProofStatus
)

# --------------------------------------------------------------------- schema
DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS project (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS page (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    image_path  TEXT    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS block (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id     INTEGER NOT NULL REFERENCES page(id) ON DELETE CASCADE,
    block_type  TEXT    NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL,
    block_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS line (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id        INTEGER NOT NULL REFERENCES block(id) ON DELETE CASCADE,
    text            TEXT    NOT NULL DEFAULT '',
    original_text   TEXT    NOT NULL DEFAULT '',
    confidence      REAL    NOT NULL DEFAULT 0.0,
    proof_status    TEXT    NOT NULL DEFAULT 'unchecked',
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    w INTEGER NOT NULL, h INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS char (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id     INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    char        TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER
);
"""


class ProjectStore:
    """负责 OcrProject 的持久化。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ open/close

    def open(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(DDL)
        self._conn.commit()

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

    # ------------------------------------------------------------------ save

    def save_project(self, project: OcrProject) -> OcrProject:
        """保存或更新整个项目（全量写入）。"""
        now = time.time()
        project.updated_at = now

        cur = self.conn.cursor()
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

        for page in project.pages:
            self._save_page(cur, page, project.id)

        self.conn.commit()
        project.db_path = self.db_path
        return project

    def _save_page(self, cur: sqlite3.Cursor, page: Page, project_id: int) -> None:
        if page.id is None:
            cur.execute(
                "INSERT INTO page (project_id, image_path, width, height, page_number) "
                "VALUES (?, ?, ?, ?, ?)",
                (project_id, page.image_path, page.width, page.height, page.page_number),
            )
            page.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE page SET image_path=?, width=?, height=?, page_number=? WHERE id=?",
                (page.image_path, page.width, page.height, page.page_number, page.id),
            )

        for block in page.blocks:
            self._save_block(cur, block, page.id)

    def _save_block(self, cur: sqlite3.Cursor, block: Block, page_id: int) -> None:
        bb = block.bbox
        if block.id is None:
            cur.execute(
                "INSERT INTO block (page_id, block_type, x, y, w, h, block_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (page_id, block.block_type.value, bb.x, bb.y, bb.w, bb.h, block.order),
            )
            block.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE block SET block_type=?, x=?, y=?, w=?, h=?, block_order=? WHERE id=?",
                (block.block_type.value, bb.x, bb.y, bb.w, bb.h, block.order, block.id),
            )

        for line in block.lines:
            self._save_line(cur, line, block.id)

    def _save_line(self, cur: sqlite3.Cursor, line: Line, block_id: int) -> None:
        bb = line.bbox
        if line.id is None:
            cur.execute(
                "INSERT INTO line (block_id, text, original_text, confidence, proof_status, "
                "x, y, w, h) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (block_id, line.text, line.original_text, line.confidence,
                 line.proof_status.value, bb.x, bb.y, bb.w, bb.h),
            )
            line.id = cur.lastrowid
        else:
            cur.execute(
                "UPDATE line SET text=?, original_text=?, confidence=?, proof_status=?, "
                "x=?, y=?, w=?, h=? WHERE id=?",
                (line.text, line.original_text, line.confidence,
                 line.proof_status.value, bb.x, bb.y, bb.w, bb.h, line.id),
            )

        for char in line.chars:
            self._save_char(cur, char, line.id)

    def _save_char(self, cur: sqlite3.Cursor, char: Char, line_id: int) -> None:
        if char.id is not None:
            return  # char 不做更新（识别后不变）
        bb = char.bbox
        x, y, w, h = (bb.x, bb.y, bb.w, bb.h) if bb else (None, None, None, None)
        cur.execute(
            "INSERT INTO char (line_id, char, confidence, x, y, w, h) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (line_id, char.char, char.confidence, x, y, w, h),
        )
        char.id = cur.lastrowid

    # ------------------------------------------------------------------ update single line

    def update_line(self, line: Line) -> None:
        """只更新单行文字（校对时使用）。"""
        bb = line.bbox
        self.conn.execute(
            "UPDATE line SET text=?, original_text=?, proof_status=? WHERE id=?",
            (line.text, line.original_text, line.proof_status.value, line.id),
        )
        self.conn.commit()

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
                text=r["text"],
                original_text=r["original_text"],
                confidence=r["confidence"],
                bbox=BBox(r["x"], r["y"], r["w"], r["h"]),
                proof_status=ProofStatus(r["proof_status"]),
                id=r["id"],
            )
            line.chars = self._load_chars(line.id)
            lines.append(line)
        return lines

    def _load_chars(self, line_id: int) -> List[Char]:
        rows = self.conn.execute(
            "SELECT * FROM char WHERE line_id=?", (line_id,)
        ).fetchall()
        chars = []
        for r in rows:
            bbox = BBox(r["x"], r["y"], r["w"], r["h"]) if r["x"] is not None else None
            chars.append(Char(
                char=r["char"],
                confidence=r["confidence"],
                bbox=bbox,
                id=r["id"],
            ))
        return chars

    # ------------------------------------------------------------------ list projects

    def list_projects(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM project ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
