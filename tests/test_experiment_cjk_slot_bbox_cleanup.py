from __future__ import annotations

import numpy as np
import sqlite3

from scripts.experiment_cjk_slot_bbox_cleanup import (
    _best_vertical_seam,
    _load_project_observations,
    _slot_x_bounds,
    _tight_foreground_bbox,
)


def _atom(index: int, text: str, bbox: tuple[int, int, int, int]) -> dict:
    return {
        "index": index,
        "text": text,
        "bbox": list(bbox),
        "source": "hanwang:micro_recblock",
        "granularity": "char",
    }


def test_vertical_seam_chooses_empty_projection_gap() -> None:
    foreground = np.zeros((30, 80), dtype=bool)
    foreground[5:25, 10:32] = True
    foreground[5:25, 45:68] = True

    seam, risky = _best_vertical_seam(foreground, (0, 30), 21.0, 56.0)

    assert 33 <= seam <= 44
    assert risky is False


def test_slot_bounds_use_projection_gap_between_cjk_atoms() -> None:
    foreground = np.zeros((30, 80), dtype=bool)
    foreground[5:25, 10:32] = True
    foreground[5:25, 45:68] = True
    atoms = [_atom(0, "民", (10, 5, 34, 25)), _atom(1, "族", (29, 5, 68, 25))]

    bounds = _slot_x_bounds(
        atoms,
        1,
        page_width=80,
        foreground=foreground,
        band=(0, 30),
    )

    assert bounds is not None
    left, right, risky = bounds
    assert 33 <= left <= 44
    assert right == 68
    assert risky is False


def test_tight_foreground_bbox_keeps_detached_parts_inside_slot() -> None:
    foreground = np.zeros((40, 60), dtype=bool)
    foreground[12:34, 22:42] = True
    foreground[5:9, 30:34] = True
    foreground[20:24, 8:11] = True

    bbox, area, component_count = _tight_foreground_bbox(
        foreground,
        (20, 3, 45, 36),
    )

    assert bbox == (22, 5, 42, 34)
    assert area == 456
    assert component_count == 2


def test_legacy_project_adapter_preserves_page_line_and_char_order() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE page (
            id INTEGER PRIMARY KEY, uid TEXT, page_number INTEGER,
            width INTEGER, height INTEGER, cache_image_path TEXT, source_path TEXT
        );
        CREATE TABLE block (id INTEGER PRIMARY KEY, page_id INTEGER);
        CREATE TABLE line (
            id INTEGER PRIMARY KEY, uid TEXT, block_id INTEGER, text TEXT,
            x INTEGER, y INTEGER, w INTEGER, h INTEGER
        );
        CREATE TABLE char_ (
            id INTEGER PRIMARY KEY, uid TEXT, line_id INTEGER, char TEXT,
            x INTEGER, y INTEGER, w INTEGER, h INTEGER,
            bbox_source TEXT, bbox_granularity TEXT
        );
        INSERT INTO page VALUES (1, 'page-a', 1, 100, 80, 'page.png', 'source.png');
        INSERT INTO block VALUES (2, 1);
        INSERT INTO line VALUES (3, 'line-a', 2, '民族', 10, 20, 50, 30);
        INSERT INTO char_ VALUES (
            5, 'char-2', 3, '族', 35, 20, 20, 30,
            'hanwang:micro_recblock', 'char'
        );
        INSERT INTO char_ VALUES (
            4, 'char-1', 3, '民', 10, 20, 20, 30,
            'hanwang:micro_recblock', 'char'
        );
        """
    )

    pages, lines, atoms, schema = _load_project_observations(connection)

    assert schema == "legacy_v1_diagnostic_adapter"
    assert pages[0]["uid"] == "page-a"
    assert lines[0]["page_uid"] == "page-a"
    assert lines[0]["bbox"] == [10, 20, 60, 50]
    assert [(atom["text"], atom["index"]) for atom in atoms] == [
        ("民", 0),
        ("族", 1),
    ]
