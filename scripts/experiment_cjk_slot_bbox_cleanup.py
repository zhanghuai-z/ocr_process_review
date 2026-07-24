#!/usr/bin/env python3
"""Evaluate CJK bbox cleanup without changing OCR text or project state.

The experiment reads persisted OCR atom observations from one v2 project.  For
each single CJK LineCut atom it derives a horizontal ownership slot from the
centres of adjacent visible atoms, derives a robust CJK vertical band from the
physical line, and proposes the tight foreground bbox inside that slot.  The
proposal is diagnostic geometry only; it is never written back to the project.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


LINECUT_SOURCE = "hanwang:micro_recblock"
MIN_COMPONENT_AREA = 3


def _is_cjk(text: str) -> bool:
    if len(text) != 1:
        return False
    codepoint = ord(text)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def _xyxy(value: Iterable[Any]) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _center_x(bbox: tuple[int, int, int, int]) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _line_pitch(atoms: list[dict[str, Any]]) -> float:
    cjk = [atom for atom in atoms if _eligible_cjk_atom(atom)]
    centres = [_center_x(_xyxy(atom["bbox"])) for atom in cjk]
    gaps = [right - left for left, right in zip(centres, centres[1:]) if right > left]
    if gaps:
        return _median(gaps)
    widths = [float(_xyxy(atom["bbox"])[2] - _xyxy(atom["bbox"])[0]) for atom in cjk]
    return _median(widths) if widths else 0.0


def _eligible_cjk_atom(atom: dict[str, Any]) -> bool:
    return bool(
        atom.get("source") == LINECUT_SOURCE
        and atom.get("granularity") == "char"
        and atom.get("bbox") is not None
        and _is_cjk(str(atom.get("text") or ""))
    )


def _visible_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        atom
        for atom in atoms
        if atom.get("bbox") is not None
        and str(atom.get("text") or "").strip()
        and atom.get("granularity") != "space"
    ]


def _slot_x_bounds(
    atoms: list[dict[str, Any]],
    target_index: int,
    *,
    page_width: int,
    foreground: np.ndarray | None = None,
    band: tuple[int, int] | None = None,
) -> tuple[int, int, bool] | None:
    visible = _visible_atoms(atoms)
    positions = [index for index, atom in enumerate(visible) if int(atom["index"]) == target_index]
    if len(positions) != 1:
        return None
    position = positions[0]
    target = visible[position]
    bbox = _xyxy(target["bbox"])
    centre = _center_x(bbox)
    pitch = _line_pitch(atoms)
    if pitch <= 0:
        return None
    seam_risk = False

    if position > 0:
        left_centre = _center_x(_xyxy(visible[position - 1]["bbox"]))
        left, risky = _best_vertical_seam(
            foreground, band, left_centre, centre
        )
        seam_risk = seam_risk or risky
    else:
        left = bbox[0]
    if position + 1 < len(visible):
        right_centre = _center_x(_xyxy(visible[position + 1]["bbox"]))
        right, risky = _best_vertical_seam(
            foreground, band, centre, right_centre
        )
        seam_risk = seam_risk or risky
    else:
        right = bbox[2]
    left = max(0, left)
    right = min(page_width, right)
    if right <= left:
        return None
    return left, right, seam_risk


def _best_vertical_seam(
    foreground: np.ndarray | None,
    band: tuple[int, int] | None,
    left_centre: float,
    right_centre: float,
) -> tuple[int, bool]:
    midpoint = round((left_centre + right_centre) / 2.0)
    if foreground is None or band is None or right_centre <= left_centre:
        return midpoint, False
    distance = right_centre - left_centre
    start = max(1, round(left_centre + distance * 0.30))
    stop = min(foreground.shape[1] - 1, round(left_centre + distance * 0.70))
    if stop < start:
        return midpoint, False
    top, bottom = band
    candidates = []
    for seam in range(start, stop + 1):
        score = int(foreground[top:bottom, seam - 1:seam + 1].sum())
        candidates.append((score, abs(seam - midpoint), seam))
    score, _distance, seam = min(candidates)
    return seam, score > 0


def _line_cjk_band(
    atoms: list[dict[str, Any]],
    line_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    boxes = [_xyxy(atom["bbox"]) for atom in atoms if _eligible_cjk_atom(atom)]
    if not boxes:
        return None
    tops = [float(box[1]) for box in boxes]
    bottoms = [float(box[3]) for box in boxes]
    heights = [float(box[3] - box[1]) for box in boxes]
    median_height = max(1.0, _median(heights))
    padding = max(1, round(median_height * 0.08))
    top = max(line_bbox[1], round(_median(tops)) - padding)
    bottom = min(line_bbox[3], round(_median(bottoms)) + padding)
    if bottom <= top:
        return None
    return top, bottom


def _tight_foreground_bbox(
    foreground: np.ndarray,
    slot_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int] | None, int, int]:
    """Return tight bbox, kept area, and component count inside one slot."""
    x1, y1, x2, y2 = slot_bbox
    crop = foreground[y1:y2, x1:x2].astype(np.uint8)
    if crop.size == 0:
        return None, 0, 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(crop, 8)
    kept = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= MIN_COMPONENT_AREA
    ]
    if not kept:
        return None, 0, 0
    mask = np.isin(labels, kept)
    ys, xs = np.where(mask)
    return (
        (x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1),
        int(mask.sum()),
        len(kept),
    )


def _intersection_area(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _ink_area(foreground: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return int(foreground[y1:y2, x1:x2].sum())


def _slot_edge_ink(
    foreground: np.ndarray,
    slot_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    x1, y1, x2, y2 = slot_bbox
    left = int(foreground[y1:y2, x1].sum()) if x1 < foreground.shape[1] else 0
    right_x = x2 - 1
    right = int(foreground[y1:y2, right_x].sum()) if right_x >= 0 else 0
    return left, right


def _load_table(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [json.loads(row[0]) for row in connection.execute(f"SELECT payload FROM {table}")]


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _load_project_observations(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Normalize strict-v2 or legacy rows at this diagnostic input boundary."""
    if "payload" in _table_columns(connection, "page"):
        return (
            _load_table(connection, "page"),
            _load_table(connection, "ocr_line"),
            _load_table(connection, "ocr_atom"),
            "strict_v2",
        )

    pages = [
        {
            "uid": uid,
            "page_number": page_number,
            "width": width,
            "height": height,
            "cache_image_path": cache_image_path,
            "source_path": source_path,
        }
        for uid, page_number, width, height, cache_image_path, source_path
        in connection.execute(
            "SELECT uid, page_number, width, height, cache_image_path, source_path "
            "FROM page ORDER BY page_number, id"
        )
    ]
    lines = [
        {
            "uid": uid,
            "page_uid": page_uid,
            "text": text,
            "bbox": [x, y, x + width, y + height],
        }
        for uid, page_uid, text, x, y, width, height
        in connection.execute(
            "SELECT l.uid, p.uid, l.text, l.x, l.y, l.w, l.h "
            "FROM line l JOIN block b ON b.id=l.block_id "
            "JOIN page p ON p.id=b.page_id ORDER BY p.page_number, l.id"
        )
    ]
    atoms: list[dict[str, Any]] = []
    current_line_id: int | None = None
    line_index = 0
    for row in connection.execute(
        "SELECT c.id, c.uid, c.line_id, l.uid, c.char, c.x, c.y, c.w, c.h, "
        "c.bbox_source, c.bbox_granularity "
        "FROM char_ c JOIN line l ON l.id=c.line_id ORDER BY c.line_id, c.id"
    ):
        _id, uid, line_id, line_uid, text, x, y, width, height, source, granularity = row
        if line_id != current_line_id:
            current_line_id = int(line_id)
            line_index = 0
        bbox = None if x is None or y is None or width is None or height is None else [
            int(x), int(y), int(x + width), int(y + height)
        ]
        atoms.append({
            "uid": uid,
            "line_uid": line_uid,
            "index": line_index,
            "text": text,
            "bbox": bbox,
            "source": source,
            "granularity": granularity,
        })
        line_index += 1
    return pages, lines, atoms, "legacy_v1_diagnostic_adapter"


def _resolve_page_image(project_path: Path, page: dict[str, Any]) -> Path:
    candidates = [
        project_path.parent / str(page.get("cache_image_path") or ""),
        Path(str(page.get("source_path") or "").replace("D:\\", "/mnt/d/").replace("\\", "/")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve page image for {project_path}")


def _records(
    project_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Path],
    list[dict[str, Any]],
    str,
]:
    connection = sqlite3.connect(f"file:{project_path}?mode=ro", uri=True)
    try:
        pages, lines, atoms, persistence_schema = _load_project_observations(connection)
    finally:
        connection.close()
    atoms_by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        atoms_by_line[str(atom["line_uid"])].append(atom)
    lines_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        lines_by_page[str(line["page_uid"])].append(line)
    records: list[dict[str, Any]] = []
    image_paths: dict[str, Path] = {}
    for page in sorted(pages, key=lambda item: int(item.get("page_number") or 0)):
        page_uid = str(page["uid"])
        image_path = _resolve_page_image(project_path, page)
        image_paths[page_uid] = image_path
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = gray < 128
        for line in lines_by_page.get(page_uid, ()):
            line_uid = str(line["uid"])
            line_atoms = atoms_by_line.get(line_uid, ())
            ordered = sorted(line_atoms, key=lambda atom: int(atom["index"]))
            line_bbox = _xyxy(line["bbox"])
            band = _line_cjk_band(ordered, line_bbox)
            if band is None:
                continue
            for atom in ordered:
                if not _eligible_cjk_atom(atom):
                    continue
                native_bbox = _xyxy(atom["bbox"])
                bounds = _slot_x_bounds(
                    ordered,
                    int(atom["index"]),
                    page_width=image.shape[1],
                    foreground=foreground,
                    band=band,
                )
                if bounds is None:
                    continue
                slot_bbox = (bounds[0], band[0], bounds[1], band[1])
                cleanup_bbox = (
                    max(native_bbox[0], slot_bbox[0]),
                    native_bbox[1],
                    min(native_bbox[2], slot_bbox[2]),
                    native_bbox[3],
                )
                proposal, proposal_ink, component_count = _tight_foreground_bbox(
                    foreground, cleanup_bbox
                )
                if proposal is None:
                    status = "no_slot_foreground"
                    proposal = native_bbox
                else:
                    status = "proposed"
                native_ink = _ink_area(foreground, native_bbox)
                shared_ink = _ink_area(
                    foreground,
                    (
                        max(native_bbox[0], proposal[0]),
                        max(native_bbox[1], proposal[1]),
                        min(native_bbox[2], proposal[2]),
                        min(native_bbox[3], proposal[3]),
                    ),
                ) if _intersection_area(native_bbox, proposal) else 0
                removed_ink = max(0, native_ink - shared_ink)
                recovered_ink = max(0, proposal_ink - shared_ink)
                left_edge, right_edge = _slot_edge_ink(foreground, slot_bbox)
                records.append({
                    "page_uid": page_uid,
                    "page_number": int(page.get("page_number") or 0),
                    "page_source": str(page.get("source_path") or image_path.name),
                    "line_uid": line_uid,
                    "line_text": line.get("text") or "",
                    "atom_uid": atom["uid"],
                    "atom_index": int(atom["index"]),
                    "text": atom["text"],
                    "source": atom["source"],
                    "native_bbox": list(native_bbox),
                    "slot_bbox": list(slot_bbox),
                    "proposal_bbox": list(proposal),
                    "status": status,
                    "native_ink": native_ink,
                    "proposal_ink": proposal_ink,
                    "removed_ink": removed_ink,
                    "removed_ink_ratio": round(removed_ink / max(1, native_ink), 6),
                    "recovered_ink": recovered_ink,
                    "recovered_ink_ratio": round(recovered_ink / max(1, proposal_ink), 6),
                    "slot_component_count": component_count,
                    "slot_left_edge_ink": left_edge,
                    "slot_right_edge_ink": right_edge,
                    "slot_edge_risk": bool(left_edge or right_edge),
                    "seam_search_risk": bool(bounds[2]),
                })
        del image
    return records, image_paths, pages, persistence_schema


def _fit_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    size: int = 112,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if crop.size == 0:
        return canvas
    scale = min((size - 12) / crop.shape[1], (size - 12) / crop.shape[0])
    resized = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _render_record(image: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    native = _xyxy(record["native_bbox"])
    slot = _xyxy(record["slot_bbox"])
    proposal = _xyxy(record["proposal_bbox"])
    native_panel = _fit_crop(image, native)
    proposal_panel = _fit_crop(image, proposal)
    margin = 8
    context_bbox = (
        max(0, min(native[0], slot[0], proposal[0]) - margin),
        max(0, min(native[1], slot[1], proposal[1]) - margin),
        min(image.shape[1], max(native[2], slot[2], proposal[2]) + margin),
        min(image.shape[0], max(native[3], slot[3], proposal[3]) + margin),
    )
    x1, y1, x2, y2 = context_bbox
    context = image[y1:y2, x1:x2].copy()
    for bbox, color in ((native, (0, 0, 255)), (slot, (255, 0, 0)), (proposal, (0, 170, 0))):
        cv2.rectangle(
            context,
            (bbox[0] - x1, bbox[1] - y1),
            (bbox[2] - x1 - 1, bbox[3] - y1 - 1),
            color,
            1,
        )
    context_panel = _fit_crop(context, (0, 0, context.shape[1], context.shape[0]))
    body = np.hstack((native_panel, proposal_panel, context_panel))
    header = np.full((42, body.shape[1], 3), 255, dtype=np.uint8)
    values = (
        f"idx={record['atom_index']} remove={record['removed_ink_ratio']:.3f} "
        f"recover={record['recovered_ink_ratio']:.3f} edge={record['slot_edge_risk']}"
    )
    cv2.putText(header, values, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
    cv2.putText(
        header,
        "native | slot-clean proposal | context: red native / blue slot / green proposal",
        (5, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (60, 60, 60),
        1,
    )
    return np.vstack((header, body))


def _contact_sheet(
    image: np.ndarray,
    records: list[dict[str, Any]],
    path: Path,
    *,
    limit: int = 120,
) -> None:
    rows = [_render_record(image, record) for record in records[:limit]]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(path)


def _contact_sheets(
    image: np.ndarray,
    records: list[dict[str, Any]],
    directory: Path,
    stem: str,
    *,
    page_size: int = 80,
) -> list[Path]:
    paths: list[Path] = []
    for offset in range(0, len(records), page_size):
        path = directory / f"{stem}_{offset // page_size + 1:03d}.png"
        _contact_sheet(image, records[offset:offset + page_size], path, limit=page_size)
        paths.append(path)
    return paths


def _write_page_overlay(
    image: np.ndarray,
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    overlay = image.copy()
    thickness = max(1, round(image.shape[1] / 1400))
    for record in records:
        native = _xyxy(record["native_bbox"])
        proposal = _xyxy(record["proposal_bbox"])
        slot = _xyxy(record["slot_bbox"])
        cv2.rectangle(
            overlay,
            (native[0], native[1]),
            (native[2] - 1, native[3] - 1),
            (0, 0, 255),
            thickness,
        )
        cv2.rectangle(
            overlay,
            (proposal[0], proposal[1]),
            (proposal[2] - 1, proposal[3] - 1),
            (0, 170, 0),
            thickness,
        )
        cv2.line(
            overlay,
            (slot[0], slot[1]),
            (slot[0], slot[3] - 1),
            (255, 0, 0),
            thickness,
        )
        cv2.line(
            overlay,
            (slot[2] - 1, slot[1]),
            (slot[2] - 1, slot[3] - 1),
            (255, 0, 0),
            thickness,
        )
    if overlay.shape[1] > 1800:
        scale = 1800 / overlay.shape[1]
        overlay = cv2.resize(
            overlay,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", overlay)[1].tofile(path)


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _crossplatform_name(path: str) -> str:
    return Path(path.replace("\\", "/")).name


def _write_html_index(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    def relative(path: str) -> str:
        return Path(path).resolve().relative_to(output_dir.resolve()).as_posix()

    rows = []
    for summary in summaries:
        links = []
        for label, key in (
            ("changed", "changed_sheets"),
            ("empty seam", "empty_seam_sheets"),
            ("risk", "risk_sheets"),
        ):
            paths = summary[key]
            if paths:
                links.append(
                    f"<span>{escape(label)}: "
                    + " ".join(
                        f'<a href="{escape(relative(path))}">{index + 1}</a>'
                        for index, path in enumerate(paths)
                    )
                    + "</span>"
                )
        overlay = escape(relative(summary["overlay"]))
        rows.append(
            "<tr>"
            f"<td>{summary['page_number']}</td>"
            f"<td>{escape(_crossplatform_name(summary['source_path']))}</td>"
            f"<td>{summary['eligible_atoms']}</td>"
            f"<td>{summary['changed']}</td>"
            f"<td>{summary['empty_seam_excluded_ink']}</td>"
            f"<td>{summary['boundary_risk']}</td>"
            f'<td><a href="{overlay}"><img src="{overlay}" alt="page overlay"></a></td>'
            f"<td>{'<br>'.join(links) or 'none'}</td>"
            "</tr>"
        )
    html_path = output_dir / "index.html"
    html_path.write_text(
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>CJK bbox cleanup review</title>
<style>
body { font-family: sans-serif; margin: 20px; color: #202020; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #bbb; padding: 6px; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #f5f5f5; }
img { width: 240px; height: 180px; object-fit: contain; background: white; }
a { color: #075ea8; }
span { display: block; white-space: nowrap; }
</style>
</head>
<body>
<h1>CJK bbox cleanup review</h1>
<p>Red: native bbox. Green: proposal. Blue: slot boundary. Diagnostic only.</p>
<table>
<thead><tr><th>Page</th><th>Source</th><th>CJK</th><th>Changed</th><th>Empty seam</th><th>Risk</th><th>Overlay</th><th>Crop sheets</th></tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + """
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def _write_report(
    output_dir: Path,
    project_path: Path,
    image_paths: dict[str, Path],
    pages: list[dict[str, Any]],
    records: list[dict[str, Any]],
    persistence_schema: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    records_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_page[str(record["page_uid"])].append(record)
    page_summaries: list[dict[str, Any]] = []
    for page in sorted(pages, key=lambda item: int(item.get("page_number") or 0)):
        page_uid = str(page["uid"])
        page_records = records_by_page.get(page_uid, [])
        ranked = sorted(
            page_records,
            key=lambda item: float(item["removed_ink_ratio"]),
            reverse=True,
        )
        changed = [
            record
            for record in ranked
            if record["proposal_bbox"] != record["native_bbox"]
        ]
        empty_seam = [
            record
            for record in ranked
            if record["removed_ink"] > 0
            and not record["slot_edge_risk"]
            and not record["seam_search_risk"]
        ]
        risky = [
            record
            for record in ranked
            if record["slot_edge_risk"] or record["seam_search_risk"]
        ]
        source_stem = Path(
            str(page.get("source_path") or page_uid).replace("\\", "/")
        ).stem
        page_dir = output_dir / "pages" / (
            f"page_{int(page.get('page_number') or 0):04d}_{source_stem}"
        )
        page_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_paths[page_uid]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        overlay_path = page_dir / "changed_overlay.png"
        _write_page_overlay(image, changed, overlay_path)
        changed_sheets = _contact_sheets(image, changed, page_dir, "changed")
        empty_sheets = _contact_sheets(
            image, empty_seam, page_dir, "empty_seam_excluded_ink"
        )
        risk_sheets = _contact_sheets(image, risky, page_dir, "boundary_risk")
        del image
        summary = {
            "page_uid": page_uid,
            "page_number": int(page.get("page_number") or 0),
            "source_path": str(page.get("source_path") or ""),
            "image_path": str(image_path.resolve()),
            "eligible_atoms": len(page_records),
            "changed": len(changed),
            "excluded_native_ink": sum(record["removed_ink"] > 0 for record in page_records),
            "empty_seam_excluded_ink": len(empty_seam),
            "boundary_risk": len(risky),
            "output_dir": str(page_dir.resolve()),
            "overlay": str(overlay_path.resolve()),
            "changed_sheets": [str(path.resolve()) for path in changed_sheets],
            "empty_seam_sheets": [str(path.resolve()) for path in empty_sheets],
            "risk_sheets": [str(path.resolve()) for path in risk_sheets],
        }
        (page_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        page_summaries.append(summary)
    html_path = _write_html_index(output_dir, page_summaries)
    payload = {
        "schema": "cjk_slot_bbox_cleanup_experiment.v2",
        "diagnostic_only": True,
        "project": str(project_path.resolve()),
        "persistence_input": persistence_schema,
        "page_count": len(pages),
        "record_count": len(records),
        "status": dict(Counter(record["status"] for record in records)),
        "pages": page_summaries,
        "records": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    proposed = [record for record in records if record["status"] == "proposed"]
    changed = [
        record for record in proposed
        if record["proposal_bbox"] != record["native_bbox"]
    ]
    removal = [record for record in proposed if record["removed_ink"] > 0]
    recovery = [record for record in proposed if record["recovered_ink"] > 0]
    risky = [
        record
        for record in proposed
        if record["slot_edge_risk"] or record["seam_search_risk"]
    ]
    empty_seam = [
        record
        for record in proposed
        if record["removed_ink"] > 0
        and not record["slot_edge_risk"]
        and not record["seam_search_risk"]
    ]
    lines = [
        "# CJK Slot Bbox Cleanup Experiment",
        "",
        "Diagnostic only. OCR text, atom order, project persistence, Proof state, and export state were not modified.",
        "",
        "## Scope",
        "",
        f"- project: `{_windows_path(project_path)}`",
        f"- persistence input: `{persistence_schema}`",
        f"- pages: {len(pages)}",
        f"- eligible persisted LineCut CJK atoms: {len(records)}",
        f"- proposals: {len(proposed)}",
        f"- bbox changed: {len(changed)}",
        f"- native ink excluded by slot proposal: {len(removal)}",
        f"- slot foreground recovered outside native bbox: {len(recovery)}",
        f"- slot-edge ink risk: {sum(record['slot_edge_risk'] for record in proposed)}",
        f"- seam-search ink risk: {sum(record['seam_search_risk'] for record in proposed)}",
        f"- excluded ink beyond an empty seam: {len(empty_seam)}",
        f"- status: `{dict(Counter(record['status'] for record in records))}`",
        "",
        "## Page Review Index",
        "",
        "| page | source | CJK atoms | changed | empty-seam ink | boundary risk | review directory |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in page_summaries:
        lines.append(
            f"| {summary['page_number']} | `{_crossplatform_name(summary['source_path'])}` | "
            f"{summary['eligible_atoms']} | {summary['changed']} | "
            f"{summary['empty_seam_excluded_ink']} | {summary['boundary_risk']} | "
            f"`{_windows_path(Path(summary['output_dir']))}` |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- JSON: `{_windows_path(json_path)}`",
        f"- per-page review root: `{_windows_path(output_dir / 'pages')}`",
        f"- HTML review index: `{_windows_path(html_path)}`",
        "",
        "Each page directory contains a full-page changed overlay, paginated changed crops, empty-seam candidates, boundary-risk crops, and summary.json. Each crop row is native, proposal, then context; context colors are red native bbox, blue ownership slot, and green proposal.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "file/0724test/test1.ocrproj",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/cjk_slot_bbox_cleanup_0724test_20260724",
    )
    args = parser.parse_args()
    records, image_paths, pages, persistence_schema = _records(args.project)
    _write_report(
        args.output_dir,
        args.project,
        image_paths,
        pages,
        records,
        persistence_schema,
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
