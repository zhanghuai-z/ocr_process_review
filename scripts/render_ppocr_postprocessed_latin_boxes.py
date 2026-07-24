#!/usr/bin/env python3
"""Render only the current postprocessed PP-OCR Latin route boxes.

The input is the diagnostic JSON emitted by ``audit_italic_token_fallback``.
No raw PP proposal boxes or EngCut character boxes are drawn, so the output
shows exactly the token-local ``RoutingSegment.bbox`` geometry used downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


GREEN = (20, 170, 20)
INK = (15, 15, 15)


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _project_page(payload: dict[str, Any], run_index: int) -> dict[str, Any]:
    matches = [
        page
        for page in payload.get("project_pages") or []
        if int(page.get("run_index") or 0) == run_index
        and page.get("status") == "ok"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one successful project page for run {run_index}, got {len(matches)}"
        )
    return matches[0]


def _tokens(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        token
        for token in page.get("tokens") or []
        if token.get("route_bbox") and not token.get("audit_error")
    ]


def _draw_full_page(
    image: np.ndarray,
    tokens: list[dict[str, Any]],
    path: Path,
) -> None:
    canvas = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 1800))
    for token in tokens:
        x1, y1, x2, y2 = _xyxy(token["route_bbox"])
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2 - 1, y2 - 1),
            GREEN,
            thickness,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", canvas)[1].tofile(path)


def _crop_row(image: np.ndarray, token: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = _xyxy(token["route_bbox"])
    height, width = image.shape[:2]
    margin_x = max(10, round((x2 - x1) * 0.08))
    margin_y = max(8, round((y2 - y1) * 0.35))
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(width, x2 + margin_x)
    cy2 = min(height, y2 + margin_y)
    crop = image[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(
        crop,
        (x1 - cx1, y1 - cy1),
        (x2 - cx1 - 1, y2 - cy1 - 1),
        GREEN,
        2,
        cv2.LINE_AA,
    )
    scale = min(2.5, 700 / max(1, crop.shape[1]))
    if scale > 1.0:
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    metrics = token.get("metrics") or {}
    header = np.full((48, max(760, crop.shape[1]), 3), 255, np.uint8)
    label = (
        f"{token.get('text')} | postprocessed route={token.get('route_bbox')} | "
        f"native={metrics.get('native_text', '')}"
    )
    cv2.putText(
        header,
        label[:130],
        (8, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        INK,
        1,
        cv2.LINE_AA,
    )
    body = np.full((crop.shape[0], header.shape[1], 3), 255, np.uint8)
    body[:, :crop.shape[1]] = crop
    return np.vstack((header, body))


def _write_contact_sheets(
    image: np.ndarray,
    tokens: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    rows = [_crop_row(image, token) for token in tokens]
    paths: list[Path] = []
    for start in range(0, len(rows), 18):
        group = rows[start:start + 18]
        width = max(row.shape[1] for row in group)
        normalized = []
        for row in group:
            canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
            canvas[:, :row.shape[1]] = row
            normalized.append(canvas)
        sheet = np.vstack(normalized)
        path = output_dir / f"postprocessed_latin_boxes_{start // 18 + 1:02d}.png"
        cv2.imencode(".png", sheet)[1].tofile(path)
        paths.append(path)
    return paths


def _write_route_crop_sheet(
    image: np.ndarray,
    tokens: list[dict[str, Any]],
    path: Path,
) -> None:
    tiles: list[np.ndarray] = []
    for token in tokens:
        x1, y1, x2, y2 = _xyxy(token["route_bbox"])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = min(2.5, 320 / max(1, crop.shape[1]))
        if scale > 1.0:
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        tile = np.full((crop.shape[0] + 34, max(340, crop.shape[1]), 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(
            tile,
            str(token.get("text") or "")[:45],
            (6, crop.shape[0] + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            INK,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    columns = 4
    cell_width = max(tile.shape[1] for tile in tiles)
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        group = tiles[start:start + columns]
        cell_height = max(tile.shape[0] for tile in group)
        row = np.full((cell_height, cell_width * columns, 3), 255, np.uint8)
        for column, tile in enumerate(group):
            row[:tile.shape[0], column * cell_width:column * cell_width + tile.shape[1]] = tile
        rows.append(row)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723/report.json",
    )
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/ppocr_postprocessed_latin_boxes_test3_20260724",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    page = _project_page(payload, args.run_index)
    image = _read_image(Path(page["source_image"]))
    tokens = _tokens(page)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_page = args.output_dir / "postprocessed_latin_boxes_full_page.png"
    route_crops = args.output_dir / "postprocessed_latin_route_crops.png"
    _draw_full_page(image, tokens, full_page)
    sheets = _write_contact_sheets(image, tokens, args.output_dir)
    _write_route_crop_sheet(image, tokens, route_crops)
    report = {
        "schema": "ppocr_postprocessed_latin_boxes_echo.v1",
        "source_image": page["source_image"],
        "token_count": len(tokens),
        "geometry": "RoutingSegment.bbox (postprocessed owned-foreground union)",
        "excludes": ["raw PP proposal bbox", "EngCut native character bbox"],
        "full_page": _windows_path(full_page),
        "route_crops": _windows_path(route_crops),
        "contact_sheets": [_windows_path(path) for path in sheets],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
