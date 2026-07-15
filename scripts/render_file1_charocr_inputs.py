#!/usr/bin/env python3
"""Render production PP-OCRv6 routes and exact pre-CharOCR inputs for file/1."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.adapters.paddle.ppocr_v6_prepass import parse_ppocr_v6_prepass_result
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.core.project_store import ProjectStore
from app.engines.hanwang.micro_recblock import (
    _engcut_masked_line_routes_from_lines,
    _linecut_masked_line_routes_from_lines,
    _materialize_engcut_masked_line_crop,
    _materialize_linecut_masked_page,
)
from app.models.charocr_routing import (
    ROUTE_SEGMENT_FORMULA,
    ROUTE_SEGMENT_TEXT_LATIN,
    ROUTE_SEGMENT_TEXT_OTHER,
)
from app.models.layout_block_view import current_layout_snapshot


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COLORS = {
    "pp_line": (0, 190, 255),
    ROUTE_SEGMENT_TEXT_OTHER: (45, 175, 55),
    ROUTE_SEGMENT_TEXT_LATIN: (0, 140, 255),
    ROUTE_SEGMENT_FORMULA: (220, 40, 220),
    "skip": (130, 130, 130),
}


def _load_raw_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    results = [
        result
        for row in rows
        if isinstance(row, dict)
        for result in row.get("result", {}).get("ocrResults", [])
        if isinstance(result, dict)
    ]
    if len(results) != 1:
        raise RuntimeError(f"expected one PP-OCRv6 page result in {path}, got {len(results)}")
    return results[0]


def _raw_path(stem: str, raw_dirs: list[Path]) -> Path:
    matches = [path for root in raw_dirs for path in [root / f"{stem}.raw-pages.json"] if path.exists()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cached PP-OCRv6 response for {stem}, got {matches}")
    return matches[0]


def _draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (max(0, x), max(14, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _copy_segments(image: np.ndarray, segments: list, kind: str) -> np.ndarray:
    canvas = np.full_like(image, 255)
    height, width = image.shape[:2]
    for segment in segments:
        if segment.kind != kind:
            continue
        x1, y1, x2, y2 = segment.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return canvas


def _formula_canvas(image: np.ndarray, segments: list) -> np.ndarray:
    canvas = np.full_like(image, 255)
    height, width = image.shape[:2]
    seen: set[tuple[int, int, int, int]] = set()
    for segment in segments:
        if segment.kind != ROUTE_SEGMENT_FORMULA:
            continue
        box = segment.content_bbox or segment.bbox
        if box in seen:
            continue
        seen.add(box)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return canvas


def _draw_disjoint_route_overlay(image: np.ndarray, lines: list) -> np.ndarray:
    """Draw final route ownership after native-input exclusion precedence."""
    height, width = image.shape[:2]
    owner = np.zeros((height, width), dtype=np.uint8)
    owner_ids = {
        ROUTE_SEGMENT_TEXT_OTHER: 1,
        ROUTE_SEGMENT_TEXT_LATIN: 2,
        ROUTE_SEGMENT_FORMULA: 3,
        "skip": 4,
    }
    # The first pass establishes the LineCut carrier. The second pass mirrors
    # production masking: Latin, formula, and skip regions replace it.
    for line in lines:
        for segment in line.segments:
            if segment.kind != ROUTE_SEGMENT_TEXT_OTHER:
                continue
            x1, y1, x2, y2 = segment.bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                owner[y1:y2, x1:x2] = owner_ids[ROUTE_SEGMENT_TEXT_OTHER]
    for line in lines:
        for segment in line.segments:
            if segment.kind == ROUTE_SEGMENT_TEXT_OTHER:
                continue
            owner_id = owner_ids.get(segment.kind, owner_ids["skip"])
            box = (
                segment.content_bbox
                if segment.kind == ROUTE_SEGMENT_FORMULA and segment.content_bbox
                else segment.bbox
            )
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                owner[y1:y2, x1:x2] = owner_id

    overlay = image.copy()
    for kind, owner_id in owner_ids.items():
        mask = np.where(owner == owner_id, 255, 0).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS.get(kind, COLORS["skip"]), 2)
    return overlay


def _panel(crop: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.copyMakeBorder(crop, 30, 1, 1, 1, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    _draw_label(panel, title, 8, 20, (25, 25, 25))
    return panel


def _line_sheet(
    image: np.ndarray,
    line,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = line.bbox
    pad_x, pad_y = 20, 12
    crop = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))
    cx1, cy1, cx2, cy2 = crop
    source = image[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(source, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), COLORS["pp_line"], 2)

    linecut = np.full_like(source, 255)
    latin = np.full_like(source, 255)
    formula = np.full_like(source, 255)
    for segment in line.segments:
        box = segment.content_bbox if segment.kind == ROUTE_SEGMENT_FORMULA and segment.content_bbox else segment.bbox
        sx1, sy1, sx2, sy2 = box
        ix1, iy1, ix2, iy2 = max(cx1, sx1), max(cy1, sy1), min(cx2, sx2), min(cy2, sy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        target = None
        if segment.kind == ROUTE_SEGMENT_TEXT_OTHER:
            target = linecut
        elif segment.kind == ROUTE_SEGMENT_TEXT_LATIN:
            target = latin
        elif segment.kind == ROUTE_SEGMENT_FORMULA:
            target = formula
        if target is not None:
            target[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = image[iy1:iy2, ix1:ix2]
    # Structural/Latin/symbol regions have the same explicit precedence as
    # the production LineCut page materializer.
    for segment in line.segments:
        if segment.kind == ROUTE_SEGMENT_TEXT_OTHER:
            continue
        sx1, sy1, sx2, sy2 = segment.bbox
        ix1, iy1, ix2, iy2 = max(cx1, sx1), max(cy1, sy1), min(cx2, sx2), min(cy2, sy2)
        if ix2 > ix1 and iy2 > iy1:
            linecut[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = 255
    panels = [
        _panel(source, f"00 raw PP-OCRv6 line L{line.index}"),
        _panel(linecut, "01 LineCut-owned other text"),
        _panel(latin, "02 EngCut-owned Latin/digits"),
        _panel(formula, "03 formula-owned pixels (not CharOCR)"),
    ]
    target_width = max(panel.shape[1] for panel in panels)
    normalized = [
        cv2.copyMakeBorder(panel, 0, 0, 0, target_width - panel.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for panel in panels
    ]
    return np.vstack(normalized)


def _copy_project_bundle(project_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in project_path.parent.glob(f"{project_path.name}*"):
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
    return target_dir / project_path.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "file/debug/all-page.ocrproj")
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "file/1")
    parser.add_argument(
        "--raw-dir",
        action="append",
        type=Path,
        default=[
            REPO_ROOT / "debug/ppocrv6_mask_stage_audit_244771/ppocrv6_raw",
            REPO_ROOT / "debug/ppocrv6_mask_stage_audit_13pages/ppocrv6_raw",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "debug/file1_charocr_input_audit_20260713",
    )
    parser.add_argument(
        "--page",
        action="append",
        default=[],
        help="Render only the named image stem; repeat for multiple pages.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_project_dir = tempfile.TemporaryDirectory(prefix="ocr-route-audit-")
    project_copy = _copy_project_bundle(args.project, Path(temp_project_dir.name))
    store = ProjectStore(project_copy)
    store.open()
    projects = store.list_projects()
    project = store.load_project(projects[0]["id"])
    store.close()
    temp_project_dir.cleanup()
    if project is None:
        raise RuntimeError(f"cannot load project: {project_copy}")

    images = {
        path.name: path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    reports = []
    totals: Counter[str] = Counter()
    for page in project.pages:
        name = Path(page.source_path).name
        if args.page and Path(name).stem not in set(args.page):
            continue
        image_path = images.get(name)
        if image_path is None:
            raise RuntimeError(f"project page has no file/1 image: {name}")
        raw_path = _raw_path(image_path.stem, args.raw_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        prepass = parse_ppocr_v6_prepass_result(
            _load_raw_result(raw_path),
            page_uid=page.uid,
            run_id=f"cached-audit:{image_path.stem}",
            width=width,
            height=height,
        )
        plan = compile_page_routing_plan(
            current_layout_snapshot(page),
            prepass,
            page_width=width,
            page_height=height,
            page_image_bgr=image,
        )
        page_dir = args.out_dir / image_path.stem
        page_dir.mkdir(parents=True, exist_ok=True)

        raw_overlay = image.copy()
        for line in prepass.lines:
            x1, y1, x2, y2 = line.bbox
            cv2.rectangle(raw_overlay, (x1, y1), (x2, y2), COLORS["pp_line"], 2)
            _draw_label(raw_overlay, f"L{line.index}", x1, y1 - 3, COLORS["pp_line"])
        cv2.imwrite(str(page_dir / "01_ppocrv6_raw_line_boxes.png"), raw_overlay)

        all_lines = [line for block in plan.blocks for line in block.plan.lines]
        segments = [segment for line in all_lines for segment in line.segments]
        route_overlay = _draw_disjoint_route_overlay(image, all_lines)
        cv2.imwrite(str(page_dir / "02_final_typed_routes.png"), route_overlay)

        issue_rows = [
            {"code": issue.code, "line_index": issue.line_index, "bbox": list(issue.bbox), "message": issue.message}
            for issue in plan.validation_issues
        ]
        page_report = {
            "page": image_path.name,
            "page_uid": page.uid,
            "raw_ppocr": str(raw_path),
            "dispatchable": plan.is_dispatchable,
            "validation_issues": issue_rows,
            "ppocr_lines": len(prepass.lines),
            "routed_lines": len(all_lines),
            "route_segments": dict(Counter(segment.kind for segment in segments)),
        }
        if not plan.is_dispatchable:
            totals["blocked_pages"] += 1
            (page_dir / "BLOCKED.json").write_text(json.dumps(page_report, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(page_report)
            print(f"[blocked] {image_path.name}: {len(issue_rows)} issue(s)", flush=True)
            continue

        linecut_routes = [
            route
            for block_index, block in enumerate(plan.blocks)
            for route in _linecut_masked_line_routes_from_lines(block_index, block.plan.lines)
        ]
        engcut_routes = [
            route
            for block_index, block in enumerate(plan.blocks)
            for route in _engcut_masked_line_routes_from_lines(block_index, block.plan.lines)
        ]
        linecut_canvas = _materialize_linecut_masked_page(image, linecut_routes)
        latin_canvas = _copy_segments(image, segments, ROUTE_SEGMENT_TEXT_LATIN)
        formulas = _formula_canvas(image, segments)
        cv2.imwrite(str(page_dir / "03_linecut_final_page_input.png"), linecut_canvas)
        cv2.imwrite(str(page_dir / "04_engcut_owned_pixels.png"), latin_canvas)
        cv2.imwrite(str(page_dir / "05_formula_owned_pixels_not_charocr.png"), formulas)

        engcut_dir = page_dir / "engcut_line_inputs"
        engcut_dir.mkdir(exist_ok=True)
        for route in engcut_routes:
            crop, offset_x, offset_y = _materialize_engcut_masked_line_crop(image, route)
            cv2.imwrite(
                str(engcut_dir / f"block_{route.block_idx:03d}_line_{route.line_idx:03d}_x{offset_x}_y{offset_y}.png"),
                crop,
            )

        sheets_dir = page_dir / "route_isolation_lines"
        sheets_dir.mkdir(exist_ok=True)
        sheet_count = 0
        for line in all_lines:
            if not any(segment.kind in {ROUTE_SEGMENT_TEXT_LATIN, ROUTE_SEGMENT_FORMULA} for segment in line.segments):
                continue
            sheet = _line_sheet(image, line)
            cv2.imwrite(str(sheets_dir / f"line_{line.index:03d}_route_isolation.png"), sheet)
            sheet_count += 1

        page_report.update({
            "linecut_page_input": str(page_dir / "03_linecut_final_page_input.png"),
            "engcut_line_inputs": len(engcut_routes),
            "route_isolation_sheets": sheet_count,
        })
        (page_dir / "manifest.json").write_text(json.dumps(page_report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(page_report)
        totals["dispatchable_pages"] += 1
        totals["engcut_line_inputs"] += len(engcut_routes)
        totals["route_isolation_sheets"] += sheet_count
        print(f"[ok] {image_path.name}: lines={len(all_lines)} engcut={len(engcut_routes)} sheets={sheet_count}", flush=True)

    summary = {
        "schema": "file1-charocr-input-audit.v1",
        "project": str(args.project),
        "image_dir": str(args.image_dir),
        "totals": dict(totals),
        "pages": reports,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out_dir.resolve())
    return 0 if not totals["blocked_pages"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
