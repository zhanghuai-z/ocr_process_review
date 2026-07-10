#!/usr/bin/env python3
"""Render the production mixed-text partition against cached PP-OCRv6 facts.

This diagnostic imports the production partition pure function instead of
copying its geometry rules.  It does not persist project data or call native
OCR.  Pure Latin and pure CJK rows are counted but deliberately omitted: they
take the whole-line EngCut and LineCut routes respectively.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.adapters.paddle.ppocr_v6_prepass import (
    PpOcrV6LineHint,
    parse_ppocr_v6_prepass_result,
)
from app.core.charocr_text_partition import partition_charocr_text_region
from app.core.ocr_ir import is_cjk_char


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LATIN_COLOR = (235, 116, 0)
OTHER_COLOR = (36, 156, 83)
TOKEN_COLOR = (54, 105, 245)
ISSUE_COLOR = (214, 45, 45)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load_pruned(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    results = []
    for row in rows:
        values = row.get("result", {}).get("ocrResults", []) if isinstance(row, dict) else []
        results.extend(item for item in values if isinstance(item, dict))
    if len(results) != 1:
        raise ValueError(f"expected one PP-OCRv6 page result in {path}, got {len(results)}")
    pruned = results[0].get("prunedResult", results[0])
    if not isinstance(pruned, dict):
        raise ValueError(f"invalid prunedResult in {path}")
    return pruned


def _has_ascii_alnum(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in text)


def _has_cjk(text: str) -> bool:
    return any(is_cjk_char(char) for char in text)


def _line_kind(line: PpOcrV6LineHint) -> str:
    has_latin = _has_ascii_alnum(line.text)
    has_cjk = _has_cjk(line.text)
    if has_latin and has_cjk:
        return "mixed"
    if has_latin:
        return "latin"
    return "other"


def _clip(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    ox, oy = origin
    draw.rectangle((box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy), outline=color, width=width)


def _masked_crop(
    page: np.ndarray,
    crop_box: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    source = page[y1:y2, x1:x2]
    canvas = np.full_like(source, 255)
    for bx1, by1, bx2, by2 in boxes:
        bx1, by1, bx2, by2 = _clip((bx1, by1, bx2, by2), page.shape[1], page.shape[0])
        ix1, iy1, ix2, iy2 = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        if ix2 > ix1 and iy2 > iy1:
            canvas[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = page[iy1:iy2, ix1:ix2]
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _panel(
    page_rgb: Image.Image,
    crop_box: tuple[int, int, int, int],
    title: str,
    boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]],
) -> Image.Image:
    crop = page_rgb.crop(crop_box)
    panel = Image.new("RGB", (crop.width, crop.height + 30), "white")
    panel.paste(crop, (0, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(0, 0, 0), font=_font(16))
    for box, color in boxes:
        _draw_box(draw, box, (crop_box[0], crop_box[1] - 30), color)
    return panel


def _masked_panel(
    page_bgr: np.ndarray,
    crop_box: tuple[int, int, int, int],
    title: str,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
) -> Image.Image:
    crop = _masked_crop(page_bgr, crop_box, boxes)
    panel = Image.new("RGB", (crop.width, crop.height + 30), "white")
    panel.paste(crop, (0, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(0, 0, 0), font=_font(16))
    for box in boxes:
        _draw_box(draw, box, (crop_box[0], crop_box[1] - 30), color)
    return panel


def _token_conflicts(line: PpOcrV6LineHint, latin_boxes: list[tuple[int, int, int, int]]) -> list[dict]:
    conflicts = []
    for token in line.words:
        if _has_ascii_alnum(token.text):
            continue
        cx = (token.bbox[0] + token.bbox[2]) / 2.0
        cy = (token.bbox[1] + token.bbox[3]) / 2.0
        owners = [box for box in latin_boxes if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]]
        if owners:
            conflicts.append({"token_index": token.token_index, "text": token.text, "bbox": token.bbox})
    return conflicts


def run_page(image_path: Path, raw_path: Path, out_dir: Path, scale: float) -> dict:
    page_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if page_bgr is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    height, width = page_bgr.shape[:2]
    page_rgb = Image.fromarray(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB))
    artifact = parse_ppocr_v6_prepass_result(
        {"prunedResult": _load_pruned(raw_path)},
        page_uid=image_path.stem,
        run_id="cached-audit",
        width=width,
        height=height,
    )
    page_dir = out_dir / image_path.stem
    lines_dir = page_dir / "mixed_lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    reports = []
    overview = page_rgb.copy()
    overview_draw = ImageDraw.Draw(overview)
    for line in artifact.lines:
        kind = _line_kind(line)
        counts[f"line:{kind}"] += 1
        if kind != "mixed":
            continue
        partition = partition_charocr_text_region(page_bgr, line, line.bbox)
        latin_boxes = [segment.bbox for segment in partition.segments if segment.kind == "text_latin"]
        other_boxes = [segment.bbox for segment in partition.segments if segment.kind == "text_other"]
        conflicts = _token_conflicts(line, latin_boxes)
        counts["mixed_issues"] += len(partition.issues)
        counts["latin_segments"] += len(latin_boxes)
        counts["other_segments"] += len(other_boxes)
        counts["latin_nonlatin_token_conflicts"] += len(conflicts)
        for box in latin_boxes:
            overview_draw.rectangle(box, outline=LATIN_COLOR, width=3)
        for issue in partition.issues:
            overview_draw.rectangle(issue.bbox, outline=ISSUE_COLOR, width=4)

        lx1, ly1, lx2, ly2 = line.bbox
        crop_box = _clip((lx1 - 24, ly1 - 18, lx2 + 24, ly2 + 18), width, height)
        token_boxes = [(token.bbox, TOKEN_COLOR) for token in line.words]
        route_boxes = [(box, LATIN_COLOR) for box in latin_boxes] + [(box, OTHER_COLOR) for box in other_boxes]
        panels = [
            _panel(page_rgb, crop_box, "00 PP-OCRv6 token proposals", token_boxes),
            _panel(page_rgb, crop_box, "01 production partition: orange EngCut, green LineCut", route_boxes),
            _masked_panel(page_bgr, crop_box, "02 EngCut-owned pixels", latin_boxes, LATIN_COLOR),
            _masked_panel(page_bgr, crop_box, "03 LineCut-owned pixels", other_boxes, OTHER_COLOR),
        ]
        canvas = Image.new("RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels) + 36), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), f"L{line.index:03d}: {line.text[:140]}", fill=(0, 0, 0), font=_font(17))
        y = 34
        for panel in panels:
            canvas.paste(panel, (0, y))
            y += panel.height
        output = lines_dir / f"line_{line.index:03d}_production_partition.png"
        canvas.save(output)
        reports.append({
            "line_index": line.index,
            "text": line.text,
            "latin_segments": [list(box) for box in latin_boxes],
            "other_segments": [list(box) for box in other_boxes],
            "issues": [{"code": issue.code, "bbox": list(issue.bbox), "message": issue.message} for issue in partition.issues],
            "nonlatin_token_conflicts": conflicts,
            "file": str(output),
        })
    if scale != 1:
        overview = overview.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    overview_path = page_dir / "production_mixed_routes.png"
    overview.save(overview_path)
    summary = {
        "schema": "production-mixed-text-partition-audit.v1",
        "image": str(image_path),
        "raw_pages": str(raw_path),
        "counts": dict(counts),
        "lines": reports,
        "overview": str(overview_path),
    }
    (page_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()
    images = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    pages = []
    totals: Counter[str] = Counter()
    for image_path in images:
        raw_path = args.raw_dir / f"{image_path.stem}.raw-pages.json"
        print(f"[page] {image_path.name}", flush=True)
        summary = run_page(image_path, raw_path, args.out_dir, args.scale)
        pages.append({"image": summary["image"], "counts": summary["counts"], "overview": summary["overview"]})
        totals.update(summary["counts"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batch = {
        "schema": "production-mixed-text-partition-audit-batch.v1",
        "input_dir": str(args.input_dir),
        "raw_dir": str(args.raw_dir),
        "pages": pages,
        "totals": dict(totals),
    }
    (args.out_dir / "batch_summary.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
