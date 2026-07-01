"""Experiment: fuse Paddle table HTML with Hanwang table crop OCR geometry.

This is intentionally outside the main workflow. It answers three questions:
- Does the current Paddle-VL table artifact contain cell-level geometry?
- Can Hanwang produce usable character/line boxes inside a table crop?
- How do in-table formulas appear in Hanwang output compared with Paddle HTML?
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines.hanwang import native_bridge  # noqa: E402
from app.engines.hanwang.micro_recblock import _line_results_from_recog, _offset_line_results  # noqa: E402
from app.export.pdf import _table_rows_from_html  # noqa: E402


@dataclass
class TableExperimentSummary:
    page_number: int
    element_id: str
    bbox: dict[str, int]
    html_row_count: int
    html_cell_count: int
    html_formula_cell_count: int
    paddle_has_cell_geometry: bool
    segimg_line_count: int
    segimg_group_count: int
    recog_line_count: int
    recog_char_count: int
    recog_formula_like_char_count: int
    recog_strategy: str
    recog_error_count: int
    crop_path: str
    segimg_overlay_path: str
    recog_overlay_path: str
    notes: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    candidate = ROOT / ".cache" / "images" / path.name
    if candidate.exists():
        return candidate
    candidate = ROOT / raw.replace("\\", "/")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve image path: {raw}")


def _bbox_xywh_to_xyxy(bbox: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x = int(round(float(bbox.get("x") or 0)))
    y = int(round(float(bbox.get("y") or 0)))
    w = int(round(float(bbox.get("w") or 0)))
    h = int(round(float(bbox.get("h") or 0)))
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left, min(width, x + w))
    bottom = max(top, min(height, y + h))
    return left, top, right, bottom


def _html_cell_texts(html: str) -> list[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned: list[str] = []
    for cell in cells:
        text = re.sub(r"<[^>]+>", "", cell)
        text = " ".join(text.split())
        if text:
            cleaned.append(text)
    return cleaned


def _formula_like(text: str) -> bool:
    if not text:
        return False
    if "$" in text:
        return True
    return bool(re.search(r"[\\_^{}]|beta|times|alpha|gamma|theta|lambda", text, flags=re.IGNORECASE))


def _draw_segimg_overlay(crop, seg: dict[str, Any], out_path: Path) -> None:
    canvas = crop.copy()
    for line in seg.get("lines") or []:
        bbox = _native_bbox_to_xyxy_exclusive(line.get("bbox"), crop.shape[1], crop.shape[0])
        if bbox:
            left, top, right, bottom = bbox
            cv2.rectangle(canvas, (left, top), (max(left, right - 1), max(top, bottom - 1)), (0, 180, 0), 2)
        for group in line.get("groups") or []:
            gb = _native_bbox_to_xyxy_exclusive(group.get("bbox"), crop.shape[1], crop.shape[0])
            if gb:
                left, top, right, bottom = gb
                cv2.rectangle(canvas, (left, top), (max(left, right - 1), max(top, bottom - 1)), (255, 128, 0), 1)
    cv2.imwrite(str(out_path), canvas)


def _draw_recog_overlay(crop, lines, out_path: Path) -> None:
    canvas = crop.copy()
    for line in lines:
        left, top, right, bottom = map(int, line.bbox)
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 180, 0), 2)
        for char in line.chars:
            if not char.bbox:
                continue
            l, t, r, b = map(int, char.bbox)
            color = (0, 128, 255) if _formula_like(char.text) else (255, 128, 0)
            cv2.rectangle(canvas, (l, t), (r, b), color, 1)
            cv2.putText(
                canvas,
                char.text[:2],
                (l, max(10, t - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out_path), canvas)


def _segimg_counts(seg: dict[str, Any]) -> tuple[int, int]:
    lines = seg.get("lines") or []
    groups = 0
    for line in lines:
        groups += len(line.get("groups") or [])
    return len(lines), groups


def run_experiment(ir_path: Path, out_dir: Path) -> list[TableExperimentSummary]:
    data = _read_json(ir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[TableExperimentSummary] = []

    for page in data.get("pages") or []:
        image_path = _resolve_image_path(str(page.get("source_image") or ""))
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cv2 cannot read image: {image_path}")
        height, width = image.shape[:2]
        for element in page.get("elements") or []:
            if element.get("kind") != "table":
                continue
            element_id = str(element.get("id") or "table")
            bbox = element.get("bbox") if isinstance(element.get("bbox"), dict) else {}
            left, top, right, bottom = _bbox_xywh_to_xyxy(bbox, width, height)
            crop = image[top:bottom, left:right].copy()
            if crop.size == 0:
                continue

            table_dir = out_dir / f"p{page.get('page_number')}_{element_id}"
            table_dir.mkdir(parents=True, exist_ok=True)
            crop_path = table_dir / "table_crop.png"
            cv2.imwrite(str(crop_path), crop)

            payload = element.get("payload") if isinstance(element.get("payload"), dict) else {}
            html = str(payload.get("text") or "")
            html_rows = _table_rows_from_html(html)
            html_cells = _html_cell_texts(html)
            has_cell_geometry = _paddle_has_cell_geometry(element)
            (table_dir / "paddle_html_rows.json").write_text(
                json.dumps(
                    [
                        {
                            "row_index": idx,
                            "text": row,
                            "formula_like": _formula_like(row),
                        }
                        for idx, row in enumerate(html_rows)
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            notes: list[str] = []
            if not has_cell_geometry:
                notes.append("Paddle artifact has table bbox/HTML only; no cell/char geometry found.")
            if any(_formula_like(cell) for cell in html_cells):
                notes.append("HTML contains formula-like cells; Hanwang main OCR may not preserve LaTeX semantics.")

            seg = native_bridge.run_linecut_segimg(crop, recblocks_xyxy=[(0, 0, crop.shape[1], crop.shape[0])], timeout=120.0)
            seg_path = table_dir / "hanwang_segimg.json"
            seg_path.write_text(json.dumps(seg, ensure_ascii=False, indent=2), encoding="utf-8")
            seg_line_count, seg_group_count = _segimg_counts(seg)
            segimg_overlay_path = table_dir / "hanwang_segimg_overlay.png"
            _draw_segimg_overlay(crop, seg, segimg_overlay_path)

            lines, recog_strategy, recog_errors = _recognize_table_crop(crop, seg)
            recog_payload = [
                {
                    "text": line.text,
                    "bbox": list(line.bbox),
                    "confidence": line.confidence,
                    "chars": [
                        {
                            "text": char.text,
                            "bbox": list(char.bbox) if char.bbox else None,
                            "confidence": char.confidence,
                            "source": char.source,
                        }
                        for char in line.chars
                    ],
                }
                for line in lines
            ]
            (table_dir / "hanwang_recog_lines.json").write_text(
                json.dumps(recog_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            recog_overlay_path = table_dir / "hanwang_recog_overlay.png"
            _draw_recog_overlay(crop, lines, recog_overlay_path)
            recog_char_count = sum(len(line.chars) for line in lines)
            formula_like_count = sum(
                1
                for line in lines
                for char in line.chars
                if _formula_like(char.text)
            )

            summaries.append(TableExperimentSummary(
                page_number=int(page.get("page_number") or 0),
                element_id=element_id,
                bbox={key: int(value) for key, value in bbox.items()},
                html_row_count=len(html_rows),
                html_cell_count=len(html_cells),
                html_formula_cell_count=sum(1 for cell in html_cells if _formula_like(cell)),
                paddle_has_cell_geometry=has_cell_geometry,
                segimg_line_count=seg_line_count,
                segimg_group_count=seg_group_count,
                recog_line_count=len(lines),
                recog_char_count=recog_char_count,
                recog_formula_like_char_count=formula_like_count,
                recog_strategy=recog_strategy,
                recog_error_count=len(recog_errors),
                crop_path=_repo_rel(crop_path),
                segimg_overlay_path=_repo_rel(segimg_overlay_path),
                recog_overlay_path=_repo_rel(recog_overlay_path),
                notes=notes,
            ))
            if recog_errors:
                (table_dir / "hanwang_recog_errors.json").write_text(
                    json.dumps(recog_errors, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            _write_row_comparison(html_rows, lines, table_dir / "row_comparison.md")
    return summaries


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _recognize_table_crop(crop, seg: dict[str, Any]):
    h, w = crop.shape[:2]
    try:
        raw = native_bridge.run_linecut_recog(crop, with_charrcg=True, timeout=120.0)
        lines = _line_results_from_recog(
            raw,
            fallback_bbox=(0, 0, w, h),
            include_chars=True,
            fallback_empty=False,
        )
        return lines, "whole_table_recog", []
    except Exception as exc:
        errors = [{
            "strategy": "whole_table_recog",
            "bbox": [0, 0, w, h],
            "error": str(exc),
        }]

    group_bboxes = _group_bboxes_from_segimg(seg, w, h)
    lines = []
    for idx, bbox in enumerate(group_bboxes):
        left, top, right, bottom = bbox
        group_crop = crop[top:bottom, left:right].copy()
        if group_crop.size == 0:
            continue
        try:
            raw = native_bridge.run_linecut_recog(group_crop, with_charrcg=True, timeout=120.0)
            local_lines = _line_results_from_recog(
                raw,
                fallback_bbox=(0, 0, right - left, bottom - top),
                include_chars=True,
                fallback_empty=False,
            )
            lines.extend(_offset_line_results(local_lines, dx=left, dy=top))
        except Exception as exc:
            errors.append({
                "strategy": "segimg_group_recog",
                "group_index": idx,
                "bbox": list(bbox),
                "error": str(exc),
            })
    return lines, "segimg_group_recog", errors


def _group_bboxes_from_segimg(seg: dict[str, Any], width: int, height: int) -> list[tuple[int, int, int, int]]:
    values: list[tuple[int, int, int, int]] = []
    for line in seg.get("lines") or []:
        for group in line.get("groups") or []:
            bbox = _native_bbox_to_xyxy_exclusive(group.get("bbox"), width, height)
            if bbox:
                values.append(bbox)
    return values


def _native_bbox_to_xyxy_exclusive(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if isinstance(value, dict):
        try:
            left = int(round(float(value.get("left", 0))))
            top = int(round(float(value.get("top", 0))))
            right = int(round(float(value.get("right", left))))
            bottom = int(round(float(value.get("bottom", top))))
        except (TypeError, ValueError):
            return None
        native_width = _optional_int(value.get("width"))
        native_height = _optional_int(value.get("height"))
        if native_width is not None and native_width == right - left + 1:
            right += 1
        if native_height is not None and native_height == bottom - top + 1:
            bottom += 1
    elif isinstance(value, list) and len(value) == 4:
        try:
            left, top, right, bottom = [int(round(float(item))) for item in value]
        except (TypeError, ValueError):
            return None
    else:
        return None
    left = max(0, min(width, left))
    right = max(left, min(width, right))
    top = max(0, min(height, top))
    bottom = max(top, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _optional_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _paddle_has_cell_geometry(element: dict[str, Any]) -> bool:
    stack: list[Any] = [element]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            keys = {str(key).lower() for key in current.keys()}
            if (
                ("cell_bbox" in keys or "cell_polygon" in keys or "cell" in keys)
                and any("bbox" in key or "polygon" in key or "points" in key for key in keys)
            ):
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _write_markdown(summaries: list[TableExperimentSummary], out_path: Path) -> None:
    lines = [
        "# Table Hanwang Fusion Experiment",
        "",
        "| page | element | Paddle rows/cells/formula-cells | Paddle cell geometry | SegImg lines/groups | Recog strategy/errors | Recog lines/chars/formula-like | crop | recog overlay |",
        "|---:|---|---:|---|---:|---|---:|---|---|",
    ]
    for item in summaries:
        lines.append(
            "| {page} | {element} | {rows}/{cells}/{formula_cells} | {cell_geo} | {seg_lines}/{seg_groups} | {strategy}/{errors} | {recog_lines}/{recog_chars}/{formula_like} | `{crop}` | `{overlay}` |".format(
                page=item.page_number,
                element=item.element_id,
                rows=item.html_row_count,
                cells=item.html_cell_count,
                formula_cells=item.html_formula_cell_count,
                cell_geo="yes" if item.paddle_has_cell_geometry else "no",
                seg_lines=item.segimg_line_count,
                seg_groups=item.segimg_group_count,
                strategy=item.recog_strategy,
                errors=item.recog_error_count,
                recog_lines=item.recog_line_count,
                recog_chars=item.recog_char_count,
                formula_like=item.recog_formula_like_char_count,
                crop=item.crop_path,
                overlay=item.recog_overlay_path,
            )
        )
    lines.extend(["", "## Notes", ""])
    for item in summaries:
        for note in item.notes:
            lines.append(f"- page {item.page_number} {item.element_id}: {note}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_row_comparison(paddle_rows: list[str], recog_lines, out_path: Path) -> None:
    max_rows = max(len(paddle_rows), len(recog_lines))
    lines = [
        "# Paddle HTML vs Hanwang Row OCR",
        "",
        "| row | Paddle HTML row | Paddle formula-like | Hanwang row OCR | Hanwang formula-like |",
        "|---:|---|---|---|---|",
    ]
    for idx in range(max_rows):
        paddle_text = paddle_rows[idx] if idx < len(paddle_rows) else ""
        hanwang_text = recog_lines[idx].text if idx < len(recog_lines) else ""
        lines.append(
            "| {idx} | {paddle} | {paddle_formula} | {hanwang} | {hanwang_formula} |".format(
                idx=idx,
                paddle=_markdown_cell(paddle_text),
                paddle_formula="yes" if _formula_like(paddle_text) else "no",
                hanwang=_markdown_cell(hanwang_text),
                hanwang_formula="yes" if _formula_like(hanwang_text) else "no",
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_cell(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=Path, default=ROOT / "file" / "temp1" / "未命名项目.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "debug" / "table_hanwang_fusion" / "temp1")
    args = parser.parse_args()

    summaries = run_experiment(args.ir, args.out_dir)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(item) for item in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = args.out_dir / "summary.md"
    _write_markdown(summaries, md_path)
    print(md_path)


if __name__ == "__main__":
    main()
