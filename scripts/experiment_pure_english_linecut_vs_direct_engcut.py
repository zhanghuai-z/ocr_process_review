#!/usr/bin/env python3
"""Compare direct EngCut with linecut-then-EngCut on pure English route lines.

This is a read-only diagnostic. It uses cached Paddle/PP-OCR route line boxes,
runs two local Hanwang paths on the same physical line, and writes visual
evidence under debug/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from app.core.latin_span_recovery import EngcutChar, engcut_chars_from_payload, offset_engcut_chars
from app.engines.hanwang import native_bridge

XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")[:140] or "item"


def _clip_box(bbox: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    x2 = max(x1, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(y1, min(int(y2), height))
    return x1, y1, x2, y2


def _bbox_from_raw(raw: object) -> XYXY | None:
    if not isinstance(raw, dict):
        return None
    try:
        left = int(raw.get("left", raw.get("x", 0)))
        top = int(raw.get("top", raw.get("y", 0)))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw.get("width", 0))
            bottom = top + int(raw.get("height", 0))
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _intersect(a: XYXY, b: XYXY) -> XYXY | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _route_text_metrics(text: str) -> dict[str, Any]:
    value = text or ""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", value))
    letters = len(re.findall(r"[A-Za-z]", value))
    digits = len(re.findall(r"[0-9]", value))
    nonspace = len(re.sub(r"\s+", "", value))
    latin_digit = letters + digits
    return {
        "letters": letters,
        "digits": digits,
        "cjk": cjk,
        "nonspace": nonspace,
        "latin_digit_ratio": round(latin_digit / max(1, nonspace), 4),
        "is_pure_english_line": cjk == 0 and letters >= 10 and latin_digit / max(1, nonspace) >= 0.45,
    }


def _candidate_routes(alignment: dict[str, Any], *, max_lines: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    routes_by_block = alignment.get("routes_by_block") or {}
    for block_key in sorted(routes_by_block, key=lambda item: int(item)):
        for route in sorted(routes_by_block[block_key], key=lambda item: int(item.get("route_idx", -1))):
            text = str(route.get("ppocr_text") or "")
            metrics = _route_text_metrics(text)
            if not metrics["is_pure_english_line"]:
                continue
            bbox = route.get("bbox") or []
            if len(bbox) != 4:
                continue
            item = {
                "block_idx": int(route.get("block_idx", block_key)),
                "route_idx": int(route.get("route_idx", -1)),
                "ppocr_line_idx": route.get("ppocr_line_idx"),
                "ppocr_text": text,
                "ppocr_confidence": route.get("ppocr_confidence"),
                "route_bbox": [int(v) for v in bbox],
                "metrics": metrics,
            }
            candidates.append(item)
            if len(candidates) >= max_lines:
                return candidates
    return candidates


def _run_engcut_on_box(image: np.ndarray, bbox: XYXY, *, timeout: float) -> dict[str, Any]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return {
            "bbox": [x1, y1, x2, y2],
            "text": "",
            "chars": [],
            "elapsed_seconds": 0.0,
            "error": "empty crop",
        }
    crop = image[y1:y2, x1:x2].copy()
    started = time.perf_counter()
    payload = native_bridge.run_eng20_recogline(crop, timeout=timeout)
    elapsed = time.perf_counter() - started
    local_chars = engcut_chars_from_payload(payload)
    page_chars = offset_engcut_chars(local_chars, dx=x1, dy=y1)
    return {
        "bbox": [x1, y1, x2, y2],
        "text": "".join(char.text for char in page_chars),
        "char_count": len(page_chars),
        "chars": [_char_to_dict(char) for char in page_chars],
        "elapsed_seconds": round(elapsed, 4),
        "error": str(payload.get("error") or ""),
    }


def _linecut_groups(image: np.ndarray, route_bbox: XYXY, *, timeout: float) -> list[dict[str, Any]]:
    seg = native_bridge.run_linecut_segimg(image, recblocks_xyxy=[route_bbox], timeout=timeout)
    groups: list[dict[str, Any]] = []
    for line_index, line in enumerate(seg.get("lines") or []):
        line_bbox = _bbox_from_raw(line.get("bbox"))
        for group_index, group in enumerate(line.get("groups") or []):
            group_bbox = _bbox_from_raw(group.get("bbox"))
            if group_bbox is None:
                continue
            clipped = _intersect(group_bbox, route_bbox)
            if clipped is None:
                continue
            groups.append({
                "line_index": line_index,
                "group_index": group_index,
                "line_bbox": list(line_bbox) if line_bbox else [],
                "bbox": list(clipped),
                "raw_bbox": list(group_bbox),
                "raw_count": group.get("raw_count"),
                "linecut_char_boxes": [
                    list(box)
                    for box in (
                        _bbox_from_raw(char.get("bbox"))
                        for char in (group.get("chars") or [])
                        if isinstance(char, dict)
                    )
                    if box is not None
                ],
            })
    return groups


def _char_to_dict(char: EngcutChar) -> dict[str, Any]:
    return {
        "text": char.text,
        "bbox": list(char.bbox) if char.bbox is not None else [],
        "code": char.code,
        "line_index": char.line_index,
        "group_index": char.group_index,
        "char_index": char.char_index,
    }


def _draw_box(canvas: np.ndarray, bbox: XYXY, color: tuple[int, int, int], *, thickness: int = 1) -> None:
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness, cv2.LINE_AA)


def _draw_page_chars(
    canvas: np.ndarray,
    chars: list[dict[str, Any]],
    color: tuple[int, int, int],
    *,
    label_chars: bool = False,
) -> None:
    for char in chars:
        bbox = char.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        _draw_box(canvas, (x1, y1, x2, y2), color)
        if label_chars and char.get("text"):
            cv2.putText(
                canvas,
                str(char["text"])[:2],
                (x1, max(12, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def _draw_linecut_char_boxes(canvas: np.ndarray, groups: list[dict[str, Any]], color: tuple[int, int, int]) -> None:
    for group in groups:
        for bbox in group.get("linecut_char_boxes") or []:
            if len(bbox) == 4:
                _draw_box(canvas, tuple(int(v) for v in bbox), color)


def _label(canvas: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _route_crop_canvas(image: np.ndarray, route_bbox: XYXY, *, pad: int = 12) -> tuple[np.ndarray, XYXY]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = route_bbox
    outer = _clip_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)
    crop = image[outer[1]:outer[3], outer[0]:outer[2]].copy()
    return crop, outer


def _shift_bbox(bbox: list[int] | tuple[int, int, int, int], origin: XYXY) -> XYXY:
    return (
        int(bbox[0]) - origin[0],
        int(bbox[1]) - origin[1],
        int(bbox[2]) - origin[0],
        int(bbox[3]) - origin[1],
    )


def _shift_chars(chars: list[dict[str, Any]], origin: XYXY) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for char in chars:
        bbox = char.get("bbox") or []
        shifted.append({
            **char,
            "bbox": list(_shift_bbox(bbox, origin)) if len(bbox) == 4 else [],
        })
    return shifted


def _write_visuals(
    image: np.ndarray,
    page_id: str,
    item: dict[str, Any],
    direct: dict[str, Any],
    groups: list[dict[str, Any]],
    linecut_runs: list[dict[str, Any]],
    out_page: Path,
) -> dict[str, str]:
    block_idx = int(item["block_idx"])
    route_idx = int(item["route_idx"])
    stem = f"{page_id}_b{block_idx:03d}_r{route_idx:03d}"
    route_bbox = tuple(int(v) for v in item["route_bbox"])

    page_canvas = image.copy()
    _draw_box(page_canvas, route_bbox, (0, 190, 0), thickness=2)
    for group in groups:
        bbox = tuple(int(v) for v in group["bbox"])
        _draw_box(page_canvas, bbox, (0, 150, 255), thickness=2)
    _draw_page_chars(page_canvas, direct.get("chars") or [], (255, 180, 0))
    for run in linecut_runs:
        _draw_page_chars(page_canvas, run.get("chars") or [], (255, 0, 255))
    _label(page_canvas, "green=PP-OCR route, orange=linecut group, cyan=direct EngCut, magenta=linecut>EngCut", (30, 50), (0, 0, 255))
    page_overlay = out_page / f"{stem}_page_overlay.png"
    cv2.imwrite(str(page_overlay), page_canvas)

    base_crop, origin = _route_crop_canvas(image, route_bbox)
    direct_crop = base_crop.copy()
    linecut_crop = base_crop.copy()
    _draw_box(direct_crop, _shift_bbox(route_bbox, origin), (0, 190, 0), thickness=2)
    _draw_page_chars(direct_crop, _shift_chars(direct.get("chars") or [], origin), (255, 180, 0), label_chars=True)
    _label(direct_crop, f"direct EngCut: {direct.get('text', '')[:100]}", (8, 18), (255, 180, 0))

    _draw_box(linecut_crop, _shift_bbox(route_bbox, origin), (0, 190, 0), thickness=1)
    for group in groups:
        _draw_box(linecut_crop, _shift_bbox(group["bbox"], origin), (0, 150, 255), thickness=2)
    shifted_groups: list[dict[str, Any]] = []
    for group in groups:
        shifted_groups.append({
            **group,
            "linecut_char_boxes": [
                list(_shift_bbox(bbox, origin))
                for bbox in group.get("linecut_char_boxes") or []
                if len(bbox) == 4
            ],
        })
    _draw_linecut_char_boxes(linecut_crop, shifted_groups, (0, 165, 255))
    for run in linecut_runs:
        _draw_page_chars(linecut_crop, _shift_chars(run.get("chars") or [], origin), (255, 0, 255), label_chars=True)
    linecut_text = " | ".join(str(run.get("text") or "") for run in linecut_runs)
    _label(linecut_crop, f"linecut > EngCut: {linecut_text[:100]}", (8, 18), (255, 0, 255))

    gap = np.full((max(direct_crop.shape[0], linecut_crop.shape[0]), 12, 3), 255, dtype=np.uint8)
    if direct_crop.shape[0] != linecut_crop.shape[0]:
        target_h = max(direct_crop.shape[0], linecut_crop.shape[0])
        direct_crop = _pad_to_height(direct_crop, target_h)
        linecut_crop = _pad_to_height(linecut_crop, target_h)
    compare = np.hstack([direct_crop, gap, linecut_crop])
    compare_path = out_page / f"{stem}_direct_vs_linecut.png"
    cv2.imwrite(str(compare_path), compare)
    return {
        "page_overlay": str(page_overlay.relative_to(REPO_ROOT)),
        "compare": str(compare_path.relative_to(REPO_ROOT)),
    }


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def _process_page(page_id: str, alignment_dir: Path, out_dir: Path, *, max_lines: int, timeout: float) -> dict[str, Any]:
    alignment_path = alignment_dir / page_id / "ppocr_route_token_alignment.json"
    alignment = _load_json(alignment_path)
    image_path = Path(str(alignment.get("source_image") or ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"source image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read source image: {image_path}")

    out_page = out_dir / page_id
    out_page.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_routes(alignment, max_lines=max_lines)
    results: list[dict[str, Any]] = []
    for item in candidates:
        route_bbox = tuple(int(v) for v in item["route_bbox"])
        direct = _run_engcut_on_box(image, route_bbox, timeout=timeout)
        groups = _linecut_groups(image, route_bbox, timeout=timeout)
        linecut_runs: list[dict[str, Any]] = []
        for group in groups:
            run = _run_engcut_on_box(image, tuple(int(v) for v in group["bbox"]), timeout=timeout)
            run["linecut_line_index"] = group.get("line_index")
            run["linecut_group_index"] = group.get("group_index")
            linecut_runs.append(run)
        visuals = _write_visuals(image, page_id, item, direct, groups, linecut_runs, out_page)
        result = {
            **item,
            "direct_engcut": direct,
            "linecut_group_count": len(groups),
            "linecut_groups": groups,
            "linecut_then_engcut": linecut_runs,
            "linecut_text_joined": " | ".join(str(run.get("text") or "") for run in linecut_runs),
            "visuals": visuals,
        }
        results.append(result)

    payload = {
        "schema": "pure_english_linecut_vs_direct_engcut.page.v0",
        "page_id": page_id,
        "source_image": str(image_path),
        "candidate_rule": "CJK count == 0, letters >= 10, (letters + digits) / nonspace >= 0.45",
        "candidate_count": len(candidates),
        "results": results,
    }
    _write_json(out_page / "pure_english_linecut_vs_direct_engcut.json", payload)
    return payload


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pure English LineCut vs Direct EngCut",
        "",
        "## Rule",
        "",
        "- Pure-English candidate: CJK count = 0, letters >= 10, latin/digit ratio >= 0.45.",
        "- Direct path: PP-OCR route line bbox -> EngCut.",
        "- LineCut path: PP-OCR route line bbox -> linecut group bbox -> EngCut.",
        "- Visual colors: green = PP-OCR route line, orange = linecut group/linecut char boxes, cyan = direct EngCut char boxes, magenta = linecut > EngCut char boxes.",
        "",
        "## Results",
        "",
        "| page | block | route | route bbox | direct text | linecut > EngCut text | compare | page overlay |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for page in payload["pages"]:
        for item in page.get("results") or []:
            direct = str((item.get("direct_engcut") or {}).get("text") or "").replace("|", "\\|")
            linecut = str(item.get("linecut_text_joined") or "").replace("|", "\\|")
            visuals = item.get("visuals") or {}
            compare = visuals.get("compare", "")
            overlay = visuals.get("page_overlay", "")
            lines.append(
                f"| {page['page_id']} | {item.get('block_idx')} | {item.get('route_idx')} | "
                f"`{item.get('route_bbox')}` | `{direct[:120]}` | `{linecut[:120]}` | "
                f"`{compare}` | `{overlay}` |"
            )
    lines.extend(["", "## PP-OCR Candidate Text", ""])
    for page in payload["pages"]:
        for item in page.get("results") or []:
            lines.append(
                f"- {page['page_id']} b{int(item.get('block_idx')):03d} r{int(item.get('route_idx')):03d}: "
                f"`{item.get('ppocr_text')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=REPO_ROOT / "debug/ppocr_route_token_alignment")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/pure_english_linecut_vs_direct_engcut")
    parser.add_argument("--pages", nargs="*", default=["120180"])
    parser.add_argument("--max-lines", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    pages: list[dict[str, Any]] = []
    for page_id in args.pages:
        pages.append(_process_page(str(page_id), args.alignment_dir, args.out_dir, max_lines=args.max_lines, timeout=args.timeout))

    summary = {
        "schema": "pure_english_linecut_vs_direct_engcut.batch.v0",
        "pages": [
            {
                "page_id": page["page_id"],
                "candidate_count": page["candidate_count"],
                "results": page["results"],
            }
            for page in pages
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "pure_english_linecut_vs_direct_engcut_summary.json", summary)
    _write_markdown(summary, args.out_dir / "pure_english_linecut_vs_direct_engcut_summary.md")
    print(args.out_dir / "pure_english_linecut_vs_direct_engcut_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
