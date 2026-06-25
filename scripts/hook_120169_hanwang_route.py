#!/usr/bin/env python3
"""Dump the 120169 PP-OCRv5 -> route -> Hanwang preinput bbox chain."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.ocr_ir_builder import build_ir_lines_from_item
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    attach_page_ocr_line_routes,
    block_bbox_xyxy,
    route_subblocks_for_block,
    text_slice_routes_for_block,
    vertical_overlap_ratio,
)
from app.core.paddle_response import iter_ocr_preferred_items
from app.models import BBox, Line, Page


XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_from_layout(path: Path) -> tuple[Page, list[dict[str, Any]]]:
    raw = _load_json(path)
    page_info = raw["page"]
    image_path = Path(str(page_info["display_image_path"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    page = Page(
        image_path=str(image_path),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    blocks, _overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    page.blocks = blocks
    return page, page.ppvl_parsing_res_list


def _load_ppocr_lines(path: Path, page: Page) -> list[Line]:
    raw = _load_json(path)
    response = raw.get("response", raw)
    fallback = BBox.from_xyxy(0, 0, page.width, page.height)
    lines: list[Line] = []
    for item in iter_ocr_preferred_items(response):
        for ir_line in build_ir_lines_from_item(
            item,
            image_shape=(page.height, page.width),
            fallback_bbox=fallback,
            include_word_boxes=False,
            existing_lines=[],
        ):
            lines.append(
                Line(
                    text=ir_line.text,
                    confidence=ir_line.confidence,
                    bbox=ir_line.bbox,
                    ocr_text=ir_line.text,
                )
            )
    return lines


def _area(box: XYXY) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersect(a: XYXY, b: XYXY) -> XYXY | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _coverage(candidate: XYXY, container: XYXY) -> float:
    overlap = _intersect(candidate, container)
    if overlap is None:
        return 0.0
    return _area(overlap) / max(1, _area(candidate))


def _as_list(box: XYXY) -> list[int]:
    return [int(value) for value in box]


def _route_summary(record: dict[str, Any], page: Page) -> dict[str, Any]:
    routes = copy.deepcopy(record.get(LAYOUT_LINE_ROUTES_FIELD) or [])
    return {
        "parent_bbox": _as_list(block_bbox_xyxy(record, page.width, page.height)),
        "subblocks": [
            {
                "label": str(item.get("label") or ""),
                "bbox": _as_list(tuple(item["bbox"])),
                "text": str(item.get("text") or ""),
            }
            for item in route_subblocks_for_block(record, page.width, page.height)
        ],
        "line_routes": routes,
        "text_slices": [
            {
                "line_idx": int(item.get("line_idx", -1)),
                "segment_idx": int(item.get("segment_idx", -1)),
                "bbox": list(item["bbox"]),
                "h": int(item["bbox"][3]) - int(item["bbox"][1]),
                "carved": bool(item.get("carved")),
            }
            for item in text_slice_routes_for_block(record, page.width, page.height)
        ],
    }


def _target_record_index(records: list[dict[str, Any]], requested: int | None) -> int:
    if requested is not None:
        return requested
    for index, record in enumerate(records):
        text = str(record.get("block_content") or record.get("text") or "")
        if "$ Time_{t} $" in text and "$ \\beta_{t} $" in text:
            return index
    raise RuntimeError("Could not find target 120169 paragraph record")


def _ppocr_lines_near_parent(lines: list[Line], parent_bbox: XYXY) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        box = line.bbox.to_xyxy()
        if _coverage(box, parent_bbox) < 0.1 and vertical_overlap_ratio(box, parent_bbox) < 0.15:
            continue
        result.append(
            {
                "idx": index,
                "text": line.text,
                "bbox": _as_list(box),
                "h": box[3] - box[1],
                "confidence": line.confidence,
            }
        )
    return result


def _run_segimg(image_path: Path, text_slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import cv2

    from app.engines.hanwang import native_bridge

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    recblocks = [tuple(int(value) for value in item["bbox"]) for item in text_slices]
    seg = native_bridge.run_linecut_segimg(image, recblocks_xyxy=recblocks, timeout=120.0)
    groups: list[dict[str, Any]] = []
    for area_idx, area in enumerate(seg.get("lines", []) or []):
        recblock = recblocks[area_idx] if area_idx < len(recblocks) else (0, 0, width, height)
        for group in area.get("groups", []) or []:
            raw_bbox = group.get("bbox") or {}
            left = int(raw_bbox.get("left", raw_bbox.get("x", recblock[0])))
            top = int(raw_bbox.get("top", raw_bbox.get("y", recblock[1])))
            if "right" in raw_bbox and "bottom" in raw_bbox:
                right = int(raw_bbox["right"])
                bottom = int(raw_bbox["bottom"])
            else:
                right = left + int(raw_bbox.get("width", max(0, recblock[2] - recblock[0])))
                bottom = top + int(raw_bbox.get("height", max(0, recblock[3] - recblock[1])))
            raw = (
                max(0, min(left, width)),
                max(0, min(top, height)),
                max(0, min(right, width)),
                max(0, min(bottom, height)),
            )
            final = _intersect(raw, recblock)
            groups.append(
                {
                    "area_idx": area_idx,
                    "route_text_slice_bbox": _as_list(recblock),
                    "segimg_group_bbox": _as_list(raw),
                    "recog_group_bbox": _as_list(final) if final is not None else None,
                    "clipped": final is not None and final != raw,
                    "dropped": final is None,
                }
            )
    return groups


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    before = payload["layout_only"]
    after = payload["after_ppocr_attach"]
    pp_lines = payload["ppocr_lines_near_parent"]
    seg_groups = payload.get("hanwang_segimg_groups") or []
    lines = [
        "# 120169 Hanwang 前路由 Hook",
        "",
        "## 结论",
        "",
        "- PP-OCRv5 原始行框高度是正常的。",
        "- 版面阶段的初始 `_layout_line_routes` 会使用 VL1.6 子块高度，目标段第 1 行会被压到 `2658..2710`。",
        "- 带 PP-OCRv5 prepass 后，`attach_page_ocr_line_routes()` 会把目标段第 1 行覆盖为 `2642..2718`。",
        "- 如果程序里仍看到 `2658..2710` 这类窄框，说明用的是版面阶段旧路由，或者当前项目没有重新跑 Hanwang OCR。",
        "",
        "## PP-OCRv5 行",
        "",
    ]
    for item in pp_lines:
        lines.append(f"- idx {item['idx']}: bbox={item['bbox']} h={item['h']} text={item['text']}")
    lines.extend(["", "## 版面阶段初始 text slices", ""])
    for item in before["text_slices"]:
        lines.append(f"- line {item['line_idx']} seg {item['segment_idx']}: bbox={item['bbox']} h={item['h']}")
    lines.extend(["", "## PP-OCRv5 attach 后 text slices", ""])
    for item in after["text_slices"]:
        lines.append(f"- line {item['line_idx']} seg {item['segment_idx']}: bbox={item['bbox']} h={item['h']}")
    if seg_groups:
        lines.extend(["", "## Hanwang SegImg groups", ""])
        for item in seg_groups:
            lines.append(
                "- area {area_idx}: route={route_text_slice_bbox} seg={segimg_group_bbox} "
                "final={recog_group_bbox} clipped={clipped} dropped={dropped}".format(**item)
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        type=Path,
        default=REPO_ROOT / "file/244771纵校/120169.layout-api.json",
    )
    parser.add_argument(
        "--ppocr",
        type=Path,
        default=Path("/mnt/d/project/ocr_process/null/claude/.cache/120169_ppocrv5_return_word_box.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "debug/120169_route_hook",
    )
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--with-segimg", action="store_true")
    args = parser.parse_args()

    page, records = _page_from_layout(args.layout)
    target_idx = _target_record_index(records, args.record_index)
    layout_only_records = copy.deepcopy(records)
    layout_only_record = layout_only_records[target_idx]
    parent_bbox = block_bbox_xyxy(layout_only_record, page.width, page.height)
    ppocr_lines = _load_ppocr_lines(args.ppocr, page)

    attached_records = copy.deepcopy(records)
    attach_page_ocr_line_routes(attached_records, ppocr_lines, page.width, page.height)
    attached_record = attached_records[target_idx]

    payload: dict[str, Any] = {
        "source_image": page.display_image_path,
        "layout_cache": str(args.layout),
        "ppocr_cache": str(args.ppocr),
        "target_record_index": target_idx,
        "target_parent_bbox": _as_list(parent_bbox),
        "ppocr_lines_near_parent": _ppocr_lines_near_parent(ppocr_lines, parent_bbox),
        "layout_only": _route_summary(layout_only_record, page),
        "after_ppocr_attach": _route_summary(attached_record, page),
    }

    if args.with_segimg:
        payload["hanwang_segimg_groups"] = _run_segimg(Path(page.display_image_path), payload["after_ppocr_attach"]["text_slices"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "120169_route_hook.json"
    out_md = args.out_dir / "120169_route_hook.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
