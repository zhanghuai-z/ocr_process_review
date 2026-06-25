#!/usr/bin/env python3
"""Echo 120169 footnote data before and after Hanwang micro-recblock."""
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

import cv2

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.ocr_ir_builder import build_ir_lines_from_item
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    attach_page_ocr_line_routes,
    block_bbox_xyxy,
    text_slice_routes_for_block,
)
from app.core.paddle_response import iter_ocr_preferred_items
from app.engines.hanwang.micro_recblock import run_micro_recblock
from app.models import BBox, Line, Page


XYXY = tuple[int, int, int, int]
TARGET_LABELS = {"footnote", "vision_footnote"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_from_layout(path: Path) -> Page:
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
    page.source_path = str(page_info.get("source_path") or "")
    page.blocks, _overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    return page


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


def _as_xyxy(value: Any) -> XYXY:
    x1, y1, x2, y2 = value
    return int(x1), int(y1), int(x2), int(y2)


def _save_crop(image, bbox: XYXY, path: Path) -> str:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    cv2.imwrite(str(path), crop)
    return path.name


def _draw_rect(canvas, bbox: XYXY, color: tuple[int, int, int], label: str = "") -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _target_records(page: Page) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(page.ppvl_parsing_res_list):
        label = str(record.get("block_label") or record.get("label") or "")
        if label in TARGET_LABELS:
            records.append((index, record))
    return records


def _pre_record_payload(index: int, record: dict[str, Any], page: Page, image, out_dir: Path) -> dict[str, Any]:
    block_bbox = block_bbox_xyxy(record, page.width, page.height)
    slices = []
    for route_index, route in enumerate(text_slice_routes_for_block(record, page.width, page.height)):
        bbox = _as_xyxy(route["bbox"])
        crop_name = _save_crop(
            image,
            bbox,
            out_dir / f"pre_idx{index:02d}_route{route_index:02d}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png",
        )
        slices.append(
            {
                "route_index": route_index,
                "line_idx": int(route.get("line_idx", -1)),
                "segment_idx": int(route.get("segment_idx", -1)),
                "bbox": list(bbox),
                "height": bbox[3] - bbox[1],
                "carved": bool(route.get("carved")),
                "crop": crop_name,
            }
        )
    return {
        "record_index": index,
        "label": str(record.get("block_label") or ""),
        "block_bbox": list(block_bbox),
        "block_content": str(record.get("block_content") or ""),
        "has_ppocr_routes": bool(record.get(LAYOUT_LINE_ROUTES_FIELD)),
        "line_routes": copy.deepcopy(record.get(LAYOUT_LINE_ROUTES_FIELD) or []),
        "text_slices": slices,
    }


def _row_payload(row: Any, page: Page, image, out_dir: Path) -> dict[str, Any]:
    value = row.to_dict()
    raw_block = value.get("raw_block") or {}
    audit = raw_block.get("_hanwang_bbox_audit") if isinstance(raw_block, dict) else {}
    groups = audit.get("hanwang_segimg_groups") if isinstance(audit, dict) else []
    for group_index, group in enumerate(groups or []):
        bbox_value = group.get("recog_group_bbox")
        if not bbox_value:
            continue
        bbox = _as_xyxy(bbox_value)
        group["crop"] = _save_crop(
            image,
            bbox,
            out_dir / f"post_idx{row.block_idx:02d}_group{group_index:02d}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png",
        )
    return value


def _write_overlay(path: Path, image, before: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    canvas = image.copy()
    for record in before:
        _draw_rect(canvas, _as_xyxy(record["block_bbox"]), (0, 0, 255), f"{record['record_index']} {record['label']}")
        for slice_item in record["text_slices"]:
            _draw_rect(canvas, _as_xyxy(slice_item["bbox"]), (0, 160, 0), f"pre {record['record_index']}.{slice_item['route_index']}")
    for row in rows:
        audit = ((row.get("raw_block") or {}).get("_hanwang_bbox_audit") or {})
        for group_index, group in enumerate(audit.get("hanwang_segimg_groups") or []):
            bbox_value = group.get("recog_group_bbox")
            if bbox_value:
                _draw_rect(canvas, _as_xyxy(bbox_value), (0, 165, 255), f"post {row['block_idx']}.{group_index}")
        for line_index, line in enumerate(row.get("lines") or []):
            for char_index, char in enumerate(line.get("chars") or []):
                bbox_value = char.get("bbox")
                if bbox_value:
                    _draw_rect(canvas, _as_xyxy(bbox_value), (255, 0, 0), f"c{line_index}.{char_index}")
    cv2.imwrite(str(path), canvas)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 120169 Footnote Hanwang Echo",
        "",
        "## 图例",
        "",
        "- 红框：VL1.6 footnote / vision_footnote 原始 block。",
        "- 绿框：PP-OCRv5 行框路由后，真正送入 Hanwang 的 text slice。",
        "- 橙框：Hanwang SegImg / Recog group。",
        "- 蓝框：Hanwang CharRcg 字符框。",
        "",
        f"Overlay: `{payload['overlay']}`",
        "",
        "## Hanwang 前",
        "",
    ]
    for record in payload["before"]:
        lines.append(f"### idx {record['record_index']} `{record['label']}`")
        lines.append(f"- block_bbox: `{record['block_bbox']}`")
        lines.append(f"- has_ppocr_routes: `{record['has_ppocr_routes']}`")
        lines.append(f"- block_content: {record['block_content']}")
        for item in record["text_slices"]:
            lines.append(
                f"- pre route {item['route_index']}: bbox=`{item['bbox']}` "
                f"h={item['height']} crop=`{item['crop']}`"
            )
        lines.append("")
    lines.append("## Hanwang 后")
    lines.append("")
    for row in payload["after"]:
        lines.append(f"### input idx {row['block_idx']} `{row['block_label']}`")
        lines.append(f"- source: `{row['source']}`")
        lines.append(f"- fallback_reason: `{row.get('fallback_reason') or ''}`")
        lines.append(f"- text: {row.get('text') or ''}")
        audit = ((row.get("raw_block") or {}).get("_hanwang_bbox_audit") or {})
        lines.append(f"- route_text_slice_bboxes: `{audit.get('route_text_slice_bboxes') or []}`")
        for group_index, group in enumerate(audit.get("hanwang_segimg_groups") or []):
            lines.append(
                f"- group {group_index}: route=`{group.get('route_text_slice_bbox')}` "
                f"seg=`{group.get('segimg_group_bbox')}` recog=`{group.get('recog_group_bbox')}` "
                f"clipped={group.get('clipped')} dropped={group.get('dropped')} crop=`{group.get('crop', '')}`"
            )
        for line_index, line in enumerate(row.get("lines") or []):
            lines.append(f"- line {line_index}: bbox=`{line.get('bbox')}` text={line.get('text') or ''}")
            for char in line.get("chars") or []:
                lines.append(
                    f"  - char `{char.get('text')}` bbox=`{char.get('bbox')}` "
                    f"source=`{char.get('source')}`"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=Path, default=REPO_ROOT / "file/244771纵校/120169.layout-api.json")
    parser.add_argument(
        "--ppocr",
        type=Path,
        default=Path("/mnt/d/project/ocr_process/null/claude/.cache/120169_ppocrv5_return_word_box.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/120169_footnote_hanwang_echo")
    args = parser.parse_args()

    page = _page_from_layout(args.layout)
    ppocr_lines = _load_ppocr_lines(args.ppocr, page)
    attach_page_ocr_line_routes(page.ppvl_parsing_res_list, ppocr_lines, page.width, page.height)
    targets = _target_records(page)
    if not targets:
        raise RuntimeError("No footnote or vision_footnote records found")

    image = cv2.imread(page.display_image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {page.display_image_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    before = [_pre_record_payload(index, record, page, image, args.out_dir) for index, record in targets]
    rows, stats = run_micro_recblock(
        image,
        [copy.deepcopy(record) for _index, record in targets],
        page_ocr_lines=ppocr_lines,
        include_chars=True,
    )
    after = [_row_payload(row, page, image, args.out_dir) for row in rows]
    payload = {
        "source_image": page.display_image_path,
        "layout_cache": str(args.layout),
        "ppocr_cache": str(args.ppocr),
        "stats": stats.__dict__,
        "before": before,
        "after": after,
        "overlay": "120169_footnote_hanwang_echo_overlay.png",
    }

    _write_overlay(args.out_dir / payload["overlay"], image, before, after)
    out_json = args.out_dir / "120169_footnote_hanwang_echo.json"
    out_md = args.out_dir / "120169_footnote_hanwang_echo.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote {args.out_dir / payload['overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
