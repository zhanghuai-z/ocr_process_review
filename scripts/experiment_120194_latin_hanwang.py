#!/usr/bin/env python3
"""Echo Hanwang Latin char boxes on sample 120194."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.paddle_line_routing import block_bbox_xyxy
from app.engines.hanwang.micro_recblock import run_micro_recblock
from app.models import Page


XYXY = tuple[int, int, int, int]
TEXT_LABELS = {
    "caption",
    "figure_caption",
    "figure_title",
    "text",
    "paragraph_title",
    "title",
    "reference",
    "paragraph_text",
    "table_caption",
    "table_note",
    "table_title",
    "footnote",
    "vision_footnote",
}


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


def _has_latin_text(text: str) -> bool:
    return any(ch.isascii() and (ch.isalnum() or ch in "/.,;:-()") for ch in text)


def _is_latin_char(text: str) -> bool:
    return len(text) == 1 and text.isascii() and (text.isalnum() or text in "/.,;:-()")


def _as_xyxy(value: Any) -> XYXY:
    x1, y1, x2, y2 = value
    return int(x1), int(y1), int(x2), int(y2)


def _draw_rect(canvas, bbox: XYXY, color: tuple[int, int, int], label: str = "") -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _save_line_crop(image, line: dict[str, Any], out_dir: Path, row_idx: int, line_idx: int) -> str:
    x1, y1, x2, y2 = _as_xyxy(line["bbox"])
    pad = 12
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2].copy()
    for char in line.get("chars") or []:
        bbox_value = char.get("bbox")
        if not bbox_value:
            continue
        bx1, by1, bx2, by2 = _as_xyxy(bbox_value)
        color = (255, 0, 0)
        if _is_latin_char(str(char.get("text") or "")):
            color = (0, 0, 255)
        cv2.rectangle(crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), color, 1)
    name = f"line_row{row_idx:02d}_{line_idx:02d}_{x1}_{y1}_{x2}_{y2}.png"
    cv2.imwrite(str(out_dir / name), crop)
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=Path, default=REPO_ROOT / "file/244771纵校/120194.layout-api.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/120194_latin_hanwang")
    args = parser.parse_args()

    page = _page_from_layout(args.layout)
    image = cv2.imread(page.display_image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {page.display_image_path}")

    targets: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(page.ppvl_parsing_res_list):
        label = str(record.get("block_label") or record.get("label") or "")
        text = str(record.get("block_content") or "")
        if label in TEXT_LABELS and _has_latin_text(text):
            targets.append((index, record))
    if not targets:
        raise RuntimeError("No Latin-bearing text blocks found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_records = [record for _index, record in targets]
    rows, stats = run_micro_recblock(image, target_records, include_chars=True)

    before = []
    index_by_row = {}
    for row_idx, (record_index, record) in enumerate(targets):
        index_by_row[row_idx] = record_index
        before.append({
            "record_index": record_index,
            "row_index": row_idx,
            "label": str(record.get("block_label") or ""),
            "block_bbox": list(block_bbox_xyxy(record, page.width, page.height)),
            "block_content": str(record.get("block_content") or ""),
        })

    after = []
    overlay = image.copy()
    for item in before:
        _draw_rect(overlay, _as_xyxy(item["block_bbox"]), (0, 0, 255), f"{item['record_index']} {item['label']}")

    for row in rows:
        value = row.to_dict()
        record_index = index_by_row.get(row.block_idx, row.block_idx)
        value["record_index"] = record_index
        latin_chars = []
        suspicious = []
        for line_idx, line in enumerate(value.get("lines") or []):
            line_has_latin = any(_is_latin_char(str(char.get("text") or "")) for char in line.get("chars") or [])
            if line_has_latin:
                line["crop"] = _save_line_crop(image, line, args.out_dir, row.block_idx, line_idx)
            for char_idx, char in enumerate(line.get("chars") or []):
                bbox_value = char.get("bbox")
                text = str(char.get("text") or "")
                if bbox_value:
                    bbox = _as_xyxy(bbox_value)
                    color = (255, 0, 0)
                    if _is_latin_char(text):
                        color = (0, 0, 255)
                    _draw_rect(overlay, bbox, color)
                if _is_latin_char(text):
                    bbox = _as_xyxy(bbox_value) if bbox_value else None
                    entry = {
                        "line_idx": line_idx,
                        "char_idx": char_idx,
                        "text": text,
                        "bbox": list(bbox) if bbox else None,
                        "width": bbox[2] - bbox[0] if bbox else None,
                        "height": bbox[3] - bbox[1] if bbox else None,
                    }
                    latin_chars.append(entry)
                    if bbox and (bbox[2] - bbox[0] > 45 or bbox[3] - bbox[1] > 55):
                        suspicious.append(entry)
        value["latin_char_count"] = len(latin_chars)
        value["latin_chars"] = latin_chars
        value["suspicious_latin_boxes"] = suspicious
        after.append(value)

    overlay_path = args.out_dir / "120194_latin_hanwang_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "source_image": page.display_image_path,
        "layout": str(args.layout),
        "stats": stats.__dict__,
        "before": before,
        "after": after,
        "overlay": overlay_path.name,
    }
    json_path = args.out_dir / "120194_latin_hanwang.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 120194 Latin Hanwang Echo",
        "",
        "## 图例",
        "",
        "- 红框：VL1.6 原始文字 block。",
        "- 蓝框：Hanwang CharRcg 字符框。",
        "- 红色小字符框：Hanwang CharRcg 中的拉丁字符 / 数字 / 常见英文标点。",
        "",
        f"Overlay: `{overlay_path.name}`",
        "",
        "## Summary",
        "",
        f"- target_blocks: `{len(targets)}`",
        f"- n_blocks_hanwang: `{stats.n_blocks_hanwang}`",
        f"- n_groups: `{stats.n_groups}`",
        f"- recog_probe_calls: `{stats.recog_probe_calls}`",
        "",
    ]
    for row in after:
        md_lines.append(f"### record {row['record_index']} `{row['block_label']}`")
        md_lines.append(f"- source: `{row['source']}`")
        md_lines.append(f"- text: {row.get('text') or ''}")
        md_lines.append(f"- latin_char_count: `{row['latin_char_count']}`")
        if row["suspicious_latin_boxes"]:
            md_lines.append(f"- suspicious_latin_boxes: `{row['suspicious_latin_boxes'][:12]}`")
        for line_idx, line in enumerate(row.get("lines") or []):
            if any(_is_latin_char(str(char.get("text") or "")) for char in line.get("chars") or []):
                md_lines.append(f"- line {line_idx}: bbox=`{line.get('bbox')}` crop=`{line.get('crop')}` text={line.get('text')}")
        md_lines.append("")
    md_path = args.out_dir / "120194_latin_hanwang.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
