#!/usr/bin/env python3
"""Render word-level fallback boxes from EngCut character boxes.

The input is the diagnostic JSON produced by
experiment_pure_english_linecut_vs_direct_engcut.py.  It does not run OCR; it
only merges existing EngCut char boxes into word/punctuation boxes for visual
review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox_union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _clip_box(box: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(x1), width)),
        max(0, min(int(y1), height)),
        max(0, min(int(x2), width)),
        max(0, min(int(y2), height)),
    )


def _shift_box(box: XYXY, origin: XYXY) -> XYXY:
    return box[0] - origin[0], box[1] - origin[1], box[2] - origin[0], box[3] - origin[1]


def _char_box(char: dict[str, Any]) -> XYXY | None:
    bbox = char.get("bbox") or []
    if len(bbox) != 4:
        return None
    box = tuple(int(v) for v in bbox)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _is_alnum(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9]$", text or ""))


def _is_letter(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]$", text or ""))


def _should_split_punctuation(chars: list[dict[str, Any]], index: int) -> bool:
    text = str(chars[index].get("text") or "")
    if _is_alnum(text):
        return False
    prev_text = str(chars[index - 1].get("text") or "") if index > 0 else ""
    next_text = str(chars[index + 1].get("text") or "") if index + 1 < len(chars) else ""
    # EngCut often inserts a dot inside a slanted word, e.g. Unbalan.ced.
    # Keep that as part of the word fallback instead of creating a false dot box.
    if text == "." and _is_letter(prev_text) and _is_letter(next_text):
        return False
    return True


def _group_chars(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for char in chars:
        box = _char_box(char)
        if box is None:
            continue
        grouped[(int(char.get("line_index") or 0), int(char.get("group_index") or 0))].append(char)
    return [
        sorted(items, key=lambda item: (_char_box(item) or (0, 0, 0, 0))[0])
        for _, items in sorted(grouped.items())
    ]


def _token_boxes_from_chars(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for group in _group_chars(chars):
        if not group:
            continue
        run: list[dict[str, Any]] = []
        for index, char in enumerate(group):
            box = _char_box(char)
            if box is None:
                continue
            if _should_split_punctuation(group, index):
                if run:
                    tokens.append(_make_token(run, kind="word"))
                    run = []
                tokens.append(_make_token([char], kind="punct"))
                continue
            run.append(char)
        if run:
            tokens.append(_make_token(run, kind="word"))
    return tokens


def _make_token(chars: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    boxes = [_char_box(char) for char in chars]
    valid_boxes = [box for box in boxes if box is not None]
    union = _bbox_union(valid_boxes)
    overlaps = 0
    small_gaps = 0
    sorted_boxes = sorted(valid_boxes)
    widths = [box[2] - box[0] for box in sorted_boxes]
    median_width = sorted(widths)[len(widths) // 2] if widths else 0
    for left, right in zip(sorted_boxes, sorted_boxes[1:]):
        gap = right[0] - left[2]
        if gap < 0:
            overlaps += 1
        if gap <= max(1, int(median_width * 0.12)):
            small_gaps += 1
    return {
        "kind": kind,
        "text": "".join(str(char.get("text") or "") for char in chars),
        "bbox": list(union),
        "char_count": len(chars),
        "overlap_pairs": overlaps,
        "small_gap_pairs": small_gaps,
        "source_chars": [
            {
                "text": char.get("text"),
                "bbox": list(_char_box(char) or (0, 0, 0, 0)),
                "group_index": char.get("group_index"),
                "char_index": char.get("char_index"),
            }
            for char in chars
        ],
    }


def _draw_box(canvas: np.ndarray, box: XYXY, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)


def _draw_label(canvas: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text[:32], point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _crop_page(image: np.ndarray, route_box: XYXY, pad: int = 16) -> tuple[np.ndarray, XYXY]:
    height, width = image.shape[:2]
    origin = _clip_box(
        (route_box[0] - pad, route_box[1] - pad, route_box[2] + pad, route_box[3] + pad),
        width,
        height,
    )
    return image[origin[1]:origin[3], origin[0]:origin[2]].copy(), origin


def _render_route(image: np.ndarray, item: dict[str, Any], out_page: Path, page_id: str) -> dict[str, Any]:
    route_box = tuple(int(v) for v in item["route_bbox"])
    chars = item.get("direct_engcut", {}).get("chars") or []
    token_boxes = _token_boxes_from_chars(chars)
    crop, origin = _crop_page(image, route_box)

    char_view = crop.copy()
    token_view = crop.copy()
    _draw_box(char_view, _shift_box(route_box, origin), (0, 180, 0), 2)
    _draw_box(token_view, _shift_box(route_box, origin), (0, 180, 0), 2)

    for char in chars:
        box = _char_box(char)
        if box is None:
            continue
        shifted = _shift_box(box, origin)
        _draw_box(char_view, shifted, (255, 120, 0), 1)
        _draw_label(char_view, str(char.get("text") or ""), (shifted[0], max(10, shifted[1] - 2)), (255, 120, 0))

    for token in token_boxes:
        box = tuple(int(v) for v in token["bbox"])
        shifted = _shift_box(box, origin)
        color = (0, 150, 255) if token["kind"] == "punct" else (0, 180, 0)
        thickness = 1 if token["kind"] == "punct" else 2
        _draw_box(token_view, shifted, color, thickness)
        _draw_label(token_view, str(token.get("text") or ""), (shifted[0], max(10, shifted[1] - 2)), color)

    if char_view.shape[0] != token_view.shape[0]:
        target_h = max(char_view.shape[0], token_view.shape[0])
        char_view = _pad_to_height(char_view, target_h)
        token_view = _pad_to_height(token_view, target_h)
    gap = np.full((char_view.shape[0], 12, 3), 255, dtype=np.uint8)
    compare = np.hstack([char_view, gap, token_view])

    block_idx = int(item["block_idx"])
    route_idx = int(item["route_idx"])
    stem = f"{page_id}_b{block_idx:03d}_r{route_idx:03d}_word_fallback"
    out_path = out_page / f"{stem}.png"
    cv2.imwrite(str(out_path), compare)
    return {
        "block_idx": block_idx,
        "route_idx": route_idx,
        "route_bbox": item["route_bbox"],
        "direct_text": item.get("direct_engcut", {}).get("text") or "",
        "token_count": len(token_boxes),
        "word_count": sum(1 for token in token_boxes if token["kind"] == "word"),
        "punct_count": sum(1 for token in token_boxes if token["kind"] == "punct"),
        "overlap_token_count": sum(1 for token in token_boxes if int(token.get("overlap_pairs") or 0) > 0),
        "small_gap_token_count": sum(1 for token in token_boxes if int(token.get("small_gap_pairs") or 0) > 0),
        "tokens": token_boxes,
        "overlay": str(out_path.relative_to(REPO_ROOT)),
    }


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# EngCut Word Fallback Overlay",
        "",
        "- Left: raw EngCut char boxes.",
        "- Right: fallback boxes. Green = merged word/number, orange = punctuation.",
        "- Interior dot between letters is kept inside the word box, because EngCut may invent dots in italic/slanted words.",
        "",
        "| page | block | route | words | punct | overlap tokens | small-gap tokens | overlay |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for page in payload["pages"]:
        for item in page["results"]:
            lines.append(
                f"| {page['page_id']} | {item['block_idx']} | {item['route_idx']} | "
                f"{item['word_count']} | {item['punct_count']} | {item['overlap_token_count']} | "
                f"{item['small_gap_token_count']} | `{item['overlay']}` |"
            )
    lines.extend(["", "## Token Samples", ""])
    for page in payload["pages"]:
        for item in page["results"][:3]:
            sample = " ".join(
                f"{token['text']}[{token['kind']}]"
                for token in item["tokens"][:30]
            )
            lines.append(f"- {page['page_id']} b{item['block_idx']:03d} r{item['route_idx']:03d}: `{sample}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "debug/pure_english_linecut_vs_direct_engcut/120180/pure_english_linecut_vs_direct_engcut.json",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/engcut_word_fallback_120180")
    args = parser.parse_args()

    source = _load_json(args.input)
    image = cv2.imread(str(source["source_image"]))
    if image is None:
        raise RuntimeError(f"Cannot read source image: {source['source_image']}")
    out_page = args.out_dir / str(source["page_id"])
    out_page.mkdir(parents=True, exist_ok=True)
    results = [_render_route(image, item, out_page, str(source["page_id"])) for item in source.get("results") or []]
    payload = {
        "schema": "engcut_word_fallback_overlay.batch.v0",
        "source": str(args.input),
        "pages": [
            {
                "page_id": str(source["page_id"]),
                "results": results,
            }
        ],
    }
    _write_json(args.out_dir / "engcut_word_fallback_summary.json", payload)
    _write_markdown(payload, args.out_dir / "engcut_word_fallback_summary.md")
    print(args.out_dir / "engcut_word_fallback_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
