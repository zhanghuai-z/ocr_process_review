#!/usr/bin/env python3
"""Render visual evidence for old fallback vs new Latin strategy."""
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
import numpy as np


OLD_COLOR = (255, 0, 255)       # magenta
PRIMARY_COLOR = (255, 80, 0)    # blue-ish
FB_EXACT_COLOR = (0, 170, 0)    # green
REVIEW_COLOR = (0, 165, 255)    # orange
TEXT_COLOR = (20, 20, 20)
BG = (248, 248, 248)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not value or len(value) != 4:
        return None
    return tuple(int(v) for v in value)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _clip(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _expand(box: tuple[int, int, int, int], width: int, height: int, pad_x: int = 180, pad_y: int = 90) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return _clip((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _draw_text(img, text: str, org: tuple[int, int], scale: float = 0.55, color=TEXT_COLOR, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_box(
    img,
    *,
    crop_origin: tuple[int, int],
    bbox: tuple[int, int, int, int],
    color,
    label: str,
    thickness: int = 2,
) -> None:
    ox, oy = crop_origin
    x1, y1, x2, y2 = bbox
    p1 = (x1 - ox, y1 - oy)
    p2 = (x2 - ox, y2 - oy)
    cv2.rectangle(img, p1, p2, color, thickness)
    label_y = max(14, p1[1] - 5)
    _draw_text(img, label, (p1[0], label_y), 0.45, color, 1)


def _fit_panel(crop, width: int = 520, height: int = 220):
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((height, width, 3), 240, np.uint8)
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 255, np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    panel[y:y + new_h, x:x + new_w] = resized
    return panel


def _status_label(status: str) -> tuple[str, tuple[int, int, int]]:
    if status == "primary_exact":
        return "PRIMARY EXACT", PRIMARY_COLOR
    if status == "fallback_exact_after_takeover":
        return "FALLBACK EXACT", FB_EXACT_COLOR
    if status == "fallback_variant_review":
        return "REVIEW CANDIDATE", REVIEW_COLOR
    return status.upper(), REVIEW_COLOR


def _render_item(item: dict[str, Any], image, out_path: Path) -> dict[str, Any]:
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    old_box = _bbox(item.get("old_bbox"))
    primary_box = _bbox(item.get("primary_bbox"))
    if old_box:
        boxes.append(old_box)
    if primary_box:
        boxes.append(primary_box)
    reverse_boxes = []
    for candidate in item.get("reverse_candidates") or []:
        candidate_box = _bbox(candidate.get("bbox"))
        if candidate_box:
            reverse_boxes.append((candidate, candidate_box))
            boxes.append(candidate_box)
    if not boxes:
        boxes.append((0, 0, width, height))
    crop_box = _expand(_union(boxes), width, height)
    x1, y1, x2, y2 = crop_box

    old_crop = image[y1:y2, x1:x2].copy()
    new_crop = image[y1:y2, x1:x2].copy()
    if old_box:
        _draw_box(
            old_crop,
            crop_origin=(x1, y1),
            bbox=old_box,
            color=OLD_COLOR,
            label=f"OLD {item.get('old_hanwang_text_span', '')}",
        )
    if primary_box:
        _draw_box(
            new_crop,
            crop_origin=(x1, y1),
            bbox=primary_box,
            color=PRIMARY_COLOR,
            label=f"PRIMARY {item['text']}",
        )
    for candidate, candidate_box in reverse_boxes:
        status = item.get("new_strategy_status")
        color = FB_EXACT_COLOR if status == "fallback_exact_after_takeover" else REVIEW_COLOR
        _draw_box(
            new_crop,
            crop_origin=(x1, y1),
            bbox=candidate_box,
            color=color,
            label=f"FB {candidate.get('text', '')}",
        )

    old_panel = _fit_panel(old_crop)
    new_panel = _fit_panel(new_crop)
    card_h = 330
    card_w = 1080
    card = np.full((card_h, card_w, 3), BG, np.uint8)
    cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (210, 210, 210), 1)

    title = f"{item['page_id']} b{item['block_idx']} l{item.get('line_idx')} token={item['text']}"
    status_text, status_color = _status_label(str(item.get("new_strategy_status") or ""))
    _draw_text(card, title, (16, 28), 0.72, TEXT_COLOR, 2)
    _draw_text(card, status_text, (16, 58), 0.65, status_color, 2)
    _draw_text(card, "left: old alignment fallback", (16, 86), 0.52, OLD_COLOR, 1)
    _draw_text(card, "right: new primary/fallback result", (555, 86), 0.52, TEXT_COLOR, 1)
    card[100:320, 16:536] = old_panel
    card[100:320, 544:1064] = new_panel
    cv2.imwrite(str(out_path), card)
    return {
        "page_id": item["page_id"],
        "token": item["text"],
        "status": item["new_strategy_status"],
        "path": str(out_path),
    }


def _make_atlas(cards: list[Path], out_path: Path) -> None:
    thumbs = []
    for path in cards:
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (540, 165), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    atlas = np.full((rows * 165, cols * 540, 3), 245, np.uint8)
    for index, thumb in enumerate(thumbs):
        r = index // cols
        c = index % cols
        atlas[r * 165:(r + 1) * 165, c * 540:(c + 1) * 540] = thumb
    cv2.imwrite(str(out_path), atlas)


def _write_md(summary: list[dict[str, Any]], atlas: Path, out_path: Path) -> None:
    lines = [
        "# Latin Fallback Image Evidence",
        "",
        "## Legend",
        "",
        "- Magenta: old alignment fallback bbox.",
        "- Blue: new primary `Paddle token + EngCut exact` bbox.",
        "- Green: new fallback exact candidate after line takeover.",
        "- Orange: fallback review candidate; geometry is useful, text is not exact.",
        "",
        f"## Atlas",
        "",
        f"![atlas]({atlas.name})",
        "",
        "## Individual Evidence",
        "",
    ]
    for item in summary:
        rel = Path(item["path"]).name
        lines.append(f"### {item['page_id']} `{item['token']}` `{item['status']}`")
        lines.append(f"![{item['page_id']} {item['token']}]({rel})")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=REPO_ROOT / "debug/old_fallback_vs_new_latin_strategy/old_fallback_vs_new_latin_strategy.json")
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "file/244771纵校")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/old_fallback_vs_new_latin_strategy/image_evidence")
    args = parser.parse_args()

    payload = _load_json(args.comparison)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    card_paths: list[Path] = []
    for index, item in enumerate(payload.get("items") or []):
        page_id = str(item["page_id"])
        if page_id not in image_cache:
            image_path = args.image_dir / f"{page_id}.tif"
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Cannot read image: {image_path}")
            image_cache[page_id] = image
        out_path = args.out_dir / f"{index + 1:02d}_{page_id}_b{item['block_idx']:03d}_l{item.get('line_idx')}_{_safe_name(str(item['text']))}.png"
        summary.append(_render_item(item, image_cache[page_id], out_path))
        card_paths.append(out_path)
    atlas_path = args.out_dir / "00_atlas.png"
    _make_atlas(card_paths, atlas_path)
    _write_md(summary, atlas_path, args.out_dir / "README.md")
    print(f"wrote {atlas_path}")
    print(f"wrote {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
