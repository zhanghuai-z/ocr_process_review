#!/usr/bin/env python3
"""Render the proposed Latin strategy overlay for page 120180.

Strategy view:
- exact English words: keep EngCut character boxes;
- inexact English words: use one Paddle-text word box on EngCut geometry;
- digits and punctuation: keep character-level boxes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = ROOT / "debug/latin_word_truth_fallback_120180/latin_word_truth_fallback_120180.json"
OUT_DIR = ROOT / "debug/latin_strategy_overlay_120180"

XYXY = tuple[int, int, int, int]


COLORS = {
    "route": (0, 150, 0),
    "exact_char": (0, 145, 0),
    "word_fallback": (190, 0, 190),
    "number": (255, 80, 0),
    "punct": (0, 140, 255),
    "raw": (255, 130, 0),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _box(value: Any) -> XYXY | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    box = tuple(int(v) for v in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _clip(box: XYXY, width: int, height: int) -> XYXY:
    return (
        max(0, min(width, box[0])),
        max(0, min(height, box[1])),
        max(0, min(width, box[2])),
        max(0, min(height, box[3])),
    )


def _shift(box: XYXY, origin: XYXY) -> XYXY:
    return box[0] - origin[0], box[1] - origin[1], box[2] - origin[0], box[3] - origin[1]


def _crop(image: np.ndarray, bbox: XYXY, pad: int = 20) -> tuple[np.ndarray, XYXY]:
    height, width = image.shape[:2]
    origin = _clip((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), width, height)
    return image[origin[1] : origin[3], origin[0] : origin[2]].copy(), origin


def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def _draw_box(canvas: np.ndarray, box: XYXY, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.36) -> None:
    cv2.putText(canvas, text[:32], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _char_boxes(unit: dict[str, Any]) -> list[tuple[str, XYXY]]:
    chars: list[tuple[str, XYXY]] = []
    truth_text = str(unit.get("text") or "")
    for index, char in enumerate(unit.get("chars") or []):
        box = _box(char.get("bbox"))
        if box is None:
            continue
        label = truth_text[index] if index < len(truth_text) else str(char.get("text") or "")
        chars.append((label, box))
    return chars


def _render_route(image: np.ndarray, route: dict[str, Any], raw_route: dict[str, Any] | None, out_page: Path) -> str:
    route_box = _box(route.get("route_bbox"))
    if route_box is None:
        raise RuntimeError(f"Invalid route bbox: {route.get('route_bbox')}")
    crop, origin = _crop(image, route_box)
    raw_view = crop.copy()
    strategy_view = crop.copy()

    _draw_box(raw_view, _shift(route_box, origin), COLORS["route"], 2)
    _draw_box(strategy_view, _shift(route_box, origin), COLORS["route"], 2)

    if raw_route is not None:
        for raw_char in raw_route.get("direct_engcut", {}).get("chars") or []:
            box = _box(raw_char.get("bbox"))
            if box is None:
                continue
            shifted = _shift(box, origin)
            _draw_box(raw_view, shifted, COLORS["raw"], 1)
            _draw_label(raw_view, str(raw_char.get("text") or ""), shifted[0], max(11, shifted[1] - 2), COLORS["raw"], 0.33)

    for unit in route.get("units") or []:
        kind = unit.get("kind")
        if kind == "word" and unit.get("exact_char_usable"):
            for label, box in _char_boxes(unit):
                shifted = _shift(box, origin)
                _draw_box(strategy_view, shifted, COLORS["exact_char"], 1)
                _draw_label(strategy_view, label, shifted[0], max(11, shifted[1] - 2), COLORS["exact_char"], 0.32)
            continue
        if kind == "word":
            box = _box(unit.get("bbox"))
            if box is None:
                continue
            shifted = _shift(box, origin)
            _draw_box(strategy_view, shifted, COLORS["word_fallback"], 2)
            _draw_label(strategy_view, str(unit.get("text") or ""), shifted[0], max(12, shifted[1] - 3), COLORS["word_fallback"], 0.4)
            continue
        for label, box in _char_boxes(unit):
            shifted = _shift(box, origin)
            color = COLORS["number"] if kind == "number" else COLORS["punct"]
            _draw_box(strategy_view, shifted, color, 1)
            _draw_label(strategy_view, label, shifted[0], max(11, shifted[1] - 2), color, 0.32)

    height = max(raw_view.shape[0], strategy_view.shape[0])
    raw_view = _pad_height(raw_view, height)
    strategy_view = _pad_height(strategy_view, height)
    gap = np.full((height, 12, 3), 255, dtype=np.uint8)
    compare = np.hstack([raw_view, gap, strategy_view])
    out_path = out_page / f"120180_b{int(route['block_idx']):03d}_r{int(route['route_idx']):03d}_strategy_overlay.png"
    cv2.imwrite(str(out_path), compare)
    return str(out_path.relative_to(ROOT))


def _write_markdown(routes: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 120180 Latin Strategy Overlay",
        "",
        "- Left: raw EngCut character boxes.",
        "- Right: proposed strategy result.",
        "- Green: exact English char boxes kept.",
        "- Magenta: English word fallback using Paddle/VL truth text.",
        "- Blue: digit char boxes.",
        "- Orange: punctuation char boxes.",
        "",
        "| block | route | exact word chars | word fallback | digits | punctuation | overlay |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for route in routes:
        units = route.get("units") or []
        exact_words = sum(1 for unit in units if unit.get("kind") == "word" and unit.get("exact_char_usable"))
        fallback_words = sum(1 for unit in units if unit.get("kind") == "word" and not unit.get("exact_char_usable"))
        digits = sum(1 for unit in units if unit.get("kind") == "number")
        punct = sum(1 for unit in units if unit.get("kind") == "punct")
        lines.append(
            f"| {route['block_idx']} | {route['route_idx']} | {exact_words} | {fallback_words} | "
            f"{digits} | {punct} | `{route['overlay']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = _load_json(INPUT)
    raw_payload = _load_json(Path(payload["source"]))
    image = cv2.imread(str(raw_payload["source_image"]))
    if image is None:
        raise RuntimeError(f"Cannot read image: {raw_payload['source_image']}")
    raw_by_route = {
        (int(route["block_idx"]), int(route["route_idx"])): route
        for route in raw_payload.get("results") or []
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_page = OUT_DIR / "120180"
    out_page.mkdir(parents=True, exist_ok=True)

    routes: list[dict[str, Any]] = []
    for route in sorted(payload.get("routes") or [], key=lambda item: (int(item["block_idx"]), int(item["route_idx"]))):
        raw_route = raw_by_route.get((int(route["block_idx"]), int(route["route_idx"])))
        copied = dict(route)
        copied["overlay"] = _render_route(image, copied, raw_route, out_page)
        routes.append(copied)

    _write_markdown(routes, OUT_DIR / "latin_strategy_overlay_120180.md")
    (OUT_DIR / "latin_strategy_overlay_120180.json").write_text(
        json.dumps({"routes": routes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUT_DIR / "latin_strategy_overlay_120180.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
