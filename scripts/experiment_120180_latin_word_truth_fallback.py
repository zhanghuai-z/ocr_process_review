#!/usr/bin/env python3
"""Build truth-text word boxes for 120180 pure Latin reference lines.

This is an offline experiment. It keeps Paddle/VL text as truth and uses
EngCut geometry only. English words are merged to word boxes; digits and
punctuation remain character-level units.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PURE_INPUT = ROOT / "debug/pure_english_linecut_vs_direct_engcut/120180/pure_english_linecut_vs_direct_engcut.json"
DISPATCH_INPUT = ROOT / "debug/latin_recovery_batch_prose_all_v2/120180/dispatch_units.json"
OUT_DIR = ROOT / "debug/latin_word_truth_fallback_120180"

XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _char_box(char: dict[str, Any]) -> XYXY | None:
    bbox = char.get("bbox") or []
    if len(bbox) != 4:
        return None
    box = tuple(int(v) for v in bbox)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _clip(box: XYXY, width: int, height: int) -> XYXY:
    return (
        max(0, min(box[0], width)),
        max(0, min(box[1], height)),
        max(0, min(box[2], width)),
        max(0, min(box[3], height)),
    )


def _shift(box: XYXY, origin: XYXY) -> XYXY:
    return box[0] - origin[0], box[1] - origin[1], box[2] - origin[0], box[3] - origin[1]


def _is_letter(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]$", text or ""))


def _is_digit(text: str) -> bool:
    return bool(re.match(r"^[0-9]$", text or ""))


def _is_alnum(text: str) -> bool:
    return _is_letter(text) or _is_digit(text)


def _group_engcut_chars(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for char in chars:
        if _char_box(char) is None:
            continue
        key = (int(char.get("line_index") or 0), int(char.get("group_index") or 0))
        grouped[key].append(char)
    return [
        sorted(values, key=lambda item: (_char_box(item) or (0, 0, 0, 0))[0])
        for _key, values in sorted(grouped.items())
    ]


def _punct_is_internal_noise(group: list[dict[str, Any]], index: int) -> bool:
    text = str(group[index].get("text") or "")
    if _is_alnum(text):
        return False
    prev_text = str(group[index - 1].get("text") or "") if index > 0 else ""
    next_text = str(group[index + 1].get("text") or "") if index + 1 < len(group) else ""
    return _is_letter(prev_text) and _is_letter(next_text)


def _engcut_units(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for group in _group_engcut_chars(chars):
        run: list[dict[str, Any]] = []
        for index, char in enumerate(group):
            text = str(char.get("text") or "")
            if _is_alnum(text) or _punct_is_internal_noise(group, index):
                run.append(char)
                continue
            if run:
                units.append(_make_engcut_unit(run))
                run = []
            units.append(_make_engcut_unit([char], forced_kind="punct"))
        if run:
            units.append(_make_engcut_unit(run))
    return units


def _make_engcut_unit(chars: list[dict[str, Any]], forced_kind: str = "") -> dict[str, Any]:
    boxes = [_char_box(char) for char in chars]
    valid = [box for box in boxes if box is not None]
    text = "".join(str(char.get("text") or "") for char in chars)
    if forced_kind:
        kind = forced_kind
    elif text.isdigit():
        kind = "number"
    elif any(_is_letter(ch) for ch in text):
        kind = "word"
    else:
        kind = "punct"
    return {
        "engcut_text": text,
        "kind": kind,
        "bbox": list(_union(valid)),
        "chars": [
            {
                "text": char.get("text"),
                "bbox": list(_char_box(char) or (0, 0, 0, 0)),
                "line_index": char.get("line_index"),
                "group_index": char.get("group_index"),
                "char_index": char.get("char_index"),
            }
            for char in chars
        ],
    }


def _normalize_truth_text(text: str) -> str:
    return (
        text.replace("〔", "[")
        .replace("〕", "]")
        .replace("，", ",")
        .replace("：", ":")
        .replace("。", ".")
    )


def _truth_units(text: str) -> list[dict[str, str]]:
    value = _normalize_truth_text(text)
    units: list[dict[str, str]] = []
    index = 0
    compact_index = 0
    while index < len(value):
        char = value[index]
        if char.isspace():
            index += 1
            continue
        if char.isascii() and char.isalpha():
            start = index
            while index < len(value) and value[index].isascii() and value[index].isalpha():
                index += 1
            text_value = value[start:index]
            units.append(
                {
                    "text": text_value,
                    "kind": "word",
                    "start": compact_index,
                    "end": compact_index + len(text_value),
                }
            )
            compact_index += len(text_value)
            continue
        if char.isascii() and char.isdigit():
            units.append({"text": char, "kind": "number", "start": compact_index, "end": compact_index + 1})
            compact_index += 1
            index += 1
            continue
        units.append({"text": char, "kind": "punct", "start": compact_index, "end": compact_index + 1})
        compact_index += 1
        index += 1
    return units


def _truth_units_by_block() -> dict[int, list[dict[str, str]]]:
    payload = _load_json(DISPATCH_INPUT)
    result: dict[int, list[dict[str, str]]] = {}
    for unit in payload.get("units") or []:
        block_idx = unit.get("block_idx")
        text = str(unit.get("text_truth") or "")
        if block_idx is None or not text:
            continue
        result[int(block_idx)] = _truth_units(text)
    return result


def _geometry_ok(unit: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    boxes = [tuple(char["bbox"]) for char in unit.get("chars") or []]
    if not boxes:
        return False, ["missing_char_boxes"]
    for left, right in zip(boxes, boxes[1:]):
        if right[0] < left[0]:
            reasons.append("non_monotonic_x")
            break
        overlap = max(0, left[2] - right[0])
        min_width = max(1, min(left[2] - left[0], right[2] - right[0]))
        # Italic Latin can make adjacent char boxes physically overlap even when
        # the text is correct. Once the overlap is visible, the char-level boxes
        # are no longer a trustworthy proof/edit target; keep the word geometry
        # instead of pretending the individual char boxes are exact.
        if overlap / min_width > 0.30:
            reasons.append("adjacent_char_overlap")
            break
    heights = [box[3] - box[1] for box in boxes]
    if heights and max(heights) > max(10, min(heights) * 2.4):
        reasons.append("height_outlier")
    return not reasons, reasons


def _normal_char(text: str) -> str:
    value = _normalize_truth_text(text or "")
    return value[:1]


def _truth_chars(units: list[dict[str, str]]) -> list[str]:
    return [char for unit in units for char in unit["text"]]


def _direct_chars_for_block(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for route in sorted(routes, key=lambda item: int(item["route_idx"])):
        route_idx = int(route["route_idx"])
        for source_char in route.get("direct_engcut", {}).get("chars") or []:
            box = _char_box(source_char)
            if box is None:
                continue
            text = _normal_char(str(source_char.get("text") or ""))
            if not text:
                continue
            chars.append(
                {
                    "text": text,
                    "bbox": list(box),
                    "route_idx": route_idx,
                    "line_index": source_char.get("line_index"),
                    "group_index": source_char.get("group_index"),
                    "char_index": source_char.get("char_index"),
                }
            )
    return chars


def _align_truth_to_engcut(truth_chars: list[str], eng_chars: list[dict[str, Any]]) -> list[int | None]:
    """Return EngCut char index for each truth char using edit-distance alignment."""
    eng_text = [str(char["text"]) for char in eng_chars]
    truth_len = len(truth_chars)
    eng_len = len(eng_text)
    dp = [[0] * (eng_len + 1) for _ in range(truth_len + 1)]
    for truth_index in range(1, truth_len + 1):
        dp[truth_index][0] = truth_index
    for eng_index in range(1, eng_len + 1):
        dp[0][eng_index] = eng_index
    for truth_index in range(1, truth_len + 1):
        truth_char = truth_chars[truth_index - 1]
        for eng_index in range(1, eng_len + 1):
            substitution_cost = 0 if truth_char == eng_text[eng_index - 1] else 1
            dp[truth_index][eng_index] = min(
                dp[truth_index - 1][eng_index] + 1,
                dp[truth_index][eng_index - 1] + 1,
                dp[truth_index - 1][eng_index - 1] + substitution_cost,
            )

    mapping: list[int | None] = [None] * truth_len
    truth_index = truth_len
    eng_index = eng_len
    while truth_index > 0 or eng_index > 0:
        if truth_index > 0 and eng_index > 0:
            substitution_cost = 0 if truth_chars[truth_index - 1] == eng_text[eng_index - 1] else 1
            if dp[truth_index][eng_index] == dp[truth_index - 1][eng_index - 1] + substitution_cost:
                mapping[truth_index - 1] = eng_index - 1
                truth_index -= 1
                eng_index -= 1
                continue
        if eng_index > 0 and dp[truth_index][eng_index] == dp[truth_index][eng_index - 1] + 1:
            eng_index -= 1
            continue
        truth_index -= 1
    return mapping


def _selected_engcut_indices(
    unit: dict[str, Any],
    mapping: list[int | None],
    eng_chars: list[dict[str, Any]],
) -> list[int]:
    start = int(unit["start"])
    end = int(unit["end"])
    direct = [mapped for mapped in mapping[start:end] if mapped is not None]
    if not direct:
        return []
    left = min(direct)
    right = max(direct)
    if unit["kind"] in {"word", "number"}:
        return list(range(left, right + 1))
    expected = unit["text"]
    exact = [index for index in direct if str(eng_chars[index]["text"]) == expected]
    return exact[:1] or [direct[0]]


def _fallback_route(unit_index: int, decorated: list[dict[str, Any]], routes: list[dict[str, Any]]) -> int:
    if decorated:
        return int(decorated[-1]["route_idx"])
    return int(sorted(routes, key=lambda item: int(item["route_idx"]))[0]["route_idx"])


def _decorate_block_routes(
    routes: list[dict[str, Any]],
    truth_units: list[dict[str, str]],
) -> dict[int, list[dict[str, Any]]]:
    eng_chars = _direct_chars_for_block(routes)
    mapping = _align_truth_to_engcut(_truth_chars(truth_units), eng_chars)
    decorated: list[dict[str, Any]] = []
    for unit_index, truth in enumerate(truth_units):
        selected_indices = _selected_engcut_indices(truth, mapping, eng_chars)
        selected_chars = [eng_chars[index] for index in selected_indices]
        if selected_chars:
            boxes = [tuple(int(v) for v in char["bbox"]) for char in selected_chars]
            route_votes: dict[int, int] = defaultdict(int)
            for char in selected_chars:
                route_votes[int(char["route_idx"])] += 1
            route_idx = max(route_votes.items(), key=lambda pair: (pair[1], -pair[0]))[0]
            bbox = list(_union(boxes))
            engcut_text = "".join(str(char["text"]) for char in selected_chars)
        else:
            route_idx = _fallback_route(unit_index, decorated, routes)
            bbox = []
            engcut_text = ""
        geometry_unit = {
            "chars": selected_chars,
            "bbox": bbox,
            "engcut_text": engcut_text,
        }
        geometry_ok, geometry_reasons = _geometry_ok(geometry_unit) if selected_chars else (False, ["missing_geometry"])
        exact_text = engcut_text == truth["text"]
        char_count_ok = len(selected_chars) == len(truth["text"])
        exact_char_usable = bool(selected_chars) and exact_text and char_count_ok and geometry_ok
        if truth["kind"] == "word":
            granularity = "word"
            source = "engcut_exact_word_chars" if exact_char_usable else "paddle_text+engcut_word_fallback"
        elif truth["kind"] == "number":
            granularity = "number"
            source = "engcut_exact_number_chars" if exact_char_usable else "paddle_text+engcut_number_fallback"
        else:
            granularity = "char"
            source = "engcut_punctuation" if exact_char_usable else "paddle_text+engcut_punctuation_fallback"
        decorated.append(
            {
                "text": truth["text"],
                "kind": truth["kind"],
                "bbox": bbox,
                "route_idx": route_idx,
                "granularity": granularity,
                "source": source,
                "exact_char_usable": exact_char_usable,
                "engcut_text": engcut_text,
                "char_count": len(selected_chars),
                "truth_char_count": len(truth["text"]),
                "geometry_reasons": geometry_reasons,
                "chars": selected_chars,
            }
        )
    units_by_route: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in decorated:
        units_by_route[int(unit["route_idx"])].append(unit)
    return units_by_route


def _draw_box(canvas: np.ndarray, box: XYXY, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text[:26], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _crop(image: np.ndarray, bbox: XYXY, pad: int = 18) -> tuple[np.ndarray, XYXY]:
    height, width = image.shape[:2]
    origin = _clip((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), width, height)
    return image[origin[1] : origin[3], origin[0] : origin[2]].copy(), origin


def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def _render(image: np.ndarray, item: dict[str, Any], units: list[dict[str, Any]], out_page: Path) -> str:
    route_bbox = tuple(int(v) for v in item["route_bbox"])
    crop, origin = _crop(image, route_bbox)
    raw_view = crop.copy()
    truth_view = crop.copy()
    _draw_box(raw_view, _shift(route_bbox, origin), (0, 170, 0), 2)
    _draw_box(truth_view, _shift(route_bbox, origin), (0, 170, 0), 2)

    for char in item.get("direct_engcut", {}).get("chars") or []:
        box = _char_box(char)
        if box is None:
            continue
        shifted = _shift(box, origin)
        _draw_box(raw_view, shifted, (255, 130, 0), 1)
        _draw_label(raw_view, str(char.get("text") or ""), shifted[0], max(12, shifted[1] - 2), (255, 130, 0))

    for unit in units:
        if not unit.get("bbox"):
            continue
        box = tuple(int(v) for v in unit["bbox"])
        shifted = _shift(box, origin)
        if unit["kind"] == "word":
            color = (0, 150, 0)
            thickness = 2
        elif unit["kind"] == "number":
            color = (255, 80, 0)
            thickness = 2
        else:
            color = (0, 140, 255)
            thickness = 1
        if not unit["exact_char_usable"] and unit["kind"] in {"word", "number"}:
            color = (180, 0, 180)
            thickness = 2
        _draw_box(truth_view, shifted, color, thickness)
        _draw_label(truth_view, unit["text"], shifted[0], max(12, shifted[1] - 2), color)

    height = max(raw_view.shape[0], truth_view.shape[0])
    raw_view = _pad_height(raw_view, height)
    truth_view = _pad_height(truth_view, height)
    gap = np.full((height, 12, 3), 255, dtype=np.uint8)
    compare = np.hstack([raw_view, gap, truth_view])
    out_path = out_page / f"120180_b{int(item['block_idx']):03d}_r{int(item['route_idx']):03d}_truth_word_fallback.png"
    cv2.imwrite(str(out_path), compare)
    return str(out_path.relative_to(ROOT))


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 120180 Latin Word Truth Fallback",
        "",
        "- Left side of overlays: raw EngCut char boxes.",
        "- Right side: Paddle/VL truth text on EngCut geometry.",
        "- Green = English word, blue = number, orange = punctuation, magenta = word fallback because exact char is not usable.",
        "",
        "## Exact Char Usable Rule",
        "",
        "A token is exact-char usable only when all conditions hold:",
        "",
        "1. EngCut text exactly equals the Paddle/VL truth token.",
        "2. Character bbox count equals truth character count.",
        "3. Bboxes are valid, x-monotonic, and do not have heavy overlap/outlier height.",
        "",
        "If any condition fails, English tokens are kept as one word box with Paddle/VL text.",
        "Digits and punctuation stay character-level unless their geometry is missing.",
        "",
        "| block | route | units | exact-char | fallback | text preview | overlay |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["routes"]:
        preview = " ".join(unit["text"] for unit in row["units"][:24])
        lines.append(
            f"| {row['block_idx']} | {row['route_idx']} | {len(row['units'])} | "
            f"{row['exact_char_count']} | {row['fallback_count']} | "
            f"`{preview[:120]}` | `{row['overlay']}` |"
        )
    lines.extend(["", "## Fallback Samples", ""])
    for row in payload["routes"]:
        samples = [
            f"{unit['engcut_text']} -> {unit['text']} ({','.join(unit['geometry_reasons']) or 'text_mismatch'})"
            for unit in row["units"]
            if str(unit["source"]).startswith("paddle_text+engcut")
        ]
        if samples:
            lines.append(f"- b{row['block_idx']:03d} r{row['route_idx']:03d}: " + "; ".join(samples))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    source = _load_json(PURE_INPUT)
    truth_by_block = _truth_units_by_block()
    image = cv2.imread(str(source["source_image"]))
    if image is None:
        raise RuntimeError(f"Cannot read image: {source['source_image']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_page = OUT_DIR / "120180"
    out_page.mkdir(parents=True, exist_ok=True)

    source_routes_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in source.get("results") or []:
        source_routes_by_block[int(item["block_idx"])].append(item)

    routes: list[dict[str, Any]] = []
    for block_idx in sorted(source_routes_by_block):
        block_routes = sorted(source_routes_by_block[block_idx], key=lambda value: int(value["route_idx"]))
        units_by_route = _decorate_block_routes(block_routes, truth_by_block.get(block_idx, []))
        for item in block_routes:
            route_idx = int(item["route_idx"])
            units = units_by_route.get(route_idx, [])
            overlay = _render(image, item, units, out_page)
            routes.append(
                {
                    "block_idx": block_idx,
                    "route_idx": route_idx,
                    "route_bbox": item["route_bbox"],
                    "ppocr_text": item.get("ppocr_text", ""),
                    "engcut_text": item.get("direct_engcut", {}).get("text") or "",
                    "truth_text_preview": " ".join(unit["text"] for unit in units),
                    "unit_count": len(units),
                    "exact_char_count": sum(1 for unit in units if unit["exact_char_usable"]),
                    "fallback_count": sum(1 for unit in units if str(unit["source"]).startswith("paddle_text+engcut")),
                    "units": units,
                    "overlay": overlay,
                }
            )

    payload = {
        "schema": "latin_word_truth_fallback_120180.v0",
        "source": str(PURE_INPUT),
        "truth_source": str(DISPATCH_INPUT),
        "routes": routes,
    }
    _write_json(OUT_DIR / "latin_word_truth_fallback_120180.json", payload)
    _write_markdown(payload, OUT_DIR / "latin_word_truth_fallback_120180.md")
    print(OUT_DIR / "latin_word_truth_fallback_120180.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
