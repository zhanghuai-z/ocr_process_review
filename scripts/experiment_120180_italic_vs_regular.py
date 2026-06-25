#!/usr/bin/env python3
"""Compare italic-like and regular-like Latin word geometry on 120180.

Offline experiment only. It uses Paddle/VL text truth + EngCut geometry from
the previous 120180 word fallback experiment, then estimates visual slant from
the word crop. The goal is to see whether style can decide when exact char
boxes are trustworthy.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = ROOT / "debug/latin_word_truth_fallback_120180/latin_word_truth_fallback_120180.json"
OUT_DIR = ROOT / "debug/italic_vs_regular_120180"

XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(box: XYXY, width: int, height: int) -> XYXY:
    return (
        max(0, min(width, box[0])),
        max(0, min(height, box[1])),
        max(0, min(width, box[2])),
        max(0, min(height, box[3])),
    )


def _pad(box: XYXY, pad: int, width: int, height: int) -> XYXY:
    return _clip((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), width, height)


def _word_slant_score(image: np.ndarray, bbox: XYXY) -> dict[str, float]:
    """Estimate italic slant by comparing upper/lower ink center drift.

    Positive values mean the top part of the glyphs is shifted to the right
    relative to the lower part. Italic Latin in this sample tends to produce a
    positive drift, but the measure is not a true font classifier.
    """
    height, width = image.shape[:2]
    left, top, right, bottom = _pad(bbox, 2, width, height)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return {"slant_px": 0.0, "slant_norm": 0.0, "ink_ratio": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # The page is high-contrast; Otsu is stable enough for this measurement.
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) < 8:
        return {"slant_px": 0.0, "slant_norm": 0.0, "ink_ratio": float(len(xs)) / max(1, binary.size)}

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    band_height = max(2, int((y_max - y_min + 1) * 0.35))
    upper_mask = ys <= y_min + band_height
    lower_mask = ys >= y_max - band_height
    if int(np.sum(upper_mask)) < 4 or int(np.sum(lower_mask)) < 4:
        return {"slant_px": 0.0, "slant_norm": 0.0, "ink_ratio": float(len(xs)) / max(1, binary.size)}

    upper_center = float(np.mean(xs[upper_mask]))
    lower_center = float(np.mean(xs[lower_mask]))
    slant_px = upper_center - lower_center
    ink_height = max(1.0, float(y_max - y_min + 1))
    return {
        "slant_px": slant_px,
        "slant_norm": slant_px / ink_height,
        "ink_ratio": float(len(xs)) / max(1, binary.size),
    }


def _box_from_unit(unit: dict[str, Any]) -> XYXY | None:
    bbox = unit.get("bbox") or []
    if len(bbox) != 4:
        return None
    box = tuple(int(v) for v in bbox)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _word_records(payload: dict[str, Any], image: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for route in payload.get("routes") or []:
        for unit in route.get("units") or []:
            if unit.get("kind") != "word":
                continue
            box = _box_from_unit(unit)
            if box is None:
                continue
            score = _word_slant_score(image, box)
            records.append(
                {
                    "block": route.get("block_idx"),
                    "route": route.get("route_idx"),
                    "text": unit.get("text"),
                    "engcut_text": unit.get("engcut_text"),
                    "source": unit.get("source"),
                    "exact_char_usable": bool(unit.get("exact_char_usable")),
                    "geometry_reasons": unit.get("geometry_reasons") or [],
                    "bbox": list(box),
                    **score,
                }
            )
    return records


def _bucket(records: list[dict[str, Any]], predicate) -> dict[str, Any]:
    selected = [record for record in records if predicate(record)]
    if not selected:
        return {"count": 0}
    exact = [record for record in selected if record["exact_char_usable"]]
    slants = [float(record["slant_norm"]) for record in selected]
    return {
        "count": len(selected),
        "exact_count": len(exact),
        "exact_ratio": round(len(exact) / len(selected), 4),
        "slant_norm_mean": round(mean(slants), 4),
        "slant_norm_median": round(median(slants), 4),
        "slant_norm_min": round(min(slants), 4),
        "slant_norm_max": round(max(slants), 4),
    }


def _render_contact_sheet(image: np.ndarray, records: list[dict[str, Any]], path: Path) -> None:
    height, width = image.shape[:2]
    rows: list[np.ndarray] = []
    for record in records:
        box = tuple(int(v) for v in record["bbox"])
        left, top, right, bottom = _pad(box, 8, width, height)
        crop = image[top:bottom, left:right].copy()
        scale = 2.0
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        status = "exact" if record["exact_char_usable"] else "word"
        label = f"{record['text']} | {status} | slant={record['slant_norm']:.2f}"
        label_h = 24
        canvas = np.full((crop.shape[0] + label_h, max(crop.shape[1], 560), 3), 255, dtype=np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        color = (0, 120, 0) if record["exact_char_usable"] else (170, 0, 170)
        cv2.putText(canvas, label[:90], (4, crop.shape[0] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        rows.append(canvas)
    if not rows:
        return
    gap = np.full((8, rows[0].shape[1], 3), 255, dtype=np.uint8)
    normalized = []
    max_w = max(row.shape[1] for row in rows)
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.hstack([row, pad])
        normalized.append(row)
    sheet = normalized[0]
    for row in normalized[1:]:
        sheet = np.vstack([sheet, gap, row])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def _write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    fallback = [record for record in records if not record["exact_char_usable"]]
    exact = [record for record in records if record["exact_char_usable"]]
    by_abs_slant = sorted(records, key=lambda record: abs(float(record["slant_norm"])), reverse=True)
    lines = [
        "# 120180 Italic vs Regular Experiment",
        "",
        "EngCut native output does not expose confidence. This experiment estimates visual slant from word crops and compares it with exact-char usability.",
        "",
        "## Summary",
        "",
        f"- word count: {len(records)}",
        f"- exact-char usable: {len(exact)}",
        f"- word fallback: {len(fallback)}",
        f"- exact contact sheet: `debug/italic_vs_regular_120180/exact_word_samples.png`",
        f"- fallback contact sheet: `debug/italic_vs_regular_120180/fallback_word_samples.png`",
        "",
        "## Buckets",
        "",
        "| bucket | count | exact | exact ratio | slant median | slant min..max |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    buckets = {
        "all": lambda _record: True,
        "exact-char usable": lambda record: record["exact_char_usable"],
        "word fallback": lambda record: not record["exact_char_usable"],
        "abs(slant) >= 0.12": lambda record: abs(float(record["slant_norm"])) >= 0.12,
        "abs(slant) < 0.12": lambda record: abs(float(record["slant_norm"])) < 0.12,
    }
    for name, predicate in buckets.items():
        value = _bucket(records, predicate)
        if not value.get("count"):
            lines.append(f"| {name} | 0 | 0 | - | - | - |")
            continue
        lines.append(
            f"| {name} | {value['count']} | {value['exact_count']} | {value['exact_ratio']} | "
            f"{value['slant_norm_median']} | {value['slant_norm_min']}..{value['slant_norm_max']} |"
        )
    lines.extend(
        [
            "",
            "## Highest Absolute Slant Samples",
            "",
            "| text | exact | slant | source | reasons |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for record in by_abs_slant[:30]:
        lines.append(
            f"| `{record['text']}` | {str(record['exact_char_usable']).lower()} | "
            f"{float(record['slant_norm']):.3f} | `{record['source']}` | "
            f"`{','.join(record['geometry_reasons'])}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = _load_json(INPUT)
    source_path = Path(_load_json(Path(payload["source"]))["source_image"])
    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"Cannot read source image: {source_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _word_records(payload, image)
    _write_json(OUT_DIR / "italic_vs_regular_120180.json", {"records": records})
    fallback_samples = [record for record in records if not record["exact_char_usable"]][:24]
    exact_samples = [record for record in records if record["exact_char_usable"]][:24]
    _render_contact_sheet(image, exact_samples, OUT_DIR / "exact_word_samples.png")
    _render_contact_sheet(image, fallback_samples, OUT_DIR / "fallback_word_samples.png")
    _write_markdown(records, OUT_DIR / "italic_vs_regular_120180.md")
    print(OUT_DIR / "italic_vs_regular_120180.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
