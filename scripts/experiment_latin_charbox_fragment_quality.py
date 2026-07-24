#!/usr/bin/env python3
"""Audit Latin native character boxes by the foreground fragments they create.

This offline experiment reuses the token ownership evidence captured by
``audit_italic_token_fallback``. It measures how one owned connected component
is divided among native EngCut character boxes. It never changes routing, OCR,
Proof, or project state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_italic_token_fallback import _component_pixel_map  # noqa: E402
from scripts.experiment_latin_token_slant_gate import (  # noqa: E402
    KNOWN_TEST3_CONTROLS,
    KNOWN_TEST3_TARGETS,
    _slant_metrics,
)


FRAGMENT_RATIO_THRESHOLDS = (0.005, 0.01, 0.02, 0.04, 0.08)
FRAGMENT_COUNT_THRESHOLDS = (1, 2, 3)


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)


def _latin_letter_count(text: str) -> int:
    return sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _read_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def _known_label(token: dict[str, Any], dataset: str, run_index: int) -> str:
    if dataset != "test3" or run_index != 1 or not token.get("route_bbox"):
        return "unlabelled"
    key = (str(token.get("text") or ""), _xyxy(token["route_bbox"]))
    if key in KNOWN_TEST3_TARGETS:
        return "target_bad_geometry"
    if key in KNOWN_TEST3_CONTROLS:
        return "control_usable_geometry"
    return "unlabelled"


def _eligible(token: dict[str, Any]) -> tuple[bool, str]:
    if _latin_letter_count(str(token.get("text") or "")) < 2:
        return False, "fewer_than_two_latin_letters"
    if token.get("symbol_conflicts"):
        return False, "symbol_ownership_conflict"
    if not token.get("route_bbox"):
        return False, "missing_route_bbox"
    if not token.get("native_chars"):
        return False, "missing_native_chars"
    metrics = token.get("metrics") or {}
    if not metrics.get("owned_components"):
        return False, "missing_owned_foreground"
    return True, "eligible"


def _char_boxes(token: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"text": str(char.get("text") or ""), "bbox": _xyxy(char["bbox"])}
        for char in token.get("native_chars") or []
        if str(char.get("text") or "") and char.get("bbox") is not None
    ]


def _component_coverage(
    xs: np.ndarray,
    ys: np.ndarray,
    boxes: list[dict[str, Any]],
) -> np.ndarray:
    return np.array([
        (xs >= box["bbox"][0])
        & (xs < box["bbox"][2])
        & (ys >= box["bbox"][1])
        & (ys < box["bbox"][3])
        for box in boxes
    ], dtype=bool)


def _fragment_metrics(
    image: np.ndarray,
    token: dict[str, Any],
    *,
    include_debug_pixels: bool = False,
) -> tuple[dict[str, Any], dict[str, list[list[int]]]]:
    boxes = _char_boxes(token)
    if not boxes:
        return {"measurable": False, "reason": "no_nonempty_char_boxes"}, {}
    line_bbox = _xyxy(token.get("line_bbox") or token["route_bbox"])
    pixel_map = _component_pixel_map(image, line_bbox)
    components = (token.get("metrics") or {}).get("owned_components") or []

    total = 0
    uncovered = 0
    multi = 0
    foreign_only = 0
    primary_overflow = 0
    split_component_area = 0
    split_component_count = 0
    foreign_fragment_count = 0
    reconstruction_missing = 0
    empty_char_ink = np.zeros(len(boxes), dtype=np.int64)
    foreign_by_char = np.zeros(len(boxes), dtype=np.int64)
    debug: dict[str, list[list[int]]] = {
        "foreign": [],
        "multi": [],
        "uncovered": [],
    }

    for component in components:
        key = (_xyxy(component["bbox"]), int(component["area"]))
        points = pixel_map.get(key)
        if points is None:
            reconstruction_missing += 1
            continue
        ys, xs = points
        area = int(len(xs))
        if area == 0:
            continue
        total += area
        coverage = _component_coverage(xs, ys, boxes)
        hits = np.sum(coverage, axis=1)
        empty_char_ink += hits
        covered_count = np.sum(coverage, axis=0)
        uncovered_mask = covered_count == 0
        multi_mask = covered_count > 1
        uncovered += int(np.sum(uncovered_mask))
        multi += int(np.sum(multi_mask))
        if not np.any(hits):
            if include_debug_pixels:
                debug["uncovered"].extend(
                    [[int(x), int(y)] for x, y in zip(xs, ys)]
                )
            continue

        primary = int(np.argmax(hits))
        primary_mask = coverage[primary]
        secondary_mask = np.any(
            np.delete(coverage, primary, axis=0), axis=0
        ) if len(boxes) > 1 else np.zeros(area, dtype=bool)
        foreign_mask = secondary_mask & ~primary_mask
        foreign_pixels = int(np.sum(foreign_mask))
        foreign_only += foreign_pixels
        primary_overflow += int(np.sum(~primary_mask))

        significant = [
            index
            for index, hit in enumerate(hits)
            if int(hit) >= max(3, round(area * 0.05))
        ]
        if len(significant) >= 2:
            split_component_count += 1
            split_component_area += area
        for index in significant:
            if index == primary:
                continue
            count = int(np.sum(coverage[index] & ~primary_mask))
            if count <= 0:
                continue
            foreign_fragment_count += 1
            foreign_by_char[index] += count

        if include_debug_pixels:
            debug["foreign"].extend(
                [[int(x), int(y)] for x, y in zip(xs[foreign_mask], ys[foreign_mask])]
            )
            debug["multi"].extend(
                [[int(x), int(y)] for x, y in zip(xs[multi_mask], ys[multi_mask])]
            )
            debug["uncovered"].extend(
                [[int(x), int(y)] for x, y in zip(xs[uncovered_mask], ys[uncovered_mask])]
            )

    if total == 0:
        return {
            "measurable": False,
            "reason": "owned_component_pixels_not_reconstructed",
            "reconstruction_missing": reconstruction_missing,
        }, debug
    empty_indices = [
        index for index, ink in enumerate(empty_char_ink) if int(ink) == 0
    ]
    metrics = {
        "measurable": reconstruction_missing == 0,
        "reason": "measured" if reconstruction_missing == 0 else "component_reconstruction_missing",
        "foreground_area": total,
        "char_box_count": len(boxes),
        "empty_char_box_count": len(empty_indices),
        "empty_char_box_indices": empty_indices,
        "foreign_fragment_count": foreign_fragment_count,
        "foreign_fragment_char_count": int(np.sum(foreign_by_char > 0)),
        "foreign_fragment_ratio": round(foreign_only / total, 6),
        "primary_component_overflow_ratio": round(primary_overflow / total, 6),
        "multi_covered_ratio": round(multi / total, 6),
        "uncovered_ratio": round(uncovered / total, 6),
        "split_component_count": split_component_count,
        "split_component_area_ratio": round(split_component_area / total, 6),
        "reconstruction_missing": reconstruction_missing,
        "char_text": "".join(box["text"] for box in boxes),
    }
    return metrics, debug


def _pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    project = [
        page
        for page in payload.get("project_pages") or []
        if int(page.get("run_index") or 0) == 1
    ]
    return project + list(payload.get("batch_pages") or [])


def _records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    images: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for page in _pages(payload):
        if page.get("status") != "ok":
            exclusions["failed_page"] += 1
            continue
        source = str(page["source_image"])
        image = images.get(source)
        if image is None:
            image = _read_image(source)
            images[source] = image
            hashes[source] = hashlib.sha256(image.tobytes()).hexdigest()
        for token in page.get("tokens") or []:
            identity = (
                hashes[source],
                str(token.get("text") or ""),
                _xyxy(token.get("route_bbox") or token.get("pp_bbox") or (0, 0, 0, 0)),
            )
            if identity in seen:
                exclusions["duplicate_page_token"] += 1
                continue
            seen.add(identity)
            eligible, reason = _eligible(token)
            if not eligible:
                exclusions[reason] += 1
                continue
            metrics, _debug = _fragment_metrics(image, token)
            slant = _slant_metrics(image, token)
            if not metrics.get("measurable"):
                exclusions[str(metrics.get("reason") or "unmeasurable")] += 1
            records.append({
                "dataset": page["dataset"],
                "source_name": page["source_name"],
                "source_image": source,
                "run_index": int(page["run_index"]),
                "text": token["text"],
                "native_text": (token.get("metrics") or {}).get("native_text"),
                "route_bbox": token["route_bbox"],
                "line_bbox": token["line_bbox"],
                "native_chars": token["native_chars"],
                "owned_components": (token.get("metrics") or {}).get("owned_components") or [],
                "known_label": _known_label(
                    token, str(page["dataset"]), int(page["run_index"])
                ),
                "current_word_fallback": bool(
                    (token.get("current_result") or {}).get("word_fallback")
                ),
                "fragment": metrics,
                "slant": slant,
            })
    return records, exclusions


def _candidate(record: dict[str, Any], ratio: float, count: int) -> bool:
    metrics = record["fragment"]
    return bool(
        metrics.get("measurable")
        and float(metrics.get("foreign_fragment_ratio") or 0.0) >= ratio
        and int(metrics.get("foreign_fragment_count") or 0) >= count
    )


def _combined_candidate(record: dict[str, Any], ratio: float, count: int) -> bool:
    slant = record["slant"]
    return bool(
        _candidate(record, ratio, count)
        and slant.get("measurable")
        and float(slant.get("slope") or 0.0) >= 0.12
        and float(slant.get("score_improvement") or 0.0) >= 0.05
    )


def _quality_reference_candidate(record: dict[str, Any]) -> bool:
    """Diagnostic quality conjunction; it does not choose OCR text truth."""
    metrics = record["fragment"]
    ratio = float(metrics.get("foreign_fragment_ratio") or 0.0)
    count = int(metrics.get("foreign_fragment_count") or 0)
    text_disagreement = str(record.get("text") or "") != str(
        record.get("native_text") or ""
    )
    return bool(
        metrics.get("measurable")
        and (
            (ratio >= 0.02 and count >= 2)
            or (ratio >= 0.06 and count >= 1 and text_disagreement)
        )
    )


def _threshold_grid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ratio in FRAGMENT_RATIO_THRESHOLDS:
        for count in FRAGMENT_COUNT_THRESHOLDS:
            selected = [record for record in records if _candidate(record, ratio, count)]
            combined = [
                record
                for record in records
                if _combined_candidate(record, ratio, count)
            ]
            result.append({
                "foreign_fragment_ratio": ratio,
                "fragment_count": count,
                "selected": len(selected),
                "known_targets": sum(
                    item["known_label"] == "target_bad_geometry" for item in selected
                ),
                "known_controls": sum(
                    item["known_label"] == "control_usable_geometry" for item in selected
                ),
                "already_fallback": sum(item["current_word_fallback"] for item in selected),
                "combined_selected": len(combined),
                "combined_known_targets": sum(
                    item["known_label"] == "target_bad_geometry" for item in combined
                ),
                "combined_known_controls": sum(
                    item["known_label"] == "control_usable_geometry" for item in combined
                ),
                "combined_already_fallback": sum(
                    item["current_word_fallback"] for item in combined
                ),
            })
    return result


def _render_row(record: dict[str, Any]) -> np.ndarray:
    image = _read_image(record["source_image"])
    token = {
        "route_bbox": record["route_bbox"],
        "line_bbox": record["line_bbox"],
        "native_chars": record["native_chars"],
        "metrics": {"owned_components": record["owned_components"]},
    }
    _metrics, debug = _fragment_metrics(image, token, include_debug_pixels=True)
    x1, y1, x2, y2 = _xyxy(record["route_bbox"])
    height, width = image.shape[:2]
    margin_x = max(10, round((x2 - x1) * 0.08))
    margin_y = max(8, round((y2 - y1) * 0.30))
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(width, x2 + margin_x), min(height, y2 + margin_y)
    raw = image[cy1:cy2, cx1:cx2].copy()
    boxes = raw.copy()
    heat = raw.copy()
    for char in record["native_chars"]:
        if char.get("bbox") is None:
            continue
        bx1, by1, bx2, by2 = _xyxy(char["bbox"])
        cv2.rectangle(
            boxes,
            (bx1 - cx1, by1 - cy1),
            (bx2 - cx1 - 1, by2 - cy1 - 1),
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )
    colors = {
        "foreign": (0, 0, 255),
        "multi": (0, 165, 255),
        "uncovered": (255, 80, 0),
    }
    for kind, points in debug.items():
        color = colors[kind]
        for px, py in points:
            if cx1 <= px < cx2 and cy1 <= py < cy2:
                heat[py - cy1, px - cx1] = color
    panels = [raw, boxes, heat]
    scale = min(2.2, 660 / max(1, raw.shape[1]))
    panels = [
        cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for panel in panels
    ]
    panel_height = max(panel.shape[0] for panel in panels)
    panel_width = max(220, max(panel.shape[1] for panel in panels))
    normalized = []
    for panel in panels:
        canvas = np.full((panel_height, panel_width, 3), 255, np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        normalized.append(canvas)
    body = np.hstack(normalized)
    header = np.full((58, body.shape[1], 3), 255, np.uint8)
    metric = record["fragment"]
    label = (
        f"{record['source_name']} | {record['text']} -> {record['native_text']} | "
        f"label={record['known_label']} fallback={record['current_word_fallback']}"
    )
    values = (
        f"foreign={metric.get('foreign_fragment_ratio')} count={metric.get('foreign_fragment_count')} "
        f"multi={metric.get('multi_covered_ratio')} uncovered={metric.get('uncovered_ratio')}"
    )
    cv2.putText(header, label[:150], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, values, (6, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack((header, body))


def _contact_sheet(records: list[dict[str, Any]], path: Path, limit: int = 80) -> None:
    selected = records[:limit]
    if not selected:
        return
    rows = [_render_row(record) for record in selected]
    width = max(row.shape[1] for row in rows)
    normalized = []
    for row in rows:
        canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
        canvas[:, :row.shape[1]] = row
        normalized.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(normalized))[1].tofile(path)


def _write_report(
    output_dir: Path,
    records: list[dict[str, Any]],
    exclusions: Counter[str],
) -> None:
    grid = _threshold_grid(records)
    labelled = [record for record in records if record["known_label"] != "unlabelled"]
    ranked = sorted(
        records,
        key=lambda item: (
            float(item["fragment"].get("foreign_fragment_ratio") or 0.0),
            int(item["fragment"].get("foreign_fragment_count") or 0),
        ),
        reverse=True,
    )
    json_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "schema": "latin_charbox_fragment_quality.v1",
        "record_count": len(records),
        "exclusions": dict(exclusions),
        "threshold_grid": grid,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    known_sheet = output_dir / "known_target_control_fragments.png"
    ranked_sheet = output_dir / "highest_fragment_candidates.png"
    combined_sheet = output_dir / "right_slant_fragment_candidates.png"
    quality_sheet = output_dir / "fragment_quality_reference_candidates.png"
    _contact_sheet(labelled, known_sheet)
    _contact_sheet(ranked, ranked_sheet)
    combined_reference = [
        record
        for record in ranked
        if _combined_candidate(record, 0.02, 1)
    ]
    _contact_sheet(combined_reference, combined_sheet)
    quality_reference = [
        record for record in ranked if _quality_reference_candidate(record)
    ]
    _contact_sheet(quality_reference, quality_sheet)

    lines = [
        "# Latin Character-box Fragment Quality Experiment",
        "",
        "Diagnostic only. No production or project state was modified.",
        "Red heat pixels are owned component fragments lying only in a non-primary native character box; orange pixels are covered by multiple native boxes; blue pixels are owned ink uncovered by every native box.",
        "",
        "## Scope",
        "",
        f"- records: {len(records)}",
        f"- measurable: {sum(item['fragment'].get('measurable') for item in records)}",
        f"- exclusions: `{dict(exclusions)}`",
        "",
        "## Known Test3 Cohort",
        "",
        "| label | text | native | foreign ratio | fragments | right slope | improvement | fallback |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in labelled:
        metric = record["fragment"]
        lines.append(
            f"| {record['known_label']} | `{record['text']}` | `{record['native_text']}` | "
            f"{metric.get('foreign_fragment_ratio')} | {metric.get('foreign_fragment_count')} | "
            f"{record['slant'].get('slope')} | {record['slant'].get('score_improvement')} | "
            f"{str(record['current_word_fallback']).lower()} |"
        )
    lines.extend((
        "",
        "## Threshold Grid",
        "",
        "Fragment-only columns measure box damage. Combined columns additionally require right slope >= 0.12 and projection improvement >= 0.05.",
        "",
        "| foreign ratio | fragments | fragment-only | targets | controls | combined | combined targets | combined controls | combined fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in grid:
        lines.append(
            f"| {row['foreign_fragment_ratio']} | {row['fragment_count']} | "
            f"{row['selected']} | {row['known_targets']} | {row['known_controls']} | "
            f"{row['combined_selected']} | {row['combined_known_targets']} | "
            f"{row['combined_known_controls']} | {row['combined_already_fallback']} |"
        )
    lines.extend((
        "",
        "## Quality Reference Conjunction",
        "",
        "For comparison only: select at least two foreign fragments with ratio >= 0.02, or one fragment with ratio >= 0.06 plus PP/native text disagreement. Text disagreement is corroborating evidence of damaged boxes; it does not decide which OCR text is authoritative.",
        "",
        f"- selected: {len(quality_reference)}",
        f"- known targets: {sum(item['known_label'] == 'target_bad_geometry' for item in quality_reference)}",
        f"- known controls: {sum(item['known_label'] == 'control_usable_geometry' for item in quality_reference)}",
        f"- already fallback: {sum(item['current_word_fallback'] for item in quality_reference)}",
        "",
        "## Outputs",
        "",
        f"- JSON: `{_windows_path(json_path)}`",
        f"- Known cohort heatmap: `{_windows_path(known_sheet)}`",
        f"- Highest fragment candidates: `{_windows_path(ranked_sheet)}`",
        f"- Right-slope + fragment candidates: `{_windows_path(combined_sheet)}`",
        f"- Fragment quality reference candidates: `{_windows_path(quality_sheet)}`",
    ))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723/report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/latin_charbox_fragment_quality_20260724",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records, exclusions = _records(payload)
    _write_report(args.output_dir, records, exclusions)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
