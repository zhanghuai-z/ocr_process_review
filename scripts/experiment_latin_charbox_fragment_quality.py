#!/usr/bin/env python3
"""Audit Latin native character boxes by hard crop boundaries cutting ink.

This offline experiment reuses the token ownership evidence captured by
``audit_italic_token_fallback``. It measures whether the vertical sides of each
native EngCut character crop cut through an owned connected component and
whether the crop contains secondary disconnected ink. It never changes routing,
OCR, Proof, or project state.
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


FRAGMENT_CHAR_RATIO_THRESHOLDS = (0.1, 0.2, 0.4, 0.6)
FRAGMENT_AREA_RATIO_THRESHOLDS = (0.005, 0.01, 0.02, 0.04)


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


def _vertical_cut_contacts(
    xs: np.ndarray,
    ys: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict[str, set[tuple[int, int]]]:
    """Return component pixels touching both sides of a crop's vertical edges."""
    x1, y1, x2, y2 = bbox
    points = {(int(x), int(y)) for x, y in zip(xs, ys)}
    contacts: dict[str, set[tuple[int, int]]] = {"left": set(), "right": set()}
    for side, inside_x, outside_x in (
        ("left", x1, x1 - 1),
        ("right", x2 - 1, x2),
    ):
        for y in range(y1, y2):
            inside = (inside_x, y)
            if inside not in points:
                continue
            for outside_y in range(y - 1, y + 2):
                outside = (outside_x, outside_y)
                if outside in points:
                    contacts[side].add(inside)
                    contacts[side].add(outside)
    return contacts


def _crop_fragments(
    points: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    char: str,
) -> tuple[list[tuple[int, int]], int]:
    """Find secondary connected ink inside one character crop."""
    x1, y1, x2, y2 = bbox
    if not points or x2 <= x1 or y2 <= y1:
        return [], 0
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for x, y in points:
        if x1 <= x < x2 and y1 <= y < y2:
            mask[y - y1, x - x1] = 1
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    components = [
        (index, tuple(int(value) for value in stats[index]))
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 2
    ]
    if len(components) <= 1:
        return [], len(components)
    main_index, main = max(components, key=lambda item: item[1][cv2.CC_STAT_AREA])
    main_left, main_top, main_width, main_height, _main_area = main
    accepted_dot: int | None = None
    if char.lower() in {"i", "j"}:
        candidates = []
        main_center = main_left + main_width / 2
        for index, (left, top, width, height, area) in components:
            if index == main_index:
                continue
            center = left + width / 2
            if top + height <= main_top + max(2, round(main_height * 0.2)):
                candidates.append((abs(center - main_center), -area, index))
        if candidates:
            accepted_dot = min(candidates)[2]
    fragments: list[tuple[int, int]] = []
    for index, _stats in components:
        if index in {main_index, accepted_dot}:
            continue
        ys, xs = np.where(labels == index)
        fragments.extend((int(x + x1), int(y + y1)) for y, x in zip(ys, xs))
    return fragments, len(components)


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
    cut_contact_pixels: set[tuple[int, int]] = set()
    cut_component_events = 0
    cut_sides_by_char: list[set[str]] = [set() for _box in boxes]
    reconstruction_missing = 0
    empty_char_ink = np.zeros(len(boxes), dtype=np.int64)
    ink_points_by_char: list[set[tuple[int, int]]] = [set() for _box in boxes]
    debug: dict[str, list[list[int]]] = {
        "cut": [],
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
        for index, covered in enumerate(coverage):
            ink_points_by_char[index].update(
                (int(x), int(y)) for x, y in zip(xs[covered], ys[covered])
            )
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

        for index, box in enumerate(boxes):
            if not hits[index]:
                continue
            contacts = _vertical_cut_contacts(xs, ys, box["bbox"])
            component_cut = False
            for side, pixels in contacts.items():
                if not pixels:
                    continue
                cut_sides_by_char[index].add(side)
                cut_contact_pixels.update(pixels)
                component_cut = True
            if component_cut:
                cut_component_events += 1

        if include_debug_pixels:
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
    cut_indices = [index for index, sides in enumerate(cut_sides_by_char) if sides]
    cut_side_count = sum(len(sides) for sides in cut_sides_by_char)
    fragment_pixels: set[tuple[int, int]] = set()
    fragment_indices: list[int] = []
    crop_component_counts: list[int] = []
    char_ink_total = 0
    fragment_area_total = 0
    for index, (box, points) in enumerate(zip(boxes, ink_points_by_char)):
        fragments, component_count = _crop_fragments(points, box["bbox"], box["text"])
        crop_component_counts.append(component_count)
        char_ink_total += len(points)
        fragment_area_total += len(fragments)
        if fragments:
            fragment_indices.append(index)
            fragment_pixels.update(fragments)
    if include_debug_pixels:
        debug["cut"] = [[x, y] for x, y in sorted(cut_contact_pixels)]
        debug["fragment"] = [[x, y] for x, y in sorted(fragment_pixels)]
    metrics = {
        "measurable": reconstruction_missing == 0,
        "reason": "measured" if reconstruction_missing == 0 else "component_reconstruction_missing",
        "foreground_area": total,
        "char_box_count": len(boxes),
        "empty_char_box_count": len(empty_indices),
        "empty_char_box_indices": empty_indices,
        "cut_char_count": len(cut_indices),
        "cut_char_indices": cut_indices,
        "cut_char_ratio": round(len(cut_indices) / len(boxes), 6),
        "cut_side_count": cut_side_count,
        "cut_component_events": cut_component_events,
        "cut_contact_pixel_count": len(cut_contact_pixels),
        "cut_contact_ratio": round(len(cut_contact_pixels) / total, 6),
        "fragment_char_count": len(fragment_indices),
        "fragment_char_indices": fragment_indices,
        "fragment_char_ratio": round(len(fragment_indices) / len(boxes), 6),
        "fragment_area": fragment_area_total,
        "fragment_area_ratio": round(fragment_area_total / max(1, char_ink_total), 6),
        "crop_component_counts": crop_component_counts,
        "multi_covered_ratio": round(multi / total, 6),
        "uncovered_ratio": round(uncovered / total, 6),
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
                "pp_bbox": token.get("pp_bbox"),
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


def _candidate(record: dict[str, Any], ratio: float, count: float) -> bool:
    metrics = record["fragment"]
    return bool(
        metrics.get("measurable")
        and float(metrics.get("fragment_char_ratio") or 0.0) >= ratio
        and float(metrics.get("fragment_area_ratio") or 0.0) >= count
    )


def _combined_candidate(record: dict[str, Any], ratio: float, count: float) -> bool:
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
    return bool(
        metrics.get("measurable")
        and float(metrics.get("fragment_char_ratio") or 0.0) >= 0.2
        and float(metrics.get("fragment_area_ratio") or 0.0) >= 0.01
    )


def _threshold_grid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ratio in FRAGMENT_CHAR_RATIO_THRESHOLDS:
        for count in FRAGMENT_AREA_RATIO_THRESHOLDS:
            selected = [record for record in records if _candidate(record, ratio, count)]
            combined = [
                record
                for record in records
                if _combined_candidate(record, ratio, count)
            ]
            result.append({
                "fragment_char_ratio": ratio,
                "fragment_area_ratio": count,
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
    cx1, cy1, cx2, cy2 = x1, y1, x2, y2
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
    if record.get("pp_bbox"):
        px1, py1, px2, py2 = _xyxy(record["pp_bbox"])
        for panel in (boxes, heat):
            cv2.rectangle(
                panel,
                (px1 - cx1, py1 - cy1),
                (px2 - cx1 - 1, py2 - cy1 - 1),
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    colors = {
        "cut": (255, 0, 255),
        "fragment": (0, 0, 255),
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
        f"cut_chars={metric.get('cut_char_count')}/{metric.get('char_box_count')} "
        f"fragments={metric.get('fragment_char_count')} area={metric.get('fragment_area_ratio')} "
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
            float(item["fragment"].get("fragment_char_ratio") or 0.0),
            float(item["fragment"].get("fragment_area_ratio") or 0.0),
        ),
        reverse=True,
    )
    json_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "schema": "latin_charbox_fragment_quality.v3",
        "supersedes": "v1 non-primary-component metric; v2 boundary-cut presence saturated on controls",
        "record_count": len(records),
        "exclusions": dict(exclusions),
        "threshold_grid": grid,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    known_sheet = output_dir / "known_target_control_fragments.png"
    ranked_sheet = output_dir / "highest_fragment_candidates.png"
    combined_sheet = output_dir / "right_slant_fragment_candidates.png"
    quality_sheet = output_dir / "fragment_quality_reference_candidates.png"
    t00031_sheet = output_dir / "T00031_pp_route_native_trace.png"
    _contact_sheet(labelled, known_sheet)
    _contact_sheet(ranked, ranked_sheet)
    combined_reference = [
        record
        for record in ranked
        if _combined_candidate(record, 0.2, 0.01)
    ]
    _contact_sheet(combined_reference, combined_sheet)
    quality_reference = [
        record for record in ranked if _quality_reference_candidate(record)
    ]
    _contact_sheet(quality_reference, quality_sheet)
    t00031_records = [
        record
        for record in records
        if record["source_name"] == "T00031_00.jpg"
        and record["text"] == "goubmieibsout"
    ]
    _contact_sheet(t00031_records, t00031_sheet, limit=1)

    lines = [
        "# Latin Character-box Fragment Quality Experiment",
        "",
        "Diagnostic only. No production or project state was modified.",
        "Red heat pixels are secondary connected fragments inside a native character crop; one plausible upper dot is exempted for i/j. Magenta pixels are same-component contacts crossing a crop's vertical side, orange pixels are covered by multiple native boxes, and blue pixels are owned ink uncovered by every native box. The crop is tight to route_bbox; a blue rectangle marks the original PP word bbox and red rectangles mark every native character bbox.",
        "",
        "## Scope",
        "",
        f"- records: {len(records)}",
        f"- measurable: {sum(item['fragment'].get('measurable') for item in records)}",
        f"- exclusions: `{dict(exclusions)}`",
        "",
        "## Known Test3 Cohort",
        "",
        "| label | text | native | fragment chars | fragment ratio | fragment area | cut ratio | right slope | improvement | fallback |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in labelled:
        metric = record["fragment"]
        lines.append(
            f"| {record['known_label']} | `{record['text']}` | `{record['native_text']}` | "
            f"{metric.get('fragment_char_count')} | {metric.get('fragment_char_ratio')} | "
            f"{metric.get('fragment_area_ratio')} | {metric.get('cut_char_ratio')} | "
            f"{record['slant'].get('slope')} | {record['slant'].get('score_improvement')} | "
            f"{str(record['current_word_fallback']).lower()} |"
        )
    lines.extend((
        "",
        "## Threshold Grid",
        "",
        "Fragment-only columns measure character crops containing secondary connected ink after exempting one plausible i/j dot. Combined columns additionally require right slope >= 0.12 and projection improvement >= 0.05. Boundary-cut presence remains reported but is not a selector because it saturates on controls.",
        "",
        "| fragment char ratio | fragment area ratio | fragment-only | targets | controls | combined | combined targets | combined controls | combined fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in grid:
        lines.append(
            f"| {row['fragment_char_ratio']} | {row['fragment_area_ratio']} | "
            f"{row['selected']} | {row['known_targets']} | {row['known_controls']} | "
            f"{row['combined_selected']} | {row['combined_known_targets']} | "
            f"{row['combined_known_controls']} | {row['combined_already_fallback']} |"
        )
    lines.extend((
        "",
        "## Quality Reference Conjunction",
        "",
        "For comparison only: select tokens where at least 20% of character crops contain secondary fragments and fragment pixels occupy at least 1% of character-crop ink. OCR text disagreement is reported separately and does not decide which OCR text is authoritative.",
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
        f"- T00031 PP/route/native trace: `{_windows_path(t00031_sheet)}`",
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
