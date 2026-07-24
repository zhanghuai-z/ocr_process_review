#!/usr/bin/env python3
"""Measure token-local Latin slant as a diagnostic word-fallback gate.

The input is the immutable JSON produced by ``audit_italic_token_fallback``.
Only uniquely routed Latin tokens without symbol ownership conflicts are
measured. The experiment writes debug evidence and never edits OCR/project
state or changes production routing.
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


SLOPE_GRID = np.linspace(-0.6, 0.6, 121)
SLOPE_THRESHOLDS = (0.08, 0.12, 0.16, 0.20)
IMPROVEMENT_THRESHOLDS = (0.01, 0.03, 0.05)
KNOWN_TEST3_TARGETS = {
    ("Unbalanced", (1249, 2388, 1448, 2421)),
    ("Incentives", (1207, 2535, 1369, 2566)),
    ("Finance", (1864, 2679, 2000, 2710)),
    ("Incentives", (302, 3051, 465, 3082)),
}
KNOWN_TEST3_CONTROLS = {
    ("Review", (843, 2464, 966, 2495)),
    ("Economic", (660, 2465, 826, 2496)),
    ("American", (480, 2467, 644, 2497)),
}


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)


def _latin_letter_count(text: str) -> int:
    return sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _eligible(token: dict[str, Any]) -> tuple[bool, str]:
    text = str(token.get("text") or "")
    if _latin_letter_count(text) < 2:
        return False, "fewer_than_two_latin_letters"
    if token.get("symbol_conflicts"):
        return False, "symbol_ownership_conflict"
    if not token.get("route_bbox"):
        return False, "missing_route_bbox"
    metrics = token.get("metrics") or {}
    if not metrics.get("owned_components"):
        return False, "missing_owned_foreground"
    if metrics.get("component_mask_reconstruction_missing"):
        return False, "incomplete_component_reconstruction"
    return True, "eligible"


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    threshold, _unused = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) <= threshold:
        return (gray > threshold).astype(np.uint8)
    return (gray <= threshold).astype(np.uint8)


def _owned_pixels(
    image: np.ndarray,
    token: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    route = _xyxy(token["route_bbox"])
    height, width = image.shape[:2]
    x1 = max(0, route[0] - 2)
    y1 = max(0, route[1] - 2)
    x2 = min(width, route[2] + 2)
    y2 = min(height, route[3] + 2)
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    binary = _foreground_mask(gray)
    owned = np.zeros_like(binary, dtype=np.uint8)

    components = (token.get("metrics") or {}).get("owned_components") or []
    total_component_area = sum(int(item.get("area") or 0) for item in components)
    ink_height = max(1, route[3] - route[1])
    accepted = 0
    rejected = 0
    for component in components:
        bbox = _xyxy(component["bbox"])
        component_height = bbox[3] - bbox[1]
        component_area = int(component.get("area") or 0)
        is_main_ink = (
            component_height >= ink_height * 0.35
            and component_area >= max(3, round(total_component_area * 0.005))
        )
        if not is_main_ink:
            rejected += 1
            continue
        cx1 = max(x1, bbox[0]) - x1
        cy1 = max(y1, bbox[1]) - y1
        cx2 = min(x2, bbox[2]) - x1
        cy2 = min(y2, bbox[3]) - y1
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        owned[cy1:cy2, cx1:cx2] |= binary[cy1:cy2, cx1:cx2]
        accepted += 1

    ys, xs = np.where(owned > 0)
    return ys.astype(np.float64), xs.astype(np.float64), {
        "accepted_components": accepted,
        "rejected_small_components": rejected,
        "pixel_count": int(len(xs)),
        "crop_bbox": [x1, y1, x2, y2],
    }


def _projection_score(xs: np.ndarray, rises: np.ndarray, slope: float) -> float:
    columns = np.rint(xs - slope * rises).astype(np.int32)
    columns -= int(columns.min())
    counts = np.bincount(columns)
    return float(np.dot(counts, counts)) / max(1.0, float(len(columns) ** 2))


def _slant_metrics(image: np.ndarray, token: dict[str, Any]) -> dict[str, Any]:
    ys, xs, support = _owned_pixels(image, token)
    if len(xs) < 24 or support["accepted_components"] == 0:
        return {
            **support,
            "measurable": False,
            "reason": "insufficient_main_ink",
        }
    baseline = float(np.max(ys))
    rises = baseline - ys
    if float(np.max(rises)) < 5.0:
        return {
            **support,
            "measurable": False,
            "reason": "insufficient_ink_height",
        }

    scores = np.array([
        _projection_score(xs, rises, float(slope)) for slope in SLOPE_GRID
    ])
    best_index = int(np.argmax(scores))
    best_slope = float(SLOPE_GRID[best_index])
    best_score = float(scores[best_index])
    zero_index = int(np.argmin(np.abs(SLOPE_GRID)))
    zero_score = float(scores[zero_index])
    improvement = (best_score - zero_score) / max(zero_score, 1e-12)
    outside_peak = np.abs(SLOPE_GRID - best_slope) >= 0.04
    runner_up = float(np.max(scores[outside_peak])) if np.any(outside_peak) else zero_score
    peak_margin = (best_score - runner_up) / max(best_score, 1e-12)
    boundary_peak = best_index in {0, len(SLOPE_GRID) - 1}
    return {
        **support,
        "measurable": not boundary_peak,
        "reason": "slope_peak_at_search_boundary" if boundary_peak else "measured",
        "slope": round(best_slope, 6),
        "abs_slope": round(abs(best_slope), 6),
        "projection_score": round(best_score, 8),
        "zero_slope_score": round(zero_score, 8),
        "score_improvement": round(improvement, 6),
        "peak_margin": round(peak_margin, 6),
    }


def _structural_conflict(token: dict[str, Any]) -> bool:
    metrics = token.get("metrics") or {}
    split = float(metrics.get("split_component_area_ratio") or 0.0)
    structural = bool(
        metrics.get("count_mismatch") or metrics.get("unexpected_internal_symbols")
    )
    return split >= 0.30 or (structural and split >= 0.10)


def _known_label(token: dict[str, Any], dataset: str, run_index: int) -> str:
    if dataset != "test3" or run_index != 1:
        return "unlabelled"
    key = (str(token.get("text") or ""), _xyxy(token.get("route_bbox") or (0, 0, 0, 0)))
    if key in KNOWN_TEST3_TARGETS:
        return "target_bad_geometry"
    if key in KNOWN_TEST3_CONTROLS:
        return "control_usable_geometry"
    return "unlabelled"


def _audit_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    project_run1 = [
        page
        for page in payload.get("project_pages") or []
        if page.get("run_index") == 1
    ]
    return project_run1 + list(payload.get("batch_pages") or [])


def _records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    image_cache: dict[str, np.ndarray] = {}
    image_hashes: dict[str, str] = {}
    seen_tokens: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for page in _audit_pages(payload):
        if page.get("status") != "ok":
            exclusions["failed_page"] += 1
            continue
        source_image = str(page["source_image"])
        image = image_cache.get(source_image)
        if image is None:
            image = cv2.imdecode(
                np.fromfile(source_image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"cannot read source image: {source_image}")
            image_cache[source_image] = image
        image_hash = image_hashes.get(source_image)
        if image_hash is None:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            image_hashes[source_image] = image_hash
        for token in page.get("tokens") or []:
            identity = (
                image_hash,
                str(token.get("text") or ""),
                _xyxy(token.get("route_bbox") or token.get("pp_bbox") or (0, 0, 0, 0)),
            )
            if identity in seen_tokens:
                exclusions["duplicate_page_token"] += 1
                continue
            seen_tokens.add(identity)
            eligible, reason = _eligible(token)
            if not eligible:
                exclusions[reason] += 1
                continue
            slant = _slant_metrics(image, token)
            if not slant.get("measurable"):
                exclusions[str(slant.get("reason") or "unmeasurable")] += 1
            records.append({
                "dataset": page["dataset"],
                "source_name": page["source_name"],
                "source_image": source_image,
                "run_index": int(page["run_index"]),
                "text": token["text"],
                "route_bbox": token["route_bbox"],
                "native_text": (token.get("metrics") or {}).get("native_text"),
                "known_label": _known_label(
                    token, str(page["dataset"]), int(page["run_index"])
                ),
                "current_word_fallback": bool(
                    (token.get("current_result") or {}).get("word_fallback")
                ),
                "structural_conflict": _structural_conflict(token),
                "split_component_area_ratio": (
                    token.get("metrics") or {}
                ).get("split_component_area_ratio"),
                "slant": slant,
            })
    return records, exclusions


def _candidate(
    record: dict[str, Any],
    slope: float,
    improvement: float,
    *,
    direction: str,
) -> bool:
    slant = record["slant"]
    observed_slope = float(slant.get("slope") or 0.0)
    slope_passes = (
        observed_slope >= slope
        if direction == "right"
        else abs(observed_slope) >= slope
    )
    return bool(
        slant.get("measurable")
        and record["structural_conflict"]
        and slope_passes
        and float(slant.get("score_improvement") or 0.0) >= improvement
    )


def _threshold_grid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in ("absolute", "right"):
        for slope in SLOPE_THRESHOLDS:
            for improvement in IMPROVEMENT_THRESHOLDS:
                selected = [
                    record
                    for record in records
                    if _candidate(
                        record,
                        slope,
                        improvement,
                        direction=direction,
                    )
                ]
                rows.append({
                    "direction": direction,
                    "slope_threshold": slope,
                    "score_improvement": improvement,
                    "selected": len(selected),
                    "known_targets": sum(
                        record["known_label"] == "target_bad_geometry"
                        for record in selected
                    ),
                    "known_controls": sum(
                        record["known_label"] == "control_usable_geometry"
                        for record in selected
                    ),
                    "current_word_fallbacks": sum(
                        record["current_word_fallback"] for record in selected
                    ),
                })
    return rows


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    selected = sorted(
        records,
        key=lambda item: (
            item["known_label"] != "unlabelled",
            item["structural_conflict"],
            float(item["slant"].get("score_improvement") or 0.0),
        ),
        reverse=True,
    )[:80]
    image_cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for record in selected:
        image = image_cache.get(record["source_image"])
        if image is None:
            image = cv2.imdecode(
                np.fromfile(record["source_image"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            image_cache[record["source_image"]] = image
        x1, y1, x2, y2 = _xyxy(record["route_bbox"])
        height, width = image.shape[:2]
        crop = image[
            max(0, y1 - 8):min(height, y2 + 8),
            max(0, x1 - 12):min(width, x2 + 12),
        ]
        scale = min(2.0, 720 / max(1, crop.shape[1]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        canvas = np.full((crop.shape[0] + 48, max(760, crop.shape[1]), 3), 255, np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        slant = record["slant"]
        label = (
            f"{record['source_name']} | {record['text']} -> {record['native_text']} | "
            f"label={record['known_label']} structural={record['structural_conflict']}"
        )
        metric = (
            f"slope={slant.get('slope')} abs={slant.get('abs_slope')} "
            f"improve={slant.get('score_improvement')} peak={slant.get('peak_margin')}"
        )
        cv2.putText(canvas, label[:130], (6, crop.shape[0] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, metric, (6, crop.shape[0] + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
        rows.append(canvas)
    if not rows:
        return
    max_width = max(row.shape[1] for row in rows)
    normalized = []
    for row in rows:
        canvas = np.full((row.shape[0], max_width, 3), 255, np.uint8)
        canvas[:, :row.shape[1]] = row
        normalized.append(canvas)
    sheet = np.vstack(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", sheet)[1].tofile(path)


def _write_report(
    output_dir: Path,
    records: list[dict[str, Any]],
    exclusions: Counter[str],
) -> None:
    grid = _threshold_grid(records)
    measurable = [record for record in records if record["slant"].get("measurable")]
    labelled = [record for record in records if record["known_label"] != "unlabelled"]
    payload = {
        "schema": "latin_token_slant_gate_experiment.v1",
        "scope": (
            "diagnostic-only uniquely routed Latin tokens without symbol conflicts"
        ),
        "record_count": len(records),
        "measurable_count": len(measurable),
        "exclusions": dict(exclusions),
        "threshold_grid": grid,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sheet_path = output_dir / "latin_slant_contact_sheet.png"
    _contact_sheet(records, sheet_path)
    right_candidates = [
        record
        for record in records
        if _candidate(record, 0.12, 0.05, direction="right")
    ]
    right_sheet_path = output_dir / "right_slant_structural_candidates.png"
    _contact_sheet(right_candidates, right_sheet_path)
    direction_rejections = [
        record
        for record in records
        if _candidate(record, 0.12, 0.05, direction="absolute")
        and not _candidate(record, 0.12, 0.05, direction="right")
    ]
    rejected_sheet_path = output_dir / "left_slant_direction_rejections.png"
    _contact_sheet(direction_rejections, rejected_sheet_path)
    added_right_candidates = [
        record for record in right_candidates if not record["current_word_fallback"]
    ]

    lines = [
        "# Latin Token Slant Gate Experiment",
        "",
        "Diagnostic only. No OCR, Proof, layout, or project state was modified.",
        "Slant is evaluated only together with an existing foreground component-split conflict.",
        "",
        "## Scope",
        "",
        f"- eligible Latin token records: {len(records)}",
        f"- measurable token records: {len(measurable)}",
        f"- known test3 targets: {sum(item['known_label'] == 'target_bad_geometry' for item in labelled)}",
        f"- known test3 controls: {sum(item['known_label'] == 'control_usable_geometry' for item in labelled)}",
        f"- exclusions: `{dict(exclusions)}`",
        "",
        "## Known Test3 Cohort",
        "",
        "| label | text | native | structural | slope | improvement | current fallback |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in labelled:
        slant = record["slant"]
        lines.append(
            f"| {record['known_label']} | `{record['text']}` | `{record['native_text']}` | "
            f"{str(record['structural_conflict']).lower()} | {slant.get('slope')} | "
            f"{slant.get('score_improvement')} | {str(record['current_word_fallback']).lower()} |"
        )
    lines.extend((
        "",
        "## Threshold Grid",
        "",
        "A row selects a token only when component-split conflict, the stated slope direction, and score improvement all pass.",
        "",
        "| direction | slope | improvement | selected | known targets | known controls | already fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in grid:
        lines.append(
            f"| {row['direction']} | {row['slope_threshold']} | "
            f"{row['score_improvement']} | {row['selected']} | "
            f"{row['known_targets']} | {row['known_controls']} | {row['current_word_fallbacks']} |"
        )
    lines.extend((
        "",
        "## Observations",
        "",
        f"- right-slope 0.12 / improvement 0.05 selects {len(right_candidates)} tokens; {sum(item['current_word_fallback'] for item in right_candidates)} already use word fallback and {len(added_right_candidates)} are additional candidates.",
        f"- additional candidate source pages: `{dict(Counter(item['source_name'] for item in added_right_candidates))}`",
        f"- direction gate rejects {len(direction_rejections)} absolute-slope candidates.",
        "- Rasterized slope peaks are strongly quantized in this dataset, so the tested slope thresholds do not yet establish a production cutoff.",
        "- The full batch has no human-labelled italic/regular truth; candidate counts and contact sheets are evidence, not accuracy measurements.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{_windows_path(json_path)}`",
        f"- General contact sheet: `{_windows_path(sheet_path)}`",
        f"- Right-slope structural candidates: `{_windows_path(right_sheet_path)}`",
        f"- Direction-gate rejections: `{_windows_path(rejected_sheet_path)}`",
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
        default=ROOT / "debug/latin_token_slant_gate_20260724",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records, exclusions = _records(payload)
    _write_report(args.output_dir, records, exclusions)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
