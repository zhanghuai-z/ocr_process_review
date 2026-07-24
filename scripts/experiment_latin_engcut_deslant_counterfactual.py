#!/usr/bin/env python3
"""Compare EngCut on original and counter-sheared token-isolated Latin ink.

This is a diagnostic experiment. It reads the immutable token ownership audit,
copies only one uniquely owned Latin route into a white canvas, and runs EngCut
on both the unchanged canvas and a counter-sheared copy. Corrected coordinates
remain experiment-local and are never written to OCR, layout, Proof, or project
state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engines.hanwang import native_bridge  # noqa: E402
from app.engines.hanwang.engcut_payload import EngcutChar, engcut_chars_from_payload  # noqa: E402
from scripts.experiment_latin_token_slant_gate import (  # noqa: E402
    _eligible,
    _known_label,
    _slant_metrics,
    _structural_conflict,
)


MIN_SLOPE = 0.12
MIN_PROJECTION_IMPROVEMENT = 0.05
SHEAR_FACTORS = (0.25, 0.5, 0.75, 1.0)


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)


def _read_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read source image: {path}")
    return image


def _token_isolated_canvas(
    image: np.ndarray,
    route_bbox: tuple[int, int, int, int],
    slope: float,
    *,
    outer_pad: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Copy only route-owned source pixels with equal guard space for both arms."""
    x1, y1, x2, y2 = route_bbox
    height, width = image.shape[:2]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid route bbox: {route_bbox}")
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    shift_guard = int(math.ceil(abs(slope) * max(1, y2 - y1))) + 2
    pad_x = outer_pad + shift_guard
    canvas = np.full(
        (y2 - y1 + 2 * outer_pad, x2 - x1 + 2 * pad_x, 3),
        255,
        dtype=image.dtype,
    )
    local_bbox = (pad_x, outer_pad, pad_x + x2 - x1, outer_pad + y2 - y1)
    lx1, ly1, lx2, ly2 = local_bbox
    canvas[ly1:ly2, lx1:lx2] = image[y1:y2, x1:x2]
    return canvas, local_bbox


def _counter_shear(
    canvas: np.ndarray,
    local_bbox: tuple[int, int, int, int],
    slope: float,
) -> np.ndarray:
    """Move upper ink left by the measured right-slant displacement."""
    _x1, _y1, _x2, baseline = local_bbox
    matrix = np.array(
        [[1.0, float(slope), -float(slope) * baseline], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        canvas,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _visible_groups(chars: list[EngcutChar]) -> list[list[EngcutChar]]:
    groups: dict[tuple[int, int], list[EngcutChar]] = {}
    order: list[tuple[int, int]] = []
    for char in chars:
        if not str(char.text or ""):
            continue
        key = char.line_index, char.group_index
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(char)
    return [groups[key] for key in order]


def _engcut_metrics(chars: list[EngcutChar]) -> dict[str, Any]:
    groups = _visible_groups(chars)
    boxes = [char.bbox for group in groups for char in group if char.bbox is not None]
    overlap_pairs = 0
    nonmonotonic_centers = 0
    for group in groups:
        visible = [char for char in group if char.bbox is not None]
        for left, right in zip(visible, visible[1:]):
            assert left.bbox is not None and right.bbox is not None
            if right.bbox[0] < left.bbox[2]:
                overlap_pairs += 1
            left_center = (left.bbox[0] + left.bbox[2]) / 2
            right_center = (right.bbox[0] + right.bbox[2]) / 2
            if right_center <= left_center:
                nonmonotonic_centers += 1
    return {
        "text": " ".join("".join(char.text for char in group) for group in groups),
        "char_count": sum(len(group) for group in groups),
        "group_count": len(groups),
        "bbox_count": len(boxes),
        "missing_bbox_count": sum(
            char.bbox is None for group in groups for char in group
        ),
        "overlap_pair_count": overlap_pairs,
        "nonmonotonic_center_count": nonmonotonic_centers,
        "chars": [
            {"text": char.text, "bbox": list(char.bbox) if char.bbox else None}
            for group in groups
            for char in group
        ],
    }


def _run_arm(
    canvas: np.ndarray,
    *,
    timeout: float,
    runner: Callable[..., dict[str, Any]] = native_bridge.run_eng20_recogline,
) -> dict[str, Any]:
    payload = runner(canvas, timeout=timeout)
    return _engcut_metrics(engcut_chars_from_payload(payload))


def _selection_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record["known_label"] != "unlabelled":
        reasons.append("known_test3")
    if record["current_word_fallback"]:
        reasons.append("current_word_fallback")
    slant = record["slant"]
    right_slant = bool(
        slant.get("measurable")
        and float(slant.get("slope") or 0.0) >= MIN_SLOPE
        and float(slant.get("score_improvement") or 0.0) >= MIN_PROJECTION_IMPROVEMENT
    )
    if right_slant and record["structural_conflict"]:
        reasons.append("right_slant_structural")
    elif right_slant:
        reasons.append("right_slant_clean_control")
    return reasons


def _records(
    payload: dict[str, Any],
    *,
    dataset: str | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pages = [
        page
        for page in payload.get("project_pages") or []
        if int(page.get("run_index") or 0) == 1
    ] + list(payload.get("batch_pages") or [])
    exclusions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for page in pages:
        if page.get("status") != "ok":
            exclusions["failed_page"] += 1
            continue
        if dataset and str(page.get("dataset")) != dataset:
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
            slant = _slant_metrics(image, token)
            record = {
                "dataset": str(page["dataset"]),
                "source_name": str(page["source_name"]),
                "source_image": source,
                "run_index": int(page["run_index"]),
                "text": str(token["text"]),
                "route_bbox": list(token["route_bbox"]),
                "known_label": _known_label(
                    token, str(page["dataset"]), int(page["run_index"])
                ),
                "current_word_fallback": bool(
                    (token.get("current_result") or {}).get("word_fallback")
                ),
                "structural_conflict": _structural_conflict(token),
                "slant": slant,
            }
            reasons = _selection_reasons(record)
            if not reasons:
                exclusions["outside_targeted_cohort"] += 1
                continue
            record["selection_reasons"] = reasons
            records.append(record)
    records.sort(
        key=lambda item: (
            item["known_label"] == "unlabelled",
            not item["structural_conflict"],
            -float(item["slant"].get("score_improvement") or 0.0),
            item["source_name"],
            item["route_bbox"],
        )
    )
    return records, exclusions


def _evaluate(
    record: dict[str, Any],
    image: np.ndarray,
    timeout: float,
    *,
    factors: tuple[float, ...] = SHEAR_FACTORS,
) -> dict[str, Any]:
    slope = float(record["slant"]["slope"])
    original, local_bbox = _token_isolated_canvas(
        image, _xyxy(record["route_bbox"]), slope
    )
    original_result = _run_arm(original, timeout=timeout)
    arms: list[dict[str, Any]] = []
    rendered_images: list[tuple[str, np.ndarray]] = [("ORIGINAL", original)]
    for factor in factors:
        corrected = _counter_shear(original, local_bbox, slope * factor)
        result = _run_arm(corrected, timeout=timeout)
        arms.append({"factor": factor, "applied_slope": slope * factor, "result": result})
        rendered_images.append((f"SHEAR {factor:.2f}", corrected))
    corrected_result = arms[-1]["result"]
    stable_arms = [
        arm for arm in arms if arm["result"]["text"] == original_result["text"]
    ]
    stable_overlap_improvements = [
        arm
        for arm in stable_arms
        if arm["result"]["overlap_pair_count"]
        < original_result["overlap_pair_count"]
        and arm["result"]["nonmonotonic_center_count"]
        <= original_result["nonmonotonic_center_count"]
    ]
    comparison = {
        "text_stable": original_result["text"] == corrected_result["text"],
        "original_matches_pp": original_result["text"] == record["text"],
        "corrected_matches_pp": corrected_result["text"] == record["text"],
        "char_count_delta": corrected_result["char_count"] - original_result["char_count"],
        "overlap_pair_delta": (
            corrected_result["overlap_pair_count"]
            - original_result["overlap_pair_count"]
        ),
        "overlap_resolved": bool(
            original_result["overlap_pair_count"] > 0
            and corrected_result["overlap_pair_count"] == 0
        ),
        "geometry_regressed": bool(
            corrected_result["overlap_pair_count"] > original_result["overlap_pair_count"]
            or corrected_result["nonmonotonic_center_count"]
            > original_result["nonmonotonic_center_count"]
        ),
        "stable_arm_count": len(stable_arms),
        "stable_geometry_improvement_factors": [
            arm["factor"] for arm in stable_overlap_improvements
        ],
        "native_text_variant_count": len({
            original_result["text"],
            *(arm["result"]["text"] for arm in arms),
        }),
    }
    return {
        **record,
        "experiment": {
            "local_route_bbox": list(local_bbox),
            "canvas_shape": list(original.shape),
            "counter_shear_slope": slope,
        },
        "original": original_result,
        "arms": arms,
        "corrected": corrected_result,
        "comparison": comparison,
        "_images": rendered_images,
    }


def _render_record(record: dict[str, Any], path: Path) -> None:
    panels: list[np.ndarray] = []
    results = [record["original"], *(arm["result"] for arm in record["arms"])]
    for index, ((label, image), result) in enumerate(zip(record["_images"], results)):
        color = (0, 0, 220) if index == 0 else (0, 140, 0)
        panel = image.copy()
        for char in result["chars"]:
            if char["bbox"] is None:
                continue
            x1, y1, x2, y2 = _xyxy(char["bbox"])
            cv2.rectangle(panel, (x1, y1), (x2 - 1, y2 - 1), color, 1, cv2.LINE_AA)
        scale = min(2.2, 420 / max(1, panel.shape[1]))
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        header = np.full((52, panel.shape[1], 3), 255, np.uint8)
        cv2.putText(header, f"{label}: {result['text']}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(
            header,
            f"chars={result['char_count']} overlap={result['overlap_pair_count']} nonmono={result['nonmonotonic_center_count']}",
            (5, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA,
        )
        panels.append(np.vstack((header, panel)))
    height = max(panel.shape[0] for panel in panels)
    normalized = []
    for panel in panels:
        canvas = np.full((height, panel.shape[1], 3), 255, np.uint8)
        canvas[:panel.shape[0]] = panel
        normalized.append(canvas)
    body = np.hstack(normalized)
    title = np.full((64, body.shape[1], 3), 255, np.uint8)
    cv2.putText(
        title,
        f"{record['source_name']} | PP={record['text']} | label={record['known_label']} | slope={record['slant'].get('slope')}",
        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
    )
    cv2.putText(
        title,
        f"reasons={','.join(record['selection_reasons'])} stable={record['comparison']['text_stable']} resolved={record['comparison']['overlap_resolved']}",
        (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack((title, body)))[1].tofile(path)


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _write_report(
    output_dir: Path,
    records: list[dict[str, Any]],
    exclusions: Counter[str],
    *,
    factors: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "comparisons"
    serializable: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_path = image_dir / f"{index + 1:04d}_{record['source_name']}_{record['text']}.png"
        safe_path = image_path.with_name("".join(char if char.isalnum() or char in "._-" else "_" for char in image_path.name))
        _render_record(record, safe_path)
        clean = {key: value for key, value in record.items() if key != "_images"}
        clean["comparison_image"] = _windows_path(safe_path)
        serializable.append(clean)
    summary = {
        "records": len(records),
        "text_stable": sum(item["comparison"]["text_stable"] for item in records),
        "overlap_resolved": sum(item["comparison"]["overlap_resolved"] for item in records),
        "geometry_regressed": sum(item["comparison"]["geometry_regressed"] for item in records),
        "original_matches_pp": sum(item["comparison"]["original_matches_pp"] for item in records),
        "corrected_matches_pp": sum(item["comparison"]["corrected_matches_pp"] for item in records),
        "has_stable_geometry_improvement": sum(
            bool(item["comparison"]["stable_geometry_improvement_factors"])
            for item in records
        ),
    }
    cohort_summary: list[dict[str, Any]] = []
    for reason in ("right_slant_structural", "right_slant_clean_control"):
        cohort = [item for item in records if reason in item["selection_reasons"]]
        for dataset in ["all", *sorted({item["dataset"] for item in cohort})]:
            subset = (
                cohort
                if dataset == "all"
                else [item for item in cohort if item["dataset"] == dataset]
            )
            if not subset:
                continue
            cohort_summary.append({
                "reason": reason,
                "dataset": dataset,
                "records": len(subset),
                "text_changed": sum(
                    item["arms"][0]["result"]["text"] != item["original"]["text"]
                    for item in subset
                ),
                "overlap_improved": sum(
                    item["arms"][0]["result"]["overlap_pair_count"]
                    < item["original"]["overlap_pair_count"]
                    for item in subset
                ),
                "overlap_regressed": sum(
                    item["arms"][0]["result"]["overlap_pair_count"]
                    > item["original"]["overlap_pair_count"]
                    for item in subset
                ),
            })
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps({
        "schema": "latin_engcut_deslant_counterfactual.v1",
        "scope": "diagnostic-only token-isolated uniquely owned Latin routes",
        "limitations": [
            "The canvas contains one token rather than the complete production masked physical line.",
            "Counter-sheared boxes remain in experiment coordinates and are not project geometry.",
            "PP text agreement is comparison evidence and does not select OCR truth.",
        ],
        "thresholds": {
            "right_slope": MIN_SLOPE,
            "projection_improvement": MIN_PROJECTION_IMPROVEMENT,
            "shear_factors": list(factors),
        },
        "summary": summary,
        "cohort_summary": cohort_summary,
        "exclusions": dict(exclusions),
        "records": serializable,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Latin EngCut De-slant Counterfactual",
        "",
        "Diagnostic only. No OCR, layout, Proof, export, or project state was modified.",
        "",
        "## Boundary",
        "",
        "- Each arm receives the same token-isolated white canvas and only pixels from the uniquely owned route bbox.",
        "- The corrected arm counter-shears those pixels using the measured token-local right slope.",
        "- Corrected character boxes are experiment-local; PP text agreement is reported but never chooses text truth.",
        "- This first experiment does not reproduce other Latin tokens from the complete production masked physical line.",
        "",
        "## Summary",
        "",
        f"- records: {summary['records']}",
        f"- text stable: {summary['text_stable']}",
        f"- original/corrected match PP: {summary['original_matches_pp']} / {summary['corrected_matches_pp']}",
        f"- native overlap fully resolved: {summary['overlap_resolved']}",
        f"- native geometry regressed: {summary['geometry_regressed']}",
        f"- has a text-stable intermediate geometry improvement: {summary['has_stable_geometry_improvement']}",
        f"- exclusions: `{dict(exclusions)}`",
        "",
        "## Cohorts",
        "",
        "The text-change column compares the original arm with the first configured shear factor. It is response sensitivity, not OCR truth selection.",
        "",
        "| reason | dataset | records | text changed | overlap improved | overlap regressed |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in cohort_summary:
        lines.append(
            f"| {row['reason']} | {row['dataset']} | {row['records']} | "
            f"{row['text_changed']} | {row['overlap_improved']} | "
            f"{row['overlap_regressed']} |"
        )
    lines.extend([
        "",
        "## Records",
        "",
        f"| source | PP | label | slope | texts at 0/{'/'.join(str(item) for item in factors)} | overlaps at 0/{'/'.join(str(item) for item in factors)} | stable improved factors | image |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ])
    for record in serializable:
        lines.append(
            f"| {record['source_name']} | `{record['text']}` | {record['known_label']} | "
            f"{record['slant'].get('slope')} | `{' / '.join([record['original']['text'], *(arm['result']['text'] for arm in record['arms'])])}` | "
            f"{' / '.join(str(value) for value in [record['original']['overlap_pair_count'], *(arm['result']['overlap_pair_count'] for arm in record['arms'])])} | "
            f"{record['comparison']['stable_geometry_improvement_factors']} | "
            f"`{record['comparison_image']}` |"
        )
    lines.extend(("", "## Outputs", "", f"- JSON: `{_windows_path(json_path)}`"))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723/report.json",
    )
    parser.add_argument("--dataset", default="test3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--factor",
        type=float,
        action="append",
        dest="factors",
        help="Counter-shear fraction; repeat for a curve (default: .25/.5/.75/1).",
    )
    parser.add_argument(
        "--reason",
        action="append",
        dest="reasons",
        help="Keep records carrying at least one repeated selection reason.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/latin_engcut_deslant_counterfactual_test3_20260724",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records, exclusions = _records(payload, dataset=args.dataset or None)
    factors = tuple(args.factors or SHEAR_FACTORS)
    if not factors or any(not 0 < factor <= 1 for factor in factors):
        parser.error("every --factor must be within (0, 1]")
    if args.reasons:
        requested = set(args.reasons)
        kept = [
            record
            for record in records
            if requested.intersection(record["selection_reasons"])
        ]
        exclusions["selection_reason_filter"] += len(records) - len(kept)
        records = kept
    if args.limit > 0:
        exclusions["limit"] += max(0, len(records) - args.limit)
        records = records[:args.limit]
    images: dict[str, np.ndarray] = {}
    evaluated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source = record["source_image"]
        if source not in images:
            images[source] = _read_image(source)
        print(f"[{index}/{len(records)}] {record['source_name']} {record['text']}", flush=True)
        evaluated.append(
            _evaluate(record, images[source], args.timeout, factors=factors)
        )
    _write_report(args.output_dir, evaluated, exclusions, factors=factors)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
