#!/usr/bin/env python3
"""Render curated evidence showing why Latin fragment metrics were rejected."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_latin_charbox_fragment_quality import (  # noqa: E402
    _quality_reference_candidate,
    _render_row,
)


def _known_cohorts(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = [
        record
        for record in records
        if record.get("known_label") == "target_bad_geometry"
    ]
    controls = [
        record
        for record in records
        if record.get("known_label") == "control_usable_geometry"
    ]
    return targets, controls


def _v3_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [record for record in records if _quality_reference_candidate(record)]
    return sorted(
        selected,
        key=lambda item: (
            not bool(item.get("current_word_fallback")),
            -float(item["fragment"].get("fragment_area_ratio") or 0.0),
            str(item.get("source_name") or ""),
        ),
    )


def _section_header(title: str, detail: str, width: int) -> np.ndarray:
    canvas = np.full((74, width, 3), 255, np.uint8)
    cv2.putText(
        canvas,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        detail[:180],
        (8, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _stack_sections(
    sections: list[tuple[str, str, list[dict[str, Any]]]],
    output: Path,
) -> None:
    rendered: list[tuple[str, str, list[np.ndarray]]] = []
    width = 900
    for title, detail, records in sections:
        rows = [_render_row(record) for record in records]
        if rows:
            width = max(width, *(row.shape[1] for row in rows))
        rendered.append((title, detail, rows))
    parts: list[np.ndarray] = []
    for title, detail, rows in rendered:
        parts.append(_section_header(title, detail, width))
        for row in rows:
            canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
            canvas[:, :row.shape[1]] = row
            parts.append(canvas)
        parts.append(np.full((16, width, 3), 245, np.uint8))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(parts))[1].tofile(output)


def _write_index(output_dir: Path, stats: dict[str, int]) -> None:
    cards = [
        (
            "01_v2_boundary_cut_saturation.png",
            "v2 boundary-cut saturation",
            "Every known bad token and every usable control has cut ratio 1.0. The metric observes a real crop-edge contact but cannot separate bad geometry from normal geometry.",
        ),
        (
            "02_v3_misses_known_bad.png",
            "v3 misses all known bad tokens",
            "All four known bad tokens have fragment count 0 and fragment area 0. The visually wrong ink remains connected to the main crop ink, so disconnected-fragment counting cannot see it.",
        ),
        (
            "03_v3_selected_examples.png",
            "What v3 actually selects",
            "The reference threshold selects other severe or already-fallback cases rather than the four target failures. These examples explain why the metric would mostly duplicate existing fallback coverage.",
        ),
    ]
    body = "\n".join(
        f"""
        <section>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(detail)}</p>
          <a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(title)}"></a>
        </section>
        """
        for name, title, detail in cards
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Latin fragment metric failure evidence</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 system-ui, sans-serif; color: #181818; background: #f3f4f6; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 26px; margin: 0 0 8px; }}
    h2 {{ font-size: 19px; margin: 0 0 6px; }}
    p {{ max-width: 1050px; margin: 0 0 14px; color: #4b5563; }}
    .summary {{ margin-bottom: 22px; }}
    section {{ margin: 0 0 24px; padding: 18px; background: white; border: 1px solid #d1d5db; border-radius: 6px; }}
    img {{ display: block; max-width: 100%; height: auto; border: 1px solid #d1d5db; }}
  </style>
</head>
<body><main>
  <h1>Latin fragment metric failure evidence</h1>
  <p class="summary">Diagnostic projection only. Source: latin_charbox_fragment_quality.v3. Known bad: {stats['targets']}; usable controls: {stats['controls']}; v3 reference candidates: {stats['candidates']}; known bad selected by v3: 0.</p>
  {body}
</main></body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "debug/latin_charbox_fragment_quality_20260724/report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/latin_fragment_metric_failure_evidence_20260724",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    targets, controls = _known_cohorts(records)
    candidates = _v3_candidates(records)
    if len(targets) != 4 or len(controls) != 3:
        raise RuntimeError(
            f"expected known cohort 4 targets / 3 controls, got {len(targets)} / {len(controls)}"
        )

    output_dir = args.output_dir
    _stack_sections(
        [
            (
                "KNOWN BAD: boundary cut ratio is 1.0",
                "Magenta edge contacts appear on every target, but this alone does not distinguish failure.",
                targets,
            ),
            (
                "USABLE CONTROLS: boundary cut ratio is also 1.0",
                "The same metric saturates on Review, Economic, and American even though their native geometry is usable.",
                controls,
            ),
        ],
        output_dir / "01_v2_boundary_cut_saturation.png",
    )
    _stack_sections(
        [
            (
                "V3 FALSE NEGATIVES: all four known bad tokens report fragments=0",
                "Red fragment heat is absent because the visually wrong ink remains connected to the main crop ink.",
                targets,
            )
        ],
        output_dir / "02_v3_misses_known_bad.png",
    )
    _stack_sections(
        [
            (
                "V3 REFERENCE CANDIDATES: examples selected elsewhere",
                "The first twelve candidates are ranked with already-fallback cases first; none is a known target.",
                candidates[:12],
            )
        ],
        output_dir / "03_v3_selected_examples.png",
    )
    stats = {
        "targets": len(targets),
        "controls": len(controls),
        "candidates": len(candidates),
    }
    _write_index(output_dir, stats)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "latin_fragment_metric_failure_evidence.v1",
                "source_schema": payload.get("schema"),
                **stats,
                "known_target_v2_cut_ratios": [
                    record["fragment"].get("cut_char_ratio") for record in targets
                ],
                "known_control_v2_cut_ratios": [
                    record["fragment"].get("cut_char_ratio") for record in controls
                ],
                "known_target_v3_fragment_counts": [
                    record["fragment"].get("fragment_char_count") for record in targets
                ],
                "known_targets_selected_by_v3": sum(
                    record.get("known_label") == "target_bad_geometry"
                    for record in candidates
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(_windows_path(output_dir / "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
