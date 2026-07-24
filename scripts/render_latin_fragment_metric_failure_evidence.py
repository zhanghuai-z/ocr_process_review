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


def _token_key(record: dict[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    return (
        str(record["source_image"]),
        str(record["text"]),
        tuple(int(item) for item in record["route_bbox"]),
    )


def _audit_token_index(payload: dict[str, Any]) -> dict[tuple[str, str, tuple[int, ...]], dict[str, Any]]:
    pages = [
        page
        for page in payload.get("project_pages") or []
        if int(page.get("run_index") or 0) == 1
    ] + list(payload.get("batch_pages") or [])
    index: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for page in pages:
        if page.get("status") != "ok":
            continue
        for token in page.get("tokens") or []:
            if not token.get("route_bbox"):
                continue
            record = {**token, "source_image": page["source_image"]}
            index.setdefault(_token_key(record), token)
    return index


def _strict_intersects(left: list[int], right: list[int]) -> bool:
    return bool(
        max(left[0], right[0]) < min(left[2], right[2])
        and max(left[1], right[1]) < min(left[3], right[3])
    )


def _production_word_quality(
    record: dict[str, Any],
    token: dict[str, Any],
) -> dict[str, Any]:
    atoms = list((token.get("current_result") or {}).get("atoms") or [])
    word_atoms = [atom for atom in atoms if atom.get("granularity") == "word"]
    if len(word_atoms) != 1 or not word_atoms[0].get("bbox"):
        raise RuntimeError(
            f"expected one persisted diagnostic word atom: {record['source_name']} {record['text']!r}"
        )
    word = word_atoms[0]
    word_bbox = [int(item) for item in word["bbox"]]
    other_boxes = [
        [int(item) for item in atom["bbox"]]
        for atom in atoms
        if atom is not word and atom.get("bbox") is not None
    ]
    components = list(record.get("owned_components") or [])
    outside_components = sum(
        not (
            word_bbox[0] <= int(component["bbox"][0])
            and word_bbox[1] <= int(component["bbox"][1])
            and word_bbox[2] >= int(component["bbox"][2])
            and word_bbox[3] >= int(component["bbox"][3])
        )
        for component in components
    )
    return {
        "word_atom": word,
        "all_atoms": atoms,
        "word_equals_route": word_bbox == [int(item) for item in record["route_bbox"]],
        "other_atom_overlap_count": sum(
            _strict_intersects(word_bbox, box) for box in other_boxes
        ),
        "owned_component_outside_count": outside_components,
    }


def _read_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read source image: {path}")
    return image


def _draw_bbox(
    image: np.ndarray,
    bbox: list[int],
    crop_bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = bbox
    cx1, cy1, _cx2, _cy2 = crop_bbox
    cv2.rectangle(
        image,
        (x1 - cx1, y1 - cy1),
        (x2 - cx1 - 1, y2 - cy1 - 1),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_word_quality_row(record: dict[str, Any]) -> np.ndarray:
    quality = record["production_word_quality"]
    image = _read_image(record["source_image"])
    boxes = [
        [int(item) for item in record["route_bbox"]],
        [int(item) for item in record["pp_bbox"]],
        *[
            [int(item) for item in atom["bbox"]]
            for atom in quality["all_atoms"]
            if atom.get("bbox") is not None
        ],
    ]
    height, width = image.shape[:2]
    crop_bbox = (
        max(0, min(box[0] for box in boxes) - 16),
        max(0, min(box[1] for box in boxes) - 10),
        min(width, max(box[2] for box in boxes) + 16),
        min(height, max(box[3] for box in boxes) + 10),
    )
    x1, y1, x2, y2 = crop_bbox
    raw = image[y1:y2, x1:x2].copy()
    native = raw.copy()
    final = raw.copy()
    for char in record.get("native_chars") or []:
        if char.get("bbox") is not None:
            _draw_bbox(native, char["bbox"], crop_bbox, (0, 0, 220), 1)
    _draw_bbox(native, record["pp_bbox"], crop_bbox, (220, 80, 0), 1)
    _draw_bbox(final, record["pp_bbox"], crop_bbox, (220, 80, 0), 1)
    for atom in quality["all_atoms"]:
        if atom.get("bbox") is None:
            continue
        if atom.get("granularity") == "word":
            _draw_bbox(final, atom["bbox"], crop_bbox, (0, 170, 0), 3)
        else:
            _draw_bbox(final, atom["bbox"], crop_bbox, (0, 140, 255), 2)

    scale = min(2.4, 480 / max(1, raw.shape[1]))
    panels: list[np.ndarray] = []
    for label, panel in (
        ("RAW CONTEXT", raw),
        ("NATIVE: RED CHAR / BLUE PP", native),
        ("FINAL: GREEN WORD / ORANGE OTHER / BLUE PP", final),
    ):
        panel = cv2.resize(
            panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        label_band = np.full((28, panel.shape[1], 3), 255, np.uint8)
        cv2.putText(
            label_band,
            label,
            (5, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.vstack((label_band, panel)))
    panel_height = max(panel.shape[0] for panel in panels)
    normalized: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((panel_height, panel.shape[1], 3), 255, np.uint8)
        canvas[:panel.shape[0]] = panel
        normalized.append(canvas)
    body = np.hstack(normalized)
    header = np.full((70, body.shape[1], 3), 255, np.uint8)
    word = quality["word_atom"]
    cv2.putText(
        header,
        f"{record['source_name']} | PP={record['text']} | FINAL WORD={word['text']} | source={word['source']}",
        (6, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"pp={record['pp_bbox']} word={word['bbox']} route_equal={quality['word_equals_route']} other_overlap={quality['other_atom_overlap_count']} owned_outside={quality['owned_component_outside_count']}",
        (6, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, body))


def _word_quality_sheets(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    page_size: int = 9,
) -> list[str]:
    names: list[str] = []
    for start in range(0, len(records), page_size):
        rows = [_render_word_quality_row(record) for record in records[start:start + page_size]]
        width = max(row.shape[1] for row in rows)
        parts = [_section_header(
            f"PRODUCTION WORD FALLBACK QUALITY {start + 1}-{start + len(rows)} / {len(records)}",
            "Green is the final production word atom. Blue is raw PP bbox. Orange is any other final atom; inspect ink containment and overlap.",
            width,
        )]
        for row in rows:
            canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
            canvas[:, :row.shape[1]] = row
            parts.append(canvas)
            parts.append(np.full((10, width, 3), 242, np.uint8))
        name = f"04_word_fallback_quality_{start // page_size + 1:02d}.png"
        cv2.imencode(".png", np.vstack(parts))[1].tofile(output_dir / name)
        names.append(name)
    return names


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


def _write_index(
    output_dir: Path,
    stats: dict[str, int],
    word_quality_sheets: list[str],
) -> None:
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
    word_quality_body = "\n".join(
        f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="Production word fallback quality page"></a>'
        for name in word_quality_sheets
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
  <section>
    <h2>Final production word-box quality</h2>
    <p>All {stats['fallback_candidates']} current fallback candidates are shown. Green is the final word atom, blue is the raw PP bbox, and orange marks any other final atom. Word equals route: {stats['word_equals_route']}; overlaps another final atom: {stats['word_overlaps_other']}.</p>
    {word_quality_body}
  </section>
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
        "--audit-input",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723/report.json",
    )
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
    audit_payload = json.loads(args.audit_input.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    targets, controls = _known_cohorts(records)
    candidates = _v3_candidates(records)
    audit_index = _audit_token_index(audit_payload)
    fallback_candidates: list[dict[str, Any]] = []
    for record in candidates:
        if not record.get("current_word_fallback"):
            continue
        token = audit_index.get(_token_key(record))
        if token is None:
            raise RuntimeError(
                f"audit token missing: {record['source_name']} {record['text']!r}"
            )
        fallback_candidates.append({
            **record,
            "production_word_quality": _production_word_quality(record, token),
        })
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
    word_sheets = _word_quality_sheets(fallback_candidates, output_dir)
    stats = {
        "targets": len(targets),
        "controls": len(controls),
        "candidates": len(candidates),
        "fallback_candidates": len(fallback_candidates),
        "word_equals_route": sum(
            record["production_word_quality"]["word_equals_route"]
            for record in fallback_candidates
        ),
        "word_overlaps_other": sum(
            bool(record["production_word_quality"]["other_atom_overlap_count"])
            for record in fallback_candidates
        ),
    }
    _write_index(output_dir, stats, word_sheets)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "latin_fragment_metric_failure_evidence.v2",
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
                "word_quality_sheets": word_sheets,
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
