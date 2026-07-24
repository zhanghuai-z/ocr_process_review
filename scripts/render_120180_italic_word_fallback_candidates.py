#!/usr/bin/env python3
"""Render post-EngCut italic word-fallback candidates on 120180.tif."""
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

from scripts.render_latin_fragment_metric_failure_evidence import (  # noqa: E402
    _audit_token_index,
    _production_word_quality,
    _render_word_quality_row,
    _section_header,
    _token_key,
    _windows_path,
)


SLOPE_THRESHOLD = 0.12
IMPROVEMENT_THRESHOLD = 0.05
PROPOSED_SOURCE = "diagnostic:ppocr_latin_word_after_slant_structural_gate"
ROUTE_DISPLAY_SOURCE = "diagnostic:route_bbox_display_only"


def _is_candidate(record: dict[str, Any], source_name: str) -> bool:
    return bool(record.get("structural_conflict") and _passes_slant(record, source_name))


def _passes_slant(record: dict[str, Any], source_name: str) -> bool:
    slant = record.get("slant") or {}
    return bool(
        record.get("source_name") == source_name
        and slant.get("measurable")
        and float(slant.get("slope") or 0.0) >= SLOPE_THRESHOLD
        and float(slant.get("score_improvement") or 0.0)
        >= IMPROVEMENT_THRESHOLD
    )


def _proposed_token(token: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_result": {
            "atoms": [{
                "text": str(token["text"]),
                "bbox": list(token["route_bbox"]),
                "granularity": "word",
                "source": PROPOSED_SOURCE,
            }]
        }
    }


def _latin_letter_count(text: str) -> int:
    return sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _source_page(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    matches = [
        page
        for page in payload.get("project_pages") or []
        if int(page.get("run_index") or 0) == 1
        and page.get("status") == "ok"
        and page.get("source_name") == source_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one audit page for {source_name!r}, got {len(matches)}")
    return matches[0]


def _display_token(token: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_result": {
            "atoms": [{
                "text": str(token["text"]),
                "bbox": list(token["route_bbox"]),
                "granularity": "word",
                "source": ROUTE_DISPLAY_SOURCE,
            }]
        }
    }


def _classification(
    token: dict[str, Any],
    slant_record: dict[str, Any] | None,
    source_name: str,
) -> tuple[str, tuple[int, int, int]]:
    if token.get("symbol_conflicts"):
        return "OWNERSHIP CONFLICT", (180, 0, 180)
    if bool((token.get("current_result") or {}).get("word_fallback")):
        return "CURRENT FINAL", (0, 145, 0)
    if slant_record is not None and _is_candidate(slant_record, source_name):
        return "GATE HIT", (0, 210, 0)
    if slant_record is not None and _passes_slant(slant_record, source_name):
        return "SLOPE RECALL", (0, 190, 220)
    return "NOT SELECTED", (145, 145, 145)


def _all_latin_records(
    audit_page: dict[str, Any],
    slant_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slant_index = {_token_key(record): record for record in slant_records}
    records: list[dict[str, Any]] = []
    for token in audit_page.get("tokens") or []:
        letter_count = _latin_letter_count(str(token.get("text") or ""))
        if letter_count == 0 or not token.get("route_bbox"):
            continue
        source_record = {
            "source_name": audit_page["source_name"],
            "source_image": audit_page["source_image"],
            "text": token["text"],
            "route_bbox": token["route_bbox"],
        }
        slant_record = slant_index.get(_token_key(source_record))
        status, color = _classification(token, slant_record, audit_page["source_name"])
        is_current = status == "CURRENT FINAL"
        base_record = {
            **source_record,
            "pp_bbox": token["pp_bbox"],
            "native_chars": token["native_chars"],
            "native_text": (token.get("metrics") or {}).get("native_text"),
            "owned_components": (token.get("metrics") or {}).get("owned_components") or [],
            "word_box_status": status,
            "word_box_color": color,
            "word_box_color_name": {
                "CURRENT FINAL": "DARK GREEN",
                "GATE HIT": "BRIGHT GREEN",
                "SLOPE RECALL": "YELLOW",
                "NOT SELECTED": "GRAY",
                "OWNERSHIP CONFLICT": "MAGENTA",
            }[status],
            "classification": status,
        }
        base_record["production_word_quality"] = _production_word_quality(
            base_record,
            token if is_current else _display_token(token),
        )
        records.append(base_record)
    records.sort(
        key=lambda item: (
            int(item["route_bbox"][1]),
            int(item["route_bbox"][0]),
            int(item["route_bbox"][3]),
        )
    )
    return (
        [record for record in records if _latin_letter_count(record["text"]) >= 2],
        [record for record in records if _latin_letter_count(record["text"]) == 1],
    )


def _render_records(
    records: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    title: str,
    detail: str,
    *,
    page_size: int = 8,
) -> list[str]:
    names: list[str] = []
    for start in range(0, len(records), page_size):
        rows = [_render_word_quality_row(record) for record in records[start:start + page_size]]
        width = max(row.shape[1] for row in rows)
        parts = [_section_header(
            f"{title} {start + 1}-{start + len(rows)} / {len(records)}",
            detail,
            width,
        )]
        for row in rows:
            canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
            canvas[:, :row.shape[1]] = row
            parts.extend((canvas, np.full((10, width, 3), 242, np.uint8)))
        name = f"{prefix}_{start // page_size + 1:02d}.png"
        cv2.imencode(".png", np.vstack(parts))[1].tofile(output_dir / name)
        names.append(name)
    return names


def _image_links(names: list[str], label: str) -> str:
    return "\n".join(
        f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(label)}"></a>'
        for name in names
    )


def _write_index(
    output_dir: Path,
    proposed: list[dict[str, Any]],
    current: list[dict[str, Any]],
    proposed_sheets: list[str],
    current_sheets: list[str],
    all_word_sheets: list[str],
    single_latin_sheets: list[str],
    all_words: list[dict[str, Any]],
    single_latin: list[dict[str, Any]],
    slope_recall: list[dict[str, Any]],
    slope_recall_sheets: list[str],
) -> None:
    document = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>120180 italic word fallback candidates</title>
<style>
body {{ margin:0; font:15px/1.5 system-ui,sans-serif; color:#181818; background:#f3f4f6; }}
main {{ max-width:1500px; margin:auto; padding:24px; }}
h1 {{ font-size:26px; margin:0 0 8px; }} h2 {{ font-size:19px; margin:0 0 6px; }}
p {{ max-width:1100px; color:#4b5563; }} section {{ margin:20px 0; padding:18px; background:white; border:1px solid #d1d5db; border-radius:6px; }}
img {{ display:block; max-width:100%; height:auto; border:1px solid #d1d5db; margin-top:12px; }}
</style></head><body><main>
<h1>120180.tif post-EngCut italic fallback experiment</h1>
<p>Diagnostic only. No image correction and no OCR rerun. Candidate gate: uniquely owned Latin token already present in the audit, right slope &gt;= {SLOPE_THRESHOLD}, projection improvement &gt;= {IMPROVEMENT_THRESHOLD}, and original EngCut structural conflict.</p>
<section><h2>Proposed additional word fallbacks ({len(proposed)})</h2>
<p>These tokens are not word fallbacks in production today. Green is the proposed PP text + route foreground word box; red is original EngCut character geometry; blue is raw PP bbox. Review this section first for false positives and word-box contamination.</p>
{_image_links(proposed_sheets, 'Proposed additional fallback candidates')}
</section>
<section><h2>Current production word fallbacks ({len(current)})</h2>
<p>Reference cohort already degraded by the current overlap rule. Green is the actual final word atom.</p>
{_image_links(current_sheets, 'Current production fallback references')}
</section>
<section><h2>Additional slope-recall candidates ({len(slope_recall)})</h2>
<p>Yellow means the token passes the same right-slope and projection-improvement thresholds but has no original structural conflict. These are acceptable extra candidates under the recall-first rule; they remain separate from the evidence-backed structural gate.</p>
{_image_links(slope_recall_sheets, 'Additional slope-recall candidates')}
</section>
<section><h2>All English-word tokens in page order ({len(all_words)})</h2>
<p>Complete cohort with at least two Latin letters. Dark green: current final word fallback. Bright green: structural gate hit. Yellow: slope-recall candidate. Gray: not selected. Magenta: ownership conflict. A regular word selected by either candidate tier is acceptable for this recall-oriented review; inspect gray rows for remaining missed italic words.</p>
{_image_links(all_word_sheets, 'All English-word tokens in page order')}
</section>
<section><h2>Single-letter Latin appendix ({len(single_latin)})</h2>
<p>Included so no Latin-bearing token is hidden from the review. These are not part of the two-letter word-gate statistics.</p>
{_image_links(single_latin_sheets, 'Single-letter Latin token appendix')}
</section>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slant-input",
        type=Path,
        default=ROOT / "debug/latin_token_slant_gate_20260724/report.json",
    )
    parser.add_argument(
        "--audit-input",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723/report.json",
    )
    parser.add_argument("--source-name", default="120180.tif")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/120180_italic_word_fallback_candidates_20260724",
    )
    args = parser.parse_args()
    slant_payload = json.loads(args.slant_input.read_text(encoding="utf-8"))
    audit_payload = json.loads(args.audit_input.read_text(encoding="utf-8"))
    audit_page = _source_page(audit_payload, args.source_name)
    audit_index = _audit_token_index(audit_payload)
    selected = [
        record
        for record in slant_payload.get("records") or []
        if _is_candidate(record, args.source_name)
    ]
    source_slant_records = [
        record
        for record in slant_payload.get("records") or []
        if record.get("source_name") == args.source_name
    ]
    all_words, single_latin = _all_latin_records(
        audit_page, source_slant_records
    )
    slope_recall = [
        record for record in all_words if record["classification"] == "SLOPE RECALL"
    ]
    proposed: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []
    for slant_record in selected:
        token = audit_index.get(_token_key(slant_record))
        if token is None:
            raise RuntimeError(
                f"audit token missing: {slant_record['source_name']} {slant_record['text']!r}"
            )
        fragment_record = {
            "source_name": slant_record["source_name"],
            "source_image": slant_record["source_image"],
            "text": token["text"],
            "route_bbox": token["route_bbox"],
            "pp_bbox": token["pp_bbox"],
            "native_chars": token["native_chars"],
            "owned_components": (token.get("metrics") or {}).get("owned_components") or [],
        }
        is_current = bool(slant_record.get("current_word_fallback"))
        result_token = token if is_current else _proposed_token(token)
        rendered = {
            **fragment_record,
            "word_box_status": "FINAL" if is_current else "PROPOSED",
            "production_word_quality": _production_word_quality(
                fragment_record, result_token
            ),
        }
        (current if is_current else proposed).append(rendered)
        report_records.append({
            "text": token["text"],
            "native_text": slant_record.get("native_text"),
            "route_bbox": token["route_bbox"],
            "pp_bbox": token["pp_bbox"],
            "current_word_fallback": is_current,
            "known_label": slant_record.get("known_label"),
            "split_component_area_ratio": slant_record.get("split_component_area_ratio"),
            "slope": slant_record["slant"].get("slope"),
            "score_improvement": slant_record["slant"].get("score_improvement"),
        })
    proposed.sort(key=lambda item: (item["text"], item["route_bbox"]))
    current.sort(key=lambda item: (item["text"], item["route_bbox"]))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    proposed_sheets = _render_records(
        proposed,
        output_dir,
        "01_proposed_additional_fallbacks",
        "PROPOSED ADDITIONAL WORD FALLBACKS",
        "Green is diagnostic PP text + route bbox, not persisted state. Inspect whether it encloses only the intended word.",
    )
    current_sheets = _render_records(
        current,
        output_dir,
        "02_current_fallback_references",
        "CURRENT PRODUCTION WORD FALLBACKS",
        "Green is the current final word atom produced by the existing overlap rule.",
    )
    slope_recall_sheets = _render_records(
        slope_recall,
        output_dir,
        "03_slope_recall_additional",
        "ADDITIONAL SLOPE-RECALL WORD CANDIDATES",
        "Yellow passes right slope + projection improvement without structural conflict. Regular-word false positives are acceptable in this recall-first review.",
    )
    all_word_sheets = _render_records(
        all_words,
        output_dir,
        "03_all_english_words_page_order",
        "ALL ENGLISH-WORD TOKENS IN PAGE ORDER",
        "Dark green=current; bright green=structural gate; yellow=slope recall; gray=not selected; magenta=ownership conflict.",
    )
    single_latin_sheets = _render_records(
        single_latin,
        output_dir,
        "04_single_latin_token_appendix",
        "SINGLE-LETTER LATIN TOKEN APPENDIX",
        "Complete appendix for Latin-bearing tokens excluded from the two-letter word cohort.",
    )
    _write_index(
        output_dir,
        proposed,
        current,
        proposed_sheets,
        current_sheets,
        all_word_sheets,
        single_latin_sheets,
        all_words,
        single_latin,
        slope_recall,
        slope_recall_sheets,
    )
    (output_dir / "report.json").write_text(
        json.dumps({
            "schema": "120180_italic_word_fallback_candidates.v1",
            "scope": "diagnostic-only post-EngCut gate; no OCR rerun or persistence",
            "source_name": args.source_name,
            "thresholds": {
                "right_slope": SLOPE_THRESHOLD,
                "projection_improvement": IMPROVEMENT_THRESHOLD,
                "structural_conflict": True,
            },
            "candidate_count": len(selected),
            "proposed_additional_count": len(proposed),
            "current_fallback_count": len(current),
            "all_english_word_count": len(all_words),
            "single_latin_token_count": len(single_latin),
            "slope_recall_additional_count": len(slope_recall),
            "all_word_classifications": {
                status: sum(record["classification"] == status for record in all_words)
                for status in (
                    "CURRENT FINAL",
                    "GATE HIT",
                    "SLOPE RECALL",
                    "NOT SELECTED",
                    "OWNERSHIP CONFLICT",
                )
            },
            "records": report_records,
            "outputs": {
                "proposed_sheets": proposed_sheets,
                "current_sheets": current_sheets,
                "slope_recall_sheets": slope_recall_sheets,
                "all_word_sheets": all_word_sheets,
                "single_latin_sheets": single_latin_sheets,
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(_windows_path(output_dir / "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
