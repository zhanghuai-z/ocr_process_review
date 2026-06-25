#!/usr/bin/env python3
"""Draft AutoRec-lite dispatch units for sample 120194.

This script is intentionally read-only for the product pipeline.  It converts
the saved Paddle/VL layout response into explicit units that can later feed a
Hanwang/Eng20 scheduler.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.paddle_labels import (
    is_hanwang_skip_label,
    is_hanwang_text_label,
    normalize_paddle_label,
)
from app.core.paddle_line_routing import (
    block_bbox_xyxy,
    block_text,
    line_routes_for_block,
    route_authority_label,
    route_subblocks_for_block,
)
from app.models import Page


FORMULA_LABELS = {
    "display_formula",
    "equation",
    "equation_block",
    "formula",
    "formula_number",
    "inline_formula",
    "isolated_formula",
}
TABLE_LABELS = {
    "table",
    "table_block",
    "table_body",
    "table_region",
}
FIGURE_LABELS = {
    "chart",
    "figure",
    "graphic",
    "image",
    "photo",
    "picture",
}
LATIN_HINT_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\\-]{1,}")
MATH_SPAN_RE = re.compile(r"\$[^$]*\$")


def _load_layout(path: Path) -> tuple[Page, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    page_info = raw["page"]
    image_path = Path(str(page_info["display_image_path"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    page = Page(
        image_path=str(image_path),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    page.source_path = str(page_info.get("source_path") or "")
    _blocks, _overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    return page, list(page.ppvl_parsing_res_list or [])


def _kind_from_label(label: str) -> str:
    normalized = normalize_paddle_label(label)
    if normalized == "formula_number":
        return "formula_number"
    if normalized == "inline_formula":
        return "inline_formula"
    if normalized in FORMULA_LABELS:
        return "display_formula"
    if normalized in TABLE_LABELS:
        return "table"
    if normalized in FIGURE_LABELS:
        return "figure"
    if is_hanwang_text_label(normalized):
        return "text"
    if is_hanwang_skip_label(normalized):
        return "skip"
    return "unknown"


def _engine_for_kind(kind: str) -> str:
    if kind == "text_slice":
        return "hanwang_chn"
    if kind in {"inline_formula", "display_formula", "formula_number", "table", "figure"}:
        return "paddle_truth"
    if kind == "latin_span":
        return "eng20"
    return "none"


def _unit(
    *,
    unit_id: str,
    block_idx: int,
    kind: str,
    bbox: tuple[int, int, int, int],
    label: str,
    route_source: str,
    text_truth: str = "",
    parent_unit_id: str | None = None,
    line_idx: int | None = None,
    segment_idx: int | None = None,
    review_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": unit_id,
        "parent_unit_id": parent_unit_id,
        "block_idx": block_idx,
        "line_idx": line_idx,
        "segment_idx": segment_idx,
        "kind": kind,
        "label": label,
        "bbox": list(bbox),
        "text_truth": text_truth,
        "geometry_source": route_source,
        "text_source": "paddle_vl" if text_truth else "",
        "engine": _engine_for_kind(kind),
        "review_flags": list(review_flags or []),
    }


def _math_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in MATH_SPAN_RE.finditer(text or "")]


def _inside_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start >= span_start and end <= span_end for span_start, span_end in spans)


def _latin_hints(text: str, block_idx: int, *, include_math_latin: bool = False) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    math_spans = _math_spans(text)
    for match in LATIN_HINT_RE.finditer(text or ""):
        token = match.group(0)
        if token.isdigit():
            continue
        in_math = _inside_any_span(match.start(), match.end(), math_spans)
        if in_math and not include_math_latin:
            continue
        hints.append({
            "block_idx": block_idx,
            "text": token,
            "start": match.start(),
            "end": match.end(),
            "status": "text_only_no_bbox",
            "context": "math" if in_math else "prose",
        })
    return hints


def build_dispatch_units(
    page: Page,
    records: list[dict[str, Any]],
    *,
    include_math_latin: bool = False,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    latin_hints: list[dict[str, Any]] = []
    width = int(page.width)
    height = int(page.height)

    for block_idx, record in enumerate(records):
        label = route_authority_label(record, "unknown")
        normalized_label = normalize_paddle_label(label)
        parent_kind = _kind_from_label(normalized_label)
        bbox = block_bbox_xyxy(record, width, height)
        text = block_text(record)
        parent_id = f"b{block_idx:03d}"
        routes = line_routes_for_block(record, width, height)
        subblocks = route_subblocks_for_block(record, width, height)
        parent_engine = "route_children" if parent_kind == "text" else _engine_for_kind(parent_kind)
        units.append({
            "id": parent_id,
            "parent_unit_id": None,
            "block_idx": block_idx,
            "line_idx": None,
            "segment_idx": None,
            "kind": f"{parent_kind}_parent",
            "label": normalized_label,
            "bbox": list(bbox),
            "text_truth": text,
            "geometry_source": "paddle_vl_block",
            "text_source": "paddle_vl" if text else "",
            "engine": parent_engine,
            "review_flags": [],
            "subblock_count": len(subblocks),
            "line_route_count": len(routes),
        })

        if parent_kind == "text":
            latin_hints.extend(_latin_hints(text, block_idx, include_math_latin=include_math_latin))
            if routes:
                for line_idx, route in enumerate(routes):
                    for segment_idx, segment in enumerate(route.get("segments") or []):
                        segment_kind = str(segment.get("kind") or "")
                        segment_label = normalize_paddle_label(segment.get("label") or "")
                        segment_bbox = tuple(int(v) for v in segment["bbox"])
                        if segment_kind == "text":
                            kind = "text_slice"
                            flags: list[str] = []
                        elif segment_kind == "formula":
                            kind = "inline_formula"
                            flags = ["route_inline_formula"]
                        else:
                            kind = "skip"
                            flags = ["route_skip_segment"]
                        units.append(_unit(
                            unit_id=f"{parent_id}.l{line_idx:02d}.s{segment_idx:02d}",
                            parent_unit_id=parent_id,
                            block_idx=block_idx,
                            line_idx=line_idx,
                            segment_idx=segment_idx,
                            kind=kind,
                            label=segment_label or segment_kind,
                            bbox=segment_bbox,
                            text_truth=str(segment.get("text") or ""),
                            route_source="_layout_line_routes",
                            review_flags=flags,
                        ))
            else:
                units.append(_unit(
                    unit_id=f"{parent_id}.text",
                    parent_unit_id=parent_id,
                    block_idx=block_idx,
                    line_idx=0,
                    segment_idx=0,
                    kind="text_slice",
                    label=normalized_label,
                    bbox=bbox,
                    text_truth="",
                    route_source="paddle_vl_block_bbox",
                    review_flags=["needs_ppocrv5_line_route"],
                ))
        elif parent_kind in {"inline_formula", "display_formula", "formula_number", "table", "figure"}:
            units.append(_unit(
                unit_id=f"{parent_id}.{parent_kind}",
                parent_unit_id=parent_id,
                block_idx=block_idx,
                kind=parent_kind,
                label=normalized_label,
                bbox=bbox,
                text_truth=text,
                route_source="paddle_vl_block_bbox",
            ))

    by_engine: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for unit in units:
        by_engine[unit["engine"]] = by_engine.get(unit["engine"], 0) + 1
        by_kind[unit["kind"]] = by_kind.get(unit["kind"], 0) + 1

    return {
        "schema": "autorec_lite_dispatch_units.v0",
        "page": {
            "page_number": page.page_number,
            "image_path": page.display_image_path,
            "width": width,
            "height": height,
        },
        "summary": {
            "unit_count": len(units),
            "latin_hint_count": len(latin_hints),
            "by_engine": by_engine,
            "by_kind": by_kind,
        },
        "units": units,
        "latin_hints": latin_hints,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 120194 AutoRec-lite Dispatch Units",
        "",
        f"- page: `{payload['page']['page_number']}`",
        f"- image: `{payload['page']['image_path']}`",
        f"- units: `{payload['summary']['unit_count']}`",
        f"- latin_hints: `{payload['summary']['latin_hint_count']}`",
        "",
        "## By Engine",
        "",
    ]
    for engine, count in sorted(payload["summary"]["by_engine"].items()):
        lines.append(f"- `{engine}`: `{count}`")
    lines.extend(["", "## Units", ""])
    for unit in payload["units"]:
        lines.append(
            "- "
            f"`{unit['id']}` "
            f"kind=`{unit['kind']}` "
            f"engine=`{unit['engine']}` "
            f"bbox=`{unit['bbox']}` "
            f"label=`{unit['label']}` "
            f"flags=`{unit['review_flags']}`"
        )
        text = str(unit.get("text_truth") or "").strip()
        if text:
            lines.append(f"  text: {text[:160]}")
    lines.extend(["", "## Latin Hints", ""])
    for hint in payload["latin_hints"]:
        lines.append(
            f"- block `{hint['block_idx']}` text=`{hint['text']}` "
            f"span=`{hint['start']}..{hint['end']}` status=`{hint['status']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        type=Path,
        default=REPO_ROOT / "file/244771纵校/120194.layout-api.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "debug/120194_autorec_lite_dispatch",
    )
    parser.add_argument("--include-math-latin", action="store_true")
    args = parser.parse_args()

    page, records = _load_layout(args.layout)
    payload = build_dispatch_units(page, records, include_math_latin=args.include_math_latin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "120194_dispatch_units.json"
    md_path = args.out_dir / "120194_dispatch_units.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
