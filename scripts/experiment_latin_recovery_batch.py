#!/usr/bin/env python3
"""Batch Latin span recovery experiment on saved layout API samples."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from scripts.experiment_120194_dispatch_units import _load_layout, build_dispatch_units
from scripts.experiment_120194_latin_binding import (
    _bind_hint,
    _crop_bindings,
    _run_eng20_for_crops,
)
from scripts.experiment_120194_latin_hanwang import TEXT_LABELS, _has_latin_text

from app.core.paddle_line_routing import block_bbox_xyxy
from app.engines.hanwang.micro_recblock import run_micro_recblock


def _norm_token(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text or "").lower()


def _layout_id(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _target_records(page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    before: list[dict[str, Any]] = []
    for record_index, record in enumerate(page.ppvl_parsing_res_list or []):
        label = str(record.get("block_label") or record.get("label") or "")
        text = str(record.get("block_content") or "")
        if label not in TEXT_LABELS or not _has_latin_text(text):
            continue
        targets.append(record)
        before.append({
            "record_index": record_index,
            "row_index": len(targets) - 1,
            "label": label,
            "block_bbox": list(block_bbox_xyxy(record, page.width, page.height)),
            "block_content": text,
        })
    return targets, before


def _echo_hanwang(page, records: list[dict[str, Any]], before: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    image = cv2.imread(page.display_image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {page.display_image_path}")

    started = time.time()
    rows, stats = run_micro_recblock(image, records, include_chars=True)
    index_by_row = {
        int(item["row_index"]): int(item["record_index"])
        for item in before
    }
    after: list[dict[str, Any]] = []
    for row in rows:
        value = row.to_dict()
        value["record_index"] = index_by_row.get(row.block_idx, row.block_idx)
        after.append(value)

    payload = {
        "source_image": page.display_image_path,
        "stats": stats.__dict__,
        "elapsed_seconds": time.time() - started,
        "before": before,
        "after": after,
    }
    _json_dump(out_dir / "hanwang_echo.json", payload)
    return payload


def _bind_dispatch(dispatch: dict[str, Any], echo: dict[str, Any], out_dir: Path, run_eng20: bool) -> dict[str, Any]:
    source_texts = {
        int(unit["block_idx"]): str(unit.get("text_truth") or "")
        for unit in dispatch.get("units") or []
        if unit.get("kind") == "text_parent" and "block_idx" in unit
    }
    rows_by_record = {
        int(row["record_index"]): row
        for row in echo.get("after") or []
        if "record_index" in row
    }
    cursors: dict[int, int] = {}
    bindings: list[dict[str, Any]] = []
    for hint in dispatch.get("latin_hints") or []:
        block_idx = int(hint["block_idx"])
        row = rows_by_record.get(block_idx)
        if row is None:
            bindings.append({
                **hint,
                "binding_status": "no_hanwang_echo_for_block",
            })
            continue
        bound, cursor = _bind_hint(
            hint,
            row,
            cursors.get(block_idx, 0),
            source_text=source_texts.get(block_idx, ""),
            alignment_fallback=True,
        )
        cursors[block_idx] = cursor
        bindings.append(bound)

    image_path = Path(str(echo["source_image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    _crop_bindings(image_path, bindings, out_dir)
    if run_eng20:
        _run_eng20_for_crops(out_dir, bindings)

    exact_count = sum(1 for item in bindings if item.get("binding_status") == "exact_hanwang_text_match")
    alignment_count = sum(1 for item in bindings if item.get("binding_status") == "alignment_fallback_match")
    eng20_ok_count = sum(
        1
        for item in bindings
        if item.get("crop") and item.get("eng20_returncode") == 0 and item.get("eng20_text") == item.get("text")
    )
    eng20_norm_ok_count = sum(
        1
        for item in bindings
        if (
            item.get("crop")
            and item.get("eng20_returncode") == 0
            and _norm_token(str(item.get("eng20_text") or "")) == _norm_token(str(item.get("text") or ""))
        )
    )
    eng20_contains_count = sum(
        1
        for item in bindings
        if (
            item.get("crop")
            and item.get("eng20_returncode") == 0
            and _norm_token(str(item.get("text") or ""))
            and _norm_token(str(item.get("text") or "")) in _norm_token(str(item.get("eng20_text") or ""))
        )
    )
    eng20_ran_count = sum(1 for item in bindings if item.get("crop") and item.get("eng20_returncode") is not None)
    payload = {
        "schema": "latin_recovery_binding_batch.v0",
        "source_image": str(image_path),
        "summary": {
            "latin_hint_count": len(bindings),
            "exact_match_count": exact_count,
            "alignment_fallback_count": alignment_count,
            "matched_count": exact_count + alignment_count,
            "unmatched_count": len(bindings) - exact_count - alignment_count,
            "eng20_ran_count": eng20_ran_count,
            "eng20_ok_count": eng20_ok_count,
            "eng20_norm_ok_count": eng20_norm_ok_count,
            "eng20_contains_count": eng20_contains_count,
        },
        "bindings": bindings,
    }
    _json_dump(out_dir / "latin_binding.json", payload)
    return payload


def _page_summary(page_id: str, dispatch: dict[str, Any], echo: dict[str, Any] | None, binding: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "page_id": page_id,
        "latin_hint_count": dispatch.get("summary", {}).get("latin_hint_count", 0),
        "unit_count": dispatch.get("summary", {}).get("unit_count", 0),
        "target_block_count": len(echo.get("before", [])) if echo else 0,
        "hanwang_seconds": echo.get("elapsed_seconds", 0) if echo else 0,
        "error": error,
    }
    if binding:
        summary.update(binding.get("summary", {}))
        statuses: dict[str, int] = {}
        for item in binding.get("bindings") or []:
            status = str(item.get("binding_status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        summary["binding_statuses"] = statuses
        summary["eng20_mismatches"] = [
            {
                "block_idx": item.get("block_idx"),
                "text": item.get("text"),
                "status": item.get("binding_status"),
                "eng20_text": item.get("eng20_text"),
                "crop": item.get("crop", ""),
            }
            for item in binding.get("bindings") or []
            if item.get("crop") and item.get("eng20_returncode") == 0 and item.get("eng20_text") != item.get("text")
        ][:20]
        summary["unmatched"] = [
            {
                "block_idx": item.get("block_idx"),
                "text": item.get("text"),
                "status": item.get("binding_status"),
            }
            for item in binding.get("bindings") or []
            if item.get("binding_status") not in {"exact_hanwang_text_match", "alignment_fallback_match"}
        ][:20]
    return summary


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    total = payload["summary"]
    lines = [
        "# Latin Recovery Batch Experiment",
        "",
        f"- pages: `{total['page_count']}`",
        f"- pages_with_latin: `{total['pages_with_latin']}`",
        f"- latin_hints: `{total['latin_hint_count']}`",
        f"- matched: `{total['matched_count']}`",
        f"- exact: `{total['exact_match_count']}`",
        f"- alignment_fallback: `{total['alignment_fallback_count']}`",
        f"- unmatched: `{total['unmatched_count']}`",
        f"- eng20_ok: `{total['eng20_ok_count']}/{total['eng20_ran_count']}`",
        f"- eng20_norm_ok: `{total.get('eng20_norm_ok_count', 0)}/{total['eng20_ran_count']}`",
        f"- eng20_contains: `{total.get('eng20_contains_count', 0)}/{total['eng20_ran_count']}`",
        "",
        "## Pages",
        "",
        "| page | hints | exact | align | unmatched | eng20 | seconds | statuses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for page in payload["pages"]:
        statuses = ", ".join(f"{key}:{value}" for key, value in sorted((page.get("binding_statuses") or {}).items()))
        lines.append(
            f"| {page['page_id']} | "
            f"{page.get('latin_hint_count', 0)} | "
            f"{page.get('exact_match_count', 0)} | "
            f"{page.get('alignment_fallback_count', 0)} | "
            f"{page.get('unmatched_count', 0)} | "
            f"{page.get('eng20_ok_count', 0)}/{page.get('eng20_ran_count', 0)} | "
            f"{float(page.get('hanwang_seconds', 0)):.1f} | "
            f"{statuses} |"
        )
    lines.extend(["", "## Unmatched Or Eng20 Mismatch", ""])
    for page in payload["pages"]:
        if not page.get("unmatched") and not page.get("eng20_mismatches") and not page.get("error"):
            continue
        lines.append(f"### {page['page_id']}")
        if page.get("error"):
            lines.append(f"- error: `{page['error']}`")
        for item in page.get("unmatched") or []:
            lines.append(f"- unmatched block=`{item['block_idx']}` text=`{item['text']}` status=`{item['status']}`")
        for item in page.get("eng20_mismatches") or []:
            lines.append(
                f"- eng20 mismatch block=`{item['block_idx']}` text=`{item['text']}` "
                f"eng20=`{item['eng20_text']}` crop=`{item['crop']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-dir", type=Path, default=REPO_ROOT / "file/244771纵校")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch")
    parser.add_argument("--pages", nargs="*", help="Optional page ids, for example 120166 120194.")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--run-eng20", action="store_true")
    parser.add_argument("--include-no-latin", action="store_true")
    args = parser.parse_args()

    layouts = sorted(args.layout_dir.glob("*.layout-api.json"))
    if args.pages:
        wanted = set(args.pages)
        layouts = [path for path in layouts if _layout_id(path) in wanted]
    if args.max_pages:
        layouts = layouts[:args.max_pages]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    totals = {
        "page_count": 0,
        "pages_with_latin": 0,
        "latin_hint_count": 0,
        "exact_match_count": 0,
        "alignment_fallback_count": 0,
        "matched_count": 0,
        "unmatched_count": 0,
        "eng20_ran_count": 0,
        "eng20_ok_count": 0,
        "eng20_norm_ok_count": 0,
        "eng20_contains_count": 0,
    }

    for layout in layouts:
        page_id = _layout_id(layout)
        page_dir = args.out_dir / page_id
        page_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{page_id}] loading {layout}")
        dispatch: dict[str, Any] = {"summary": {"latin_hint_count": 0, "unit_count": 0}}
        try:
            page, records = _load_layout(layout)
            dispatch = build_dispatch_units(page, records)
            _json_dump(page_dir / "dispatch_units.json", dispatch)
            if not dispatch["summary"]["latin_hint_count"] and not args.include_no_latin:
                summary = _page_summary(page_id, dispatch, None, None)
                pages.append(summary)
                continue

            target_records, before = _target_records(page)
            if not target_records:
                summary = _page_summary(page_id, dispatch, None, None, "no_latin_target_records")
                pages.append(summary)
                continue

            echo = _echo_hanwang(page, target_records, before, page_dir)
            binding = _bind_dispatch(dispatch, echo, page_dir, args.run_eng20)
            summary = _page_summary(page_id, dispatch, echo, binding)
            print(
                f"[{page_id}] hints={summary.get('latin_hint_count', 0)} "
                f"matched={summary.get('matched_count', 0)} "
                f"unmatched={summary.get('unmatched_count', 0)} "
                f"eng20={summary.get('eng20_ok_count', 0)}/{summary.get('eng20_ran_count', 0)}"
            )
            pages.append(summary)
        except Exception as exc:
            pages.append(_page_summary(page_id, dispatch, None, None, f"{type(exc).__name__}: {exc}"))
            print(f"[{page_id}] ERROR {type(exc).__name__}: {exc}")

    for page in pages:
        totals["page_count"] += 1
        if int(page.get("latin_hint_count") or 0) > 0:
            totals["pages_with_latin"] += 1
        for key in (
            "latin_hint_count",
            "exact_match_count",
            "alignment_fallback_count",
            "matched_count",
            "unmatched_count",
            "eng20_ran_count",
            "eng20_ok_count",
            "eng20_norm_ok_count",
            "eng20_contains_count",
        ):
            totals[key] += int(page.get(key) or 0)

    payload = {
        "schema": "latin_recovery_batch.v0",
        "layout_dir": str(args.layout_dir),
        "run_eng20": bool(args.run_eng20),
        "summary": totals,
        "pages": pages,
    }
    _json_dump(args.out_dir / "latin_recovery_batch_summary.json", payload)
    _write_markdown(payload, args.out_dir / "latin_recovery_batch_summary.md")
    print(f"wrote {args.out_dir / 'latin_recovery_batch_summary.json'}")
    print(f"wrote {args.out_dir / 'latin_recovery_batch_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
