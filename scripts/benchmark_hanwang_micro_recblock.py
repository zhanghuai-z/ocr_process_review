"""Benchmark Hanwang micro-recblock on the native probe path.

Usage:
    PYTHONPATH=. python scripts/benchmark_hanwang_micro_recblock.py page.tif \
        --ppvl-json page.layout-api.json

The PP-VL JSON may be either a raw parsing_res_list array or an object that
contains parsing_res_list. Without it, the script benchmarks one full-page text
block, which is useful for smoke timing but less representative.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines.hanwang.micro_recblock import (
    MAX_RECOG_BATCH_GROUPS,
    run_micro_recblock,
)


def _load_ppvl_blocks(path: Path | None, width: int, height: int) -> list[dict[str, Any]]:
    if path is None:
        return [{
            "block_label": "text",
            "block_bbox": [0, 0, width, height],
            "block_content": "",
        }]

    def extract_records(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("parsing_res_list", "ppvl_parsing_res_list"):
            records = value.get(key)
            if isinstance(records, list):
                return [dict(item) for item in records if isinstance(item, dict)]
        response_records = extract_records(value.get("response"))
        if response_records:
            return response_records
        result = value.get("result")
        if isinstance(result, dict):
            direct = extract_records(result)
            if direct:
                return direct
            layout_results = result.get("layoutParsingResults")
            if isinstance(layout_results, list):
                extracted: list[dict[str, Any]] = []
                for item in layout_results:
                    if isinstance(item, dict):
                        extracted.extend(extract_records(item.get("prunedResult")))
                if extracted:
                    return extracted
        pruned = value.get("prunedResult")
        if isinstance(pruned, dict):
            return extract_records(pruned)
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    records = extract_records(data)
    if records:
        return records
    raise SystemExit(f"Cannot find parsing_res_list in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Hanwang micro-recblock")
    parser.add_argument("image", type=Path)
    parser.add_argument("--ppvl-json", type=Path, default=None)
    parser.add_argument("--no-chars", action="store_true", help="Disable char output for diagnostic comparison only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    height, width = image.shape[:2]
    blocks = _load_ppvl_blocks(args.ppvl_json, width, height)

    events: list[tuple[int, int, str]] = []
    started = time.perf_counter()
    rows, stats = run_micro_recblock(
        image,
        blocks,
        include_chars=not args.no_chars,
        progress_callback=lambda current, total, message: events.append((current, total, message)),
    )
    elapsed = time.perf_counter() - started

    payload = {
        "image": str(args.image),
        "ppvl_json": str(args.ppvl_json) if args.ppvl_json else None,
        "elapsed_seconds": elapsed,
        "batch_limits": {
            "max_groups": MAX_RECOG_BATCH_GROUPS,
        },
        "stats": {
            "n_blocks_total": stats.n_blocks_total,
            "n_blocks_hanwang": stats.n_blocks_hanwang,
            "n_blocks_ppvl": stats.n_blocks_ppvl,
            "n_blocks_fallback": stats.n_blocks_fallback,
            "n_groups": stats.n_groups,
            "seg_seconds": stats.seg_seconds,
            "recog_seconds": stats.recog_seconds,
            "recog_probe_calls": stats.recog_probe_calls,
            "recog_group_failures": stats.recog_group_failures,
            "recog_group_retry_attempts": stats.recog_group_retry_attempts,
            "recog_group_retry_successes": stats.recog_group_retry_successes,
            "recog_group_retry_failures": stats.recog_group_retry_failures,
            "recog_batch_chunks": stats.recog_batch_chunks,
            "recog_batch_failures": stats.recog_batch_failures,
            "recog_batch_disabled": stats.recog_batch_disabled,
            "recog_batch_guarded_chunks": stats.recog_batch_guarded_chunks,
            "recog_max_batch_crop_width": stats.recog_max_batch_crop_width,
            "recog_max_batch_crop_height": stats.recog_max_batch_crop_height,
            "recog_max_batch_crop_pixels": stats.recog_max_batch_crop_pixels,
            "recog_full_page_pixels": stats.recog_full_page_pixels,
            "recog_crop_pixels": stats.recog_crop_pixels,
            "latin_engcut_probe_calls": stats.latin_engcut_probe_calls,
            "latin_engcut_probe_failures": stats.latin_engcut_probe_failures,
            "latin_engcut_exact_tokens": stats.latin_engcut_exact_tokens,
            "latin_engcut_review_tokens": stats.latin_engcut_review_tokens,
            "latin_engcut_disabled": stats.latin_engcut_disabled,
        },
        "rows": [
            {
                "block_idx": row.block_idx,
                "block_label": row.block_label,
                "source": row.source,
                "group_count": row.group_count,
                "text_length": len(row.text),
                "fallback_reason": row.fallback_reason,
            }
            for row in rows
        ],
        "progress_events": events[-20:],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
