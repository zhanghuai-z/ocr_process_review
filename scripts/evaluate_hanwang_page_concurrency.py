"""Evaluate bounded page-level concurrency for Hanwang micro-recblock.

This is an explicit benchmark entrypoint only; the application default remains
serial because the native probe has process-wide stability limits and expensive
subprocess/temp-file pressure.

Usage:
    PYTHONPATH=. python scripts/evaluate_hanwang_page_concurrency.py page1.tif page2.tif \
        --ppvl-jsons page1.layout-api.json page2.layout-api.json --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_hanwang_micro_recblock import _load_ppvl_blocks
from app.engines.hanwang.micro_recblock import (
    MAX_RECOG_BATCH_GROUPS,
    run_micro_recblock,
)


def _run_page(image_path: Path, ppvl_json: Path | None, include_chars: bool) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    blocks = _load_ppvl_blocks(ppvl_json, width, height)
    events: list[tuple[int, int, str]] = []
    started = time.perf_counter()
    rows, stats = run_micro_recblock(
        image,
        blocks,
        include_chars=include_chars,
        progress_callback=lambda current, total, message: events.append((current, total, message)),
    )
    return {
        "image": str(image_path),
        "ppvl_json": str(ppvl_json) if ppvl_json else None,
        "elapsed_seconds": time.perf_counter() - started,
        "stats": asdict(stats),
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


def _ppvl_paths(values: list[Path] | None, n_images: int) -> list[Path | None]:
    if not values:
        return [None] * n_images
    if len(values) != n_images:
        raise SystemExit("--ppvl-jsons must have the same count as images")
    return list(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded page-level concurrency for Hanwang micro-recblock"
    )
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--ppvl-jsons", type=Path, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1, help="Bounded worker count; default preserves serial evaluation")
    parser.add_argument("--no-chars", action="store_true", help="Disable char output for diagnostic comparison only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    ppvl_jsons = _ppvl_paths(args.ppvl_jsons, len(args.images))

    started = time.perf_counter()
    page_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_page, image_path, ppvl_json, not args.no_chars)
            for image_path, ppvl_json in zip(args.images, ppvl_jsons)
        ]
        for future in as_completed(futures):
            try:
                page_results.append(future.result())
            except Exception as exc:
                errors.append({"error": str(exc)})

    page_results.sort(key=lambda item: item["image"])
    total_elapsed = time.perf_counter() - started
    payload = {
        "mode": "hanwang_page_concurrency_evaluation",
        "default_runtime_changed": False,
        "workers": args.workers,
        "total_elapsed_seconds": total_elapsed,
        "batch_limits": {
            "max_groups": MAX_RECOG_BATCH_GROUPS,
        },
        "risk_notes": [
            "Native linecut_recogimg_probe.exe has shown AccessViolation risk when multiple recblocks are passed to one Recog call.",
            "Current batch-list mode keeps one safe recblock per crop and batches only at the wrapper-process level.",
            "Concurrent pages multiply subprocess and temp-file pressure.",
            "Process-wide batch disable state can make per-page timings non-independent.",
            "UI progress/writeback ordering must remain page-keyed before enabling runtime concurrency.",
        ],
        "pages": page_results,
        "errors": errors,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
