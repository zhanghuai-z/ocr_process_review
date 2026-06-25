#!/usr/bin/env python3
"""Summarize Hanwang native probe call-count evidence.

This experiment is intentionally read-only. It does not invoke Hanwang native
probes; it aggregates existing debug artifacts so call-count strategy changes
can be discussed without machine/network/cache noise.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sum_stats(paths: list[Path]) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    for path in paths:
        payload = _load_json(path)
        stats = payload.get("stats") or {}
        page_id = path.parent.name
        page = {
            "page_id": page_id,
            "elapsed_seconds": round(_safe_float(payload.get("elapsed_seconds")), 3),
            "n_groups": _safe_int(stats.get("n_groups")),
            "recog_probe_calls": _safe_int(stats.get("recog_probe_calls")),
            "recog_batch_chunks": _safe_int(stats.get("recog_batch_chunks")),
            "recog_batch_guarded_chunks": _safe_int(stats.get("recog_batch_guarded_chunks")),
            "recog_batch_failures": _safe_int(stats.get("recog_batch_failures")),
            "seg_seconds": round(_safe_float(stats.get("seg_seconds")), 3),
            "recog_seconds": round(_safe_float(stats.get("recog_seconds")), 3),
        }
        pages.append(page)
        for key in (
            "n_groups",
            "recog_probe_calls",
            "recog_batch_chunks",
            "recog_batch_guarded_chunks",
            "recog_batch_failures",
        ):
            totals[key] += page[key]
        for key in ("elapsed_seconds", "seg_seconds", "recog_seconds"):
            seconds[key] += page[key]

    groups = int(totals["n_groups"])
    actual_recog = int(totals["recog_probe_calls"])
    ideal_6 = sum(math.ceil(max(0, page["n_groups"]) / 6) for page in pages)
    ideal_4 = sum(math.ceil(max(0, page["n_groups"]) / 4) for page in pages)
    return {
        "page_count": len(pages),
        "totals": {
            **dict(totals),
            **{key: round(value, 3) for key, value in seconds.items()},
            "ideal_stable_batch_calls_max6": ideal_6,
            "ideal_stable_batch_calls_max4": ideal_4,
            "actual_calls_per_group": round(actual_recog / groups, 3) if groups else 0,
            "guarded_chunk_ratio": round(totals["recog_batch_guarded_chunks"] / max(1, totals["recog_batch_chunks"]), 3),
        },
        "pages": pages,
    }


def _native_direct_summary(root: Path) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/hanwang_native_direct.json")):
        payload = _load_json(path)
        item: dict[str, Any] = {"page_id": str(payload.get("page_id") or path.parent.name)}
        for result in payload.get("results") or []:
            mode = str(result.get("mode") or "")
            if mode == "ppvl_micro_recblock_current":
                stats = result.get("stats") or {}
                item["ppvl_current"] = {
                    "elapsed_seconds": _safe_float(result.get("elapsed_seconds")),
                    "blocks": _safe_int(result.get("blocks")),
                    "lines": _safe_int(result.get("lines")),
                    "recog_probe_calls": _safe_int(stats.get("recog_probe_calls")),
                    "latin_engcut_probe_calls": _safe_int(stats.get("latin_engcut_probe_calls")),
                    "n_groups": _safe_int(stats.get("n_groups")),
                    "seg_seconds": _safe_float(stats.get("seg_seconds")),
                    "recog_seconds": _safe_float(stats.get("recog_seconds")),
                }
            elif mode == "direct_full_page_linecut":
                item["direct_full_page"] = {
                    "elapsed_seconds": _safe_float(result.get("elapsed_seconds")),
                    "lines": _safe_int(result.get("lines")),
                    "chars": _safe_int(result.get("chars")),
                    "risk": "full-page linecut has known AccessViolation/fallback risk",
                }
            elif mode == "native_docseg_then_ocr":
                item["native_docseg_then_ocr"] = {
                    "elapsed_seconds": _safe_float(result.get("elapsed_seconds")),
                    "layout_seconds": _safe_float(result.get("layout_seconds")),
                    "ocr_seconds": _safe_float(result.get("ocr_seconds")),
                    "blocks": _safe_int(result.get("blocks")),
                    "lines": _safe_int(result.get("lines")),
                    "risk": "uses Hanwang layout semantics, not Paddle/VL structure truth",
                }
        pages.append(item)
    return {"page_count": len(pages), "pages": pages}


def _engcut_summary(benchmark_root: Path, binding_summary_path: Path, alignment_summary_path: Path) -> dict[str, Any]:
    def load_benchmark(name: str) -> dict[str, Any]:
        path = benchmark_root / name / "benchmark.json"
        if not path.exists():
            return {}
        payload = _load_json(path)
        return dict(payload.get("summary") or {})

    route_w4 = load_benchmark("route_with_unresolved_block_lines_w4")
    route_w8 = load_benchmark("route_with_unresolved_block_lines_w8")
    route_w12 = load_benchmark("route_with_unresolved_block_lines_w12")
    route_w16 = load_benchmark("route_with_unresolved_block_lines_w16")
    route_w24 = load_benchmark("route_with_unresolved_block_lines_w24")
    old_same_pages = load_benchmark("old_w4/pages_120167_120168_120171_120172_120173_120174_120175_120183_120184_120185_120186_120187")
    route_same_pages = load_benchmark("route_w4/pages_120167_120168_120171_120172_120173_120174_120175_120183_120184_120185_120186_120187")
    binding = _load_json(binding_summary_path) if binding_summary_path.exists() else {}
    alignment = _load_json(alignment_summary_path) if alignment_summary_path.exists() else {}
    totals = alignment.get("totals") or {}

    old_calls = _safe_int(old_same_pages.get("task_count"))
    route_calls = _safe_int(route_same_pages.get("task_count"))
    old_time = _safe_float(old_same_pages.get("elapsed_seconds"))
    route_time = _safe_float(route_same_pages.get("elapsed_seconds"))
    return {
        "alignment_totals": totals,
        "same_page_comparison": {
            "old_full_line_calls": old_calls,
            "route_token_calls": route_calls,
            "call_reduction": old_calls - route_calls,
            "call_reduction_ratio": round((old_calls - route_calls) / old_calls, 3) if old_calls else 0,
            "old_full_line_seconds": round(old_time, 3),
            "route_token_seconds": round(route_time, 3),
            "time_reduction_seconds": round(old_time - route_time, 3),
            "time_reduction_ratio": round((old_time - route_time) / old_time, 3) if old_time else 0,
        },
        "route_token_workers": [
            {
                "workers": _safe_int(item.get("workers")),
                "task_count": _safe_int(item.get("task_count")),
                "elapsed_seconds": round(_safe_float(item.get("elapsed_seconds")), 3),
                "avg_task_elapsed_seconds": round(_safe_float(item.get("avg_task_elapsed_seconds")), 3),
                "errors": len(item.get("errors") or []),
            }
            for item in (route_w4, route_w8, route_w12, route_w16, route_w24)
            if item
        ],
        "binding_totals": binding.get("totals") or {},
    }


def _timing_log_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    by_probe: dict[str, Counter[str]] = {}
    seconds_by_probe: dict[str, Counter[str]] = {}
    records = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        probe = str(payload.get("probe") or "unknown")
        by_probe.setdefault(probe, Counter())
        seconds_by_probe.setdefault(probe, Counter())
        records += 1
        by_probe[probe]["calls"] += 1
        if payload.get("cache_hit"):
            by_probe[probe]["cache_hits"] += 1
        if payload.get("ok"):
            by_probe[probe]["ok"] += 1
        if payload.get("error"):
            by_probe[probe]["errors"] += 1
        for key, value in payload.items():
            if key.endswith("_seconds"):
                seconds_by_probe[probe][key] += _safe_float(value)
    return {
        "path": str(path),
        "records": records,
        "by_probe": {
            probe: {
                **dict(counts),
                "seconds": {key: round(value, 3) for key, value in seconds_by_probe.get(probe, {}).items()},
            }
            for probe, counts in sorted(by_probe.items())
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    linecut = report["linecut_recog"]
    linecut_totals = linecut["totals"]
    engcut = report["engcut"]
    direct = report["native_direct_pages"]
    lines = [
        "# Hanwang Native Probe Call Count Experiment",
        "",
        "## Scope",
        "",
        "- `linecut_segimg_probe.exe`: page/text-route segmentation.",
        "- `linecut_recogimg_probe.exe`: Chinese/main OCR recognition.",
        "- `eng20_probe.exe`: Latin/digit geometry enhancement.",
        "- This report aggregates existing debug artifacts and does not invoke native OCR.",
        "",
        "## LineCut Recog Calls",
        "",
        f"- Sample pages: `{linecut['page_count']}`.",
        f"- SegImg groups: `{linecut_totals['n_groups']}`.",
        f"- Actual `linecut_recogimg_probe.exe` calls: `{linecut_totals['recog_probe_calls']}`.",
        f"- Actual calls per group: `{linecut_totals['actual_calls_per_group']}`.",
        f"- Batch chunks: `{linecut_totals['recog_batch_chunks']}`, guarded chunks: `{linecut_totals['recog_batch_guarded_chunks']}` "
        f"({linecut_totals['guarded_chunk_ratio'] * 100:.1f}%).",
        f"- Ideal stable max-6 batch lower bound: `{linecut_totals['ideal_stable_batch_calls_max6']}` calls.",
        f"- Ideal stable max-4 batch lower bound: `{linecut_totals['ideal_stable_batch_calls_max4']}` calls.",
        "",
        "Interpretation: current linecut recognition is effectively one native call per SegImg group. "
        "The theoretical batch reduction is large, but existing wide-line/collage batch evidence is guarded or unstable, so this is not a safe product path without a native bridge/probe redesign.",
        "",
        "## Direct Page Evidence",
        "",
        "| page | current PPVL time | current recog calls | current EngCut calls | full-page time | docseg time | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for page in direct.get("pages") or []:
        cur = page.get("ppvl_current") or {}
        full = page.get("direct_full_page") or {}
        docseg = page.get("native_docseg_then_ocr") or {}
        lines.append(
            f"| {page['page_id']} | {cur.get('elapsed_seconds', 0):.3f}s | "
            f"{cur.get('recog_probe_calls', 0)} | {cur.get('latin_engcut_probe_calls', 0)} | "
            f"{full.get('elapsed_seconds', 0):.3f}s | {docseg.get('elapsed_seconds', 0):.3f}s | "
            "full-page loses routing; docseg is not Paddle/VL truth |"
        )
    lines.extend([
        "",
        "## EngCut Calls",
        "",
    ])
    same = engcut["same_page_comparison"]
    lines.extend([
        f"- Same-page old full-line EngCut: `{same['old_full_line_calls']}` calls, `{same['old_full_line_seconds']}`s.",
        f"- Same-page route-token EngCut: `{same['route_token_calls']}` calls, `{same['route_token_seconds']}`s.",
        f"- Call reduction: `{same['call_reduction']}` ({same['call_reduction_ratio'] * 100:.1f}%).",
        f"- Time reduction: `{same['time_reduction_seconds']}`s ({same['time_reduction_ratio'] * 100:.1f}%).",
        f"- Binding success: `{engcut['binding_totals'].get('success_rate', '')}` "
        f"({engcut['binding_totals'].get('engcut_exact', 0)} exact, "
        f"{engcut['binding_totals'].get('engcut_variant', 0)} variant, "
        f"{engcut['binding_totals'].get('unresolved', 0)} unresolved).",
        "",
        "| workers | calls | elapsed | avg task | errors |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in engcut["route_token_workers"]:
        lines.append(
            f"| {item['workers']} | {item['task_count']} | {item['elapsed_seconds']}s | "
            f"{item['avg_task_elapsed_seconds']}s | {item['errors']} |"
        )
    timing = report.get("timing_log") or {}
    if timing:
        lines.extend([
            "",
            "## Timing Log",
            "",
            f"- Source: `{timing.get('path')}`.",
            f"- Records: `{timing.get('records', 0)}`.",
            "",
            "| probe | calls | cache hits | ok | errors | total | subprocess | cache read | image write |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for probe, item in (timing.get("by_probe") or {}).items():
            seconds = item.get("seconds") or {}
            lines.append(
                f"| {probe} | {item.get('calls', 0)} | {item.get('cache_hits', 0)} | "
                f"{item.get('ok', 0)} | {item.get('errors', 0)} | "
                f"{seconds.get('total_seconds', 0)}s | {seconds.get('subprocess_seconds', 0)}s | "
                f"{seconds.get('cache_read_seconds', 0)}s | {seconds.get('temp_image_write_seconds', 0)}s |"
            )
    lines.extend([
        "",
        "## Decision",
        "",
        "1. Short-term valid optimization is EngCut call targeting: keep route-token targeting and worker cap around 12-16 for experiment/high-performance mode.",
        "2. LineCut Recog call reduction requires native-side batching or a resident bridge. Current Python-side collage/multi-recblock batching is not safe enough.",
        "3. Full-page or Hanwang-docseg paths can reduce routing complexity in some pages, but they break the Paddle/VL structure authority needed for formula/table/manual boxes.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linecut-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch_all")
    parser.add_argument("--native-direct-dir", type=Path, default=REPO_ROOT / "debug/hanwang_native_direct")
    parser.add_argument("--benchmark-dir", type=Path, default=REPO_ROOT / "debug/route_token_engcut_benchmark")
    parser.add_argument("--binding-summary", type=Path, default=REPO_ROOT / "debug/route_token_engcut_binding/route_token_engcut_binding_summary.json")
    parser.add_argument("--alignment-summary", type=Path, default=REPO_ROOT / "debug/ppocr_route_token_alignment/ppocr_route_token_alignment_summary.json")
    parser.add_argument("--timing-log", type=Path, default=None, help="Optional HANWANG_NATIVE_TIMING_LOG JSONL from a real app run")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/native_probe_call_count_experiment")
    args = parser.parse_args()

    linecut_paths = sorted(args.linecut_dir.glob("*/hanwang_echo.json"))
    report = {
        "schema": "native_probe_call_count_experiment.v1",
        "inputs": {
            "linecut_dir": str(args.linecut_dir),
            "native_direct_dir": str(args.native_direct_dir),
            "benchmark_dir": str(args.benchmark_dir),
            "binding_summary": str(args.binding_summary),
            "alignment_summary": str(args.alignment_summary),
        },
        "linecut_recog": _sum_stats(linecut_paths),
        "native_direct_pages": _native_direct_summary(args.native_direct_dir),
        "engcut": _engcut_summary(args.benchmark_dir, args.binding_summary, args.alignment_summary),
        "timing_log": _timing_log_summary(args.timing_log),
    }
    _write_json(args.out_dir / "native_probe_call_count_experiment.json", report)
    _write_markdown(report, args.out_dir / "native_probe_call_count_experiment.md")
    print(args.out_dir / "native_probe_call_count_experiment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
