#!/usr/bin/env python3
"""Benchmark EngCut calls for route-token line crops.

The script is read-only for application data. It consumes
``debug/ppocr_route_token_alignment`` and optionally the old
``debug/engcut_line_binding_batch_v2`` artifacts to compare the new
route-token call set against the previous full-line EngCut call set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Task:
    source: str
    page_id: str
    task_id: str
    bbox: tuple[int, int, int, int]
    image_path: Path
    old_crop_path: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _win_path(path: Path) -> str:
    text = path.resolve().as_posix()
    for drive in ("d", "e", "c"):
        prefix = f"/mnt/{drive}/"
        if text.startswith(prefix):
            text = f"{drive.upper()}:/" + text[len(prefix):]
            break
    return text.replace("/", "\\")


def _decode(value: bytes) -> str:
    for encoding in ("utf-8", "gbk", "cp936"):
        try:
            return value.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace").strip()


def _extract_char_count(payload: dict[str, Any]) -> int:
    count = 0
    for line in payload.get("lines") or []:
        for group in line.get("groups") or []:
            count += len(group.get("chars") or [])
    return count


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _route_tasks(
    alignment_dir: Path,
    *,
    pages: list[str] | None,
    include_unresolved_block_lines: bool,
) -> list[Task]:
    summary = _load_json(alignment_dir / "ppocr_route_token_alignment_summary.json")
    page_ids = pages or [str(page["page_id"]) for page in summary.get("pages") or []]
    tasks: dict[tuple[str, int, int], Task] = {}
    for page_id in page_ids:
        path = alignment_dir / page_id / "ppocr_route_token_alignment.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        image_path = Path(str(payload["source_image"]))
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path

        wanted: set[tuple[int, int]] = set()
        for binding in payload.get("bindings") or []:
            status = str(binding.get("route_status") or "")
            if status in {"ppocr_route_exact", "ppocr_route_variant", "ppocr_route_cross_line_variant"}:
                route_indices = binding.get("route_indices")
                if isinstance(route_indices, list) and route_indices:
                    for route_idx in route_indices:
                        wanted.add((int(binding.get("block_idx", -1)), int(route_idx)))
                elif binding.get("route_idx") is not None:
                    wanted.add((int(binding.get("block_idx", -1)), int(binding.get("route_idx", -1))))

        if include_unresolved_block_lines:
            unresolved_blocks = {
                int(item.get("block_idx", -1))
                for item in payload.get("unresolved") or []
                if item.get("block_idx") is not None
            }
            for block_idx in unresolved_blocks:
                for route in (payload.get("routes_by_block") or {}).get(str(block_idx), []):
                    wanted.add((block_idx, int(route.get("route_idx", -1))))

        routes_by_block = payload.get("routes_by_block") or {}
        for block_idx, route_idx in wanted:
            block_routes = routes_by_block.get(str(block_idx), [])
            route = next((item for item in block_routes if int(item.get("route_idx", -999)) == route_idx), None)
            if not route:
                continue
            bbox = tuple(int(v) for v in route.get("bbox") or [0, 0, 0, 0])
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            key = (page_id, block_idx, route_idx)
            tasks[key] = Task(
                source="route_token",
                page_id=page_id,
                task_id=f"b{block_idx:03d}_r{route_idx:03d}",
                bbox=bbox,  # type: ignore[arg-type]
                image_path=image_path,
            )
    return sorted(tasks.values(), key=lambda item: (item.page_id, item.task_id))


def _old_full_line_tasks(old_dir: Path, pages: list[str] | None) -> list[Task]:
    page_dirs = sorted(path for path in old_dir.iterdir() if path.is_dir())
    if pages:
        wanted = set(pages)
        page_dirs = [path for path in page_dirs if path.name in wanted]
    tasks: list[Task] = []
    for page_dir in page_dirs:
        payload_path = page_dir / "engcut_line_binding.json"
        if not payload_path.exists():
            continue
        payload = _load_json(payload_path)
        source_image = Path(str(payload["source_image"]))
        if not source_image.is_absolute():
            source_image = REPO_ROOT / source_image
        for index, line in enumerate(payload.get("lines") or []):
            crop_rel = str(line.get("crop") or "")
            crop_path = page_dir / crop_rel if crop_rel else None
            bbox = tuple(int(v) for v in line.get("bbox") or [0, 0, 0, 0])
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if crop_path is None or not crop_path.exists():
                continue
            tasks.append(Task(
                source="old_full_line",
                page_id=page_dir.name,
                task_id=f"old_{index:03d}",
                bbox=bbox,  # type: ignore[arg-type]
                image_path=source_image,
                old_crop_path=crop_path,
            ))
    return tasks


def _prepare_crop(task: Task, out_dir: Path) -> Path:
    crop_dir = out_dir / "crops" / task.page_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{_safe_name(task.task_id)}.png"
    if task.old_crop_path is not None:
        return task.old_crop_path

    image = cv2.imread(str(task.image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {task.image_path}")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = task.bbox
    x1 = max(0, min(x1, width))
    x2 = max(x1, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(y1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop bbox: {task.bbox}")
    cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
    return crop_path


def _run_task(task: Task, out_dir: Path, *, force: bool, timeout: float) -> dict[str, Any]:
    result_dir = out_dir / "results" / task.page_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{_safe_name(task.task_id)}.eng20.json"
    crop_path = _prepare_crop(task, out_dir)
    if result_path.exists() and not force:
        payload = _load_json(result_path)
        return {
            "page_id": task.page_id,
            "task_id": task.task_id,
            "source": task.source,
            "bbox": list(task.bbox),
            "elapsed_seconds": 0.0,
            "reused": True,
            "returncode": 0,
            "char_count": _extract_char_count(payload),
            "error": "",
        }

    exe = REPO_ROOT / "resources/hanwang_native/bin/eng20_probe.exe"
    cmd = [
        str(exe),
        _win_path(crop_path),
        _win_path(result_path),
        "__missing_rb.tsv",
        "recogline_engstr",
        "packed",
        "tbrl",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {}
    if result_path.exists():
        payload = _load_json(result_path)
    return {
        "page_id": task.page_id,
        "task_id": task.task_id,
        "source": task.source,
        "bbox": list(task.bbox),
        "elapsed_seconds": elapsed,
        "reused": False,
        "returncode": proc.returncode,
        "char_count": _extract_char_count(payload),
        "stdout": _decode(proc.stdout),
        "stderr": _decode(proc.stderr),
        "error": "" if proc.returncode == 0 else f"returncode={proc.returncode}",
    }


def _benchmark(tasks: list[Task], out_dir: Path, *, workers: int, force: bool, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_run_task, task, out_dir, force=force, timeout=timeout) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    errors = [item for item in results if item.get("error")]
    return {
        "task_count": len(tasks),
        "workers": workers,
        "elapsed_seconds": elapsed,
        "avg_task_elapsed_seconds": (
            sum(float(item.get("elapsed_seconds") or 0.0) for item in results) / max(1, len(results))
        ),
        "char_count": sum(int(item.get("char_count") or 0) for item in results),
        "errors": errors,
        "results": sorted(results, key=lambda item: (str(item["page_id"]), str(item["task_id"]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=REPO_ROOT / "debug/ppocr_route_token_alignment")
    parser.add_argument("--old-dir", type=Path, default=REPO_ROOT / "debug/engcut_line_binding_batch_v2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/route_token_engcut_benchmark")
    parser.add_argument("--mode", choices=["route", "route_with_unresolved_block_lines", "old"], default="route")
    parser.add_argument("--pages", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.mode == "old":
        tasks = _old_full_line_tasks(args.old_dir, args.pages)
    else:
        tasks = _route_tasks(
            args.alignment_dir,
            pages=args.pages,
            include_unresolved_block_lines=args.mode == "route_with_unresolved_block_lines",
        )

    run_dir = args.out_dir / f"{args.mode}_w{max(1,args.workers)}"
    if args.pages:
        run_dir = run_dir / ("pages_" + "_".join(args.pages))
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "route_token_engcut_benchmark.v0",
        "mode": args.mode,
        "pages": args.pages,
        "force": bool(args.force),
        "summary": _benchmark(tasks, run_dir, workers=max(1, args.workers), force=bool(args.force), timeout=args.timeout),
    }
    _write_json(run_dir / "benchmark.json", payload)
    compact = dict(payload["summary"])
    compact.pop("results", None)
    print(json.dumps({"mode": args.mode, **compact}, ensure_ascii=False, indent=2))
    print(run_dir / "benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
