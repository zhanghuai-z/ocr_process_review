#!/usr/bin/env python3
"""Probe linecut_recogimg native call shapes.

This experiment invokes ``linecut_recogimg_probe.exe`` directly. It compares:

- per-line crop calls: current safe shape, one native call per line/group
- full-page multi-recblock call: one native call with N page-space recblocks
- vertical/horizontal collage calls: one native call with N crop recblocks

The goal is to test whether native call count can be reduced without relying on
the application cache or the Python OCR pipeline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_route_bboxes(alignment_json: Path) -> list[dict[str, Any]]:
    payload = _load_json(alignment_json)
    routes: list[dict[str, Any]] = []
    for block_routes in (payload.get("routes_by_block") or {}).values():
        for route in block_routes:
            bbox = [int(v) for v in route.get("bbox") or []]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            routes.append({
                "block_idx": int(route.get("block_idx", -1)),
                "route_idx": int(route.get("route_idx", -1)),
                "bbox": bbox,
                "text": str(route.get("ppocr_text") or ""),
            })
    routes.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["block_idx"], item["route_idx"]))
    return routes


def _write_recblocks(path: Path, recblocks: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{l}\t{t}\t{r}\t{b}" for l, t, r, b in recblocks) + "\n",
        encoding="utf-8",
    )


def _extract_counts(payload: dict[str, Any]) -> dict[str, int]:
    line_count = 0
    group_count = 0
    char_count = 0
    for line in payload.get("lines") or []:
        line_count += 1
        for group in line.get("groups") or []:
            group_count += 1
            char_count += len(group.get("chars") or [])
    return {"lines": line_count, "groups": group_count, "chars": char_count}


def _run_probe(
    *,
    image_path: Path,
    recblocks: list[tuple[int, int, int, int]],
    out_json: Path,
    rb_path: Path,
    timeout: float,
    mode: int = 71,
    postprocess: int = 1,
    split_mode: int = 0,
    via_seg: bool = True,
    with_charrcg: bool = True,
) -> dict[str, Any]:
    exe = REPO_ROOT / "resources/hanwang_native/bin/linecut_recogimg_probe.exe"
    bin_dir = exe.parent
    _write_recblocks(rb_path, recblocks)
    cmd = [
        str(exe),
        _win_path(image_path),
        _win_path(out_json),
        _win_path(rb_path),
        str(mode),
        str(postprocess),
        str(split_mode),
    ]
    if via_seg:
        cmd.append("via-seg")
    if with_charrcg:
        cmd.append("with-charrcg")
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(bin_dir),
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -999,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"timeout after {timeout}s",
            "stdout": _decode(exc.stdout or b""),
            "stderr": _decode(exc.stderr or b""),
        }

    payload: dict[str, Any] = {}
    counts = {"lines": 0, "groups": 0, "chars": 0}
    if out_json.exists():
        try:
            payload = _load_json(out_json)
            counts = _extract_counts(payload)
        except Exception as exc:
            payload = {"json_error": str(exc)}
    error = ""
    if proc.returncode != 0:
        error = f"returncode={proc.returncode}"
    if isinstance(payload, dict) and payload.get("error"):
        error = str(payload.get("error"))
    return {
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
        "stdout": _decode(proc.stdout),
        "stderr": _decode(proc.stderr),
        **counts,
        "init": payload.get("init") if isinstance(payload, dict) else None,
        "charrcg_init": payload.get("charrcg_init") if isinstance(payload, dict) else None,
        "charrcg_ran": payload.get("charrcg_ran") if isinstance(payload, dict) else None,
        "mode": mode,
        "postprocess": postprocess,
        "split_mode": split_mode,
        "via_seg": via_seg,
        "with_charrcg": with_charrcg,
        "output": str(out_json),
    }


def _crop(image: np.ndarray, bbox: list[int], pad: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(w, r + pad)
    b = min(h, b + pad)
    return image[t:b, l:r].copy(), (l, t, r, b)


def _vertical_collage(crops: list[np.ndarray], *, gap: int = 8) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    width = max(crop.shape[1] for crop in crops)
    height = sum(crop.shape[0] for crop in crops) + gap * max(0, len(crops) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    recblocks: list[tuple[int, int, int, int]] = []
    y = 0
    for crop in crops:
        h, w = crop.shape[:2]
        canvas[y:y + h, 0:w] = crop
        recblocks.append((0, y, w, y + h))
        y += h + gap
    return canvas, recblocks


def _horizontal_collage(crops: list[np.ndarray], *, gap: int = 12) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    width = sum(crop.shape[1] for crop in crops) + gap * max(0, len(crops) - 1)
    height = max(crop.shape[0] for crop in crops)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    recblocks: list[tuple[int, int, int, int]] = []
    x = 0
    for crop in crops:
        h, w = crop.shape[:2]
        canvas[0:h, x:x + w] = crop
        recblocks.append((x, 0, x + w, h))
        x += w + gap
    return canvas, recblocks


def _run_per_line(
    image: np.ndarray,
    routes: list[dict[str, Any]],
    out_dir: Path,
    *,
    size: int,
    timeout: float,
) -> dict[str, Any]:
    case_dir = out_dir / f"per_line_{size}"
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for idx, route in enumerate(routes[:size]):
        crop, _ = _crop(image, route["bbox"], pad=2)
        crop_path = case_dir / f"crop_{idx:03d}.png"
        cv2.imwrite(str(crop_path), crop)
        h, w = crop.shape[:2]
        result = _run_probe(
            image_path=crop_path,
            recblocks=[(0, 0, w, h)],
            out_json=case_dir / f"crop_{idx:03d}.json",
            rb_path=case_dir / f"crop_{idx:03d}.rb.tsv",
            timeout=timeout,
        )
        results.append(result)
    elapsed = time.perf_counter() - started
    return {
        "shape": "per_line_crop",
        "requested_recblocks": size,
        "native_calls": size,
        "elapsed_seconds": round(elapsed, 3),
        "errors": [item for item in results if item.get("error")],
        "lines": sum(int(item.get("lines") or 0) for item in results),
        "groups": sum(int(item.get("groups") or 0) for item in results),
        "chars": sum(int(item.get("chars") or 0) for item in results),
        "results": results,
    }


def _run_batch_list_case(
    image: np.ndarray,
    routes: list[dict[str, Any]],
    out_dir: Path,
    *,
    size: int,
    timeout: float,
) -> dict[str, Any]:
    case_dir = out_dir / f"batch_list_{size}"
    case_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = case_dir / "tasks.tsv"
    summary_path = case_dir / "summary.json"
    task_lines: list[str] = []
    for idx, route in enumerate(routes[:size]):
        crop, _ = _crop(image, route["bbox"], pad=2)
        crop_path = case_dir / f"crop_{idx:03d}.png"
        out_json = case_dir / f"crop_{idx:03d}.json"
        cv2.imwrite(str(crop_path), crop)
        task_lines.append(f"{_win_path(crop_path)}\t{_win_path(out_json)}")
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    exe = REPO_ROOT / "resources/hanwang_native/bin/linecut_recog_batch_probe.exe"
    cmd = [
        str(exe),
        "--batch-list",
        _win_path(tasks_path),
        _win_path(summary_path),
        "71",
        "1",
        "0",
        "via-seg",
        "with-charrcg",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(exe.parent),
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        return {
            "shape": "batch_list_single_process",
            "requested_recblocks": size,
            "native_calls": 1,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "returncode": -999,
            "lines": 0,
            "groups": 0,
            "chars": 0,
            "error": f"timeout after {timeout}s",
            "stdout": _decode(exc.stdout or b""),
            "stderr": _decode(exc.stderr or b""),
        }
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
        except Exception as exc:
            summary = {"json_error": str(exc)}
    tasks = summary.get("tasks") if isinstance(summary, dict) else []
    if not isinstance(tasks, list):
        tasks = []
    errors = [
        item for item in tasks
        if isinstance(item, dict) and not item.get("ok")
    ]
    return {
        "shape": "batch_list_single_process",
        "requested_recblocks": size,
        "native_calls": 1,
        "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
        "lines": sum(int(item.get("lines") or 0) for item in tasks if isinstance(item, dict)),
        "groups": sum(int(item.get("lines") or 0) for item in tasks if isinstance(item, dict)),
        "chars": sum(int(item.get("chars") or 0) for item in tasks if isinstance(item, dict)),
        "error": "" if proc.returncode == 0 and not errors else f"returncode={proc.returncode}; task_errors={len(errors)}",
        "stdout": _decode(proc.stdout),
        "stderr": _decode(proc.stderr),
        "summary": str(summary_path),
    }


def _run_single_call_case(
    image_path: Path,
    image: np.ndarray,
    routes: list[dict[str, Any]],
    out_dir: Path,
    *,
    shape: str,
    size: int,
    timeout: float,
) -> dict[str, Any]:
    case_dir = out_dir / f"{shape}_{size}"
    case_dir.mkdir(parents=True, exist_ok=True)
    selected = routes[:size]
    if shape == "full_page_multirb":
        recblocks = [tuple(route["bbox"]) for route in selected]
        run_image = image_path
    else:
        crops = [_crop(image, route["bbox"], pad=2)[0] for route in selected]
        if shape == "vertical_collage":
            collage, recblocks = _vertical_collage(crops)
        elif shape == "horizontal_collage":
            collage, recblocks = _horizontal_collage(crops)
        else:
            raise ValueError(shape)
        run_image = case_dir / "collage.png"
        cv2.imwrite(str(run_image), collage)

    result = _run_probe(
        image_path=run_image,
        recblocks=recblocks,
        out_json=case_dir / "result.json",
        rb_path=case_dir / "recblocks.tsv",
        timeout=timeout,
    )
    return {
        "shape": shape,
        "requested_recblocks": size,
        "native_calls": 1,
        "recblocks": [list(item) for item in recblocks],
        **result,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# LineCut Recog Probe Shape Experiment",
        "",
        f"- Page: `{payload['page_id']}`",
        f"- Image: `{payload['image']}`",
        f"- Route source: `{payload['alignment_json']}`",
        "",
        "| shape | recblocks | native calls | elapsed | rc | lines | groups | chars | errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload["cases"]:
        errors = item.get("errors")
        error_text = ""
        if isinstance(errors, list):
            error_text = str(len(errors))
        else:
            error_text = str(item.get("error") or "")
        lines.append(
            f"| {item['shape']} | {item['requested_recblocks']} | {item['native_calls']} | "
            f"{item.get('elapsed_seconds', 0)}s | {item.get('returncode', '')} | "
            f"{item.get('lines', 0)} | {item.get('groups', 0)} | {item.get('chars', 0)} | "
            f"{error_text} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `per_line_crop` is the current safe call shape: one native call per line/group.",
        "- `batch_list_single_process` is the new experimental wrapper: one process loops over N single-recblock calls.",
        "- `full_page_multirb` tests whether the native probe can process several page-space recblocks in one call.",
        "- `vertical_collage` and `horizontal_collage` test whether batching crops into one image is stable.",
    ])
    variant_cases = payload.get("variant_cases") or []
    if variant_cases:
        lines.extend([
            "",
            "## Variant Sweep",
            "",
            "| shape | recblocks | via-seg | charrcg | postprocess | split | elapsed | rc | lines | chars | error |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in variant_cases:
            lines.append(
                f"| {item['shape']} | {item['requested_recblocks']} | {item.get('via_seg')} | "
                f"{item.get('with_charrcg')} | {item.get('postprocess')} | {item.get('split_mode')} | "
                f"{item.get('elapsed_seconds', 0)}s | {item.get('returncode', '')} | "
                f"{item.get('lines', 0)} | {item.get('chars', 0)} | {item.get('error', '')} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_variant_sweep(
    image_path: Path,
    image: np.ndarray,
    routes: list[dict[str, Any]],
    out_dir: Path,
    *,
    size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    variants = [
        {"name": "via_charrcg", "via_seg": True, "with_charrcg": True, "postprocess": 1, "split_mode": 0},
        {"name": "via_no_charrcg", "via_seg": True, "with_charrcg": False, "postprocess": 1, "split_mode": 0},
        {"name": "no_via_charrcg", "via_seg": False, "with_charrcg": True, "postprocess": 1, "split_mode": 0},
        {"name": "no_via_no_charrcg", "via_seg": False, "with_charrcg": False, "postprocess": 1, "split_mode": 0},
        {"name": "via_post0", "via_seg": True, "with_charrcg": True, "postprocess": 0, "split_mode": 0},
        {"name": "via_split1", "via_seg": True, "with_charrcg": True, "postprocess": 1, "split_mode": 1},
    ]
    results: list[dict[str, Any]] = []
    for shape in ("full_page_multirb", "vertical_collage", "horizontal_collage"):
        for variant in variants:
            case_dir = out_dir / f"variant_{shape}_{size}_{variant['name']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            selected = routes[:size]
            if shape == "full_page_multirb":
                recblocks = [tuple(route["bbox"]) for route in selected]
                run_image = image_path
            else:
                crops = [_crop(image, route["bbox"], pad=2)[0] for route in selected]
                if shape == "vertical_collage":
                    collage, recblocks = _vertical_collage(crops)
                else:
                    collage, recblocks = _horizontal_collage(crops)
                run_image = case_dir / "collage.png"
                cv2.imwrite(str(run_image), collage)
            result = _run_probe(
                image_path=run_image,
                recblocks=recblocks,
                out_json=case_dir / "result.json",
                rb_path=case_dir / "recblocks.tsv",
                timeout=timeout,
                via_seg=bool(variant["via_seg"]),
                with_charrcg=bool(variant["with_charrcg"]),
                postprocess=int(variant["postprocess"]),
                split_mode=int(variant["split_mode"]),
            )
            results.append({
                "shape": shape,
                "requested_recblocks": size,
                "variant": variant["name"],
                **result,
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", default="120186")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--alignment-json", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--variant-sweep", action="store_true")
    parser.add_argument("--variant-size", type=int, default=2)
    parser.add_argument("--batch-list", action="store_true", help="Run experimental single-process batch wrapper if available")
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Skip known-unsafe multi-recblock/collage cases and compare only per-line calls vs batch-list",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/linecut_recog_probe_shapes")
    args = parser.parse_args()

    image_path = args.image or REPO_ROOT / "file" / "244771纵校" / f"{args.page_id}.tif"
    alignment_json = args.alignment_json or REPO_ROOT / "debug/ppocr_route_token_alignment" / args.page_id / "ppocr_route_token_alignment.json"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    routes = _load_route_bboxes(alignment_json)
    if not routes:
        raise SystemExit(f"No routes found: {alignment_json}")

    out_dir = args.out_dir / args.page_id
    out_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, Any]] = []
    for size in args.sizes:
        if size > len(routes):
            continue
        case_results.append(_run_per_line(image, routes, out_dir, size=size, timeout=args.timeout))
        if args.batch_list:
            case_results.append(_run_batch_list_case(image, routes, out_dir, size=size, timeout=args.timeout))
        if not args.batch_only:
            for shape in ("full_page_multirb", "vertical_collage", "horizontal_collage"):
                case_results.append(_run_single_call_case(
                    image_path,
                    image,
                    routes,
                    out_dir,
                    shape=shape,
                    size=size,
                    timeout=args.timeout,
                ))
    variant_cases = (
        _run_variant_sweep(
            image_path,
            image,
            routes,
            out_dir,
            size=args.variant_size,
            timeout=args.timeout,
        )
        if args.variant_sweep
        else []
    )
    payload = {
        "schema": "linecut_recog_probe_shapes.v1",
        "page_id": args.page_id,
        "image": str(image_path),
        "alignment_json": str(alignment_json),
        "route_count": len(routes),
        "sizes": args.sizes,
        "cases": case_results,
        "variant_cases": variant_cases,
    }
    _write_json(out_dir / "linecut_recog_probe_shapes.json", payload)
    _write_markdown(payload, out_dir / "linecut_recog_probe_shapes.md")
    print(out_dir / "linecut_recog_probe_shapes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
