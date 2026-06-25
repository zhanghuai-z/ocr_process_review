#!/usr/bin/env python3
"""Probe Eng20/engstr segmentation on pure crops and mixed line crops."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2


DEFAULT_SAMPLES = [
    {
        "name": "pure_specialization_120194",
        "kind": "pure_latin_crop",
        "expected": "Specialization",
        "path": "debug/latin_recovery_batch_prose_all_v2/120194/crops/001_b002_Specialization_1263_564_1540_622.png",
    },
    {
        "name": "pure_pevc_120194",
        "kind": "pure_latin_crop",
        "expected": "PE/VC",
        "path": "debug/latin_recovery_batch_prose_all_v2/120194/crops/017_b010_PE_VC_357_2234_496_2281.png",
    },
    {
        "name": "mixed_line_specialization_120194",
        "kind": "mixed_line_crop",
        "expected": "Specialization SPE",
        "path": "debug/120194_latin_hanwang_with_footnote/line_row01_01_272_552_2145_628.png",
    },
    {
        "name": "mixed_line_tfp_li_lyv_120194",
        "kind": "mixed_line_crop",
        "expected": "TFP Li Lyv",
        "path": "debug/120194_latin_hanwang_with_footnote/line_row02_06_273_1670_2146_1744.png",
    },
    {
        "name": "mixed_line_pevc_tfp_120194",
        "kind": "mixed_line_crop",
        "expected": "PE/VC TFP",
        "path": "debug/120194_latin_hanwang_with_footnote/line_row03_02_274_2217_2147_2290.png",
    },
    {
        "name": "problem_pevc_120186",
        "kind": "known_mismatch_crop",
        "expected": "PE/VC",
        "path": "debug/latin_recovery_batch_prose_all_v2/120186/crops/002_b003_PE_VC_1833_1418_1971_1465.png",
    },
]


def _win_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _decode(value: bytes) -> str:
    for encoding in ("utf-8", "gbk", "cp936"):
        try:
            return value.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace").strip()


def _code_to_text(code: int) -> str:
    if 32 <= code <= 126:
        return chr(code)
    if code:
        return f"\\u{code:04x}"
    return ""


def _extract_chars(payload: dict[str, Any]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for line_index, line in enumerate(payload.get("lines") or []):
        for group_index, group in enumerate(line.get("groups") or []):
            for char in group.get("chars") or []:
                codes = char.get("codes") or []
                code = int(codes[0]) if codes else 0
                bbox = char.get("bbox") or {}
                chars.append({
                    "line_index": line_index,
                    "group_index": group_index,
                    "char_index": char.get("index"),
                    "text": _code_to_text(code),
                    "code": code,
                    "bbox": {
                        "left": int(bbox.get("left", 0)),
                        "top": int(bbox.get("top", 0)),
                        "right": int(bbox.get("right", 0)),
                        "bottom": int(bbox.get("bottom", 0)),
                    },
                })
    return chars


def _draw_overlay(image_path: Path, chars: list[dict[str, Any]], out_path: Path, title: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    canvas = image.copy()
    for item in chars:
        bbox = item["bbox"]
        left = int(bbox["left"])
        top = int(bbox["top"])
        right = int(bbox["right"])
        bottom = int(bbox["bottom"])
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 255), 1)
        text = item["text"]
        if text:
            cv2.putText(
                canvas,
                text[:3],
                (left, max(10, top - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(canvas, title[:80], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _run_probe(image_path: Path, output_json: Path, call: str) -> dict[str, Any]:
    exe = REPO_ROOT / "resources/hanwang_native/bin/eng20_probe.exe"
    cmd = [
        str(exe),
        _win_path(image_path),
        _win_path(output_json),
        "__missing_rb.tsv",
        call,
        "packed",
        "tbrl",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": _decode(proc.stdout),
        "stderr": _decode(proc.stderr),
    }
    if output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        chars = _extract_chars(payload)
        result.update({
            "json": str(output_json),
            "init": payload.get("init"),
            "error": payload.get("error"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "line_count": len(payload.get("lines") or []),
            "char_count": len(chars),
            "text": "".join(item["text"] for item in chars),
            "chars": chars,
        })
    return result


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Eng20 Segmentation Probe",
        "",
        "## Summary",
        "",
        "| sample | kind | call | expected | text | chars | overlay |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in payload["results"]:
        for run in item["runs"]:
            lines.append(
                f"| {item['name']} | {item['kind']} | {run['call']} | "
                f"`{item.get('expected', '')}` | `{run.get('text', '')}` | "
                f"{run.get('char_count', 0)} | `{run.get('overlay', '')}` |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `recogline_engstr` treats the input as one English line.",
        "- `recogimg_engstr` lets Eng20 segment image-level English regions.",
        "- Red boxes are Eng20 returned character boxes in crop-local coordinates.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/eng20_segmentation_probe")
    parser.add_argument("--sample", action="append", default=[], help="Optional image path. Can be repeated.")
    parser.add_argument("--calls", nargs="*", default=["recogline_engstr", "recogimg_engstr"])
    args = parser.parse_args()

    samples = list(DEFAULT_SAMPLES)
    for index, path in enumerate(args.sample):
        samples.append({
            "name": f"custom_{index:02d}",
            "kind": "custom",
            "expected": "",
            "path": path,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for sample in samples:
        image_path = Path(sample["path"])
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        item = {
            **sample,
            "image_path": str(image_path),
            "runs": [],
        }
        if not image_path.exists():
            item["error"] = "missing image"
            results.append(item)
            continue
        for call in args.calls:
            json_path = args.out_dir / f"{sample['name']}.{call}.json"
            run = _run_probe(image_path, json_path, call)
            run["call"] = call
            overlay_path = args.out_dir / f"{sample['name']}.{call}.overlay.png"
            _draw_overlay(image_path, run.get("chars") or [], overlay_path, f"{sample['name']} {call}")
            run["overlay"] = overlay_path.name
            item["runs"].append(run)
            print(f"{sample['name']} {call}: {run.get('text', '')}")
        results.append(item)

    payload = {
        "schema": "eng20_segmentation_probe.v0",
        "results": results,
    }
    json_path = args.out_dir / "eng20_segmentation_probe.json"
    md_path = args.out_dir / "eng20_segmentation_probe.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
