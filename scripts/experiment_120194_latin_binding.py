#!/usr/bin/env python3
"""Bind 120194 latin hints to Hanwang char geometry and optional Eng20 crops."""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2


XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _pad_bbox(bbox: XYXY, width: int, height: int, pad_x: int = 4, pad_y: int = 6) -> XYXY:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _char_stream(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    full_text_parts: list[str] = []
    chars: list[dict[str, Any]] = []
    offset = 0
    for line_idx, line in enumerate(row.get("lines") or []):
        line_chars = list(line.get("chars") or [])
        for char_idx, char in enumerate(line_chars):
            text = str(char.get("text") or "")
            if not text:
                continue
            for local_idx, codepoint in enumerate(text):
                entry = dict(char)
                entry["text"] = codepoint
                entry["_line_idx"] = line_idx
                entry["_char_idx"] = char_idx
                entry["_local_idx"] = local_idx
                entry["_global_idx"] = offset
                chars.append(entry)
                full_text_parts.append(codepoint)
                offset += 1
    return "".join(full_text_parts), chars


def _bind_hint(
    hint: dict[str, Any],
    row: dict[str, Any],
    cursor: int,
    *,
    source_text: str = "",
    alignment_fallback: bool = False,
) -> tuple[dict[str, Any], int]:
    token = str(hint.get("text") or "")
    full_text, chars = _char_stream(row)
    found = full_text.find(token, cursor)
    search_from = cursor
    if found < 0:
        if alignment_fallback and source_text:
            aligned = _bind_hint_by_alignment(hint, source_text, full_text, chars, search_from)
            if aligned:
                return aligned, int(aligned["hanwang_match_end"])
        return {
            **hint,
            "binding_status": "not_found_in_hanwang_text",
            "hanwang_text_excerpt": full_text[max(0, cursor - 40):cursor + 80],
        }, cursor

    end = found + len(token)
    char_slice = chars[found:end]
    if len(char_slice) != len(token) or any(not item.get("bbox") for item in char_slice):
        return {
            **hint,
            "binding_status": "found_without_complete_char_bbox",
            "hanwang_match_start": found,
            "hanwang_match_end": end,
            "search_from": search_from,
        }, end

    boxes: list[XYXY] = [
        tuple(int(v) for v in item["bbox"])
        for item in char_slice
    ]
    line_indices = sorted({int(item["_line_idx"]) for item in char_slice})
    return {
        **hint,
        "binding_status": "exact_hanwang_text_match",
        "binding_method": "exact_text",
        "hanwang_match_start": found,
        "hanwang_match_end": end,
        "search_from": search_from,
        "bbox": list(_union(boxes)),
        "line_indices": line_indices,
        "chars": [
            {
                "text": item.get("text"),
                "bbox": item.get("bbox"),
                "candidates": item.get("candidates") or [],
                "line_idx": item["_line_idx"],
                "char_idx": item["_char_idx"],
            }
            for item in char_slice
        ],
    }, end


def _bind_hint_by_alignment(
    hint: dict[str, Any],
    source_text: str,
    target_text: str,
    chars: list[dict[str, Any]],
    search_from: int,
) -> dict[str, Any] | None:
    source_start = int(hint["start"])
    source_end = int(hint["end"])
    intervals: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(None, source_text, target_text, autojunk=False)
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if a2 <= source_start or a1 >= source_end:
            continue
        if tag == "equal":
            overlap_start = max(source_start, a1)
            overlap_end = min(source_end, a2)
            intervals.append((
                b1 + (overlap_start - a1),
                b1 + (overlap_end - a1),
            ))
        elif b1 < b2:
            intervals.append((b1, b2))

    if not intervals:
        return None

    found = min(start for start, _ in intervals)
    end = max(end for _, end in intervals)
    if found >= end or found >= len(chars):
        return None
    char_slice = chars[found:min(end, len(chars))]
    if not char_slice or any(not item.get("bbox") for item in char_slice):
        return None

    boxes: list[XYXY] = [
        tuple(int(v) for v in item["bbox"])
        for item in char_slice
    ]
    line_indices = sorted({int(item["_line_idx"]) for item in char_slice})
    return {
        **hint,
        "binding_status": "alignment_fallback_match",
        "binding_method": "source_to_hanwang_alignment",
        "hanwang_match_start": found,
        "hanwang_match_end": found + len(char_slice),
        "search_from": search_from,
        "hanwang_text_span": target_text[found:found + len(char_slice)],
        "bbox": list(_union(boxes)),
        "line_indices": line_indices,
        "chars": [
            {
                "text": item.get("text"),
                "bbox": item.get("bbox"),
                "candidates": item.get("candidates") or [],
                "line_idx": item["_line_idx"],
                "char_idx": item["_char_idx"],
            }
            for item in char_slice
        ],
    }


def _crop_bindings(
    image_path: Path,
    bindings: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for index, binding in enumerate(bindings):
        if binding.get("binding_status") not in {"exact_hanwang_text_match", "alignment_fallback_match"}:
            continue
        bbox = tuple(int(v) for v in binding["bbox"])
        crop_bbox = _pad_bbox(bbox, width, height)
        x1, y1, x2, y2 = crop_bbox
        crop = image[y1:y2, x1:x2].copy()
        name = f"{index:03d}_b{binding['block_idx']:03d}_{binding['text']}_{x1}_{y1}_{x2}_{y2}.png"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        cv2.imwrite(str(crop_dir / safe_name), crop)
        binding["crop"] = str((crop_dir / safe_name).relative_to(out_dir))
        binding["crop_bbox"] = list(crop_bbox)


def _eng20_text(result_json: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in result_json.get("lines") or []:
        for group in line.get("groups") or []:
            for char in group.get("chars") or []:
                codes = char.get("codes") or []
                if not codes:
                    continue
                code = int(codes[0])
                if 32 <= code < 127:
                    parts.append(chr(code))
                elif code:
                    parts.append(f"\\u{code:04x}")
    return "".join(parts)


def _win_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/"):]
    return text.replace("/", "\\")


def _decode_process_output(value: bytes) -> str:
    for encoding in ("utf-8", "gbk", "cp936"):
        try:
            return value.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace").strip()


def _run_eng20_for_crops(out_dir: Path, bindings: list[dict[str, Any]]) -> None:
    exe = REPO_ROOT / "resources/hanwang_native/bin/eng20_probe.exe"
    for binding in bindings:
        crop = binding.get("crop")
        if not crop:
            continue
        crop_path = out_dir / crop
        result_path = crop_path.with_suffix(".eng20.json")
        cmd = [
            str(exe),
            _win_path(crop_path),
            _win_path(result_path),
            "__missing_rb.tsv",
            "recogline_engstr",
            "packed",
            "tbrl",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        binding["eng20_returncode"] = proc.returncode
        binding["eng20_stderr"] = _decode_process_output(proc.stderr)
        binding["eng20_stdout"] = _decode_process_output(proc.stdout)
        if result_path.exists():
            result = _load_json(result_path)
            binding["eng20_json"] = str(result_path.relative_to(out_dir))
            binding["eng20_text"] = _eng20_text(result)
            binding["eng20_error"] = result.get("error", "")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 120194 Latin Binding Experiment",
        "",
        f"- latin_hints: `{payload['summary']['latin_hint_count']}`",
        f"- exact_matches: `{payload['summary']['exact_match_count']}`",
        f"- alignment_fallback_matches: `{payload['summary'].get('alignment_fallback_count', 0)}`",
        f"- total_matches: `{payload['summary'].get('matched_count', payload['summary']['exact_match_count'])}`",
        f"- unmatched: `{payload['summary']['unmatched_count']}`",
        f"- eng20_ran: `{payload['summary']['eng20_ran']}`",
        "",
        "## Bindings",
        "",
    ]
    for item in payload["bindings"]:
        lines.append(
            "- "
            f"block=`{item['block_idx']}` text=`{item['text']}` "
            f"status=`{item['binding_status']}` "
            f"bbox=`{item.get('bbox')}` crop=`{item.get('crop', '')}`"
        )
        if item.get("eng20_text") is not None:
            lines.append(f"  eng20: `{item.get('eng20_text')}` error=`{item.get('eng20_error', '')}`")
        if item.get("hanwang_text_span") is not None:
            lines.append(f"  hanwang_span: `{item.get('hanwang_text_span')}` method=`{item.get('binding_method')}`")
        if item["binding_status"] not in {"exact_hanwang_text_match", "alignment_fallback_match"}:
            excerpt = str(item.get("hanwang_text_excerpt") or "")
            if excerpt:
                lines.append(f"  excerpt: {excerpt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatch",
        type=Path,
        default=REPO_ROOT / "debug/120194_autorec_lite_dispatch/120194_dispatch_units.json",
    )
    parser.add_argument(
        "--echo",
        type=Path,
        default=REPO_ROOT / "debug/120194_latin_hanwang/120194_latin_hanwang.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "debug/120194_latin_binding",
    )
    parser.add_argument("--run-eng20", action="store_true")
    parser.add_argument("--alignment-fallback", action="store_true")
    args = parser.parse_args()

    dispatch = _load_json(args.dispatch)
    echo = _load_json(args.echo)
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
            alignment_fallback=args.alignment_fallback,
        )
        cursors[block_idx] = cursor
        bindings.append(bound)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(str(echo["source_image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    _crop_bindings(image_path, bindings, args.out_dir)
    if args.run_eng20:
        _run_eng20_for_crops(args.out_dir, bindings)

    exact_count = sum(1 for item in bindings if item.get("binding_status") == "exact_hanwang_text_match")
    alignment_count = sum(1 for item in bindings if item.get("binding_status") == "alignment_fallback_match")
    payload = {
        "schema": "latin_hint_binding.v0",
        "source_image": str(image_path),
        "dispatch": str(args.dispatch),
        "echo": str(args.echo),
        "summary": {
            "latin_hint_count": len(bindings),
            "exact_match_count": exact_count,
            "alignment_fallback_count": alignment_count,
            "matched_count": exact_count + alignment_count,
            "unmatched_count": len(bindings) - exact_count - alignment_count,
            "eng20_ran": bool(args.run_eng20),
        },
        "bindings": bindings,
    }
    json_path = args.out_dir / "120194_latin_binding.json"
    md_path = args.out_dir / "120194_latin_binding.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
