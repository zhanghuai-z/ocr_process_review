#!/usr/bin/env python3
"""Bind Paddle/VL Latin tokens to Eng20 character boxes from full line crops.

This is a read-only experiment for the OCR routing review.  It uses existing
batch artifacts:

- ``dispatch_units.json``: Paddle/VL block text and Latin tokens.
- ``hanwang_echo.json``: current pre-Hanwang physical line bboxes.

For each physical line, the script crops the original page image, asks Eng20 to
segment the full line, then binds Paddle tokens by exact substring only.  It
also emits exact Hanwang+Eng20 same-line consensus candidates so we can inspect
what remains possible when Paddle misses a Latin token.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2


LATIN_HINT_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\\-]{1,}")
XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return "~"


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
                    "bbox_local": [
                        int(bbox.get("left", 0)),
                        int(bbox.get("top", 0)),
                        int(bbox.get("right", 0)),
                        int(bbox.get("bottom", 0)),
                    ],
                })
    return chars


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _clip_bbox(bbox: list[int] | tuple[int, int, int, int], width: int, height: int, pad: int = 0) -> XYXY:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _run_eng20_line(crop_path: Path, result_path: Path, *, force: bool = False) -> dict[str, Any]:
    if result_path.exists() and not force:
        payload = _load_json(result_path)
        chars = _extract_chars(payload)
        return {
            "returncode": 0,
            "reused": True,
            "stdout": "",
            "stderr": "",
            "payload": payload,
            "chars": chars,
            "text": "".join(item["text"] for item in chars),
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
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    payload: dict[str, Any] = {}
    chars: list[dict[str, Any]] = []
    if result_path.exists():
        payload = _load_json(result_path)
        chars = _extract_chars(payload)
    return {
        "returncode": proc.returncode,
        "reused": False,
        "stdout": _decode(proc.stdout),
        "stderr": _decode(proc.stderr),
        "payload": payload,
        "chars": chars,
        "text": "".join(item["text"] for item in chars),
    }


def _draw_line_overlay(image_path: Path, chars: list[dict[str, Any]], out_path: Path, title: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    canvas = image.copy()
    for item in chars:
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox_local") or [0, 0, 0, 0]]
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
        text = str(item.get("text") or "")
        if text and text.isascii():
            cv2.putText(
                canvas,
                text[:2],
                (x1, max(10, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(canvas, title[:96], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _line_records(
    echo: dict[str, Any],
    image,
    page_dir: Path,
    *,
    target_blocks: set[int],
    pad: int,
    force: bool,
) -> list[dict[str, Any]]:
    source_image = Path(str(echo["source_image"]))
    if not source_image.is_absolute():
        source_image = REPO_ROOT / source_image
    height, width = image.shape[:2]

    lines_dir = page_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in echo.get("after") or []:
        block_idx = int(row.get("record_index", row.get("block_idx", -1)))
        if block_idx not in target_blocks:
            continue
        row_block_idx = int(row.get("block_idx", -1))
        for line_idx, line in enumerate(row.get("lines") or []):
            bbox = _clip_bbox(line.get("bbox") or [0, 0, 0, 0], width, height, pad=pad)
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            base = _safe_name(f"b{block_idx:03d}_r{row_block_idx:03d}_l{line_idx:02d}_{x1}_{y1}_{x2}_{y2}")
            crop_path = lines_dir / f"{base}.png"
            result_path = lines_dir / f"{base}.eng20.json"
            overlay_path = lines_dir / f"{base}.eng20.overlay.png"
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            run = _run_eng20_line(crop_path, result_path, force=force)
            chars: list[dict[str, Any]] = []
            for char_offset, char in enumerate(run.get("chars") or []):
                lx1, ly1, lx2, ly2 = [int(v) for v in char.get("bbox_local") or [0, 0, 0, 0]]
                item = dict(char)
                item["char_offset"] = char_offset
                item["bbox_page"] = [x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2]
                chars.append(item)
            _draw_line_overlay(crop_path, run.get("chars") or [], overlay_path, f"block {block_idx} line {line_idx}")
            records.append({
                "block_idx": block_idx,
                "row_block_idx": row_block_idx,
                "line_idx": line_idx,
                "bbox": list(bbox),
                "hanwang_text": str(line.get("text") or ""),
                "crop": str(crop_path.relative_to(page_dir)),
                "eng20_json": str(result_path.relative_to(page_dir)),
                "eng20_overlay": str(overlay_path.relative_to(page_dir)),
                "eng20_returncode": run["returncode"],
                "eng20_reused": bool(run.get("reused")),
                "eng20_stdout": run.get("stdout", ""),
                "eng20_stderr": run.get("stderr", ""),
                "eng20_text": str(run.get("text") or ""),
                "eng20_char_count": len(chars),
                "chars": chars,
            })
    return records


def _target_blocks(dispatch: dict[str, Any], echo: dict[str, Any]) -> set[int]:
    blocks = {int(hint["block_idx"]) for hint in dispatch.get("latin_hints") or []}
    for row in echo.get("after") or []:
        block_idx = int(row.get("record_index", row.get("block_idx", -1)))
        for line in row.get("lines") or []:
            if LATIN_HINT_RE.search(str(line.get("text") or "")):
                blocks.add(block_idx)
                break
    return blocks


def _bind_hints(dispatch: dict[str, Any], lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines_by_block: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        lines_by_block.setdefault(int(line["block_idx"]), []).append(line)
    for block_lines in lines_by_block.values():
        block_lines.sort(key=lambda item: (int(item["line_idx"]), int(item["bbox"][1]), int(item["bbox"][0])))

    streams = _block_streams(lines_by_block)
    cursors: dict[int, int] = {}
    bindings: list[dict[str, Any]] = []
    hints = sorted(
        list(dispatch.get("latin_hints") or []),
        key=lambda item: (int(item.get("block_idx", -1)), int(item.get("start", -1)), int(item.get("end", -1))),
    )
    hints_by_block: dict[int, list[dict[str, Any]]] = {}
    for hint in hints:
        hints_by_block.setdefault(int(hint["block_idx"]), []).append(hint)

    for block_idx, block_hints in hints_by_block.items():
        stream = streams.get(block_idx) or {"text": "", "entries": []}
        text = str(stream.get("text") or "")
        entries = list(stream.get("entries") or [])
        cursor = cursors.get(block_idx, 0)
        for hint_idx, hint in enumerate(block_hints):
            token = str(hint.get("text") or "")
            found = text.find(token, cursor)
            if found >= 0:
                blocker = _source_order_blocker(text, cursor, found, block_hints[hint_idx + 1:])
                if blocker:
                    bindings.append({
                        **hint,
                        "binding_status": "not_found_in_engcut_line_exact",
                        "binding_method": "exact_rejected_by_source_order",
                        "search_from": cursor,
                        "rejected_match_start": found,
                        "rejected_match_end": found + len(token),
                        "blocked_by_future_token": blocker,
                        "eng20_block_excerpt": text[max(0, cursor - 80):cursor + 500],
                    })
                    continue

            if found < 0:
                bindings.append({
                    **hint,
                    "binding_status": "not_found_in_engcut_line_exact",
                    "search_from": cursor,
                    "eng20_block_excerpt": text[max(0, cursor - 80):cursor + 500],
                })
                continue

            end = found + len(token)
            entry_slice = entries[found:end]
            if len(entry_slice) != len(token) or any(not item or not item.get("bbox_page") for item in entry_slice):
                line = _first_line(entry_slice)
                bindings.append({
                    **hint,
                    "binding_status": "found_without_complete_eng20_char_bbox",
                    "line_idx": line.get("line_idx") if line else None,
                    "line_order": line.get("line_order") if line else None,
                    "eng20_match_start": found,
                    "eng20_match_end": end,
                    "eng20_line_text": line.get("eng20_text", "") if line else "",
                    "hanwang_line_text": line.get("hanwang_text", "") if line else "",
                    "line_crop": line.get("crop", "") if line else "",
                    "line_overlay": line.get("eng20_overlay", "") if line else "",
                    "chars": [
                        {
                            "text": item.get("text") if item else "",
                            "bbox": item.get("bbox_page") if item else None,
                            "bbox_local": item.get("bbox_local") if item else None,
                        }
                        for item in entry_slice
                    ],
                })
                cursor = end
                cursors[block_idx] = cursor
                continue

            boxes = [tuple(int(v) for v in item["bbox_page"]) for item in entry_slice]
            line_refs = [
                item["line"]
                for item in entry_slice
                if item and item.get("line") is not None
            ]
            line = line_refs[0] if line_refs else {}
            line_indices = sorted({int(ref.get("line_idx", -1)) for ref in line_refs})
            bindings.append({
                **hint,
                "binding_status": "paddle_token_engcut_line_exact",
                "binding_method": "paddle_token_exact_in_eng20_line_source_ordered",
                "line_idx": line.get("line_idx"),
                "line_order": line.get("line_order"),
                "line_indices": line_indices,
                "eng20_match_start": found,
                "eng20_match_end": end,
                "bbox": list(_union(boxes)),
                "line_bbox": line.get("bbox", []),
                "line_crop": line.get("crop", ""),
                "line_overlay": line.get("eng20_overlay", ""),
                "eng20_line_text": line.get("eng20_text", ""),
                "hanwang_line_text": line.get("hanwang_text", ""),
                "chars": [
                    {
                        "text": item.get("text"),
                        "bbox": item.get("bbox_page"),
                        "bbox_local": item.get("bbox_local"),
                    }
                    for item in entry_slice
                ],
            })
            cursor = end
            cursors[block_idx] = cursor
    return bindings


def _block_streams(lines_by_block: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    streams: dict[int, dict[str, Any]] = {}
    for block_idx, block_lines in lines_by_block.items():
        text_parts: list[str] = []
        entries: list[dict[str, Any] | None] = []
        for line_order, line in enumerate(block_lines):
            line["line_order"] = line_order
            for char in line.get("chars") or []:
                text_parts.append(str(char.get("text") or ""))
                entry = dict(char)
                entry["line"] = line
                entry["line_order"] = line_order
                entries.append(entry)
            text_parts.append("\n")
            entries.append(None)
        streams[block_idx] = {
            "text": "".join(text_parts),
            "entries": entries,
        }
    return streams


def _source_order_blocker(
    block_text: str,
    cursor: int,
    candidate_start: int,
    future_hints: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for future in future_hints:
        future_text = str(future.get("text") or "")
        if not future_text:
            continue
        future_pos = block_text.find(future_text, cursor)
        if 0 <= future_pos < candidate_start:
            return {
                "text": future_text,
                "start": int(future.get("start", -1)),
                "eng20_match_start": future_pos,
            }
    return None


def _first_line(entry_slice: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    for item in entry_slice:
        if item and item.get("line") is not None:
            return dict(item["line"])
    return None


def _latin_tokens(text: str) -> list[dict[str, Any]]:
    return [
        {"text": match.group(0), "start": match.start(), "end": match.end()}
        for match in LATIN_HINT_RE.finditer(text or "")
        if not match.group(0).isdigit()
    ]


def _consensus_candidates(lines: list[dict[str, Any]], dispatch: dict[str, Any]) -> list[dict[str, Any]]:
    paddle_counts_by_block: dict[int, Counter[str]] = {}
    for hint in dispatch.get("latin_hints") or []:
        paddle_counts_by_block.setdefault(int(hint["block_idx"]), Counter())[str(hint.get("text") or "")] += 1

    consumed_by_block: dict[int, Counter[str]] = {}
    candidates: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: (int(item["block_idx"]), int(item["line_idx"]))):
        block_idx = int(line["block_idx"])
        hanwang_tokens = _latin_tokens(str(line.get("hanwang_text") or ""))
        eng20_text = str(line.get("eng20_text") or "")
        eng20_tokens = _latin_tokens(eng20_text)
        eng20_tokens_by_text: dict[str, list[dict[str, Any]]] = {}
        for token in eng20_tokens:
            eng20_tokens_by_text.setdefault(token["text"], []).append(token)
        eng20_seen = Counter()
        for token in hanwang_tokens:
            text = token["text"]
            eng20_matches = eng20_tokens_by_text.get(text) or []
            occurrence = eng20_seen[text]
            if occurrence >= len(eng20_matches):
                continue
            eng20_seen[text] += 1
            eng20_pos = int(eng20_matches[occurrence]["start"])
            chars = line.get("chars") or []
            char_slice = chars[eng20_pos:eng20_pos + len(text)]
            if len(char_slice) != len(text) or any(not item.get("bbox_page") for item in char_slice):
                continue
            boxes = [tuple(int(v) for v in item["bbox_page"]) for item in char_slice]
            consumed = consumed_by_block.setdefault(block_idx, Counter())
            paddle_counts = paddle_counts_by_block.get(block_idx, Counter())
            source = "also_in_paddle_hint"
            if consumed[text] >= paddle_counts[text]:
                source = "hanwang_engcut_consensus_only"
            consumed[text] += 1
            candidates.append({
                "block_idx": block_idx,
                "line_idx": line["line_idx"],
                "text": text,
                "status": source,
                "bbox": list(_union(boxes)),
                "line_bbox": line["bbox"],
                "line_crop": line["crop"],
                "line_overlay": line["eng20_overlay"],
                "hanwang_line_text": line.get("hanwang_text", ""),
                "eng20_line_text": eng20_text,
            })
    return candidates


def _draw_page_overlay(image, lines: list[dict[str, Any]], bindings: list[dict[str, Any]], out_path: Path) -> None:
    canvas = image.copy()
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 0), 1)
    for item in bindings:
        if item.get("binding_status") != "paddle_token_engcut_line_exact":
            continue
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            canvas,
            str(item.get("text") or "")[:12],
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 130, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), canvas)


def _page_summary(page_id: str, bindings: list[dict[str, Any]], lines: list[dict[str, Any]], consensus: list[dict[str, Any]], error: str = "") -> dict[str, Any]:
    statuses = Counter(str(item.get("binding_status") or "unknown") for item in bindings)
    return {
        "page_id": page_id,
        "error": error,
        "line_count": len(lines),
        "latin_hint_count": len(bindings),
        "exact_count": statuses.get("paddle_token_engcut_line_exact", 0),
        "not_found_count": statuses.get("not_found_in_engcut_line_exact", 0),
        "incomplete_bbox_count": statuses.get("found_without_complete_eng20_char_bbox", 0),
        "binding_statuses": dict(statuses),
        "consensus_count": len(consensus),
        "consensus_only_count": sum(1 for item in consensus if item.get("status") == "hanwang_engcut_consensus_only"),
        "unmatched": [
            {
                "block_idx": item.get("block_idx"),
                "text": item.get("text"),
                "status": item.get("binding_status"),
                "excerpt": item.get("eng20_block_excerpt", ""),
            }
            for item in bindings
            if item.get("binding_status") != "paddle_token_engcut_line_exact"
        ][:30],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    total = payload["summary"]
    lines = [
        "# EngCut Line Binding Experiment",
        "",
        "## Scope",
        "",
        "- 输入：已有 `dispatch_units.json` 的 Paddle/VL Latin token，以及 `hanwang_echo.json` 的物理行框。",
        "- 动作：按整行裁原图，调用 Eng20/EngCut 切字符，只做 exact substring 绑定。",
        "- 目的：验证 `Paddle token -> clean line crop -> Eng20 char bbox` 是否可作为拉丁字母确定性几何来源。",
        "- 约束：不使用评分、不使用模糊匹配；匹配失败就进入 review/manual。",
        "",
        "## Summary",
        "",
        f"- pages: `{total['page_count']}`",
        f"- lines: `{total['line_count']}`",
        f"- paddle_latin_hints: `{total['latin_hint_count']}`",
        f"- exact: `{total['exact_count']}`",
        f"- not_found: `{total['not_found_count']}`",
        f"- incomplete_bbox: `{total['incomplete_bbox_count']}`",
        f"- consensus_candidates: `{total['consensus_count']}`",
        f"- consensus_only_candidates: `{total['consensus_only_count']}`",
        "",
        "## Pages",
        "",
        "| page | lines | hints | exact | not_found | incomplete | consensus_only | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for page in payload["pages"]:
        status_text = ", ".join(f"{key}:{value}" for key, value in sorted((page.get("binding_statuses") or {}).items()))
        if page.get("error"):
            status_text = f"ERROR {page['error']}"
        lines.append(
            f"| {page['page_id']} | {page.get('line_count', 0)} | {page.get('latin_hint_count', 0)} | "
            f"{page.get('exact_count', 0)} | {page.get('not_found_count', 0)} | "
            f"{page.get('incomplete_bbox_count', 0)} | {page.get('consensus_only_count', 0)} | {status_text} |"
        )
    lines.extend(["", "## Unmatched", ""])
    for page in payload["pages"]:
        if not page.get("unmatched") and not page.get("error"):
            continue
        lines.append(f"### {page['page_id']}")
        if page.get("error"):
            lines.append(f"- error: `{page['error']}`")
        for item in page.get("unmatched") or []:
            excerpt = str(item.get("excerpt") or "").replace("\n", " ")
            lines.append(
                f"- block=`{item.get('block_idx')}` token=`{item.get('text')}` "
                f"status=`{item.get('status')}` excerpt=`{excerpt[:160]}`"
            )
        lines.append("")
    lines.extend([
        "## Reading The Output",
        "",
        "- 每页目录下 `lines/*.png` 是送入 Eng20 的整行原始裁图。",
        "- 每页目录下 `lines/*.eng20.overlay.png` 是 Eng20 返回的字符框回显，红框为 Eng20 字符框。",
        "- 每页 `page_overlay.png` 中青色框是输入行框，绿色框是 Paddle token exact 绑定后的 Eng20 字符联合框。",
        "- `consensus_only_candidates` 不是自动真值，只表示同一物理行内 Hanwang 文本和 Eng20 文本完全一致、但 Paddle token 列表未覆盖的拉丁 token，需要人工 review。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_page(page_id: str, page_dir: Path, out_root: Path, *, pad: int, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    dispatch_path = page_dir / "dispatch_units.json"
    echo_path = page_dir / "hanwang_echo.json"
    if not dispatch_path.exists():
        raise FileNotFoundError(dispatch_path)
    if not echo_path.exists():
        raise FileNotFoundError(echo_path)

    dispatch = _load_json(dispatch_path)
    echo = _load_json(echo_path)
    source_image = Path(str(echo["source_image"]))
    if not source_image.is_absolute():
        source_image = REPO_ROOT / source_image
    image = cv2.imread(str(source_image))
    if image is None:
        raise RuntimeError(f"Cannot read image: {source_image}")

    out_page_dir = out_root / page_id
    out_page_dir.mkdir(parents=True, exist_ok=True)
    target_blocks = _target_blocks(dispatch, echo)
    lines = _line_records(echo, image, out_page_dir, target_blocks=target_blocks, pad=pad, force=force)
    bindings = _bind_hints(dispatch, lines)
    consensus = _consensus_candidates(lines, dispatch)
    overlay_path = out_page_dir / "page_overlay.png"
    _draw_page_overlay(image, lines, bindings, overlay_path)

    page_payload = {
        "schema": "engcut_line_binding.page.v0",
        "page_id": page_id,
        "source_image": str(source_image),
        "dispatch": str(dispatch_path),
        "echo": str(echo_path),
        "line_pad": pad,
        "page_overlay": str(overlay_path.relative_to(out_page_dir)),
        "lines": lines,
        "bindings": bindings,
        "consensus_candidates": consensus,
        "summary": _page_summary(page_id, bindings, lines, consensus),
    }
    _write_json(out_page_dir / "engcut_line_binding.json", page_payload)
    return page_payload["summary"], page_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch_prose_all_v2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/engcut_line_binding")
    parser.add_argument("--pages", nargs="*", help="Page ids, for example 120194 120169.")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--line-pad", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Rerun Eng20 even if line JSON already exists.")
    args = parser.parse_args()

    page_dirs = sorted(path for path in args.batch_dir.iterdir() if path.is_dir())
    if args.pages:
        wanted = set(args.pages)
        page_dirs = [path for path in page_dirs if path.name in wanted]
    if args.max_pages:
        page_dirs = page_dirs[:args.max_pages]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for page_dir in page_dirs:
        page_id = page_dir.name
        print(f"[{page_id}] line Eng20 binding")
        try:
            summary, _payload = _process_page(page_id, page_dir, args.out_dir, pad=args.line_pad, force=args.force)
            pages.append(summary)
            print(
                f"[{page_id}] hints={summary['latin_hint_count']} "
                f"exact={summary['exact_count']} not_found={summary['not_found_count']} "
                f"consensus_only={summary['consensus_only_count']}"
            )
        except Exception as exc:
            error_summary = _page_summary(page_id, [], [], [], f"{type(exc).__name__}: {exc}")
            pages.append(error_summary)
            print(f"[{page_id}] ERROR {type(exc).__name__}: {exc}")

    totals = {
        "page_count": len(pages),
        "line_count": sum(int(page.get("line_count") or 0) for page in pages),
        "latin_hint_count": sum(int(page.get("latin_hint_count") or 0) for page in pages),
        "exact_count": sum(int(page.get("exact_count") or 0) for page in pages),
        "not_found_count": sum(int(page.get("not_found_count") or 0) for page in pages),
        "incomplete_bbox_count": sum(int(page.get("incomplete_bbox_count") or 0) for page in pages),
        "consensus_count": sum(int(page.get("consensus_count") or 0) for page in pages),
        "consensus_only_count": sum(int(page.get("consensus_only_count") or 0) for page in pages),
    }
    payload = {
        "schema": "engcut_line_binding.batch.v0",
        "batch_dir": str(args.batch_dir),
        "line_pad": args.line_pad,
        "summary": totals,
        "pages": pages,
    }
    _write_json(args.out_dir / "engcut_line_binding_summary.json", payload)
    _write_markdown(payload, args.out_dir / "engcut_line_binding_summary.md")
    print(f"wrote {args.out_dir / 'engcut_line_binding_summary.json'}")
    print(f"wrote {args.out_dir / 'engcut_line_binding_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
