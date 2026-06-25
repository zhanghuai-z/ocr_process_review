#!/usr/bin/env python3
"""Experiment: Paddle token preprocessing + reverse-selected EngCut fallback.

This script is read-only for the application.  It combines existing debug
artifacts:

- Paddle/VL text truth: debug/latin_recovery_batch_prose_all_v2/<page>/dispatch_units.json
- EngCut line chars: debug/engcut_line_binding_batch_v2/<page>/engcut_line_binding.json
- Hanwang-confidence reverse selection:
  debug/linecut_conf_reverse_select_batch_rerun/<page>/linecut_conf_reverse_select.json

The experiment checks whether Paddle tokens that fail direct EngCut exact
binding can be recovered from reverse-selected EngCut candidates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\-]{1,}|[0-9]{2,}(?:[./\-][0-9A-Za-z]+)*")
FORMULA_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)
SEPARATOR_VARIANTS = {
    "/": ["/", "f", "!", "l", "I", "1", "|"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_token(text: str) -> str:
    value = (text or "").strip()
    value = value.translate(str.maketrans({
        "／": "/",
        "⁄": "/",
        "∕": "/",
        "－": "-",
        "—": "-",
        "–": "-",
        "＋": "+",
        "＆": "&",
        "．": ".",
    }))
    return value.strip(" \t\r\n,，.。;；:：()（）[]【】{}")


def token_variants(token: str) -> list[str]:
    token = normalize_token(token)
    variants = [token]
    if "/" in token:
        expanded = [""]
        for ch in token:
            replacements = SEPARATOR_VARIANTS.get(ch, [ch])
            expanded = [prefix + replacement for prefix in expanded for replacement in replacements]
        variants.extend(expanded)
        variants.append(token.replace("/", ""))
    seen: set[str] = set()
    result: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_paddle_tokens(dispatch: dict[str, Any]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for unit in dispatch.get("units") or []:
        if unit.get("kind") != "text_parent":
            continue
        label = str(unit.get("label") or "")
        if label in {"header", "number", "page_number"}:
            continue
        text = str(unit.get("text_truth") or "")
        if not text:
            continue
        formula_ranges = [
            range(match.start(), match.end())
            for match in FORMULA_RE.finditer(text)
        ]
        block_idx = int(unit.get("block_idx", -1))
        for match in TOKEN_RE.finditer(text):
            if any(match.start() < span.stop and match.end() > span.start for span in formula_ranges):
                continue
            raw = match.group(0)
            token = normalize_token(raw)
            if len(token) < 2:
                continue
            token_type = "number" if token.replace(".", "").replace("/", "").replace("-", "").isdigit() else "latin"
            tokens.append({
                "block_idx": block_idx,
                "text": token,
                "raw_text": raw,
                "start": match.start(),
                "end": match.end(),
                "token_type": token_type,
                "unit_id": unit.get("id", ""),
                "unit_label": unit.get("label", ""),
            })
    tokens.sort(key=lambda item: (int(item["block_idx"]), int(item["start"]), int(item["end"])))
    return tokens


def union_boxes(boxes: list[list[int]]) -> list[int]:
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def block_streams(engcut: dict[str, Any]) -> dict[int, dict[str, Any]]:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for line in engcut.get("lines") or []:
        by_block.setdefault(int(line.get("block_idx", -1)), []).append(line)
    streams: dict[int, dict[str, Any]] = {}
    for block_idx, lines in by_block.items():
        text_parts: list[str] = []
        entries: list[dict[str, Any] | None] = []
        for line_order, line in enumerate(sorted(lines, key=lambda item: (int(item.get("line_idx", -1)), int((item.get("bbox") or [0, 0])[1])))):
            for char in line.get("chars") or []:
                text_parts.append(str(char.get("text") or ""))
                entry = dict(char)
                entry["line_idx"] = int(line.get("line_idx", -1))
                entry["line_order"] = line_order
                entry["line_bbox"] = line.get("bbox") or []
                entry["hanwang_line_text"] = line.get("hanwang_text") or ""
                entry["eng20_line_text"] = line.get("eng20_text") or ""
                entries.append(entry)
            text_parts.append("\n")
            entries.append(None)
        streams[block_idx] = {"text": "".join(text_parts), "entries": entries}
    return streams


def reverse_groups(reverse: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for idx, group in enumerate(reverse.get("selected_tokens_at_overlay_threshold") or []):
        item = dict(group)
        item["group_order"] = idx
        by_block.setdefault(int(item.get("block_idx", -1)), []).append(item)
    for groups in by_block.values():
        groups.sort(key=lambda item: (
            int(item.get("line_idx", -1)),
            int((item.get("bbox") or [0, 0, 0, 0])[1]),
            int((item.get("bbox") or [0, 0, 0, 0])[0]),
            int(item.get("group_order", 0)),
        ))
    return by_block


def future_variant_before(
    block_text: str,
    cursor: int,
    candidate_start: int,
    future_tokens: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for future in future_tokens:
        for variant in token_variants(str(future.get("text") or "")):
            pos = block_text.find(variant, cursor)
            if 0 <= pos < candidate_start:
                return {"text": future.get("text"), "variant": variant, "eng20_match_start": pos}
    return None


def bind_exact(
    token: dict[str, Any],
    future_tokens: list[dict[str, Any]],
    stream: dict[str, Any],
    cursor: int,
) -> tuple[dict[str, Any] | None, int]:
    text = str(stream.get("text") or "")
    entries = list(stream.get("entries") or [])
    best: tuple[int, str] | None = None
    for variant in token_variants(str(token.get("text") or "")):
        pos = text.find(variant, cursor)
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, variant)
    if best is None:
        return None, cursor
    found, variant = best
    blocker = future_variant_before(text, cursor, found, future_tokens)
    if blocker:
        return None, cursor
    end = found + len(variant)
    entry_slice = entries[found:end]
    if len(entry_slice) != len(variant) or any(not item or not item.get("bbox_page") for item in entry_slice):
        return None, cursor
    boxes = [list(item["bbox_page"]) for item in entry_slice if item and item.get("bbox_page")]
    first = next((item for item in entry_slice if item), {})
    status = "engcut_exact"
    if variant != token.get("text"):
        status = "engcut_exact_preprocessed_variant"
    return {
        **token,
        "status": status,
        "matched_variant": variant,
        "bbox": union_boxes(boxes),
        "line_idx": first.get("line_idx"),
        "line_order": first.get("line_order"),
        "chars": [
            {"text": item.get("text"), "bbox": item.get("bbox_page")}
            for item in entry_slice
            if item
        ],
    }, end


def reverse_group_matches(group_text: str, token: str) -> tuple[bool, str]:
    for variant in token_variants(token):
        if group_text == variant:
            return True, variant
    return False, ""


def reverse_part_matches(group_text: str, part: str) -> tuple[bool, str]:
    group_text = normalize_token(group_text)
    part = normalize_token(part)
    if not group_text or not part:
        return False, ""
    for variant in token_variants(part):
        if group_text == variant:
            return True, variant
        stripped = group_text.strip("/f!lI1|")
        if stripped == variant:
            return True, group_text
    return False, ""


def group_after_binding(group: dict[str, Any], binding: dict[str, Any]) -> bool:
    group_line = int(group.get("line_idx", -1))
    bind_line = int(binding.get("line_idx", -1))
    group_box = group.get("bbox") or [0, 0, 0, 0]
    bind_box = binding.get("bbox") or [0, 0, 0, 0]
    if group_line > bind_line:
        return True
    if group_line < bind_line:
        return False
    return int(group_box[0]) >= int(bind_box[2]) - 2


def advance_reverse_cursor(groups: list[dict[str, Any]], cursor: int, binding: dict[str, Any]) -> int:
    if not binding.get("bbox") or binding.get("line_idx") is None:
        return cursor
    for idx in range(cursor, len(groups)):
        if group_after_binding(groups[idx], binding):
            return idx
    return len(groups)


def bind_reverse(
    token: dict[str, Any],
    groups: list[dict[str, Any]],
    cursor: int,
) -> tuple[dict[str, Any] | None, int]:
    token_text = str(token.get("text") or "")
    for idx in range(cursor, len(groups)):
        group = groups[idx]
        ok, variant = reverse_group_matches(str(group.get("text") or ""), token_text)
        if ok:
            status = "reverse_exact" if variant == token_text else "reverse_preprocessed_variant"
            return {
                **token,
                "status": status,
                "matched_variant": variant,
                "bbox": group.get("bbox") or [],
                "line_idx": group.get("line_idx"),
                "chars": [
                    {"text": char.get("text"), "bbox": char.get("bbox")}
                    for char in group.get("chars") or []
                ],
            }, idx + 1

    if "/" in token_text:
        parts = [part for part in re.split(r"/+", token_text) if len(part) >= 2]
        if len(parts) >= 2:
            for idx in range(cursor, len(groups)):
                first = groups[idx]
                first_ok, first_variant = reverse_part_matches(str(first.get("text") or ""), parts[0])
                if not first_ok:
                    continue
                matched = [first]
                variants = [first_variant]
                scan = idx + 1
                for part in parts[1:]:
                    found_idx = -1
                    found_variant = ""
                    for j in range(scan, min(len(groups), scan + 4)):
                        candidate = groups[j]
                        if int(candidate.get("line_idx", -999)) > int(first.get("line_idx", -998)) + 1:
                            continue
                        ok, variant = reverse_part_matches(str(candidate.get("text") or ""), part)
                        if ok:
                            found_idx = j
                            found_variant = variant
                            break
                    if found_idx < 0:
                        break
                    matched.append(groups[found_idx])
                    variants.append(found_variant)
                    scan = found_idx + 1
                if len(matched) == len(parts):
                    boxes = [list(item.get("bbox") or []) for item in matched if item.get("bbox")]
                    line_indices = sorted({int(item.get("line_idx", -1)) for item in matched})
                    return {
                        **token,
                        "status": "reverse_compound_split",
                        "matched_variant": "/".join(variants),
                        "bbox": union_boxes(boxes),
                        "line_idx": first.get("line_idx"),
                        "line_indices": line_indices,
                        "chars": [
                            {"text": char.get("text"), "bbox": char.get("bbox")}
                            for item in matched
                            for char in item.get("chars") or []
                        ],
                    }, scan
    return None, cursor


def process_page(page_id: str, args: argparse.Namespace) -> dict[str, Any]:
    dispatch = load_json(args.dispatch_dir / page_id / "dispatch_units.json")
    engcut = load_json(args.engcut_dir / page_id / "engcut_line_binding.json")
    reverse = load_json(args.reverse_dir / page_id / "linecut_conf_reverse_select.json")
    tokens = extract_paddle_tokens(dispatch)
    streams = block_streams(engcut)
    reverse_by_block = reverse_groups(reverse)
    exact_cursors: dict[int, int] = {}
    reverse_cursors: dict[int, int] = {}
    bindings: list[dict[str, Any]] = []
    tokens_by_block: dict[int, list[dict[str, Any]]] = {}
    for token in tokens:
        tokens_by_block.setdefault(int(token["block_idx"]), []).append(token)

    for block_idx, block_tokens in tokens_by_block.items():
        stream = streams.get(block_idx, {"text": "", "entries": []})
        groups = reverse_by_block.get(block_idx, [])
        for token_index, token in enumerate(block_tokens):
            exact, next_exact = bind_exact(
                token,
                block_tokens[token_index + 1:],
                stream,
                exact_cursors.get(block_idx, 0),
            )
            if exact is not None:
                bindings.append(exact)
                exact_cursors[block_idx] = next_exact
                reverse_cursors[block_idx] = advance_reverse_cursor(
                    groups,
                    reverse_cursors.get(block_idx, 0),
                    exact,
                )
                continue
            reverse, next_reverse = bind_reverse(
                token,
                groups,
                reverse_cursors.get(block_idx, 0),
            )
            if reverse is not None:
                bindings.append(reverse)
                reverse_cursors[block_idx] = next_reverse
                continue
            bindings.append({
                **token,
                "status": "unresolved",
                "matched_variant": "",
                "bbox": [],
            })

    status_counts = Counter(str(item.get("status") or "") for item in bindings)
    type_counts = Counter(str(item.get("token_type") or "") for item in bindings)
    page_payload = {
        "schema": "paddle_token_reverse_fallback.page.v0",
        "page_id": page_id,
        "token_count": len(tokens),
        "status_counts": dict(status_counts),
        "token_type_counts": dict(type_counts),
        "bindings": bindings,
        "unresolved": [item for item in bindings if item.get("status") == "unresolved"],
        "fallback": [item for item in bindings if str(item.get("status") or "").startswith("reverse_")],
    }
    page_out = args.out_dir / page_id
    write_json(page_out / "paddle_token_reverse_fallback.json", page_payload)
    return {
        "page_id": page_id,
        "token_count": len(tokens),
        "status_counts": dict(status_counts),
        "token_type_counts": dict(type_counts),
        "unresolved": len(page_payload["unresolved"]),
        "fallback": len(page_payload["fallback"]),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paddle Token Reverse Fallback Experiment",
        "",
        "## Method",
        "",
        "1. 从 Paddle/VL `text_parent.text_truth` 抽取英文/复合英文 token 和 2 位以上数字 token。",
        "2. 对 token 做轻量预处理：跳过页眉/页码/公式 span，统一全角符号；`/` 允许匹配 `/ f ! l I 1 |` 这类 EngCut 常见误切变体。",
        "3. 优先在 EngCut 整 block 字符流里 source-order exact 绑定。",
        "4. 失败后，在 linecut 高置信中文/标点占用区反选后的 EngCut 候选里重新找。",
        "5. 反选 fallback 只产出候选 bbox，本实验不修改主程序。",
        "",
        "## Totals",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(payload["total_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## Pages",
        "",
        "| page | tokens | exact | exact_variant | reverse | unresolved | number tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for page in payload["pages"]:
        counts = page.get("status_counts") or {}
        reverse_total = sum(int(counts.get(key, 0)) for key in counts if str(key).startswith("reverse_"))
        lines.append(
            f"| {page['page_id']} | {page['token_count']} | "
            f"{counts.get('engcut_exact', 0)} | {counts.get('engcut_exact_preprocessed_variant', 0)} | "
            f"{reverse_total} | {page.get('unresolved', 0)} | "
            f"{(page.get('token_type_counts') or {}).get('number', 0)} |"
        )
    lines.extend([
        "",
        "## Focus 120186",
        "",
    ])
    focus = payload.get("focus_120186") or {}
    for item in focus.get("fallback", []):
        lines.append(
            f"- fallback `{item.get('text')}` -> `{item.get('status')}` "
            f"variant=`{item.get('matched_variant')}` line={item.get('line_idx')} bbox={item.get('bbox')}"
        )
    if focus.get("unresolved"):
        lines.append("")
        lines.append("Unresolved:")
        for item in focus["unresolved"]:
            lines.append(f"- `{item.get('text')}` type={item.get('token_type')} block={item.get('block_idx')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch_prose_all_v2")
    parser.add_argument("--engcut-dir", type=Path, default=REPO_ROOT / "debug/engcut_line_binding_batch_v2")
    parser.add_argument("--reverse-dir", type=Path, default=REPO_ROOT / "debug/linecut_conf_reverse_select_batch_rerun")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/paddle_token_reverse_fallback")
    parser.add_argument("--pages", nargs="*", default=None)
    args = parser.parse_args()

    if args.pages:
        pages = args.pages
    else:
        pages = sorted(
            path.name
            for path in args.engcut_dir.iterdir()
            if path.is_dir()
            and (args.dispatch_dir / path.name / "dispatch_units.json").exists()
            and (args.reverse_dir / path.name / "linecut_conf_reverse_select.json").exists()
        )

    page_summaries: list[dict[str, Any]] = []
    total_status = Counter()
    total_types = Counter()
    focus_120186: dict[str, Any] = {}
    for page_id in pages:
        print(f"[{page_id}] paddle token reverse fallback")
        summary = process_page(page_id, args)
        page_summaries.append(summary)
        total_status.update(summary.get("status_counts") or {})
        total_types.update(summary.get("token_type_counts") or {})
        if page_id == "120186":
            focus_120186 = load_json(args.out_dir / page_id / "paddle_token_reverse_fallback.json")

    payload = {
        "schema": "paddle_token_reverse_fallback.batch.v0",
        "pages": page_summaries,
        "total_status_counts": dict(total_status),
        "total_token_type_counts": dict(total_types),
        "focus_120186": focus_120186,
    }
    write_json(args.out_dir / "paddle_token_reverse_fallback_summary.json", payload)
    write_markdown(payload, args.out_dir / "paddle_token_reverse_fallback_summary.md")
    print(args.out_dir / "paddle_token_reverse_fallback_summary.md")


if __name__ == "__main__":
    main()
