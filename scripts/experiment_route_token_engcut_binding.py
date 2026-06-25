#!/usr/bin/env python3
"""Bind Paddle tokens to EngCut output from route-token benchmark results."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEPARATOR_VARIANTS = {
    "/": ["/", "f", "!", "l", "I", "1", "|"],
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _normalize_token(text: str) -> str:
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


def _token_variants(token: str) -> list[str]:
    token = _normalize_token(token)
    variants = [token]
    if "/" in token:
        variants.append(token.replace("/", "/\n"))
        variants.append(token.replace("/", "\n"))
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


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _future_variant_before(
    stream_text: str,
    cursor: int,
    candidate_start: int,
    future_tokens: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for future in future_tokens:
        for variant in _token_variants(str(future.get("text") or "")):
            pos = stream_text.find(variant, cursor)
            if 0 <= pos < candidate_start:
                return {
                    "text": future.get("text"),
                    "variant": variant,
                    "engcut_match_start": pos,
                }
    return None


def _engcut_streams_for_page(page_id: str, alignment_page: dict[str, Any], result_dir: Path) -> dict[int, dict[str, Any]]:
    streams: dict[int, dict[str, Any]] = {}
    routes_by_block = alignment_page.get("routes_by_block") or {}
    for block_key, routes in routes_by_block.items():
        block_idx = int(block_key)
        text_parts: list[str] = []
        entries: list[dict[str, Any] | None] = []
        for route in sorted(routes, key=lambda item: int(item.get("route_idx", -1))):
            route_idx = int(route.get("route_idx", -1))
            task_id = f"b{block_idx:03d}_r{route_idx:03d}"
            eng_path = result_dir / page_id / f"{task_id}.eng20.json"
            if not eng_path.exists():
                continue
            chars = _extract_chars(_load_json(eng_path))
            for char in chars:
                text_parts.append(str(char.get("text") or ""))
                entries.append({
                    **char,
                    "block_idx": block_idx,
                    "route_idx": route_idx,
                    "route_bbox": route.get("bbox") or [],
                })
            text_parts.append("\n")
            entries.append(None)
        streams[block_idx] = {
            "text": "".join(text_parts),
            "entries": entries,
        }
    return streams


def _bind_page(page_id: str, alignment_dir: Path, result_dir: Path) -> dict[str, Any]:
    alignment_page = _load_json(alignment_dir / page_id / "ppocr_route_token_alignment.json")
    streams = _engcut_streams_for_page(page_id, alignment_page, result_dir)
    tokens_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for token in alignment_page.get("bindings") or []:
        tokens_by_block[int(token.get("block_idx", -1))].append(token)

    bindings: list[dict[str, Any]] = []
    for block_idx, tokens in sorted(tokens_by_block.items()):
        tokens.sort(key=lambda item: (_safe_int(item.get("start")), _safe_int(item.get("end"))))
        stream = streams.get(block_idx, {"text": "", "entries": []})
        stream_text = str(stream.get("text") or "")
        cursor = 0
        for token_index, token in enumerate(tokens):
            best: tuple[int, str] | None = None
            for variant in _token_variants(str(token.get("text") or "")):
                pos = stream_text.find(variant, cursor)
                if pos >= 0 and (best is None or pos < best[0]):
                    best = (pos, variant)
            if best is None:
                bindings.append({
                    **token,
                    "engcut_status": "unresolved_in_engcut_stream",
                    "matched_variant": "",
                    "engcut_excerpt": stream_text[max(0, cursor - 80):cursor + 240],
                })
                continue
            found, variant = best
            blocker = _future_variant_before(stream_text, cursor, found, tokens[token_index + 1:])
            if blocker:
                bindings.append({
                    **token,
                    "engcut_status": "rejected_by_source_order",
                    "matched_variant": variant,
                    "blocked_by_future_token": blocker,
                    "engcut_excerpt": stream_text[max(0, cursor - 80):cursor + 240],
                })
                continue
            status = "engcut_exact" if variant == token.get("text") else "engcut_variant"
            bindings.append({
                **token,
                "engcut_status": status,
                "matched_variant": variant,
                "engcut_match_start": found,
                "engcut_match_end": found + len(variant),
            })
            cursor = found + len(variant)

    status_counts = Counter(str(item.get("engcut_status") or "") for item in bindings)
    return {
        "schema": "route_token_engcut_binding.page.v0",
        "page_id": page_id,
        "token_count": len(bindings),
        "status_counts": dict(status_counts),
        "bindings": bindings,
        "unresolved": [
            item for item in bindings
            if item.get("engcut_status") not in {"engcut_exact", "engcut_variant"}
        ],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    totals = payload["totals"]
    lines = [
        "# Route Token EngCut Binding Experiment",
        "",
        "## Totals",
        "",
    ]
    for key, value in totals.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Pages",
        "",
        "| page | tokens | exact | variant | unresolved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for page in payload["pages"]:
        counts = page.get("status_counts") or {}
        unresolved = page.get("unresolved_count", 0)
        lines.append(
            f"| {page['page_id']} | {page['token_count']} | "
            f"{counts.get('engcut_exact', 0)} | {counts.get('engcut_variant', 0)} | {unresolved} |"
        )
    lines.extend(["", "## Unresolved Samples", ""])
    for page in payload.get("unresolved_samples") or []:
        lines.append(f"### {page['page_id']}")
        for item in page.get("items") or []:
            excerpt = str(item.get("engcut_excerpt") or "").replace("\n", " ")
            lines.append(
                f"- block={item.get('block_idx')} token=`{item.get('text')}` "
                f"route_status={item.get('route_status')} excerpt=`{excerpt[:180]}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=REPO_ROOT / "debug/ppocr_route_token_alignment")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_ROOT / "debug/route_token_engcut_benchmark/route_with_unresolved_block_lines_w4",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/route_token_engcut_binding")
    parser.add_argument("--pages", nargs="*", default=None)
    args = parser.parse_args()

    summary = _load_json(args.alignment_dir / "ppocr_route_token_alignment_summary.json")
    page_ids = args.pages or [str(page["page_id"]) for page in summary.get("pages") or []]
    result_dir = args.benchmark_dir / "results"

    page_summaries: list[dict[str, Any]] = []
    unresolved_samples: list[dict[str, Any]] = []
    total_status = Counter()
    for page_id in page_ids:
        page_payload = _bind_page(page_id, args.alignment_dir, result_dir)
        page_out = args.out_dir / page_id
        _write_json(page_out / "route_token_engcut_binding.json", page_payload)
        total_status.update(page_payload["status_counts"])
        unresolved = page_payload.get("unresolved") or []
        if unresolved:
            unresolved_samples.append({
                "page_id": page_id,
                "items": unresolved[:10],
            })
        page_summaries.append({
            "page_id": page_id,
            "token_count": page_payload["token_count"],
            "status_counts": page_payload["status_counts"],
            "unresolved_count": len(unresolved),
        })

    token_count = sum(int(page["token_count"]) for page in page_summaries)
    success_count = int(total_status.get("engcut_exact", 0)) + int(total_status.get("engcut_variant", 0))
    totals = {
        "page_count": len(page_summaries),
        "token_count": token_count,
        "engcut_exact": int(total_status.get("engcut_exact", 0)),
        "engcut_variant": int(total_status.get("engcut_variant", 0)),
        "unresolved": token_count - success_count,
        "success_rate": f"{(success_count / max(1, token_count)) * 100:.2f}%",
        "status_counts": dict(total_status),
    }
    payload = {
        "schema": "route_token_engcut_binding.batch.v0",
        "benchmark_dir": str(args.benchmark_dir),
        "totals": totals,
        "pages": page_summaries,
        "unresolved_samples": unresolved_samples,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "route_token_engcut_binding_summary.json", payload)
    _write_markdown(payload, args.out_dir / "route_token_engcut_binding_summary.md")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(args.out_dir / "route_token_engcut_binding_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
