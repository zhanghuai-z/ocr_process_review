#!/usr/bin/env python3
"""Compare old Hanwang alignment fallbacks with the new Latin strategy.

Old baseline:
    debug/latin_recovery_batch_prose_all_v2/*/latin_binding.json
    binding_status == alignment_fallback_match

New strategy:
    primary  = Paddle token + EngCut exact
    fallback = linecut confidence reverse selection + EngCut candidates
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


XYXY = tuple[int, int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox(raw: Any) -> XYXY | None:
    if not raw or len(raw) != 4:
        return None
    return tuple(int(v) for v in raw)


def _area(box: XYXY) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersect(a: XYXY, b: XYXY) -> XYXY | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlap_ratio(target: XYXY, candidate: XYXY) -> float:
    overlap = _intersect(target, candidate)
    if overlap is None:
        return 0.0
    return _area(overlap) / max(1, min(_area(target), _area(candidate)))


def _item_key(item: dict[str, Any]) -> tuple[int, str, int, int]:
    return (
        int(item.get("block_idx", -1)),
        str(item.get("text") or ""),
        int(item.get("start", -1)),
        int(item.get("end", -1)),
    )


def _old_fallback_items(old_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(old_dir.glob("*/latin_binding.json")):
        page_id = path.parent.name
        payload = _load_json(path)
        for binding in payload.get("bindings") or []:
            if binding.get("binding_status") != "alignment_fallback_match":
                continue
            line_indices = binding.get("line_indices") or []
            items.append({
                "page_id": page_id,
                "block_idx": int(binding.get("block_idx", -1)),
                "line_idx": int(line_indices[0]) if line_indices else None,
                "text": str(binding.get("text") or ""),
                "start": int(binding.get("start", -1)),
                "end": int(binding.get("end", -1)),
                "old_status": binding.get("binding_status"),
                "old_bbox": binding.get("bbox"),
                "old_hanwang_text_span": binding.get("hanwang_text_span", ""),
                "old_eng20_crop_text": binding.get("eng20_text", ""),
                "old_crop": binding.get("crop", ""),
            })
    return items


def _primary_index(primary_path: Path) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    if not primary_path.exists():
        return {}
    payload = _load_json(primary_path)
    return {
        _item_key(item): item
        for item in payload.get("bindings") or []
    }


def _reverse_candidates(reverse_path: Path) -> list[dict[str, Any]]:
    if not reverse_path.exists():
        return []
    payload = _load_json(reverse_path)
    return list(payload.get("selected_tokens_at_overlay_threshold") or [])


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": candidate.get("text", ""),
        "block_idx": candidate.get("block_idx"),
        "line_idx": candidate.get("line_idx"),
        "bbox": candidate.get("bbox"),
    }


def _find_reverse_match(item: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    block_idx = int(item["block_idx"])
    line_idx = item.get("line_idx")
    text = str(item["text"])
    old_box = _bbox(item.get("old_bbox"))
    same_line = [
        candidate
        for candidate in candidates
        if int(candidate.get("block_idx", -999)) == block_idx
        and (line_idx is None or int(candidate.get("line_idx", -999)) == int(line_idx))
    ]

    exact = [candidate for candidate in same_line if str(candidate.get("text") or "") == text]
    if exact:
        return "fallback_exact_same_line", [_candidate_summary(candidate) for candidate in exact[:5]]

    contains = [
        candidate
        for candidate in same_line
        if text in str(candidate.get("text") or "") or str(candidate.get("text") or "") in text
    ]
    if contains:
        return "fallback_contains_or_split_review", [_candidate_summary(candidate) for candidate in contains[:8]]

    if old_box is not None:
        overlaps = []
        for candidate in same_line:
            candidate_box = _bbox(candidate.get("bbox"))
            if candidate_box is None:
                continue
            overlap = _overlap_ratio(old_box, candidate_box)
            if overlap >= 0.20:
                value = _candidate_summary(candidate)
                value["overlap_ratio"] = overlap
                overlaps.append(value)
        if overlaps:
            overlaps.sort(key=lambda value: float(value.get("overlap_ratio") or 0.0), reverse=True)
            return "fallback_overlap_variant_review", overlaps[:8]

    same_block = [
        candidate
        for candidate in candidates
        if int(candidate.get("block_idx", -999)) == block_idx
        and str(candidate.get("text") or "") == text
    ]
    if same_block:
        return "fallback_exact_same_block_wrong_line_review", [_candidate_summary(candidate) for candidate in same_block[:8]]
    return "fallback_unresolved", []


def _classify_item(
    item: dict[str, Any],
    *,
    primary: dict[tuple[int, str, int, int], dict[str, Any]],
    reverse_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    key = (
        int(item["block_idx"]),
        str(item["text"]),
        int(item["start"]),
        int(item["end"]),
    )
    primary_item = primary.get(key)
    primary_status = str(primary_item.get("binding_status") if primary_item else "missing")
    result = dict(item)
    result["primary_status"] = primary_status
    result["primary_bbox"] = primary_item.get("bbox") if primary_item else None
    result["primary_line_idx"] = primary_item.get("line_idx") if primary_item else None
    if primary_status == "paddle_token_engcut_line_exact":
        result["new_strategy_status"] = "primary_exact"
        result["reverse_status"] = "not_needed"
        result["reverse_candidates"] = []
        return result

    reverse_status, candidates = _find_reverse_match(item, reverse_candidates)
    result["reverse_status"] = reverse_status
    result["reverse_candidates"] = candidates
    if reverse_status == "fallback_exact_same_line":
        result["new_strategy_status"] = "fallback_exact_after_takeover"
    elif reverse_status == "fallback_unresolved":
        result["new_strategy_status"] = "fallback_unresolved_review"
    else:
        result["new_strategy_status"] = "fallback_variant_review"
    return result


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Old Fallback vs New Latin Strategy",
        "",
        "## Summary",
        "",
        f"- old_alignment_fallback_items: `{summary['old_alignment_fallback_items']}`",
        f"- primary_exact: `{summary['primary_exact']}`",
        f"- fallback_exact_after_takeover: `{summary['fallback_exact_after_takeover']}`",
        f"- fallback_variant_review: `{summary['fallback_variant_review']}`",
        f"- fallback_unresolved_review: `{summary['fallback_unresolved_review']}`",
        "",
        "## Items",
        "",
        "| page | block | line | token | old span | new primary | reverse fallback | final | candidates |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["items"]:
        candidates = "; ".join(
            f"{cand.get('text')}@b{cand.get('block_idx')}.l{cand.get('line_idx')} {cand.get('bbox')}"
            for cand in item.get("reverse_candidates") or []
        )
        lines.append(
            f"| {item['page_id']} | {item['block_idx']} | {item.get('line_idx')} | "
            f"`{item['text']}` | `{item.get('old_hanwang_text_span', '')}` | "
            f"`{item['primary_status']}` | `{item['reverse_status']}` | "
            f"`{item['new_strategy_status']}` | {candidates} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `primary_exact` means the new primary route already fixes the old fallback case; fallback is not needed.",
        "- `fallback_exact_after_takeover` means primary fails, but linecut reverse selection can recover an exact same-line token candidate.",
        "- `fallback_variant_review` means fallback exposes useful geometry, but text is not exact, so it must go to review/manual instead of being auto-accepted.",
        "- The experiment intentionally does not normalize `PEfVC`, `PE!VC`, or `Lemer` to the Paddle token. That would be fuzzy correction and is outside the lightweight fallback boundary.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch_prose_all_v2")
    parser.add_argument("--primary-dir", type=Path, default=REPO_ROOT / "debug/engcut_line_binding_batch_v2")
    parser.add_argument("--reverse-dir", type=Path, default=REPO_ROOT / "debug/linecut_conf_reverse_select_batch_v1")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/old_fallback_vs_new_latin_strategy")
    args = parser.parse_args()

    old_items = _old_fallback_items(args.old_dir)
    results: list[dict[str, Any]] = []
    for item in old_items:
        page_id = item["page_id"]
        primary = _primary_index(args.primary_dir / page_id / "engcut_line_binding.json")
        reverse = _reverse_candidates(args.reverse_dir / page_id / "linecut_conf_reverse_select.json")
        results.append(_classify_item(item, primary=primary, reverse_candidates=reverse))

    counts = Counter(item["new_strategy_status"] for item in results)
    payload = {
        "schema": "old_fallback_vs_new_latin_strategy.v0",
        "summary": {
            "old_alignment_fallback_items": len(results),
            "primary_exact": counts.get("primary_exact", 0),
            "fallback_exact_after_takeover": counts.get("fallback_exact_after_takeover", 0),
            "fallback_variant_review": counts.get("fallback_variant_review", 0),
            "fallback_unresolved_review": counts.get("fallback_unresolved_review", 0),
        },
        "items": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "old_fallback_vs_new_latin_strategy.json", payload)
    _write_markdown(payload, args.out_dir / "old_fallback_vs_new_latin_strategy.md")
    print(f"wrote {args.out_dir / 'old_fallback_vs_new_latin_strategy.json'}")
    print(f"wrote {args.out_dir / 'old_fallback_vs_new_latin_strategy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
