#!/usr/bin/env python3
"""Test Hanwang confidence based Chinese occupancy + EngCut reverse selection.

This experiment uses saved artifacts only:

- ``hanwang_echo.json`` from ``debug/latin_recovery_batch_prose_all_v2``.
- ``engcut_line_binding.json`` from ``debug/engcut_line_binding_batch_v2``.

The goal is to test whether high-confidence Hanwang CJK/punctuation character
boxes can be treated as occupied regions, leaving EngCut ASCII boxes as Latin
candidates.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2


LATIN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./&+-")
CHINESE_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】［］〔〕—…·．")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _area(box: list[int] | tuple[int, int, int, int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _center(box: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _contains_point(box: list[int] | tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x, y = point
    return int(box[0]) <= x <= int(box[2]) and int(box[1]) <= y <= int(box[3])


def _intersect(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlap_ratio(candidate: list[int], blocker: list[int]) -> float:
    overlap = _intersect(candidate, blocker)
    if overlap is None:
        return 0.0
    return _area(overlap) / max(1, _area(candidate))


def _is_cjk(char: str) -> bool:
    return any(
        "\u3400" <= ch <= "\u4dbf"
        or "\u4e00" <= ch <= "\u9fff"
        or "\uf900" <= ch <= "\ufaff"
        for ch in char
    )


def _is_hanwang_occupied_char(char: dict[str, Any], threshold: float) -> bool:
    text = str(char.get("text") or "")
    if not text:
        return False
    conf = float(char.get("confidence") or 0.0)
    if conf < threshold:
        return False
    if _is_cjk(text):
        return True
    return text in CHINESE_PUNCT


def _is_latin_candidate_char(text: str) -> bool:
    return len(text) == 1 and text in LATIN_CHARS


def _char_class(text: str) -> str:
    if not text:
        return "empty"
    if all(ch in LATIN_CHARS for ch in text):
        return "latin_ascii"
    if _is_cjk(text):
        return "cjk"
    if text in CHINESE_PUNCT:
        return "chinese_punct"
    if text.isascii():
        return "ascii_other"
    return "other_non_ascii"


def _confidence_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(v) for v in values)
    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = round((len(ordered) - 1) * q)
        return ordered[index]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": pct(0.10),
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p90": pct(0.90),
        "max": ordered[-1],
    }


def _hanwang_chars(echo: dict[str, Any]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for row in echo.get("after") or []:
        block_idx = int(row.get("record_index", row.get("block_idx", -1)))
        for line_idx, line in enumerate(row.get("lines") or []):
            for char_idx, char in enumerate(line.get("chars") or []):
                bbox = char.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                item = dict(char)
                item["block_idx"] = block_idx
                item["line_idx"] = int(line.get("line_idx", line_idx))
                item["char_idx"] = char_idx
                item["bbox"] = [int(v) for v in bbox]
                chars.append(item)
    return chars


def _truth_bindings(engcut: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in engcut.get("bindings") or []
        if item.get("binding_status") == "paddle_token_engcut_line_exact"
        and item.get("bbox")
        and item.get("chars")
    ]


def _truth_char_boxes(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth: list[dict[str, Any]] = []
    for token_index, binding in enumerate(bindings):
        token_bbox = [int(v) for v in binding["bbox"]]
        for char_index, char in enumerate(binding.get("chars") or []):
            bbox = char.get("bbox")
            if not bbox:
                continue
            truth.append({
                "token_index": token_index,
                "token": str(binding.get("text") or ""),
                "text": str(char.get("text") or ""),
                "bbox": [int(v) for v in bbox],
                "token_bbox": token_bbox,
                "block_idx": int(binding.get("block_idx", -1)),
                "line_idx": int(binding.get("line_idx", -1)),
                "char_index": char_index,
            })
    return truth


def _truth_token_hit(point: tuple[float, float], bindings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for binding in bindings:
        bbox = binding.get("bbox")
        if bbox and _contains_point(bbox, point):
            return binding
    return None


def _truth_char_hit(point: tuple[float, float], truth_chars: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in truth_chars:
        if _contains_point(item["bbox"], point):
            return item
    return None


def _engcut_chars(engcut: dict[str, Any]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for line in engcut.get("lines") or []:
        block_idx = int(line.get("block_idx", -1))
        line_idx = int(line.get("line_idx", -1))
        for char_idx, char in enumerate(line.get("chars") or []):
            text = str(char.get("text") or "")
            bbox = char.get("bbox_page")
            if not bbox or len(bbox) != 4:
                continue
            chars.append({
                "text": text,
                "bbox": [int(v) for v in bbox],
                "block_idx": block_idx,
                "line_idx": line_idx,
                "char_idx": char_idx,
            })
    return chars


def _occupied_boxes(hanwang_chars: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        char
        for char in hanwang_chars
        if _is_hanwang_occupied_char(char, threshold)
    ]


def _blocking_box(candidate_box: list[int], occupied: list[dict[str, Any]]) -> dict[str, Any] | None:
    point = _center(candidate_box)
    for blocker in occupied:
        if _contains_point(blocker["bbox"], point):
            return blocker
    for blocker in occupied:
        if _overlap_ratio(candidate_box, blocker["bbox"]) >= 0.55:
            return blocker
    return None


def _evaluate_threshold(
    *,
    threshold: float,
    hanwang_chars: list[dict[str, Any]],
    engcut_chars: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    truth_chars: list[dict[str, Any]],
) -> dict[str, Any]:
    occupied = _occupied_boxes(hanwang_chars, threshold)
    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    truth_blocked: set[tuple[int, int]] = set()
    truth_recalled: set[tuple[int, int]] = set()
    token_blocked: set[int] = set()
    token_recalled_chars: Counter[int] = Counter()
    token_char_counts = Counter(item["token_index"] for item in truth_chars)

    for char in engcut_chars:
        if not _is_latin_candidate_char(str(char.get("text") or "")):
            continue
        blocker = _blocking_box(char["bbox"], occupied)
        truth_char = _truth_char_hit(_center(char["bbox"]), truth_chars)
        truth_token = _truth_token_hit(_center(char["bbox"]), bindings)
        if blocker is not None:
            blocked_item = {
                **char,
                "blocked_by": {
                    "text": blocker.get("text"),
                    "confidence": blocker.get("confidence"),
                    "bbox": blocker.get("bbox"),
                    "class": _char_class(str(blocker.get("text") or "")),
                },
            }
            blocked.append(blocked_item)
            if truth_char is not None:
                truth_blocked.add((int(truth_char["token_index"]), int(truth_char["char_index"])))
                token_blocked.add(int(truth_char["token_index"]))
            continue
        selected_item = {
            **char,
            "truth_token": truth_token.get("text") if truth_token else "",
            "truth_char": truth_char.get("text") if truth_char else "",
        }
        selected.append(selected_item)
        if truth_char is not None:
            key = (int(truth_char["token_index"]), int(truth_char["char_index"]))
            truth_recalled.add(key)
            token_recalled_chars[int(truth_char["token_index"])] += 1

    recalled_token_count = sum(
        1
        for token_index, count in token_char_counts.items()
        if token_recalled_chars[token_index] >= count and token_index not in token_blocked
    )
    selected_truth_hits = sum(1 for item in selected if item.get("truth_char"))
    selected_token_hits = sum(1 for item in selected if item.get("truth_token"))
    selected_extra = len(selected) - selected_token_hits
    return {
        "threshold": threshold,
        "occupied_hanwang_count": len(occupied),
        "selected_engcut_latin_chars": len(selected),
        "blocked_engcut_latin_chars": len(blocked),
        "truth_latin_chars": len(truth_chars),
        "truth_latin_chars_recalled": len(truth_recalled),
        "truth_latin_chars_blocked": len(truth_blocked),
        "truth_tokens": len(bindings),
        "truth_tokens_recalled": recalled_token_count,
        "truth_tokens_blocked": len(token_blocked),
        "selected_truth_char_hits": selected_truth_hits,
        "selected_truth_token_hits": selected_token_hits,
        "selected_extra_chars": selected_extra,
        "blocked_examples": blocked[:20],
        "selected_extra_examples": [item for item in selected if not item.get("truth_token")][:20],
    }


def _confidence_audit(hanwang_chars: list[dict[str, Any]], bindings: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {
        "inside_truth_latin_top_latin_ascii": [],
        "inside_truth_latin_top_cjk_or_chinese_punct": [],
        "inside_truth_latin_top_other": [],
        "outside_truth_top_cjk_or_chinese_punct": [],
        "outside_truth_top_latin_ascii": [],
        "outside_truth_top_other": [],
    }
    examples: dict[str, list[dict[str, Any]]] = {key: [] for key in buckets}
    for char in hanwang_chars:
        bbox = char.get("bbox")
        if not bbox:
            continue
        cls = _char_class(str(char.get("text") or ""))
        inside = _truth_token_hit(_center(bbox), bindings) is not None
        if cls == "latin_ascii":
            suffix = "top_latin_ascii"
        elif cls in {"cjk", "chinese_punct"}:
            suffix = "top_cjk_or_chinese_punct"
        else:
            suffix = "top_other"
        key = ("inside_truth_latin_" if inside else "outside_truth_") + suffix
        conf = float(char.get("confidence") or 0.0)
        buckets[key].append(conf)
        if len(examples[key]) < 20:
            examples[key].append({
                "text": char.get("text"),
                "confidence": conf,
                "bbox": char.get("bbox"),
                "candidates": char.get("candidates") or [],
            })
    return {
        "stats": {key: _confidence_stats(values) for key, values in buckets.items()},
        "examples": examples,
    }


def _group_selected_tokens(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_key: tuple[int, int] | None = None
    last_char_idx = -999
    for item in sorted(selected, key=lambda c: (int(c["block_idx"]), int(c["line_idx"]), int(c["char_idx"]))):
        key = (int(item["block_idx"]), int(item["line_idx"]))
        if (
            current
            and key == last_key
            and int(item["char_idx"]) == last_char_idx + 1
            and _is_latin_candidate_char(str(item.get("text") or ""))
        ):
            current.append(item)
        else:
            if current:
                groups.append(_token_group(current))
            current = [item]
        last_key = key
        last_char_idx = int(item["char_idx"])
    if current:
        groups.append(_token_group(current))
    return [group for group in groups if len(group["text"]) >= 2]


def _token_group(chars: list[dict[str, Any]]) -> dict[str, Any]:
    x1 = min(int(item["bbox"][0]) for item in chars)
    y1 = min(int(item["bbox"][1]) for item in chars)
    x2 = max(int(item["bbox"][2]) for item in chars)
    y2 = max(int(item["bbox"][3]) for item in chars)
    return {
        "text": "".join(str(item.get("text") or "") for item in chars),
        "bbox": [x1, y1, x2, y2],
        "block_idx": int(chars[0]["block_idx"]),
        "line_idx": int(chars[0]["line_idx"]),
        "chars": chars,
    }


def _selected_for_overlay(
    *,
    threshold: float,
    hanwang_chars: list[dict[str, Any]],
    engcut_chars: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    occupied = _occupied_boxes(hanwang_chars, threshold)
    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for char in engcut_chars:
        if not _is_latin_candidate_char(str(char.get("text") or "")):
            continue
        blocker = _blocking_box(char["bbox"], occupied)
        if blocker is not None:
            blocked.append(char)
        else:
            selected.append(char)
    return occupied, selected, blocked


def _draw_overlay(
    *,
    source_image: Path,
    out_path: Path,
    threshold: float,
    hanwang_chars: list[dict[str, Any]],
    engcut_chars: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> None:
    image = cv2.imread(str(source_image))
    if image is None:
        return
    canvas = image.copy()
    occupied, selected, blocked = _selected_for_overlay(
        threshold=threshold,
        hanwang_chars=hanwang_chars,
        engcut_chars=engcut_chars,
    )
    for binding in bindings:
        x1, y1, x2, y2 = [int(v) for v in binding["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for item in occupied:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 170, 0), 1)
    for item in blocked:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 220), 1)
    for item in selected:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
        text = str(item.get("text") or "")
        if text:
            cv2.putText(
                canvas,
                text[:1],
                (x1, max(10, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"blue truth, green Hanwang occupied >= {threshold:.2f}, red selected EngCut, yellow blocked",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), canvas)


def _process_page(
    page_id: str,
    *,
    echo_dir: Path,
    engcut_dir: Path,
    out_dir: Path,
    thresholds: list[float],
    overlay_threshold: float,
) -> dict[str, Any]:
    echo = _load_json(echo_dir / page_id / "hanwang_echo.json")
    engcut = _load_json(engcut_dir / page_id / "engcut_line_binding.json")
    source_image = Path(str(echo["source_image"]))
    if not source_image.is_absolute():
        source_image = REPO_ROOT / source_image

    hanwang_chars = _hanwang_chars(echo)
    eng_chars = _engcut_chars(engcut)
    bindings = _truth_bindings(engcut)
    truth_chars = _truth_char_boxes(bindings)
    threshold_results = [
        _evaluate_threshold(
            threshold=threshold,
            hanwang_chars=hanwang_chars,
            engcut_chars=eng_chars,
            bindings=bindings,
            truth_chars=truth_chars,
        )
        for threshold in thresholds
    ]

    occupied, selected, _blocked = _selected_for_overlay(
        threshold=overlay_threshold,
        hanwang_chars=hanwang_chars,
        engcut_chars=eng_chars,
    )
    selected_tokens = _group_selected_tokens(selected)

    page_out = out_dir / page_id
    page_out.mkdir(parents=True, exist_ok=True)
    overlay_path = page_out / f"reverse_select_t{overlay_threshold:.2f}.png"
    _draw_overlay(
        source_image=source_image,
        out_path=overlay_path,
        threshold=overlay_threshold,
        hanwang_chars=hanwang_chars,
        engcut_chars=eng_chars,
        bindings=bindings,
    )
    page_payload = {
        "schema": "linecut_conf_reverse_select.page.v0",
        "page_id": page_id,
        "source_image": str(source_image),
        "overlay_threshold": overlay_threshold,
        "overlay": str(overlay_path.relative_to(page_out)),
        "hanwang_char_count": len(hanwang_chars),
        "engcut_char_count": len(eng_chars),
        "truth_token_count": len(bindings),
        "truth_latin_char_count": len(truth_chars),
        "confidence_audit": _confidence_audit(hanwang_chars, bindings),
        "thresholds": threshold_results,
        "selected_tokens_at_overlay_threshold": selected_tokens[:200],
        "occupied_count_at_overlay_threshold": len(occupied),
    }
    _write_json(page_out / "linecut_conf_reverse_select.json", page_payload)
    return {
        "page_id": page_id,
        "overlay": str(overlay_path),
        "hanwang_char_count": len(hanwang_chars),
        "engcut_char_count": len(eng_chars),
        "truth_token_count": len(bindings),
        "truth_latin_char_count": len(truth_chars),
        "thresholds": [
            {
                key: value
                for key, value in result.items()
                if key not in {"blocked_examples", "selected_extra_examples"}
            }
            for result in threshold_results
        ],
        "confidence_stats": page_payload["confidence_audit"]["stats"],
    }


def _sum_thresholds(pages: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    totals: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        total = {
            "threshold": threshold,
            "occupied_hanwang_count": 0,
            "selected_engcut_latin_chars": 0,
            "blocked_engcut_latin_chars": 0,
            "truth_latin_chars": 0,
            "truth_latin_chars_recalled": 0,
            "truth_latin_chars_blocked": 0,
            "truth_tokens": 0,
            "truth_tokens_recalled": 0,
            "truth_tokens_blocked": 0,
            "selected_truth_char_hits": 0,
            "selected_truth_token_hits": 0,
            "selected_extra_chars": 0,
        }
        for page in pages:
            if len(page.get("thresholds") or []) <= index:
                continue
            result = page["thresholds"][index]
            for key in list(total.keys()):
                if key == "threshold":
                    continue
                total[key] += int(result.get(key) or 0)
        totals.append(total)
    return totals


def _merge_confidence_stats(page_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    # The per-page payload stores stats only. For exact merged quantiles we read
    # page JSON files in main where raw bucket values are intentionally not kept
    # small; here we report aggregate counts and weighted means as a compact view.
    merged: dict[str, dict[str, float]] = {}
    for page in page_payloads:
        for key, stat in (page.get("confidence_stats") or {}).items():
            count = int(stat.get("count") or 0)
            if count <= 0:
                continue
            bucket = merged.setdefault(key, {"count": 0, "weighted_mean_sum": 0.0})
            bucket["count"] += count
            bucket["weighted_mean_sum"] += float(stat.get("mean") or 0.0) * count
    return {
        key: {
            "count": int(value["count"]),
            "mean": value["weighted_mean_sum"] / max(1, value["count"]),
        }
        for key, value in merged.items()
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Linecut Confidence Reverse Selection Experiment",
        "",
        "## Scope",
        "",
        "- 目标：验证 Hanwang/linecut 的高置信中文/标点框是否能作为占用区，反选 EngCut 的 Latin 字符框。",
        "- 输入：已有 `hanwang_echo.json` 和 `engcut_line_binding.json`，不重新调用 native。",
        "- 蓝框：Paddle token + EngCut exact 的已知 Latin token；绿色：Hanwang 高置信中文/中文标点占用框；红色：反选保留的 EngCut Latin 字符；黄色：被占用区挡掉的 EngCut Latin 字符。",
        "",
        "## Threshold Summary",
        "",
        "| threshold | truth chars | recalled chars | blocked truth chars | truth tokens | recalled tokens | blocked tokens | selected chars | selected extra chars | occupied boxes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["threshold_totals"]:
        lines.append(
            f"| {float(row['threshold']):.2f} | "
            f"{row['truth_latin_chars']} | {row['truth_latin_chars_recalled']} | {row['truth_latin_chars_blocked']} | "
            f"{row['truth_tokens']} | {row['truth_tokens_recalled']} | {row['truth_tokens_blocked']} | "
            f"{row['selected_engcut_latin_chars']} | {row['selected_extra_chars']} | {row['occupied_hanwang_count']} |"
        )
    lines.extend([
        "",
        "## Confidence Buckets",
        "",
        "| bucket | count | weighted mean confidence |",
        "| --- | ---: | ---: |",
    ])
    for key, stat in sorted(payload.get("confidence_stats_compact", {}).items()):
        lines.append(f"| {key} | {stat['count']} | {float(stat['mean']):.3f} |")
    lines.extend([
        "",
        "## Pages",
        "",
        "| page | truth tokens | truth chars | overlay |",
        "| --- | ---: | ---: | --- |",
    ])
    for page in payload["pages"]:
        lines.append(
            f"| {page['page_id']} | {page['truth_token_count']} | {page['truth_latin_char_count']} | `{page['overlay']}` |"
        )
    lines.extend([
        "",
        "## Initial Reading",
        "",
        "- 置信度字段来自 Hanwang native score 的线性归一化：`confidence = 1 - score / 100`，不是后处理随意生成。",
        "- 如果阈值过低，英文被误识别成中文标点/汉字的低置信框会挡住 EngCut 的真实英文。",
        "- 如果阈值过高，低置信但真实的中文/标点不会进入占用区，EngCut 在中文区域产生的 ASCII 噪声会增加。",
        "- 因此置信度可以参与“占用区过滤”，但不能单独成为绝对规则；还需要字符类别和后续 exact 校验。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo-dir", type=Path, default=REPO_ROOT / "debug/latin_recovery_batch_prose_all_v2")
    parser.add_argument("--engcut-dir", type=Path, default=REPO_ROOT / "debug/engcut_line_binding_batch_v2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/linecut_conf_reverse_select")
    parser.add_argument("--pages", nargs="*", help="Optional page ids.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.85, 0.95])
    parser.add_argument("--overlay-threshold", type=float, default=0.5)
    args = parser.parse_args()

    page_ids = sorted(path.name for path in args.engcut_dir.iterdir() if path.is_dir())
    if args.pages:
        wanted = set(args.pages)
        page_ids = [page_id for page_id in page_ids if page_id in wanted]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for page_id in page_ids:
        echo_path = args.echo_dir / page_id / "hanwang_echo.json"
        engcut_path = args.engcut_dir / page_id / "engcut_line_binding.json"
        if not echo_path.exists() or not engcut_path.exists():
            continue
        print(f"[{page_id}] reverse selection")
        pages.append(_process_page(
            page_id,
            echo_dir=args.echo_dir,
            engcut_dir=args.engcut_dir,
            out_dir=args.out_dir,
            thresholds=args.thresholds,
            overlay_threshold=args.overlay_threshold,
        ))

    payload = {
        "schema": "linecut_conf_reverse_select.batch.v0",
        "echo_dir": str(args.echo_dir),
        "engcut_dir": str(args.engcut_dir),
        "overlay_threshold": args.overlay_threshold,
        "thresholds": args.thresholds,
        "threshold_totals": _sum_thresholds(pages, args.thresholds),
        "confidence_stats_compact": _merge_confidence_stats(pages),
        "pages": pages,
    }
    _write_json(args.out_dir / "linecut_conf_reverse_select_summary.json", payload)
    _write_markdown(payload, args.out_dir / "linecut_conf_reverse_select_summary.md")
    print(f"wrote {args.out_dir / 'linecut_conf_reverse_select_summary.json'}")
    print(f"wrote {args.out_dir / 'linecut_conf_reverse_select_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
