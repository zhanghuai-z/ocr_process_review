#!/usr/bin/env python3
"""Experiment: PP-OCRv5 line routes as the authority for token-level EngCut.

This is a read-only experiment.  It does not call Paddle or Hanwang.  It uses
saved Paddle/VL layout JSON plus saved PP-OCRv5 `/ocr` response caches, attaches
the PP-OCRv5 lines to the layout records, then checks where Latin/number tokens
land in the route lines that the Hanwang workflow will consume.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.ocr_ir_builder import build_ir_lines_from_item
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    attach_page_ocr_line_routes,
    block_bbox_xyxy,
    block_text,
)
from app.core.paddle_response import iter_ocr_preferred_items
from app.models import BBox, Line, Page


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./&+\-]{1,}|[0-9]{2,}(?:[./\-][0-9A-Za-z]+)*")
FORMULA_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)
SEPARATOR_VARIANTS = {
    "/": ["/", "f", "!", "l", "I", "1", "|"],
}
XYXY = tuple[int, int, int, int]
SKIP_TOKEN_LABELS = {
    "table",
    "figure",
    "image",
    "chart",
    "display_formula",
    "equation",
    "equation_block",
    "formula",
    "formula_number",
    "inline_formula",
    "isolated_formula",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _xyxy(value: Any) -> XYXY:
    x1, y1, x2, y2 = value
    return int(x1), int(y1), int(x2), int(y2)


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


def _iou(a: XYXY, b: XYXY) -> float:
    overlap = _intersect(a, b)
    if overlap is None:
        return 0.0
    inter_area = _area(overlap)
    union_area = _area(a) + _area(b) - inter_area
    return inter_area / max(1, union_area)


def _vertical_overlap_ratio(a: XYXY, b: XYXY) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1, min(a[3] - a[1], b[3] - b[1]))
    return overlap / denom


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


def _page_from_layout(path: Path) -> tuple[Page, list[dict[str, Any]]]:
    raw = _load_json(path)
    page_info = raw["page"]
    image_path = Path(str(page_info["display_image_path"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    page = Page(
        image_path=str(image_path),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    blocks, _overlays = LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    page.blocks = blocks
    return page, page.ppvl_parsing_res_list


def _load_ppocr_lines(path: Path, page: Page) -> list[Line]:
    raw = _load_json(path)
    response = raw.get("response", raw)
    fallback = BBox.from_xyxy(0, 0, page.width, page.height)
    lines: list[Line] = []
    for item in iter_ocr_preferred_items(response):
        for ir_line in build_ir_lines_from_item(
            item,
            image_shape=(page.height, page.width),
            fallback_bbox=fallback,
        ):
            lines.append(
                Line(
                    text=ir_line.text,
                    confidence=ir_line.confidence,
                    bbox=ir_line.bbox,
                    ocr_text=ir_line.text,
                )
            )
    return lines


def _line_lookup(lines: list[Line]) -> dict[XYXY, dict[str, Any]]:
    return {
        _xyxy(line.bbox.to_xyxy()): {
            "idx": idx,
            "text": line.text,
            "confidence": line.confidence,
            "bbox": list(_xyxy(line.bbox.to_xyxy())),
        }
        for idx, line in enumerate(lines)
    }


def _nearest_line(route_bbox: XYXY, lines: list[Line]) -> dict[str, Any] | None:
    best_score = 0.0
    best: dict[str, Any] | None = None
    for idx, line in enumerate(lines):
        line_bbox = _xyxy(line.bbox.to_xyxy())
        score = _iou(route_bbox, line_bbox) * 0.7 + _vertical_overlap_ratio(route_bbox, line_bbox) * 0.3
        if score > best_score:
            best_score = score
            best = {
                "idx": idx,
                "text": line.text,
                "confidence": line.confidence,
                "bbox": list(line_bbox),
                "match_score": score,
            }
    if best is None or best_score < 0.5:
        return None
    return best


def _route_lines_for_block(
    *,
    block_idx: int,
    block: dict[str, Any],
    ppocr_lines: list[Line],
) -> list[dict[str, Any]]:
    by_bbox = _line_lookup(ppocr_lines)
    result: list[dict[str, Any]] = []
    for route_idx, route in enumerate(block.get(LAYOUT_LINE_ROUTES_FIELD) or []):
        route_bbox = _xyxy(route.get("bbox") or [0, 0, 0, 0])
        line = by_bbox.get(route_bbox) or _nearest_line(route_bbox, ppocr_lines) or {}
        segments = [
            {
                "kind": str(segment.get("kind") or ""),
                "label": str(segment.get("label") or ""),
                "bbox": list(segment.get("bbox") or []),
                "text": str(segment.get("text") or ""),
            }
            for segment in route.get("segments") or []
            if isinstance(segment, dict)
        ]
        result.append({
            "block_idx": block_idx,
            "route_idx": route_idx,
            "bbox": list(route_bbox),
            "ppocr_line_idx": line.get("idx"),
            "ppocr_text": line.get("text", ""),
            "ppocr_confidence": line.get("confidence", 0.0),
            "ppocr_bbox": line.get("bbox", []),
            "segments": segments,
            "segment_kinds": dict(Counter(segment["kind"] for segment in segments)),
        })
    return result


def _extract_block_tokens(block_idx: int, block: dict[str, Any]) -> list[dict[str, Any]]:
    label = str(block.get("block_label") or block.get("label") or "").strip()
    if label in SKIP_TOKEN_LABELS:
        return []
    text = block_text(block)
    if not text:
        return []
    if text.lstrip().lower().startswith("<table"):
        return []
    formula_ranges = [range(match.start(), match.end()) for match in FORMULA_RE.finditer(text)]
    tokens: list[dict[str, Any]] = []
    for match in TOKEN_RE.finditer(text):
        if any(match.start() < span.stop and match.end() > span.start for span in formula_ranges):
            continue
        raw = match.group(0)
        token = _normalize_token(raw)
        if len(token) < 2:
            continue
        token_type = "number" if token.replace(".", "").replace("/", "").replace("-", "").isdigit() else "latin"
        tokens.append({
            "block_idx": block_idx,
            "block_label": label,
            "text": token,
            "raw_text": raw,
            "start": match.start(),
            "end": match.end(),
            "token_type": token_type,
        })
    return tokens


def _stream_for_routes(routes: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    entries: list[dict[str, Any] | None] = []
    for line_order, route in enumerate(routes):
        route_text = str(route.get("ppocr_text") or "")
        for ch in route_text:
            text_parts.append(ch)
            entries.append({
                "line_order": line_order,
                "route_idx": route.get("route_idx"),
                "ppocr_line_idx": route.get("ppocr_line_idx"),
                "route_bbox": route.get("bbox") or [],
                "ppocr_bbox": route.get("ppocr_bbox") or [],
                "ppocr_text": route_text,
                "segment_kinds": route.get("segment_kinds") or {},
            })
        text_parts.append("\n")
        entries.append(None)
    return {"text": "".join(text_parts), "entries": entries}


def _union_boxes(boxes: list[list[int]]) -> list[int]:
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def _future_variant_before(
    block_text_value: str,
    cursor: int,
    candidate_start: int,
    future_tokens: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for future in future_tokens:
        for variant in _token_variants(str(future.get("text") or "")):
            pos = block_text_value.find(variant, cursor)
            if 0 <= pos < candidate_start:
                return {
                    "text": future.get("text"),
                    "variant": variant,
                    "ppocr_match_start": pos,
                }
    return None


def _bind_tokens_to_routes(
    tokens: list[dict[str, Any]],
    routes_by_block: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    tokens_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for token in tokens:
        tokens_by_block[int(token["block_idx"])].append(token)

    bindings: list[dict[str, Any]] = []
    for block_idx, block_tokens in sorted(tokens_by_block.items()):
        block_tokens.sort(key=lambda item: (_safe_int(item.get("start")), _safe_int(item.get("end"))))
        stream = _stream_for_routes(routes_by_block.get(block_idx, []))
        stream_text = str(stream.get("text") or "")
        entries = list(stream.get("entries") or [])
        cursor = 0
        for token_index, token in enumerate(block_tokens):
            best: tuple[int, str] | None = None
            for variant in _token_variants(str(token.get("text") or "")):
                pos = stream_text.find(variant, cursor)
                if pos >= 0 and (best is None or pos < best[0]):
                    best = (pos, variant)
            if best is None:
                bindings.append({
                    **token,
                    "route_status": "unresolved_in_ppocr_route_text",
                    "matched_variant": "",
                    "route_idx": None,
                    "ppocr_line_idx": None,
                    "route_bbox": [],
                    "ppocr_text": "",
                })
                continue
            found, variant = best
            blocker = _future_variant_before(stream_text, cursor, found, block_tokens[token_index + 1:])
            if blocker:
                bindings.append({
                    **token,
                    "route_status": "rejected_by_source_order",
                    "matched_variant": variant,
                    "blocked_by_future_token": blocker,
                    "route_idx": None,
                    "ppocr_line_idx": None,
                    "route_bbox": [],
                    "ppocr_text": "",
                })
                continue
            end = found + len(variant)
            entry_slice = [entry for entry in entries[found:end] if entry]
            entry = entry_slice[0] if entry_slice else {}
            route_indices = sorted({
                int(item.get("route_idx", -1))
                for item in entry_slice
                if item.get("route_idx") is not None
            })
            ppocr_line_indices = sorted({
                int(item.get("ppocr_line_idx", -1))
                for item in entry_slice
                if item.get("ppocr_line_idx") is not None
            })
            route_bboxes = []
            for item in entry_slice:
                box = item.get("route_bbox") or []
                if len(box) == 4 and box not in route_bboxes:
                    route_bboxes.append(box)
            if "\n" in variant or len(route_indices) > 1:
                status = "ppocr_route_cross_line_variant"
            elif variant == token.get("text"):
                status = "ppocr_route_exact"
            else:
                status = "ppocr_route_variant"
            bindings.append({
                **token,
                "route_status": status,
                "matched_variant": variant,
                "route_idx": entry.get("route_idx"),
                "route_indices": route_indices,
                "ppocr_line_idx": entry.get("ppocr_line_idx"),
                "ppocr_line_indices": ppocr_line_indices,
                "route_bbox": _union_boxes(route_bboxes) if route_bboxes else (entry.get("route_bbox") or []),
                "route_bboxes": route_bboxes,
                "ppocr_bbox": entry.get("ppocr_bbox") or [],
                "ppocr_text": entry.get("ppocr_text") or "",
                "segment_kinds": entry.get("segment_kinds") or {},
            })
            cursor = end
    return bindings


def _existing_engcut_statuses(page_id: str) -> tuple[Counter[str], list[dict[str, Any]]]:
    path = REPO_ROOT / "debug/paddle_token_reverse_fallback_batch_v2" / page_id / "paddle_token_reverse_fallback.json"
    if not path.exists():
        return Counter(), []
    payload = _load_json(path)
    bindings = list(payload.get("bindings") or [])
    return Counter(str(item.get("status") or "") for item in bindings), bindings


def _existing_full_line_engcut_calls(page_id: str) -> int | None:
    path = REPO_ROOT / "debug/engcut_line_binding_batch_v2" / page_id / "engcut_line_binding.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    return len(payload.get("lines") or [])


def _process_page(page_id: str, args: argparse.Namespace) -> dict[str, Any]:
    layout_path = args.file_dir / f"{page_id}.layout-api.json"
    ppocr_path = args.ppocr_cache_dir / f"{page_id}_ppocrv5_return_word_box.json"
    if not layout_path.exists():
        raise FileNotFoundError(layout_path)
    if not ppocr_path.exists():
        raise FileNotFoundError(ppocr_path)

    page, records = _page_from_layout(layout_path)
    ppocr_lines = _load_ppocr_lines(ppocr_path, page)
    attached_records = copy.deepcopy(records)
    attach_page_ocr_line_routes(attached_records, ppocr_lines, page.width, page.height)

    routes_by_block: dict[int, list[dict[str, Any]]] = {}
    tokens: list[dict[str, Any]] = []
    segment_counts = Counter()
    formula_split_lines = 0
    line_count_with_tokens: set[tuple[int, int]] = set()
    for block_idx, block in enumerate(attached_records):
        block_tokens = _extract_block_tokens(block_idx, block)
        tokens.extend(block_tokens)
        routes = _route_lines_for_block(block_idx=block_idx, block=block, ppocr_lines=ppocr_lines)
        routes_by_block[block_idx] = routes
        for route in routes:
            kinds = route.get("segment_kinds") or {}
            segment_counts.update(kinds)
            if int(kinds.get("formula", 0)) > 0 and int(kinds.get("text", 0)) > 0:
                formula_split_lines += 1

    bindings = _bind_tokens_to_routes(tokens, routes_by_block)
    for item in bindings:
        if item.get("route_status") in {
            "ppocr_route_exact",
            "ppocr_route_variant",
            "ppocr_route_cross_line_variant",
        }:
            route_indices = item.get("route_indices")
            if isinstance(route_indices, list) and route_indices:
                for route_idx in route_indices:
                    line_count_with_tokens.add((int(item.get("block_idx", -1)), int(route_idx)))
            else:
                line_count_with_tokens.add((int(item.get("block_idx", -1)), int(item.get("route_idx", -1))))

    existing_counts, existing_bindings = _existing_engcut_statuses(page_id)
    existing_line_calls = _existing_full_line_engcut_calls(page_id)
    status_counts = Counter(str(item.get("route_status") or "") for item in bindings)
    token_type_counts = Counter(str(item.get("token_type") or "") for item in bindings)

    payload = {
        "schema": "ppocr_route_token_alignment.page.v0",
        "page_id": page_id,
        "layout_path": str(layout_path),
        "ppocr_cache": str(ppocr_path),
        "source_image": page.display_image_path,
        "ppocr_line_count": len(ppocr_lines),
        "layout_block_count": len(attached_records),
        "route_line_count": sum(len(value) for value in routes_by_block.values()),
        "route_segment_counts": dict(segment_counts),
        "formula_split_lines": formula_split_lines,
        "token_count": len(bindings),
        "token_type_counts": dict(token_type_counts),
        "route_status_counts": dict(status_counts),
        "unique_route_lines_with_tokens": len(line_count_with_tokens),
        "existing_engcut_status_counts": dict(existing_counts),
        "existing_full_line_engcut_calls": existing_line_calls,
        "routes_by_block": routes_by_block,
        "bindings": bindings,
        "unresolved": [
            item for item in bindings
            if item.get("route_status") not in {
                "ppocr_route_exact",
                "ppocr_route_variant",
                "ppocr_route_cross_line_variant",
            }
        ],
        "existing_engcut_bindings": existing_bindings,
    }
    out_page = args.out_dir / page_id
    _write_json(out_page / "ppocr_route_token_alignment.json", payload)
    return {
        "page_id": page_id,
        "ppocr_line_count": payload["ppocr_line_count"],
        "route_line_count": payload["route_line_count"],
        "token_count": payload["token_count"],
        "route_status_counts": payload["route_status_counts"],
        "token_type_counts": payload["token_type_counts"],
        "unique_route_lines_with_tokens": payload["unique_route_lines_with_tokens"],
        "formula_split_lines": payload["formula_split_lines"],
        "route_segment_counts": payload["route_segment_counts"],
        "existing_engcut_status_counts": payload["existing_engcut_status_counts"],
        "existing_full_line_engcut_calls": payload["existing_full_line_engcut_calls"],
        "unresolved": len(payload["unresolved"]),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# PP-OCRv5 Route Token Alignment Experiment",
        "",
        "## 方法",
        "",
        "1. 读取真实页 `*.layout-api.json`，得到 VL1.6 版面父框和子框。",
        "2. 读取旧缓存 `*_ppocrv5_return_word_box.json`，只取 PP-OCRv5 行级 bbox/text。",
        "3. 调用主程序同一个 `attach_page_ocr_line_routes()`，把 PP-OCRv5 行挂到 VL 父框。",
        "4. 从 Paddle/VL 父框文本里抽取英文和 2 位以上数字 token。",
        "5. 在同一个 block 的 PP-OCRv5 route-line 文本流里按 source-order exact/variant 绑定 token。",
        "",
        "## 总计",
        "",
    ]
    totals = summary["totals"]
    for key in (
        "page_count",
        "ppocr_line_count",
        "route_line_count",
        "token_count",
        "unique_route_lines_with_tokens",
        "existing_full_line_engcut_calls",
        "new_token_line_calls_on_existing_pages",
        "engcut_call_reduction_on_existing_pages",
        "formula_split_lines",
        "unresolved",
    ):
        if totals.get(key) is not None:
            lines.append(f"- {key}: `{totals.get(key, 0)}`")
    if totals.get("engcut_call_reduction_ratio_on_existing_pages") is not None:
        lines.append(
            "- engcut_call_reduction_ratio_on_existing_pages: "
            f"`{totals['engcut_call_reduction_ratio_on_existing_pages']:.2%}`"
        )
    lines.extend([
        f"- route_status_counts: `{totals.get('route_status_counts', {})}`",
        f"- token_type_counts: `{totals.get('token_type_counts', {})}`",
        f"- route_segment_counts: `{totals.get('route_segment_counts', {})}`",
        "",
        "## 逐页",
        "",
        "| page | ppocr lines | route lines | tokens | token lines | old line calls | exact | variant | unresolved | formula split lines | existing EngCut |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for page in summary["pages"]:
        route_counts = page.get("route_status_counts") or {}
        existing = ", ".join(f"{k}:{v}" for k, v in sorted((page.get("existing_engcut_status_counts") or {}).items()))
        lines.append(
            f"| {page['page_id']} | {page.get('ppocr_line_count', 0)} | {page.get('route_line_count', 0)} | "
            f"{page.get('token_count', 0)} | {page.get('unique_route_lines_with_tokens', 0)} | "
            f"{page.get('existing_full_line_engcut_calls') if page.get('existing_full_line_engcut_calls') is not None else ''} | "
            f"{route_counts.get('ppocr_route_exact', 0)} | "
            f"{route_counts.get('ppocr_route_variant', 0) + route_counts.get('ppocr_route_cross_line_variant', 0)} | "
            f"{page.get('unresolved', 0)} | {page.get('formula_split_lines', 0)} | {existing} |"
        )

    focus = summary.get("focus_120186") or {}
    if focus:
        lines.extend(["", "## 120186 token 明细", ""])
        for item in focus.get("bindings", []):
            lines.append(
                "- block={block_idx} token=`{text}` type={token_type} "
                "route={route_status} line={route_idx} bbox={route_bbox} "
                "ppocr=`{ppocr_text}`".format(**{
                    "block_idx": item.get("block_idx"),
                    "text": item.get("text"),
                    "token_type": item.get("token_type"),
                    "route_status": item.get("route_status"),
                    "route_idx": item.get("route_idx"),
                    "route_bbox": item.get("route_bbox"),
                    "ppocr_text": str(item.get("ppocr_text") or "")[:100],
                })
            )
    lines.extend([
        "",
        "## 读数解释",
        "",
        "- `route lines` 是 Hanwang 前的行路由；有公式时，一条 PP-OCRv5 行会被拆成 text/formula/text segment。",
        "- `token lines` 是真正包含英文/数字 token 的 route line 数，可作为 EngCut 调用上限。",
        "- `existing EngCut` 是旧实验的 token 级绑定状态，用来横向看 exact/fallback 是否一致。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-dir", type=Path, default=REPO_ROOT / "file/244771纵校")
    parser.add_argument(
        "--ppocr-cache-dir",
        type=Path,
        default=Path("/mnt/d/project/ocr_process/null/claude/.cache"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "debug/ppocr_route_token_alignment")
    parser.add_argument("--pages", nargs="*", default=None)
    args = parser.parse_args()

    if args.pages:
        page_ids = args.pages
    else:
        page_ids = sorted(
            path.stem.split("_ppocrv5", 1)[0]
            for path in args.ppocr_cache_dir.glob("*_ppocrv5_return_word_box.json")
            if (args.file_dir / f"{path.stem.split('_ppocrv5', 1)[0]}.layout-api.json").exists()
        )

    pages: list[dict[str, Any]] = []
    total_status = Counter()
    total_types = Counter()
    total_segments = Counter()
    focus_120186: dict[str, Any] = {}
    for page_id in page_ids:
        print(f"[{page_id}] PP-OCRv5 route token alignment")
        page_summary = _process_page(page_id, args)
        pages.append(page_summary)
        total_status.update(page_summary.get("route_status_counts") or {})
        total_types.update(page_summary.get("token_type_counts") or {})
        total_segments.update(page_summary.get("route_segment_counts") or {})
        if page_id == "120186":
            focus_120186 = _load_json(args.out_dir / page_id / "ppocr_route_token_alignment.json")

    totals = {
        "page_count": len(pages),
        "ppocr_line_count": sum(int(page.get("ppocr_line_count") or 0) for page in pages),
        "route_line_count": sum(int(page.get("route_line_count") or 0) for page in pages),
        "token_count": sum(int(page.get("token_count") or 0) for page in pages),
        "unique_route_lines_with_tokens": sum(int(page.get("unique_route_lines_with_tokens") or 0) for page in pages),
        "existing_full_line_engcut_calls": sum(
            int(page.get("existing_full_line_engcut_calls") or 0)
            for page in pages
            if page.get("existing_full_line_engcut_calls") is not None
        ),
        "new_token_line_calls_on_existing_pages": sum(
            int(page.get("unique_route_lines_with_tokens") or 0)
            for page in pages
            if page.get("existing_full_line_engcut_calls") is not None
        ),
        "formula_split_lines": sum(int(page.get("formula_split_lines") or 0) for page in pages),
        "unresolved": sum(int(page.get("unresolved") or 0) for page in pages),
        "route_status_counts": dict(total_status),
        "token_type_counts": dict(total_types),
        "route_segment_counts": dict(total_segments),
    }
    old_calls = int(totals["existing_full_line_engcut_calls"])
    new_calls = int(totals["new_token_line_calls_on_existing_pages"])
    if old_calls > 0:
        totals["engcut_call_reduction_on_existing_pages"] = old_calls - new_calls
        totals["engcut_call_reduction_ratio_on_existing_pages"] = 1.0 - (new_calls / old_calls)
    else:
        totals["engcut_call_reduction_on_existing_pages"] = None
        totals["engcut_call_reduction_ratio_on_existing_pages"] = None
    payload = {
        "schema": "ppocr_route_token_alignment.batch.v0",
        "pages": pages,
        "totals": totals,
        "focus_120186": focus_120186,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "ppocr_route_token_alignment_summary.json", payload)
    _write_markdown(payload, args.out_dir / "ppocr_route_token_alignment_summary.md")
    print(args.out_dir / "ppocr_route_token_alignment_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
