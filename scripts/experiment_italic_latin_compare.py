#!/usr/bin/env python3
"""Compare Paddle and Hanwang recognition on italic/slanted Latin samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines.hanwang import native_bridge
from app.engines.hanwang.micro_recblock import _line_results_from_recog


OUT_DIR = ROOT / "debug" / "italic_latin_paddle_hanwang_compare"
PURE_120180 = ROOT / "debug/pure_english_linecut_vs_direct_engcut/120180/pure_english_linecut_vs_direct_engcut.json"
ROUTE_BINDING_120180 = ROOT / "debug/route_token_engcut_binding/120180/route_token_engcut_binding.json"
LATIN_BINDING_120186 = ROOT / "debug/latin_recovery_batch_prose_all_v2/120186/latin_binding.json"


FOCUS_120180 = {
    (16, 0),
    (17, 0),
    (20, 0),
    (20, 1),
}
FOCUS_120186 = {"PE/VC", "Lerner", "Guariglia", "Gompers"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_text(value: str, limit: int = 140) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _hanwang_recog_text(image, bbox: list[int]) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return {"text": "", "error": "empty_crop"}
    try:
        raw = native_bridge.run_linecut_recog(crop, with_charrcg=True, timeout=60.0)
    except Exception as exc:
        return {"text": "", "error": str(exc)}
    lines = _line_results_from_recog(
        raw,
        fallback_bbox=(0, 0, crop.shape[1], crop.shape[0]),
        include_chars=True,
        fallback_empty=False,
    )
    return {
        "text": "".join(line.text for line in lines),
        "line_texts": [line.text for line in lines],
        "char_count": sum(len(line.chars) for line in lines),
        "error": "",
    }


def _binding_tokens_by_route() -> dict[tuple[int, int], list[dict[str, Any]]]:
    payload = _load_json(ROUTE_BINDING_120180)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in payload.get("bindings") or []:
        route_idx = item.get("route_idx")
        if route_idx is None:
            continue
        key = (int(item.get("block_idx", -1)), int(route_idx))
        grouped.setdefault(key, []).append(item)
    return grouped


def _summarize_tokens(tokens: list[dict[str, Any]]) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    for item in tokens:
        status = str(item.get("engcut_status") or "")
        route_status = str(item.get("route_status") or "")
        mark = "ok" if status == "engcut_exact" else "bad"
        if route_status != "ppocr_route_exact":
            mark = "route?"
        parts.append(f"{item.get('text')}[{mark}]")
    return ", ".join(parts)


def _rows_120180() -> list[dict[str, Any]]:
    payload = _load_json(PURE_120180)
    image = cv2.imread(str(payload["source_image"]))
    if image is None:
        raise RuntimeError(f"Cannot read {payload['source_image']}")
    tokens_by_route = _binding_tokens_by_route()
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        key = (int(item["block_idx"]), int(item["route_idx"]))
        if key not in FOCUS_120180:
            continue
        direct = item.get("direct_engcut") or {}
        linecut_text = str(item.get("linecut_text_joined") or "")
        if not linecut_text:
            linecut_values = item.get("linecut_then_engcut") or item.get("linecut_engcut") or []
            if isinstance(linecut_values, dict):
                linecut_text = str(linecut_values.get("text") or "")
            elif isinstance(linecut_values, list):
                linecut_text = "".join(str(value.get("text") or "") for value in linecut_values if isinstance(value, dict))
        recog = _hanwang_recog_text(image, item["route_bbox"])
        tokens = tokens_by_route.get(key, [])
        rows.append({
            "page_id": "120180",
            "sample": f"b{key[0]:03d}_r{key[1]:03d}",
            "bbox": item["route_bbox"],
            "paddle_ppocr_line": item.get("ppocr_text", ""),
            "paddle_tokens": [token.get("text") for token in tokens],
            "paddle_token_status": _summarize_tokens(tokens),
            "hanwang_recog_text": recog["text"],
            "hanwang_recog_error": recog["error"],
            "engcut_direct_text": direct.get("text", ""),
            "engcut_linecut_text": linecut_text,
            "visual_compare": (
                f"debug/pure_english_linecut_vs_direct_engcut/120180/"
                f"120180_b{key[0]:03d}_r{key[1]:03d}_direct_vs_linecut.png"
            ),
            "word_fallback_visual": (
                f"debug/engcut_word_fallback_120180/120180/"
                f"120180_b{key[0]:03d}_r{key[1]:03d}_word_fallback.png"
            ),
        })
    return rows


def _rows_120186() -> list[dict[str, Any]]:
    payload = _load_json(LATIN_BINDING_120186)
    rows: list[dict[str, Any]] = []
    for item in payload.get("bindings") or []:
        token = str(item.get("text") or "")
        if token not in FOCUS_120186:
            continue
        crop = str(item.get("crop") or "")
        rows.append({
            "page_id": "120186",
            "sample": f"b{int(item.get('block_idx', -1)):03d}_{token}",
            "paddle_token": token,
            "hanwang_binding_status": item.get("binding_status", ""),
            "hanwang_text_span": item.get("hanwang_text_span", token),
            "engcut_text": item.get("eng20_text", ""),
            "bbox": item.get("bbox") or [],
            "crop": f"debug/latin_recovery_batch_prose_all_v2/120186/{crop}" if crop else "",
        })
    return rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Italic Latin Paddle vs Hanwang Compare",
        "",
        "## Scope",
        "",
        "- Paddle/VL token: block or table/formula text returned by Paddle/VL.",
        "- PP-OCR route line: Paddle line text used only for route alignment.",
        "- Hanwang Recog: Hanwang Chinese/mixed recognizer on the same crop.",
        "- EngCut: Hanwang English module; useful for geometry, not authoritative text.",
        "",
        "## 120180 Reference Lines",
        "",
        "| sample | Paddle/PP-OCR route text | Paddle tokens | Hanwang Recog | EngCut direct | visual |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["page_120180"]:
        visual = row["visual_compare"]
        lines.append(
            f"| {row['sample']} | `{_safe_text(row['paddle_ppocr_line'], 90)}` | "
            f"`{_safe_text(row['paddle_token_status'], 90)}` | "
            f"`{_safe_text(row['hanwang_recog_text'], 90)}` | "
            f"`{_safe_text(row['engcut_direct_text'], 90)}` | "
            f"[compare](../../{visual}) |"
        )
    lines.extend([
        "",
        "## 120186 Mixed Chinese/Latin",
        "",
        "| sample | Paddle token | Hanwang span | EngCut text | crop |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in payload["page_120186"]:
        crop = row.get("crop") or ""
        crop_link = f"[crop](../../{crop})" if crop else ""
        lines.append(
            f"| {row['sample']} | `{row['paddle_token']}` | "
            f"`{_safe_text(row['hanwang_text_span'], 60)}` | "
            f"`{_safe_text(row['engcut_text'], 60)}` | {crop_link} |"
        )
    lines.extend([
        "",
        "## Initial Reading",
        "",
        "- Paddle/VL text is the stronger source for token truth on italic Latin words.",
        "- PP-OCR route text is good enough for many route lines, but may drop or merge letters in dense reference lines.",
        "- Hanwang Recog handles many mixed Latin tokens, but italic/adhered letters can collapse: `Lerner -> Lemer`, `Guariglia -> Gua吨lia` before EngCut correction.",
        "- EngCut gives useful geometry and often exact text, but italic reference words can acquire false dots or letter substitutions: `Macroeconomics -> Macroecortomics`, `Unbalanced -> Unbalan.ced`.",
        "- Product rule should remain: Paddle token is truth; EngCut supplies geometry when exact; fuzzy EngCut text must not overwrite truth.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "italic_latin_paddle_hanwang_compare.v0",
        "page_120180": _rows_120180(),
        "page_120186": _rows_120186(),
    }
    _write_json(OUT_DIR / "italic_latin_compare.json", payload)
    _write_markdown(payload, OUT_DIR / "italic_latin_compare.md")
    print(OUT_DIR / "italic_latin_compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
