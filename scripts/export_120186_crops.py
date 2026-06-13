from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAGE_ID = "120186"
IMAGE_PATH = ROOT / "file/244771纵校/120186.tif"
HANWANG_ECHO = ROOT / "debug/latin_recovery_batch_prose_all_v2/120186/hanwang_echo.json"
ENGCUT_BINDING = ROOT / "debug/engcut_line_binding_batch_v2/120186/engcut_line_binding.json"
REVERSE_SELECT = ROOT / "debug/linecut_conf_reverse_select_batch_v1/120186/linecut_conf_reverse_select.json"
OUT_DIR = ROOT / "debug/120186_crops"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy(raw: Any) -> tuple[int, int, int, int] | None:
    if not raw or len(raw) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in raw]
    return x1, y1, x2, y2


def clamp_box(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop(image: np.ndarray, bbox: tuple[int, int, int, int], *, pad: int = 0) -> np.ndarray | None:
    height, width = image.shape[:2]
    clamped = clamp_box(bbox, width, height, pad=pad)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return image[y1:y2, x1:x2].copy()


def safe_text(text: str, *, max_len: int = 16) -> str:
    if not text:
        return "empty"
    chunks = []
    for ch in text[:max_len]:
        if re.match(r"[A-Za-z0-9_.-]", ch):
            chunks.append(ch)
        else:
            chunks.append(f"U{ord(ch):04X}")
    return "_".join(chunks)


def write_crop(path: Path, image: np.ndarray | None) -> bool:
    if image is None or image.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def draw_box(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    offset: tuple[int, int] = (0, 0),
    label: str = "",
    thickness: int = 1,
) -> None:
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    x1 -= ox
    x2 -= ox
    y1 -= oy
    y2 -= oy
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def make_contact_sheet(items: list[dict[str, Any]], out_path: Path) -> None:
    if not items:
        return
    thumb_w = 68
    thumb_h = 68
    label_h = 28
    cols = 12
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 255, dtype=np.uint8)
    for idx, item in enumerate(items):
        img = item["image"]
        row = idx // cols
        col = idx % cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        if img is not None and img.size:
            h, w = img.shape[:2]
            scale = min((thumb_w - 8) / max(1, w), (thumb_h - 8) / max(1, h))
            resized = cv2.resize(
                img,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            rh, rw = resized.shape[:2]
            px = x + (thumb_w - rw) // 2
            py = y + 4 + (thumb_h - 8 - rh) // 2
            sheet[py:py + rh, px:px + rw] = resized
        label = item["label"]
        cv2.putText(
            sheet,
            label[:12],
            (x + 3, y + thumb_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            item.get("sub", "")[:12],
            (x + 3, y + thumb_h + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {IMAGE_PATH}")
    height, width = image.shape[:2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    echo = load_json(HANWANG_ECHO)
    engcut = load_json(ENGCUT_BINDING)
    reverse = load_json(REVERSE_SELECT)
    rows = echo["after"]

    engcut_lines_by_key = {
        (line.get("block_idx"), line.get("line_idx")): line
        for line in engcut.get("lines", [])
    }
    bindings = engcut.get("bindings", [])
    reverse_tokens = reverse.get("selected_tokens_at_overlay_threshold", [])
    manifest: dict[str, Any] = {
        "page_id": PAGE_ID,
        "source_image": str(IMAGE_PATH),
        "line_count": 0,
        "hanwang_char_count": 0,
        "engcut_char_count": 0,
        "binding_token_count": 0,
        "reverse_token_count": 0,
        "guariglia_focus": [],
        "guariglia_engcut_focus": [],
        "lines": [],
    }

    for row in rows:
        record_index = int(row.get("record_index") or row.get("block_idx") or 0)
        compact_idx = row.get("block_idx")
        for line_idx, line in enumerate(row.get("lines", [])):
            line_bbox = xyxy(line.get("bbox"))
            if line_bbox is None:
                continue
            key = (record_index, line_idx)
            line_id = f"b{record_index:03d}_l{line_idx:02d}"
            line_text = str(line.get("text", ""))
            line_crop = crop(image, line_bbox, pad=8)
            line_path = OUT_DIR / "lines" / f"{line_id}.png"
            write_crop(line_path, line_crop)

            overlay = crop(image, line_bbox, pad=36)
            clamped = clamp_box(line_bbox, width, height, pad=36)
            if overlay is not None and clamped is not None:
                offset = (clamped[0], clamped[1])
                draw_box(overlay, line_bbox, (0, 180, 0), offset=offset, label="line", thickness=2)
                for ci, char in enumerate(line.get("chars", [])):
                    bbox = xyxy(char.get("bbox"))
                    if bbox is not None:
                        draw_box(overlay, bbox, (0, 150, 255), offset=offset, label=str(ci), thickness=1)
                for binding in bindings:
                    if binding.get("block_idx") == record_index and binding.get("line_idx") == line_idx:
                        bbox = xyxy(binding.get("bbox"))
                        if bbox is not None:
                            draw_box(overlay, bbox, (255, 0, 255), offset=offset, label=str(binding.get("text", "")), thickness=2)
                for token in reverse_tokens:
                    if token.get("block_idx") == record_index and token.get("line_idx") == line_idx:
                        bbox = xyxy(token.get("bbox"))
                        if bbox is not None:
                            draw_box(overlay, bbox, (0, 0, 255), offset=offset, label=str(token.get("text", "")), thickness=2)
                write_crop(OUT_DIR / "line_overlays" / f"{line_id}_overlay.png", overlay)

            sheet_items: list[dict[str, Any]] = []
            char_entries: list[dict[str, Any]] = []
            for ci, char in enumerate(line.get("chars", [])):
                bbox = xyxy(char.get("bbox"))
                text = str(char.get("text") or "")
                char_img = crop(image, bbox, pad=3) if bbox is not None else None
                file_name = f"{line_id}_c{ci:03d}_{safe_text(text, max_len=1)}.png"
                rel = Path("hanwang_chars") / f"b{record_index:03d}" / f"l{line_idx:02d}" / file_name
                write_crop(OUT_DIR / rel, char_img)
                label = f"{ci}:{safe_text(text, max_len=1)}"
                sub = ""
                if bbox is not None:
                    sub = f"{bbox[0]},{bbox[1]}"
                sheet_items.append({"image": char_img, "label": label, "sub": sub})
                entry = {
                    "index": ci,
                    "text": text,
                    "bbox": list(bbox) if bbox else None,
                    "path": str(rel),
                    "candidates": char.get("candidates", [])[:10],
                    "source": char.get("source", ""),
                    "bbox_granularity": char.get("bbox_granularity", ""),
                    "token_text": char.get("token_text", ""),
                }
                char_entries.append(entry)
                manifest["hanwang_char_count"] += 1
                if "Gua" in line_text or "吨lia" in line_text or text == "吨":
                    if text == "吨" or 37 <= ci <= 47:
                        manifest["guariglia_focus"].append({
                            "line_id": line_id,
                            **entry,
                        })
            make_contact_sheet(sheet_items, OUT_DIR / "contact_sheets" / f"{line_id}_hanwang_chars.png")

            eng_line = engcut_lines_by_key.get(key)
            eng_entries: list[dict[str, Any]] = []
            if eng_line:
                eng_sheet: list[dict[str, Any]] = []
                for ci, char in enumerate(eng_line.get("chars", [])):
                    bbox = xyxy(char.get("bbox_page"))
                    text = str(char.get("text") or "")
                    char_img = crop(image, bbox, pad=3) if bbox is not None else None
                    file_name = f"{line_id}_e{ci:03d}_{safe_text(text, max_len=1)}.png"
                    rel = Path("engcut_chars") / f"b{record_index:03d}" / f"l{line_idx:02d}" / file_name
                    write_crop(OUT_DIR / rel, char_img)
                    eng_sheet.append({
                        "image": char_img,
                        "label": f"{ci}:{safe_text(text, max_len=1)}",
                        "sub": f"{bbox[0]},{bbox[1]}" if bbox else "",
                    })
                    eng_entries.append({
                        "index": ci,
                        "text": text,
                        "bbox": list(bbox) if bbox else None,
                        "path": str(rel),
                    })
                    manifest["engcut_char_count"] += 1
                make_contact_sheet(eng_sheet, OUT_DIR / "contact_sheets" / f"{line_id}_engcut_chars.png")

            manifest["line_count"] += 1
            manifest["lines"].append({
                "line_id": line_id,
                "record_index": record_index,
                "compact_block_idx": compact_idx,
                "line_idx": line_idx,
                "text": line_text,
                "bbox": list(line_bbox),
                "line_crop": str(line_path.relative_to(OUT_DIR)),
                "overlay": str(Path("line_overlays") / f"{line_id}_overlay.png"),
                "hanwang_char_count": len(char_entries),
                "engcut_char_count": len(eng_entries),
                "hanwang_contact_sheet": str(Path("contact_sheets") / f"{line_id}_hanwang_chars.png"),
                "engcut_contact_sheet": str(Path("contact_sheets") / f"{line_id}_engcut_chars.png") if eng_entries else "",
            })

    for idx, binding in enumerate(bindings):
        bbox = xyxy(binding.get("bbox"))
        if bbox is None:
            continue
        text = str(binding.get("text") or "")
        block_idx = int(binding.get("block_idx") or 0)
        line_idx = binding.get("line_idx")
        line_part = "lxx" if line_idx is None else f"l{int(line_idx):02d}"
        rel = Path("binding_tokens") / f"b{block_idx:03d}" / f"{PAGE_ID}_b{block_idx:03d}_{line_part}_t{idx:03d}_{safe_text(text)}.png"
        write_crop(OUT_DIR / rel, crop(image, bbox, pad=5))
        manifest["binding_token_count"] += 1

        if text == "Guariglia":
            focus_sheet: list[dict[str, Any]] = []
            for ci, char in enumerate(binding.get("chars") or []):
                char_bbox = xyxy(char.get("bbox"))
                char_text = str(char.get("text") or "")
                char_img = crop(image, char_bbox, pad=3) if char_bbox is not None else None
                char_rel = (
                    Path("focus")
                    / "guariglia_engcut_chars"
                    / f"g{ci:02d}_{safe_text(char_text, max_len=1)}.png"
                )
                write_crop(OUT_DIR / char_rel, char_img)
                focus_sheet.append({
                    "image": char_img,
                    "label": f"{ci}:{safe_text(char_text, max_len=1)}",
                    "sub": f"{char_bbox[0]},{char_bbox[1]}" if char_bbox else "",
                })
                manifest["guariglia_engcut_focus"].append({
                    "index": ci,
                    "text": char_text,
                    "bbox": list(char_bbox) if char_bbox else None,
                    "path": str(char_rel),
                    "source_binding_status": binding.get("binding_status", ""),
                    "source_binding_method": binding.get("binding_method", ""),
                })
            make_contact_sheet(
                focus_sheet,
                OUT_DIR / "focus" / "guariglia_engcut_chars_sheet.png",
            )

    for idx, token in enumerate(reverse_tokens):
        bbox = xyxy(token.get("bbox"))
        if bbox is None:
            continue
        text = str(token.get("text") or "")
        block_idx = int(token.get("block_idx") or 0)
        line_idx = token.get("line_idx")
        line_part = "lxx" if line_idx is None else f"l{int(line_idx):02d}"
        rel = Path("reverse_tokens") / f"b{block_idx:03d}" / f"{PAGE_ID}_b{block_idx:03d}_{line_part}_r{idx:03d}_{safe_text(text)}.png"
        write_crop(OUT_DIR / rel, crop(image, bbox, pad=5))
        manifest["reverse_token_count"] += 1

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    focus_lines = [
        item for item in manifest["lines"]
        if "Gua" in item["text"] or "吨lia" in item["text"] or "PE!VC" in item["text"]
    ]
    md_lines = [
        f"# {PAGE_ID} crops export",
        "",
        f"- source image: `{IMAGE_PATH.relative_to(ROOT)}`",
        f"- line crops: `{OUT_DIR.relative_to(ROOT)}/lines/`",
        f"- line overlays: `{OUT_DIR.relative_to(ROOT)}/line_overlays/`",
        f"- Hanwang char crops: `{OUT_DIR.relative_to(ROOT)}/hanwang_chars/`",
        f"- EngCut char crops: `{OUT_DIR.relative_to(ROOT)}/engcut_chars/`",
        f"- contact sheets: `{OUT_DIR.relative_to(ROOT)}/contact_sheets/`",
        f"- manifest: `{manifest_path.relative_to(ROOT)}`",
        "",
        "Counts:",
        f"- lines: {manifest['line_count']}",
        f"- Hanwang chars: {manifest['hanwang_char_count']}",
        f"- EngCut chars: {manifest['engcut_char_count']}",
        f"- binding token crops: {manifest['binding_token_count']}",
        f"- reverse token crops: {manifest['reverse_token_count']}",
        "",
        "Focus lines:",
    ]
    for item in focus_lines:
        md_lines.append(
            f"- `{item['line_id']}` bbox={item['bbox']} text=`{item['text']}` "
            f"overlay=`{item['overlay']}` hanwang_sheet=`{item['hanwang_contact_sheet']}`"
        )
    md_lines.extend([
        "",
        "Guariglia focus chars:",
    ])
    for item in manifest["guariglia_focus"]:
        md_lines.append(
            f"- `{item['line_id']}` c{item['index']:03d} text=`{item['text']}` "
            f"bbox={item['bbox']} path=`{item['path']}` candidates={item['candidates']}"
        )
    md_lines.extend([
        "",
        "Guariglia EngCut exact focus chars:",
        "- sheet: `focus/guariglia_engcut_chars_sheet.png`",
    ])
    for item in manifest["guariglia_engcut_focus"]:
        md_lines.append(
            f"- g{item['index']:02d} text=`{item['text']}` "
            f"bbox={item['bbox']} path=`{item['path']}`"
        )
    (OUT_DIR / "index.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "index.md")


if __name__ == "__main__":
    main()
