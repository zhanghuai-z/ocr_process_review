from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
PAGE_ID = "120186"
IMAGE_PATH = ROOT / "file/244771纵校/120186.tif"
OUT_DIR = ROOT / "debug/120186_stage_echo"

HANWANG_ECHO = ROOT / "debug/latin_recovery_batch_prose_all_v2/120186/hanwang_echo.json"
ENGCUT_BINDING = ROOT / "debug/engcut_line_binding_batch_v2/120186/engcut_line_binding.json"
REVERSE_SELECT = ROOT / "debug/linecut_conf_reverse_select_batch_v1/120186/linecut_conf_reverse_select.json"


COLORS = {
    "block": (255, 120, 0),
    "line": (0, 180, 0),
    "hanwang_char": (0, 150, 255),
    "engcut_char": (255, 255, 0),
    "binding": (255, 0, 255),
    "reverse": (0, 0, 255),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy(bbox: list[int] | tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return x1, y1, x2, y2


def bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def draw_box(
    image,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    label: str = "",
    *,
    offset: tuple[int, int] = (0, 0),
    thickness: int = 2,
) -> None:
    if bbox is None:
        return
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
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def crop_rect(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    pad_x: int = 90,
    pad_y: int = 70,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def text_preview(text: str, limit: int = 42) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot load image: {IMAGE_PATH}")
    height, width = image.shape[:2]

    echo = load_json(HANWANG_ECHO)
    engcut = load_json(ENGCUT_BINDING)
    reverse = load_json(REVERSE_SELECT)

    rows = echo["after"]
    rows_by_record = {row.get("record_index"): row for row in rows}
    lines = engcut["lines"]
    bindings = engcut["bindings"]
    reverse_tokens = reverse["selected_tokens_at_overlay_threshold"]

    full = image.copy()
    for row in rows:
        record_index = row.get("record_index")
        draw_box(full, xyxy(row.get("block_bbox")), COLORS["block"], f"VL b{record_index}", thickness=3)
        for line_idx, line in enumerate(row.get("lines", [])):
            draw_box(full, xyxy(line.get("bbox")), COLORS["line"], f"HW l{line_idx}", thickness=1)
    for binding in bindings:
        bbox = xyxy(binding.get("bbox"))
        if bbox:
            draw_box(full, bbox, COLORS["binding"], binding.get("text", ""), thickness=2)
    for token in reverse_tokens:
        bbox = xyxy(token.get("bbox"))
        if bbox:
            draw_box(full, bbox, COLORS["reverse"], token.get("text", ""), thickness=1)

    full_path = OUT_DIR / f"{PAGE_ID}_stage_overlay_full.png"
    cv2.imwrite(str(full_path), full)
    preview = cv2.resize(full, (int(width * 0.45), int(height * 0.45)), interpolation=cv2.INTER_AREA)
    preview_path = OUT_DIR / f"{PAGE_ID}_stage_overlay_preview.png"
    cv2.imwrite(str(preview_path), preview)

    focus_specs = [
        ("b003_l0_pevc_exact", 3, 0, "PE/VC exact baseline"),
        ("b003_l1_pevc_misread", 3, 1, "PE/VC slash misread as PE!VC"),
        ("b004_l1_lemer", 4, 1, "Lerner source adhesion / Lemer output"),
        ("b005_l4_pevc_exact", 5, 4, "PE/VC exact after current binding"),
    ]
    focus_paths: list[tuple[str, Path, str]] = []
    findings: list[str] = []

    for slug, block_idx, line_idx, title in focus_specs:
        line = next((item for item in lines if item.get("block_idx") == block_idx and item.get("line_idx") == line_idx), None)
        if not line:
            continue
        line_bbox = xyxy(line.get("bbox"))
        if line_bbox is None:
            continue
        crop = crop_rect(line_bbox, width, height)
        cx1, cy1, cx2, cy2 = crop
        title_h = 44
        canvas = cv2.copyMakeBorder(
            image[cy1:cy2, cx1:cx2].copy(),
            title_h,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        offset = (cx1, cy1 - title_h)
        draw_box(canvas, line_bbox, COLORS["line"], "Hanwang line", offset=offset, thickness=3)

        row = rows_by_record.get(block_idx)
        if row:
            row_line = row.get("lines", [])[line_idx] if line_idx < len(row.get("lines", [])) else None
            if row_line:
                for char in row_line.get("chars", []):
                    draw_box(canvas, xyxy(char.get("bbox")), COLORS["hanwang_char"], "", offset=offset, thickness=1)

        for char in line.get("chars", []):
            bbox = xyxy(char.get("bbox_page"))
            if bbox and bbox_intersects(bbox, crop):
                draw_box(canvas, bbox, COLORS["engcut_char"], "", offset=offset, thickness=1)

        for binding in bindings:
            if binding.get("block_idx") == block_idx and binding.get("line_idx") == line_idx:
                label = f"{binding.get('text')} {binding.get('binding_status', '')}"
                draw_box(canvas, xyxy(binding.get("bbox")), COLORS["binding"], label, offset=offset, thickness=3)

        for token in reverse_tokens:
            if token.get("block_idx") == block_idx and token.get("line_idx") == line_idx:
                draw_box(canvas, xyxy(token.get("bbox")), COLORS["reverse"], token.get("text", ""), offset=offset, thickness=2)
                for char in token.get("chars", []):
                    draw_box(canvas, xyxy(char.get("bbox")), COLORS["reverse"], "", offset=offset, thickness=1)

        cv2.putText(
            canvas,
            title,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        out = OUT_DIR / f"{PAGE_ID}_{slug}.png"
        cv2.imwrite(str(out), canvas)
        focus_paths.append((slug, out, title))

        exact = [
            b for b in bindings
            if b.get("block_idx") == block_idx and b.get("line_idx") == line_idx
        ]
        reverse_hits = [
            t for t in reverse_tokens
            if t.get("block_idx") == block_idx and t.get("line_idx") == line_idx
        ]
        findings.append(
            f"- `{slug}`: Hanwang line=`{text_preview(line.get('hanwang_text', ''))}`; "
            f"EngCut line=`{text_preview(line.get('eng20_text', ''))}`; "
            f"binding={[b.get('text') + ':' + str(b.get('binding_status')) for b in exact]}; "
            f"reverse={[t.get('text') for t in reverse_hits]}."
        )

    report = OUT_DIR / f"{PAGE_ID}_stage_echo.md"
    report.write_text(
        "\n".join(
            [
                f"# {PAGE_ID} stage box echo",
                "",
                "Legend:",
                "- Blue: Paddle/VL block sent to Hanwang route.",
                "- Green: Hanwang line bbox.",
                "- Orange: Hanwang char bbox before EngCut enhancement.",
                "- Cyan: EngCut chars from whole-line probe.",
                "- Magenta: Paddle token / EngCut exact binding bbox.",
                "- Red: LineCut confidence reverse-select fallback bbox.",
                "",
                "Images:",
                f"- `{preview_path.relative_to(ROOT)}`",
                f"- `{full_path.relative_to(ROOT)}`",
                *[f"- `{path.relative_to(ROOT)}`: {title}" for _slug, path, title in focus_paths],
                "",
                "Evidence summary:",
                *findings,
                "",
                "Interpretation:",
                "- b003/l1 keeps a good Hanwang line bbox, but both Hanwang text and EngCut line text see `PE!VC`; reverse-select only recovers `PE` and `VC`, so this is primarily slash recognition ambiguity, not a missing line box.",
                "- b004/l1 shows `Lemer` where truth expects `Lerner`; the source glyphs are visually adhered, so automatic separation is unreliable and should remain a manual correction case.",
                "- b005/l4 is the control sample: current EngCut line binding recovers the full `PE/VC` bbox, which means the new exact path is viable when line crop and recognizer output agree.",
                "- The PP-OCR/Hanwang line bboxes in these focus samples are vertically loose enough; partial punctuation clipping is more likely from char crop/display padding or downstream char bbox tightness, not from the physical line bbox.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
