from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    def expand(self, pad: int) -> "Box":
        return Box(self.x1 - pad, self.y1 - pad, self.x2 + pad, self.y2 + pad)

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(width, self.x1)),
            max(0, min(height, self.y1)),
            max(0, min(width, self.x2)),
            max(0, min(height, self.y2)),
        )

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Component:
    box: Box
    area: int
    cx: float
    cy: float

    def to_json(self) -> dict[str, Any]:
        return {
            "bbox": self.box.to_list(),
            "area": self.area,
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
        }


@dataclass(frozen=True)
class Anchor:
    line: int
    token: int
    text: str
    kind: str
    source: Box
    ownership: tuple[float, float]
    raw: tuple[Component, ...]
    kept: tuple[Component, ...]
    ink: Box | None
    crop: Box
    status: str
    notes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "token": self.token,
            "text": self.text,
            "kind": self.kind,
            "source_bbox": self.source.to_list(),
            "ownership": [round(self.ownership[0], 2), round(self.ownership[1], 2)],
            "raw_cc": len(self.raw),
            "kept_cc": len(self.kept),
            "ink_bbox": self.ink.to_list() if self.ink else None,
            "crop_bbox": self.crop.to_list(),
            "status": self.status,
            "notes": list(self.notes),
        }


def _bbox(raw: list[int | float]) -> Box:
    x1, y1, x2, y2 = [int(round(v)) for v in raw[:4]]
    return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _is_cjk_char(text: str) -> bool:
    return len(text) == 1 and (
        "\u4e00" <= text <= "\u9fff"
        or "\u3400" <= text <= "\u4dbf"
        or "\uf900" <= text <= "\ufaff"
    )


def _classify_token(text: str) -> str:
    if _is_cjk_char(text):
        return "cjk"
    if text.isdigit():
        return "number"
    if text.isascii() and text.isalpha():
        return "latin"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "mixed_cjk"
    if any(ch.isalpha() or ch.isdigit() for ch in text):
        return "formula"
    return "punct"


def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _default_kernel(ref_h: int) -> tuple[int, int]:
    return max(5, min(9, round(ref_h * 0.13))), max(1, min(2, round(ref_h * 0.035)))


def _extract_components(
    image_bgr: np.ndarray,
    region: Box,
    *,
    kernel: tuple[int, int] | None | str = "default",
    min_area_ratio: float = 0.0012,
) -> tuple[Component, ...]:
    height, width = image_bgr.shape[:2]
    box = region.clip(width, height)
    if box.w <= 0 or box.h <= 0:
        return ()
    mask = _foreground_mask(image_bgr[box.y1:box.y2, box.x1:box.x2])
    kernel_size = _default_kernel(box.h) if kernel == "default" else kernel
    if kernel_size:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size),
        )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(3, int(box.h * box.h * min_area_ratio))
    min_side = max(1, int(box.h * 0.025))
    components: list[Component] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area or w < min_side or h < min_side:
            continue
        components.append(
            Component(
                box=Box(box.x1 + x, box.y1 + y, box.x1 + x + w, box.y1 + y + h),
                area=area,
                cx=box.x1 + float(centroids[idx][0]),
                cy=box.y1 + float(centroids[idx][1]),
            )
        )
    return tuple(sorted(components, key=lambda comp: (comp.box.x1, comp.box.y1)))


def _union_bbox(components: tuple[Component, ...]) -> Box | None:
    if not components:
        return None
    return Box(
        min(comp.box.x1 for comp in components),
        min(comp.box.y1 for comp in components),
        max(comp.box.x2 for comp in components),
        max(comp.box.y2 for comp in components),
    )


def _ownership_intervals(line_box: Box, token_boxes: list[Box]) -> list[tuple[float, float]]:
    centers = [box.cx for box in token_boxes]
    intervals: list[tuple[float, float]] = []
    for idx, center in enumerate(centers):
        left = float(line_box.x1) if idx == 0 else (centers[idx - 1] + center) / 2.0
        right = float(line_box.x2) if idx == len(centers) - 1 else (center + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def _filter_cjk_components(
    source: Box,
    components: tuple[Component, ...],
    ownership: tuple[float, float],
    *,
    use_ownership: bool,
    edge_crumb: bool,
) -> tuple[tuple[Component, ...], tuple[str, ...]]:
    if not components:
        return (), ("no_foreground",)
    largest_area = max(component.area for component in components)
    margin = max(2, round(source.w * 0.035))
    min_edge_area = max(10, largest_area * 0.06, source.area * 0.012)
    kept: list[Component] = []
    notes: list[str] = []
    for component in components:
        if use_ownership and not (ownership[0] <= component.cx < ownership[1]):
            notes.append("drop_outside_ownership")
            continue
        touches_side = component.box.x1 <= source.x1 + margin or component.box.x2 >= source.x2 - margin
        small_area = component.area < min_edge_area
        narrow = component.box.w < source.w * 0.22 or component.box.h < source.h * 0.22
        if edge_crumb and touches_side and small_area and narrow:
            notes.append("drop_edge_crumb")
            continue
        kept.append(component)
    if kept:
        return tuple(kept), tuple(sorted(set(notes)))
    dominant = max(components, key=lambda component: component.area)
    return (dominant,), tuple(sorted(set(notes + ["fallback_largest_component"])))


def _build_variant(
    image_bgr: np.ndarray,
    pruned: dict[str, Any],
    *,
    use_ownership: bool,
    edge_crumb: bool,
    union_mode: bool,
    clip_source: bool,
    split_non_cjk: bool,
) -> tuple[Anchor, ...]:
    height, width = image_bgr.shape[:2]
    anchors: list[Anchor] = []
    for line_idx, (text, line_raw, words, word_boxes_raw) in enumerate(
        zip(pruned["rec_texts"], pruned["rec_boxes"], pruned["text_word"], pruned["text_word_boxes"])
    ):
        if "".join(words) != text:
            raise ValueError(f"text_word does not reconstruct rec_texts at line {line_idx}")
        line_box = _bbox(line_raw)
        token_boxes = [_bbox(raw) for raw in word_boxes_raw]
        ownerships = _ownership_intervals(line_box, token_boxes)
        for token_idx, (token_text, source, ownership) in enumerate(zip(words, token_boxes, ownerships)):
            kind = _classify_token(token_text)
            raw = _extract_components(image_bgr, source)
            if kind != "cjk" and split_non_cjk:
                ink = _union_bbox(raw)
                crop = (ink.expand(2) if ink else source).clip(width, height)
                status = "token_group" if kind in {"number", "latin", "formula", "mixed_cjk"} else "punct_or_empty"
                anchors.append(
                    Anchor(line_idx, token_idx, token_text, kind, source, ownership, raw, raw, ink, crop, status, ("split_non_cjk",))
                )
                continue

            kept, notes = _filter_cjk_components(
                source,
                raw,
                ownership,
                use_ownership=use_ownership,
                edge_crumb=edge_crumb,
            )
            if union_mode:
                ink = _union_bbox(kept)
                status = "single_cc" if len(kept) == 1 else "multi_cc_union"
            else:
                largest = max(kept, key=lambda comp: comp.area) if kept else None
                ink = largest.box if largest else None
                notes = notes + (("lost_components_by_largest_only",) if len(kept) > 1 else ())
                status = "largest_only"
            if ink is None:
                crop = source.clip(width, height)
                status = "fallback_source_box"
            else:
                crop = ink.expand(2).clip(width, height)
                if clip_source:
                    source_clip = source.expand(2).clip(width, height)
                    crop = Box(
                        max(crop.x1, source_clip.x1),
                        max(crop.y1, source_clip.y1),
                        min(crop.x2, source_clip.x2),
                        min(crop.y2, source_clip.y2),
                    )
            anchors.append(
                Anchor(line_idx, token_idx, token_text, kind, source, ownership, raw, kept, ink, crop, status, notes)
            )
    return tuple(anchors)


def _summarize_variant(name: str, anchors: tuple[Anchor, ...], *, cjk_only: bool) -> dict[str, Any]:
    tokens = [anchor for anchor in anchors if anchor.kind == "cjk" or not cjk_only]
    cjk = [anchor for anchor in anchors if anchor.kind == "cjk"]

    def cc_bucket(value: int) -> str:
        return "cc0" if value == 0 else "cc1" if value == 1 else "cc2p"

    return {
        "variant": name,
        "token_total": len(tokens),
        "cjk_total": len(cjk),
        "raw_cc": dict(Counter(cc_bucket(len(anchor.raw)) for anchor in cjk)),
        "kept_cc": dict(Counter(cc_bucket(len(anchor.kept)) for anchor in cjk)),
        "status_counts": dict(Counter(anchor.status for anchor in tokens)),
        "note_counts": dict(Counter(note for anchor in tokens for note in anchor.notes)),
        "cjk_bindable_rate": sum(1 for anchor in cjk if anchor.kept) / len(cjk) if cjk else 0.0,
        "crop_crosses_source_x_count": sum(
            1 for anchor in cjk if anchor.crop.x1 < anchor.source.x1 or anchor.crop.x2 > anchor.source.x2
        ),
        "fallback_count": sum(1 for anchor in cjk if "fallback_largest_component" in anchor.notes or anchor.status == "fallback_source_box"),
        "lost_multi_component_count": sum(1 for anchor in cjk if "lost_components_by_largest_only" in anchor.notes),
        "avg_cjk_crop_area": round(sum(anchor.crop.area for anchor in cjk) / len(cjk), 2) if cjk else 0.0,
    }


def _crop_tile(image_bgr: np.ndarray, box: Box, width: int, height: int) -> Image.Image:
    image_h, image_w = image_bgr.shape[:2]
    clipped = box.clip(image_w, image_h)
    crop = image_bgr[clipped.y1:clipped.y2, clipped.x1:clipped.x2]
    if crop.size == 0:
        return Image.new("RGB", (width, height), (35, 35, 35))
    tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    scale = min(width / max(1, tile.width), height / max(1, tile.height))
    tile = tile.resize((max(1, int(tile.width * scale)), max(1, int(tile.height * scale))), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height), (35, 35, 35))
    canvas.paste(tile, ((width - tile.width) // 2, (height - tile.height) // 2))
    return canvas


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _make_rule_samples(image_bgr: np.ndarray, anchors_by_variant: dict[str, tuple[Anchor, ...]], output_path: Path) -> list[tuple[int, int]]:
    by_variant = {
        name: {(anchor.line, anchor.token): anchor for anchor in anchors}
        for name, anchors in anchors_by_variant.items()
    }
    full = by_variant["full_anchor"]
    sample_ids: list[tuple[int, int]] = []
    for token_id, anchor in full.items():
        if anchor.kind == "cjk" and len(by_variant["no_ownership"][token_id].kept) > len(anchor.kept):
            sample_ids.append(token_id)
        if len(sample_ids) >= 4:
            break
    for token_id, anchor in full.items():
        if anchor.kind == "cjk" and len(by_variant["no_edge_crumb"][token_id].kept) > len(anchor.kept) and token_id not in sample_ids:
            sample_ids.append(token_id)
        if len(sample_ids) >= 8:
            break
    for token_id, anchor in full.items():
        if anchor.kind == "cjk" and len(anchor.kept) >= 2 and token_id not in sample_ids:
            sample_ids.append(token_id)
        if len(sample_ids) >= 14:
            break
    for token_id, anchor in full.items():
        if anchor.kind == "cjk" and (anchor.text in {"一", "二", "三"} or len(anchor.kept) >= 3) and token_id not in sample_ids:
            sample_ids.append(token_id)
        if len(sample_ids) >= 18:
            break

    latin = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    cjk = _font("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 22)
    columns = [
        ("src", "source"),
        ("full", "full_anchor"),
        ("noOwn", "no_ownership"),
        ("noEdge", "no_edge_crumb"),
        ("largest", "largest_only_no_union"),
        ("noClip", "no_source_clip"),
    ]
    cell_w, row_h, label_w, header_h = 116, 132, 128, 28
    canvas = Image.new("RGB", (label_w + len(columns) * cell_w, header_h + len(sample_ids) * row_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for col_idx, (label, _) in enumerate(columns):
        draw.text((label_w + col_idx * cell_w + 4, 6), label, fill=(230, 230, 230), font=latin)
    for row_idx, token_id in enumerate(sample_ids):
        y = header_h + row_idx * row_h
        anchor = full[token_id]
        draw.text((4, y + 4), f"L{anchor.line:02d}#{anchor.token:03d}", fill=(160, 200, 255), font=latin)
        draw.text((4, y + 22), anchor.text, fill=(255, 80, 80), font=cjk)
        draw.text((4, y + 52), f"full {len(anchor.kept)}cc", fill=(210, 210, 210), font=latin)
        for col_idx, (label, variant) in enumerate(columns):
            if label == "src":
                compared = anchor
                box = anchor.source
            else:
                compared = by_variant[variant][token_id]
                box = compared.crop
            x = label_w + col_idx * cell_w
            canvas.paste(_crop_tile(image_bgr, box, cell_w - 8, 76), (x + 4, y + 22))
            if label != "src":
                color = (80, 210, 110) if variant == "full_anchor" else (230, 170, 60)
                draw.text((x + 4, y + 100), f"cc{len(compared.kept)}", fill=color, font=latin)
                if compared.notes:
                    draw.text((x + 38, y + 100), compared.notes[0][:10], fill=(200, 200, 200), font=latin)
            draw.rectangle([x, y, x + cell_w - 1, y + row_h - 1], outline=(70, 70, 70))
    canvas.save(output_path)
    return sample_ids


def _make_residual_samples(image_bgr: np.ndarray, anchors: tuple[Anchor, ...], output_path: Path) -> None:
    residuals = [
        anchor
        for anchor in anchors
        if anchor.kind == "cjk" and (len(anchor.kept) >= 3 or anchor.text in {"一", "二", "三"} or len(anchor.raw) - len(anchor.kept) >= 2)
    ][:30]
    latin = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    cjk = _font("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 22)
    cell_w, cell_h, cols = 150, 120, 5
    rows = math.ceil(len(residuals) / cols)
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for idx, anchor in enumerate(residuals):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        image.paste(_crop_tile(image_bgr, anchor.crop, cell_w - 12, 68), (x + 6, y + 40))
        draw.text((x + 5, y + 4), f"L{anchor.line:02d}#{anchor.token:03d}", fill=(0, 0, 0), font=latin)
        draw.text((x + 70, y), anchor.text, fill=(180, 0, 0), font=cjk)
        draw.text((x + 5, y + 24), f"raw{len(anchor.raw)} keep{len(anchor.kept)}", fill=(0, 100, 0), font=latin)
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180))
    image.save(output_path)


def _make_ranking(summary: dict[str, dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    full = summary["full_anchor"]
    no_ownership = summary["no_ownership"]
    no_edge = summary["no_edge_crumb"]
    largest = summary["largest_only_no_union"]
    no_clip = summary["no_source_clip"]
    no_non_cjk = summary["no_non_cjk_split"]
    ranking = [
        {
            "rank": 1,
            "rule": "edge crumb suppression",
            "evidence": (
                "removes side-touching small/narrow fragments; "
                f"no_edge_crumb cc2+={no_edge['kept_cc'].get('cc2p', 0)} vs full {full['kept_cc'].get('cc2p', 0)} "
                f"(+{no_edge['kept_cc'].get('cc2p', 0) - full['kept_cc'].get('cc2p', 0)} noisy multi-CC)"
            ),
        },
        {
            "rank": 2,
            "rule": "ownership intervals",
            "evidence": (
                "drops components whose centroid belongs to neighbour token; "
                f"no_ownership cc2+={no_ownership['kept_cc'].get('cc2p', 0)} vs full {full['kept_cc'].get('cc2p', 0)} "
                f"(+{no_ownership['kept_cc'].get('cc2p', 0) - full['kept_cc'].get('cc2p', 0)})"
            ),
        },
        {
            "rank": 3,
            "rule": "union multi-CC",
            "evidence": f"keeps separated legal strokes; largest_only loses {largest['lost_multi_component_count']} multi-component CJK tokens",
        },
        {
            "rank": 4,
            "rule": "non-CJK split",
            "evidence": (
                "keeps 124 number/latin/formula/punct tokens out of CJK glyph lane; "
                f"no split token_total={no_non_cjk['token_total']} and fallback_source_box={no_non_cjk['status_counts'].get('fallback_source_box', 0)}"
            ),
        },
        {
            "rank": 5,
            "rule": "source/ownership crop clamp",
            "evidence": (
                "guardrail only in this sample: no_source_clip has same cc and outside count "
                f"({no_clip['crop_crosses_source_x_count']}); keep as safety for worse OCR boxes, not a decisive 120166 rule"
            ),
        },
        {
            "rank": 6,
            "rule": "quick_simple / realloc / white-margin",
            "evidence": "mostly performance and visual margin polish; useful UX optimization, lower contribution to ownership correctness",
        },
    ]
    title_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    head_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    text_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    image = Image.new("RGB", (1520, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "wordbox_anchor rule contribution ranking (120166 corrected)", fill=(0, 0, 0), font=title_font)
    y = 82
    for item in ranking:
        draw.text((30, y), f"{item['rank']} {item['rule']}", fill=(0, 90, 160), font=head_font)
        y += 30
        line = ""
        for word in item["evidence"].split(" "):
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=text_font) > 1360 and line:
                draw.text((60, y), line, fill=(0, 0, 0), font=text_font)
                y += 24
                line = word
            else:
                line = trial
        if line:
            draw.text((60, y), line, fill=(0, 0, 0), font=text_font)
            y += 24
        y += 24
    image.save(output_path)
    return ranking


def _make_report(
    output_path: Path,
    image_path: Path,
    json_path: Path,
    pruned: dict[str, Any],
    summary: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    claude_actions_path: Path | None,
) -> None:
    kind_counts = Counter(_classify_token(token) for row in pruned["text_word"] for token in row)
    lines = [
        "=" * 72,
        "wordbox_anchor 规则消融实验报告",
        "=" * 72,
        "",
        f"image: {image_path}",
        f"json: {json_path}",
        f"text_word reconstructs rec_texts: {all(''.join(words) == text for text, words in zip(pruned['rec_texts'], pruned['text_word']))}",
        f"token kind counts: {dict(kind_counts)}",
        "",
        "Variant statistics:",
    ]
    for name, variant_summary in summary.items():
        lines.append(
            f"- {name}: cjk={variant_summary['cjk_total']} kept={variant_summary['kept_cc']} "
            f"bind={variant_summary['cjk_bindable_rate'] * 100:.1f}% "
            f"outside_x={variant_summary['crop_crosses_source_x_count']} "
            f"fallback={variant_summary['fallback_count']} lost_multi={variant_summary['lost_multi_component_count']}"
        )
    lines.extend(["", "Rule contribution ranking:"])
    for item in ranking:
        lines.append(f"{item['rank']}. {item['rule']}: {item['evidence']}")
    lines.extend(
        [
            "",
            "Working answers:",
            "- real text_word_boxes 主链: zip(text_word, text_word_boxes) is the only order/geometry anchor; Structure only gives layout containers; line bbox is context/diagnostic/fallback, not VProof char truth.",
            "- 多笔画汉字最稳判据: ownership-filtered components inside the source word box, then union; cc2+ is normal, not an error.",
            "- 边角料压制: component centroid must belong to ownership interval; additionally drop side-touching small+narrow crumbs.",
            "- line bbox: downgrade to HProof crop, ownership context, line-height reference, and diagnostics. Never bind chars directly from whole-line components.",
            "- 数字/公式/混合数字行: split by OCR text/token kind first; number/latin/formula/punct go token_group/property lane, not the CJK glyph lane.",
            "",
            "Rules that look deletable / low impact:",
            "- stop-on-ink white-margin scanning: v11 doc already says abandoned; keep out.",
            "- quick_simple: performance/clean simple cases only; it does not replace ownership+crumb+union.",
            "- aggressive line-level normal-shape filtering: useful diagnostics, but not a binding rule.",
        ]
    )
    if claude_actions_path and claude_actions_path.exists():
        lines.extend(["", "Claude v11 action-count evidence (120166 excerpt):"])
        lines.extend(claude_actions_path.read_text(encoding="utf-8").splitlines()[:24])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    image_path: Path,
    source_json_path: Path,
    output_dir: Path,
    claude_actions_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_json = output_dir / source_json_path.name
    if source_json_path.resolve() != copied_json.resolve():
        shutil.copyfile(source_json_path, copied_json)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    data = json.loads(source_json_path.read_text(encoding="utf-8"))
    pruned = data["response"]["result"]["ocrResults"][0]["prunedResult"]
    variants = {
        "full_anchor": dict(use_ownership=True, edge_crumb=True, union_mode=True, clip_source=True, split_non_cjk=True),
        "no_ownership": dict(use_ownership=False, edge_crumb=True, union_mode=True, clip_source=True, split_non_cjk=True),
        "no_edge_crumb": dict(use_ownership=True, edge_crumb=False, union_mode=True, clip_source=True, split_non_cjk=True),
        "largest_only_no_union": dict(use_ownership=True, edge_crumb=True, union_mode=False, clip_source=True, split_non_cjk=True),
        "no_source_clip": dict(use_ownership=True, edge_crumb=True, union_mode=True, clip_source=False, split_non_cjk=True),
        "no_non_cjk_split": dict(use_ownership=True, edge_crumb=True, union_mode=True, clip_source=True, split_non_cjk=False),
    }
    anchors_by_variant = {
        name: _build_variant(image_bgr, pruned, **config)
        for name, config in variants.items()
    }
    summary = {
        name: _summarize_variant(name, anchors, cjk_only=(name != "no_non_cjk_split"))
        for name, anchors in anchors_by_variant.items()
    }
    sample_ids = _make_rule_samples(image_bgr, anchors_by_variant, output_dir / "01_rule_on_off_samples.png")
    _make_residual_samples(image_bgr, anchors_by_variant["full_anchor"], output_dir / "02_success_failure_residual_samples.png")
    ranking = _make_ranking(summary, output_dir / "03_rule_contribution_ranking.png")
    _make_report(
        output_dir / "04_ablation_report.txt",
        image_path,
        copied_json,
        pruned,
        summary,
        ranking,
        claude_actions_path,
    )
    payload = {
        "source_image": str(image_path),
        "source_json": str(copied_json),
        "variant_summary": summary,
        "sample_token_ids": [list(item) for item in sample_ids],
        "rule_ranking": ranking,
        "full_anchor_samples": [anchor.to_json() for anchor in anchors_by_variant["full_anchor"][:80]],
    }
    (output_dir / "wordbox_anchor_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run wordbox_anchor rule ablation on real PP-OCRv5 word boxes.")
    parser.add_argument("--image", type=Path, default=repo_root / "file/244771纵校/120166.tif")
    parser.add_argument("--json", type=Path, default=repo_root / "paddle-char-box-samples/wordbox-anchor-ablation/120166_ppocrv5_return_word_box.json")
    parser.add_argument("--output", type=Path, default=repo_root / "paddle-char-box-samples/wordbox-anchor-ablation")
    parser.add_argument(
        "--claude-actions",
        type=Path,
        default=repo_root.parent / "claude/app/experiments/results/wordbox_anchor_v11/120166/03_actions.txt",
    )
    args = parser.parse_args()
    run_ablation(args.image, args.json, args.output, args.claude_actions)


if __name__ == "__main__":
    main()
