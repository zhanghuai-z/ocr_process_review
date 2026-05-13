from __future__ import annotations

import argparse
import json
import math
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
class ZoneSpec:
    name: str
    left_inset_ratio: float
    right_inset_ratio: float
    y_inset_ratio: float = 0.15
    tol_ratio: float = 0.08
    min_tol: int = 3
    y_pad_ratio: float = 0.15
    stroke_extension_ratio: float = 0.0
    adaptive_hard_bound_ratio: float = 0.0
    weak_y_overlap_ratio: float = 0.25
    weak_area_ratio: float = 0.05
    rescue_left_guard_ratio: float = 0.0
    rescue_right_guard_ratio: float = 0.05


@dataclass(frozen=True)
class TokenRef:
    line: int
    token: int
    text: str
    line_box: Box
    source: Box
    ownership: tuple[float, float]

    @property
    def id(self) -> tuple[int, int]:
        return (self.line, self.token)


@dataclass(frozen=True)
class ZoneEval:
    token: TokenRef
    spec: ZoneSpec
    initial_hard_bound: Box
    hard_bound: Box
    stroke_bound: Box
    blue_expanded_sides: tuple[str, ...]
    strong: Box
    rescue_domain: tuple[float, float]
    components: tuple[Component, ...]
    component_labels: tuple[str, ...]
    ink: Box | None
    crop: Box
    no_strong: bool

    @property
    def counts(self) -> Counter[str]:
        return Counter(self.component_labels)

    def to_json(self) -> dict[str, Any]:
        return {
            "line": self.token.line,
            "token": self.token.token,
            "text": self.token.text,
            "spec": self.spec.name,
            "source_bbox": self.token.source.to_list(),
            "ownership": [round(self.token.ownership[0], 2), round(self.token.ownership[1], 2)],
            "initial_hard_bound": self.initial_hard_bound.to_list(),
            "hard_bound": self.hard_bound.to_list(),
            "stroke_bound": self.stroke_bound.to_list(),
            "blue_expanded_sides": list(self.blue_expanded_sides),
            "strong": self.strong.to_list(),
            "rescue_domain": [round(self.rescue_domain[0], 2), round(self.rescue_domain[1], 2)],
            "counts": dict(self.counts),
            "ink_bbox": self.ink.to_list() if self.ink else None,
            "crop_bbox": self.crop.to_list(),
            "no_strong": self.no_strong,
            "components": [
                {**component.to_json(), "label": label}
                for component, label in zip(self.components, self.component_labels)
            ],
        }


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _bbox(raw: list[int | float]) -> Box:
    x1, y1, x2, y2 = [int(round(value)) for value in raw[:4]]
    return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _is_cjk_char(text: str) -> bool:
    return len(text) == 1 and (
        "\u4e00" <= text <= "\u9fff"
        or "\u3400" <= text <= "\u4dbf"
        or "\uf900" <= text <= "\ufaff"
    )


def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _extract_components(image_bgr: np.ndarray, region: Box, *, min_area_ratio: float = 0.0012) -> tuple[Component, ...]:
    height, width = image_bgr.shape[:2]
    box = region.clip(width, height)
    if box.w <= 0 or box.h <= 0:
        return ()
    mask = _foreground_mask(image_bgr[box.y1:box.y2, box.x1:box.x2])
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(3, int(box.h * box.h * min_area_ratio))
    min_side = max(1, int(box.h * 0.025))
    components: list[Component] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
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
    return tuple(sorted(components, key=lambda item: (item.box.x1, item.box.y1)))


def _intersects(a: Box, b: Box) -> bool:
    return min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1)


def _touch_sides(box: Box, bound: Box, margin: int = 1) -> tuple[str, ...]:
    sides: list[str] = []
    if box.x1 <= bound.x1 + margin:
        sides.append("L")
    if box.x2 >= bound.x2 - margin:
        sides.append("R")
    if box.y1 <= bound.y1 + margin:
        sides.append("U")
    if box.y2 >= bound.y2 - margin:
        sides.append("D")
    return tuple(sides)


def _expand_sides(bound: Box, sides: tuple[str, ...], pad_x: int, pad_y: int, width: int, height: int) -> Box:
    side_set = set(sides)
    return Box(
        bound.x1 - (pad_x if "L" in side_set else 0),
        bound.y1 - (pad_y if "U" in side_set else 0),
        bound.x2 + (pad_x if "R" in side_set else 0),
        bound.y2 + (pad_y if "D" in side_set else 0),
    ).clip(width, height)


def _ownership_intervals(line_box: Box, token_boxes: list[Box]) -> list[tuple[float, float]]:
    centers = [box.cx for box in token_boxes]
    intervals: list[tuple[float, float]] = []
    for idx, center in enumerate(centers):
        left = float(line_box.x1) if idx == 0 else (centers[idx - 1] + center) / 2.0
        right = float(line_box.x2) if idx == len(centers) - 1 else (center + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def _load_tokens(pruned: dict[str, Any]) -> tuple[TokenRef, ...]:
    tokens: list[TokenRef] = []
    for line_idx, (text, line_raw, words, word_boxes_raw) in enumerate(
        zip(pruned["rec_texts"], pruned["rec_boxes"], pruned["text_word"], pruned["text_word_boxes"])
    ):
        if "".join(words) != text:
            raise ValueError(f"text_word does not reconstruct rec_texts at line {line_idx}")
        line_box = _bbox(line_raw)
        word_boxes = [_bbox(raw) for raw in word_boxes_raw]
        ownerships = _ownership_intervals(line_box, word_boxes)
        for token_idx, (word, source, ownership) in enumerate(zip(words, word_boxes, ownerships)):
            if _is_cjk_char(word):
                tokens.append(TokenRef(line_idx, token_idx, word, line_box, source, ownership))
    return tuple(tokens)


def _evaluate_zone(image_bgr: np.ndarray, token: TokenRef, spec: ZoneSpec) -> ZoneEval:
    height, width = image_bgr.shape[:2]
    source = token.source
    source_w = max(1, source.w)
    source_h = max(1, source.h)
    tol = max(spec.min_tol, int(source_w * spec.tol_ratio))
    y_pad = max(2, int(source_h * spec.y_pad_ratio))
    own_lo, own_hi = token.ownership
    initial_hard_bound = Box(
        max(0, int(own_lo) - tol),
        max(0, source.y1 - y_pad),
        min(width, int(own_hi) + tol + 1),
        min(height, source.y2 + y_pad),
    )
    hard_bound = initial_hard_bound
    strong = Box(
        source.x1 + max(0, int(source_w * spec.left_inset_ratio)),
        source.y1 + max(1, int(source_h * spec.y_inset_ratio)),
        source.x2 - max(0, int(source_w * spec.right_inset_ratio)),
        source.y2 - max(1, int(source_h * spec.y_inset_ratio)),
    )
    if strong.x1 >= strong.x2 or strong.y1 >= strong.y2:
        return ZoneEval(token, spec, initial_hard_bound, hard_bound, hard_bound, (), strong, (own_lo, own_hi), (), (), None, source.clip(width, height), True)

    def build_stroke_bound(bound: Box) -> Box:
        if spec.stroke_extension_ratio <= 0:
            return bound
        stroke_pad_x = max(tol * 2, int(source_w * spec.stroke_extension_ratio), 6)
        stroke_pad_y = max(y_pad * 2, int(source_h * spec.stroke_extension_ratio), 6)
        return Box(
            bound.x1 - stroke_pad_x,
            bound.y1 - stroke_pad_y,
            bound.x2 + stroke_pad_x,
            bound.y2 + stroke_pad_y,
        ).clip(width, height)

    def extract_strong(bound: Box) -> tuple[Box, tuple[Component, ...], list[Component]]:
        search_bound = build_stroke_bound(bound)
        found_components = _extract_components(image_bgr, search_bound)
        found_strong = [component for component in found_components if _intersects(component.box, strong)]
        return search_bound, found_components, found_strong

    stroke_bound, components, strong_components = extract_strong(hard_bound)
    blue_sides: tuple[str, ...] = ()
    if spec.adaptive_hard_bound_ratio > 0 and strong_components:
        sides = sorted({side for component in strong_components for side in _touch_sides(component.box, hard_bound)})
        blue_sides = tuple(sides)
        if blue_sides:
            pad_x = max(1, int(source_w * spec.adaptive_hard_bound_ratio))
            pad_y = max(1, int(source_h * spec.adaptive_hard_bound_ratio))
            hard_bound = _expand_sides(hard_bound, blue_sides, pad_x, pad_y, width, height)
            stroke_bound, components, strong_components = extract_strong(hard_bound)

    if spec.stroke_extension_ratio > 0:
        for _ in range(3):
            if not strong_components:
                break
            stroke_sides = tuple(sorted({side for component in strong_components for side in _touch_sides(component.box, stroke_bound)}))
            if not stroke_sides:
                break
            extend_x = max(2, int(source_w * spec.stroke_extension_ratio))
            extend_y = max(2, int(source_h * spec.stroke_extension_ratio))
            new_stroke_bound = _expand_sides(stroke_bound, stroke_sides, extend_x, extend_y, width, height)
            if new_stroke_bound == stroke_bound:
                break
            stroke_bound = new_stroke_bound
            components = _extract_components(image_bgr, stroke_bound)
            strong_components = [component for component in components if _intersects(component.box, strong)]

    main_area = max((component.area for component in strong_components), default=max((c.area for c in components), default=0))
    weak_area_threshold = max(8, int(main_area * spec.weak_area_ratio))
    rescue_lo = max(own_lo, source.x1 + source_w * spec.rescue_left_guard_ratio)
    rescue_hi = min(own_hi, source.x2 - source_w * spec.rescue_right_guard_ratio)

    labels: list[str] = []
    for component in components:
        if _intersects(component.box, strong):
            labels.append("strong")
            continue
        y_overlap = max(0, min(component.box.y2, source.y2) - max(component.box.y1, source.y1))
        if (
            rescue_lo <= component.cx <= rescue_hi
            and y_overlap >= source_h * spec.weak_y_overlap_ratio
            and component.area >= weak_area_threshold
        ):
            labels.append("weak_rescued")
        else:
            labels.append("anti_dropped")
    kept = [
        component
        for component, label in zip(components, labels)
        if label in {"strong", "weak_rescued"}
    ]
    ink: Box | None = None
    crop = source.clip(width, height)
    if kept:
        clip_lo = int(round(own_lo))
        clip_hi = int(round(own_hi))
        xs1: list[int] = []
        ys1: list[int] = []
        xs2: list[int] = []
        ys2: list[int] = []
        for component in kept:
            x1 = max(component.box.x1, clip_lo)
            x2 = min(component.box.x2, clip_hi)
            if x2 <= x1:
                continue
            xs1.append(x1)
            ys1.append(component.box.y1)
            xs2.append(x2)
            ys2.append(component.box.y2)
        if xs1:
            ink = Box(min(xs1), min(ys1), max(xs2), max(ys2))
            pad = 4
            margin_y = max(pad, int(source_h * 0.10))
            crop = Box(
                max(0, clip_lo, ink.x1 - pad),
                max(0, ink.y1 - margin_y),
                min(width, clip_hi, ink.x2 + pad),
                min(height, ink.y2 + margin_y),
            ).clip(width, height)
    return ZoneEval(
        token=token,
        spec=spec,
        initial_hard_bound=initial_hard_bound,
        hard_bound=hard_bound,
        stroke_bound=stroke_bound,
        blue_expanded_sides=blue_sides,
        strong=strong,
        rescue_domain=(rescue_lo, rescue_hi),
        components=components,
        component_labels=tuple(labels),
        ink=ink,
        crop=crop,
        no_strong=not strong_components,
    )


def _quality_label(current: ZoneEval, adjusted: ZoneEval) -> str:
    source = current.token.source
    components_in_source = [component for component in current.components if _intersects(component.box, source)]
    if not components_in_source:
        return "Q3 no ink / fallback"
    ink = Box(
        min(component.box.x1 for component in components_in_source),
        min(component.box.y1 for component in components_in_source),
        max(component.box.x2 for component in components_in_source),
        max(component.box.y2 for component in components_in_source),
    )
    left_margin = (ink.x1 - source.x1) / max(1, source.w)
    right_margin = (source.x2 - ink.x2) / max(1, source.w)
    delta = _effective_counts(current) != _effective_counts(adjusted) or current.crop != adjusted.crop or current.no_strong != adjusted.no_strong
    if delta:
        return "Q2 param-sensitive"
    if current.counts.get("anti_dropped", 0) or current.counts.get("weak_rescued", 0) or len(components_in_source) >= 3:
        return "Q2 loose/noisy"
    if left_margin < 0.02 or right_margin < 0.02 or left_margin > 0.25 or right_margin > 0.25:
        return "Q1 edge-risk"
    return "Q0 stable"


def _outside(outer: Box, inner: Box) -> bool:
    return inner.x1 < outer.x1 or inner.y1 < outer.y1 or inner.x2 > outer.x2 or inner.y2 > outer.y2


def _effective_counts(zone: ZoneEval) -> tuple[int, int]:
    counts = zone.counts
    return counts.get("strong", 0), counts.get("weak_rescued", 0)


def _breaks_hard_bound(zone: ZoneEval) -> bool:
    if _outside(zone.hard_bound, zone.crop):
        return True
    return any(
        label in {"strong", "weak_rescued"} and _outside(zone.hard_bound, component.box)
        for component, label in zip(zone.components, zone.component_labels)
    )


def _draw_translucent(base: Image.Image, rect: tuple[int, int, int, int], fill: tuple[int, int, int, int]) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(rect, fill=fill)
    base.alpha_composite(overlay)


def _rect_local(box: Box, origin: Box, scale: float) -> tuple[int, int, int, int]:
    return (
        int(round((box.x1 - origin.x1) * scale)),
        int(round((box.y1 - origin.y1) * scale)),
        int(round((box.x2 - origin.x1) * scale)),
        int(round((box.y2 - origin.y1) * scale)),
    )


def _draw_zone_panel(
    image_bgr: np.ndarray,
    zone: ZoneEval,
    *,
    panel_w: int,
    panel_h: int,
    latin: ImageFont.FreeTypeFont,
) -> Image.Image:
    height, width = image_bgr.shape[:2]
    view_boxes = [zone.hard_bound, zone.stroke_bound, zone.token.source, zone.crop]
    if zone.ink:
        view_boxes.append(zone.ink)
    view = Box(
        min(box.x1 for box in view_boxes),
        min(box.y1 for box in view_boxes),
        max(box.x2 for box in view_boxes),
        max(box.y2 for box in view_boxes),
    ).expand(max(12, zone.token.source.w // 2)).clip(width, height)
    crop = image_bgr[view.y1:view.y2, view.x1:view.x2]
    if crop.size == 0:
        return Image.new("RGB", (panel_w, panel_h), (30, 30, 30))
    tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
    scale = min((panel_w - 12) / max(1, tile.width), (panel_h - 44) / max(1, tile.height))
    tile = tile.resize((max(1, int(tile.width * scale)), max(1, int(tile.height * scale))), Image.Resampling.NEAREST)
    panel = Image.new("RGBA", (panel_w, panel_h), (248, 248, 248, 255))
    ox, oy = (panel_w - tile.width) // 2, 34
    panel.alpha_composite(tile, (ox, oy))
    shifted_origin = Box(view.x1 - int(round(ox / scale)), view.y1 - int(round(oy / scale)), view.x2, view.y2)

    def rect(box: Box) -> tuple[int, int, int, int]:
        return _rect_local(box, shifted_origin, scale)

    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), zone.spec.name, fill=(20, 20, 20), font=latin)

    source = zone.token.source
    left_anti = Box(source.x1, source.y1, zone.strong.x1, source.y2)
    right_anti = Box(zone.strong.x2, source.y1, source.x2, source.y2)
    _draw_translucent(panel, rect(left_anti), (255, 80, 80, 65))
    _draw_translucent(panel, rect(right_anti), (255, 80, 80, 65))
    _draw_translucent(panel, rect(zone.strong), (0, 190, 80, 70))

    rescue_box = Box(int(round(zone.rescue_domain[0])), source.y1, int(round(zone.rescue_domain[1])), source.y2)
    _draw_translucent(panel, rect(rescue_box), (255, 190, 0, 42))

    draw = ImageDraw.Draw(panel)
    draw.rectangle(rect(zone.stroke_bound), outline=(150, 70, 210), width=2)
    draw.rectangle(rect(zone.hard_bound), outline=(60, 120, 255), width=2)
    draw.rectangle(rect(source), outline=(0, 0, 0), width=2)
    draw.rectangle(rect(zone.strong), outline=(0, 150, 60), width=2)
    draw.rectangle(rect(rescue_box), outline=(230, 150, 0), width=2)
    own_box = Box(int(round(zone.token.ownership[0])), source.y1, int(round(zone.token.ownership[1])), source.y2)
    draw.rectangle(rect(own_box), outline=(0, 180, 210), width=1)

    colors = {
        "strong": (0, 150, 60),
        "weak_rescued": (230, 145, 0),
        "anti_dropped": (220, 0, 0),
    }
    for component, label in zip(zone.components, zone.component_labels):
        draw.rectangle(rect(component.box), outline=colors[label], width=3)

    counts = zone.counts
    draw.text(
        (8, panel_h - 24),
        f"S{counts.get('strong', 0)} R{counts.get('weak_rescued', 0)} D{counts.get('anti_dropped', 0)}",
        fill=(20, 20, 20),
        font=latin,
    )
    return panel.convert("RGB")


def _draw_crop_panel(
    image_bgr: np.ndarray,
    zone: ZoneEval,
    *,
    panel_w: int,
    panel_h: int,
    latin: ImageFont.FreeTypeFont,
) -> Image.Image:
    height, width = image_bgr.shape[:2]
    crop_box = zone.crop.clip(width, height)
    crop = image_bgr[crop_box.y1:crop_box.y2, crop_box.x1:crop_box.x2]
    panel = Image.new("RGB", (panel_w, panel_h), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{zone.spec.name} crop", fill=(20, 20, 20), font=latin)
    draw.text((8, 28), f"{crop_box.w}x{crop_box.h}", fill=(80, 80, 80), font=latin)
    if crop.size == 0:
        draw.text((8, 58), "empty", fill=(180, 0, 0), font=latin)
        return panel
    tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    scale = min((panel_w - 16) / max(1, tile.width), (panel_h - 78) / max(1, tile.height))
    tile = tile.resize((max(1, int(tile.width * scale)), max(1, int(tile.height * scale))), Image.Resampling.NEAREST)
    panel.paste(tile, ((panel_w - tile.width) // 2, 62 + (panel_h - 78 - tile.height) // 2))
    draw.rectangle([4, 54, panel_w - 5, panel_h - 5], outline=(120, 120, 120))
    return panel


def _select_samples(
    tokens: tuple[TokenRef, ...],
    current_evals: dict[tuple[int, int], ZoneEval],
    adjusted_evals: dict[tuple[int, int], ZoneEval],
    limit: int = 10,
) -> list[TokenRef]:
    scored: list[tuple[int, TokenRef]] = []
    for token in tokens:
        current = current_evals[token.id]
        adjusted = adjusted_evals[token.id]
        counts = current.counts
        score = 0
        if _breaks_hard_bound(current) or _breaks_hard_bound(adjusted):
            score += 220
        if current.crop != adjusted.crop:
            score += 180
        if _effective_counts(current) != _effective_counts(adjusted) or current.no_strong != adjusted.no_strong:
            score += 120
        score += counts.get("anti_dropped", 0) * 25
        score += counts.get("weak_rescued", 0) * 20
        score += max(0, len(current.components) - 1) * 10
        if token.text in {"一", "二", "三"}:
            score += 45
        if current.no_strong:
            score += 60
        if score:
            scored.append((score, token))
    scored.sort(key=lambda item: (-item[0], item[1].line, item[1].token))
    selected: list[TokenRef] = []
    used_texts: Counter[str] = Counter()
    for _, token in scored:
        if used_texts[token.text] >= 2:
            continue
        selected.append(token)
        used_texts[token.text] += 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for token in tokens:
            if token not in selected:
                selected.append(token)
            if len(selected) >= limit:
                break
    return selected


def _make_scope_comparison(
    image_bgr: np.ndarray,
    samples: list[TokenRef],
    current_evals: dict[tuple[int, int], ZoneEval],
    adjusted_evals: dict[tuple[int, int], ZoneEval],
    output_path: Path,
    *,
    page_id: str,
) -> None:
    latin = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    latin_small = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    cjk = _font("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 24)
    scope_w, crop_w, row_h, label_w, header_h = 310, 150, 250, 250, 96
    image = Image.new("RGB", (label_w + (scope_w + crop_w) * 2, header_h + len(samples) * row_h), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), f"Radiation scope comparison on real PP-OCRv5 word boxes - page {page_id}", fill=(0, 0, 0), font=latin)
    draw.text(
        (16, 38),
        "green=strong, red=anti/weak, orange=rescue, purple=stroke-search, blue=initial hard_bound only",
        fill=(40, 40, 40),
        font=latin_small,
    )
    draw.text((label_w + 8, 62), "current scope", fill=(20, 20, 20), font=latin_small)
    draw.text((label_w + scope_w + 8, 62), "current final crop", fill=(20, 20, 20), font=latin_small)
    draw.text((label_w + scope_w + crop_w + 8, 62), "adjusted scope", fill=(20, 20, 20), font=latin_small)
    draw.text((label_w + scope_w * 2 + crop_w + 8, 62), "adjusted final crop", fill=(20, 20, 20), font=latin_small)
    draw.text((label_w + 8, 80), "cur: red L10/R20", fill=(80, 80, 80), font=latin_small)
    draw.text((label_w + scope_w + crop_w + 8, 80), "adj: red L0/R25; blue +5% on touch", fill=(80, 80, 80), font=latin_small)
    for row, token in enumerate(samples):
        y = header_h + row * row_h
        current = current_evals[token.id]
        adjusted = adjusted_evals[token.id]
        qlabel = _quality_label(current, adjusted)
        draw.text((10, y + 12), f"L{token.line:02d}#{token.token:03d}", fill=(0, 90, 160), font=latin)
        draw.text((10, y + 38), token.text, fill=(180, 0, 0), font=cjk)
        draw.text((54, y + 42), qlabel, fill=(40, 40, 40), font=latin_small)
        draw.text((10, y + 62), f"crop changed: {current.crop != adjusted.crop}", fill=(120, 0, 0), font=latin_small)
        draw.text((10, y + 82), f"break blue: { _breaks_hard_bound(current) or _breaks_hard_bound(adjusted) }", fill=(120, 0, 0), font=latin_small)
        draw.text((10, y + 104), f"src={token.source.to_list()}", fill=(40, 40, 40), font=latin_small)
        draw.text(
            (10, y + 126),
            f"cur {dict(current.counts)}",
            fill=(40, 40, 40),
            font=latin_small,
        )
        draw.text(
            (10, y + 148),
            f"adj {dict(adjusted.counts)}",
            fill=(40, 40, 40),
            font=latin_small,
        )
        draw.text((10, y + 170), f"cur crop={current.crop.to_list()}", fill=(40, 40, 40), font=latin_small)
        draw.text((10, y + 192), f"adj crop={adjusted.crop.to_list()}", fill=(40, 40, 40), font=latin_small)
        x = label_w
        image.paste(_draw_zone_panel(image_bgr, current, panel_w=scope_w, panel_h=row_h, latin=latin_small), (x, y))
        x += scope_w
        image.paste(_draw_crop_panel(image_bgr, current, panel_w=crop_w, panel_h=row_h, latin=latin_small), (x, y))
        x += crop_w
        image.paste(_draw_zone_panel(image_bgr, adjusted, panel_w=scope_w, panel_h=row_h, latin=latin_small), (x, y))
        x += scope_w
        image.paste(_draw_crop_panel(image_bgr, adjusted, panel_w=crop_w, panel_h=row_h, latin=latin_small), (x, y))
        draw.rectangle([0, y, image.width - 1, y + row_h - 1], outline=(205, 205, 205))
    image.save(output_path)


def _make_tier_matrix(output_path: Path) -> None:
    title_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    head_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    text_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    rows = [
        ("Q0 stable", "single/clean cc, margins normal", "keep current v11; no adaptive widening", "safe default"),
        ("Q1 edge-risk", "ink close to one source edge or bbox too tight", "small side-specific relax only; keep raw bbox", "mark review"),
        ("Q2 loose/noisy", "anti_dropped/rescue/multi-cc or parameter-sensitive", "use stricter ownership + edge crumb; compare parameter profiles", "yellow"),
        ("Q3 fallback", "no strong cc, no ink, or non-CJK mixed token", "do not force glyph crop; stay token/raw bbox", "manual/diagnostic"),
    ]
    image = Image.new("RGB", (1500, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 22), "Safe parameter tiering for variable PP-OCRv5 bbox quality", fill=(0, 0, 0), font=title_font)
    headers = ["tier", "bbox quality signal", "allowed tuning", "UI/export rule"]
    xs = [30, 250, 650, 1120]
    y = 92
    for x, header in zip(xs, headers):
        draw.text((x, y), header, fill=(0, 90, 160), font=head_font)
    y += 42
    for row in rows:
        draw.rectangle([20, y - 8, 1480, y + 78], outline=(220, 220, 220))
        for x, value in zip(xs, row):
            line = ""
            yy = y
            for word in value.split(" "):
                trial = f"{line} {word}".strip()
                if draw.textlength(trial, font=text_font) > 360 and line:
                    draw.text((x, yy), line, fill=(20, 20, 20), font=text_font)
                    yy += 22
                    line = word
                else:
                    line = trial
            if line:
                draw.text((x, yy), line, fill=(20, 20, 20), font=text_font)
        y += 92
    draw.text(
        (30, 472),
        "Rule: adaptive parameters can promote a candidate level, but must never overwrite raw_token_bbox; high-risk crops stay reviewable.",
        fill=(120, 0, 0),
        font=text_font,
    )
    image.save(output_path)


def _make_page_report(
    output_path: Path,
    page_id: str,
    samples: list[TokenRef],
    current_evals: dict[tuple[int, int], ZoneEval],
    adjusted_evals: dict[tuple[int, int], ZoneEval],
) -> None:
    changed = [
        token
        for token in current_evals
        if _effective_counts(current_evals[token]) != _effective_counts(adjusted_evals[token])
        or current_evals[token].crop != adjusted_evals[token].crop
        or current_evals[token].no_strong != adjusted_evals[token].no_strong
        or adjusted_evals[token].blue_expanded_sides
    ]
    crop_changed = [token for token in current_evals if current_evals[token].crop != adjusted_evals[token].crop]
    break_blue = [
        token
        for token in current_evals
        if _breaks_hard_bound(current_evals[token]) or _breaks_hard_bound(adjusted_evals[token])
    ]
    current_no_strong = sum(1 for eval_item in current_evals.values() if eval_item.no_strong)
    adjusted_no_strong = sum(1 for eval_item in adjusted_evals.values() if eval_item.no_strong)
    lines = [
        f"radiation zone comparison - page {page_id}",
        "=" * 72,
        "current v11: left strong inset=10%, right strong inset=20%, rescue right anti guard=5%",
        "adjusted: left strong extends 10% (left red anti shrinks 10%->0%), right strong shrinks 5% (right red anti grows 20%->25%)",
        "orange is rescue domain, not anti-radiation; it is shown to explain which weak components may be rescued.",
        "strong preservation: any component intersecting the green strong zone is kept whole before ownership clipping; red anti does not cut its stroke.",
        "blue adapts by 5% in the same direction when the main strong component touches it (10% strong : 5% blue = 2:1); purple stroke-search then follows the connected component to completion.",
        "",
        f"cjk tokens evaluated: {len(current_evals)}",
        f"effective parameter-sensitive tokens: {len(changed)}",
        f"final crop changed tokens: {len(crop_changed)}",
        f"break-blue stroke tokens: {len(break_blue)}",
        f"no strong cc current/adjusted: {current_no_strong}/{adjusted_no_strong}",
        "",
        "selected samples:",
    ]
    for token in samples:
        current = current_evals[token.id]
        adjusted = adjusted_evals[token.id]
        lines.append(
            f"- L{token.line:02d}#{token.token:03d} {token.text}: "
            f"{_quality_label(current, adjusted)}, crop_changed={current.crop != adjusted.crop}, "
            f"break_blue={_breaks_hard_bound(current) or _breaks_hard_bound(adjusted)}, "
            f"current={dict(current.counts)}, adjusted={dict(adjusted.counts)}"
        )
    lines.extend(
        [
            "",
            "interpretation:",
            "- The requested adjustment removes the left anti band for the adjusted profile and keeps the right side stricter.",
            "- This is a useful side-specific profile for PP-OCRv5 boxes whose left radicals are often clipped while right neighbor crumbs leak in.",
            "- It is not safe as a global replacement yet; use it as a Q1/Q2 profile selected by bbox quality signals.",
            "- Raw PP-OCRv5 word boxes must remain L0 fallback truth while refined crops carry risk flags.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _make_overview(output_path: Path, page_summaries: list[dict[str, Any]]) -> None:
    title_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    head_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    text_font = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    row_h = 34
    image = Image.new("RGB", (1380, 90 + row_h * len(page_summaries)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), "Page-level radiation scope sensitivity overview", fill=(0, 0, 0), font=title_font)
    draw.text(
        (24, 54),
        "red=anti/weak; orange=rescue; purple=stroke-search. Adjusted blue expands 5% when touched; strong-hit strokes complete beyond it.",
        fill=(80, 0, 0),
        font=text_font,
    )
    columns = [
        ("page", 24),
        ("cjk", 130),
        ("sensitive", 220),
        ("crop changed", 350),
        ("break blue", 500),
        ("current no-strong", 640),
        ("adjusted no-strong", 810),
        ("selected examples", 1010),
    ]
    y = 92
    for label, x in columns:
        draw.text((x, y), label, fill=(0, 90, 160), font=head_font)
    y += 30
    for summary in page_summaries:
        examples = ", ".join(f"L{line:02d}#{token:03d}{text}" for line, token, text in summary["selected_examples"][:6])
        values = [
            summary["page_id"],
            str(summary["cjk_total"]),
            str(summary["parameter_sensitive_count"]),
            str(summary["crop_changed_count"]),
            str(summary["break_blue_count"]),
            str(summary["current_no_strong_count"]),
            str(summary["adjusted_no_strong_count"]),
            examples,
        ]
        for (_, x), value in zip(columns, values):
            draw.text((x, y), value, fill=(20, 20, 20), font=text_font)
        y += row_h
    image.save(output_path)


def _make_batch_report(output_path: Path, page_summaries: list[dict[str, Any]]) -> None:
    total_cjk = sum(item["cjk_total"] for item in page_summaries)
    total_sensitive = sum(item["parameter_sensitive_count"] for item in page_summaries)
    total_crop_changed = sum(item["crop_changed_count"] for item in page_summaries)
    total_break_blue = sum(item["break_blue_count"] for item in page_summaries)
    lines = [
        "radiation zone comparison - page batch",
        "=" * 72,
        "Clarification: red is anti/weak radiation edge; orange is weak-zone rescue domain, not anti-radiation.",
        "Current v11: left strong inset=10%, right strong inset=20%, rescue right guard=5%.",
        "Adjusted profile: left strong extends 10% (left anti shrinks 10%->0%); right strong shrinks 5% (right anti grows 20%->25%).",
        "Strong preservation: a CC that intersects green strong is retained whole; red anti only filters fully-outside-strong CCs, then ownership may clip the final union.",
        "Adaptive blue rule: if the main strong component touches blue, blue expands 5% in the same direction (2:1 against the 10% strong-left extension). Purple stroke-search then follows the connected component to completion.",
        "",
        f"pages evaluated: {len(page_summaries)}",
        f"cjk tokens evaluated: {total_cjk}",
        f"parameter-sensitive tokens: {total_sensitive}",
        f"final crop changed tokens: {total_crop_changed}",
        f"break-blue stroke tokens: {total_break_blue}",
        "",
        "per-page summary:",
    ]
    for summary in page_summaries:
        examples = ", ".join(f"L{line:02d}#{token:03d}{text}" for line, token, text in summary["selected_examples"][:8])
        lines.append(
            f"- {summary['page_id']}: cjk={summary['cjk_total']} sensitive={summary['parameter_sensitive_count']} "
            f"crop_changed={summary['crop_changed_count']} break_blue={summary['break_blue_count']} "
            f"no_strong={summary['current_no_strong_count']}/{summary['adjusted_no_strong_count']} examples={examples}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_page_comparison(
    page_id: str,
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    *,
    samples_per_page: int,
    write_top_level: bool = False,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pruned = data["response"]["result"]["ocrResults"][0]["prunedResult"]
    tokens = _load_tokens(pruned)
    current_spec = ZoneSpec(
        name="current v11",
        left_inset_ratio=0.10,
        right_inset_ratio=0.20,
        rescue_left_guard_ratio=0.00,
        rescue_right_guard_ratio=0.05,
    )
    adjusted_spec = ZoneSpec(
        name="adjusted scope",
        left_inset_ratio=0.00,
        right_inset_ratio=0.25,
        stroke_extension_ratio=0.25,
        adaptive_hard_bound_ratio=0.05,
        rescue_left_guard_ratio=0.00,
        rescue_right_guard_ratio=0.10,
    )
    current_evals = {token.id: _evaluate_zone(image_bgr, token, current_spec) for token in tokens}
    adjusted_evals = {token.id: _evaluate_zone(image_bgr, token, adjusted_spec) for token in tokens}
    samples = _select_samples(tokens, current_evals, adjusted_evals, limit=samples_per_page)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_image = pages_dir / f"{page_id}_radiation_scope_current_vs_adjusted.png"
    _make_scope_comparison(image_bgr, samples, current_evals, adjusted_evals, page_image, page_id=page_id)
    if write_top_level:
        _make_scope_comparison(
            image_bgr,
            samples,
            current_evals,
            adjusted_evals,
            output_dir / "01_radiation_scope_current_vs_adjusted.png",
            page_id=page_id,
        )
        _make_page_report(output_dir / "03_radiation_scope_report.txt", page_id, samples, current_evals, adjusted_evals)
    changed = [
        token_id
        for token_id in current_evals
        if _effective_counts(current_evals[token_id]) != _effective_counts(adjusted_evals[token_id])
        or current_evals[token_id].crop != adjusted_evals[token_id].crop
        or current_evals[token_id].no_strong != adjusted_evals[token_id].no_strong
        or adjusted_evals[token_id].blue_expanded_sides
    ]
    crop_changed = [token_id for token_id in current_evals if current_evals[token_id].crop != adjusted_evals[token_id].crop]
    break_blue = [
        token_id
        for token_id in current_evals
        if _breaks_hard_bound(current_evals[token_id]) or _breaks_hard_bound(adjusted_evals[token_id])
    ]
    payload: dict[str, Any] = {
        "page_id": page_id,
        "source_image": str(image_path),
        "source_json": str(json_path),
        "page_visual": str(page_image),
        "current_spec": current_spec.__dict__,
        "adjusted_spec": adjusted_spec.__dict__,
        "cjk_total": len(tokens),
        "parameter_sensitive_count": len(changed),
        "crop_changed_count": len(crop_changed),
        "break_blue_count": len(break_blue),
        "current_no_strong_count": sum(1 for eval_item in current_evals.values() if eval_item.no_strong),
        "adjusted_no_strong_count": sum(1 for eval_item in adjusted_evals.values() if eval_item.no_strong),
        "sample_token_ids": [list(token.id) for token in samples],
        "selected_examples": [(token.line, token.token, token.text) for token in samples],
        "samples": [
            {
                "id": list(token.id),
                "text": token.text,
                "quality": _quality_label(current_evals[token.id], adjusted_evals[token.id]),
                "current": current_evals[token.id].to_json(),
                "adjusted": adjusted_evals[token.id].to_json(),
            }
            for token in samples
        ],
    }
    return payload


def run_batch_comparison(
    image_dir: Path,
    json_dir: Path,
    output_dir: Path,
    *,
    page_ids: list[str],
    samples_per_page: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_payloads: list[dict[str, Any]] = []
    for index, page_id in enumerate(page_ids):
        image_path = image_dir / f"{page_id}.tif"
        json_path = json_dir / f"{page_id}_ppocrv5_return_word_box.json"
        if not image_path.exists() or not json_path.exists():
            continue
        page_payloads.append(
            run_page_comparison(
                page_id,
                image_path,
                json_path,
                output_dir,
                samples_per_page=samples_per_page,
                write_top_level=(index == 0),
            )
        )
    if not page_payloads:
        raise RuntimeError(f"No matching page image/json pairs under {image_dir} and {json_dir}")
    _make_tier_matrix(output_dir / "02_bbox_quality_tier_matrix.png")
    _make_overview(output_dir / "04_page_scope_overview.png", page_payloads)
    _make_batch_report(output_dir / "05_page_scope_batch_report.txt", page_payloads)
    payload = {
        "pages": page_payloads,
        "legend": {
            "green": "strong direct-keep radiation zone",
            "red": "anti/weak edge zone",
            "orange": "weak-zone rescue domain, not anti-radiation",
            "cyan": "ownership interval",
            "purple": "larger stroke-search area used for component extraction",
            "blue": "initial hard_bound / diagnostic box, not the final stroke limit",
            "final_crop_columns": "current and adjusted crop boxes after keeping strong/rescued components and clipping to ownership",
        },
    }
    (output_dir / "radiation_zone_comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualize current and adjusted wordbox_anchor radiation scopes.")
    parser.add_argument("--image-dir", type=Path, default=repo_root / "file/244771纵校")
    parser.add_argument("--json-dir", type=Path, default=repo_root.parent / "claude/.cache")
    parser.add_argument("--output", type=Path, default=repo_root / "paddle-char-box-samples/radiation-zone-comparison")
    parser.add_argument(
        "--pages",
        default="120166,120167,120168,120169,120170,120171,120172,120173,120174,120175,120176,120177,120178,120179,120180,120183,120184,120185,120186,120187",
        help="Comma-separated page ids to process.",
    )
    parser.add_argument("--samples-per-page", type=int, default=8)
    args = parser.parse_args()
    page_ids = [item.strip() for item in args.pages.split(",") if item.strip()]
    run_batch_comparison(args.image_dir, args.json_dir, args.output, page_ids=page_ids, samples_per_page=args.samples_per_page)


if __name__ == "__main__":
    main()
