"""Pixel-only geometry postprocessing for Hanwang character observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


XYXY = tuple[int, int, int, int]
_MIN_COMPONENT_AREA = 3
_LATIN_SLOPE_GRID = np.linspace(-0.6, 0.6, 121)


@dataclass(frozen=True)
class LineAtomGeometry:
    index: int
    text: str
    bbox: XYXY | None
    source: str
    granularity: str


@dataclass(frozen=True)
class LatinSlantMeasurement:
    measurable: bool
    slope: float = 0.0
    score_improvement: float = 0.0
    pixel_count: int = 0


def conservative_cjk_bbox_cleanup(
    image_bgr: np.ndarray,
    line_bbox: XYXY,
    atoms: Sequence[LineAtomGeometry],
    *,
    linecut_source: str,
) -> dict[int, XYXY]:
    """Tighten eligible CJK boxes only across foreground-free ownership seams."""
    eligible = [
        atom
        for atom in atoms
        if atom.source == linecut_source
        and atom.granularity == "char"
        and atom.bbox is not None
        and _is_cjk(atom.text)
    ]
    if not eligible:
        return {}
    pitch = _median_positive_gaps([_center_x(atom.bbox) for atom in eligible if atom.bbox])
    if pitch <= 0:
        widths = [atom.bbox[2] - atom.bbox[0] for atom in eligible if atom.bbox]
        pitch = _median([float(width) for width in widths]) if widths else 0.0
    if pitch <= 0:
        return {}

    foreground = _dark_foreground(image_bgr)
    band = _cjk_band(eligible, line_bbox)
    if band is None:
        return {}
    visible = [
        atom for atom in atoms
        if atom.bbox is not None and atom.text.strip() and atom.granularity != "space"
    ]
    positions = {atom.index: position for position, atom in enumerate(visible)}
    proposals: dict[int, XYXY] = {}
    for atom in eligible:
        assert atom.bbox is not None
        position = positions.get(atom.index)
        if position is None:
            continue
        center = _center_x(atom.bbox)
        seam_risk = False
        if position > 0:
            left, risky = _best_vertical_seam(
                foreground, band, _center_x(visible[position - 1].bbox), center
            )
            seam_risk = seam_risk or risky
        else:
            left = atom.bbox[0]
        if position + 1 < len(visible):
            right, risky = _best_vertical_seam(
                foreground, band, center, _center_x(visible[position + 1].bbox)
            )
            seam_risk = seam_risk or risky
        else:
            right = atom.bbox[2]
        left = max(0, left)
        right = min(image_bgr.shape[1], right)
        if seam_risk or right <= left:
            continue
        slot = (left, band[0], right, band[1])
        if _slot_edge_has_ink(foreground, slot):
            continue
        cleanup = (
            max(atom.bbox[0], left),
            atom.bbox[1],
            min(atom.bbox[2], right),
            atom.bbox[3],
        )
        proposal = _tight_foreground_bbox(foreground, cleanup)
        if proposal is not None and proposal != atom.bbox:
            proposals[atom.index] = proposal
    return proposals


def measure_latin_right_slant(
    image_bgr: np.ndarray,
    bbox: XYXY,
) -> LatinSlantMeasurement:
    """Measure a token-local right slant without altering the EngCut input."""
    height, width = image_bgr.shape[:2]
    x1 = max(0, bbox[0] - 2)
    y1 = max(0, bbox[1] - 2)
    x2 = min(width, bbox[2] + 2)
    y2 = min(height, bbox[3] + 2)
    if x2 <= x1 or y2 <= y1:
        return LatinSlantMeasurement(False)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    foreground = _polarity_aware_foreground(gray)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, 8)
    component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    total_area = sum(component_areas)
    ink_height = max(1, bbox[3] - bbox[1])
    kept = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_HEIGHT]) >= ink_height * 0.35
        and int(stats[index, cv2.CC_STAT_AREA]) >= max(3, round(total_area * 0.005))
    ]
    if not kept:
        return LatinSlantMeasurement(False)
    ys, xs = np.where(np.isin(labels, kept))
    if len(xs) < 24:
        return LatinSlantMeasurement(False, pixel_count=int(len(xs)))
    rises = float(np.max(ys)) - ys.astype(np.float64)
    if float(np.max(rises)) < 5.0:
        return LatinSlantMeasurement(False, pixel_count=int(len(xs)))
    xs_float = xs.astype(np.float64)
    scores = np.array([
        _projection_score(xs_float, rises, float(slope))
        for slope in _LATIN_SLOPE_GRID
    ])
    best_index = int(np.argmax(scores))
    if best_index in {0, len(_LATIN_SLOPE_GRID) - 1}:
        return LatinSlantMeasurement(False, pixel_count=int(len(xs)))
    best_score = float(scores[best_index])
    zero_score = float(scores[int(np.argmin(np.abs(_LATIN_SLOPE_GRID)))])
    return LatinSlantMeasurement(
        measurable=True,
        slope=float(_LATIN_SLOPE_GRID[best_index]),
        score_improvement=(best_score - zero_score) / max(zero_score, 1e-12),
        pixel_count=int(len(xs)),
    )


def is_latin_right_slant_fallback(
    measurement: LatinSlantMeasurement,
    *,
    minimum_slope: float = 0.12,
    minimum_improvement: float = 0.05,
) -> bool:
    return bool(
        measurement.measurable
        and measurement.slope >= minimum_slope
        and measurement.score_improvement >= minimum_improvement
    )


def _is_cjk(text: str) -> bool:
    if len(text) != 1:
        return False
    codepoint = ord(text)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def _center_x(bbox: XYXY | None) -> float:
    assert bbox is not None
    return (bbox[0] + bbox[2]) / 2.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _median_positive_gaps(centers: list[float]) -> float:
    return _median([
        right - left for left, right in zip(centers, centers[1:]) if right > left
    ])


def _cjk_band(atoms: Sequence[LineAtomGeometry], line_bbox: XYXY) -> tuple[int, int] | None:
    boxes = [atom.bbox for atom in atoms if atom.bbox is not None]
    if not boxes:
        return None
    heights = [float(box[3] - box[1]) for box in boxes]
    padding = max(1, round(max(1.0, _median(heights)) * 0.08))
    top = max(line_bbox[1], round(_median([float(box[1]) for box in boxes])) - padding)
    bottom = min(line_bbox[3], round(_median([float(box[3]) for box in boxes])) + padding)
    return (top, bottom) if bottom > top else None


def _best_vertical_seam(
    foreground: np.ndarray,
    band: tuple[int, int],
    left_center: float,
    right_center: float,
) -> tuple[int, bool]:
    midpoint = round((left_center + right_center) / 2.0)
    if right_center <= left_center:
        return midpoint, False
    distance = right_center - left_center
    start = max(1, round(left_center + distance * 0.30))
    stop = min(foreground.shape[1] - 1, round(left_center + distance * 0.70))
    if stop < start:
        return midpoint, False
    top, bottom = band
    candidates = [
        (int(foreground[top:bottom, seam - 1:seam + 1].sum()), abs(seam - midpoint), seam)
        for seam in range(start, stop + 1)
    ]
    score, _distance, seam = min(candidates)
    return seam, score > 0


def _tight_foreground_bbox(foreground: np.ndarray, bbox: XYXY) -> XYXY | None:
    x1, y1, x2, y2 = bbox
    crop = foreground[y1:y2, x1:x2].astype(np.uint8)
    if crop.size == 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(crop, 8)
    kept = [
        index for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= _MIN_COMPONENT_AREA
    ]
    if not kept:
        return None
    ys, xs = np.where(np.isin(labels, kept))
    return x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1


def _slot_edge_has_ink(foreground: np.ndarray, bbox: XYXY) -> bool:
    x1, y1, x2, y2 = bbox
    left = int(foreground[y1:y2, x1].sum()) if x1 < foreground.shape[1] else 0
    right_x = x2 - 1
    right = int(foreground[y1:y2, right_x].sum()) if right_x >= 0 else 0
    return bool(left or right)


def _dark_foreground(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return (gray < 128).astype(np.uint8)


def _polarity_aware_foreground(gray: np.ndarray) -> np.ndarray:
    threshold, _unused = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) <= threshold:
        return (gray > threshold).astype(np.uint8)
    return (gray <= threshold).astype(np.uint8)


def _projection_score(xs: np.ndarray, rises: np.ndarray, slope: float) -> float:
    columns = np.rint(xs - slope * rises).astype(np.int32)
    columns -= int(columns.min())
    counts = np.bincount(columns)
    return float(np.dot(counts, counts)) / max(1.0, float(len(columns) ** 2))


__all__ = [
    "LatinSlantMeasurement",
    "LineAtomGeometry",
    "conservative_cjk_bbox_cleanup",
    "is_latin_right_slant_fallback",
    "measure_latin_right_slant",
]
