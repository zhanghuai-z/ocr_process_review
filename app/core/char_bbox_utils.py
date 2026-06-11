from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from app.models import BBox, Char, Line


LINE_DIRECTION_HORIZONTAL = "horizontal"
LINE_DIRECTION_VERTICAL = "vertical"
MISSING_LINE_BBOX_FLAG = "missing_line_bbox"
BBOX_SOURCE_UNAVAILABLE = "unavailable"
BBOX_GRANULARITY_UNAVAILABLE = "unavailable"

# Sources whose char-granularity bboxes are inherently trustworthy and must NOT
# be subject to the neighbor-repair heuristic in ensure_line_char_bboxes.
# Engine adapters may still provide explicit per-character boxes as source
# "ocr"; Hanwang ("hanwang:*") delivers CharRcg boxes from its native
# recognition pass. These are first-party char-level evidence.
_TRUSTED_OCR_CHAR_SOURCES = ("ocr",)
_TRUSTED_OCR_CHAR_SOURCE_PREFIXES = ("hanwang:",)


def _is_trusted_char_bbox_source(bbox_source: str) -> bool:
    """Return True when the bbox source should be trusted at char granularity.

    Trusted sources are exempt from the neighbor-repair heuristic that replaces
    an explicit char bbox with a split-line estimate when the bbox centre appears
    to land on an adjacent slot.  Applying that repair to Hanwang CharRcg boxes
    would silently overwrite valid, high-quality per-character coordinates.
    """
    s = (bbox_source or "").strip().lower()
    if s in _TRUSTED_OCR_CHAR_SOURCES:
        return True
    return any(s.startswith(pfx) for pfx in _TRUSTED_OCR_CHAR_SOURCE_PREFIXES)


def infer_line_direction(line_bbox: BBox, text_length: int) -> str:
    """根据行框长宽比推断字符分布方向。"""
    if text_length <= 1:
        return LINE_DIRECTION_HORIZONTAL
    bbox = line_bbox.normalize()
    if bbox.h > bbox.w * 1.4:
        return LINE_DIRECTION_VERTICAL
    return LINE_DIRECTION_HORIZONTAL


def infer_bbox_direction(line_bbox: BBox) -> str:
    bbox = line_bbox.normalize()
    if bbox.h > bbox.w * 1.4:
        return LINE_DIRECTION_VERTICAL
    return LINE_DIRECTION_HORIZONTAL


def split_line_bbox_into_char_bboxes(line_bbox: BBox, text: str) -> List[BBox]:
    """在缺少字符级 bbox 时，按行框方向切分出字符框。"""
    text_length = len(text)
    if text_length <= 0:
        return []

    bbox = line_bbox.normalize()
    if bbox.w <= 0 or bbox.h <= 0:
        return []
    if text_length == 1:
        return [bbox]

    direction = infer_line_direction(bbox, text_length)
    char_bboxes: List[BBox] = []
    if direction == LINE_DIRECTION_VERTICAL:
        edges = [
            bbox.y + round(idx * bbox.h / float(text_length))
            for idx in range(text_length + 1)
        ]
        for idx in range(text_length):
            y1 = edges[idx]
            y2 = max(y1 + 1, edges[idx + 1])
            char_bboxes.append(BBox(bbox.x, y1, bbox.w, y2 - y1))
    else:
        edges = [
            bbox.x + round(idx * bbox.w / float(text_length))
            for idx in range(text_length + 1)
        ]
        for idx in range(text_length):
            x1 = edges[idx]
            x2 = max(x1 + 1, edges[idx + 1])
            char_bboxes.append(BBox(x1, bbox.y, x2 - x1, bbox.h))
    return char_bboxes


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return thresh > 0


def _nonzero_bounds(projection: np.ndarray) -> Optional[tuple[int, int]]:
    nonzero = np.flatnonzero(projection > 0)
    if nonzero.size == 0:
        return None
    return int(nonzero[0]), int(nonzero[-1]) + 1


def _projection_runs(projection: np.ndarray) -> List[tuple[int, int]]:
    active = projection > 0
    if not np.any(active):
        return []
    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _select_dominant_run(
    projection: np.ndarray,
    *,
    target_center: float,
) -> Optional[tuple[int, int]]:
    runs = _projection_runs(projection)
    if not runs:
        return None

    def score(run: tuple[int, int]) -> tuple[float, float, int]:
        start, end = run
        ink = float(projection[start:end].sum())
        center = (start + end) / 2.0
        distance = abs(center - target_center)
        return (ink, -distance, end - start)

    return max(runs, key=score)


def _nearest_projection_valley(
    projection: np.ndarray,
    target: int,
    *,
    left_bound: int,
    right_bound: int,
    search_radius: int,
) -> int:
    left = max(left_bound, target - search_radius)
    right = min(right_bound, target + search_radius)
    window = projection[left:right + 1]
    if window.size == 0:
        return target
    smoothed = np.convolve(window, np.ones(5) / 5.0, mode="same")
    min_value = float(smoothed.min())
    candidate_offsets = np.flatnonzero(np.isclose(smoothed, min_value))
    best_offset = min(candidate_offsets, key=lambda idx: abs((left + int(idx)) - target))
    return left + int(best_offset)


def _projection_guided_boundaries(
    projection: np.ndarray,
    segment_count: int,
) -> List[int]:
    primary_len = int(projection.shape[0])
    if segment_count <= 1 or primary_len <= segment_count:
        return [0, primary_len]

    avg_span = max(primary_len / float(segment_count), 1.0)
    search_radius = max(2, int(round(avg_span * 0.6)))
    boundaries = [0]
    for boundary_idx in range(1, segment_count):
        target = int(round(boundary_idx * primary_len / float(segment_count)))
        min_boundary = boundaries[-1] + 1
        max_boundary = primary_len - (segment_count - boundary_idx)
        refined = _nearest_projection_valley(
            projection,
            target,
            left_bound=min_boundary,
            right_bound=max_boundary,
            search_radius=search_radius,
        )
        boundaries.append(min(max(refined, min_boundary), max_boundary))
    boundaries.append(primary_len)
    return boundaries


def _tighten_segment_to_foreground(
    mask: np.ndarray,
    segment_bbox: BBox,
) -> BBox:
    if mask.size == 0 or not mask.any():
        return segment_bbox

    rows = np.flatnonzero(mask.sum(axis=1) > 0)
    cols = np.flatnonzero(mask.sum(axis=0) > 0)
    if rows.size == 0 or cols.size == 0:
        return segment_bbox

    pad = 1
    x1 = max(0, int(cols[0]) - pad)
    y1 = max(0, int(rows[0]) - pad)
    x2 = min(mask.shape[1], int(cols[-1]) + 1 + pad)
    y2 = min(mask.shape[0], int(rows[-1]) + 1 + pad)
    return BBox(
        segment_bbox.x + x1,
        segment_bbox.y + y1,
        max(1, x2 - x1),
        max(1, y2 - y1),
    )


def refine_line_char_bboxes(
    line_bbox: BBox,
    text: str,
    page_image: np.ndarray,
) -> List[BBox]:
    """根据真实墨迹收紧字符框，减少按整行均分导致的切图偏差。"""
    base_boxes = split_line_bbox_into_char_bboxes(line_bbox, text)
    if not text or page_image.size == 0 or not base_boxes:
        return base_boxes

    bbox = line_bbox.normalize().clamp(page_image.shape[1], page_image.shape[0])
    if bbox.w <= 0 or bbox.h <= 0:
        return base_boxes

    crop = page_image[bbox.y:bbox.y2, bbox.x:bbox.x2]
    if crop.size == 0:
        return base_boxes

    mask = _foreground_mask(crop)
    if not mask.any():
        return base_boxes

    direction = infer_line_direction(bbox, len(text))
    primary_projection = mask.sum(axis=0) if direction == LINE_DIRECTION_HORIZONTAL else mask.sum(axis=1)
    secondary_projection = mask.sum(axis=1) if direction == LINE_DIRECTION_HORIZONTAL else mask.sum(axis=0)
    primary_bounds = _nonzero_bounds(primary_projection)
    secondary_bounds = _nonzero_bounds(secondary_projection)
    if primary_bounds is None or secondary_bounds is None:
        return base_boxes

    p0, p1 = primary_bounds
    s0, s1 = secondary_bounds
    if direction == LINE_DIRECTION_HORIZONTAL:
        tight_mask = mask[s0:s1, p0:p1]
    else:
        tight_mask = mask[p0:p1, s0:s1]
    if tight_mask.size == 0 or not tight_mask.any():
        return base_boxes

    tight_primary_projection = (
        tight_mask.sum(axis=0)
        if direction == LINE_DIRECTION_HORIZONTAL
        else tight_mask.sum(axis=1)
    )
    boundaries = _projection_guided_boundaries(tight_primary_projection, len(text))
    refined_boxes: List[BBox] = []
    for idx in range(len(text)):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if direction == LINE_DIRECTION_HORIZONTAL:
            segment_bbox = BBox(bbox.x + p0 + start, bbox.y + s0, max(1, end - start), max(1, s1 - s0))
            segment_mask = tight_mask[:, start:end]
        else:
            segment_bbox = BBox(bbox.x + s0, bbox.y + p0 + start, max(1, s1 - s0), max(1, end - start))
            segment_mask = tight_mask[start:end, :]
        refined_boxes.append(_tighten_segment_to_foreground(segment_mask, segment_bbox))
    return refined_boxes


def refine_line_bbox(
    line_bbox: BBox,
    page_image: np.ndarray,
) -> BBox:
    """根据真实墨迹收紧整行 bbox，并在松散框中选择主文本带。"""
    bbox = line_bbox.normalize().clamp(page_image.shape[1], page_image.shape[0])
    if bbox.w <= 0 or bbox.h <= 0:
        return bbox

    crop = page_image[bbox.y:bbox.y2, bbox.x:bbox.x2]
    if crop.size == 0:
        return bbox

    mask = _foreground_mask(crop)
    if not mask.any():
        return bbox

    direction = infer_bbox_direction(bbox)
    if direction == LINE_DIRECTION_HORIZONTAL:
        secondary_projection = mask.sum(axis=1)
        dominant = _select_dominant_run(
            secondary_projection,
            target_center=(bbox.h / 2.0),
        )
        if dominant is None:
            return bbox
        y0, y1 = dominant
        band_mask = mask[y0:y1, :]
        primary_projection = band_mask.sum(axis=0)
        primary_bounds = _nonzero_bounds(primary_projection)
        if primary_bounds is None:
            return bbox
        x0, x1 = primary_bounds
        return _tighten_segment_to_foreground(
            band_mask[:, x0:x1],
            BBox(bbox.x + x0, bbox.y + y0, max(1, x1 - x0), max(1, y1 - y0)),
        )

    secondary_projection = mask.sum(axis=0)
    dominant = _select_dominant_run(
        secondary_projection,
        target_center=(bbox.w / 2.0),
    )
    if dominant is None:
        return bbox
    x0, x1 = dominant
    band_mask = mask[:, x0:x1]
    primary_projection = band_mask.sum(axis=1)
    primary_bounds = _nonzero_bounds(primary_projection)
    if primary_bounds is None:
        return bbox
    y0, y1 = primary_bounds
    return _tighten_segment_to_foreground(
        band_mask[y0:y1, :],
        BBox(bbox.x + x0, bbox.y + y0, max(1, x1 - x0), max(1, y1 - y0)),
    )


def measure_bbox_foreground(
    page_image: Optional[np.ndarray],
    bbox: Optional[BBox],
) -> dict:
    if page_image is None or bbox is None:
        return {
            "ink_pixels": 0,
            "fill_ratio": 0.0,
            "tight_w": 0,
            "tight_h": 0,
        }
    normalized = bbox.normalize().clamp(page_image.shape[1], page_image.shape[0])
    if normalized.w <= 0 or normalized.h <= 0:
        return {
            "ink_pixels": 0,
            "fill_ratio": 0.0,
            "tight_w": 0,
            "tight_h": 0,
        }

    crop = page_image[normalized.y:normalized.y2, normalized.x:normalized.x2]
    if crop.size == 0:
        return {
            "ink_pixels": 0,
            "fill_ratio": 0.0,
            "tight_w": 0,
            "tight_h": 0,
        }

    mask = _foreground_mask(crop)
    ink_pixels = int(mask.sum())
    if ink_pixels <= 0:
        return {
            "ink_pixels": 0,
            "fill_ratio": 0.0,
            "tight_w": 0,
            "tight_h": 0,
        }

    rows = np.flatnonzero(mask.sum(axis=1) > 0)
    cols = np.flatnonzero(mask.sum(axis=0) > 0)
    tight_h = int(rows[-1] - rows[0] + 1) if rows.size else 0
    tight_w = int(cols[-1] - cols[0] + 1) if cols.size else 0
    return {
        "ink_pixels": ink_pixels,
        "fill_ratio": float(ink_pixels) / float(normalized.w * normalized.h),
        "tight_w": tight_w,
        "tight_h": tight_h,
    }


def is_meaningful_text_bbox(
    page_image: Optional[np.ndarray],
    bbox: Optional[BBox],
    text: str = "",
) -> bool:
    if bbox is None or bbox.area <= 0:
        return False
    if page_image is None:
        return True

    metrics = measure_bbox_foreground(page_image, bbox)
    ink_pixels = int(metrics["ink_pixels"])
    if ink_pixels <= 0:
        return False

    core_text = "".join(ch for ch in text if not ch.isspace())
    expected_len = max(1, len(core_text))
    tight_primary = max(int(metrics["tight_w"]), int(metrics["tight_h"]))
    tight_secondary = min(int(metrics["tight_w"]), int(metrics["tight_h"]))

    if tight_primary < max(2, expected_len) and ink_pixels < expected_len * 3:
        return False
    if tight_secondary <= 1 and ink_pixels < max(3, expected_len * 2):
        return False
    if float(metrics["fill_ratio"]) < 0.01 and ink_pixels < max(4, expected_len * 2):
        return False
    return True


def _bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    first = first.normalize()
    second = second.normalize()
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / float(min(first.area, second.area))


def _bbox_center_inside(bbox: BBox, target: BBox) -> bool:
    cx = bbox.x + bbox.w / 2.0
    cy = bbox.y + bbox.h / 2.0
    return target.x <= cx <= target.x2 and target.y <= cy <= target.y2


def explicit_char_bbox_points_to_neighbor(
    explicit_bbox: BBox,
    expected_bboxes: List[BBox],
    idx: int,
) -> bool:
    """Return True when an OCR char bbox clearly lands on an adjacent slot."""
    if len(expected_bboxes) <= 1 or idx < 0 or idx >= len(expected_bboxes):
        return False
    explicit = explicit_bbox.normalize()
    expected = expected_bboxes[idx].normalize()
    tolerance = max(2, int(max(expected.w, expected.h) * 0.20))
    expected_zone = expected.expand(tolerance)
    explicit_center_x = explicit.x + explicit.w / 2.0
    explicit_center_y = explicit.y + explicit.h / 2.0
    expected_center_x = expected.x + expected.w / 2.0
    expected_center_y = expected.y + expected.h / 2.0
    if expected.w >= expected.h:
        if abs(explicit_center_x - expected_center_x) <= expected.w * 0.75:
            return False
    elif abs(explicit_center_y - expected_center_y) <= expected.h * 0.75:
        return False
    own_overlap = _bbox_overlap_ratio(explicit, expected)
    neighbor_overlap = 0.0
    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(expected_bboxes):
            neighbor_overlap = max(
                neighbor_overlap,
                _bbox_overlap_ratio(explicit, expected_bboxes[neighbor_idx]),
            )
    if _bbox_center_inside(explicit, expected_zone) and own_overlap >= neighbor_overlap:
        return False
    return neighbor_overlap > max(own_overlap * 1.25, 0.10)


def ensure_line_char_bboxes(
    line: Line,
    page_image: Optional[np.ndarray] = None,
) -> List[Char]:
    """确保 line.chars 至少拥有与文本长度一致的 page-space bbox。"""
    text = line.display_text
    if not text:
        line.chars = []
        return []

    if MISSING_LINE_BBOX_FLAG in line.review_flags:
        line.chars = [
            Char(
                char=glyph,
                confidence=float(line.confidence),
                bbox=None,
                bbox_source=BBOX_SOURCE_UNAVAILABLE,
                bbox_granularity=BBOX_GRANULARITY_UNAVAILABLE,
                token_text=glyph,
            )
            for glyph in text
        ]
        return line.chars

    refined_line_bbox = (
        refine_line_bbox(line.bbox, page_image)
        if page_image is not None
        else line.bbox.normalize()
    )
    line.bbox = refined_line_bbox
    split_bboxes = (
        refine_line_char_bboxes(refined_line_bbox, text, page_image)
        if page_image is not None
        else split_line_bbox_into_char_bboxes(refined_line_bbox, text)
    )
    chars: List[Char] = []
    for idx, glyph in enumerate(text):
        existing = line.chars[idx] if idx < len(line.chars) else None
        has_explicit_bbox = (
            existing is not None
            and existing.bbox is not None
            and existing.bbox.area > 0
        )
        repaired_neighbor_bbox = False
        if has_explicit_bbox:
            bbox = existing.bbox.normalize()
            if (
                existing is not None
                and existing.bbox_granularity == "char"
                and not _is_trusted_char_bbox_source(existing.bbox_source)
                and explicit_char_bbox_points_to_neighbor(bbox, split_bboxes, idx)
            ):
                bbox = split_bboxes[idx]
                repaired_neighbor_bbox = True
        else:
            bbox = split_bboxes[idx]
        confidence = (
            float(existing.confidence)
            if existing is not None
            else float(line.confidence)
        )
        char_id = existing.id if existing is not None else None
        bbox_source = (
            existing.bbox_source
            if has_explicit_bbox and not repaired_neighbor_bbox and existing and existing.bbox_source
            else "fallback"
        )
        bbox_granularity = (
            existing.bbox_granularity
            if has_explicit_bbox and not repaired_neighbor_bbox and existing and existing.bbox_granularity
            else "fallback"
        )
        token_text = (
            existing.token_text
            if existing is not None and existing.token_text
            else glyph
        )
        chars.append(Char(
            char=glyph,
            confidence=confidence,
            bbox=bbox,
            id=char_id,
            bbox_source=bbox_source,
            bbox_granularity=bbox_granularity,
            token_text=token_text,
        ))
    line.chars = chars
    return chars
