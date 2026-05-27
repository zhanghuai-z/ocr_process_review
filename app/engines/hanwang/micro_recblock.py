"""Page-level PP-VL block -> Hanwang linecut micro-recblock integration."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.bbox_extraction import bbox_from_variant
from app.core.logging import get_logger
from app.core.paddle_labels import authoritative_paddle_label, normalize_paddle_label
from app.core.proof_status import proof_status_for
from app.models import BBox, Block, BlockSource, BlockType, Char, Line, Page

from . import native_bridge

logger = get_logger(__name__)

TEXT_LABELS: set[str] = {
    "text",
    "paragraph",
    "paragraph_text",
    "paragraph_title",
    "plain_text",
    "body",
    "body_text",
    "title",
    "header",
    "footer",
    "footnote",
    "number",
    "page_number",
    "reference",
    "references",
    "reference_list",
    "bibliography",
    "caption",
    "figure_caption",
    "figure_title",
    "table_caption",
    "table_title",
    "table_note",
}

SKIP_LABELS: set[str] = {
    "display_formula",
    "inline_formula",
    "isolated_formula",
    "formula",
    "formula_number",
    "equation",
    "table",
    "table_region",
    "table_block",
    "table_body",
    "figure",
    "chart",
    "graphic",
    "image",
    "picture",
    "photo",
    "seal",
    "stamp",
}

FALLBACK_RATIO_THRESHOLD = 0.85
MAX_RECOG_BATCH_GROUPS = 6
MAX_RECOG_COLLAGE_WIDTH = 1600
MAX_RECOG_COLLAGE_HEIGHT = 1600
MAX_RECOG_COLLAGE_PIXELS = 2_000_000
MAX_RECOG_COLLAGE_ASPECT = 4.5
_BATCH_DISABLED_FOR_SESSION = False
_BATCH_DISABLE_REASON = ""


@dataclass
class CharResult:
    text: str
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    candidates: list[str] = field(default_factory=list)
    source: str = "hanwang:micro_recblock"


@dataclass
class LineResult:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    chars: list[CharResult] = field(default_factory=list)
    source: str = "hanwang"


@dataclass
class BlockResult:
    block_idx: int
    block_label: str
    block_bbox: tuple[int, int, int, int]
    source: str
    text: str
    ppvl_text: str
    group_count: int = 0
    lines: list[LineResult] = field(default_factory=list)
    fallback_reason: str = ""
    raw_block: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_idx": self.block_idx,
            "block_label": self.block_label,
            "block_bbox": list(self.block_bbox),
            "source": self.source,
            "text": self.text,
            "ppvl_text": self.ppvl_text,
            "group_count": self.group_count,
            "fallback_reason": self.fallback_reason,
            "raw_block": dict(self.raw_block),
            "lines": [
                {
                    "text": line.text,
                    "bbox": list(line.bbox),
                    "confidence": line.confidence,
                    "source": line.source,
                    "chars": [
                        {
                            "text": char.text,
                            "confidence": char.confidence,
                            "bbox": list(char.bbox) if char.bbox else None,
                            "candidates": list(char.candidates),
                            "source": char.source,
                        }
                        for char in line.chars
                    ],
                }
                for line in self.lines
            ],
        }


@dataclass
class RunStats:
    n_blocks_total: int = 0
    n_blocks_hanwang: int = 0
    n_blocks_ppvl: int = 0
    n_blocks_fallback: int = 0
    n_groups: int = 0
    seg_seconds: float = 0.0
    recog_seconds: float = 0.0
    recog_full_page_pixels: int = 0
    recog_crop_pixels: int = 0
    recog_probe_calls: int = 0
    recog_batch_chunks: int = 0
    recog_batch_failures: int = 0
    recog_batch_disabled: bool = False
    recog_batch_guarded_chunks: int = 0
    recog_max_collage_width: int = 0
    recog_max_collage_height: int = 0
    recog_max_collage_pixels: int = 0


@dataclass
class _GroupPlacement:
    area_idx: int
    page_bbox: tuple[int, int, int, int]
    collage_bbox: tuple[int, int, int, int]


def _label_from_block(block: dict[str, Any], default: str = "unknown") -> str:
    return normalize_paddle_label(authoritative_paddle_label(block, default))


def _is_skip_label(label: str) -> bool:
    if label in TEXT_LABELS:
        return False
    if label in SKIP_LABELS:
        return True
    return label.startswith((
        "equation",
        "formula",
        "table",
        "figure",
        "image",
        "chart",
        "seal",
        "stamp",
    ))


def _is_text_label(label: str) -> bool:
    return label in TEXT_LABELS or not _is_skip_label(label)


def decode_gbk(code: int) -> str:
    """Decode Hanwang little-endian GBK code and strip NUL/control noise."""
    if not code:
        return ""
    try:
        text = code.to_bytes(2, "little").decode("gbk", errors="ignore")
    except Exception:
        return ""
    return "".join(ch for ch in text if ch >= " " or ch in "\n\t")


def _score_to_confidence(score: object) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    conf = 1.0 - value / 100.0
    return max(0.0, min(1.0, conf))


def _code_to_int(code: object) -> int:
    if isinstance(code, str):
        return int(code, 0) if code.startswith(("0x", "0X")) else int(code, 16)
    return int(code)


def _bbox_tuple(raw: object, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not isinstance(raw, dict):
        return fallback
    try:
        left = int(raw.get("left", raw.get("x", fallback[0])))
        top = int(raw.get("top", raw.get("y", fallback[1])))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw.get("width", max(0, fallback[2] - fallback[0])))
            bottom = top + int(raw.get("height", max(0, fallback[3] - fallback[1])))
    except (TypeError, ValueError):
        return fallback
    if right <= left or bottom <= top:
        return fallback
    return left, top, right, bottom


def _clamp_xyxy(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left = max(0, min(int(left), width))
    top = max(0, min(int(top), height))
    right = max(left, min(int(right), width))
    bottom = max(top, min(int(bottom), height))
    return left, top, right, bottom


def _block_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int]:
    bbox = bbox_from_variant(
        raw.get("block_bbox") or raw.get("bbox") or raw.get("coordinate"),
        max_w=width,
        max_h=height,
    )
    if bbox is None or bbox.area <= 0:
        return 0, 0, width, height
    return bbox.to_xyxy()


def _char_result(raw: dict, fallback_bbox: tuple[int, int, int, int]) -> CharResult:
    codes = raw.get("codes") or []
    scores = raw.get("scores") or []
    candidates: list[str] = []
    for code in codes:
        try:
            text = decode_gbk(_code_to_int(code))
        except (TypeError, ValueError):
            text = ""
        if text:
            candidates.append(text)
    text = candidates[0] if candidates else ""
    confidence = _score_to_confidence(scores[0] if scores else 100)
    return CharResult(
        text=text,
        confidence=confidence,
        bbox=_bbox_tuple(raw.get("bbox"), fallback_bbox),
        candidates=candidates,
    )


def _fallback_line(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    source: str,
) -> LineResult:
    return LineResult(
        text=text,
        bbox=bbox,
        confidence=0.0,
        source=source,
        chars=[
            CharResult(text=ch, confidence=0.0, bbox=None, candidates=[ch], source=source)
            for ch in text
        ],
    )


def _line_results_from_recog(
    raw: dict,
    *,
    fallback_bbox: tuple[int, int, int, int],
    include_chars: bool,
    fallback_empty: bool = True,
) -> list[LineResult]:
    lines: list[LineResult] = []
    for area in raw.get("lines", []) or []:
        for group in area.get("groups", []) or []:
            line_bbox = _bbox_tuple(group.get("bbox"), fallback_bbox)
            raw_chars = group.get("chars") or []
            chars = [_char_result(char, line_bbox) for char in raw_chars]
            text = "".join(char.text for char in chars).strip()
            if not text and not chars:
                continue
            confidence = (
                sum(char.confidence for char in chars) / len(chars)
                if chars else 0.0
            )
            lines.append(
                LineResult(
                    text=text,
                    bbox=line_bbox,
                    confidence=confidence,
                    chars=chars if include_chars else [],
                )
            )
    if not lines and fallback_empty:
        lines.append(_fallback_line("", fallback_bbox, source="hanwang_empty"))
    return lines


def _offset_line_results(
    lines: list[LineResult],
    *,
    dx: int,
    dy: int,
) -> list[LineResult]:
    shifted: list[LineResult] = []
    for line in lines:
        lx1, ly1, lx2, ly2 = line.bbox
        chars: list[CharResult] = []
        for char in line.chars:
            bbox = None
            if char.bbox is not None:
                cx1, cy1, cx2, cy2 = char.bbox
                bbox = (cx1 + dx, cy1 + dy, cx2 + dx, cy2 + dy)
            chars.append(
                CharResult(
                    text=char.text,
                    confidence=char.confidence,
                    bbox=bbox,
                    candidates=list(char.candidates),
                    source=char.source,
                )
            )
        shifted.append(
            LineResult(
                text=line.text,
                bbox=(lx1 + dx, ly1 + dy, lx2 + dx, ly2 + dy),
                confidence=line.confidence,
                chars=chars,
                source=line.source,
            )
        )
    return shifted


def _intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0, right - left) * max(0, bottom - top)


def _placement_for_line(
    line: LineResult,
    placements: list[_GroupPlacement],
) -> _GroupPlacement | None:
    if not placements:
        return None
    x1, y1, x2, y2 = line.bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    for placement in placements:
        left, top, right, bottom = placement.collage_bbox
        if left <= cx <= right and top <= cy <= bottom:
            return placement
    return max(
        placements,
        key=lambda placement: _intersection_area(line.bbox, placement.collage_bbox),
    )


def _build_group_collage(
    image_bgr: np.ndarray,
    group_bboxes: list[tuple[int, int, int, int]],
    area_indices: list[int],
) -> tuple[np.ndarray, list[_GroupPlacement]]:
    if not group_bboxes:
        return image_bgr[:0, :0].copy(), []
    gap = 2
    crops: list[np.ndarray] = []
    max_width = 1
    total_height = 0
    for bbox in group_bboxes:
        left, top, right, bottom = bbox
        crop = image_bgr[top:bottom, left:right].copy()
        crops.append(crop)
        crop_h, crop_w = crop.shape[:2]
        max_width = max(max_width, crop_w)
        total_height += crop_h
    total_height += gap * max(0, len(crops) - 1)
    if image_bgr.ndim == 2:
        collage = np.full((total_height, max_width), 255, dtype=image_bgr.dtype)
    else:
        collage = np.full((total_height, max_width, image_bgr.shape[2]), 255, dtype=image_bgr.dtype)
    placements: list[_GroupPlacement] = []
    y = 0
    for crop, page_bbox, area_idx in zip(crops, group_bboxes, area_indices):
        crop_h, crop_w = crop.shape[:2]
        collage[y:y + crop_h, 0:crop_w] = crop
        placements.append(
            _GroupPlacement(
                area_idx=area_idx,
                page_bbox=page_bbox,
                collage_bbox=(0, y, crop_w, y + crop_h),
            )
        )
        y += crop_h + gap
    return collage, placements


def _estimate_collage_shape(
    group_bboxes: list[tuple[int, int, int, int]],
    *,
    gap: int = 2,
) -> tuple[int, int, int]:
    if not group_bboxes:
        return 0, 0, 0
    width = max(max(0, right - left) for left, _top, right, _bottom in group_bboxes)
    height = sum(max(0, bottom - top) for _left, top, _right, bottom in group_bboxes)
    height += gap * max(0, len(group_bboxes) - 1)
    return width, height, width * height


def _chunk_group_bboxes(
    group_bboxes: list[tuple[int, int, int, int]],
    area_indices: list[int],
    *,
    max_groups: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_pixels: int | None = None,
    max_aspect: float | None = None,
) -> list[tuple[list[tuple[int, int, int, int]], list[int]]]:
    if max_groups is None:
        max_groups = MAX_RECOG_BATCH_GROUPS
    if max_width is None:
        max_width = MAX_RECOG_COLLAGE_WIDTH
    if max_height is None:
        max_height = MAX_RECOG_COLLAGE_HEIGHT
    if max_pixels is None:
        max_pixels = MAX_RECOG_COLLAGE_PIXELS
    if max_aspect is None:
        max_aspect = MAX_RECOG_COLLAGE_ASPECT
    chunks: list[tuple[list[tuple[int, int, int, int]], list[int]]] = []
    current_bboxes: list[tuple[int, int, int, int]] = []
    current_areas: list[int] = []
    for bbox, area_idx in zip(group_bboxes, area_indices):
        candidate_bboxes = [*current_bboxes, bbox]
        width, height, pixels = _estimate_collage_shape(candidate_bboxes)
        aspect = (width / height) if height > 0 else 0.0
        over_limit = (
            len(candidate_bboxes) > max_groups
            or width > max_width
            or height > max_height
            or pixels > max_pixels
            or aspect > max_aspect
        )
        if current_bboxes and over_limit:
            chunks.append((current_bboxes, current_areas))
            current_bboxes = [bbox]
            current_areas = [area_idx]
        else:
            current_bboxes = candidate_bboxes
            current_areas = [*current_areas, area_idx]
    if current_bboxes:
        chunks.append((current_bboxes, current_areas))
    return chunks


def _text_length(text: str) -> int:
    return len("".join(ch for ch in text if not ch.isspace()))


def _fallback_needed(hw_text: str, ppvl_text: str) -> str:
    hw_len = _text_length(hw_text)
    ppvl_len = _text_length(ppvl_text)
    if ppvl_len <= 0:
        return ""
    if hw_len <= 0:
        return "empty_hanwang_text"
    if hw_len < ppvl_len * FALLBACK_RATIO_THRESHOLD:
        return f"short_hanwang_text:{hw_len}/{ppvl_len}"
    return ""


def run_micro_recblock(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict],
    *,
    seg_timeout: float = 120.0,
    recog_timeout: float = 60.0,
    include_chars: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[BlockResult], RunStats]:
    """Run Hanwang Recog for text-like PP-VL blocks and keep PP-VL for others."""
    global _BATCH_DISABLED_FOR_SESSION, _BATCH_DISABLE_REASON
    height, width = image_bgr.shape[:2]
    stats = RunStats(n_blocks_total=len(ppvl_blocks))
    text_indices: list[int] = []
    skip_indices: list[int] = []
    for idx, block in enumerate(ppvl_blocks):
        label = _label_from_block(block)
        if _is_skip_label(label):
            skip_indices.append(idx)
        elif _is_text_label(label):
            text_indices.append(idx)

    stats.n_blocks_hanwang = len(text_indices)
    stats.n_blocks_ppvl = len(skip_indices)
    rows: list[BlockResult | None] = [None] * len(ppvl_blocks)

    for idx in skip_indices:
        block = ppvl_blocks[idx]
        label = _label_from_block(block)
        bbox = _block_bbox(block, width, height)
        ppvl_text = str(block.get("block_content") or block.get("text") or "").strip()
        rows[idx] = BlockResult(
            block_idx=idx,
            block_label=label,
            block_bbox=bbox,
            source="ppvl",
            text=ppvl_text,
            ppvl_text=ppvl_text,
            lines=[_fallback_line(ppvl_text, bbox, source="ppvl")],
            raw_block=dict(block),
        )

    if text_indices:
        recblocks = [
            _clamp_xyxy(_block_bbox(ppvl_blocks[idx], width, height), width, height)
            for idx in text_indices
        ]
        if progress_callback:
            progress_callback(
                0,
                max(1, len(text_indices)),
                "Hanwang micro-recblock SegImg 分块中…",
            )
        started = time.time()
        seg = native_bridge.run_linecut_segimg(
            image_bgr,
            recblocks_xyxy=recblocks,
            timeout=seg_timeout,
        )
        stats.seg_seconds = time.time() - started

        groups: list[dict] = []
        for area_idx, area in enumerate(seg.get("lines", []) or []):
            for group in area.get("groups", []) or []:
                group["_area_idx"] = area_idx
                groups.append(group)
        stats.n_groups = len(groups)
        if progress_callback:
            progress_callback(
                0,
                max(1, len(groups)),
                f"Hanwang micro-recblock Recog 准备中：{len(groups)} 个 group",
            )

        started = time.time()
        grouped_lines: dict[int, list[LineResult]] = {idx: [] for idx in range(len(text_indices))}
        total_groups = max(1, len(groups))
        group_bboxes: list[tuple[int, int, int, int]] = []
        group_area_indices: list[int] = []
        for group in groups:
            bbox = _clamp_xyxy(
                _bbox_tuple(group.get("bbox"), recblocks[group["_area_idx"]]),
                width,
                height,
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            group_bboxes.append(bbox)
            group_area_indices.append(group["_area_idx"])

        def recognize_individually(placements: list[_GroupPlacement]) -> None:
            for placement in placements:
                left, top, right, bottom = placement.page_bbox
                crop = image_bgr[top:bottom, left:right].copy()
                crop_h, crop_w = crop.shape[:2]
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        crop,
                        recblock_xyxy=None,
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    logger.warning("Hanwang micro_recblock group failed bbox=%s: %s", placement.page_bbox, exc)
                    raw = {}
                local_lines = _line_results_from_recog(
                    raw,
                    fallback_bbox=(0, 0, crop_w, crop_h),
                    include_chars=include_chars,
                )
                grouped_lines[placement.area_idx].extend(
                    _offset_line_results(local_lines, dx=left, dy=top)
                )

        batch_enabled = not _BATCH_DISABLED_FOR_SESSION
        stats.recog_batch_disabled = _BATCH_DISABLED_FOR_SESSION
        completed_groups = 0
        chunks = _chunk_group_bboxes(group_bboxes, group_area_indices)
        for chunk_index, (chunk_bboxes, chunk_area_indices) in enumerate(chunks):
            collage, placements = _build_group_collage(image_bgr, chunk_bboxes, chunk_area_indices)
            if not placements:
                continue
            stats.recog_batch_chunks += 1
            collage_h, collage_w = collage.shape[:2]
            collage_pixels = collage_w * collage_h
            stats.recog_max_collage_width = max(stats.recog_max_collage_width, collage_w)
            stats.recog_max_collage_height = max(stats.recog_max_collage_height, collage_h)
            stats.recog_max_collage_pixels = max(stats.recog_max_collage_pixels, collage_pixels)
            stats.recog_full_page_pixels += width * height * len(placements)
            stats.recog_crop_pixels += sum(
                (placement.page_bbox[2] - placement.page_bbox[0])
                * (placement.page_bbox[3] - placement.page_bbox[1])
                for placement in placements
            )
            collage_aspect = (collage_w / collage_h) if collage_h > 0 else 0.0
            shape_guarded = (
                collage_w > MAX_RECOG_COLLAGE_WIDTH
                or collage_h > MAX_RECOG_COLLAGE_HEIGHT
                or collage_pixels > MAX_RECOG_COLLAGE_PIXELS
                or collage_aspect > MAX_RECOG_COLLAGE_ASPECT
            )
            if shape_guarded:
                stats.recog_batch_guarded_chunks += 1
            use_batch = batch_enabled and len(placements) > 1 and not shape_guarded
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    (
                        "Hanwang micro-recblock 分块批量识别中… "
                        f"chunk {chunk_index + 1}/{len(chunks)}, groups={len(placements)}, "
                        f"collage={collage_w}x{collage_h}"
                    ),
                )
            if use_batch:
                try:
                    stats.recog_probe_calls += 1
                    raw = native_bridge.run_linecut_recog(
                        collage,
                        recblocks_xyxy=[placement.collage_bbox for placement in placements],
                        with_charrcg=True,
                        timeout=recog_timeout,
                    )
                except Exception as exc:
                    stats.recog_batch_failures += 1
                    stats.recog_batch_disabled = True
                    batch_enabled = False
                    _BATCH_DISABLED_FOR_SESSION = True
                    _BATCH_DISABLE_REASON = (
                        f"chunk={chunk_index + 1} groups={len(placements)} "
                        f"collage={collage_w}x{collage_h}: {exc}"
                    )
                    logger.warning(
                        "Hanwang micro_recblock batch failed; disabling batch for this process "
                        "chunk=%d groups=%d collage=%dx%d: %s",
                        chunk_index + 1,
                        len(placements),
                        collage_w,
                        collage_h,
                        exc,
                    )
                    recognize_individually(placements)
                else:
                    local_lines = _line_results_from_recog(
                        raw,
                        fallback_bbox=(0, 0, collage_w, collage_h),
                        include_chars=include_chars,
                        fallback_empty=False,
                    )
                    for local_line in local_lines:
                        placement = _placement_for_line(local_line, placements)
                        if placement is None:
                            continue
                        dx = placement.page_bbox[0] - placement.collage_bbox[0]
                        dy = placement.page_bbox[1] - placement.collage_bbox[1]
                        grouped_lines[placement.area_idx].extend(
                            _offset_line_results([local_line], dx=dx, dy=dy)
                        )
            else:
                recognize_individually(placements)

            completed_groups += len(placements)
            if progress_callback:
                progress_callback(
                    completed_groups,
                    total_groups,
                    f"Hanwang micro-recblock 已完成 {completed_groups}/{total_groups} groups",
                )
        stats.recog_seconds = time.time() - started

        for area_idx, block_idx in enumerate(text_indices):
            block = ppvl_blocks[block_idx]
            label = _label_from_block(block)
            bbox = _block_bbox(block, width, height)
            ppvl_text = str(block.get("block_content") or block.get("text") or "").strip()
            lines = grouped_lines.get(area_idx, [])
            lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
            hw_text = "".join(line.text for line in lines).strip()
            fallback_reason = _fallback_needed(hw_text, ppvl_text)
            source = "hanwang"
            text = hw_text
            if fallback_reason:
                source = "ppvl_fallback"
                text = ppvl_text
                lines = [_fallback_line(ppvl_text, bbox, source="ppvl_fallback")]
                stats.n_blocks_fallback += 1
            rows[block_idx] = BlockResult(
                block_idx=block_idx,
                block_label=label,
                block_bbox=bbox,
                source=source,
                text=text,
                ppvl_text=ppvl_text,
                group_count=len(lines),
                lines=lines,
                fallback_reason=fallback_reason,
                raw_block=dict(block),
            )

    return [row for row in rows if row is not None], stats


def _bbox_from_xyxy_tuple(raw: tuple[int, int, int, int], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = _clamp_xyxy(raw, width, height)
    return BBox.from_xyxy(x1, y1, x2, y2)


def _char_to_model(char: CharResult) -> Char:
    bbox = BBox.from_xyxy(*char.bbox) if char.bbox else None
    return Char(
        char=char.text,
        confidence=char.confidence,
        bbox=bbox,
        bbox_source=char.source,
        bbox_granularity="char" if bbox is not None else "fallback",
        token_text=char.text,
    )


def _line_to_model(line: LineResult, width: int, height: int, review_flags: list[str]) -> Line:
    chars = [_char_to_model(char) for char in line.chars]
    model = Line(
        text=line.text,
        final_text=line.text,
        confidence=line.confidence,
        bbox=_bbox_from_xyxy_tuple(line.bbox, width, height),
        chars=chars,
        ocr_text=line.text,
        original_text=line.text,
        review_flags=list(review_flags),
        proof_status=proof_status_for(line.confidence, review_flags),
    )
    return model


def _page_blocks_from_layout(page: Page) -> list[dict]:
    blocks: list[dict] = []
    for block in page.blocks:
        raw_payload = dict(block.raw_payload)
        source_label = (
            authoritative_paddle_label(raw_payload)
            or block.source_label
            or block.block_type.value
        )
        blocks.append(
            {
                **raw_payload,
                "block_label": source_label,
                "block_bbox": list(block.bbox.to_xyxy()),
                "block_content": block.full_text or block.note,
                "source_label": block.source_label or source_label,
            }
        )
    return blocks


class HanwangMicroRecBlockEngine:
    """OcrPipeline page-level engine for PP-VL layout + Hanwang text OCR."""

    prefer_page_hybrid_blocks = True
    bbox_space = "page"

    def __init__(
        self,
        *,
        seg_timeout: float = 120.0,
        recog_timeout: float = 60.0,
        runner=run_micro_recblock,
    ) -> None:
        self._seg_timeout = seg_timeout
        self._recog_timeout = recog_timeout
        self._runner = runner

    def recognize_page_blocks(
        self,
        image_bgr: np.ndarray,
        page: Page,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> RunStats:
        ppvl_blocks = page.ppvl_parsing_res_list or _page_blocks_from_layout(page)
        if not ppvl_blocks:
            raise RuntimeError("Hanwang micro_recblock requires PP-VL parsing_res_list blocks")

        rows, stats = self._runner(
            image_bgr,
            ppvl_blocks,
            seg_timeout=self._seg_timeout,
            recog_timeout=self._recog_timeout,
            include_chars=True,
            progress_callback=progress_callback,
        )

        new_blocks: list[Block] = []
        height, width = image_bgr.shape[:2]
        for order, row in enumerate(rows):
            bbox = _bbox_from_xyxy_tuple(row.block_bbox, width, height)
            block_type = BlockType.from_paddle(row.block_label)
            flags: list[str] = []
            if row.fallback_reason:
                flags.append("hanwang_micro_recblock_fallback")
            lines = [_line_to_model(line, width, height, flags) for line in row.lines if line.text]
            note_parts = [
                f"source_label={row.block_label}",
                f"micro_recblock_source={row.source}",
            ]
            if row.fallback_reason:
                note_parts.append(f"fallback_reason={row.fallback_reason}")
            if row.ppvl_text:
                note_parts.append(f"ppvl_text={row.ppvl_text[:120]}")
            new_blocks.append(
                Block(
                    block_type=block_type,
                    bbox=bbox,
                    lines=lines,
                    order=order,
                    source=BlockSource.AUTO_LAYOUT,
                    recognizable=row.source == "hanwang",
                    note=" | ".join(note_parts),
                    source_label=row.block_label,
                    raw_payload=dict(row.raw_block),
                )
            )

        page.blocks = new_blocks
        logger.info(
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d fallback=%d "
            "groups=%d chunks=%d guarded_chunks=%d batch_failures=%d batch_disabled=%s "
            "max_collage=%dx%d probe_calls=%d recog_pixels=%d/%d",
            page.page_number,
            stats.n_blocks_total,
            stats.n_blocks_hanwang,
            stats.n_blocks_ppvl,
            stats.n_blocks_fallback,
            stats.n_groups,
            stats.recog_batch_chunks,
            stats.recog_batch_guarded_chunks,
            stats.recog_batch_failures,
            stats.recog_batch_disabled,
            stats.recog_max_collage_width,
            stats.recog_max_collage_height,
            stats.recog_probe_calls,
            stats.recog_crop_pixels,
            stats.recog_full_page_pixels,
        )
        return stats


__all__ = [
    "TEXT_LABELS",
    "SKIP_LABELS",
    "FALLBACK_RATIO_THRESHOLD",
    "CharResult",
    "LineResult",
    "BlockResult",
    "RunStats",
    "HanwangMicroRecBlockEngine",
    "decode_gbk",
    "run_micro_recblock",
]
