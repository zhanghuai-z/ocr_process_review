"""Page-level PP-VL block -> Hanwang linecut micro-recblock integration."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.bbox_extraction import bbox_from_variant
from app.core.logging import get_logger
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


def _normalize_label(label: object) -> str:
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_skip_label(label: str) -> bool:
    if label in SKIP_LABELS:
        return True
    return any(token in label for token in ("formula", "equation")) and "caption" not in label


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
    if not lines:
        lines.append(_fallback_line("", fallback_bbox, source="hanwang_empty"))
    return lines


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
    height, width = image_bgr.shape[:2]
    stats = RunStats(n_blocks_total=len(ppvl_blocks))
    text_indices: list[int] = []
    skip_indices: list[int] = []
    for idx, block in enumerate(ppvl_blocks):
        label = _normalize_label(block.get("block_label") or block.get("label"))
        if _is_skip_label(label):
            skip_indices.append(idx)
        elif _is_text_label(label):
            text_indices.append(idx)

    stats.n_blocks_hanwang = len(text_indices)
    stats.n_blocks_ppvl = len(skip_indices)
    rows: list[BlockResult | None] = [None] * len(ppvl_blocks)

    for idx in skip_indices:
        block = ppvl_blocks[idx]
        label = _normalize_label(block.get("block_label") or block.get("label"))
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
        for group_index, group in enumerate(groups):
            bbox = _bbox_tuple(group.get("bbox"), recblocks[group["_area_idx"]])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if progress_callback:
                progress_callback(
                    group_index,
                    total_groups,
                    f"Hanwang micro-recblock 识别中… group {group_index + 1}/{total_groups}",
                )
            try:
                raw = native_bridge.run_linecut_recog(
                    image_bgr,
                    recblock_xyxy=bbox,
                    with_charrcg=True,
                    timeout=recog_timeout,
                )
            except Exception as exc:
                logger.warning("Hanwang micro_recblock group failed bbox=%s: %s", bbox, exc)
                raw = {}
            grouped_lines[group["_area_idx"]].extend(
                _line_results_from_recog(raw, fallback_bbox=bbox, include_chars=include_chars)
            )
            if progress_callback:
                progress_callback(
                    group_index + 1,
                    total_groups,
                    f"Hanwang micro-recblock 已完成 group {group_index + 1}/{total_groups}",
                )
        stats.recog_seconds = time.time() - started

        for area_idx, block_idx in enumerate(text_indices):
            block = ppvl_blocks[block_idx]
            label = _normalize_label(block.get("block_label") or block.get("label"))
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
            str(raw_payload.get("block_label") or "")
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
            "Hanwang micro_recblock page=%s blocks=%d hanwang=%d ppvl=%d fallback=%d groups=%d",
            page.page_number,
            stats.n_blocks_total,
            stats.n_blocks_hanwang,
            stats.n_blocks_ppvl,
            stats.n_blocks_fallback,
            stats.n_groups,
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
