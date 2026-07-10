"""Typed PP-OCRv6 prepass adapter used only by the CharOCR routing boundary."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.api_image_codec import encode_image_bytes_for_paddle
from app.core.bbox_extraction import bbox_from_variant
from app.core.paddle_v16_client import PaddleV16LayoutClient


PPOCR_V6_MODEL = "PP-OCRv6"


def build_ppocr_v6_prepass_options() -> dict[str, object]:
    """Return the fixed PP-OCRv6 request contract for CharOCR routing."""
    return {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
        "returnWordBox": True,
        "textDetLimitSideLen": 1536,
        "textDetLimitType": "max",
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 2.0,
        "textRecScoreThresh": 0.0,
    }


@dataclass(frozen=True)
class PpOcrV6WordBox:
    """One PP-OCRv6 word/digit/punctuation proposal inside a physical line."""

    line_index: int
    token_index: int
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PpOcrV6LineHint:
    """A PP-OCRv6 physical row plus its optional word-box proposals."""

    index: int
    text: str
    bbox: tuple[int, int, int, int]
    words: tuple[PpOcrV6WordBox, ...]


@dataclass(frozen=True)
class PpOcrV6PrepassArtifact:
    """External PP-OCRv6 observation for one page and one request run.

    It is transient routing input.  It must not be projected to proof lines,
    blocks, or characters.
    """

    page_uid: str
    run_id: str
    lines: tuple[PpOcrV6LineHint, ...]

    def __post_init__(self) -> None:
        if not self.page_uid:
            raise ValueError("PP-OCRv6 prepass artifact requires page_uid")


class PpOcrV6PrepassClient:
    """Use the common Paddle jobs transport with the PP-OCRv6 model contract."""

    def __init__(self, transport: PaddleV16LayoutClient) -> None:
        self._transport = transport

    def analyze_page(
        self,
        image_bgr: np.ndarray,
        *,
        page_uid: str,
        batch_id: str = "",
        filename: str = "page.png",
    ) -> PpOcrV6PrepassArtifact:
        image_bytes = encode_image_bytes_for_paddle(image_bgr)
        if not image_bytes:
            raise RuntimeError("Cannot encode image for PP-OCRv6 routing prepass")
        return self.analyze_page_bytes(
            image_bytes,
            page_uid=page_uid,
            batch_id=batch_id,
            filename=filename,
        )

    def analyze_page_bytes(
        self,
        image_bytes: bytes,
        *,
        page_uid: str,
        batch_id: str = "",
        filename: str = "page.png",
    ) -> PpOcrV6PrepassArtifact:
        job_id = self._transport.submit_image_bytes(
            image_bytes,
            model=PPOCR_V6_MODEL,
            optional_payload=build_ppocr_v6_prepass_options(),
            batch_id=batch_id,
            filename=filename,
        )
        json_url, _job_data = self._transport.wait_for_result_json_url(job_id)
        return parse_ppocr_v6_prepass_jsonl(
            self._transport.download_jsonl(json_url),
            page_uid=page_uid,
            run_id=job_id,
        )


def parse_ppocr_v6_prepass_jsonl(
    jsonl_text: str,
    *,
    page_uid: str,
    run_id: str = "",
    width: int | None = None,
    height: int | None = None,
) -> PpOcrV6PrepassArtifact:
    """Parse PP-OCRv6 jobs JSONL without coercing it into layout/proof data."""
    result_items: list[dict[str, Any]] = []
    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError("PP-OCRv6 result contains invalid JSONL") from exc
        if not isinstance(row, dict):
            continue
        result = row.get("result", row)
        if not isinstance(result, dict):
            continue
        values = result.get("ocrResults", [])
        if isinstance(values, list):
            result_items.extend(value for value in values if isinstance(value, dict))

    if not result_items:
        raise ValueError("PP-OCRv6 result contains no ocrResults")
    if len(result_items) != 1:
        raise ValueError(f"PP-OCRv6 prepass expected one page result, got {len(result_items)}")
    return parse_ppocr_v6_prepass_result(
        result_items[0],
        page_uid=page_uid,
        run_id=run_id,
        width=width,
        height=height,
    )


def parse_ppocr_v6_prepass_result(
    result: dict[str, Any],
    *,
    page_uid: str,
    run_id: str = "",
    width: int | None = None,
    height: int | None = None,
) -> PpOcrV6PrepassArtifact:
    """Normalize one PP-OCRv6 result item into a strict routing artifact."""
    pruned = result.get("prunedResult", result)
    if not isinstance(pruned, dict):
        raise ValueError("PP-OCRv6 result prunedResult must be an object")
    texts = pruned.get("rec_texts", [])
    boxes = pruned.get("rec_boxes", [])
    words_by_line = pruned.get("text_word", [])
    word_boxes_by_line = pruned.get("text_word_boxes", [])
    if not isinstance(texts, list) or not isinstance(boxes, list):
        raise ValueError("PP-OCRv6 result missing rec_texts/rec_boxes lists")
    if len(texts) != len(boxes):
        raise ValueError(
            "PP-OCRv6 result has mismatched rec_texts/rec_boxes lengths: "
            f"{len(texts)} != {len(boxes)}"
        )
    if words_by_line is not None and not isinstance(words_by_line, list):
        raise ValueError("PP-OCRv6 result text_word must be a list")
    if word_boxes_by_line is not None and not isinstance(word_boxes_by_line, list):
        raise ValueError("PP-OCRv6 result text_word_boxes must be a list")

    lines: list[PpOcrV6LineHint] = []
    for line_index, raw_text in enumerate(texts):
        line_bbox = _parse_bbox(boxes[line_index], width=width, height=height)
        if line_bbox is None:
            raise ValueError(f"PP-OCRv6 line {line_index} has invalid rec_box")
        raw_words = words_by_line[line_index] if line_index < len(words_by_line) else []
        raw_word_boxes = word_boxes_by_line[line_index] if line_index < len(word_boxes_by_line) else []
        if raw_words is None:
            raw_words = []
        if raw_word_boxes is None:
            raw_word_boxes = []
        if not isinstance(raw_words, list) or not isinstance(raw_word_boxes, list):
            raise ValueError(f"PP-OCRv6 line {line_index} word boxes must be lists")
        if len(raw_words) != len(raw_word_boxes):
            raise ValueError(
                f"PP-OCRv6 line {line_index} has mismatched text_word/text_word_boxes lengths: "
                f"{len(raw_words)} != {len(raw_word_boxes)}"
            )
        words: list[PpOcrV6WordBox] = []
        for token_index, (raw_word, raw_bbox) in enumerate(zip(raw_words, raw_word_boxes)):
            word_bbox = _parse_bbox(raw_bbox, width=width, height=height)
            if word_bbox is None:
                raise ValueError(f"PP-OCRv6 line {line_index} token {token_index} has invalid word box")
            words.append(PpOcrV6WordBox(
                line_index=line_index,
                token_index=token_index,
                text=str(raw_word or ""),
                bbox=word_bbox,
            ))
        lines.append(PpOcrV6LineHint(
            index=line_index,
            text=str(raw_text or ""),
            bbox=line_bbox,
            words=tuple(words),
        ))
    return PpOcrV6PrepassArtifact(page_uid=page_uid, run_id=run_id, lines=tuple(lines))


def _parse_bbox(value: object, *, width: int | None, height: int | None) -> tuple[int, int, int, int] | None:
    bbox = bbox_from_variant(value, max_w=width, max_h=height)
    if bbox is None or bbox.area <= 0:
        return None
    return bbox.to_xyxy()


__all__ = [
    "PPOCR_V6_MODEL",
    "PpOcrV6LineHint",
    "PpOcrV6PrepassArtifact",
    "PpOcrV6PrepassClient",
    "PpOcrV6WordBox",
    "build_ppocr_v6_prepass_options",
    "parse_ppocr_v6_prepass_jsonl",
    "parse_ppocr_v6_prepass_result",
]
